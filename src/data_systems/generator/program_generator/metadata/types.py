"""
型システム

現行システム（argument_generator.py）の4分類型システムを完全継承し、拡張
"""
import random
from enum import Enum
from typing import Set, Dict, List, Optional, TYPE_CHECKING, NamedTuple, Any

if TYPE_CHECKING:
    from .commands import CommandMetadata


class SemanticType(Enum):
    """意味的な型（現行システムの4分類 + 拡張）"""

    # ========================================
    # 1. COLOR - 完全独立（算術演算禁止）
    # ========================================
    COLOR = "COLOR"

    # ========================================
    # 2. CORE_SPATIAL - 自由演算（28.5%）
    # ========================================
    COORDINATE = "COORDINATE"       # x, y座標
    OFFSET = "OFFSET"               # dx, dy（相対移動）
    SIZE = "SIZE"                   # width, height, pixels
    DISTANCE_AXIS = "DISTANCE_AXIS" # GET_X_DISTANCE, GET_Y_DISTANCE

    # ========================================
    # 3. MEASUREMENT - 制限的（0.3-1.2%）
    # ========================================
    DISTANCE = "DISTANCE"           # GET_DISTANCE（ユークリッド距離）

    # カウント系の細分化（意味的に独立した値を区別）
    COUNT = "COUNT"                 # 汎用カウント（LEN等）
    LOOP_INDEX = "LOOP_INDEX"        # ループ変数（i, j, kなど）
    COUNT_HOLES = "COUNT_HOLES"     # 穴の数（COUNT_HOLES, GET_HOLE_COUNT）
    COUNT_ADJACENT = "COUNT_ADJACENT"  # 隣接ピクセル数
    COUNT_OVERLAP = "COUNT_OVERLAP"    # 重複ピクセル数

    # ========================================
    # 4. PERCENTAGE - 比較専用（算術演算0%）
    # ========================================
    PERCENTAGE = "PERCENTAGE"       # GET_SYMMETRY_SCORE (0-100)

    # ========================================
    # 比率型 - 比較専用（基本値との比較で使用）
    # ========================================
    ASPECT_RATIO = "ASPECT_RATIO"   # GET_ASPECT_RATIO (幅/高さ * 100、通常1-1000程度)
    DENSITY = "DENSITY"             # GET_DENSITY (ピクセル数/bbox面積 * 100、0-100)

    # ========================================
    # その他の数値型
    # ========================================
    ANGLE = "ANGLE"                 # 0, 90, 180, 270
    SCALE = "SCALE"                 # 2, 3, 4

    # ========================================
    # 真偽値型
    # ========================================
    BOOL = "BOOL"                   # 真偽値


    # ========================================
    # 特殊型
    # ========================================
    VOID = "VOID"                   # 戻り値なし

    # ========================================
    # 文字列型
    # ========================================
    STRING = "STRING"               # 汎用文字列型
    DIRECTION = "DIRECTION"         # "X", "Y", "-X", "-Y", "XY", "-XY", "X-Y", "-X-Y", "C"
    AXIS = "AXIS"                   # "X", "Y"
    LINE_TYPE = "LINE_TYPE"         # "X", "Y", "XY", "-XY", "C"
    RECT_TYPE = "RECT_TYPE"         # "filled", "hollow", "none"
    SIDE = "SIDE"                   # "start", "end"
    ORDER = "ORDER"                 # "asc", "desc"
    ALIGN_MODE = "ALIGN_MODE"       # "left", "right", "top", "bottom", "center_x", "center_y", "center"

    # ========================================
    # オブジェクト型
    # ========================================
    OBJECT = "OBJECT"               # 単一オブジェクト



class ReturnType(Enum):
    """戻り値の基本型"""
    OBJECT = "object"           # 単一オブジェクト
    INT = "int"                 # 整数
    BOOL = "bool"               # 真偽値
    STRING = "string"           # 文字列
    VOID = "void"               # 戻り値なし（RENDER_GRID等）
    LIST_INT = "list[int]"      # 整数配列（GET_INPUT_GRID_SIZE）
    LIST_COLOR = "list[color]"  # 色配列（GET_COLORS）


class TypeInfo(NamedTuple):
    """型情報（ReturnType、SemanticType、配列情報を含む）"""
    return_type: ReturnType
    semantic_type: SemanticType
    is_array: bool = False

    def __str__(self) -> str:
        """文字列表現（デバッグ用）"""
        if self.is_array:
            return f"{self.semantic_type.value}_ARRAY"
        return self.semantic_type.value

    @property
    def variable_name_suffix(self) -> str:
        """変数名の接尾辞（配列の場合は's'を付加）"""
        return "s" if self.is_array else ""

    @property
    def actual_semantic_type(self) -> SemanticType:
        """実際のSemanticType（配列かどうかに関係なく元の型を返す）"""
        return self.semantic_type

    @classmethod
    def create_from_semantic_type(cls, semantic_type: SemanticType, is_array: bool = False) -> 'TypeInfo':
        """SemanticTypeからTypeInfoを作成（ReturnTypeを自動推定）"""
        # SemanticTypeからReturnTypeを推定
        return_type_mapping = {
            SemanticType.OBJECT: ReturnType.OBJECT,
            SemanticType.COLOR: ReturnType.INT,
            SemanticType.COORDINATE: ReturnType.INT,
            SemanticType.OFFSET: ReturnType.INT,
            SemanticType.SIZE: ReturnType.INT,
            SemanticType.DISTANCE: ReturnType.INT,
            SemanticType.DISTANCE_AXIS: ReturnType.INT,
            SemanticType.COUNT: ReturnType.INT,
            SemanticType.COUNT_HOLES: ReturnType.INT,
            SemanticType.COUNT_ADJACENT: ReturnType.INT,
            SemanticType.COUNT_OVERLAP: ReturnType.INT,
            SemanticType.LOOP_INDEX: ReturnType.INT,
            SemanticType.PERCENTAGE: ReturnType.INT,
            SemanticType.ASPECT_RATIO: ReturnType.INT,
            SemanticType.DENSITY: ReturnType.INT,
            SemanticType.ANGLE: ReturnType.INT,
            SemanticType.SCALE: ReturnType.INT,
            SemanticType.BOOL: ReturnType.BOOL,
            SemanticType.VOID: ReturnType.VOID,
            SemanticType.STRING: ReturnType.STRING,
            SemanticType.DIRECTION: ReturnType.STRING,
            SemanticType.AXIS: ReturnType.STRING,
            SemanticType.LINE_TYPE: ReturnType.STRING,
            SemanticType.RECT_TYPE: ReturnType.STRING,
            SemanticType.SIDE: ReturnType.STRING,
            SemanticType.ORDER: ReturnType.STRING,
            SemanticType.ALIGN_MODE: ReturnType.STRING,
        }

        return_type = return_type_mapping.get(semantic_type, ReturnType.INT)

        # 配列の場合はReturnTypeを調整
        if is_array:
            if return_type == ReturnType.OBJECT:
                return_type = ReturnType.OBJECT  # オブジェクト配列もOBJECTとして扱う
            elif return_type == ReturnType.INT:
                return_type = ReturnType.LIST_INT
            elif return_type == ReturnType.STRING:
                return_type = ReturnType.STRING  # 文字列配列もSTRINGとして扱う

        return cls(return_type=return_type, semantic_type=semantic_type, is_array=is_array)





class TypeSystem:
    """型システム（現行argument_generator.pyの_are_types_compatibleを継承）"""

    # ============================================================
    # 型の互換性マトリックス
    # ============================================================

    COMPATIBILITY_MATRIX: Dict[SemanticType, Set[SemanticType]] = {
        # COLOR - 完全独立
        SemanticType.COLOR: {
            SemanticType.COLOR,  # 自分自身のみ
        },

        # CORE_SPATIAL - 相互互換
        SemanticType.COORDINATE: {
            SemanticType.COORDINATE,
            SemanticType.OFFSET,
            SemanticType.SIZE,
            SemanticType.DISTANCE_AXIS,
            SemanticType.LOOP_INDEX,  # ループ変数と互換（算術演算で使用される）
        },
        SemanticType.OFFSET: {
            SemanticType.COORDINATE,
            SemanticType.OFFSET,
            SemanticType.SIZE,
            SemanticType.DISTANCE_AXIS,
            SemanticType.LOOP_INDEX,  # ループ変数と互換（算術演算で使用される）
        },
        SemanticType.SIZE: {
            SemanticType.COORDINATE,
            SemanticType.OFFSET,
            SemanticType.SIZE,
            SemanticType.DISTANCE_AXIS,
            SemanticType.LOOP_INDEX,  # ループ変数と互換（算術演算で使用される）
        },
        SemanticType.DISTANCE_AXIS: {
            SemanticType.COORDINATE,
            SemanticType.OFFSET,
            SemanticType.SIZE,
            SemanticType.DISTANCE_AXIS,
            SemanticType.LOOP_INDEX,  # ループ変数と互換（算術演算で使用される）
        },

        # MEASUREMENT - 自分自身のみ（制限的）
        SemanticType.DISTANCE: {
            SemanticType.DISTANCE,
        },

        # カウント系 -
        SemanticType.COUNT: {
            SemanticType.COUNT,  # 汎用カウント同士のみ互換
            SemanticType.LOOP_INDEX,  # ループ変数と互換
        },
        SemanticType.LOOP_INDEX: {
            SemanticType.LOOP_INDEX,  # ループ変数同士のみ互換
            SemanticType.COUNT,  # 汎用カウントと互換
            # CORE_SPATIALグループと互換（算術演算で使用されるため）
            SemanticType.COORDINATE,
            SemanticType.OFFSET,
            SemanticType.SIZE,
            SemanticType.DISTANCE_AXIS,
        },
        SemanticType.COUNT_HOLES: {
            SemanticType.COUNT_HOLES,  # 穴の数のみ互換
        },
        SemanticType.COUNT_ADJACENT: {
            SemanticType.COUNT_ADJACENT,  # 隣接ピクセル数のみ互換
        },
        SemanticType.COUNT_OVERLAP: {
            SemanticType.COUNT_OVERLAP,  # 重複ピクセル数のみ互換
        },

        # PERCENTAGE - 自分自身のみ
        SemanticType.PERCENTAGE: {
            SemanticType.PERCENTAGE,
        },

        # 比率型 - 自分自身のみ（基本値との比較で使用）
        SemanticType.ASPECT_RATIO: {
            SemanticType.ASPECT_RATIO,  # GET_ASPECT_RATIO同士のみ互換
        },
        SemanticType.DENSITY: {
            SemanticType.DENSITY,  # GET_DENSITY同士のみ互換
        },

        # その他 - 自分自身のみ
        SemanticType.ANGLE: {
            SemanticType.ANGLE,
        },
        SemanticType.SCALE: {
            SemanticType.SCALE,
        },
        SemanticType.BOOL: {
            SemanticType.BOOL,
        },
        SemanticType.VOID: {
            SemanticType.VOID,
        },

        # 文字列型 - 自分自身のみ
        SemanticType.STRING: {
            SemanticType.STRING,
        },
        SemanticType.DIRECTION: {
            SemanticType.DIRECTION,
        },
        SemanticType.AXIS: {
            SemanticType.AXIS,
        },
        SemanticType.LINE_TYPE: {
            SemanticType.LINE_TYPE,
        },
        SemanticType.RECT_TYPE: {
            SemanticType.RECT_TYPE,
        },
        SemanticType.SIDE: {
            SemanticType.SIDE,
        },
        SemanticType.ORDER: {
            SemanticType.ORDER,
        },
        SemanticType.ALIGN_MODE: {
            SemanticType.ALIGN_MODE,
        },

        # オブジェクト型 - 自分自身のみ
        SemanticType.OBJECT: {
            SemanticType.OBJECT,
        },

    }



    # ============================================================
    # 算術演算・比例演算の組み合わせ制約
    # ============================================================

    # 算術演算で使用可能な型（効率化）
    # 比例演算で使用可能な型（算術演算型 + COLOR + DISTANCE + BOOL + 比較専用型）
    # 比較専用型（PERCENTAGE, ASPECT_RATIO, DENSITY）は算術演算では使用しないが、
    # 比例演算（GREATER, LESS, EQUAL等）では比較に使用される
    # LINE_TYPEとRECT_TYPEは文字列型なので、EQUAL/NOT_EQUALでのみ使用可能（GREATER/LESSでは意味がない）

    # 算術演算可能な型: CORE_SPATIALグループ + カウント系
    ARITHMETIC_TYPES = {
        # CORE_SPATIALグループ
        SemanticType.COORDINATE,
        SemanticType.OFFSET,
        SemanticType.SIZE,
        SemanticType.DISTANCE_AXIS,
        # カウント系
        SemanticType.COUNT,
        SemanticType.LOOP_INDEX,  # ループ変数（算術演算で使用される）
        SemanticType.COUNT_HOLES,
        SemanticType.COUNT_ADJACENT,
        SemanticType.COUNT_OVERLAP,
    }

    # 比例演算可能な型（比較演算で使用可能な型）
    PROPORTIONAL_TYPES = {
        # CORE_SPATIALグループ
        SemanticType.COORDINATE,
        SemanticType.OFFSET,
        SemanticType.SIZE,
        SemanticType.DISTANCE_AXIS,
        # カウント系
        SemanticType.COUNT,
        SemanticType.COUNT_HOLES,
        SemanticType.COUNT_ADJACENT,
        SemanticType.COUNT_OVERLAP,
        SemanticType.LOOP_INDEX,  # ループ変数（算術演算で使用される）

        SemanticType.COLOR,
        SemanticType.DISTANCE,
        SemanticType.BOOL,
        # 比較専用型（算術演算では使用しない）
        SemanticType.PERCENTAGE,
        SemanticType.ASPECT_RATIO,
        SemanticType.DENSITY,
        # 形状タイプ（文字列型、EQUAL/NOT_EQUALでのみ使用可能）
        SemanticType.LINE_TYPE,
        SemanticType.RECT_TYPE,
        # その他の数値型
        SemanticType.ANGLE,
        SemanticType.SCALE,
    }
    # 型互換性マトリックスは基本的なCOMPATIBILITY_MATRIXで十分


    # SORT_BYの条件で使用可能な型とその重み:
    SORT_BY_TYPES = {
        # CORE_SPATIALグループ（高確率）
        SemanticType.SIZE: 5.0,              # 最も一般的（サイズでソート）
        SemanticType.COORDINATE: 3.0,        # 座標でソート（位置関係）
        SemanticType.OFFSET: 2.0,            # オフセットでソート
        SemanticType.DISTANCE_AXIS: 1.5,     # 距離軸でソート
        # カウント系（中確率）
        SemanticType.COUNT: 2.5,             # カウントでソート
        SemanticType.COUNT_HOLES: 1.0,      # 穴の数でソート
        SemanticType.COUNT_ADJACENT: 1.0,   # 隣接数でソート
        SemanticType.COUNT_OVERLAP: 1.0,    # 重複数でソート
        # その他（中確率）
        SemanticType.ASPECT_RATIO: 2.0,      # アスペクト比でソート
        SemanticType.DENSITY: 1.5,           # 密度でソート
        SemanticType.PERCENTAGE: 1.0,        # パーセンテージでソート
    }

    # ============================================================
    # 型互換性チェック
    # ============================================================

    @staticmethod
    def are_compatible(type1: SemanticType, type2: SemanticType) -> bool:
        """2つの型が互換性があるかチェック

        Args:
            type1: 型1
            type2: 型2

        Returns:
            互換性があればTrue

        例:
            are_compatible(COORDINATE, SIZE) → True
            are_compatible(COLOR, COORDINATE) → False
        """
        if type1 is None or type2 is None:
            return True  # Noneは任意の型と互換（デフォルト）

        return type2 in TypeSystem.COMPATIBILITY_MATRIX.get(type1, set())

    @staticmethod
    def is_compatible(type1: SemanticType, type2: SemanticType) -> bool:
        """2つの型が互換性があるかチェック（are_compatibleのエイリアス）

        Args:
            type1: 型1
            type2: 型2

        Returns:
            互換性があればTrue
        """
        return TypeSystem.are_compatible(type1, type2)

    @staticmethod
    def get_compatible_types(semantic_type: SemanticType) -> List[SemanticType]:
        """指定された型と互換性のある型のリストを取得

        Args:
            semantic_type: 型

        Returns:
            互換性のある型のリスト
        """
        return list(TypeSystem.COMPATIBILITY_MATRIX.get(semantic_type, set()))


    @staticmethod
    def can_mix_in_operation(type1: SemanticType, type2: SemanticType) -> bool:
        """2つの型を同じ演算・比較で使用可能かチェック

        ★重要★: COLOR型の完全分離を保証

        Args:
            type1: 型1
            type2: 型2

        Returns:
            同じ演算・比較で使用可能ならTrue

        例:
            can_mix_in_operation(COLOR, COLOR) → True
            can_mix_in_operation(COLOR, COORDINATE) → False (0%)
            can_mix_in_operation(COORDINATE, SIZE) → True
        """
        if type1 is None or type2 is None:
            return True

        # COLORが含まれる場合、両方がCOLORでなければならない
        if type1 == SemanticType.COLOR or type2 == SemanticType.COLOR:
            return type1 == SemanticType.COLOR and type2 == SemanticType.COLOR

        # それ以外は互換性マトリックスに従う
        return TypeSystem.are_compatible(type1, type2)

    @staticmethod
    def filter_compatible_commands(
        target_type: SemanticType,
        available_commands: List[str],
        command_metadata: Dict[str, 'CommandMetadata']
    ) -> List[str]:
        """指定された型と互換性のあるコマンドのみをフィルタリング

        Args:
            target_type: 目標の型
            available_commands: 候補コマンドのリスト
            command_metadata: コマンドメタデータ

        Returns:
            型互換なコマンドのリスト

        例:
            target_type = COLOR
            available = ['GET_COLOR', 'GET_X', 'GET_Y']
            → ['GET_COLOR'] のみ（GET_X, GET_Yは除外）
        """
        compatible = []

        for cmd_name in available_commands:
            cmd = command_metadata.get(cmd_name)
            if not cmd:
                continue

            # 戻り値の意味的な型を取得
            cmd_return_type = getattr(cmd, 'semantic_return_type', None)
            if cmd_return_type is None:
                # 型情報がない場合は互換とみなす
                compatible.append(cmd_name)
                continue

            # 型互換性をチェック
            if TypeSystem.can_mix_in_operation(target_type, cmd_return_type):
                compatible.append(cmd_name)

        return compatible

    @staticmethod
    def get_default_range(semantic_type: SemanticType, grid_width: Optional[int] = None, grid_height: Optional[int] = None) -> tuple:
        """型のデフォルト値範囲を取得

        Args:
            semantic_type: 型
            grid_width: グリッド幅（指定されていない場合はデフォルト値を使用）
            grid_height: グリッド高さ（指定されていない場合はデフォルト値を使用）

        Returns:
            (min_val, max_val) のタプル
        """
        # デフォルトのグリッドサイズ
        max_grid_size = max(grid_width or 30, grid_height or 30)

        if semantic_type == SemanticType.COLOR:
            return (0, 9)

        elif semantic_type == SemanticType.COORDINATE:
            # グリッドサイズ依存: 0 から max(grid_width, grid_height) - 1
            return (0, max_grid_size - 1)

        elif semantic_type == SemanticType.OFFSET:
            # OFFSETは負の値も取る（dx, dy、距離）
            # グリッドサイズの50%を最大値とする（移動距離制限）
            max_offset = int(max_grid_size * 0.5)
            return (-max_offset, max_offset)

        elif semantic_type == SemanticType.SIZE:
            # グリッドサイズに応じて範囲を調整（最小1、最大グリッドサイズ）
            return (1, max_grid_size)

        elif semantic_type in {SemanticType.DISTANCE, SemanticType.DISTANCE_AXIS}:
            # グリッドサイズに応じて範囲を調整
            return (0, max_grid_size)

        elif semantic_type == SemanticType.COUNT:
            return (0, 10)

        elif semantic_type == SemanticType.LOOP_INDEX:
            # ループ変数は通常0から配列の長さ-1までの範囲
            # グリッドサイズに応じて範囲を調整（最大でもグリッドサイズ程度）
            return (0, max_grid_size - 1)

        elif semantic_type == SemanticType.COUNT_HOLES:
            # 穴の数は通常少ない（0-10程度）
            return (0, 10)

        elif semantic_type == SemanticType.COUNT_ADJACENT:
            # 隣接ピクセル数はグリッドサイズに依存（最大でもグリッドサイズ程度）
            return (0, max_grid_size)

        elif semantic_type == SemanticType.COUNT_OVERLAP:
            # 重複ピクセル数はグリッドサイズに依存（最大でもグリッドサイズ程度）
            return (0, max_grid_size)

        elif semantic_type == SemanticType.PERCENTAGE:
            return (0, 100)

        elif semantic_type == SemanticType.ASPECT_RATIO:
            # アスペクト比（幅/高さ * 100）
            # 1~default_heightから乱数で高さを選び、1~default_widthから乱数で幅を選んで計算
            default_width = grid_width or 30
            default_height = grid_height or 30
            # 1~default_heightから乱数で高さを選ぶ
            height = random.randint(1, default_height)
            # 1~default_widthから乱数で幅を選ぶ
            width = random.randint(1, default_width)
            # アスペクト比を計算: (幅/高さ) * 100
            aspect_ratio_val = int((width / height) * 100) if height > 0 else 1
            # 範囲ではなく計算された値を返す（特別な処理のため、タプルで返す）
            return (aspect_ratio_val, aspect_ratio_val)

        elif semantic_type == SemanticType.DENSITY:
            # 密度（ピクセル数/bbox面積 * 100）、0-100
            return (0, 100)

        elif semantic_type == SemanticType.ANGLE:
            return None  # 選択肢のみ（0, 90, 180, 270）

        elif semantic_type == SemanticType.SCALE:
            return (2, 4)

        else:
            return (0, 30)  # デフォルト

    @staticmethod
    def get_literal_choices(semantic_type: SemanticType, is_array: bool = False) -> Optional[List[Any]]:
        """リテラル値の選択肢を取得（BOOL型、OBJECT型、文字列型に対応）

        Args:
            semantic_type: 型
            is_array: 配列かどうか

        Returns:
            選択肢のリスト（該当しない場合はNone）
        """
        # BOOL型
        if semantic_type == SemanticType.BOOL:
            if is_array:
                return None  # BOOL型の配列はリテラル値を生成しない
            return [True, False]

        # OBJECT型
        if semantic_type == SemanticType.OBJECT:
            if is_array:
                return ["objects"]  # オブジェクト配列: objects
            else:
                return ["objects[0]"]  # オブジェクト単体: objects[0]

        # 文字列型
        if semantic_type == SemanticType.DIRECTION:
            # エグゼキュータ準拠: 8方向 + C（GET_CENTROID, GET_DIRECTIONが"C"を返す可能性がある）
            return ["X", "Y", "-X", "-Y", "XY", "-XY", "X-Y", "-X-Y", "C"]

        elif semantic_type == SemanticType.AXIS:
            # エグゼキュータ準拠: X, Y のみ
            return ["X", "Y"]

        elif semantic_type == SemanticType.LINE_TYPE:
            # エグゼキュータの実装に準拠（core.py:5083-5132）
            return ["X", "Y", "XY", "-XY", "C"]

        elif semantic_type == SemanticType.RECT_TYPE:
            # エグゼキュータの実装に準拠（core.py:5138-5195）
            return ["filled", "hollow", "none"]

        elif semantic_type == SemanticType.SIDE:
            return ["start", "end"]

        elif semantic_type == SemanticType.ORDER:
            return ["asc", "desc"]

        elif semantic_type == SemanticType.ANGLE:
            # 数値だが選択肢で表現（数値として返す）
            return [0, 90, 180, 270]

        elif semantic_type == SemanticType.ALIGN_MODE:
            return ["left", "right", "top", "bottom", "center_x", "center_y", "center"]

        return None



    @staticmethod
    def are_arithmetically_compatible(type1: SemanticType, type2: SemanticType) -> bool:
        """算術演算で互換性があるかチェック

        Args:
            type1: 型1
            type2: 型2

        Returns:
            算術演算で互換性があればTrue
        """
        if type1 is None or type2 is None:
            return True

        # COMPATIBILITY_MATRIXを使用して型互換性をチェック
        # 算術演算で使える型は、COMPATIBILITY_MATRIXで互換性がある型のみ
        return TypeSystem.are_compatible(type1, type2)

    @staticmethod
    def are_proportionally_compatible(type1: SemanticType, type2: SemanticType) -> bool:
        """比例演算で互換性があるかチェック

        Args:
            type1: 型1
            type2: 型2

        Returns:
            比例演算で互換性があればTrue
        """
        if type1 is None or type2 is None:
            return True

        # 比例演算で使用可能な型かチェック
        if type1 not in TypeSystem.PROPORTIONAL_TYPES or type2 not in TypeSystem.PROPORTIONAL_TYPES:
            return False

        # 基本的な型互換性のみチェック
        return type2 in TypeSystem.COMPATIBILITY_MATRIX.get(type1, set())

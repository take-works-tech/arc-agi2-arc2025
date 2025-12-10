"""
引数スキーマ定義

各コマンドの引数を統一的に定義
"""
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Any, Dict, Tuple
import random
from .types import ReturnType, SemanticType, TypeInfo
from .variable_manager import variable_manager


# ========================================
# 動的範囲計算関数
# ========================================

def _generate_offset_with_third_probability(grid_size: int, axis: str) -> int:
    """オフセット値を生成（1/3の確率を高く）

    Args:
        grid_size: グリッドサイズ
        axis: 軸（'x' or 'y'）

    Returns:
        オフセット値
    """
    # 最大範囲（グリッドサイズの50%）
    max_offset = int(grid_size * 0.5)

    # 1/3の範囲
    third_offset = grid_size // 3

    # 確率分布: 1/3範囲内 70%, 1/3範囲外 30%
    if random.random() < 0.7:
        # 1/3範囲内（高確率）- 0を除外
        values = [x for x in range(-third_offset, third_offset + 1) if x != 0]
        if values:  # 空でないことを確認
            return random.choice(values)
        else:
            # フォールバック: 単純な範囲
            return random.choice([-1, 1])
    else:
        # 1/3範囲外（低確率）
        # 1/3範囲外の値を生成（ただし最大50%まで）
        if random.random() < 0.5:
            # 負の値
            return random.randint(-max_offset, -third_offset - 1)
        else:
            # 正の値
            return random.randint(third_offset + 1, max_offset)





@dataclass
class ArgumentSchema:
    """引数のスキーマ定義"""

    # ========================================
    # 基本情報
    # ========================================
    type_info: TypeInfo                 # 型情報（ReturnType、SemanticType、Is array）
    required: bool = True               # 必須かオプションか

    # ========================================
    # 生成確率
    # ========================================
    literal_prob: float = 0.30           # リテラル値生成確率（変数使用率向上のためさらに下げる）
    variable_prob: float = 0.50         # 変数生成確率（変数使用率向上のためさらに上げる）
    nested_prob: float = 0.20           # ネストしたコマンド生成確率

    # ========================================
    # 値の範囲・選択肢
    # ========================================
    range_min: Optional[int] = None     # 最小値（int型の場合）
    range_max: Optional[int] = None     # 最大値（int型の場合）
    choices: Optional[List[str]] = None # 選択肢（string型や固定値の場合）

    # ========================================
    # その他
    # ========================================
    description: Optional[str] = None   # 説明
    depends_on_grid_size: bool = False  # グリッドサイズに依存するか
    range_calculator: Optional[Callable] = None  # 範囲計算関数
    allowed_nested_commands: Optional[List[str]] = None  # 許可されるネストしたコマンド
    custom_generator: Optional[Callable] = None  # カスタム生成関数
    nesting_control: Optional[Dict] = None  # ネスト制御
    # 追加: 代替で許可するSemanticType（is_arrayはtype_infoに従う）
    allowed_alt_semantic_types: Optional[List[SemanticType]] = None

    def __post_init__(self):
        """初期化後にnameを動的に生成"""
        self.name = variable_manager.get_argument_name(self.type_info)

    def get_range(self, context: Optional[Dict] = None) -> tuple:
        """範囲を取得（動的計算対応）

        Args:
            context: コンテキスト（グリッドサイズ等の情報）

        Returns:
            (min_val, max_val) のタプル
        """
        # COLOR型の場合は常に0-9を返す
        from .types import SemanticType
        if self.type_info and self.type_info.semantic_type == SemanticType.COLOR:
            return (0, 9)

        # 動的計算が設定されている場合
        if self.range_calculator and context:
            range_result = self.range_calculator(context)
            # COLOR型の場合は0-9に制限
            if self.type_info and self.type_info.semantic_type == SemanticType.COLOR:
                range_min, range_max = range_result
                range_min = max(0, range_min)
                range_max = min(9, range_max)
                if range_min > range_max:
                    range_min, range_max = 0, 9
                return (range_min, range_max)
            return range_result

        # 固定範囲
        if self.range_min is not None and self.range_max is not None:
            # COLOR型の場合は0-9に制限
            if self.type_info and self.type_info.semantic_type == SemanticType.COLOR:
                range_min = max(0, self.range_min)
                range_max = min(9, self.range_max)
                if range_min > range_max:
                    range_min, range_max = 0, 9
                return (range_min, range_max)
            return (self.range_min, self.range_max)

        # デフォルト範囲（semantic_typeから推論）
        if self.type_info.semantic_type:
            from .types import TypeSystem
            # グリッドサイズをcontextから取得
            grid_width = context.get('output_grid_width') if context else None
            grid_height = context.get('output_grid_height') if context else None
            default_range = TypeSystem.get_default_range(self.type_info.semantic_type, grid_width, grid_height)
            if default_range:
                return default_range

        # フォールバック（グリッドサイズを使用）
        max_grid_size = max(context.get('output_grid_width', 30) if context else 30,
                           context.get('output_grid_height', 30) if context else 30)
        # COLOR型の場合は0-9を返す
        if self.type_info and self.type_info.semantic_type == SemanticType.COLOR:
            return (0, 9)
        return (0, max_grid_size)


class CompatibleArgumentSchema(ArgumentSchema):
    """互換性のある型を動的に選択する引数スキーマ"""

    # ========================================
    # 基本情報
    # ========================================
    base_type: SemanticType             # 基準となる型
    compatible_types: Optional[List[SemanticType]] = None  # 互換性のある型のリスト

    # 親クラスの引数をオーバーライド
    type_info: TypeInfo = field(default_factory=lambda: TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False))

    def __post_init__(self):
        """初期化後に互換性のある型を設定"""
        from .types import TypeSystem

        # 互換性のある型を取得
        if self.compatible_types is None:
            self.compatible_types = list(TypeSystem.COMPATIBILITY_MATRIX.get(self.base_type, {self.base_type}))

        # 基準型を互換性リストに含める
        if self.base_type not in self.compatible_types:
            self.compatible_types.append(self.base_type)

        # デフォルトの型情報を設定（実行時に動的に変更される）
        # 注意: is_arrayフラグは動的に決定されるべきだが、デフォルトはFalse
        # 実際の使用時はget_type_info()メソッドで正しいis_arrayフラグが返される
        self.type_info = TypeInfo.create_from_semantic_type(self.base_type, is_array=False)
        self.name = variable_manager.get_argument_name(self.type_info)

    def get_compatible_type(self) -> SemanticType:
        """互換性のある型をランダムに選択"""
        return random.choice(self.compatible_types)

    def get_type_info(self) -> TypeInfo:
        """現在選択されている型のTypeInfoを取得"""
        selected_type = self.get_compatible_type()
        # 元のtype_infoのis_arrayフラグを保持
        return TypeInfo.create_from_semantic_type(selected_type, self.type_info.is_array)


    # ========================================
    # 値の範囲・選択肢
    # ========================================
    range_min: Optional[int] = None     # 最小値（int型の場合）
    range_max: Optional[int] = None     # 最大値（int型の場合）
    choices: Optional[List[str]] = None # 選択肢（string型や固定値の場合）

    # 動的な範囲計算（グリッドサイズ依存の場合）
    range_calculator: Optional[Callable[[Dict], tuple]] = None

    # カスタム生成関数（確率分布制御等）
    custom_generator: Optional[Callable[[Dict], Any]] = None

    # ========================================
    # 生成確率（3つの選択肢の確率）
    # ========================================
    literal_prob: float = 0.30          # リテラル値の確率（定数を下げる）
    variable_prob: float = 0.50         # 変数使用の確率（変数を上げる）
    nested_prob: float = 0.20           # ネストコマンドの確率

    # ========================================
    # 制約
    # ========================================
    depends_on_grid_size: bool = False  # グリッドサイズに依存するか

    # ネストコマンドの制約
    allowed_nested_commands: Optional[List[str]] = None     # 使用可能なネストコマンド

    # ネスト制御（FILTER、MATCH_PAIRS、SORT_BY用）
    nesting_control: Optional[Dict[str, Any]] = None        # ネスト制御設定

    # ========================================
    # メタ情報
    # ========================================
    description: str = ""               # 説明
    examples: List[str] = field(default_factory=list)  # 使用例

    def get_range(self, context: Optional[Dict] = None) -> tuple:
        """範囲を取得（動的計算対応）

        Args:
            context: コンテキスト（グリッドサイズ等の情報）

        Returns:
            (min_val, max_val) のタプル
        """
        # 動的計算が設定されている場合
        if self.range_calculator and context:
            return self.range_calculator(context)

        # 固定範囲
        if self.range_min is not None and self.range_max is not None:
            return (self.range_min, self.range_max)

        # デフォルト範囲（semantic_typeから推論）
        if self.type_info.semantic_type:
            from .types import TypeSystem
            default_range = TypeSystem.get_default_range(self.type_info.semantic_type)
            if default_range:
                return default_range

        # フォールバック
        return (0, 30)

    def get_choices(self) -> Optional[List[str]]:
        """選択肢を取得

        Returns:
            選択肢のリスト（選択肢がない場合はNone）
        """
        # 明示的な選択肢
        if self.choices:
            return self.choices

        # semantic_typeから推論
        if self.type_info.semantic_type:
            from .types import TypeSystem
            # get_literal_choicesを使用し、文字列型のみを返す
            result = TypeSystem.get_literal_choices(
                self.type_info.semantic_type,
                is_array=self.type_info.is_array
            )
            if result is None:
                return None
            # すべての要素が文字列型かチェック
            if all(isinstance(item, str) for item in result):
                return result  # type: ignore
            return None

        return None

    def validate_value(self, value: Any) -> bool:
        """値が有効かチェック

        Args:
            value: チェックする値

        Returns:
            有効ならTrue
        """
        # 選択肢がある場合
        choices = self.get_choices()
        if choices:
            return str(value) in choices or value in choices

        # 範囲チェック（int型の場合）
        if self.type_info.return_type == ReturnType.INT:
            range_vals = self.get_range()
            if isinstance(value, int):
                return range_vals[0] <= value <= range_vals[1]

        return True

    def is_compatible_with(self, other_type_info: TypeInfo, allow_color_comparison: bool = False, is_arithmetic_operation: bool = False, is_proportional_operation: bool = False) -> bool:
        """他の型情報との互換性をチェック

        Args:
            other_type_info: 比較する型情報
            allow_color_comparison: 色の比較を許可するか（比較演算用）
            is_arithmetic_operation: 算術演算かどうか
            is_proportional_operation: 比例演算かどうか

        Returns:
            互換性がある場合True
        """
        from .types import TypeSystem

        # 算術演算の場合、専用の互換性チェックを使用（is_arrayフラグも考慮）
        if is_arithmetic_operation:
            return (TypeSystem.are_arithmetically_compatible(
                self.type_info.semantic_type,
                other_type_info.semantic_type
            ) and self.type_info.is_array == other_type_info.is_array)

        # 比例演算の場合、専用の互換性チェックを使用（is_arrayフラグも考慮）
        if is_proportional_operation:
            return (TypeSystem.are_proportionally_compatible(
                self.type_info.semantic_type,
                other_type_info.semantic_type
            ) and self.type_info.is_array == other_type_info.is_array)

        # 基本的な型互換性チェック（is_arrayフラグも考慮）
        if (TypeSystem.is_compatible(self.type_info.semantic_type, other_type_info.semantic_type) and
            self.type_info.is_array == other_type_info.is_array):
                return True

        # 代替で許可するSemanticType（LENの色配列など）
        if self.allowed_alt_semantic_types:
            if (other_type_info.semantic_type in self.allowed_alt_semantic_types and
            self.type_info.is_array == other_type_info.is_array):
                return True

        # 比較演算の場合、COLOR型との比較も許可
        if allow_color_comparison:
            from .types import SemanticType
            if (self.type_info.semantic_type == SemanticType.COLOR and
                other_type_info.semantic_type == SemanticType.COLOR):
                return True

        return False

    def get_compatible_semantic_types(self) -> List[SemanticType]:
        """互換性のあるSemanticTypeのリストを取得

        Returns:
            互換性のあるSemanticTypeのリスト
        """
        from .types import TypeSystem
        # 注意: このメソッドはSemanticTypeのみを返すため、is_arrayフラグは考慮されない
        # 完全な型互換性チェックにはis_compatible_withメソッドを使用すること
        return TypeSystem.get_compatible_types(self.type_info.semantic_type)

    # 使用されていない複雑な型互換性関数は削除



# ============================================================
# 共通の引数スキーマ（再利用可能）
# ============================================================

# オブジェクト引数（ほぼすべてのコマンドで使用）
def create_object_arg() -> ArgumentSchema:
    """オブジェクト引数を作成"""
    return ArgumentSchema(
        type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
        literal_prob=0.0,  # 常に変数または配列要素
        variable_prob=1.0,
        nested_prob=0.0,
        description='対象オブジェクト',
    )

OBJECT_ARG = create_object_arg()

# 座標引数（TELEPORT, PATHFIND, CREATE_LINE, CREATE_RECT等で使用）
def create_coordinate_arg(is_x: bool = True) -> ArgumentSchema:
    """座標引数を作成"""
    description = 'X座標' if is_x else 'Y座標'
    range_calculator = (lambda ctx: (0, ctx.get('output_grid_width', 30) - 1)) if is_x else (lambda ctx: (0, ctx.get('output_grid_height', 30) - 1))

    return ArgumentSchema(
        type_info=TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False),
        range_min=0,
        range_max=29,
        depends_on_grid_size=True,
        range_calculator=range_calculator,
        literal_prob=0.30,  # リテラル値の確率（変数使用率向上のためさらに下げる）
        variable_prob=0.50,  # 変数確率（変数使用率向上のためさらに上げる）
        nested_prob=0.20,    # ネストコマンドの確率（実際の確率: (1-0.30)×(1-0.50) = 35%）
        allowed_nested_commands=None,  # 型互換性チェックで自動選択
        description=description,
    )

COORDINATE_X_ARG = create_coordinate_arg(True)  # X座標
COORDINATE_Y_ARG = create_coordinate_arg(False)  # Y座標

# オフセット引数（MOVE, DRAW等で使用）
OFFSET_DX_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.OFFSET, is_array=False),
    range_min=-10,
    range_max=10,
    depends_on_grid_size=True,
    range_calculator=lambda ctx: (
        -int(ctx.get('output_grid_width', 30) * 0.5),
        int(ctx.get('output_grid_width', 30) * 0.5)
    ),
    custom_generator=lambda ctx: _generate_offset_with_third_probability(
        ctx.get('output_grid_width', 30), 'x'
    ),
    literal_prob=0.75,  # リテラル値の確率（実際の確率: 75%）
    variable_prob=0.10,  # 変数確率（実際の確率: (1-0.75)×0.10 = 2.5%）
    nested_prob=0.25,    # ネストコマンドの確率（実際の確率: (1-0.75)×(1-0.10) = 37.5%）
    allowed_nested_commands=None,  # 型互換性チェックで自動選択
    description='X方向のオフセット（グリッドサイズの50%以内、1/3の確率を高く）',
)

OFFSET_DY_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.OFFSET, is_array=False),
    range_min=-10,
    range_max=10,
    depends_on_grid_size=True,
    range_calculator=lambda ctx: (
        -int(ctx.get('output_grid_height', 30) * 0.5),
        int(ctx.get('output_grid_height', 30) * 0.5)
    ),
    custom_generator=lambda ctx: _generate_offset_with_third_probability(
        ctx.get('output_grid_height', 30), 'y'
    ),
    literal_prob=0.75,  # リテラル値の確率（実際の確率: 75%）
    variable_prob=0.10,  # 変数確率（実際の確率: (1-0.75)×0.10 = 2.5%）
    nested_prob=0.25,    # ネストコマンドの確率（実際の確率: (1-0.75)×(1-0.10) = 37.5%）
    allowed_nested_commands=None,  # 型互換性チェックで自動選択
    description='Y方向のオフセット（グリッドサイズの50%以内、1/3の確率を高く）',
)

# 色引数（SET_COLOR, FILL_HOLES等で使用）
COLOR_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.COLOR, is_array=False),
    range_min=0,
    range_max=9,
    literal_prob=0.30,  # リテラル値の確率（変数使用率向上のためさらに下げる）
    variable_prob=0.50,  # 変数確率（変数使用率向上のためさらに上げる）
    nested_prob=0.20,    # ネストコマンドの確率（実際の確率: (1-0.30)×(1-0.50) = 35%）
    allowed_nested_commands=None,  # 型互換性チェックで自動選択
    description='色（0-9）',
)

# 真偽値引数（AND, OR等で使用）
# 注意: リテラル値の選択肢（True/False）はtypes.pyのTypeSystem.get_literal_choices()で管理
BOOL_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    literal_prob=0.01,  # 20%の確率でリテラル（True/False）を許可
    variable_prob=0.10, # 10%の確率で変数
    nested_prob=0.89,   # 70%の確率でネストしたコマンド（比較演算子、論理演算子など）
    # choicesは削除: types.pyのTypeSystem.get_literal_choices()で管理
    allowed_nested_commands=None,  # 型互換性チェックで自動選択
    description='真偽値（True/False）',
)

# サイズ引数（SCALE, EXPAND等で使用）
SIZE_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    range_min=1,
    range_max=30,
    depends_on_grid_size=True,
    range_calculator=lambda ctx: (1, min(ctx.get('output_grid_width', 30), ctx.get('output_grid_height', 30))),
    literal_prob=0.30,  # リテラル値の確率（変数使用率向上のためさらに下げる）
    variable_prob=0.50,  # 変数確率（変数使用率向上のためさらに上げる）
    nested_prob=0.20,    # ネストコマンドの確率（実際の確率: (1-0.30)×(1-0.50) = 35%）
    allowed_nested_commands=None,  # 型互換性チェックで自動選択
    description='サイズ',
)

# スケール引数（SCALE, SCALE_DOWN等で使用）
SCALE_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.SCALE, is_array=False),
    range_min=2,
    range_max=4,
    custom_generator=lambda ctx: random.choices(
        [2, 3, 4],
        weights=[0.82, 0.17, 0.01]
    )[0],
    literal_prob=0.30,  # リテラル値の確率（変数使用率向上のためさらに下げる）
    variable_prob=0.50,  # 変数確率（変数使用率向上のためさらに上げる）
    nested_prob=0.0,     # ネストコマンドの確率を0に設定（SCALE型はコマンドを選択しない）
    allowed_nested_commands=None,  # nested_prob=0.0のため使用されない
    description='スケール係数（2倍82%, 3倍17%, 4倍1%、グリッド爆発抑制）',
)

# 角度引数（ROTATE専用）
ANGLE_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.ANGLE, is_array=False),
    choices=[90, 180, 270],  # 0度を削除（無意味な回転を防ぐ）
    literal_prob=1.0,  # 常にリテラル
    variable_prob=0.0,
    nested_prob=0.0,
    description='回転角度',
)

# 軸引数（FLIP, GET_SYMMETRY_SCORE等で使用）
AXIS_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.AXIS, is_array=False),
    choices=["X", "Y"],  # エグゼキュータ準拠
    literal_prob=1.0,
    variable_prob=0.0,
    nested_prob=0.0,
    description='軸（X or Y）',
)

# 方向引数（統一版：FLOW, SLIDE, LAY, CREATE_LINEで使用）
# 各コマンドでの動作:
# - FLOW: 4方向（X, Y, -X, -Y）のみ対応、斜め方向（XY, -XY, X-Y, -X-Y）は自動的にCに変換
# - SLIDE, LAY: 8方向対応、Cは変化なし
# - CREATE_LINE: 8方向対応、Cは点生成
DIRECTION_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.DIRECTION, is_array=False),
    choices=["X", "Y", "-X", "-Y", "XY", "-XY", "X-Y", "-X-Y", "C"],  # 8方向 + C
    literal_prob=0.50,  # リテラル値の確率（変数使用率向上のためさらに下げる）
    variable_prob=0.30,  # 変数確率（変数使用率向上のためさらに上げる）
    nested_prob=0.20,  # ネストコマンドの確率（実際の確率: (1-0.50)×(1-0.30) = 35%）
    allowed_nested_commands=None,  # 型互換性チェックで自動選択
    description='方向（8方向 + C）',
)

# 順序引数（SORT_BY専用）
ORDER_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.ORDER, is_array=False),
    choices=["asc", "desc"],
    literal_prob=1.0,
    variable_prob=0.0,
    nested_prob=0.0,
    description='ソート順序',
)

# 整列モード引数（ALIGN専用）
ALIGN_MODE_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.ALIGN_MODE, is_array=False),
    choices=["left", "right", "top", "bottom", "center_x", "center_y", "center"],
    literal_prob=1.0,
    variable_prob=0.0,
    nested_prob=0.0,
    description='整列モード',
)

# オブジェクト配列引数
OBJECT_ARRAY_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    literal_prob=0.0,  # 常に変数
    variable_prob=1.0,
    nested_prob=0.0,
    description='オブジェクト配列',
)

# 色配列引数（LEN等で使用）
COLOR_ARRAY_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.COLOR, is_array=True),
    literal_prob=0.0,  # 常に変数
    variable_prob=1.0,
    nested_prob=0.0,
    description='色配列',
)

# LEN専用引数（オブジェクト配列のみ）
LEN_ARRAY_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    literal_prob=0.0,  # 常に変数
    variable_prob=1.0,
    nested_prob=0.0,
    description='オブジェクト配列',
    allowed_nested_commands=None,  # LENは配列変数のみ使用
)

# 連結性引数（GET_ALL_OBJECTS, SPLIT_CONNECTED専用）
CONNECTIVITY_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.COUNT, is_array=False),
    choices=[4, 8],  # 文字列ではなく整数に修正
    literal_prob=1.0,
    variable_prob=0.0,
    nested_prob=0.0,
    description='連結性（4 or 8）',
)


# ピクセル数引数（EXPAND, SHRINK専用）
PIXELS_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    range_min=1,
    range_max=5,
    literal_prob=0.30,  # リテラル値の確率（変数使用率向上のためさらに下げる）
    variable_prob=0.50,  # 変数確率（変数使用率向上のためさらに上げる）
    nested_prob=0.20,    # ネストコマンドの確率（実際の確率: (1-0.30)×(1-0.50) = 35%）
    allowed_nested_commands=None,  # 型互換性チェックで自動選択
    description='拡張/縮小ピクセル数',
)

# 長さ引数（CREATE_LINE専用）
LENGTH_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    range_min=2,
    range_max=25,
    depends_on_grid_size=True,
    range_calculator=lambda ctx: (2, min(ctx.get('output_grid_width', 30), ctx.get('output_grid_height', 30))),
    literal_prob=0.30,  # リテラル値の確率（変数使用率向上のためさらに下げる）
    variable_prob=0.50,  # 変数確率（変数使用率向上のためさらに上げる）
    nested_prob=0.20,    # ネストコマンドの確率（実際の確率: (1-0.30)×(1-0.50) = 35%）
    allowed_nested_commands=None,  # 型互換性チェックで自動選択
    description='線の長さ',
)


# キー式引数（SORT_BY専用）
KEY_EXPR_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    literal_prob=0.0,
    variable_prob=0.0,
    nested_prob=1.0,
    allowed_nested_commands=None,  # 型互換性チェックで自動選択（SORT_BYでは様々な型が使用可能）
    description='ソートキー式（$objプレースホルダー使用）',
)

# 側引数（EXTEND_PATTERN専用）
SIDE_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.SIDE, is_array=False),
    choices=["start", "end"],
    literal_prob=1.0,
    description='延長する側',
)

# カウント引数（EXTEND_PATTERN専用）
COUNT_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.COUNT, is_array=False),
    range_min=1,
    range_max=5,
    literal_prob=0.50,  # リテラル値の確率（変数使用率向上のため下げる、繰り返し回数なので50%に設定）
    variable_prob=0.30,  # 変数確率（変数使用率向上のため上げる）
    nested_prob=0.20,  # ネストコマンドの確率（実際の確率: (1-0.50)×(1-0.30) = 35%）
    description='繰り返し回数',
)

# 条件引数（FILTER専用）
# 部分プログラム生成で使えるコマンドのみを許可
CONDITION_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    literal_prob=0.0,
    variable_prob=0.0,
    nested_prob=1.0,
    allowed_nested_commands=None,  # 型互換性チェックで自動選択（BOOL型を返すコマンドのみ）
    description='フィルタ条件（$objプレースホルダー使用）',
)

# MATCH_PAIRS用の条件引数
MATCH_PAIRS_CONDITION_ARG = ArgumentSchema(
    type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    literal_prob=0.0,
    variable_prob=0.0,
    nested_prob=1.0,
    allowed_nested_commands=None,  # 型互換性チェックで自動選択（BOOL型を返すコマンドのみ）
    # 注意: ネストしたコマンドの引数（例: EQUAL(GET_COLOR($obj1), GET_COLOR($obj2))のGET_COLOR）は、
    # それぞれの引数スキーマ（OBJECT_ARGなど）に基づいて生成されるため、型互換性チェックで適切に処理される
    description='ペアマッチング条件（$obj1, $obj2プレースホルダー使用）',
)

# ArgumentSchemaクラスにminimal_naming_systemを活用するメソッドを追加
def add_naming_system_methods():
    """ArgumentSchemaクラスにminimal_naming_systemを活用するメソッドを追加"""

    def get_argument_name_from_naming_system(self) -> str:
        """minimal_naming_systemから引数名を取得

        Returns:
            引数名
        """
        return variable_manager.get_argument_name(self.type_info)

    def create_with_position(self) -> 'ArgumentSchema':
        """新しいArgumentSchemaを作成

        Returns:
            新しいArgumentSchemaインスタンス
        """
        return ArgumentSchema(
            type_info=self.type_info,
            required=self.required,
            range_min=self.range_min,
            range_max=self.range_max,
            choices=self.choices,
            range_calculator=self.range_calculator,
            custom_generator=self.custom_generator,
            literal_prob=self.literal_prob,
            variable_prob=self.variable_prob,
            nested_prob=self.nested_prob,
            depends_on_grid_size=self.depends_on_grid_size,
            allowed_nested_commands=self.allowed_nested_commands,
            description=self.description,
            examples=self.examples,
        )

    # メソッドをArgumentSchemaクラスに追加
    ArgumentSchema.get_argument_name_from_naming_system = get_argument_name_from_naming_system
    ArgumentSchema.create_with_position = create_with_position

# コマンド別の引数定義（コマンドクイックリファレンス.md準拠）
COMMAND_ARGUMENTS = {
    # 基本操作 (10)
    'MOVE': [OBJECT_ARG, OFFSET_DX_ARG, OFFSET_DY_ARG],
    'TELEPORT': [OBJECT_ARG, COORDINATE_X_ARG, COORDINATE_Y_ARG],
    'SLIDE': [OBJECT_ARG, DIRECTION_ARG, OBJECT_ARRAY_ARG],
    'PATHFIND': [OBJECT_ARG, COORDINATE_X_ARG, COORDINATE_Y_ARG, OBJECT_ARRAY_ARG],
    'ROTATE': [OBJECT_ARG, ANGLE_ARG],
    'FLIP': [OBJECT_ARG, AXIS_ARG],
    'SCALE': [OBJECT_ARG, SCALE_ARG],
    'SCALE_DOWN': [OBJECT_ARG, SCALE_ARG],
    'EXPAND': [OBJECT_ARG, PIXELS_ARG],
    'FILL_HOLES': [OBJECT_ARG, COLOR_ARG],
    'SET_COLOR': [OBJECT_ARG, COLOR_ARG],

    # 高度操作 (9)
    'OUTLINE': [OBJECT_ARG, COLOR_ARG],
    'HOLLOW': [OBJECT_ARG],
    'BBOX': [OBJECT_ARG, COLOR_ARG],
    'INTERSECTION': [OBJECT_ARG, OBJECT_ARG],
    'SUBTRACT': [OBJECT_ARG, OBJECT_ARG],
    'FLOW': [OBJECT_ARG, DIRECTION_ARG, OBJECT_ARRAY_ARG],
    'DRAW': [OBJECT_ARG, COORDINATE_X_ARG, COORDINATE_Y_ARG],
    'LAY': [OBJECT_ARG, DIRECTION_ARG, OBJECT_ARRAY_ARG],
    'ALIGN': [OBJECT_ARG, ALIGN_MODE_ARG],

    # オブジェクト生成 (4)
    'MERGE': [OBJECT_ARRAY_ARG],
    'CREATE_LINE': [COORDINATE_X_ARG, COORDINATE_Y_ARG, LENGTH_ARG, DIRECTION_ARG, COLOR_ARG],
    'CREATE_RECT': [COORDINATE_X_ARG, COORDINATE_Y_ARG, SIZE_ARG, SIZE_ARG, COLOR_ARG],
    'TILE': [OBJECT_ARG, COUNT_ARG, COUNT_ARG],  # obj, count_x, count_y

    # オブジェクト分割/抽出 (5)
    'SPLIT_CONNECTED': [OBJECT_ARG, CONNECTIVITY_ARG],
    'CROP': [OBJECT_ARG, COORDINATE_X_ARG, COORDINATE_Y_ARG, SIZE_ARG, SIZE_ARG],
    'EXTRACT_RECTS': [OBJECT_ARG],
    'EXTRACT_HOLLOW_RECTS': [OBJECT_ARG],
    'EXTRACT_LINES': [OBJECT_ARG],

    # フィッティング (3)
    'FIT_SHAPE': [OBJECT_ARG, OBJECT_ARG],
    'FIT_SHAPE_COLOR': [OBJECT_ARG, OBJECT_ARG],
    'FIT_ADJACENT': [OBJECT_ARG, OBJECT_ARG],

    # 配列操作 (9)
    'APPEND': [OBJECT_ARRAY_ARG, OBJECT_ARG],
    'LEN': [LEN_ARRAY_ARG],
    'REVERSE': [OBJECT_ARRAY_ARG],
    'CONCAT': [OBJECT_ARRAY_ARG, OBJECT_ARRAY_ARG],
    'EXCLUDE': [OBJECT_ARRAY_ARG, OBJECT_ARRAY_ARG],
    'SORT_BY': [OBJECT_ARRAY_ARG, KEY_EXPR_ARG, ORDER_ARG],
    'EXTEND_PATTERN': [OBJECT_ARRAY_ARG, SIDE_ARG, COUNT_ARG],
    'ARRANGE_GRID': [OBJECT_ARRAY_ARG, COUNT_ARG, SIZE_ARG, SIZE_ARG],  # objs, cols, w, h
    'MATCH_PAIRS': [OBJECT_ARRAY_ARG, OBJECT_ARRAY_ARG, MATCH_PAIRS_CONDITION_ARG],

    # 条件フィルター (1)
    'FILTER': [OBJECT_ARRAY_ARG, CONDITION_ARG],

    # GET関数 - 基本情報 (11)
    'GET_SIZE': [OBJECT_ARG],
    'GET_WIDTH': [OBJECT_ARG],
    'GET_HEIGHT': [OBJECT_ARG],
    'GET_X': [OBJECT_ARG],
    'GET_Y': [OBJECT_ARG],
    'GET_COLOR': [OBJECT_ARG],
    'GET_COLORS': [OBJECT_ARG],
    'GET_CENTER_X': [OBJECT_ARG],
    'GET_CENTER_Y': [OBJECT_ARG],
    'GET_MAX_X': [OBJECT_ARG],
    'GET_MAX_Y': [OBJECT_ARG],

    # GET関数 - 形状情報 (8)
    'COUNT_HOLES': [OBJECT_ARG],
    'GET_SYMMETRY_SCORE': [OBJECT_ARG, AXIS_ARG],
    'GET_LINE_TYPE': [OBJECT_ARG],
    'GET_RECTANGLE_TYPE': [OBJECT_ARG],
    'GET_ASPECT_RATIO': [OBJECT_ARG],
    'GET_DENSITY': [OBJECT_ARG],
    'GET_CENTROID': [OBJECT_ARG],

    # GET関数 - 距離/関係 (7)
    'GET_DISTANCE': [OBJECT_ARG, OBJECT_ARG],
    'GET_X_DISTANCE': [OBJECT_ARG, OBJECT_ARG],
    'GET_Y_DISTANCE': [OBJECT_ARG, OBJECT_ARG],
    'GET_DIRECTION': [OBJECT_ARG, OBJECT_ARG],
    'GET_NEAREST': [OBJECT_ARG, OBJECT_ARRAY_ARG],
    'COUNT_ADJACENT': [OBJECT_ARG, OBJECT_ARG],
    'COUNT_OVERLAP': [OBJECT_ARG, OBJECT_ARG],

    # GET関数 - グリッド情報 (3)
    'GET_ALL_OBJECTS': [CONNECTIVITY_ARG],
    'GET_BACKGROUND_COLOR': [],
    'GET_INPUT_GRID_SIZE': [],

    # IS関数 - 真偽値判定 (4)
    'IS_INSIDE': [OBJECT_ARG, COORDINATE_X_ARG, COORDINATE_Y_ARG, SIZE_ARG, SIZE_ARG],
    'IS_SAME_SHAPE': [OBJECT_ARG, OBJECT_ARG],
    'IS_SAME_STRUCT': [OBJECT_ARG, OBJECT_ARG],
    'IS_IDENTICAL': [OBJECT_ARG, OBJECT_ARG],

    # 算術演算・比較演算は動的に生成されるため除外

    # 論理演算 (2)
    'AND': [BOOL_ARG, BOOL_ARG],
    'OR': [BOOL_ARG, BOOL_ARG],

    # 最終出力 (1)
    'RENDER_GRID': [OBJECT_ARRAY_ARG, COLOR_ARG, SIZE_ARG, SIZE_ARG],
}

# メソッドを追加
add_naming_system_methods()

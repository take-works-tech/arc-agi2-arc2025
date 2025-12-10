"""
クリーンなコマンドメタデータ

無効・廃止済みコマンドを完全に除去した、実装済みコマンドのみのメタデータ
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .types import ReturnType, SemanticType, TypeInfo, TypeSystem
from .argument_schema import (
    ArgumentSchema,
    CompatibleArgumentSchema,
    COMMAND_ARGUMENTS,
    OBJECT_ARRAY_ARG,
)
from .variable_manager import variable_manager


def create_argument_schema_with_naming_system(type_info: TypeInfo, **kwargs) -> ArgumentSchema:
    """minimal_naming_systemを活用してArgumentSchemaを作成

    Args:
        type_info: 型情報（is_arrayフラグを含む）
        **kwargs: ArgumentSchemaのその他のパラメータ

    Returns:
        ArgumentSchemaインスタンス
    """
    # ArgumentSchemaを作成（nameは__post_init__で自動生成）
    # type_infoのis_arrayフラグはそのまま保持される
    return ArgumentSchema(
        type_info=type_info,
        **kwargs
    )





@dataclass
class CommandMetadata:
    """コマンドのメタデータ定義"""

    # ========================================
    # 基本情報
    # ========================================
    name: str                           # コマンド名
    category: str                       # カテゴリ（transform, filter, create, etc.）
    return_type_info: TypeInfo          # 戻り値の型情報（必須）

    # ========================================
    # 引数定義
    # ========================================
    arguments: List[ArgumentSchema] = field(default_factory=list)  # 引数リスト

    # ========================================
    # 使用確率・重み
    # ========================================
    base_weight: float = 1.0            # 基本重み（CommandPoolから継承）
    complexity_weights: Optional[Dict[int, float]] = None  # 複雑度ごとの重み調整

    # ========================================
    # コンテキスト依存
    # ========================================
    usage_contexts: List[str] = field(default_factory=lambda: ['for_loop'])  # 使用コンテキスト
    requires_array: bool = False        # 配列が必要か
    requires_object: bool = False       # オブジェクトが必要か
    modifies_in_place: bool = True      # 元のオブジェクトを変更するか

    # ========================================
    # 制約
    # ========================================
    min_complexity: int = 1             # 最小複雑度
    max_complexity: int = 8             # 最大複雑度


    # ========================================
    # 説明
    # ========================================
    description: str = ""               # 説明
    examples: List[str] = field(default_factory=list)  # 使用例


# ============================================================
# 全74コマンドのメタデータ定義
# ============================================================

COMMAND_METADATA: Dict[str, CommandMetadata] = {}


# ========================================
# 基本操作 (10コマンド)
# ========================================

COMMAND_METADATA['MOVE'] = CommandMetadata(
    name='MOVE',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['MOVE'],
    base_weight=3.0,  # 現行システムから継承
    usage_contexts=['for_loop'],
    requires_object=True,
    modifies_in_place=True,
    description='オブジェクトを相対移動',
    examples=['MOVE(obj, 5, -3)', 'MOVE(obj, GET_X(other), 0)'],
)

COMMAND_METADATA['TELEPORT'] = CommandMetadata(
    name='TELEPORT',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['TELEPORT'],
    base_weight=2.5,
    usage_contexts=['for_loop'],
    requires_object=True,
    modifies_in_place=True,
    description='オブジェクトを絶対座標に移動',
    examples=['TELEPORT(obj, 10, 15)'],
)

COMMAND_METADATA['ALIGN'] = CommandMetadata(
    name='ALIGN',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['ALIGN'],
    base_weight=2.0,  # 整列操作の簡潔化
    usage_contexts=['for_loop'],
    requires_object=True,
    modifies_in_place=True,
    description='オブジェクトを整列',
    examples=['ALIGN(obj, "left")', 'ALIGN(obj, "center")'],
)

COMMAND_METADATA['SLIDE'] = CommandMetadata(
    name='SLIDE',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['SLIDE'],
    base_weight=1.5,  # 0.8 → 1.5 に向上（移動操作の使用頻度向上）
    usage_contexts=['for_loop'],
    requires_object=True,
    requires_array=True,
    description='障害物まで移動（8方向対応）',
    examples=['SLIDE(obj, "Y", others)', 'SLIDE(obj, "XY", others)'],
)

COMMAND_METADATA['PATHFIND'] = CommandMetadata(
    name='PATHFIND',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['PATHFIND'],
    base_weight=1.5,  # 1.0 → 1.5 に向上（経路探索の使用頻度向上）
    usage_contexts=['for_loop'],
    requires_object=True,
    requires_array=True,
    description='経路探索移動（障害物回避、8方向対応）',
    examples=['PATHFIND(obj, 10, 15, obstacles)', 'PATHFIND(obj, GET_X(target), GET_Y(target), others)'],
)

COMMAND_METADATA['ROTATE'] = CommandMetadata(
    name='ROTATE',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['ROTATE'],
    base_weight=2.0,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='オブジェクトを回転',
    examples=['ROTATE(obj, 90)'],
)

COMMAND_METADATA['FLIP'] = CommandMetadata(
    name='FLIP',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['FLIP'],
    base_weight=2.0,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='オブジェクトを反転',
    examples=['FLIP(obj, "X")'],
)

COMMAND_METADATA['SCALE'] = CommandMetadata(
    name='SCALE',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['SCALE'],
    base_weight=1.5,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='オブジェクトを整数倍拡大',
    examples=['SCALE(obj, 2)'],
)

COMMAND_METADATA['SCALE_DOWN'] = CommandMetadata(
    name='SCALE_DOWN',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['SCALE_DOWN'],
    base_weight=0.5,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='オブジェクトを1/n倍縮小',
    examples=['SCALE_DOWN(obj, 2)'],
)

COMMAND_METADATA['EXPAND'] = CommandMetadata(
    name='EXPAND',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['EXPAND'],
    base_weight=1.2,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='オブジェクトを全方向に拡張',
    examples=['EXPAND(obj, 3)'],
)

COMMAND_METADATA['FILL_HOLES'] = CommandMetadata(
    name='FILL_HOLES',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['FILL_HOLES'],
    base_weight=1.8,  # 1.2 → 1.8 に向上（穴埋めの使用頻度向上）
    usage_contexts=['for_loop', 'assignment'],  # 変数代入でも使用可能に
    requires_object=True,
    description='オブジェクトの穴を埋める',
    examples=['FILL_HOLES(obj, 0)'],
)


COMMAND_METADATA['SET_COLOR'] = CommandMetadata(
    name='SET_COLOR',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['SET_COLOR'],
    base_weight=4.0,  # 10.0 → 15.0 に向上（大幅改善）
    usage_contexts=['for_loop'],
    requires_object=True,
    description='オブジェクトの色を変更',
    examples=['SET_COLOR(obj, 3)', 'SET_COLOR(obj, GET_BACKGROUND_COLOR())'],
)

# ========================================
# 高度操作 (8コマンド)
# ========================================

COMMAND_METADATA['OUTLINE'] = CommandMetadata(
    name='OUTLINE',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['OUTLINE'],
    base_weight=1.0,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='輪郭抽出',
    examples=['OUTLINE(obj, 1)'],
)

COMMAND_METADATA['HOLLOW'] = CommandMetadata(
    name='HOLLOW',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['HOLLOW'],
    base_weight=1.0,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='中空化',
    examples=['HOLLOW(obj)'],
)

COMMAND_METADATA['BBOX'] = CommandMetadata(
    name='BBOX',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['BBOX'],
    base_weight=0.8,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='境界矩形抽出',
    examples=['BBOX(obj, 2)'],
)

COMMAND_METADATA['INTERSECTION'] = CommandMetadata(
    name='INTERSECTION',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['INTERSECTION'],
    base_weight=1.5,  # 0.5 → 1.5 に向上（関係性生成に重要）
    usage_contexts=['for_loop'],
    requires_object=True,
    description='共通部分',
    examples=['INTERSECTION(obj1, obj2)'],
)

COMMAND_METADATA['SUBTRACT'] = CommandMetadata(
    name='SUBTRACT',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['SUBTRACT'],
    base_weight=1.5,  # 0.5 → 1.5 に向上（関係性生成に重要）
    usage_contexts=['for_loop'],
    requires_object=True,
    description='オブジェクト差分',
    examples=['SUBTRACT(obj1, obj2)'],
)

COMMAND_METADATA['FLOW'] = CommandMetadata(
    name='FLOW',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['FLOW'],
    base_weight=2.0,
    usage_contexts=['for_loop'],
    requires_object=True,
    requires_array=True,
    description='液体シミュレーション',
    examples=['FLOW(obj, "Y", others)'],
)

COMMAND_METADATA['DRAW'] = CommandMetadata(
    name='DRAW',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['DRAW'],
    base_weight=2.0,
    usage_contexts=['for_loop'],
    requires_object=True,
    description='軌跡描画',
    examples=['DRAW(obj, 5, 5)'],
)

COMMAND_METADATA['LAY'] = CommandMetadata(
    name='LAY',
    category='transform',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['LAY'],
    base_weight=2.0,
    usage_contexts=['for_loop'],
    requires_object=True,
    requires_array=True,
    description='重力配置（8方向対応）',
    examples=['LAY(obj, "Y", others)', 'LAY(obj, "XY", others)'],
)

# ========================================
# オブジェクト生成 (3コマンド)
# ========================================

COMMAND_METADATA['MERGE'] = CommandMetadata(
    name='MERGE',
    category='create',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['MERGE'],
    base_weight=1.0,  # 15.0 → 25.0 に向上（大幅改善）
    usage_contexts=['creation', 'for_loop'],
    requires_array=True,description='オブジェクト結合',
    examples=['MERGE([obj1, obj2])'],
)

COMMAND_METADATA['CREATE_LINE'] = CommandMetadata(
    name='CREATE_LINE',
    category='create',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['CREATE_LINE'],
    base_weight=1.5,  # 6.0 → 10.0 に上げる（大幅改善）
    usage_contexts=['creation'],description='線作成',
    examples=['CREATE_LINE(5, 10, 3, "X", 2)'],
)

COMMAND_METADATA['CREATE_RECT'] = CommandMetadata(
    name='CREATE_RECT',
    category='create',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['CREATE_RECT'],
    base_weight=1.5,  # 0.7 → 1.5 に向上（矩形作成の使用頻度向上）
    usage_contexts=['creation'],description='矩形作成',
    examples=['CREATE_RECT(0, 0, 5, 3, 1)'],
)

COMMAND_METADATA['TILE'] = CommandMetadata(
    name='TILE',
    category='create',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['TILE'],
    base_weight=1.0,  # 1.0 → 1.5 に向上（タイル操作の使用頻度向上）
    usage_contexts=['array_operation'],
    requires_object=True,
    requires_array=True,
    description='オブジェクトをタイル状に複製',
    examples=['TILE(obj, 3, 2)'],  # 3列×2行のタイル
)

# ========================================
# オブジェクト分割/抽出 (5コマンド)
# ========================================

COMMAND_METADATA['SPLIT_CONNECTED'] = CommandMetadata(
    name='SPLIT_CONNECTED',
    category='split',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments= COMMAND_ARGUMENTS['SPLIT_CONNECTED'],
    base_weight=1.2,
    usage_contexts=['for_loop', 'step_operation'],
    requires_object=True,description='連結成分に分割',
    examples=['SPLIT_CONNECTED(obj, 4)'],
)

# SPLIT_BY_CONNECTIONはSPLIT_CONNECTEDに改名されました

COMMAND_METADATA['CROP'] = CommandMetadata(
    name='CROP',
    category='split',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['CROP'],
    base_weight=0.8,
    usage_contexts=['for_loop'],
    requires_object=True,description='矩形範囲切り取り',
    examples=['CROP(obj, 0, 0, 10, 10)'],
)

COMMAND_METADATA['EXTRACT_RECTS'] = CommandMetadata(
    name='EXTRACT_RECTS',
    category='extract',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['EXTRACT_RECTS'],
    base_weight=0.8,
    usage_contexts=['step_operation'],
    requires_object=True,description='矩形抽出',
    examples=['EXTRACT_RECTS(obj)'],
)

COMMAND_METADATA['EXTRACT_HOLLOW_RECTS'] = CommandMetadata(
    name='EXTRACT_HOLLOW_RECTS',
    category='extract',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['EXTRACT_HOLLOW_RECTS'],
    base_weight=0.8,
    usage_contexts=['step_operation'],
    requires_object=True,description='中空矩形抽出',
    examples=['EXTRACT_HOLLOW_RECTS(obj)'],
)


COMMAND_METADATA['EXTRACT_LINES'] = CommandMetadata(
    name='EXTRACT_LINES',
    category='extract',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['EXTRACT_LINES'],
    base_weight=0.8,
    usage_contexts=['step_operation'],
    requires_object=True,description='直線抽出',
    examples=['EXTRACT_LINES(obj)'],
)

# ========================================
# フィッティング (3コマンド)
# ========================================

COMMAND_METADATA['FIT_SHAPE'] = CommandMetadata(
    name='FIT_SHAPE',
    category='placement',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments= COMMAND_ARGUMENTS['FIT_SHAPE'],
    base_weight=1.5,  # 0.1 → 1.5 に向上（関係性生成に重要）
    usage_contexts=['for_loop'],
    requires_object=True,description='形状フィット',
    examples=['FIT_SHAPE(obj1, obj2)'],
)

COMMAND_METADATA['FIT_SHAPE_COLOR'] = CommandMetadata(
    name='FIT_SHAPE_COLOR',
    category='placement',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['FIT_SHAPE_COLOR'],
    base_weight=1.5,  # 0.1 → 1.5 に向上（関係性生成に重要）
    usage_contexts=['for_loop'],
    requires_object=True,description='形状・色フィット',
    examples=['FIT_SHAPE_COLOR(obj1, obj2)'],
)

COMMAND_METADATA['FIT_ADJACENT'] = CommandMetadata(
    name='FIT_ADJACENT',
    category='placement',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['FIT_ADJACENT'],
    base_weight=1.5,  # 0.1 → 1.5 に向上（関係性生成に重要）
    usage_contexts=['for_loop'],
    requires_object=True,description='隣接フィット',
    examples=['FIT_ADJACENT(obj1, obj2)'],
)

# ========================================
# 配列操作 (8コマンド)
# ========================================

COMMAND_METADATA['APPEND'] = CommandMetadata(
    name='APPEND',
    category='array',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['APPEND'],
    base_weight=0.8,  # 型安全性の問題を解決したため有効化
    usage_contexts=['array_operation', 'assignment'],  # 使用コンテキストを復活
    requires_array=True,
    description='配列に要素を追加（型安全なオブジェクトのみ）',
    examples=['APPEND(objects, objects[0])'],
)

COMMAND_METADATA['LEN'] = CommandMetadata(
    name='LEN',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COUNT, is_array=False),
    arguments=COMMAND_ARGUMENTS['LEN'],
    base_weight=0.5,
    usage_contexts=['condition', 'nested_arg'],
    requires_array=True,
    description='配列の長さ',
    examples=['LEN(objects)'],
)

COMMAND_METADATA['REVERSE'] = CommandMetadata(
    name='REVERSE',
    category='array',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['REVERSE'],
    base_weight=0.8,  # 配列の逆順は中程度の優先度
    usage_contexts=['array_operation'],
    requires_array=True,
    description='配列を逆順にする',
    examples=['REVERSE(objects)'],
)

COMMAND_METADATA['CONCAT'] = CommandMetadata(
    name='CONCAT',
    category='array',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['CONCAT'],
    base_weight=1.0,  # 1.5 → 3.0 に向上（APPENDの代替として使用頻度を上げる）
    usage_contexts=['array_operation', 'assignment'],
    requires_array=True,
    description='配列連結（型安全）',
    examples=['CONCAT(arr1, arr2)'],
)

COMMAND_METADATA['SORT_BY'] = CommandMetadata(
    name='SORT_BY',
    category='array',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['SORT_BY'],
    base_weight=1.0,  # 0.8 → 2.5 に上げる（大幅改善）
    usage_contexts=['array_operation'],
    requires_array=True,
    description='ソート',
    examples=['SORT_BY(objs, GET_SIZE($obj), "asc")'],
)

COMMAND_METADATA['EXTEND_PATTERN'] = CommandMetadata(
    name='EXTEND_PATTERN',
    category='array',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['EXTEND_PATTERN'],
    base_weight=1.5,  # 1.2 → 2.0 に向上（関係性生成に重要）
    usage_contexts=['array_operation', 'assignment'],  # 変数代入でも使用可能に
    requires_array=True,
    description='パターン延長',
    examples=['EXTEND_PATTERN(objs, "end", 3)'],
)

COMMAND_METADATA['EXCLUDE'] = CommandMetadata(
    name='EXCLUDE',
    category='array_operation',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['EXCLUDE'],
    base_weight=0.5,
    usage_contexts=['array_operation', 'step_operation'],
    requires_array=True,description='オブジェクト除外（形状+色+位置で完全一致）',
    examples=['EXCLUDE(all_objects, background)', 'EXCLUDE(objects, processed)'],
)

COMMAND_METADATA['FILTER'] = CommandMetadata(
    name='FILTER',
    category='array',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['FILTER'],
    base_weight=1.0,  # 1.0 → 2.0 に向上（フィルタ操作の使用頻度向上）
    usage_contexts=['step_operation'],
    requires_array=True,
    description='条件フィルタ',
    examples=['FILTER(objs, GREATER(GET_SIZE($obj), 10))'],
)

COMMAND_METADATA['ARRANGE_GRID'] = CommandMetadata(
    name='ARRANGE_GRID',
    category='array',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['ARRANGE_GRID'],
    base_weight=1.5,  # 1.0 → 2.5 に上げる（大幅改善）
    usage_contexts=['array_operation'],
    requires_array=True,description='グリッド配置',
    examples=['ARRANGE_GRID(objs, 3, 10, 10)'],
)

COMMAND_METADATA['MATCH_PAIRS'] = CommandMetadata(
    name='MATCH_PAIRS',
    category='array',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['MATCH_PAIRS'],
    base_weight=1.0,  # 0.8 → 2.0 に向上（関係性生成に重要）
    usage_contexts=['array_operation'],
    requires_array=True,description='ペアマッチング',
    examples=['MATCH_PAIRS(objs1, objs2, IS_SAME_SHAPE($obj1, $obj2))'],
)

# ========================================
# GET関数 - 基本情報 (7コマンド)
# ========================================

COMMAND_METADATA['GET_SIZE'] = CommandMetadata(
    name='GET_SIZE',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_SIZE'],
    base_weight=2.0,
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='ピクセル数',
    examples=['GET_SIZE(obj)'],
)

COMMAND_METADATA['GET_WIDTH'] = CommandMetadata(
    name='GET_WIDTH',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_WIDTH'],
    base_weight=8.0,
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='幅',
    examples=['GET_WIDTH(obj)'],
)

COMMAND_METADATA['GET_HEIGHT'] = CommandMetadata(
    name='GET_HEIGHT',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_HEIGHT'],
    base_weight=2.0,
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='高さ',
    examples=['GET_HEIGHT(obj)'],
)

COMMAND_METADATA['GET_X'] = CommandMetadata(
    name='GET_X',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_X'],
    base_weight=8.0,
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='X座標',
    examples=['GET_X(obj)'],
)

COMMAND_METADATA['GET_Y'] = CommandMetadata(
    name='GET_Y',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_Y'],
    base_weight=8.0,
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='Y座標',
    examples=['GET_Y(obj)'],
)

COMMAND_METADATA['GET_COLOR'] = CommandMetadata(
    name='GET_COLOR',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COLOR, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_COLOR'],
    base_weight=3.0,  # 4.0 → 6.0 に上げる（大幅改善）
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='支配色',
    examples=['GET_COLOR(obj)'],
)

COMMAND_METADATA['GET_COLORS'] = CommandMetadata(
    name='GET_COLORS',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COLOR, is_array=True),  # 色の配列
    arguments=COMMAND_ARGUMENTS['GET_COLORS'],
    base_weight=1.8,  # 1.0 → 1.8 に向上（色リストの使用頻度向上）
    usage_contexts=['condition', 'nested_arg', 'assignment'],  # 変数代入でも使用可能に
    description='全色リスト',
    examples=['GET_COLORS(obj)'],
)

# ========================================
# GET関数 - 形状情報 (4コマンド)
# ========================================

COMMAND_METADATA['COUNT_HOLES'] = CommandMetadata(
    name='COUNT_HOLES',
    category='getter',    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COUNT_HOLES, is_array=False),  # 細分化された型
    arguments=COMMAND_ARGUMENTS['COUNT_HOLES'],
    base_weight=0.5,  # 重み削減済み
    usage_contexts=['condition'],
    description='穴の数',
    examples=['COUNT_HOLES(obj)'],
)

# GET_HOLE_COUNTは廃止されました（COUNT_HOLESを使用）

COMMAND_METADATA['GET_SYMMETRY_SCORE'] = CommandMetadata(
    name='GET_SYMMETRY_SCORE',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.PERCENTAGE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_SYMMETRY_SCORE'],
    base_weight=0.5,  # 0.0 → 1.5 に変更（条件やソートキーでの使用を許可）
    usage_contexts=['condition', 'sort_key'],  # ネスト引数では使用しない（条件やソートキーでのみ使用）
    description='対称性スコア（0-100）',
    examples=['GET_SYMMETRY_SCORE(obj, "X")'],
)

COMMAND_METADATA['GET_LINE_TYPE'] = CommandMetadata(
    name='GET_LINE_TYPE',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.LINE_TYPE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_LINE_TYPE'],
    base_weight=0.5,  # 0.8 → 1.5 に向上（線タイプの使用頻度向上）
    usage_contexts=['condition', 'nested_arg', 'assignment'],  # 比例演算の引数として使用可能に
    description='線タイプ（"C"は線ではない）',
    examples=['GET_LINE_TYPE(obj)'],
)

COMMAND_METADATA['GET_RECTANGLE_TYPE'] = CommandMetadata(
    name='GET_RECTANGLE_TYPE',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.RECT_TYPE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_RECTANGLE_TYPE'],
    base_weight=0.5,  # 0.8 → 1.5 に向上（矩形タイプの使用頻度向上）
    usage_contexts=['condition', 'nested_arg', 'assignment'],  # 変数代入でも使用可能に
    description='矩形タイプ',
    examples=['GET_RECTANGLE_TYPE(obj)'],
)

# ========================================
# GET関数 - 距離/関係 (5コマンド)
# ========================================

COMMAND_METADATA['GET_DISTANCE'] = CommandMetadata(
    name='GET_DISTANCE',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.DISTANCE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_DISTANCE'],
    base_weight=0.6,  # 5.5 → 6.0 に微増（空間系の活用を強化）
    usage_contexts=['condition', 'sort_key', 'nested_arg', 'assignment', 'for_loop'],  # より多くのコンテキストで使用可能に
    description='ユークリッド距離',
    examples=['GET_DISTANCE(obj1, obj2)'],
)

COMMAND_METADATA['GET_X_DISTANCE'] = CommandMetadata(
    name='GET_X_DISTANCE',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.DISTANCE_AXIS, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_X_DISTANCE'],
    base_weight=4.0,
    usage_contexts=['condition', 'nested_arg', 'sort_key', 'assignment', 'for_loop'],  # より多くのコンテキストで使用可能に
    description='X軸距離',
    examples=['GET_X_DISTANCE(obj1, obj2)'],
)

COMMAND_METADATA['GET_Y_DISTANCE'] = CommandMetadata(
    name='GET_Y_DISTANCE',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.DISTANCE_AXIS, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_Y_DISTANCE'],
    base_weight=4.0,
    usage_contexts=['condition', 'nested_arg', 'sort_key', 'assignment', 'for_loop'],  # より多くのコンテキストで使用可能に
    description='Y軸距離',
    examples=['GET_Y_DISTANCE(obj1, obj2)'],
)

COMMAND_METADATA['COUNT_ADJACENT'] = CommandMetadata(
    name='COUNT_ADJACENT',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COUNT_ADJACENT, is_array=False),
    arguments=COMMAND_ARGUMENTS['COUNT_ADJACENT'],
    base_weight=1.0,  # 0.8 → 1.5 に向上（関係性測定に重要）
    usage_contexts=['condition'],
    description='隣接ピクセル数',
    examples=['COUNT_ADJACENT(obj1, obj2)'],
)

COMMAND_METADATA['COUNT_OVERLAP'] = CommandMetadata(
    name='COUNT_OVERLAP',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COUNT_OVERLAP, is_array=False),
    arguments=COMMAND_ARGUMENTS['COUNT_OVERLAP'],
    base_weight=1.0,  # 0.8 → 1.5 に向上（関係性測定に重要）
    usage_contexts=['condition'],
    description='重複ピクセル数',
    examples=['COUNT_OVERLAP(obj1, obj2)'],
)

COMMAND_METADATA['GET_DIRECTION'] = CommandMetadata(
    name='GET_DIRECTION',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.DIRECTION, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_DIRECTION'],
    base_weight=3.0,  # ARCタスクで頻繁に必要
    usage_contexts=['condition', 'nested_arg', 'assignment'],
    description='2つのオブジェクト間の方向（"C"は同じオブジェクトまたは同じ位置の場合）',
    examples=['GET_DIRECTION(obj1, obj2)'],
)

COMMAND_METADATA['GET_NEAREST'] = CommandMetadata(
    name='GET_NEAREST',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_NEAREST'],
    base_weight=1.5,  # オプション、SORT_BYで代替可能
    usage_contexts=['assignment', 'nested_arg'],
    requires_array=True,
    description='最も近いオブジェクトを取得',
    examples=['GET_NEAREST(obj, candidates)'],
)

# ========================================
# GET関数 - 形状情報（拡張） (3コマンド)
# ========================================

COMMAND_METADATA['GET_ASPECT_RATIO'] = CommandMetadata(
    name='GET_ASPECT_RATIO',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.ASPECT_RATIO, is_array=False),  # int（幅/高さ * 100）
    arguments=COMMAND_ARGUMENTS['GET_ASPECT_RATIO'],
    base_weight=2.5,  # 最も基本で重要
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='アスペクト比（幅/高さ * 100、int）',
    examples=['GET_ASPECT_RATIO(obj)'],
)

COMMAND_METADATA['GET_DENSITY'] = CommandMetadata(
    name='GET_DENSITY',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.DENSITY, is_array=False),  # int（ピクセル数/bbox面積 * 100）
    arguments=COMMAND_ARGUMENTS['GET_DENSITY'],
    base_weight=1.5,  # 既存実装あり、コストが最低
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='密度（ピクセル数/バウンディングボックス面積 * 100、int、0-100）',
    examples=['GET_DENSITY(obj)'],
)

COMMAND_METADATA['GET_CENTROID'] = CommandMetadata(
    name='GET_CENTROID',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.DIRECTION, is_array=False),  # 方向文字列を返す（"C"は中央を表す）
    arguments=COMMAND_ARGUMENTS['GET_CENTROID'],
    base_weight=1.5,  # 非対称形状の識別に重要
    usage_contexts=['condition', 'nested_arg', 'assignment'],
    description='重心位置（方向文字列、"C"は中央）',
    examples=['GET_CENTROID(obj)'],
)

# ========================================
# GET関数 - 座標情報（拡張） (4コマンド)
# ========================================

COMMAND_METADATA['GET_CENTER_X'] = CommandMetadata(
    name='GET_CENTER_X',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_CENTER_X'],
    base_weight=3.0,  # 中心座標取得は高頻度
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='中心X座標',
    examples=['GET_CENTER_X(obj)'],
)

COMMAND_METADATA['GET_CENTER_Y'] = CommandMetadata(
    name='GET_CENTER_Y',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_CENTER_Y'],
    base_weight=3.0,  # 中心座標取得は高頻度
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='中心Y座標',
    examples=['GET_CENTER_Y(obj)'],
)

COMMAND_METADATA['GET_MAX_X'] = CommandMetadata(
    name='GET_MAX_X',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_MAX_X'],
    base_weight=2.5,  # 境界座標取得
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='最大X座標（右端）',
    examples=['GET_MAX_X(obj)'],
)

COMMAND_METADATA['GET_MAX_Y'] = CommandMetadata(
    name='GET_MAX_Y',
    category='getter',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_MAX_Y'],
    base_weight=2.5,  # 境界座標取得
    usage_contexts=['condition', 'nested_arg', 'sort_key'],
    description='最大Y座標（下端）',
    examples=['GET_MAX_Y(obj)'],
)

# ========================================
# GET関数 - グリッド情報 (3コマンド)
# ========================================

COMMAND_METADATA['GET_ALL_OBJECTS'] = CommandMetadata(
    name='GET_ALL_OBJECTS',
    category='basic',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.OBJECT, is_array=True),
    arguments=COMMAND_ARGUMENTS['GET_ALL_OBJECTS'],
    base_weight=1.0,
    usage_contexts=['initialization'],
    min_complexity=1,
    max_complexity=8,
)

COMMAND_METADATA['GET_BACKGROUND_COLOR'] = CommandMetadata(
    name='GET_BACKGROUND_COLOR',
    category='global',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.COLOR, is_array=False),
    arguments=COMMAND_ARGUMENTS['GET_BACKGROUND_COLOR'],
    base_weight=2.0,  # 0.8 → 2.0 に向上（背景色の使用頻度向上）
    usage_contexts=['condition', 'nested_arg', 'render_arg'],
    description='背景色を取得',
    examples=['GET_BACKGROUND_COLOR()'],
)

COMMAND_METADATA['GET_INPUT_GRID_SIZE'] = CommandMetadata(
    name='GET_INPUT_GRID_SIZE',
    category='basic',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=True),
    arguments=COMMAND_ARGUMENTS['GET_INPUT_GRID_SIZE'],
    base_weight=1.0,  # 0.9 → 8.0 に大幅向上（初期化で必須）
    usage_contexts=['initialization'],
    min_complexity=1,
    max_complexity=8,
)

# ========================================
# IS関数 - 真偽値判定 (4コマンド)
# ========================================

COMMAND_METADATA['IS_SAME_SHAPE'] = CommandMetadata(
    name='IS_SAME_SHAPE',
    category='judgment',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    arguments=COMMAND_ARGUMENTS['IS_SAME_SHAPE'],
    base_weight=1.0,  # 0.1 → 1.0 に向上（関係性判定に重要）
    usage_contexts=['condition'],
    description='形状が同じか',
    examples=['IS_SAME_SHAPE(obj1, obj2)'],
)

COMMAND_METADATA['IS_SAME_STRUCT'] = CommandMetadata(
    name='IS_SAME_STRUCT',
    category='judgment',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    arguments=COMMAND_ARGUMENTS['IS_SAME_STRUCT'],
    base_weight=1.0,  # 0.1 → 1.0 に向上（関係性判定に重要）
    usage_contexts=['condition'],
    description='構造が同じか',
    examples=['IS_SAME_STRUCT(obj1, obj2)'],
)

COMMAND_METADATA['IS_IDENTICAL'] = CommandMetadata(
    name='IS_IDENTICAL',
    category='judgment',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    arguments=COMMAND_ARGUMENTS['IS_IDENTICAL'],
    base_weight=1.0,  # 0.1 → 1.0 に向上（関係性判定に重要）
    usage_contexts=['condition'],
    description='完全に同じか',
    examples=['IS_IDENTICAL(obj1, obj2)'],
)


COMMAND_METADATA['IS_INSIDE'] = CommandMetadata(
    name='IS_INSIDE',
    category='judgment',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    arguments=COMMAND_ARGUMENTS['IS_INSIDE'],
    base_weight=1.0,  # 0.3 → 1.0 に向上（関係性判定に重要）
    usage_contexts=['condition', 'nested_arg'],  # 条件とネストした引数で使用可能
    requires_object=True,
    description='オブジェクトが矩形内にあるか',
    examples=['IS_INSIDE(obj, 0, 0, 10, 10)'],
)

# ========================================
# 算術演算 (5コマンド)
# ========================================

def create_arithmetic_command(base_name: str, base_metadata: dict, first_arg_type: SemanticType = None) -> CommandMetadata:
    """算術演算コマンドを作成

    Args:
        base_name: 基本コマンド名（'ADD', 'SUB', etc.）
        base_metadata: 基本メタデータ
        first_arg_type: 第一引数の型（使用されない、一貫性のため残している）

    Returns:
        算術演算コマンドのメタデータ

    注意:
        - 引数の型情報は、COMMAND_METADATA取得段階ではダミー値（COORDINATE型）を設定
        - 実際の引数生成時には、argument_generator.pyで動的に決定される（target_typeに依存）
        - これにより、比較演算と算術演算の引数のSemanticType決定を一貫性を持たせる
    """
    from .types import SemanticType

    # 戻り値の型は、実際の生成時にターゲット型に設定されるため、ダミー値を使用
    # 注意: これはCOMMAND_METADATAに登録される時のデフォルト値
    # 実際の生成時には、node_generators.pyでターゲット型がreturn_type_infoに設定される
    return_type_info = TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False)

    # 引数の型情報は、COMMAND_METADATA取得段階ではダミー値（COORDINATE型）を設定
    # 実際の引数生成時には、argument_generator.pyで動的に決定される
    dummy_type_info = TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False)

    return CommandMetadata(
        name=base_name,
        category=base_metadata['category'],
        return_type_info=return_type_info,
        arguments=[
            # 第1引数: ダミー型情報（実際の生成時には無視される）
            # 注意: allowed_nested_commandsは指定しない（None）
            # 実際の引数生成時には特別処理で新しいArgumentSchemaを作成するため、
            # ここで設定したallowed_nested_commandsは使われない
            create_argument_schema_with_naming_system(
                dummy_type_info,
                literal_prob=base_metadata['literal_prob'],
                variable_prob=base_metadata['variable_prob'],
                nested_prob=base_metadata['nested_prob'],
                allowed_nested_commands=None
            ),
            # 第2引数: ダミー型情報（実際の生成時には無視される）
            create_argument_schema_with_naming_system(
                dummy_type_info,
                literal_prob=base_metadata['literal_prob'],
                variable_prob=base_metadata['variable_prob'],
                nested_prob=base_metadata['nested_prob'],
                # 注意: allowed_nested_commandsは指定しない（None）
                # 実際の引数生成時には特別処理で新しいArgumentSchemaを作成するため、
                # ここで設定したallowed_nested_commandsは使われない
                allowed_nested_commands=None
            ),
        ],
        base_weight=base_metadata['base_weight'],
        usage_contexts=base_metadata['usage_contexts'],
        description=base_metadata['description'],
        examples=base_metadata['examples'],
    )

def create_proportional_command(base_name: str, base_metadata: dict, first_arg_type: SemanticType = None) -> CommandMetadata:
    """比例演算コマンドを作成

    Args:
        base_name: 基本コマンド名（'EQUAL', 'GREATER', etc.）
        base_metadata: 基本メタデータ
        first_arg_type: 第一引数の型（使用されない、一貫性のため残している）

    Returns:
        比例演算コマンドのメタデータ

    注意:
        - 引数の型情報は、COMMAND_METADATA取得段階ではダミー値（COORDINATE型）を設定
        - 実際の引数生成時には、argument_generator.pyで動的に決定される（PROPORTIONAL_TYPESから選択）
        - これにより、比較演算と算術演算の引数のSemanticType決定を一貫性を持たせる
    """
    from .types import SemanticType

    # 引数の型情報は、COMMAND_METADATA取得段階ではダミー値（COORDINATE型）を設定
    # 実際の引数生成時には、argument_generator.pyで動的に決定される
    dummy_type_info = TypeInfo.create_from_semantic_type(SemanticType.COORDINATE, is_array=False)

    return CommandMetadata(
        name=base_name,
        category=base_metadata['category'],
        return_type_info=base_metadata['return_type_info'],
        arguments=[
            # 第1引数: ダミー型情報（実際の生成時には無視される）
            create_argument_schema_with_naming_system(
                dummy_type_info,
                literal_prob=base_metadata['literal_prob'],
                variable_prob=base_metadata['variable_prob'],
                nested_prob=base_metadata['nested_prob'],
                # allowed_nested_commandsをNoneにして、型互換性チェックで自動的に適切なコマンドを選択
                # （算術演算はtarget_typeが算術演算可能な型に含まれる場合のみ許可される）
                allowed_nested_commands=None
            ),
            # 第2引数: ダミー型情報（実際の生成時には無視される）
            create_argument_schema_with_naming_system(
                dummy_type_info,
                literal_prob=base_metadata['literal_prob'],
                variable_prob=base_metadata['variable_prob'],
                nested_prob=base_metadata['nested_prob'],
                # allowed_nested_commandsをNoneにして、型互換性チェックで自動的に適切なコマンドを選択
                # （算術演算はtarget_typeが算術演算可能な型に含まれる場合のみ許可される）
                allowed_nested_commands=None
            ),
        ],
        base_weight=base_metadata['base_weight'],
        usage_contexts=base_metadata['usage_contexts'],
        description=base_metadata['description'],
        examples=base_metadata['examples'],
    )

# ADDコマンドの組み合わせバリエーションを生成
add_metadata = {
    'category': 'arithmetic',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    'literal_prob': 0.40,
    'variable_prob': 0.40,
    'nested_prob': 0.20,
    'base_weight': 15.0,  # 8.0 → 12.0 → 15.0 → 20.0 に向上（ADDの使用頻度向上）
    'usage_contexts': ['nested_arg', 'assignment', 'for_loop', 'condition'],
    'description': '加算',
    'examples': ['ADD(5, 3)', 'ADD(GET_X(obj), 10)'],
}

COMMAND_METADATA['ADD'] = create_arithmetic_command('ADD', add_metadata)  # 動的型選択

# SUBコマンドの組み合わせバリエーションを生成
sub_metadata = {
    'category': 'arithmetic',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    'literal_prob': 0.40,
    'variable_prob': 0.40,
    'nested_prob': 0.20,
    'base_weight': 15.0,  # 8.0 → 12.0 → 15.0 → 20.0 に上げる（大幅改善）
    'usage_contexts': ['nested_arg', 'assignment', 'for_loop', 'condition'],
    'description': '減算',
    'examples': ['SUB(10, 3)', 'SUB(GET_X(obj), 5)'],
}

COMMAND_METADATA['SUB'] = create_arithmetic_command('SUB', sub_metadata)  # 動的型選択

# MULTIPLYコマンドの組み合わせバリエーションを生成
multiply_metadata = {
    'category': 'arithmetic',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    'literal_prob': 0.50,
    'variable_prob': 0.40,
    'nested_prob': 0.10,
    'base_weight': 15.0,  # 8.5 → 12.0 → 18.0 に向上（MULTIPLYの使用頻度向上）
    'usage_contexts': ['nested_arg', 'assignment', 'condition', 'for_loop'],
    'description': '乗算',
    'examples': ['MULTIPLY(GET_WIDTH(obj), 2)'],
}

COMMAND_METADATA['MULTIPLY'] = create_arithmetic_command('MULTIPLY', multiply_metadata)  # 動的型選択

# DIVIDEコマンドの組み合わせバリエーションを生成
divide_metadata = {
    'category': 'arithmetic',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    'literal_prob': 0.50,
    'variable_prob': 0.40,
    'nested_prob': 0.10,
    'base_weight': 12.0,  # 6.5 → 10.0 → 15.0 に向上（DIVIDEの使用頻度向上）
    'usage_contexts': ['nested_arg', 'assignment', 'for_loop', 'condition'],
    'description': '整数除算',
    'examples': ['DIVIDE(GET_WIDTH(obj), 2)'],
}

COMMAND_METADATA['DIVIDE'] = create_arithmetic_command('DIVIDE', divide_metadata)  # 動的型選択

# MODコマンドの組み合わせバリエーションを生成
mod_metadata = {
    'category': 'arithmetic',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.SIZE, is_array=False),
    'literal_prob': 0.70,
    'variable_prob': 0.20,
    'nested_prob': 0.10,
    'base_weight': 4.0,  # 3.0 → 8.0 → 11.0 → 16.0 に大幅向上（MODの使用頻度向上）
    'usage_contexts': ['nested_arg', 'assignment', 'condition', 'for_loop'],  # conditionとfor_loopを追加
    'description': '剰余',
    'examples': ['MOD(GET_X(obj), 10)'],
}

COMMAND_METADATA['MOD'] = create_arithmetic_command('MOD', mod_metadata)  # 動的型選択

# ========================================
# 比較演算 (4コマンド)
# ========================================

# EQUALコマンドの組み合わせバリエーションを生成
equal_metadata = {
    'category': 'comparison',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    'literal_prob': 0.50,  # リテラル値を増加
    'variable_prob': 0.10,  # 変数を大幅に減少
    'nested_prob': 0.40,
    'base_weight': 8.0,  # 12.0 → 20.0 に上げる（大幅改善）
    'usage_contexts': ['condition'],
    'description': '等価判定',
    'examples': ['EQUAL(GET_COLOR($obj), 5)', 'EQUAL(GET_WIDTH($obj), GET_HEIGHT($obj))'],
}

COMMAND_METADATA['EQUAL'] = create_proportional_command('EQUAL', equal_metadata)

# NOT_EQUALコマンドの組み合わせバリエーションを生成
not_equal_metadata = {
    'category': 'comparison',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    'literal_prob': 0.50,  # リテラル値を増加
    'variable_prob': 0.10,  # 変数を大幅に減少
    'nested_prob': 0.40,
    'base_weight': 7.2,  # 2.0 → 1.2 に下げる
    'usage_contexts': ['condition'],
    'description': '非等価判定',
    'examples': ['NOT_EQUAL(GET_COLOR($obj), 5)', 'NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR())'],
}

COMMAND_METADATA['NOT_EQUAL'] = create_proportional_command('NOT_EQUAL', not_equal_metadata)

# GREATERコマンドの組み合わせバリエーションを生成
greater_metadata = {
    'category': 'comparison',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    'literal_prob': 0.50,  # リテラル値を増加
    'variable_prob': 0.10,  # 変数を大幅に減少
    'nested_prob': 0.40,
    'base_weight': 3.5,  # 0.8 → 1.5 に上げる
    'usage_contexts': ['condition'],
    'description': '大なり判定',
    'examples': ['GREATER(GET_SIZE($obj), 10)', 'GREATER(GET_WIDTH($obj), GET_HEIGHT($obj))'],
}

COMMAND_METADATA['GREATER'] = create_proportional_command('GREATER', greater_metadata)

# LESSコマンドの組み合わせバリエーションを生成
less_metadata = {
    'category': 'comparison',
    'return_type_info': TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    'literal_prob': 0.50,  # リテラル値を増加
    'variable_prob': 0.10,  # 変数を大幅に減少
    'nested_prob': 0.40,
    'base_weight': 3.5,  # 0.8 → 1.5 に上げる
    'usage_contexts': ['condition'],
    'description': '小なり判定',
    'examples': ['LESS(GET_SIZE($obj), 20)', 'LESS(GET_WIDTH($obj), GET_HEIGHT($obj))'],
}

COMMAND_METADATA['LESS'] = create_proportional_command('LESS', less_metadata)

# ========================================
# 論理演算 (2コマンド)
# ========================================

COMMAND_METADATA['AND'] = CommandMetadata(
    name='AND',
    category='logic',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    arguments=COMMAND_ARGUMENTS['AND'],
    base_weight=1.0,  # 1.5 → 2.5 に向上（論理演算の使用頻度大幅向上）
    usage_contexts=['condition', 'nested_arg', 'assignment'],  # 変数代入でも使用可能に
    description='論理AND',
    examples=['AND(EQUAL(GET_COLOR($obj), 5), GREATER(GET_SIZE($obj), 10))'],
)

COMMAND_METADATA['OR'] = CommandMetadata(
    name='OR',
    category='logic',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.BOOL, is_array=False),
    arguments=COMMAND_ARGUMENTS['OR'],
    base_weight=1.0,  # 1.5 → 2.5 に向上（論理演算の使用頻度大幅向上）
    usage_contexts=['condition', 'nested_arg', 'assignment'],  # 変数代入でも使用可能に
    description='論理OR',
    examples=['OR(EQUAL(GET_COLOR($obj), 1), EQUAL(GET_COLOR($obj), 2))'],
)

# ========================================
# 制御構造 (1コマンド - RENDER_GRID)
# ========================================

COMMAND_METADATA['RENDER_GRID'] = CommandMetadata(
    name='RENDER_GRID',
    category='finalization',
    return_type_info=TypeInfo.create_from_semantic_type(SemanticType.VOID, is_array=False),
    arguments=COMMAND_ARGUMENTS['RENDER_GRID'],
    base_weight=1.0,
    usage_contexts=['finalization'],
    description='グリッド描画',
    examples=['RENDER_GRID(objects, 0, 30, 30)', 'RENDER_GRID(objects, GET_BACKGROUND_COLOR(), grid_size[0], grid_size[1])'],
)

# 注: FOR, IF, END は制御構造であり、コマンドメタデータでは定義しない
#     ノードクラス（ForLoopNode, IfBranchNode）で直接実装

# ============================================================
# コマンド数の検証
# ============================================================

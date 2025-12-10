"""
input_grid_generator用のコマンドメタデータ

有効性検証と置き換え機能を備えたコマンドメタデータ定義
program_generator/metadata/commands.pyのメタデータをベースに、検証用フィールドを追加
"""
from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional
from src.data_systems.generator.program_generator.metadata.commands import (
    COMMAND_METADATA as BASE_COMMAND_METADATA,
    CommandMetadata as BaseCommandMetadata
)


@dataclass
class CommandMetadata(BaseCommandMetadata):
    """コマンドのメタデータ定義（検証機能付き）

    program_generator/metadata/commands.pyのCommandMetadataを拡張し、
    input_grid_generator用の検証フィールドを追加
    """
    # ========================================
    # input_grid_generator用の検証設定
    # ========================================
    fallback_template: Optional[str] = None  # フォールバック置き換え形式（第一引数を{arg0}として使用可能）
    validate_effectiveness: bool = False  # 有効性検証を行うか


# ============================================================
# 全86コマンドのメタデータ定義（BASE_COMMAND_METADATAから動的に読み込み）
# ============================================================
#
# program_generator/metadata/commands.pyのメタデータをベースに、
# fallback_templateとvalidate_effectivenessを追加/上書き
#
# 注意: 新しいコマンドはBASE_COMMAND_METADATAに追加されれば自動的に利用可能

COMMAND_METADATA: Dict[str, CommandMetadata] = {}

def create_validation_metadata(cmd_name: str, fallback_template: Optional[str] = None,
                                validate_effectiveness: bool = False, **override_kwargs) -> CommandMetadata:
    """既存のメタデータをベースに検証用メタデータを作成

    Args:
        cmd_name: コマンド名
        fallback_template: フォールバックテンプレート
        validate_effectiveness: 有効性検証を行うか
        **override_kwargs: その他の上書きするフィールド

    Returns:
        CommandMetadataインスタンス
    """
    base_metadata = BASE_COMMAND_METADATA.get(cmd_name)
    if base_metadata is None:
        raise ValueError(f"Command '{cmd_name}' not found in BASE_COMMAND_METADATA")

    # BaseCommandMetadataのフィールドを取得して新しいCommandMetadataインスタンスを作成
    # dataclassのfields()を使って全てのフィールドを取得
    from dataclasses import fields, asdict

    # base_metadataを辞書に変換（新しいフィールドは含まれない）
    base_dict = asdict(base_metadata)

    # 検証用フィールドを追加
    base_dict['fallback_template'] = fallback_template
    base_dict['validate_effectiveness'] = validate_effectiveness

    # override_kwargsで上書き
    base_dict.update(override_kwargs)

    # 新しいCommandMetadataインスタンスを作成
    return CommandMetadata(**base_dict)


# ========================================
# 基本操作 (10コマンド)
# ========================================

COMMAND_METADATA['MOVE'] = create_validation_metadata(
    'MOVE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['TELEPORT'] = create_validation_metadata(
    'TELEPORT',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['SLIDE'] = create_validation_metadata(
    'SLIDE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['PATHFIND'] = create_validation_metadata(
    'PATHFIND',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['ROTATE'] = create_validation_metadata(
    'ROTATE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['FLIP'] = create_validation_metadata(
    'FLIP',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['SCALE'] = create_validation_metadata(
    'SCALE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['SCALE_DOWN'] = create_validation_metadata(
    'SCALE_DOWN',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['EXPAND'] = create_validation_metadata(
    'EXPAND',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['FILL_HOLES'] = create_validation_metadata(
    'FILL_HOLES',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['SET_COLOR'] = create_validation_metadata(
    'SET_COLOR',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

# ========================================
# 高度操作 (8コマンド)
# ========================================
COMMAND_METADATA['OUTLINE'] = create_validation_metadata(
    'OUTLINE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['HOLLOW'] = create_validation_metadata(
    'HOLLOW',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['BBOX'] = create_validation_metadata(
    'BBOX',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['INTERSECTION'] = create_validation_metadata(
    'INTERSECTION',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['SUBTRACT'] = create_validation_metadata(
    'SUBTRACT',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['FLOW'] = create_validation_metadata(
    'FLOW',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['DRAW'] = create_validation_metadata(
    'DRAW',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['LAY'] = create_validation_metadata(
    'LAY',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['ALIGN'] = create_validation_metadata(
    'ALIGN',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

# ========================================
# オブジェクト生成 (4コマンド)
# ========================================
COMMAND_METADATA['MERGE'] = create_validation_metadata(
    'MERGE',
    fallback_template='{arg0}[0]',
    validate_effectiveness=True
)

COMMAND_METADATA['CREATE_LINE'] = create_validation_metadata(
    'CREATE_LINE',
    fallback_template=None,
    validate_effectiveness=False
)

COMMAND_METADATA['CREATE_RECT'] = create_validation_metadata(
    'CREATE_RECT',
    fallback_template=None,
    validate_effectiveness=False
)

COMMAND_METADATA['TILE'] = create_validation_metadata(
    'TILE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

# ========================================
# オブジェクト分割/抽出 (5コマンド)
# ========================================
COMMAND_METADATA['SPLIT_CONNECTED'] = create_validation_metadata(
    'SPLIT_CONNECTED',
    fallback_template='[{arg0}]',
    validate_effectiveness=True
)

COMMAND_METADATA['CROP'] = create_validation_metadata(
    'CROP',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['EXTRACT_RECTS'] = create_validation_metadata(
    'EXTRACT_RECTS',
    fallback_template='[{arg0}]',
    validate_effectiveness=True
)

COMMAND_METADATA['EXTRACT_HOLLOW_RECTS'] = create_validation_metadata(
    'EXTRACT_HOLLOW_RECTS',
    fallback_template='[{arg0}]',
    validate_effectiveness=True
)

COMMAND_METADATA['EXTRACT_LINES'] = create_validation_metadata(
    'EXTRACT_LINES',
    fallback_template='[{arg0}]',
    validate_effectiveness=True
)

# ========================================
# フィッティング (3コマンド)
# ========================================
COMMAND_METADATA['FIT_SHAPE'] = create_validation_metadata(
    'FIT_SHAPE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['FIT_SHAPE_COLOR'] = create_validation_metadata(
    'FIT_SHAPE_COLOR',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['FIT_ADJACENT'] = create_validation_metadata(
    'FIT_ADJACENT',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

# ========================================
# 配列操作 (8コマンド)
# ========================================
COMMAND_METADATA['APPEND'] = create_validation_metadata(
    'APPEND',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['LEN'] = create_validation_metadata(
    'LEN',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['CONCAT'] = create_validation_metadata(
    'CONCAT',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['SORT_BY'] = create_validation_metadata(
    'SORT_BY',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['EXTEND_PATTERN'] = create_validation_metadata(
    'EXTEND_PATTERN',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['EXCLUDE'] = create_validation_metadata(
    'EXCLUDE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['FILTER'] = create_validation_metadata(
    'FILTER',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['ARRANGE_GRID'] = create_validation_metadata(
    'ARRANGE_GRID',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['MATCH_PAIRS'] = create_validation_metadata(
    'MATCH_PAIRS',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

COMMAND_METADATA['REVERSE'] = create_validation_metadata(
    'REVERSE',
    fallback_template='{arg0}',
    validate_effectiveness=True
)

# ========================================
# GET関数 - 基本情報 (7コマンド)
# ========================================
COMMAND_METADATA['GET_SIZE'] = create_validation_metadata(
    'GET_SIZE',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_WIDTH'] = create_validation_metadata(
    'GET_WIDTH',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_HEIGHT'] = create_validation_metadata(
    'GET_HEIGHT',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_X'] = create_validation_metadata(
    'GET_X',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_Y'] = create_validation_metadata(
    'GET_Y',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_COLOR'] = create_validation_metadata(
    'GET_COLOR',
    fallback_template=None,
    validate_effectiveness=False
)

COMMAND_METADATA['GET_COLORS'] = create_validation_metadata(
    'GET_COLORS',
    fallback_template=None,
    validate_effectiveness=False
)

# ========================================
# GET関数 - 形状情報 (4コマンド)
# ========================================
COMMAND_METADATA['COUNT_HOLES'] = create_validation_metadata(
    'COUNT_HOLES',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_SYMMETRY_SCORE'] = create_validation_metadata(
    'GET_SYMMETRY_SCORE',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_LINE_TYPE'] = create_validation_metadata(
    'GET_LINE_TYPE',
    fallback_template=None,
    validate_effectiveness=False
)

COMMAND_METADATA['GET_RECTANGLE_TYPE'] = create_validation_metadata(
    'GET_RECTANGLE_TYPE',
    fallback_template=None,
    validate_effectiveness=False
)

# ========================================
# GET関数 - 距離/関係 (5コマンド)
# ========================================
COMMAND_METADATA['GET_DISTANCE'] = create_validation_metadata(
    'GET_DISTANCE',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_X_DISTANCE'] = create_validation_metadata(
    'GET_X_DISTANCE',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_Y_DISTANCE'] = create_validation_metadata(
    'GET_Y_DISTANCE',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['COUNT_ADJACENT'] = create_validation_metadata(
    'COUNT_ADJACENT',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['COUNT_OVERLAP'] = create_validation_metadata(
    'COUNT_OVERLAP',
    fallback_template='0',
    validate_effectiveness=False
)

# ========================================
# GET関数 - グリッド情報 (3コマンド)
# ========================================
COMMAND_METADATA['GET_ALL_OBJECTS'] = create_validation_metadata(
    'GET_ALL_OBJECTS',
    fallback_template=None,
    validate_effectiveness=False
)

COMMAND_METADATA['GET_BACKGROUND_COLOR'] = create_validation_metadata(
    'GET_BACKGROUND_COLOR',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_INPUT_GRID_SIZE'] = create_validation_metadata(
    'GET_INPUT_GRID_SIZE',
    fallback_template='[0, 0]',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_ASPECT_RATIO'] = create_validation_metadata(
    'GET_ASPECT_RATIO',
    fallback_template='1.0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_DENSITY'] = create_validation_metadata(
    'GET_DENSITY',
    fallback_template='1.0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_CENTROID'] = create_validation_metadata(
    'GET_CENTROID',
    fallback_template='"center"',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_CENTER_X'] = create_validation_metadata(
    'GET_CENTER_X',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_CENTER_Y'] = create_validation_metadata(
    'GET_CENTER_Y',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_MAX_X'] = create_validation_metadata(
    'GET_MAX_X',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_MAX_Y'] = create_validation_metadata(
    'GET_MAX_Y',
    fallback_template='0',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_DIRECTION'] = create_validation_metadata(
    'GET_DIRECTION',
    fallback_template='"X"',
    validate_effectiveness=False
)

COMMAND_METADATA['GET_NEAREST'] = create_validation_metadata(
    'GET_NEAREST',
    fallback_template='{arg0}',
    validate_effectiveness=False
)

# ========================================
# IS関数 - 真偽値判定 (4コマンド)
# ========================================
COMMAND_METADATA['IS_SAME_SHAPE'] = create_validation_metadata(
    'IS_SAME_SHAPE',
    fallback_template='True',
    validate_effectiveness=False
)

COMMAND_METADATA['IS_SAME_STRUCT'] = create_validation_metadata(
    'IS_SAME_STRUCT',
    fallback_template='True',
    validate_effectiveness=False
)

COMMAND_METADATA['IS_IDENTICAL'] = create_validation_metadata(
    'IS_IDENTICAL',
    fallback_template='True',
    validate_effectiveness=False
)

COMMAND_METADATA['IS_INSIDE'] = create_validation_metadata(
    'IS_INSIDE',
    fallback_template='True',
    validate_effectiveness=False
)

# ========================================
# 算術演算 (5コマンド)
# ========================================
# 注意: ADD, SUB, MULTIPLY, DIVIDE, MODは動的型選択のため、
# BASE_COMMAND_METADATAから直接使用します（動的生成のため）

COMMAND_METADATA['ADD'] = create_validation_metadata(
    'ADD',
    fallback_template='{arg0}',
    validate_effectiveness=False
)

COMMAND_METADATA['SUB'] = create_validation_metadata(
    'SUB',
    fallback_template='{arg0}',
    validate_effectiveness=False
)

COMMAND_METADATA['MULTIPLY'] = create_validation_metadata(
    'MULTIPLY',
    fallback_template='{arg0}',
    validate_effectiveness=False
)

COMMAND_METADATA['DIVIDE'] = create_validation_metadata(
    'DIVIDE',
    fallback_template='{arg0}',
    validate_effectiveness=False
)

COMMAND_METADATA['MOD'] = create_validation_metadata(
    'MOD',
    fallback_template='{arg0}',
    validate_effectiveness=False
)

# ========================================
# 比較演算 (4コマンド)
# ========================================
# 注意: EQUAL, NOT_EQUAL, GREATER, LESSは動的型選択のため、
# BASE_COMMAND_METADATAから直接使用します（動的生成のため）

COMMAND_METADATA['EQUAL'] = create_validation_metadata(
    'EQUAL',
    fallback_template='True',
    validate_effectiveness=False
)

COMMAND_METADATA['NOT_EQUAL'] = create_validation_metadata(
    'NOT_EQUAL',
    fallback_template='True',
    validate_effectiveness=False
)

COMMAND_METADATA['GREATER'] = create_validation_metadata(
    'GREATER',
    fallback_template='True',
    validate_effectiveness=False
)

COMMAND_METADATA['LESS'] = create_validation_metadata(
    'LESS',
    fallback_template='True',
    validate_effectiveness=False
)

# ========================================
# 論理演算 (2コマンド)
# ========================================
COMMAND_METADATA['AND'] = create_validation_metadata(
    'AND',
    fallback_template='True',
    validate_effectiveness=False
)

COMMAND_METADATA['OR'] = create_validation_metadata(
    'OR',
    fallback_template='True',
    validate_effectiveness=False
)

# ========================================
# 制御構造 (1コマンド - RENDER_GRID)
# ========================================
COMMAND_METADATA['RENDER_GRID'] = create_validation_metadata(
    'RENDER_GRID',
    fallback_template=None,
    validate_effectiveness=False
)


# 注: FOR, IF, END は制御構造であり、コマンドメタデータでは定義しない
#     ノードクラス（ForLoopNode, IfBranchNode）で直接実装

# ============================================================
# input_grid_generator用の検証設定について
# ============================================================
#
# 各コマンドには以下の検証設定フィールドを追加できます：
#
# 1. fallback_template: Optional[str] = None
#    - コマンドが有効でないと判断された場合のフォールバック置き換え形式
#    - 第一引数を {arg0} として使用可能
#    - 例: 'MOVE({arg0}, 1, 0)' → MOVE(obj, 1, 0) に置き換え
#    - 第一引数がオブジェクトでない場合（CREATE_LINEなど）は固定値を使用
#    - Noneの場合は置き換えを実行しない
#
# 2. validate_effectiveness: bool = False
#    - 有効性検証を行うかどうか
#    - True: コマンドが実際に効果を持っているか検証する
#    - False: 検証をスキップする（情報取得コマンドなど）
#
#
# 設定パターンの例：
#
# 【変換操作（MOVE, SET_COLORなど）】
# - fallback_template: 'COMMAND({arg0}, デフォルト値)'
# - validate_effectiveness: True
#
# 【情報取得（GET_*など）】
# - fallback_template: None
# - validate_effectiveness: False
#
# 【配列操作（FILTER, SORT_BYなど）】
# - fallback_template: 'COMMAND({arg0}, デフォルト条件)'
# - validate_effectiveness: True
#
# 【生成コマンド（CREATE_*など）】
# - fallback_template: 'COMMAND(固定値, ...)' （{arg0}は使用不可）
# - validate_effectiveness: True
#
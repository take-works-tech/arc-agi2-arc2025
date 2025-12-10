"""
ノード選択重み計算
"""
import os
import re
from typing import Dict, Optional
from .nodes import (
    Node,
    FilterNode, ConcatNode, AppendNode, MergeNode, EmptyArrayNode,
    ExcludeNode, ExtendPatternNode, ArrangeGridNode, SplitConnectedNode,
    MatchPairsNode, IfBranchNode, RenderNode,
    InitializationNode,
)
from .program_context import ProgramContext

# ログ出力制御（Falseに固定）
# デバッグ時のみTrueに設定（elementノードの選択状況を確認するため）
ENABLE_DEBUG_OUTPUT = os.environ.get('ENABLE_ELEMENT_DEBUG', 'false').lower() in ('true', '1', 'yes')
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ELEMENT_DEBUG', 'false').lower() in ('true', '1', 'yes')

# selected（小文字）からnode_type（大文字）へのマッピング
SELECTED_TO_NODE_TYPE = {
    "filter": "FILTER", "sort": "SORT", "merge": "MERGE", "concat": "CONCAT",
    "append": "APPEND", "create": "CREATE", "object_operations": "OBJECT_OPS",
    "element": "ELEMENT_ASSIGN", "exclude": "EXCLUDE", "extract_shape": "EXTRACT_RECTS",
    "extend_pattern": "EXTEND_PATTERN", "split_connected": "SPLIT_CONNECTED",
    "match_pairs": "MATCH_PAIRS", "empty": "EMPTY_ARRAY", "for": "FOR_LOOP", "if": "IF_CONDITION",
    "get_objects": "GET_OBJECTS",
    "reverse": "REVERSE", "tile": "TILE", "arrange_grid": "ARRANGE_GRID",
    "single_object_array": "SINGLE_OBJECT_ARRAY"
}


def classify_node_type(node) -> str:
    """ノードをタイプ別に分類

    Args:
        node: 分類するノードオブジェクト

    Returns:
        str: ノードタイプの識別子

    Note:
        より具体的なパターンを先にチェックする順序で実装されています。
        エラーが発生した場合は 'OTHER' を返します。
    """
    try:
        # ノードコードを取得
        if hasattr(node, 'generate'):
            try:
                node_code = node.generate()
            except Exception as e:
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[WARNING] classify_node_type: ノードの生成に失敗: {type(e).__name__}: {e}", flush=True)
                node_code = str(node)
        else:
            node_code = str(node)

        if not node_code or not isinstance(node_code, str):
            return 'OTHER'

        # より具体的なパターンを先にチェック（順序が重要）

        # 1. 初期化・取得系（最も具体的なパターンから）
        if 'GET_ALL_OBJECTS' in node_code:
            return 'GET_OBJECTS'

        # 2. 配列操作系（具体的なコマンド名で判定）
        if re.search(r'\bFILTER\s*\(', node_code):
            return 'FILTER'
        if re.search(r'\bSORT_BY\s*\(', node_code):
            return 'SORT'
        if re.search(r'\bCONCAT\s*\(', node_code):
            return 'CONCAT'
        if re.search(r'\bAPPEND\s*\(', node_code):
            return 'APPEND'
        if re.search(r'\bEXCLUDE\s*\(', node_code):
            return 'EXCLUDE'
        if re.search(r'\bMERGE\s*\(', node_code):
            return 'MERGE'
        if re.search(r'\bMATCH_PAIRS\s*\(', node_code):
            return 'MATCH_PAIRS'
        if re.search(r'\bSPLIT_CONNECTED\s*\(', node_code):
            return 'SPLIT_CONNECTED'
        if re.search(r'\bEXTEND_PATTERN\s*\(', node_code):
            return 'EXTEND_PATTERN'
        if re.search(r'\bARRANGE_GRID\s*\(', node_code):
            return 'ARRANGE_GRID'
        if re.search(r'\bREVERSE\s*\(', node_code):
            return 'REVERSE'

        # 3. 抽出系（EXTRACT_で始まるコマンド）
        if re.search(r'\bEXTRACT_(RECTS|LINES|HOLLOW_RECTS)\s*\(', node_code):
            # EXTRACT_RECTS, EXTRACT_LINES, EXTRACT_HOLLOW_RECTSのいずれか（配列を返す）
            # ノードタイプ識別子としてEXTRACT_RECTSを使用（EXTRACT_SHAPEは存在しないコマンド）
            return 'EXTRACT_RECTS'

        # 4. 作成系
        if re.search(r'\bCREATE_(LINE|RECT)\s*\(', node_code):
            return 'CREATE'
        if re.search(r'\bTILE\s*\(', node_code):
            return 'TILE'

        # 5. オブジェクト操作系（BBOXを含む）
        object_ops_patterns = [
            r'\bSET_COLOR\s*\(',
            r'\bTELEPORT\s*\(',
            r'\bPATHFIND\s*\(',
            r'\bROTATE\s*\(',
            r'\bFLIP\s*\(',
            r'\bMOVE\s*\(',
            r'\bSLIDE\s*\(',
            r'\bSCALE(_DOWN)?\s*\(',
            r'\bEXPAND\s*\(',
            r'\bBBOX\s*\(',
            r'\bFILL_HOLES\s*\(',
            r'\bOUTLINE\s*\(',
            r'\bHOLLOW\s*\(',
            r'\bINTERSECTION\s*\(',
            r'\bSUBTRACT\s*\(',
            r'\bFLOW\s*\(',
            r'\bDRAW\s*\(',
            r'\bLAY\s*\(',
            r'\bCROP\s*\(',
            r'\bFIT_SHAPE(_COLOR)?\s*\(',
            r'\bFIT_ADJACENT\s*\(',
            r'\bALIGN\s*\(',
        ]
        if any(re.search(pattern, node_code) for pattern in object_ops_patterns):
            return 'OBJECT_OPS'

        # 6. プロパティ取得系・情報取得系
        if re.search(r'\bGET_(X|Y|COLOR|WIDTH|HEIGHT|SIZE|COLORS|DISTANCE|X_DISTANCE|Y_DISTANCE|DIRECTION|NEAREST|ASPECT_RATIO|DENSITY|CENTROID|CENTER_X|CENTER_Y|MAX_X|MAX_Y|LINE_TYPE|RECTANGLE_TYPE|SYMMETRY_SCORE|ALL_OBJECTS|BACKGROUND_COLOR|INPUT_GRID_SIZE)\s*\(', node_code):
            return 'GET_PROPERTY'
        if re.search(r'\bCOUNT_(HOLES|ADJACENT|OVERLAP)\s*\(', node_code):
            return 'GET_PROPERTY'

        # 7. 制御構造（FOR/IF/WHILEはより正確なパターンで判定）
        if re.search(r'\bFOR\s+', node_code) or re.search(r'\bFOR\s*\(', node_code):
            return 'FOR_LOOP'
        if re.search(r'\bIF\s+', node_code) or re.search(r'\bIF\s*\(', node_code):
            return 'IF_CONDITION'
        if re.search(r'\bWHILE\s+', node_code) or re.search(r'\bWHILE\s*\(', node_code):
            return 'WHILE_LOOP'

        # 8. 配列要素代入（空配列を先に判定）
        if re.search(r'=\s*\[\s*\]', node_code):
            # 空配列の作成（例: objects1 = []）を先に判定
            return 'EMPTY_ARRAY'
        if re.search(r'\[.*\]\s*=', node_code):
            # 配列要素への代入（例: objects1[0] = ...）
            return 'ELEMENT_ASSIGN'

        # 9. レンダリング・終了
        if re.search(r'\bRENDER_GRID\s*\(', node_code):
            return 'RENDER'
        if re.search(r'\bEND\b', node_code):
            return 'END'

        # 10. 判定関数・比較演算・論理演算
        if re.search(r'\bIS_(SAME_SHAPE|SAME_STRUCT|IDENTICAL|INSIDE)\s*\(', node_code):
            return 'COMPARISON_OPERATIONS'
        if re.search(r'\b(EQUAL|NOT_EQUAL|GREATER|LESS)\s*\(', node_code):
            return 'PROPORTIONAL_OPERATIONS'
        if re.search(r'\b(AND|OR)\s*\(', node_code):
            return 'LOGICAL_OPERATIONS'

        # 11. 代入文（FOR/IF/RENDER_GRIDを含まない）
        if '=' in node_code:
            # FOR/IF/RENDER_GRIDを含まない代入文のみを対象
            if not re.search(r'\b(FOR|IF|RENDER_GRID)\b', node_code):
                # 単純な代入文の詳細分類
                if re.search(r'\b(GET_INPUT_GRID_SIZE|GET_BACKGROUND_COLOR)\s*\(', node_code):
                    return 'GRID_INFO'
                if re.search(r'\b(ADD|SUB|MULTIPLY|DIVIDE|MOD)\s*\(', node_code):
                    return 'ARITHMETIC'
                if re.search(r'\bLEN\s*\(', node_code):
                    return 'UTILITY'
                if re.search(r'\b(TRUE|FALSE)\b', node_code) or re.search(r'\b[0-9]+\b', node_code):
                    return 'CONSTANT'
                return 'ASSIGNMENT'

        # 11. その他
        return 'OTHER'

    except Exception as e:
        # エラーが発生した場合は 'OTHER' を返す
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[WARNING] classify_node_type: 分類中にエラーが発生: {type(e).__name__}: {e}", flush=True)
        return 'OTHER'


def get_node_type_from_instance(node) -> Optional[str]:
    """ノードのインスタンス型から直接ノードタイプを取得（高速化）

    Args:
        node: 分類するノードオブジェクト

    Returns:
        Optional[str]: ノードタイプの識別子（判定できない場合はNone）

    Note:
        主要なノードタイプについては isinstance チェックで高速に判定します。
        判定できない場合は None を返し、呼び出し側で classify_node_type を使用します。
    """
    # 主要なノードタイプを isinstance で判定（高速）
    if isinstance(node, FilterNode):
        return 'FILTER'
    elif isinstance(node, ConcatNode):
        return 'CONCAT'
    elif isinstance(node, AppendNode):
        return 'APPEND'
    elif isinstance(node, MergeNode):
        return 'MERGE'
    elif isinstance(node, EmptyArrayNode):
        return 'EMPTY_ARRAY'
    elif isinstance(node, ExcludeNode):
        return 'EXCLUDE'
    elif isinstance(node, ExtendPatternNode):
        return 'EXTEND_PATTERN'
    elif isinstance(node, ArrangeGridNode):
        return 'ARRANGE_GRID'
    elif isinstance(node, SplitConnectedNode):
        return 'SPLIT_CONNECTED'
    elif isinstance(node, MatchPairsNode):
        return 'MATCH_PAIRS'
    elif isinstance(node, IfBranchNode):
        return 'IF_CONDITION'
    elif isinstance(node, RenderNode):
        return 'RENDER'
    elif isinstance(node, InitializationNode):
        return 'GET_OBJECTS'

    # 判定できない場合は None を返す（呼び出し側で classify_node_type を使用）
    return None


def _get_last_node_type(context: ProgramContext) -> Optional[str]:
    """最後のノードタイプを取得

    Args:
        context: プログラムコンテキスト

    Returns:
        Optional[str]: 最後のノードタイプ（取得できない場合はNone）
    """
    # last_selectedを優先的に使用（Node分類より正確）
    last_selected = getattr(context, 'last_selected', None)
    if last_selected:
        return SELECTED_TO_NODE_TYPE.get(last_selected, last_selected.upper())

    # last_selectedがない場合、previous_nodesから分類
    previous_nodes = getattr(context, 'previous_nodes', [])
    if previous_nodes:
        last_node = previous_nodes[-1]
        # まず isinstance チェックで高速に判定を試みる
        last_node_type = get_node_type_from_instance(last_node)
        # 判定できない場合は classify_node_type を使用
        if last_node_type is None:
            last_node_type = classify_node_type(last_node)
        return last_node_type

    return None


def _apply_stage_adjustments(selection_weights: Dict[str, float], step_progress: float) -> None:
    """段階別調整を適用（ARC-AGI2データセット分析に基づく）

    Args:
        selection_weights: 選択重み辞書（変更される）
        step_progress: ステップ進行度（0.0-1.0）
    """
    if step_progress < 0.3:  # 初期段階（0-30%）
        # 初期段階：複数のFILTER、SORT_BY、FOR開始が重要
        selection_weights["filter"] *= 1.0      # GET_ALL_OBJECTS後の複数フィルタリング
        selection_weights["sort"] *= 2.0        # 初期ソート
        selection_weights["for"] *= 1.0         # 初期段階でFORを大幅強化
        selection_weights["append"] *= 1.5      # APPENDも増やす
        selection_weights["create"] *= 2.0      # CREATE_LINEなど新規作成
        selection_weights["merge"] *= 1.5       # MERGEも使用
        selection_weights["if"] *= 0.5          # 初期段階ではIFは少なめ
        selection_weights["arrange_grid"] *= 1.0  # 初期段階では配置は稀
        selection_weights["empty"] *= 0.5       # 初期段階でもEMPTYは控えめ
        selection_weights["match_pairs"] *= 2.0  # 初期段階では稀
        selection_weights["define"] *= 15.0      # 初期段階では変数定義を使用
        selection_weights["redefine"] *= 3.0    # 初期段階では変数再定義を使用

    elif step_progress < 0.7:  # 中間段階（30-70%）
        # 中間段階：メイン処理（FOR、IF、APPEND、CREATE、ELEMENT）が最重要
        selection_weights["for"] *= 2.0        # メイン処理でのFORループ大幅強化
        selection_weights["if"] *= 2.0          # 条件分岐処理強化
        selection_weights["append"] *= 1.5      # 結果蓄積最重要
        selection_weights["create"] *= 1.5      # 新規オブジェクト作成
        selection_weights["element"] *= 2.0     # 要素操作強化
        selection_weights["object_operations"] *= 1.0  # オブジェクト変換
        selection_weights["merge"] *= 1.8       # MERGEも使用
        selection_weights["concat"] *= 1.3      # 配列結合
        selection_weights["filter"] *= 0.8      # 中間段階ではフィルタはやや抑制
        selection_weights["define"] *= 1.0      # 中間段階では変数定義を使用
        selection_weights["redefine"] *= 1.5    # 中間段階では変数再定義を使用

    else:  # 終了段階（70-100%）
        # 終了段階：結果統合・最終処理
        selection_weights["merge"] *= 1.5       # 最終統合
        selection_weights["concat"] *= 2.0      # 最終結合
        selection_weights["arrange_grid"] *= 2.0  # 最終配置は中程度
        selection_weights["for"] *= 1.0         # 終了段階ではFORは稀
        selection_weights["if"] *= 0.5          # 終了段階ではIFは稀
        selection_weights["filter"] *= 0.3      # 終了段階ではフィルタは稀
        selection_weights["create"] *= 0.3      # 終了段階では新規作成は稀
        selection_weights["append"] *= 0.5      # 終了段階ではAPPENDは稀


def _apply_for_loop_adjustments(selection_weights: Dict[str, float], context: ProgramContext, step_progress: float) -> None:
    """FORループ関連の調整を適用

    Args:
        selection_weights: 選択重み辞書（変更される）
        context: プログラムコンテキスト
        step_progress: ステップ進行度（0.0-1.0）
    """
    current_scope_depth = context.get_scope_nesting_depth()
    max_scope_depth = context.get_max_scope_nesting_depth()

    if context.get_for_nesting_depth() > 0:
        # FORループ内では要素操作・APPEND・CREATEの確率を大幅に上げる
        selection_weights["append"] *= 5.0     # FORループ内での追加操作最重要
        selection_weights["create"] *= 5.0     # FORループ内での作成操作
        selection_weights["filter"] *= 0.1     # FOR内でFILTER呼び出し（確率を少し下げる）
        selection_weights["sort"] *= 0.4       # FOR内でSORT呼び出し（確率を少し下げる）
        selection_weights["arrange_grid"] *= 0.4  # FOR内でARRANGE_GRID呼び出し（確率を少し下げる）
        selection_weights["match_pairs"] *= 0.4  # FOR内でMATCH_PAIRS呼び出し（確率を少し下げる）
        selection_weights["empty"] *= 0.0      # FOR内ではEMPTY_ARRAYを生成しない（FOR前のみ）
        selection_weights["element"] *= 10.0   # FORループ内では要素操作を大幅に強化（配列要素代入を促進）
        selection_weights["if"] *= 4.0         # FORループ内でのIF分岐を大幅に促進（ネスト深度向上）
        selection_weights["object_operations"] *= 1.5  # FORループ内でのオブジェクト操作
        selection_weights["merge"] *= 1.3      # FOR内でMERGEも使用

        # FORループ内でのFORループ生成を大幅に抑制（過剰なネストを防ぐ）
        selection_weights["for"] *= 0.2       # FOR内でのFORは5%に抑制

        # 既に深いネストの場合はさらに抑制
        if current_scope_depth >= 2:
            selection_weights["for"] *= 0.1    # 深度2以上ではさらに10%に抑制
        if current_scope_depth >= 3:
            selection_weights["for"] *= 0.1    # 深度3以上ではさらに10%に抑制（合計0.05%）
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] FORループ内: スコープ深度={current_scope_depth}/{max_scope_depth}, FOR選択重みを抑制", flush=True)

    # FORループが一度も使用されていない場合の特別処理
    if context.get_for_nesting_depth() == 0:
        # プログラム内でFORループが使用されていない場合、確率を大幅に上げる
        selection_weights["for"] *= 5.0        # FORループ未使用時は確率をさらに大きく上げる
        # 特に1回目のFORループの確率をさらに上げる
        if step_progress < 0.5:  # 前半の段階では特に強化
            selection_weights["for"] *= 2.0     # 前半段階ではさらに4倍に強化

    # context.for_arraysにNoneが含まれている場合、APPENDの確率を上げる（定数FORループで要素追加が一般的）
    if None in context.for_arrays:
        selection_weights["append"] *= 3.5  # Noneを含む場合はAPPEND確率を上げる（10.0→3.5に削減）
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] for_arraysにNone検出: APPEND確率を3.5倍に調整", flush=True)

    # FOR内のIF内では、APPENDの確率をさらに上げる（条件付きで要素追加が一般的）
    if context.get_for_nesting_depth() > 0 and context.get_if_nesting_depth() > 0:
        selection_weights["append"] *= 3.5  # FOR内のIF内ではAPPEND確率を上げる（10.0→3.5に削減）
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] FOR内のIF内検出: APPEND確率を3.5倍に調整（FOR深度={context.get_for_nesting_depth()}, IF深度={context.get_if_nesting_depth()}）", flush=True)


def calculate_selection_weights(
    context: ProgramContext,
    command_weight_adjustments: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """ノード選択重みを計算する

    Args:
        context: プログラムコンテキスト
        command_weight_adjustments: コマンド重み調整マップ（オプション）

    Returns:
        Dict[str, float]: ノードタイプごとの選択重み
    """
    if command_weight_adjustments is None:
        command_weight_adjustments = {}

    # ノード選択タイプと重みの対応表（ARC-AGI2データセット分析に基づく）
    base_weights = {
        "if": 9.0,                    # IF文頻度を大幅強化（FOR内で多用される、ARC-AGI2で重要）
        "for": 10.0,                  # FOR文頻度を大幅強化（ARC-AGI2の核心）
        "element": 3.5,               # 要素操作基本重み（FOR内で重要）
        "filter": 0.5,                # フィルタリング最重要（複数連続が一般的）
        "sort": 1.2,                  # ソートは中程度（位置関係の整理）
        "concat": 0.5,                # 配列結合は中程度
        "create": 3.0,                # CREATE_LINE等の新規作成を強化
        "merge": 0.5,                 # MERGEは頻繁に使用される（配列から単体への変換）
        "match_pairs": 3.0,          # 0.3 → 1.5 → 2.2 → 3.0 に向上（関係性生成に重要、ペアマッチングパターン）
        "empty": 0.0,                 # 空配列作成は中程度（APPENDとの組み合わせ）
        "append": 0.5,                # 要素追加は最重要（FOR内で多用）
        "exclude": 0.8,               # 除外は控えめ
        "extract_shape": 0.8,         # 形状抽出は控えめ（配列を返す）
        "extend_pattern": 1.5,        # 0.8 → 1.5 に向上（関係性生成に重要）
        "arrange_grid": 1.5,          # 0.5 → 1.5 に向上（関係性生成に重要）
        "single_object_array": 0.5,   # 単一オブジェクト配列は稀
        "split_connected": 1.5,       # 1.0 → 1.5 に向上（関係性生成に寄与）
        "object_operations": 0.1,     # オブジェクト操作は重要（SET_COLOR, TELEPORT等）
        "reverse": 1.5,               # REVERSE: 配列を逆順にする
        "tile": 1.5,                  # TILE: オブジェクトをタイル状に配置
        "define": 1.0,                # 変数定義ノード（オブジェクトとオブジェクト配列以外）
        "redefine": 3.0,              # 変数再定義ノード（スコープネスト内でのみ有効、初期値は0）
    }

    # コマンド名からnode_typeへの逆引きマッピング（すべてのコマンドをカバー）
    command_to_node_type = {
        # IF/FOR文
        "IF_CONDITION": "if",
        "FOR_LOOP": "for",
        # 配列操作
        "FILTER": "filter",
        "SORT_BY": "sort",
        "CONCAT": "concat",
        "APPEND": "append",
        "EXCLUDE": "exclude",
        "ARRANGE_GRID": "arrange_grid",
        "MATCH_PAIRS": "match_pairs",
        "EXTEND_PATTERN": "extend_pattern",
        "EMPTY_ARRAY": "empty",
        "SINGLE_OBJECT_ARRAY": "single_object_array",
        "MERGE": "merge",
        "REVERSE": "reverse",
        "TILE": "tile",
        "LEN": "utility",  # 配列長さ取得
        # 抽出系（配列を返す）
        "EXTRACT_RECTS": "extract_shape",
        "EXTRACT_LINES": "extract_shape",
        "EXTRACT_HOLLOW_RECTS": "extract_shape",
        "SPLIT_CONNECTED": "split_connected",
        # 作成系
        "CREATE_LINE": "create",
        "CREATE_RECT": "create",
        # オブジェクト操作系（object_operationsにマッピング）
        "MOVE": "object_operations",
        "TELEPORT": "object_operations",
        "SLIDE": "object_operations",
        "PATHFIND": "object_operations",
        "ROTATE": "object_operations",
        "FLIP": "object_operations",
        "SCALE": "object_operations",
        "SCALE_DOWN": "object_operations",
        "EXPAND": "object_operations",
        "FILL_HOLES": "object_operations",
        "SET_COLOR": "object_operations",
        "OUTLINE": "object_operations",
        "HOLLOW": "object_operations",
        "BBOX": "object_operations",  # オブジェクト操作に統合（オブジェクト単体を返す）
        "INTERSECTION": "object_operations",
        "SUBTRACT": "object_operations",
        "FLOW": "object_operations",
        "DRAW": "object_operations",
        "LAY": "object_operations",
        "CROP": "object_operations",
        "FIT_SHAPE": "object_operations",
        "FIT_SHAPE_COLOR": "object_operations",
        "FIT_ADJACENT": "object_operations",
        "ALIGN": "object_operations",
        # GETTERコマンド（情報取得系）
        # ネストした引数として使用されることが多いが、command_weight_adjustmentsで使用される可能性があるためマッピングを追加
        "GET_SIZE": "getter_operations",
        "GET_WIDTH": "getter_operations",
        "GET_HEIGHT": "getter_operations",
        "GET_X": "getter_operations",
        "GET_Y": "getter_operations",
        "GET_COLOR": "getter_operations",
        "GET_COLORS": "getter_operations",
        "COUNT_HOLES": "getter_operations",
        "GET_SYMMETRY_SCORE": "getter_operations",
        "GET_LINE_TYPE": "getter_operations",
        "GET_RECTANGLE_TYPE": "getter_operations",
        "GET_DISTANCE": "getter_operations",
        "GET_X_DISTANCE": "getter_operations",
        "GET_Y_DISTANCE": "getter_operations",
        "COUNT_ADJACENT": "getter_operations",
        "COUNT_OVERLAP": "getter_operations",
        "GET_ALL_OBJECTS": "getter_operations",
        "GET_BACKGROUND_COLOR": "getter_operations",
        "GET_INPUT_GRID_SIZE": "getter_operations",
        "GET_ASPECT_RATIO": "getter_operations",
        "GET_DENSITY": "getter_operations",
        "GET_CENTROID": "getter_operations",
        "GET_CENTER_X": "getter_operations",
        "GET_CENTER_Y": "getter_operations",
        "GET_MAX_X": "getter_operations",
        "GET_MAX_Y": "getter_operations",
        "GET_DIRECTION": "getter_operations",
        "GET_NEAREST": "getter_operations",
        # 比較・演算系（base_weightsにない場合はdefault_weights_for_new_typesを使用）
        "IS_SAME_SHAPE": "comparison_operations",
        "IS_SAME_STRUCT": "comparison_operations",
        "IS_IDENTICAL": "comparison_operations",
        "IS_INSIDE": "comparison_operations",
        "ADD": "arithmetic_operations",
        "SUB": "arithmetic_operations",
        "MULTIPLY": "arithmetic_operations",
        "DIVIDE": "arithmetic_operations",
        "MOD": "arithmetic_operations",
        "EQUAL": "proportional_operations",
        "NOT_EQUAL": "proportional_operations",
        "GREATER": "proportional_operations",
        "LESS": "proportional_operations",
        "AND": "logical_operations",
        "OR": "logical_operations",
        "RENDER_GRID": "render_operations",
    }

    # 新しいnode_typeのデフォルト重み（base_weightsにない場合）
    default_weights_for_new_types = {
        "comparison_operations": 1.0,
        "arithmetic_operations": 2.5,  # 1.0 → 1.5 → 2.5 に向上（算術演算の使用頻度向上）
        "proportional_operations": 1.0,
        "logical_operations": 1.0,
        "render_operations": 1.0,
        "getter_operations": 1.0,  # GET_*コマンド用
        "utility": 1.0,  # LEN等のユーティリティコマンド用
    }

    # コマンド重み調整を適用（統計で収集されたすべてのコマンドに対して）
    if command_weight_adjustments:
        for command_name, adjustment in command_weight_adjustments.items():
            node_type = command_to_node_type.get(command_name)
            if node_type:
                # base_weightsにない場合はデフォルト重みを追加
                if node_type not in base_weights:
                    if node_type in default_weights_for_new_types:
                        base_weights[node_type] = default_weights_for_new_types[node_type]
                    else:
                        base_weights[node_type] = 1.0  # 完全に新しいタイプの場合は1.0
                    if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                        print(f"[重み調整] 新しいnode_type '{node_type}' を追加（デフォルト重み: {base_weights[node_type]}）", flush=True)

                base_weights[node_type] *= adjustment
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    print(f"[重み調整] {command_name} -> {node_type}: {adjustment:.2f}倍適用（新しい重み: {base_weights[node_type]:.2f}）", flush=True)
            else:
                if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                    # マッピングがないコマンド（未対応のコマンド）は警告のみ
                    print(f"[重み調整警告] コマンド '{command_name}' に対応するnode_typeが見つかりません（調整スキップ）", flush=True)

    # プログラム段階に応じた重み調整
    selection_weights = base_weights.copy()

    # 現在のステップ位置を取得（0.0-1.0の範囲）
    current_step = getattr(context, 'current_step', 0)
    total_steps = getattr(context, 'total_steps', 1)
    step_progress = current_step / max(total_steps, 1) if total_steps > 0 else 0.0

    # 段階別調整を適用
    _apply_stage_adjustments(selection_weights, step_progress)

    # FORループ関連の調整を適用
    _apply_for_loop_adjustments(selection_weights, context, step_progress)

    # 前のノードに基づく連続パターン調整（ARC-AGI2手動タスク分析に基づく）
    last_node_type = _get_last_node_type(context)


    if last_node_type:

        # ARC-AGI2データセット分析に基づく連続パターンの確率調整
        if last_node_type == 'GET_OBJECTS':
            # GET_ALL_OBJECTS後は必ずFILTERまたはMERGEが続く
            # 注意: 部分プログラムフローでは、部分プログラムの最後のノードがFILTERまたはEXCLUDEの場合、
            # last_selectedが適切に設定されるため、このコードは適用されない
            selection_weights["filter"] *= 100.0  # 初期ノード後第1ステップでFILTER確率を最大に
            selection_weights["merge"] *= 1.0
            # FILTERとMERGE以外の確率を0にする
            for key in selection_weights:
                if key not in ["filter", "merge"]:
                    selection_weights[key] = 0.0

        elif last_node_type == 'FILTER':
            # FILTER後はさらにFILTER、SORT、またはFOR
            selection_weights["filter"] *= 0.001   # FILTER後のFILTER確率を下げる（連続を抑制）
            selection_weights["sort"] *= 2.5     # SORT_BYも重要
            selection_weights["for"] *= 8.5      # FILTER後のFOR確率を上げる（10.0→3.5に削減）
            selection_weights["merge"] *= 2.0    # FILTER→MERGEも頻出
            selection_weights["append"] *= 0.5
            selection_weights["empty"] *= 2.0
            selection_weights["match_pairs"] *= 2.5  # FILTER→MATCH_PAIRS（フィルタリングした配列同士をマッチング）


        elif last_node_type == 'FOR_LOOP':
            # FOR後は内部処理：APPEND、CREATE、FILTER、ELEMENT、IF
            selection_weights["append"] *= 1.0    # FOR後のAPPEND最重要
            selection_weights["create"] *= 4.0    # CREATE_LINE等の作成
            selection_weights["filter"] *= 0.3    # FOR内でFILTER呼び出し
            selection_weights["if"] *= 3.5        # IF分岐
            selection_weights["element"] *= 5.0   # ELEMENT操作
            selection_weights["object_operations"] *= 1.5  # オブジェクト操作
            selection_weights["merge"] *= 1.3     # FOR内でMERGEも使用
            selection_weights["for"] *= 2.0
            selection_weights["define"] *= 2.0      # FOR内で変数定義を使用
            selection_weights["redefine"] *= 3.0    # FOR内で変数再定義を使用
        elif last_node_type == 'IF_CONDITION':
            # IF後はOBJECT_OPSまたはELEMENT
            selection_weights["element"] *= 3.5    # IF後の要素操作最重要（8.0→3.5に削減）
            selection_weights["object_operations"] *= 4.0  # OBJECT_OPS
            selection_weights["append"] *= 3.0
            selection_weights["create"] *= 1.2
            selection_weights["define"] *= 2.0      # IF内で変数定義を使用
            selection_weights["redefine"] *= 6.0    # IF内で変数再定義を使用
        elif last_node_type == 'MERGE':
            # MERGE後はOBJECT_OPS、APPEND、または他の処理
            selection_weights["object_operations"] *= 5.0  # SET_COLOR等
            selection_weights["append"] *= 2.5    # MERGE→APPEND
            selection_weights["for"] *= 1.2
            selection_weights["merge"] *= 0.5
            # MERGE後はオブジェクト単体から配列を生成する操作を促進
            selection_weights["split_connected"] *= 4.5    # MERGE→SPLIT_CONNECTED
            selection_weights["single_object_array"] *= 2.5  # MERGE→SINGLE_OBJECT_ARRAY
            selection_weights["extract_shape"] *= 2.5      # MERGE→EXTRACT_SHAPE

        elif last_node_type == 'OBJECT_OPS':
            # OBJECT_OPS後はAPPENDまたは次の処理
            selection_weights["append"] *= 4.0    # OBJECT_OPS→APPEND
            selection_weights["merge"] *= 1.5
            selection_weights["object_operations"] *= 0.3  # 連続を抑制

        elif last_node_type == 'SORT':
            # SORT後はFOR、CONCAT、またはAPPEND
            selection_weights["for"] *= 5.0      # SORT→FOR頻出
            selection_weights["append"] *= 3.0    # APPEND
            selection_weights["concat"] *= 2.0
            selection_weights["filter"] *= 1.0
            selection_weights["match_pairs"] *= 2.5  # SORT→MATCH_PAIRS（ソートした配列同士をマッチング）

        elif last_node_type == 'CONCAT':
            # CONCAT後は終了処理が多い
            selection_weights["merge"] *= 1.5
            selection_weights["arrange_grid"] *= 1.2
            selection_weights["for"] *= 0.2
            selection_weights["match_pairs"] *= 2.0  # CONCAT→MATCH_PAIRS（結合した配列同士をマッチング）

        elif last_node_type == 'APPEND':
            # APPEND後はさらにAPPEND、または次の処理
            selection_weights["append"] *= 3.5    # 連続APPEND
            selection_weights["merge"] *= 1.5
            selection_weights["for"] *= 0.8

        elif last_node_type == 'ARRANGE_GRID':
            # ARRANGE_GRID後は終了
            for key in selection_weights:
                selection_weights[key] *= 0.05  # ほぼ終了

        elif last_node_type == 'ELEMENT_ASSIGN':
            # ELEMENT後は次の処理へ
            selection_weights["create"] *= 2.0   # 新しいオブジェクト作成
            selection_weights["object_operations"] *= 2.0  # オブジェクト操作
            selection_weights["append"] *= 1.5
            selection_weights["element"] *= 0.2  # 連続を抑制

        elif last_node_type == 'EXCLUDE':
            # EXCLUDE後はCONCAT、FILTER、またはFOR
            selection_weights["concat"] *= 2.0
            selection_weights["for"] *= 1.2
            selection_weights["match_pairs"] *= 2.5  # EXCLUDE→MATCH_PAIRS（除外した配列同士をマッチング）

        elif last_node_type == 'EXTRACT_RECTS' or last_node_type == 'EXTRACT_LINES' or last_node_type == 'EXTRACT_HOLLOW_RECTS':
            # 形状抽出（配列を返す）後は統合処理
            selection_weights["merge"] *= 2.0
            selection_weights["append"] *= 2.0
            selection_weights["for"] *= 1.2

        elif last_node_type == 'EXTEND_PATTERN':
            # パターン拡張後は配置または結合
            selection_weights["arrange_grid"] *= 2.0
            selection_weights["concat"] *= 1.5
            selection_weights["for"] *= 0.5

        elif last_node_type == 'SPLIT_CONNECTED':
            # 分割後はループ処理
            selection_weights["for"] *= 3.0
            selection_weights["if"] *= 2.0
            selection_weights["element"] *= 1.5

        elif last_node_type == 'MATCH_PAIRS':
            # ペアマッチング後はループ処理
            selection_weights["for"] *= 4.5       # MATCH_PAIRS→FOR頻出（8.0→3.5に削減）
            selection_weights["if"] *= 2.5
            selection_weights["append"] *= 2.0
            selection_weights["merge"] *= 1.5

        elif last_node_type == 'EMPTY_ARRAY':
            # 空配列作成後はFOR
            selection_weights["for"] *= 9.5      # 空配列後のFOR確率を上げる（20.0→3.5に削減）
            selection_weights["append"] *= 0.2
            selection_weights["create"] *= 1.0

        # 直前と同じタイプの確率を下げる（多様性向上）
        if last_node_type in ["FILTER", "SORT", "MERGE", "CONCAT", "APPEND", "CREATE",
                              "OBJECT_OPS", "ELEMENT_ASSIGN", "EXCLUDE", "EXTRACT_RECTS",
                              "EXTEND_PATTERN", "SPLIT_CONNECTED", "MATCH_PAIRS", "EMPTY_ARRAY",
                              "REVERSE", "TILE", "ARRANGE_GRID", "SINGLE_OBJECT_ARRAY"]:
            # 対応するノードタイプを探す
            for node_type_key, node_type_class in [
                ("filter", "FILTER"), ("sort", "SORT"), ("merge", "MERGE"), ("concat", "CONCAT"),
                ("append", "APPEND"), ("create", "CREATE"), ("object_operations", "OBJECT_OPS"),
                ("element", "ELEMENT_ASSIGN"), ("exclude", "EXCLUDE"), ("extract_shape", "EXTRACT_RECTS"),
                ("extend_pattern", "EXTEND_PATTERN"), ("split_connected", "SPLIT_CONNECTED"),
                ("match_pairs", "MATCH_PAIRS"), ("empty", "EMPTY_ARRAY"),
                ("reverse", "REVERSE"), ("tile", "TILE"), ("arrange_grid", "ARRANGE_GRID"),
                ("single_object_array", "SINGLE_OBJECT_ARRAY")
            ]:
                if last_node_type == node_type_class and node_type_key in selection_weights:
                    old_weight = selection_weights[node_type_key]
                    selection_weights[node_type_key] *= 0.3  # 同じノードの連続を大幅に抑制
                    if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                        print(f"[DEBUG] 同じノード連続抑制: {last_node_type} -> {node_type_key}を0.3倍 ({old_weight:.6f} -> {selection_weights[node_type_key]:.6f})", flush=True)

    # context.for_arraysにNoneが含まれている場合、APPENDの確率を上げる（定数FORループで要素追加が一般的）
    if None in context.for_arrays:
        selection_weights["append"] *= 3.5  # Noneを含む場合はAPPEND確率を上げる（10.0→3.5に削減）
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] for_arraysにNone検出: APPEND確率を3.5倍に調整", flush=True)

    # FOR内のIF内では、APPENDの確率をさらに上げる（条件付きで要素追加が一般的）
    if context.get_for_nesting_depth() > 0 and context.get_if_nesting_depth() > 0:
        selection_weights["append"] *= 3.5  # FOR内のIF内ではAPPEND確率を上げる（10.0→3.5に削減）
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] FOR内のIF内検出: APPEND確率を3.5倍に調整（FOR深度={context.get_for_nesting_depth()}, IF深度={context.get_if_nesting_depth()}）", flush=True)

    # ネスト内（スコープ深度1以上）では、filterの重みを下げる
    current_scope_depth = context.get_scope_nesting_depth()
    if current_scope_depth > 0:
        selection_weights["filter"] *= 0.05  # ネスト内ではfilterの重みを0.2倍に下げる
        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
            print(f"[DEBUG] ネスト内検出（深度={current_scope_depth}）: filterの重みを0.2倍に調整", flush=True)

    return selection_weights

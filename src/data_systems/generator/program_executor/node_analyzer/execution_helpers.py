"""
実行ヘルパー関数

プログラム実行、変数状態取得、fallback_template実行などのヘルパー関数
"""
import numpy as np
from typing import List, Any, Optional, Dict, Tuple
from copy import deepcopy

from .node_analyzer import is_if_node, is_for_node, is_end_node
from ..validation.commands import COMMAND_METADATA
from ..node_replacer import parse_fallback_code, find_and_replace_command_in_tree


def _find_outermost_for_loop_info(nodes_list: List[Any], current_index: int) -> Optional[Tuple[int, str]]:
    """現在のノード位置から最も外側のFORループの情報を取得

    Args:
        nodes_list: ノードリスト
        current_index: 現在のノードのインデックス

    Returns:
        (for_loop_index, loop_var_name) のタプル、見つからない場合はNone
    """
    for_depth = 0
    outermost_for_index = None
    outermost_loop_var = None

    for i in range(current_index + 1):
        node = nodes_list[i]
        if is_for_node(node):
            if for_depth == 0:
                # 最も外側のFORループ
                outermost_for_index = i
                # ループ変数名を取得
                if hasattr(node, 'loop_var'):
                    outermost_loop_var = node.loop_var
                elif hasattr(node, 'variable'):
                    outermost_loop_var = node.variable
            for_depth += 1
        elif is_end_node(node):
            if for_depth > 0:
                for_depth -= 1
                if for_depth == 0:
                    # 最も外側のFORループが閉じられた
                    if outermost_for_index is not None and i >= current_index:
                        # 現在のノードがFORループの外にある場合はNoneを返す
                        return None

    if outermost_for_index is not None and for_depth > 0:
        return (outermost_for_index, outermost_loop_var)
    return None


def _is_single_object_assignment(node: Any, assignment_variable: Optional[str]) -> bool:
    """代入先がオブジェクト単体かどうかを判定

    Args:
        node: 検証対象のノード
        assignment_variable: 代入先変数名

    Returns:
        オブジェクト単体の場合はTrue、配列や配列要素の場合はFalse
    """
    if assignment_variable is None:
        return False

    # 配列要素アクセス（例: objects[i]）の場合はFalse
    if '[' in assignment_variable and ']' in assignment_variable:
        return False

    # ノードタイプを確認
    node_type = type(node).__name__

    # ArrayAssignmentNodeの場合は配列要素への代入なのでFalse
    if node_type == 'ArrayAssignmentNode':
        return False

    # 配列を生成するNode型の場合はFalse
    # target_array属性を持つNode型（配列を生成）
    if hasattr(node, 'target_array'):
        return False

    # array_name属性を持つNode型（配列を生成）
    if hasattr(node, 'array_name'):
        return False

    # MergeNodeはオブジェクト単体を生成するのでTrue（チェック不要）
    # ObjectAccessNodeはオブジェクト単体を取得するのでTrue（チェック不要）
    # AssignmentNodeは式の結果によるが、デフォルトでTrueとして扱う

    # その他の場合はオブジェクト単体の可能性がある
    return True


def _add_test_objects_tracking(nodes_list: List[Any], current_index: int, assignment_variable: Optional[str]) -> Tuple[List[Any], Optional[str]]:
    """FORループ内のオブジェクト単体の代入を追跡するためのノードを追加

    Args:
        nodes_list: ノードリスト
        current_index: 現在のノードのインデックス
        assignment_variable: 代入先変数名

    Returns:
        (修正されたノードリスト, test_objects変数名) のタプル
        追加不要の場合は元のリストとNoneを返す
    """
    # FORループ内にいるか、代入先がオブジェクト単体かを確認
    for_info = _find_outermost_for_loop_info(nodes_list, current_index)
    if for_info is None:
        return nodes_list, None

    if not _is_single_object_assignment(nodes_list[current_index], assignment_variable):
        return nodes_list, None

    outermost_for_index, loop_var = for_info
    if loop_var is None:
        return nodes_list, None

    # ユニークな変数名を生成
    test_objects_var = "test_objects"
    # 既存の変数名と衝突しないように確認
    existing_vars = set()
    for node in nodes_list:
        if hasattr(node, 'variable'):
            existing_vars.add(node.variable)
        if hasattr(node, 'array'):
            existing_vars.add(node.array)
        if hasattr(node, 'target_array'):
            existing_vars.add(node.target_array)
        if hasattr(node, 'source_array'):
            existing_vars.add(node.source_array)
    counter = 0
    while test_objects_var in existing_vars:
        test_objects_var = f"test_objects_{counter}"
        counter += 1

    # 新しいノードリストを作成
    modified_nodes = nodes_list.copy()

    # 1. 最も外側のFORループの直前に `test_objects = []` を挿入
    from src.data_systems.generator.program_generator.generation.nodes import EmptyArrayNode
    empty_array_node = EmptyArrayNode(array_name=test_objects_var)
    modified_nodes.insert(outermost_for_index, empty_array_node)

    # インデックスがずれたので調整
    new_current_index = current_index + 1

    # 2. 現在のノード（検証ノード）の直後に `test_objects[loop_var] = APPEND(test_objects, assignment_variable)` を挿入
    from src.data_systems.generator.program_generator.generation.nodes import ArrayAssignmentNode, VariableNode, CommandNode
    # APPENDコマンドを作成
    append_command = CommandNode(
        command="APPEND",
        arguments=[VariableNode(name=test_objects_var), VariableNode(name=assignment_variable)],
        context=None
    )
    # APPENDの結果を配列要素に代入
    array_assignment_node = ArrayAssignmentNode(
        array=test_objects_var,
        index=loop_var,
        expression=append_command
    )

    # 現在のノードの直後に挿入
    modified_nodes.insert(new_current_index + 1, array_assignment_node)

    return modified_nodes, test_objects_var


def complete_incomplete_control_structures(nodes_list: List[Any]) -> List[Any]:
    """未完了のIF文とFORループを検出してENDノードを追加

    Args:
        nodes_list: Nodeリスト

    Returns:
        ENDノードが追加されたNodeリスト（IF文のENDから順に追加）
    """
    completed_nodes = []
    if_depth = 0
    for_loop_depth = 0

    for node in nodes_list:
        completed_nodes.append(node)

        if is_if_node(node):
            if_depth += 1
        elif is_for_node(node):
            for_loop_depth += 1
        elif is_end_node(node):
            # ENDノードは、最も内側のIFまたはFORを閉じる
            # スタック順に従い、FORループが優先される（一般的な実装に合わせる）
            if for_loop_depth > 0:
                for_loop_depth -= 1
            elif if_depth > 0:
                if_depth -= 1

    # 未完了の構造があればENDノードを追加
    # IF文のENDを先に追加（内側から外側へ）、その後FORループのENDを追加
    if if_depth > 0 or for_loop_depth > 0:
        from src.data_systems.generator.program_generator.generation.nodes import EndNode
        print(f"      [END追加] 未完了の構造を検出: IF_depth={if_depth}, FOR_depth={for_loop_depth}")
        # IF文のENDを先に追加
        for _ in range(if_depth):
            completed_nodes.append(EndNode())
        # FORループのENDを追加
        for _ in range(for_loop_depth):
            completed_nodes.append(EndNode())
        print(f"      [END追加] ENDノードを追加完了: IF={if_depth}個, FOR={for_loop_depth}個")
    else:
        print(f"      [END追加] 未完了の構造なし（すべて閉じられています）")

    return completed_nodes


def _extract_object_details(snapshot: Any, obj: Any) -> Optional[Dict[str, Any]]:
    """オブジェクトインスタンスから詳細情報を取得

    Args:
        snapshot: ExecutionSnapshotインスタンス
        obj: Objectインスタンスまたはobject_id（str）

    Returns:
        オブジェクトの詳細情報の辞書。見つからない場合はNone
    """
    if snapshot is None or snapshot.objects is None:
        return None

    # objがObjectインスタンスの場合、object_idを取得
    object_id = None
    if hasattr(obj, 'object_id'):
        object_id = obj.object_id
    elif isinstance(obj, str):
        object_id = obj

    if object_id is None:
        return None

    # snapshot.objectsから詳細情報を取得
    if object_id in snapshot.objects:
        return snapshot.objects[object_id]

    return None


def _extract_variable_info_from_snapshot(
    snapshot: Any,
    assignment_variable: str
) -> Dict[str, Any]:
    """スナップショットから変数情報を抽出

    Args:
        snapshot: ExecutionSnapshotインスタンス
        assignment_variable: 代入先変数名

    Returns:
        変数情報の辞書。オブジェクトの場合は詳細情報も含む
    """
    variable_value = None
    variable_type = None
    object_details = None
    object_array_details = None

    if snapshot is not None:
        if assignment_variable in snapshot.variables:
            variable_value = snapshot.variables[assignment_variable]

            # 変数値がリストの場合、配列として扱う（オブジェクト配列の可能性）
            if isinstance(variable_value, list):
                variable_type = 'array'

                # 配列の各要素がObjectインスタンスまたはobject_idの場合、詳細情報を取得
                if len(variable_value) > 0:
                    from src.data_systems.data_models.core.object import Object
                    object_array_details = []
                    for obj in variable_value:
                        obj_details = None
                        if isinstance(obj, Object):
                            obj_details = _extract_object_details(snapshot, obj)
                        elif isinstance(obj, str):
                            # 要素が文字列（object_id）の場合
                            obj_details = _extract_object_details(snapshot, obj)

                        if obj_details:
                            object_array_details.append(obj_details)
            else:
                variable_type = 'variable'

                # 変数値がObjectインスタンスの場合、詳細情報を取得
                from src.data_systems.data_models.core.object import Object
                if isinstance(variable_value, Object):
                    object_details = _extract_object_details(snapshot, variable_value)
                elif isinstance(variable_value, str):
                    # 変数値が文字列（object_id）の場合
                    object_details = _extract_object_details(snapshot, variable_value)

        elif assignment_variable in snapshot.arrays:
            variable_value = snapshot.arrays[assignment_variable]
            variable_type = 'array'

            # 配列の各要素がObjectインスタンスまたはobject_idの場合、詳細情報を取得
            if isinstance(variable_value, list) and len(variable_value) > 0:
                from src.data_systems.data_models.core.object import Object
                object_array_details = []
                for obj in variable_value:
                    obj_details = None
                    if isinstance(obj, Object):
                        obj_details = _extract_object_details(snapshot, obj)
                    elif isinstance(obj, str):
                        # 要素が文字列（object_id）の場合
                        obj_details = _extract_object_details(snapshot, obj)

                    if obj_details:
                        object_array_details.append(obj_details)

    variable_value_type = type(variable_value).__name__ if variable_value is not None else 'None'
    array_length = len(variable_value) if variable_type == 'array' and isinstance(variable_value, list) else None

    result = {
        'assignment_variable': assignment_variable,
        'variable_type': variable_type,
        'variable_value': variable_value,
        'variable_value_type': variable_value_type,
        'array_length': array_length
    }

    # オブジェクト詳細情報を追加
    if object_details is not None:
        result['object_details'] = object_details
    if object_array_details is not None:
        result['object_array_details'] = object_array_details

    return result


def _normalize_pixels(pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """ピクセルリストを正規化（位置を相対化）

    Args:
        pixels: ピクセル座標のリスト

    Returns:
        正規化されたピクセルリスト（最小座標を(0,0)にシフト）
    """
    if not pixels:
        return []

    min_x = min(p[0] for p in pixels)
    min_y = min(p[1] for p in pixels)

    return [(p[0] - min_x, p[1] - min_y) for p in pixels]


def _compare_object_details(
    obj1_details: Dict[str, Any],
    obj2_details: Dict[str, Any]
) -> bool:
    """2つのオブジェクト詳細情報を比較（位置、形状、色）

    Args:
        obj1_details: オブジェクト1の詳細情報
        obj2_details: オブジェクト2の詳細情報

    Returns:
        同じ場合はTrue（位置、形状、色がすべて同じ）
        異なる場合はFalse
    """
    # ピクセル数のチェック
    pixels1 = obj1_details.get('pixels', [])
    pixels2 = obj2_details.get('pixels', [])

    if len(pixels1) != len(pixels2):
        return False

    if len(pixels1) == 0:
        # 両方空の場合は同じとみなす
        return True

    # 位置を考慮した比較（bboxで位置を確認）
    bbox1 = obj1_details.get('bbox', (0, 0, 0, 0))
    bbox2 = obj2_details.get('bbox', (0, 0, 0, 0))

    # bboxが異なる場合は位置が異なる
    if bbox1 != bbox2:
        # ただし、正規化後の形状と色が同じかも確認
        # （位置だけ異なる場合もあるため）
        pass  # 後続の形状・色比較で判定

    # ピクセル単位の形状を比較（位置を正規化して比較）
    normalized_pixels1 = set(_normalize_pixels(pixels1))
    normalized_pixels2 = set(_normalize_pixels(pixels2))

    if normalized_pixels1 != normalized_pixels2:
        return False  # 形状が異なる

    # ピクセル単位の色を比較
    pixel_colors1 = obj1_details.get('pixel_colors', {})
    pixel_colors2 = obj2_details.get('pixel_colors', {})

    # 正規化されたピクセル座標で色を比較
    # 元の座標を正規化座標にマッピング
    min_x1 = min(p[0] for p in pixels1) if pixels1 else 0
    min_y1 = min(p[1] for p in pixels1) if pixels1 else 0
    min_x2 = min(p[0] for p in pixels2) if pixels2 else 0
    min_y2 = min(p[1] for p in pixels2) if pixels2 else 0

    color_map1 = {}
    for (x, y), color in pixel_colors1.items():
        norm_pos = (x - min_x1, y - min_y1)
        color_map1[norm_pos] = color

    color_map2 = {}
    for (x, y), color in pixel_colors2.items():
        norm_pos = (x - min_x2, y - min_y2)
        color_map2[norm_pos] = color

    # 正規化された位置での色を比較
    if set(color_map1.keys()) != set(color_map2.keys()):
        return False  # 色がある位置が異なる

    for norm_pos in color_map1.keys():
        color1_val = color_map1.get(norm_pos)
        color2_val = color_map2.get(norm_pos)
        # numpy.int64などの型の違いを考慮して比較
        if color1_val != color2_val and int(color1_val) != int(color2_val):
            return False  # 色が異なる

    # 位置の比較（bboxの左上座標を比較）
    if bbox1[:2] != bbox2[:2]:
        # 位置が異なるが、形状と色は同じ → False（位置も含めて比較）
        return False

    return True  # 位置、形状、色がすべて同じ


def compare_variable_info(
    variable_info: Optional[Dict[str, Any]],
    fallback_variable_info: Optional[Dict[str, Any]]
) -> Optional[bool]:
    """元のコードとfallback版の変数情報を比較してコマンドの有効性を判定

    Args:
        variable_info: 元のコード（置き換えなし）の変数情報
        fallback_variable_info: fallback版（置き換えあり）の変数情報

    Returns:
        True: 置き換えが有効（fallback版を使うべき、コマンドを置き換えるべき）
        False: 元のコードが有効（置き換えないべき、コマンドを残すべき）
        None: 比較できない（エラーまたは情報不足）
    """
    if variable_info is None or fallback_variable_info is None:
        return None

    # オブジェクト単体の場合
    if 'object_details' in variable_info or 'object_details' in fallback_variable_info:
        obj_details_original = variable_info.get('object_details')
        obj_details_fallback = fallback_variable_info.get('object_details')

        pixels_original = obj_details_original.get('pixels', []) if obj_details_original else []
        pixels_fallback = obj_details_fallback.get('pixels', []) if obj_details_fallback else []

        if len(pixels_original):
            return True

        # 置き換え前が空リスト、置き換え後が空でない → True
        if len(pixels_original) == 0 and len(pixels_fallback) > 0:
            return True

        # 置き換え後が空リスト、置き換え前が空でない → False
        if len(pixels_fallback) == 0 and len(pixels_original) > 0:
            return False

        # 両方とも詳細情報がある場合、位置・形状・色を比較
        if obj_details_original is not None and obj_details_fallback is not None:
            if not _compare_object_details(obj_details_original, obj_details_fallback):
                return False  # 1つでも異なればFalse

        # 両方とも同じ → True（置き換えが有効、fallback版を使用）
        return True

    # オブジェクト配列の場合
    if 'object_array_details' in variable_info or 'object_array_details' in fallback_variable_info:
        obj_array_original = variable_info.get('object_array_details', [])
        obj_array_fallback = fallback_variable_info.get('object_array_details', [])

        # 配列長さの比較を最初に行う（長さが異なりかつそれぞれ1以上かつ置き換え前 < 置き換え後の場合のみFalse）
        # 置き換え後の配列長さが置き換え前より大きく、かつ両方とも1以上の場合、元のコードを維持（False）
        if len(obj_array_original) == 0:
            return True

        # 長さが異なり、かつそれぞれ1以上、かつ置き換え前 < 置き換え後の場合のみFalse
        if len(obj_array_original) >= 1 and len(obj_array_fallback) >= 1 and len(obj_array_original) < len(obj_array_fallback):
            return False  # 配列の長さが異なり、置き換え後が長い場合はFalse（元のコードを維持）
        # それ以外（片方が0、または置き換え前 >= 置き換え後）の場合は、後続のチェックで処理される

        # 置き換え前が空リスト、置き換え後が空でない → True
        if len(obj_array_original) == 0 and len(obj_array_fallback) > 0:
            return True

        # 置き換え後が空リスト、置き換え前が空でない → False
        if len(obj_array_fallback) == 0 and len(obj_array_original) > 0:
            return False

        # 両方とも配列がある場合（長さは同じ）、各要素を比較

        # 各オブジェクトを順序を考慮して比較
        # 同じインデックスのオブジェクト同士を比較（順番が入れ替わっていてもFalse）

        for i, (original_obj, fallback_obj) in enumerate(zip(obj_array_original, obj_array_fallback)):
            if not _compare_object_details(fallback_obj, original_obj):
                return False  # 同じインデックスで異なるオブジェクトがあればFalse（順番が入れ替わっている場合もFalse）

        # すべてのオブジェクトが順序も含めて一致 → True（置き換えが有効、fallback版を使用）
        return True

    # オブジェクト情報がない場合は比較できない
    return None


def execute_and_get_variable_state(
    executor: Any,  # CoreExecutor型だが循環参照を避けるためAnyを使用
    nodes_list: List[Any],  # 補完済みノードリスト（メインループで補完済み）
    input_grid: np.ndarray,
    line_number: int,
    background_color: int,
    assignment_variable: Optional[str] = None,
    node: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """ノードリストまで実行して変数情報を取得

    Args:
        executor: CoreExecutorインスタンス
        nodes_list: 実行するNodeリスト（補完済み、未完了のIF/FORは既にENDノードで閉じられている）
        input_grid: 入力グリッド
        line_number: 実行する行番号（ASTノードのインデックス、0から始まる）
        background_color: 背景色
        assignment_variable: 代入先変数名（指定されない場合、nodeから取得）
        node: 現在のノード（assignment_variableが指定されない場合に使用）

    Returns:
        変数情報の辞書。以下のキーを含む:
        - assignment_variable: 代入先変数名（str）
        - variable_type: 変数の型（'variable'または'array'またはNone）
        - variable_value: 変数の値（Any）
        - variable_value_type: 値の型名（str）
        - array_length: 配列の場合の長さ（int、配列でない場合はNone）
        エラーが発生した場合や変数が見つからない場合はNoneを返す
    """
    try:
        # Nodeリストをプログラム文字列に変換（既に補完済み）
        program_code = executor._nodes_to_string(nodes_list)

        # デバッグ: 生成されたプログラムコードを出力
        print(f"      [DEBUG] 生成されたプログラムコード（{len(nodes_list)}ノード）:")
        for line_idx, line in enumerate(program_code.split('\n')):
            if line.strip():  # 空行以外
                print(f"      [DEBUG]   行{line_idx + 1}: {line}")

        # FORループが正しく閉じられているかを確認
        for_count = sum(1 for node in nodes_list if is_for_node(node))
        end_count = sum(1 for node in nodes_list if is_end_node(node))
        print(f"      [DEBUG] 制御構造の数: FOR={for_count}個, END={end_count}個")
        if for_count > end_count:
            print(f"      [WARNING] FORループが{for_count - end_count}個未閉じです！")
        elif for_count < end_count:
            print(f"      [WARNING] ENDノードが{end_count - for_count}個余っています！")

        # パース後のASTノードインデックスを計算
        # InitializationNodeが複数行（1行または2行）を生成する可能性があるため、
        # ノードインデックスと行番号が一致しない
        # 補完後のノードリストをパースして、実際のASTノード数を取得
        from src.core_systems.executor.parsing.tokenizer import Tokenizer
        from src.core_systems.executor.parsing.parser import Parser

        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(program_code)
        parser = Parser(tokens)
        parsed_ast = parser.parse()
        print(f"      [DEBUG] パース後のASTノード数: {len(parsed_ast)}")

        # パース後のASTノードの最後のインデックスを使用
        # （nodes_up_to_currentの最後のノードが実行された時点の状態を取得するため）
        actual_line_number = len(parsed_ast) - 1 if parsed_ast else 0

        # 代入先変数を決定（core_executor.pyと同じロジックを使用）
        if assignment_variable is None:
            if node is None:
                print(f"      [WARNING] node が None です")
                return None

            node_type_name = type(node).__name__

            # Node型に応じた代入先変数の取得（core_executor.pyと同じロジック）
            if hasattr(node, 'variable'):
                assignment_variable = node.variable if isinstance(node.variable, str) else str(node.variable)
            elif node_type_name == 'ArrayAssignmentNode':
                # ArrayAssignmentNodeの場合は配列全体（array）の情報を取得して比較
                if hasattr(node, 'array'):
                    array_name = node.array if isinstance(node.array, str) else (node.array.name if hasattr(node.array, 'name') else str(node.array))
                    assignment_variable = array_name
            elif hasattr(node, 'target_array'):
                assignment_variable = node.target_array if isinstance(node.target_array, str) else str(node.target_array)
            elif hasattr(node, 'target_obj'):
                assignment_variable = node.target_obj if isinstance(node.target_obj, str) else str(node.target_obj)
            elif hasattr(node, 'obj_var'):
                assignment_variable = node.obj_var if isinstance(node.obj_var, str) else str(node.obj_var)
            elif hasattr(node, 'array_name'):
                assignment_variable = node.array_name if isinstance(node.array_name, str) else str(node.array_name)

            if assignment_variable is None:
                print(f"      [WARNING] 代入先変数を取得できませんでした（node型: {node_type_name}）")
                return None

        # 実行コンテキストを準備して実行
        snapshot = executor.executor.get_state_at_line(
            program_code=program_code,
            input_grid=input_grid,
            line_number=actual_line_number,
            input_image_index=0,
            background_color=background_color
        )

        if snapshot is None:
            print(f"      [WARNING] スナップショットが取得できませんでした")
            return None

        # 変数情報を抽出
        variable_info = _extract_variable_info_from_snapshot(snapshot, assignment_variable)

        print(f"    代入先変数: {assignment_variable}, 型: {variable_info['variable_type']}, 値の型: {variable_info['variable_value_type']}")

        # デバッグ情報: 変数の実際の値を確認
        if variable_info['variable_value'] is not None:
            from src.data_systems.data_models.core.object import Object
            if isinstance(variable_info['variable_value'], Object):
                print(f"      [DEBUG] 変数値はObjectインスタンス: object_id={variable_info['variable_value'].object_id}")
            elif isinstance(variable_info['variable_value'], str) and variable_info['variable_value_type'] == 'str':
                print(f"      [DEBUG] 変数値は文字列: '{variable_info['variable_value']}'")
            elif isinstance(variable_info['variable_value'], list):
                print(f"      [DEBUG] 変数値はリスト: 長さ={len(variable_info['variable_value'])}")
                if len(variable_info['variable_value']) > 0:
                    first_elem = variable_info['variable_value'][0]
                    if isinstance(first_elem, Object):
                        print(f"      [DEBUG] リストの最初の要素はObject: object_id={first_elem.object_id}")
                    else:
                        print(f"      [DEBUG] リストの最初の要素の型: {type(first_elem).__name__}")

        # デバッグ情報: snapshot.objectsの状態を確認
        if snapshot is not None:
            print(f"      [DEBUG] snapshot.objectsのキー数: {len(snapshot.objects) if snapshot.objects else 0}")
            if snapshot.objects and len(snapshot.objects) > 0:
                print(f"      [DEBUG] snapshot.objectsの最初のキー: {list(snapshot.objects.keys())[0]}")

        # オブジェクト詳細情報の表示
        if 'object_details' in variable_info and variable_info['object_details']:
            obj_details = variable_info['object_details']
            print(f"      オブジェクト詳細:")
            print(f"        object_id: {obj_details.get('object_id', 'N/A')}")
            print(f"        pixels数: {len(obj_details.get('pixels', []))}")
            print(f"        bbox: {obj_details.get('bbox', 'N/A')}")
            print(f"        color: {obj_details.get('color', 'N/A')}")
            if obj_details.get('pixel_colors'):
                print(f"        pixel_colors: {obj_details['pixel_colors']}")

        # オブジェクト配列詳細情報の表示
        if 'object_array_details' in variable_info and variable_info['object_array_details']:
            obj_array_details = variable_info['object_array_details']
            print(f"      オブジェクト配列詳細: {len(obj_array_details)}個")
            for i, obj_details in enumerate(obj_array_details[:3]):  # 最初の3個のみ表示
                print(f"        [{i}] object_id: {obj_details.get('object_id', 'N/A')}, pixels数: {len(obj_details.get('pixels', []))}, bbox: {obj_details.get('bbox', 'N/A')}")
            if len(obj_array_details) > 3:
                print(f"        ... 他 {len(obj_array_details) - 3}個")

        # 通常の変数値表示（オブジェクトでない場合）
        if variable_info['variable_type'] == 'array' and variable_info['array_length'] is not None and 'object_array_details' not in variable_info:
            print(f"      配列の長さ: {variable_info['array_length']}")
        elif variable_info['variable_type'] == 'variable' and variable_info['variable_value'] is not None and 'object_details' not in variable_info:
            print(f"      変数の値: {variable_info['variable_value']}")

        return variable_info

    except Exception as e:
        print(f"    [WARNING] ノード実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

    return None


def replace_command_with_fallback(
    cmd_name: str,
    cmd_node: Any,
    nodes_list: List[Any],
    grid_width: Optional[int] = None,
    grid_height: Optional[int] = None
) -> Optional[Tuple[List[Any], str]]:
    """コマンドをfallback_templateで置き換え

    Args:
        cmd_name: コマンド名
        cmd_node: コマンドノード
        nodes_list: ノードリスト

    Returns:
        (置き換え後のノードリスト, fallback_code) のタプル。失敗時はNone
    """
    try:
        # 特別なノードタイプの場合の処理（fallback_templateチェックより先に処理）
        node_type = type(cmd_node).__name__
        from src.data_systems.generator.program_generator.generation.nodes import AssignmentNode, VariableNode

        if node_type == 'FilterNode':
            # FilterNodeの場合、第一引数はsource_array
            if not hasattr(cmd_node, 'source_array'):
                return None
            first_arg_str = cmd_node.source_array
            target_array_name = cmd_node.target_array if hasattr(cmd_node, 'target_array') else 'objects1'

            # 条件部分を解析して、常にTrueまたはFalseになる場合を検出
            if hasattr(cmd_node, 'condition'):
                condition_str = cmd_node.condition
                # EQUAL(GET_COLOR($obj), GET_COLOR($obj))のように、同じ引数のEQUALを検出
                import re

                # 括弧のネストを考慮して、引数を抽出する関数
                def extract_args(func_call_str):
                    """関数呼び出し文字列から引数を抽出"""
                    func_call_str = func_call_str.strip()
                    # 関数名と引数部分を分離（最初の開き括弧から最後の閉じ括弧まで）
                    # 例: "EQUAL(GET_COLOR($obj), GET_COLOR($obj))" → func_name="EQUAL", args_str="GET_COLOR($obj), GET_COLOR($obj)"
                    match = re.match(r'(\w+)\s*\((.*)\)\s*$', func_call_str)
                    if not match:
                        return None, []
                    func_name = match.group(1)
                    args_str = match.group(2)

                    # 括弧のネストを考慮して引数を分割
                    args = []
                    depth = 0
                    current_arg = ""
                    for char in args_str:
                        if char == '(':
                            depth += 1
                            current_arg += char
                        elif char == ')':
                            depth -= 1
                            current_arg += char
                        elif char == ',' and depth == 0:
                            args.append(current_arg.strip())
                            current_arg = ""
                        else:
                            current_arg += char
                    if current_arg.strip():
                        args.append(current_arg.strip())

                    return func_name, args

                func_name, args = extract_args(condition_str)

                if func_name and len(args) == 2:
                    # 引数が完全に同じかチェック（空白を無視）
                    arg1 = args[0].strip()
                    arg2 = args[1].strip()

                    if arg1 == arg2:
                        # 常にTrueになる条件: EQUAL(同じ引数, 同じ引数)
                        if func_name == 'EQUAL':
                            # objects = objects に置き換え
                            from src.data_systems.generator.program_generator.generation.nodes import VariableNode, AssignmentNode
                            replacement_expr = VariableNode(first_arg_str, cmd_node.context if hasattr(cmd_node, 'context') else None)
                            replacement_node = AssignmentNode(target_array_name, replacement_expr, cmd_node.context if hasattr(cmd_node, 'context') else None)
                            fallback_code = f"{target_array_name} = {first_arg_str}"
                            return ([replacement_node], fallback_code)

                        # 常にFalseになる条件: NOT_EQUAL/GREATER/LESS(同じ引数, 同じ引数)
                        if func_name in ['NOT_EQUAL', 'GREATER', 'LESS']:
                            # objects = [] に置き換え
                            from src.data_systems.generator.program_generator.generation.nodes import LiteralNode, AssignmentNode
                            replacement_expr = LiteralNode([], cmd_node.context if hasattr(cmd_node, 'context') else None)
                            replacement_node = AssignmentNode(target_array_name, replacement_expr, cmd_node.context if hasattr(cmd_node, 'context') else None)
                            fallback_code = f"{target_array_name} = []"
                            return ([replacement_node], fallback_code)

                # 条件チェックで置き換えできなかった場合、fallback_templateを使用するために処理を続行
                # （このブロックを抜けて、869行目の処理に進む）

        elif node_type == 'ExcludeNode':
            # ExcludeNodeの場合、第一引数はsource_array
            if not hasattr(cmd_node, 'source_array'):
                return None
            first_arg_str = cmd_node.source_array
            target_array_name = cmd_node.target_array if hasattr(cmd_node, 'target_array') else 'objects1'

        elif node_type == 'ConcatNode':
            # ConcatNodeの場合、第一引数はarray1
            if not hasattr(cmd_node, 'array1'):
                return None
            first_arg_str = cmd_node.array1
            target_array_name = cmd_node.target_array if hasattr(cmd_node, 'target_array') else 'objects1'

        elif node_type == 'AppendNode':
            # AppendNodeの場合、第一引数はarray
            if not hasattr(cmd_node, 'array'):
                return None
            first_arg_str = cmd_node.array
            target_array_name = cmd_node.target_array if hasattr(cmd_node, 'target_array') else 'objects1'

        elif node_type == 'MergeNode':
            # MergeNodeの場合、第一引数はobjects_array
            if not hasattr(cmd_node, 'objects_array'):
                return None
            first_arg_str = cmd_node.objects_array
            target_obj_name = cmd_node.target_obj if hasattr(cmd_node, 'target_obj') else 'object'
            target_array_name = None  # MergeNodeはtarget_array_nameを使用しない

            # MERGEのfallbackはCREATE_LINEなので、ランダムな引数を生成
            import random
            from src.data_systems.generator.program_generator.generation.nodes import CommandNode, LiteralNode

            # grid_width/grid_heightを取得（引数 > context > デフォルト値）
            context = cmd_node.context if hasattr(cmd_node, 'context') and cmd_node.context else {}
            if grid_width is None:
                grid_width = context.get('output_grid_width', 30)
            if grid_height is None:
                grid_height = context.get('output_grid_height', 30)

            # CREATE_LINEの引数をランダム生成
            # x: 0 ～ grid_width - 1
            x = random.randint(0, max(0, grid_width - 1))
            # y: 0 ～ grid_height - 1
            y = random.randint(0, max(0, grid_height - 1))
            # length: 2 ～ min(grid_width, grid_height)
            max_length = min(grid_width, grid_height)
            length = random.randint(2, max(2, max_length))
            # direction: 8方向からランダム選択
            directions = ["X", "Y", "-X", "-Y", "XY", "-XY", "X-Y", "-X-Y"]
            direction = random.choice(directions)
            # color: 0 ～ 9
            color = random.randint(0, 9)

            # CREATE_LINEのCommandNodeを作成
            create_line_args = [
                LiteralNode(x, context),
                LiteralNode(y, context),
                LiteralNode(length, context),
                LiteralNode(direction, context),
                LiteralNode(color, context)
            ]
            create_line_node = CommandNode('CREATE_LINE', create_line_args, context)

            # AssignmentNodeとして作成
            replacement_node = AssignmentNode(target_obj_name, create_line_node, context)

            # fallback_codeは生成用に保持
            fallback_code = f"{target_obj_name} = CREATE_LINE({x}, {y}, {length}, \"{direction}\", {color})"

        elif node_type in ['MatchPairsNode', 'ExtendPatternNode', 'ArrangeGridNode', 'ExtractShapeNode']:
            # これらのノードタイプも第一引数はsource_array
            if not hasattr(cmd_node, 'source_array'):
                return None
            first_arg_str = cmd_node.source_array
            target_array_name = cmd_node.target_array if hasattr(cmd_node, 'target_array') else 'objects1'

        else:
            # 上記以外の場合は従来通りCommandNodeとして処理
            first_arg_str = None
            target_array_name = None

        # FilterNodeの場合は既に処理済み（条件チェックで早期リターン）
        if node_type == 'FilterNode':
            # 条件チェックで置き換えできなかった場合は、fallback_templateを使用
            cmd_metadata = COMMAND_METADATA.get(cmd_name)
            if not cmd_metadata or not cmd_metadata.fallback_template:
                return None
            # fallback_templateの{arg0}を第一引数に置き換え
            # fallback_template='{arg0}'の場合、fallback_code='objects'になる
            fallback_code_str = cmd_metadata.fallback_template.replace('{arg0}', first_arg_str)
            # fallback_codeは置き換え後のコード（例: "objects = objects"）
            fallback_code = f"{target_array_name} = {fallback_code_str}"

            # ノード全体を置き換えるために、AssignmentNodeとして扱う
            # fallback_codeは単なる変数名なので、target_array = source_array という代入に変換
            replacement_expr = VariableNode(first_arg_str, cmd_node.context if hasattr(cmd_node, 'context') else None)
            replacement_node = AssignmentNode(target_array_name, replacement_expr, cmd_node.context if hasattr(cmd_node, 'context') else None)
        # 特別なノードタイプで、第一引数を取得できた場合（MergeNodeは既に処理済み）
        elif first_arg_str is not None and target_array_name is not None and node_type != 'MergeNode':
            cmd_metadata = COMMAND_METADATA.get(cmd_name)
            if not cmd_metadata or not cmd_metadata.fallback_template:
                return None
            # fallback_templateの{arg0}を第一引数に置き換え
            fallback_code = cmd_metadata.fallback_template.replace('{arg0}', first_arg_str)

            # ノード全体を置き換えるために、AssignmentNodeとして扱う
            # fallback_codeは単なる変数名なので、target_array = source_array という代入に変換
            replacement_expr = VariableNode(first_arg_str, cmd_node.context if hasattr(cmd_node, 'context') else None)
            replacement_node = AssignmentNode(target_array_name, replacement_expr, cmd_node.context if hasattr(cmd_node, 'context') else None)
        elif node_type == 'MergeNode':
            # MergeNodeは既に処理済みなので何もしない（fallback_codeとreplacement_nodeは既に設定済み）
            pass
        else:
            # 通常のCommandNodeの場合
            # 第一引数を取得
            if not (hasattr(cmd_node, 'arguments') and cmd_node.arguments and len(cmd_node.arguments) > 0):
                return None

            first_arg_node = cmd_node.arguments[0]
            first_arg_str = first_arg_node.generate() if hasattr(first_arg_node, 'generate') else str(first_arg_node)
            if not first_arg_str:
                return None

            # cmd_metadataを取得
            cmd_metadata = COMMAND_METADATA.get(cmd_name)
            if not cmd_metadata or not cmd_metadata.fallback_template:
                return None

            # fallback_templateの{arg0}を第一引数に置き換え
            fallback_code = cmd_metadata.fallback_template.replace('{arg0}', first_arg_str)
            replacement_node = parse_fallback_code(fallback_code)

        # ノードツリーをコピーして置き換え
        nodes_with_fallback = deepcopy(nodes_list)

        # 特別なノードタイプの場合は、ノードリスト内で直接ノードを見つけて置き換え
        special_node_types = ['FilterNode', 'ExcludeNode', 'ConcatNode', 'AppendNode', 'MergeNode',
                              'MatchPairsNode', 'ExtendPatternNode', 'ArrangeGridNode', 'ExtractShapeNode']
        if node_type in special_node_types:
            # 特別なノードタイプをIDまたはオブジェクト参照で探して置き換え
            target_node_id = getattr(cmd_node, 'id', None)
            replaced_nodes = []
            for orig_node, copied_node in zip(nodes_list, nodes_with_fallback):
                if (hasattr(copied_node, '__class__') and
                    copied_node.__class__.__name__ == node_type):
                    # IDで比較
                    if target_node_id is not None:
                        copied_id = getattr(copied_node, 'id', None)
                        if copied_id == target_node_id:
                            replaced_nodes.append(replacement_node)
                            continue
                    # オブジェクト参照で比較（ノードタイプに応じた属性で比較）
                    if node_type == 'FilterNode' or node_type == 'ExcludeNode':
                        if (hasattr(copied_node, 'source_array') and
                            hasattr(copied_node, 'target_array') and
                            copied_node.source_array == getattr(cmd_node, 'source_array', None) and
                            copied_node.target_array == getattr(cmd_node, 'target_array', None)):
                            replaced_nodes.append(replacement_node)
                            continue
                    elif node_type == 'ConcatNode':
                        if (hasattr(copied_node, 'array1') and
                            hasattr(copied_node, 'target_array') and
                            copied_node.array1 == getattr(cmd_node, 'array1', None) and
                            copied_node.target_array == getattr(cmd_node, 'target_array', None)):
                            replaced_nodes.append(replacement_node)
                            continue
                    elif node_type == 'AppendNode':
                        if (hasattr(copied_node, 'array') and
                            hasattr(copied_node, 'target_array') and
                            copied_node.array == getattr(cmd_node, 'array', None) and
                            copied_node.target_array == getattr(cmd_node, 'target_array', None)):
                            replaced_nodes.append(replacement_node)
                            continue
                    elif node_type == 'MergeNode':
                        if (hasattr(copied_node, 'objects_array') and
                            hasattr(copied_node, 'target_obj') and
                            copied_node.objects_array == getattr(cmd_node, 'objects_array', None) and
                            copied_node.target_obj == getattr(cmd_node, 'target_obj', None)):
                            replaced_nodes.append(replacement_node)
                            continue
                    elif node_type in ['MatchPairsNode', 'ExtendPatternNode', 'ArrangeGridNode', 'ExtractShapeNode']:
                        if (hasattr(copied_node, 'source_array') and
                            hasattr(copied_node, 'target_array') and
                            copied_node.source_array == getattr(cmd_node, 'source_array', None) and
                            copied_node.target_array == getattr(cmd_node, 'target_array', None)):
                            replaced_nodes.append(replacement_node)
                            continue
                replaced_nodes.append(copied_node)
            nodes_with_fallback_replaced = replaced_nodes
        else:
            nodes_with_fallback_replaced = find_and_replace_command_in_tree(
                nodes_with_fallback, cmd_node, replacement_node, original_nodes=nodes_list
            )

        return (nodes_with_fallback_replaced, fallback_code)
    except Exception as e:
        print(f"    [WARNING] Fallback置き換えエラー: {e}")
        return None


def execute_with_fallback(
    executor: Any,
    cmd_name: str,
    cmd_node: Any,
    nodes_up_to_current: List[Any],
    temp_input_grid: np.ndarray,
    background_color: int,
    assignment_variable: Optional[str] = None,
    line_number: int = 0
) -> Optional[Dict[str, Any]]:
    """fallback_templateによる置き換えと実行、変数情報取得

    Args:
        executor: CoreExecutorインスタンス
        cmd_name: コマンド名
        cmd_node: コマンドノード
        nodes_up_to_current: 現在までのノードリスト
        temp_input_grid: 一時入力グリッド
        background_color: 背景色
        assignment_variable: 代入先変数名
        line_number: 行番号（使用されないが、共通インターフェースのため）

    Returns:
        変数情報の辞書（execute_and_get_variable_stateと同じ構造 + fallback_code, output_grid_shape）
    """
    # 1. 置き換え処理
    replace_result = replace_command_with_fallback(cmd_name, cmd_node, nodes_up_to_current)
    if replace_result is None:
        return None

    nodes_with_fallback, fallback_code = replace_result

    # 2. 置き換え後のプログラムを実行（変数情報取得と同じ処理を使用）
    variable_info = execute_and_get_variable_state(
        executor, nodes_with_fallback, temp_input_grid, line_number,
        background_color, assignment_variable=assignment_variable
    )

    if variable_info is None:
        return None

    # 3. fallback固有の情報を追加
    # 出力グリッドの形状を取得（実行は既に完了しているため、再実行は不要）
    # ただし、変数情報取得時には出力グリッドは取得していないため、ここではNoneとする
    variable_info['fallback_code'] = fallback_code
    variable_info['output_grid_shape'] = None  # 必要に応じて追加取得

    print(f"    [Fallback] 置き換え後の実行結果:")
    print(f"      元のコード: {cmd_name}(...)")
    print(f"      置き換え後: {fallback_code}")

    return variable_info

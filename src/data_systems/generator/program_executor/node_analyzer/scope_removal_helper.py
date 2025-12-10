"""
スコープ削除検証のヘルパー関数
"""
from typing import List, Tuple, Optional, Any, Dict
from .node_analyzer import is_if_node, is_for_node, is_end_node


def find_scope_pairs(nodes: List[Any]) -> List[Dict[str, Any]]:
    """ノードリストからIF/END、FOR/ENDのスコープペアを検出し、深さ順にソートして返す

    Args:
        nodes: ノードリスト

    Returns:
        スコープ情報のリスト [{'type': 'IF' or 'FOR', 'start_idx': int, 'end_idx': int, 'depth': int}, ...]
        深いスコープから順にソート済み（depthが大きい順）
    """
    scope_pairs = []
    scope_stack = []  # [(type, start_idx, depth), ...]

    for i, node in enumerate(nodes):
        if is_if_node(node):
            # 現在の深さを計算
            current_depth = len(scope_stack)
            scope_stack.append(('IF', i, current_depth))
        elif is_for_node(node):
            current_depth = len(scope_stack)
            scope_stack.append(('FOR', i, current_depth))
        elif is_end_node(node):
            if scope_stack:
                scope_type, start_idx, depth = scope_stack.pop()
                scope_pairs.append({
                    'type': scope_type,
                    'start_idx': start_idx,
                    'end_idx': i,
                    'depth': depth
                })

    # 深いスコープから順にソート（depthが大きい順）
    scope_pairs.sort(key=lambda x: x['depth'], reverse=True)

    return scope_pairs


def remove_scope_from_nodes(nodes: List[Any], scope_info: Dict[str, Any]) -> List[Any]:
    """ノードリストから指定されたスコープ（IF/END、FOR/END）を削除

    Args:
        nodes: 元のノードリスト
        scope_info: スコープ情報 {'type': str, 'start_idx': int, 'end_idx': int, 'depth': int}

    Returns:
        スコープを削除したノードリスト（新しいリスト）
    """
    start_idx = scope_info['start_idx']
    end_idx = scope_info['end_idx']

    # スコープ内のノード（IF/FORからENDまで）をすべて削除
    new_nodes = nodes[:start_idx] + nodes[end_idx + 1:]

    return new_nodes

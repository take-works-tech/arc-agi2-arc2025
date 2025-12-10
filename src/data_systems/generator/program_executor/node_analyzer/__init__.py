"""
Node解析ユーティリティモジュール
"""
from .node_analyzer import (
    is_assignment_node,
    get_assignment_nodes,
    is_if_node,
    is_for_node,
    is_end_node,
    count_if_and_for_nodes,
    extract_commands_from_node,
    extract_all_node_info,
    extract_commands_with_depth,
    get_commands_sorted_by_depth
)
from .execution_helpers import (
    complete_incomplete_control_structures,
    execute_and_get_variable_state,
    replace_command_with_fallback,
    compare_variable_info
)
from .variable_analyzer import (
    remove_unused_variables,
    remove_self_assignments,
    extract_variable_definitions_from_node,
    extract_variable_usages_from_node
)

__all__ = [
    'is_assignment_node',
    'get_assignment_nodes',
    'is_if_node',
    'is_for_node',
    'is_end_node',
    'count_if_and_for_nodes',
    'extract_commands_from_node',
    'extract_all_node_info',
    'extract_commands_with_depth',
    'get_commands_sorted_by_depth',
    'complete_incomplete_control_structures',
    'execute_and_get_variable_state',
    'replace_command_with_fallback',
    'compare_variable_info',
    'remove_unused_variables',
    'remove_self_assignments',
    'extract_variable_definitions_from_node',
    'extract_variable_usages_from_node'
]

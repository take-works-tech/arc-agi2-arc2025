"""
Program Executor - プログラム実行・検証・修正モジュール

入力グリッド生成以外の処理を担当:
- プログラム実行（CoreExecutor）
- ノード検証とオブジェクト調整（node_validator_output）
- ノード置き換え（node_replacer）
- ノード分析（node_analyzer）
- コマンド検証メタデータ（validation）
- パフォーマンスプロファイラー（performance_profiler）
"""

# プログラム実行系モジュール
try:
    from .core_executor import (
        CoreExecutor,
        main as core_executor_main
    )
except ImportError as e:
    print(f"Warning: Failed to import core_executor: {e}")
    CoreExecutor = None
    core_executor_main = None

# ノード検証系モジュール
try:
    from .node_validator_output import (
        validate_nodes_and_adjust_objects,
        check_output_conditions,
        extract_all_commands_from_nodes
    )
except ImportError as e:
    print(f"Warning: Failed to import node_validator_output: {e}")
    validate_nodes_and_adjust_objects = None
    check_output_conditions = None
    extract_all_commands_from_nodes = None

# ノード置き換え系モジュール
try:
    from .node_replacer import (
        parse_fallback_code,
        find_and_replace_command_in_tree,
        replace_command_node_in_tree,
        replace_command_node_by_id,
        replace_command_node_by_code
    )
except ImportError as e:
    print(f"Warning: Failed to import node_replacer: {e}")
    parse_fallback_code = None
    find_and_replace_command_in_tree = None
    replace_command_node_in_tree = None
    replace_command_node_by_id = None
    replace_command_node_by_code = None

# ノード分析系モジュール
try:
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
    from .node_analyzer.execution_helpers import (
        complete_incomplete_control_structures,
        execute_and_get_variable_state,
        replace_command_with_fallback,
        compare_variable_info
    )
    from .node_analyzer.variable_analyzer import (
        remove_unused_variables,
        remove_self_assignments,
        extract_variable_definitions_from_node,
        extract_variable_usages_from_node
    )
except ImportError as e:
    print(f"Warning: Failed to import node_analyzer: {e}")
    is_assignment_node = None
    get_commands_sorted_by_depth = None
    replace_command_with_fallback = None

# コマンド検証メタデータ
try:
    from .validation.commands import (
        COMMAND_METADATA,
        CommandMetadata
    )
except ImportError as e:
    print(f"Warning: Failed to import validation.commands: {e}")
    COMMAND_METADATA = {}
    CommandMetadata = None

# パフォーマンスプロファイラー
try:
    from .performance_profiler import (
        PerformanceProfiler,
        profile_function,
        profile_code_block,
        get_profiler,
        reset_profiler,
        print_profiling_statistics
    )
except ImportError as e:
    print(f"Warning: Failed to import performance_profiler: {e}")
    PerformanceProfiler = None
    profile_code_block = None

__all__ = [
    # プログラム実行
    'CoreExecutor',
    'core_executor_main',
    # ノード検証
    'validate_nodes_and_adjust_objects',
    'check_output_conditions',
    'extract_all_commands_from_nodes',
    # ノード置き換え
    'parse_fallback_code',
    'find_and_replace_command_in_tree',
    'replace_command_node_in_tree',
    'replace_command_node_by_id',
    'replace_command_node_by_code',
    # ノード分析
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
    'extract_variable_usages_from_node',
    # コマンド検証
    'COMMAND_METADATA',
    'CommandMetadata',
    # パフォーマンスプロファイラー
    'PerformanceProfiler',
    'profile_function',
    'profile_code_block',
    'get_profiler',
    'reset_profiler',
    'print_profiling_statistics'
]

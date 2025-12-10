"""
部分プログラム生成ヘルパー

インプットグリッドから部分プログラムを生成するためのヘルパー関数
"""
from typing import Tuple, Optional, Dict, List
import numpy as np
import re
import random
import sys

from src.hybrid_system.inference.object_matching.rule_based_matcher import RuleBasedObjectMatcher
from src.hybrid_system.core.data_structures.task import Task
from src.data_systems.generator.program_generator.generation.nodes import (
    Node, InitializationNode, FilterNode
)
from src.data_systems.generator.program_generator.generation.program_context import ProgramContext
from src.data_systems.generator.program_generator.metadata.types import SemanticType
from src.data_systems.generator.config import get_config


def generate_partial_program_from_input_grid(
    input_grid: np.ndarray,
    grid_width: int,
    grid_height: int,
    original_background_color: Optional[int] = None
) -> Tuple[Optional[str], Optional[Dict[str, str]]]:
    """
    インプットグリッドから部分プログラムを生成（既存のオブジェクトマッチング機能を使用）

    Args:
        input_grid: 入力グリッド（2D numpy配列）
        grid_width: グリッド幅
        grid_height: グリッド高さ
        original_background_color: データセット生成時限定：入力グリッド生成時に決定された背景色
            - Noneの場合: 推論パイプライン（従来通り）
            - 指定された場合: データセット生成時。推論背景色と一致する場合はGET_BACKGROUND_COLOR()を使用、
              不一致の場合はリテラル値を使用

    Returns:
        (部分プログラム文字列, カテゴリ変数マッピング) または (None, None)

        カテゴリ変数マッピングの形式: {category_id: variable_name}
        例: {'0': 'objects1', '1': 'objects2'}
    """
    try:
        # 1. RuleBasedObjectMatcherを初期化
        matcher = RuleBasedObjectMatcher()

        # 2. Taskオブジェクトを作成（1つの訓練ペアのみ）
        # 注意: match_objects()は複数ペアを想定しているが、1つのペアでも動作する
        # 入力と出力を同じにする（オブジェクトマッチングは入力グリッドからオブジェクトを抽出し、
        # カテゴリ分類を行うため、出力グリッドは使用されない）
        task = Task(
            train=[{
                'input': input_grid.tolist(),
                'output': input_grid.tolist()  # ダミー（カテゴリ分類には使用されない）
            }],
            test=[],
            program=''  # プログラムは空でOK
        )

        # 3. オブジェクトマッチングを実行（プログラム生成フロー専用の最適化版）
        # 1つのパターンだけを試行して部分プログラムを生成（処理時間を大幅に短縮）
        # 内部で以下が実行される：
        # - オブジェクト抽出
        # - 背景色推論
        # - カテゴリ分類
        # - 部分プログラム生成（1つのパターンのみ）
        # constants.pyのMAX_PARTIAL_PROGRAM_GENERATION_ATTEMPTSを使用
        from src.data_systems.generator.program_generator.metadata.constants import MAX_PARTIAL_PROGRAM_GENERATION_ATTEMPTS

        # データセット生成時限定：背景色情報を渡す
        # 連結性をランダムに選択（4連結と8連結の両方から選択して多様性を確保）
        connectivity = random.choice([4, 8])
        matching_result = matcher.match_objects_single_pattern(
            task,
            connectivity=connectivity,
            max_attempts=MAX_PARTIAL_PROGRAM_GENERATION_ATTEMPTS,
            original_background_color=original_background_color  # データセット生成時限定
        )

        # 4. タイミング情報をログに出力（成功時も失敗時も）
        debug_info = matching_result.get('debug_info', {})
        timing_info = debug_info.get('timing', {})
        if timing_info:
            total_time = timing_info.get('total', 0.0)
            extract_time = timing_info.get('object_extraction', 0.0)
            bg_time = timing_info.get('background_color_inference', 0.0)
            bg_strategy_time = timing_info.get('background_strategy_decision', 0.0)
            pattern_time = timing_info.get('pattern_analysis', 0.0)

            # ループ処理の詳細情報（新しいタイミング情報）
            loop_time = timing_info.get('loop_total', 0.0)
            loop_attempts = timing_info.get('loop_attempts', 0)
            loop_avg_per_attempt = timing_info.get('loop_avg_per_attempt', 0.0)
            loop_max_attempt = timing_info.get('loop_max_attempt', 0.0)
            loop_min_attempt = timing_info.get('loop_min_attempt', 0.0)
            total_category_time = timing_info.get('total_category_time', 0.0)
            total_partial_time = timing_info.get('total_partial_time', 0.0)

            # ループ処理の時間が取得できない場合は、合計時間から他の処理時間を引く（後方互換性）
            if loop_time == 0.0:
                loop_time = total_time - extract_time - bg_time - bg_strategy_time - pattern_time

            # 割合を計算
            if total_time > 0:
                extract_ratio = (extract_time / total_time) * 100
                bg_ratio = (bg_time / total_time) * 100
                bg_strategy_ratio = (bg_strategy_time / total_time) * 100
                pattern_ratio = (pattern_time / total_time) * 100
                loop_ratio = (loop_time / total_time) * 100

                # 詳細タイミングログの制御
                config = get_config()
                if config.enable_detailed_timing_logs:
                    print(f"[TIMING] 部分プログラム生成の時間配分:", file=sys.stderr, flush=True)
                    print(f"  合計時間: {total_time:.3f}秒", file=sys.stderr, flush=True)
                    print(f"  ① オブジェクト抽出: {extract_time:.3f}秒 ({extract_ratio:.1f}%)", file=sys.stderr, flush=True)
                    print(f"  ② 背景色推論: {bg_time:.3f}秒 ({bg_ratio:.1f}%)", file=sys.stderr, flush=True)
                    print(f"  ③ 背景色戦略決定: {bg_strategy_time:.3f}秒 ({bg_strategy_ratio:.1f}%)", file=sys.stderr, flush=True)
                    print(f"  ④ 変換パターン分析: {pattern_time:.3f}秒 ({pattern_ratio:.1f}%)", file=sys.stderr, flush=True)
                    print(f"  ⑤ ループ処理（カテゴリ分類+部分プログラム生成）: {loop_time:.3f}秒 ({loop_ratio:.1f}%)", file=sys.stderr, flush=True)

                # ループ処理の詳細情報を出力
                if loop_attempts > 0:
                    print(f"    試行回数: {loop_attempts}回", file=sys.stderr, flush=True)
                    print(f"    平均時間/試行: {loop_avg_per_attempt:.3f}秒", file=sys.stderr, flush=True)
                    print(f"    最大時間/試行: {loop_max_attempt:.3f}秒", file=sys.stderr, flush=True)
                    print(f"    最小時間/試行: {loop_min_attempt:.3f}秒", file=sys.stderr, flush=True)
                    if total_category_time > 0 or total_partial_time > 0:
                        print(f"    カテゴリ分類合計: {total_category_time:.3f}秒", file=sys.stderr, flush=True)
                        print(f"    部分プログラム生成合計: {total_partial_time:.3f}秒", file=sys.stderr, flush=True)

        # 5. 部分プログラムとカテゴリ変数マッピングを取得
        if not matching_result.get('success', False):
            return None, None

        # matching_resultから部分プログラムを取得
        # 注意: match_objects_single_pattern()は1つのパターンだけを試行して部分プログラムを生成する
        # （処理時間を大幅に短縮：463個のパターン → 1個のパターン）
        partial_programs = matching_result.get('partial_programs', [])
        if not partial_programs:
            return None, None

        # 部分プログラムを取得（1つのみ）
        partial_program = partial_programs[0]

        # カテゴリ変数マッピングを取得
        # 注意: category_var_mappingsは {pattern_key: {category_id: variable_name}} の形式
        # pattern_keyは "connectivity_pattern_idx" の形式（例: "4_0"）
        category_var_mappings = matching_result.get('category_var_mappings', {})
        if category_var_mappings:
            # 最初のパターンのマッピングを使用
            first_pattern_key = list(category_var_mappings.keys())[0]
            category_var_mapping = category_var_mappings[first_pattern_key]
        else:
            # 代替: partial_program_to_category_var_mappingから取得
            partial_program_to_category_var_mapping = matching_result.get(
                'partial_program_to_category_var_mapping', {}
            )
            if partial_program in partial_program_to_category_var_mapping:
                category_var_mapping = partial_program_to_category_var_mapping[partial_program]
            else:
                category_var_mapping = {}

        return partial_program, category_var_mapping

    except Exception as e:
        # エラーが発生した場合はNoneを返す
        import traceback
        print(f"[generate_partial_program_from_input_grid] エラーが発生しました: {e}", file=sys.stderr)
        if __name__ == '__main__' or True:  # デバッグ時のみ
            traceback.print_exc()
        return None, None


def parse_partial_program_to_nodes(
    partial_program: str,
    category_var_mapping: Dict[str, str],
    context: ProgramContext
) -> Tuple[List[Node], Dict[str, str]]:
    """
    部分プログラム文字列をNodeリストに変換

    Args:
        partial_program: 部分プログラム文字列
        category_var_mapping: カテゴリ変数マッピング（カテゴリID -> 変数名）
        context: プログラムコンテキスト

    Returns:
        (Nodeリスト, カテゴリ変数マッピング)
    """
    from src.data_systems.generator.program_generator.generation.nodes import (
        Node, InitializationNode, FilterNode
    )
    from src.data_systems.generator.program_generator.metadata.types import SemanticType
    from src.data_systems.generator.program_generator.generation import OBJECT_ARRAY_TYPE
    import re

    nodes = []
    lines = partial_program.strip().split('\n')

    # カテゴリ変数を特定してcontextに設定
    # category_var_mappingには、カテゴリのオブジェクトを持っている変数のみが含まれる
    # EXCLUDEの場合: 最後のカテゴリ以外の変数名（FILTERで生成）と最後のカテゴリの変数名（EXCLUDEで生成）の両方が含まれる
    # FILTERのみの場合: すべてのカテゴリ変数が含まれる
    # プログラム生成側では、category_var_mappingの値をそのまま使用する
    category_arrays = list(category_var_mapping.values()) if category_var_mapping else []

    # contextにカテゴリ変数を設定
    context.category_arrays = category_arrays

    # 行をパースしてNodeを生成
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            # コメント行はスキップ
            continue

        # 行をパースしてNodeを生成
        node = _parse_line_to_node(line, context)
        if node:
            nodes.append(node)

    return nodes, category_var_mapping


def _parse_line_to_node(line: str, context: ProgramContext) -> Optional[Node]:
    """
    1行の部分プログラムをパースしてNodeを生成

    Args:
        line: 部分プログラムの1行
        context: プログラムコンテキスト

    Returns:
        Nodeオブジェクト（パースに失敗した場合はNone）
    """
    from src.data_systems.generator.program_generator.generation.nodes import (
        InitializationNode, FilterNode, ExcludeNode, MergeNode
    )
    from src.data_systems.generator.program_generator.metadata.types import SemanticType
    from src.data_systems.generator.program_generator.generation import OBJECT_ARRAY_TYPE
    import re

    # 1. GET_ALL_OBJECTSのパターン
    # 例: "objects = GET_ALL_OBJECTS(4)"
    match = re.match(r'(\w+)\s*=\s*GET_ALL_OBJECTS\((\d+)\)', line)
    if match:
        var_name = match.group(1)
        connectivity = int(match.group(2))

        # InitializationNodeを生成
        # 注意: InitializationNodeはgenerate()で複数行を返す可能性があるが、
        # 部分プログラムでは1行のみを想定
        # skip_filter=Trueを設定して、部分プログラムに既に含まれている背景色フィルタと
        # 重複しないようにする
        init_node = InitializationNode(connectivity=connectivity, context=context.to_dict(), skip_filter=True)

        # 変数を定義済みとしてマーク
        context.variable_manager.define_variable(var_name, SemanticType.OBJECT, is_array=True)
        context.add_scope_variable(var_name)

        return init_node

    # 2. FILTERのパターン
    # 例: "objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), 0))"
    # 例: "objects1 = FILTER(objects, EQUAL(GET_COLOR($obj), 3))"
    match = re.match(r'(\w+)\s*=\s*FILTER\((\w+),\s*(.+)\)', line)
    if match:
        target_array = match.group(1)
        source_array = match.group(2)
        condition = match.group(3).strip()

        # FilterNodeを生成
        filter_node = FilterNode(
            source_array=source_array,
            target_array=target_array,
            condition=condition,
            context=context.to_dict()
        )

        # 変数を定義済みとしてマーク
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        context.add_scope_variable(target_array)

        # ソース配列の使用を記録
        context.variable_manager.register_variable_usage(source_array, "argument", OBJECT_ARRAY_TYPE)

        return filter_node

    # 3. EXCLUDEのパターン
    # 例: "objects2 = EXCLUDE(objects, objects1)"
    match = re.match(r'(\w+)\s*=\s*EXCLUDE\((\w+),\s*(\w+)\)', line)
    if match:
        target_array = match.group(1)
        source_array = match.group(2)
        targets_array = match.group(3)

        # ExcludeNodeを生成
        exclude_node = ExcludeNode(
            source_array=source_array,
            target_array=target_array,
            targets_array=targets_array,
            context=context.to_dict()
        )

        # ターゲット配列を定義済みとして登録
        context.variable_manager.define_variable(target_array, SemanticType.OBJECT, is_array=True)
        context.add_scope_variable(target_array)

        # ソース配列の使用を記録
        context.variable_manager.register_variable_usage(source_array, "argument", OBJECT_ARRAY_TYPE)

        # 除外対象配列の使用を記録
        context.variable_manager.register_variable_usage(targets_array, "argument", OBJECT_ARRAY_TYPE)

        return exclude_node

    # 4. MERGEのパターン
    # 例: "object = MERGE(objects)"
    match = re.match(r'(\w+)\s*=\s*MERGE\((\w+)\)', line)
    if match:
        target_obj = match.group(1)
        objects_array = match.group(2)

        # MergeNodeを生成
        merge_node = MergeNode(
            objects_array=objects_array,
            target_obj=target_obj,
            context=context.to_dict()
        )

        # ターゲットオブジェクトを定義済みとして登録（配列ではない）
        context.variable_manager.define_variable(target_obj, SemanticType.OBJECT, is_array=False)
        context.add_scope_variable(target_obj)

        # オブジェクト配列の使用を記録
        context.variable_manager.register_variable_usage(objects_array, "argument", OBJECT_ARRAY_TYPE)

        return merge_node

    # パースに失敗した場合はNoneを返す
    return None

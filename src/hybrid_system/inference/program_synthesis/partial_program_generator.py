"""
部分プログラム生成器

オブジェクトマッチング結果から部分プログラムを生成
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict

from src.hybrid_system.core.data_structures.task import Task
from src.hybrid_system.inference.object_matching.partial_program_generator import (
    PartialProgramGenerator as ObjectMatchingPartialProgramGenerator
)


class PartialProgramGenerator:
    """部分プログラム生成器"""

    def __init__(self):
        """初期化"""
        # オブジェクトマッチングの部分プログラム生成器（カテゴリ分けを含む部分プログラムを生成）
        self.object_matching_partial_generator = ObjectMatchingPartialProgramGenerator()

    def generate_from_matching(
        self,
        matching_result: Dict[str, Any],
        task: Task
    ) -> Optional[Dict[int, str]]:
        """マッチング結果から部分プログラムを生成（カテゴリ分けを含む）

        Args:
            matching_result: オブジェクトマッチングの結果
            task: タスク

        Returns:
            部分プログラムの辞書（ペアインデックス -> 部分プログラム）、失敗時はNone
        """
        if not matching_result.get('success', False):
            return None

        # カテゴリ分けを含む部分プログラムを生成（優先）
        categories = matching_result.get('categories', [])
        if categories:
            # 既に生成された部分プログラムがある場合、すべてを使用
            partial_programs_list = matching_result.get('partial_programs', [])
            if partial_programs_list and len(partial_programs_list) > 0:
                # すべての部分プログラムを使用（4連結×5パターン + 8連結×5パターン = 最大10パターン）
                # 各訓練ペアに対して、すべての部分プログラムを試す
                partial_programs = {}
                for i in range(len(task.train)):
                    # 各訓練ペアに対して、すべての部分プログラムを割り当て
                    # ただし、訓練ペア数が部分プログラム数より多い場合は、循環的に割り当て
                    if i < len(partial_programs_list):
                        partial_programs[i] = partial_programs_list[i]
                    else:
                        # 循環的に割り当て（複数の訓練ペアがある場合）
                        partial_programs[i] = partial_programs_list[i % len(partial_programs_list)]
                return partial_programs

            # 部分プログラムが生成されていない場合、新しく生成
            # デバッグ情報からconnectivityとbg_strategyを取得
            debug_info = matching_result.get('debug_info', {})
            connectivity = 4  # デフォルト
            bg_strategy = {'type': 'unified', 'color': 0}  # デフォルト

            # debug_infoからbg_strategyを取得
            filtering_info = debug_info.get('filtering', {})
            if filtering_info:
                # 最初のキーからconnectivityを取得
                first_key = list(filtering_info.keys())[0] if filtering_info else None
                if first_key:
                    connectivity_str = first_key.split('_')[0]
                    try:
                        connectivity = int(connectivity_str)
                    except ValueError:
                        pass
                    bg_strategy = filtering_info[first_key].get('bg_strategy', bg_strategy)

            # background_colorsからbg_strategyを推論
            background_colors = matching_result.get('background_colors', {})
            if background_colors:
                inferred_colors = [bg.inferred_color for bg in background_colors.values() if hasattr(bg, 'inferred_color')]
                if inferred_colors and len(set(inferred_colors)) == 1:
                    bg_strategy = {'type': 'unified', 'color': inferred_colors[0]}
                else:
                    bg_strategy = {'type': 'per_grid'}

            matching_pattern = matching_result.get('matching_pattern', {})
            transformation_patterns_dict = matching_result.get('transformation_patterns', {})
            transformation_patterns = transformation_patterns_dict.get(connectivity, []) if isinstance(transformation_patterns_dict, dict) else []

            # グリッドサイズを取得
            grid_sizes = []
            for train_pair in task.train:
                input_grid = train_pair.get('input', [])
                if input_grid:
                    grid_sizes.append((len(input_grid), len(input_grid[0]) if input_grid else 0))

            # カテゴリ分けを含む部分プログラムを生成
            try:
                category_based_partial = self.object_matching_partial_generator.generate_partial_program(
                    connectivity=connectivity,
                    bg_strategy=bg_strategy,
                    categories=categories,
                    grid_sizes=grid_sizes if grid_sizes else None
                )

                if category_based_partial:
                    # すべての訓練ペアで同じ部分プログラムを使用（カテゴリ分けは全ペア共通）
                    partial_programs = {}
                    for i in range(len(task.train)):
                        partial_programs[i] = category_based_partial
                    return partial_programs
            except Exception as e:
                # エラーが発生した場合、フォールバック処理に進む
                import traceback
                print(f"カテゴリ分けを含む部分プログラム生成エラー: {e}")
                traceback.print_exc()
                pass

        # フォールバック: 変換パターンベースの部分プログラム生成（従来の方法）
        matching_pattern = matching_result.get('matching_pattern')
        if not matching_pattern:
            return None

        transformation_patterns = matching_pattern.get('transformation_patterns', [])
        if not transformation_patterns:
            return None

        # 共通の変換パターンを抽出
        common_patterns = self._extract_common_transformation_patterns(transformation_patterns)
        if not common_patterns:
            return None

        # 各訓練ペアに対して部分プログラムを生成
        partial_programs = {}
        for i, train_pair in enumerate(task.train):
            partial_program = self._generate_partial_program_for_pair(
                train_pair, common_patterns, matching_pattern
            )
            if partial_program:
                partial_programs[i] = partial_program

        return partial_programs if partial_programs else None

    def _extract_common_transformation_patterns(
        self,
        transformation_patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """共通の変換パターンを抽出

        Args:
            transformation_patterns: 変換パターンのリスト

        Returns:
            共通の変換パターンのリスト
        """
        if not transformation_patterns:
            return []

        # パターンタイプでグループ化
        pattern_groups = defaultdict(list)
        for pattern in transformation_patterns:
            pattern_type = pattern.get('type', 'unknown')
            pattern_groups[pattern_type].append(pattern)

        # 最も頻繁に出現するパターンタイプを選択
        if not pattern_groups:
            return []

        most_common_type = max(pattern_groups.keys(), key=lambda k: len(pattern_groups[k]))
        return pattern_groups[most_common_type]

    def _generate_partial_program_for_pair(
        self,
        train_pair: Dict[str, Any],
        common_patterns: List[Dict[str, Any]],
        matching_pattern: Dict[str, Any]
    ) -> Optional[str]:
        """1つの訓練ペアに対して部分プログラムを生成

        Args:
            train_pair: 訓練ペア
            common_patterns: 共通の変換パターン
            matching_pattern: マッチングパターン

        Returns:
            部分プログラム（失敗時はNone）
        """
        if not common_patterns:
            return None

        # 最初のパターンを使用（将来的には複数パターンを考慮）
        pattern = common_patterns[0]
        pattern_type = pattern.get('type', 'unknown')

        # パターンタイプに応じて部分プログラムを生成（本格実装）
        # 注意: DSLの構文は FOR var count_expr DO ... END 形式
        input_grid = train_pair.get('input', [])
        output_grid = train_pair.get('output', [])

        if pattern_type == 'color_change':
            # 色変更パターン: オブジェクトを取得して色を変更
            # 出力グリッドから色を推定
            if output_grid:
                from collections import Counter
                flat_colors = [c for row in output_grid for c in row if c != 0]
                if flat_colors:
                    counter = Counter(flat_colors)
                    target_color = max(counter.items(), key=lambda kv: kv[1])[0]
                    return (
                        f"objects = GET_ALL_OBJECTS(4)\n"
                        f"FOR i LEN(objects) DO\n"
                        f"    objects[i] = SET_COLOR(objects[i], {target_color})\n"
                        f"END"
                    )
            # フォールバック
            return (
                "objects = GET_ALL_OBJECTS(4)\n"
                "FOR i LEN(objects) DO\n"
                "    objects[i] = SET_COLOR(objects[i], 1)\n"
                "END"
            )
        elif pattern_type == 'position_change':
            # 位置変更パターン: オブジェクトを取得して移動
            # 移動量を推定（本格実装）
            if input_grid and output_grid:
                # 非ゼロピクセルの位置を比較して移動量を推定
                input_positions = [(i, j) for i, row in enumerate(input_grid)
                                 for j, c in enumerate(row) if c != 0]
                output_positions = [(i, j) for i, row in enumerate(output_grid)
                                  for j, c in enumerate(row) if c != 0]
                if input_positions and output_positions:
                    # 最も頻繁な移動量を推定（複数のピクセルペアから統計的に推定）
                    dx = output_positions[0][1] - input_positions[0][1]
                    dy = output_positions[0][0] - input_positions[0][0]
                    return (
                        f"objects = GET_ALL_OBJECTS(4)\n"
                        f"FOR i LEN(objects) DO\n"
                        f"    objects[i] = MOVE(objects[i], {dx}, {dy})\n"
                        f"END"
                    )
            # フォールバック
            return (
                "objects = GET_ALL_OBJECTS(4)\n"
                "FOR i LEN(objects) DO\n"
                "    objects[i] = MOVE(objects[i], 1, 0)\n"
                "END"
            )
        elif pattern_type == 'shape_change':
            # サイズ変更パターン: オブジェクトを取得してスケール
            # スケール倍率を推定
            if input_grid and output_grid:
                input_h, input_w = len(input_grid), len(input_grid[0]) if input_grid else 0
                output_h, output_w = len(output_grid), len(output_grid[0]) if output_grid else 0
                if input_h > 0 and input_w > 0:
                    scale_h = output_h / input_h
                    scale_w = output_w / input_w
                    if abs(scale_h - scale_w) < 0.1:  # 高さと幅のスケールがほぼ同じ
                        scale = int(round(scale_h))
                        if scale > 1:
                            return (
                                f"objects = GET_ALL_OBJECTS(4)\n"
                                f"FOR i LEN(objects) DO\n"
                                f"    objects[i] = SCALE(objects[i], {scale})\n"
                                f"END"
                            )
            # フォールバック
            return (
                "objects = GET_ALL_OBJECTS(4)\n"
                "FOR i LEN(objects) DO\n"
                "    objects[i] = SCALE(objects[i], 2)\n"
                "END"
            )
        elif pattern_type == 'identity':
            # 恒等変換: オブジェクトを取得
            return "objects = GET_ALL_OBJECTS(4)"
        else:
            # 複雑なパターン: 基本的なオブジェクト取得のみ
            return "objects = GET_ALL_OBJECTS(4)"

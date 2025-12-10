"""
ルールベースオブジェクトマッチャー

設計書に基づいたメインクラス
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import numpy as np
import time
import traceback
import os
from copy import deepcopy

from src.core_systems.executor.core import ExecutorCore
from src.hybrid_system.core.data_structures.task import Task

from .config import ObjectMatchingConfig
from .data_structures import ObjectInfo, CategoryInfo, BackgroundColorInfo, get_object_signature
from .similarity_utils import get_pixels_set
from .object_extractor import ObjectExtractor
from .similarity_calculator import SimilarityCalculator
from .correspondence_detector import CorrespondenceDetector
from .background_color_inferencer import BackgroundColorInferencer
from .category_classifier import CategoryClassifier
from .partial_program_generator import PartialProgramGenerator

# プログラム生成フロー用の定数

from src.data_systems.generator.program_generator.metadata.constants import MAX_PARTIAL_PROGRAM_GENERATION_ATTEMPTS


# ログ出力制御（デフォルトで詳細ログを無効化）
ENABLE_VERBOSE_LOGGING = os.environ.get('ENABLE_VERBOSE_LOGGING', 'false').lower() in ('true', '1', 'yes')
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')
ENABLE_DEBUG_OUTPUT = os.environ.get('ENABLE_DEBUG_OUTPUT', 'false').lower() in ('true', '1', 'yes')


class RuleBasedObjectMatcher:
    """ルールベースオブジェクトマッチャー"""

    def __init__(self, config: Optional[ObjectMatchingConfig] = None):
        """
        初期化

        Args:
            config: オブジェクトマッチング設定
        """
        self.config = config or ObjectMatchingConfig()
        self.executor = ExecutorCore()

        # 各モジュールを初期化
        self.object_extractor = ObjectExtractor()
        self.similarity_calculator = SimilarityCalculator(self.config)
        self.correspondence_detector = CorrespondenceDetector(
            self.config, self.similarity_calculator
        )
        # 拡張背景色推論（色役割分類統合）を使用するか
        use_enhanced = getattr(self.config, 'use_enhanced_background_inference', False)
        self.background_color_inferencer = BackgroundColorInferencer(
            use_enhanced_inference=use_enhanced,
            color_role_classifier=None  # 必要に応じて後で設定可能
        )
        self.category_classifier = CategoryClassifier(
            self.config, self.similarity_calculator
        )
        self.partial_program_generator = PartialProgramGenerator(self.config)

    def _get_background_color_for_grid(
        self, bg_strategy: Dict[str, Any], grid_idx: int,
        all_input_grids: List, num_train_pairs: int
    ) -> int:
        """
        グリッドインデックスに基づいて背景色を取得

        Args:
            bg_strategy: 背景色戦略
            grid_idx: グリッドインデックス
            all_input_grids: すべての入力グリッド
            num_train_pairs: 訓練ペア数

        Returns:
            背景色
        """
        if bg_strategy['type'] == 'unified':
            return bg_strategy.get('color', 0)
        elif bg_strategy['type'] == 'per_grid':
            # per_grid戦略の場合、bg_strategy['colors']にはすべてのグリッド（訓練+テスト）の背景色が含まれている
            if 'colors' in bg_strategy and grid_idx in bg_strategy['colors']:
                return bg_strategy['colors'][grid_idx]
            else:
                # bg_strategy['colors']に含まれていない場合は予期しない状況
                # フォールバック: グリッドから個別に推論（安全性のため）
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"per_grid戦略でgrid_idx {grid_idx}がbg_strategy['colors']に含まれていません。"
                    f"グリッドから個別に推論します。bg_strategy['colors']のキー: {list(bg_strategy.get('colors', {}).keys())}"
                )
                if grid_idx < len(all_input_grids):
                    import numpy as np
                    grid = all_input_grids[grid_idx]
                    grid_np = np.array(grid)
                    bg_info = self.background_color_inferencer._infer_single_background_color(grid_np, grid_idx)
                    return bg_info.inferred_color
                else:
                    return 0
        else:
            return 0

    def _filter_objects_by_background_color(
        self, objects: List[ObjectInfo], bg_color: int
    ) -> List[ObjectInfo]:
        """
        背景色オブジェクトを除外

        Args:
            objects: オブジェクトのリスト
            bg_color: 背景色

        Returns:
            フィルタリング後のオブジェクトのリスト
        """
        return [obj for obj in objects if obj.color != bg_color]

    def _calculate_max_colors_per_grid(
        self, objects_data: Dict[str, List[List[ObjectInfo]]],
        bg_strategy: Dict[str, Any], task: Task, connectivity: int
    ) -> int:
        """
        背景色を除外した色数の最大値を計算

        Args:
            objects_data: オブジェクトデータ（連結性ごと）
            bg_strategy: 背景色戦略
            task: Taskオブジェクト
            connectivity: 連結性

        Returns:
            背景色を除外した色数の最大値
        """
        max_colors = 1
        input_objects_list = objects_data['input_grids']
        all_input_grids = task.get_all_inputs()
        num_train_pairs = len(task.train)

        for grid_idx, input_objects in enumerate(input_objects_list):
            if not input_objects:
                continue

            # 背景色を取得
            bg_color = self._get_background_color_for_grid(
                bg_strategy, grid_idx, all_input_grids, num_train_pairs
            )

            # 背景色を除外した色のバリエーションを取得
            unique_colors = set([
                obj.color for obj in input_objects
                if obj.color is not None and obj.color != bg_color
            ])

            num_colors = len(unique_colors) if unique_colors else 1
            max_colors = max(max_colors, num_colors)

        return max_colors

    def _get_num_patterns(self, connectivity: int) -> int:
        """
        連結性に基づいてパターン数を取得

        Args:
            connectivity: 連結性（4または8）

        Returns:
            パターン数
        """
        if connectivity == 4:
            return self.config.num_category_patterns_4
        elif connectivity == 8:
            return self.config.num_category_patterns_8
        else:
            # デフォルトとして4連結のパターン数を使用
            return self.config.num_category_patterns_4

    def _check_all_grids_have_zero(self, all_input_grids: List) -> bool:
        """
        すべてのグリッドに0が含まれるかチェック

        Args:
            all_input_grids: すべての入力グリッド

        Returns:
            すべてのグリッドに0が含まれる場合はTrue
        """
        if not all_input_grids:
            return False
        return all(any(0 in row for row in grid) for grid in all_input_grids)

    def _generate_pattern_key(self, connectivity: int, pattern_idx: int) -> str:
        """
        パターンキーを生成

        Args:
            connectivity: 連結性（4または8）
            pattern_idx: パターンインデックス

        Returns:
            パターンキー文字列
        """
        return f"{connectivity}_{pattern_idx}"

    def _select_valid_categories(self, categories: List[CategoryInfo]) -> List[CategoryInfo]:
        """
        有効なカテゴリを選択（少なくとも1つのグリッドにオブジェクトが存在するカテゴリのみ）

        Args:
            categories: カテゴリ情報のリスト

        Returns:
            有効なカテゴリのリスト
        """
        valid_categories = []
        for category in categories:
            if category.object_count_per_grid:
                # 少なくとも1つのグリッドにオブジェクトが存在する場合のみ有効
                if any(count > 0 for count in category.object_count_per_grid):
                    valid_categories.append(category)
            else:
                # object_count_per_gridが存在しない場合、カテゴリにオブジェクトが存在する場合は有効（互換性のため）
                if category.objects:
                    valid_categories.append(category)
        return valid_categories

    def match_objects(self, task: Task) -> Dict[str, Any]:
        """
        オブジェクトマッチングを実行

        Args:
            task: Taskオブジェクト

        Returns:
            マッチング結果
        """
        total_start_time = time.time()
        try:
            # タスク情報を一度だけ取得（複数回使用するため）
            all_input_grids = task.get_all_inputs()
            num_train_pairs = len(task.train)

            # ① オブジェクト抽出
            extract_start = time.time()
            objects_data = self._extract_all_objects(task)
            extract_time = time.time() - extract_start

            # デバッグ: オブジェクト抽出の結果を確認
            debug_info = {
                'object_extraction': {},
                'filtering': {},
                'category_classification': {}
            }
            for connectivity in self.config.connectivities:
                total_input = sum(len(objs) for objs in objects_data[connectivity]['input_grids'])
                total_output = sum(len(objs) for objs in objects_data[connectivity]['output_grids'])
                debug_info['object_extraction'][connectivity] = {
                    'total_input': total_input,
                    'total_output': total_output
                }

            # 連結性最適化のチェック
            optimization_info = self._check_connectivity_optimization(objects_data, task)
            debug_info['connectivity_optimization'] = optimization_info

            # ② 背景色推論
            bg_start = time.time()
            background_colors = self.background_color_inferencer.infer_background_colors(task)
            bg_time = time.time() - bg_start

            # ③ 背景色戦略の決定（全パターン共通）
            bg_strategy_start = time.time()
            bg_strategy = self._decide_background_color_strategy(background_colors, task)
            bg_strategy_time = time.time() - bg_strategy_start

            # ④ 対応関係分析
            pattern_start = time.time()
            # 条件②: すべての訓練ペアの入出力オブジェクトが完全一致の場合、4連結のみ実行して8連結に複製
            if optimization_info['skip_correspondence_8']:
                # 4連結のみ対応関係分析を実行
                transformation_patterns_4, pattern_analysis_details = self._analyze_transformation_patterns_single_connectivity(
                    task, objects_data, bg_strategy, 4
                )
                # 8連結に複製（オブジェクトの対応関係情報を8連結のオブジェクトに設定）
                transformation_patterns_8 = self._copy_transformation_patterns_to_connectivity(
                    transformation_patterns_4, objects_data, 4, 8
                )
                transformation_patterns = {4: transformation_patterns_4[4], 8: transformation_patterns_8}
                # タイミング情報は4連結のみのものを使用
            else:
                transformation_patterns, pattern_analysis_details = self._analyze_transformation_patterns(task, objects_data, bg_strategy)
            pattern_time = time.time() - pattern_start
            debug_info['pattern_analysis_details'] = pattern_analysis_details

            # グリッドサイズ情報を取得（位置分類に使用）
            grid_sizes = [(len(grid), len(grid[0]) if grid else 0) for grid in all_input_grids]

            # グリッドサイズと背景色を各訓練ペアごとに計算してキャッシュ
            # 候補生成で繰り返し計算するのを避けるため
            from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import get_grid_size, get_background_color
            grid_info_cache = {}  # {pair_idx: {'input_size': (h, w), 'output_size': (h, w), 'input_bg': int, 'output_bg': int}}
            for pair_idx, train_pair in enumerate(task.train):
                input_grid = train_pair.get('input', [])
                output_grid = train_pair.get('output', [])
                if input_grid and output_grid:
                    # グリッドサイズは既にgrid_sizesから取得可能だが、
                    # output_gridのサイズは別途計算が必要
                    input_h, input_w = grid_sizes[pair_idx] if pair_idx < len(grid_sizes) else (len(input_grid), len(input_grid[0]) if input_grid else 0)
                    output_h, output_w = get_grid_size(output_grid)
                    # 入力背景色はbg_strategyから取得（一貫性のため）
                    input_bg = self._get_background_color_for_grid(
                        bg_strategy, pair_idx, all_input_grids, num_train_pairs
                    )
                    # 出力背景色はグリッドから直接取得（出力グリッドの背景色はbg_strategyでは決定していない）
                    output_bg = get_background_color(output_grid)
                    grid_info_cache[pair_idx] = {
                        'input_size': (input_h, input_w),
                        'output_size': (output_h, output_w),
                        'input_bg': input_bg,
                        'output_bg': output_bg
                    }

            # 部分プログラムリスト（重複を除外するため、setで既存の部分プログラムを記録）
            partial_programs = []
            partial_programs_set = set()  # 重複チェック用
            all_categories = []  # すべてのカテゴリを集約
            # 部分プログラムとカテゴリ情報の対応関係を追跡
            # {partial_program: List[CategoryInfo]} の形式
            # 同じ部分プログラムが複数のパターンで生成された場合、すべてのカテゴリ情報を保持
            partial_program_to_categories: Dict[str, List] = {}
            # 部分プログラムとcategory_var_mappingの対応関係を追跡
            # {partial_program: Dict[str, str]} の形式（{category_id: variable_name}）
            # 同じ部分プログラムが複数のパターンで生成された場合、最初のマッピングを使用
            partial_program_to_category_var_mapping: Dict[str, Dict[str, str]] = {}
            all_category_var_mappings = {}  # カテゴリIDと変数名の対応関係 {pattern_key: {category_id: variable_name}}

            # ループ処理時間を計測
            loop_times = {
                'category_classification': 0.0,
                'partial_program_generation': 0.0,
                'total_loop': 0.0
            }

            # ループ1: 連結性ごと（逐次処理）
            # 条件①: すべての入力オブジェクトが完全一致の場合、8連結のパターン処理をスキップ
            # （8連結側は空のまま、複製しない）
            connectivities_to_process = self.config.connectivities
            if optimization_info['skip_pattern_loop_8'] and 8 in connectivities_to_process:
                # 4連結のみ処理（8連結はスキップ）
                connectivities_to_process = [c for c in connectivities_to_process if c != 8]

            for connectivity in connectivities_to_process:
                # 連結性ごとのパターン数を取得
                num_patterns = self._get_num_patterns(connectivity)

                # 逐次処理: 各パターンを順番に処理
                none_count = 0
                exception_count = 0
                success_count = 0
                empty_objects_count = 0

                # オブジェクト数を確認（変換パターン分析時には背景色オブジェクトも含まれる）
                # 一度だけ計算して再利用
                total_input_objects = sum(len(objs) for objs in objects_data[connectivity]['input_grids'])

                # タスク内の各入力グリッドの色数の最大値を計算（パターン処理の前に一度だけ計算）
                # カテゴリ分類用の色数計算のため、背景色を除外した色のバリエーションを取得
                # すべての色の一覧は、各入力グリッドから、背景色（グリッドごとの設定ならグリッドごとに、統一の設定なら統一）を除外した場合の色数
                max_colors_per_grid = self._calculate_max_colors_per_grid(
                    objects_data[connectivity], bg_strategy, task, connectivity
                )

                for pattern_idx in range(num_patterns):
                    try:
                        # デバッグ: objects_data[connectivity]の内容を確認（環境変数で制御）
                        if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                            import logging
                            import sys
                            logger = logging.getLogger(__name__)
                            if pattern_idx < 3:  # 最初の3個のみデバッグ出力
                                connectivity_objects = objects_data[connectivity]
                                debug_msg = (
                                    f"[DEBUG] Before _process_single_pattern: "
                                    f"Pattern Index {pattern_idx}, Connectivity {connectivity}, "
                                    f"type: {type(connectivity_objects)}, keys: {list(connectivity_objects.keys()) if isinstance(connectivity_objects, dict) else 'N/A'}, "
                                    f"input_grids length: {len(connectivity_objects.get('input_grids', [])) if isinstance(connectivity_objects, dict) else 0}, "
                                    f"total objects: {sum(len(objs) for objs in connectivity_objects.get('input_grids', [])) if isinstance(connectivity_objects, dict) else 0}"
                                )
                                logger.warning(debug_msg)
                                print(debug_msg, file=sys.stderr)

                        # 全パターン共通の背景色戦略とmax_colors_per_gridを使用
                        result = self._process_single_pattern(
                            connectivity, pattern_idx, objects_data[connectivity],
                            bg_strategy, transformation_patterns.get(connectivity, []),
                            grid_sizes, task, num_patterns, max_colors_per_grid,
                            is_debug_attempt=False  # 推論パイプラインではデバッグ出力を抑制
                        )

                        if result is None:
                            none_count += 1
                            # デバッグ: resultがNoneの場合の情報を記録
                            key = self._generate_pattern_key(connectivity, pattern_idx)
                            if key not in debug_info.get('filtering', {}):
                                # エラー情報がない場合のみ記録（例外が発生した場合は既に記録されている）
                                # オブジェクト数を記録（変換パターン分析時に既に背景色オブジェクトが除外されている）
                                # total_input_objectsは既に計算済みなので再利用
                                before_filter = total_input_objects

                                # 全パターン共通の背景色戦略を使用
                                try:
                                    # オブジェクト数を計算（デバッグ用、実際には既にフィルタリング済み）
                                    # 注意: objects_dataは既に変換パターン分析時にフィルタリングされているため、
                                    # この計算はデバッグ情報取得のためだけに使用される
                                    test_objects = self._filter_background_objects(
                                        objects_data[connectivity], bg_strategy, task
                                    )
                                    after_filter = len(test_objects)

                                    debug_info.setdefault('_process_single_pattern_debug', {})[key] = {
                                        'result': 'None (no error recorded)',
                                        'before_filter': before_filter,
                                        'after_filter': after_filter,
                                        'bg_strategy': bg_strategy,
                                        'note': 'This may be due to empty objects after filtering'
                                    }
                                except Exception as e:
                                    debug_info.setdefault('_process_single_pattern_debug', {})[key] = {
                                        'result': 'None (error getting filter info)',
                                        'before_filter': before_filter,
                                        'error': str(e)
                                    }

                                if before_filter == 0:
                                    empty_objects_count += 1
                            # 結果がNoneの場合でも、all_category_var_mappingsに空のマッピングを設定
                            # これにより、部分プログラムが生成されなかった場合でも、visualize_partial_program_variations.pyで正しく処理される
                            all_category_var_mappings[key] = {}
                            continue

                        success_count += 1

                        key, categories, partial_program, category_time, partial_time, category_timing, filter_info, category_var_mapping = result

                        # デバッグ情報を記録
                        debug_info['filtering'][key] = filter_info
                        # カテゴリ情報を詳細に記録（カテゴリ分類失敗の原因分析用）
                        category_debug_info = {
                            'input_objects': filter_info['after'],
                            'categories': len(categories),
                            'valid_categories': len([c for c in categories if any(count > 0 for count in (c.object_count_per_grid or [])) or (not c.object_count_per_grid and c.objects)]),
                            'timing': category_timing,
                            # 特徴量情報を追加（カテゴリ数=1の原因分析用）
                            'selected_main_features': category_timing.get('selected_main_features', []) if category_timing else [],
                            'selected_shape_details': category_timing.get('selected_shape_details', []) if category_timing else [],
                            'selected_shape_other_details': category_timing.get('selected_shape_other_details', []) if category_timing else [],
                            'selected_shape_feature': category_timing.get('selected_shape_feature', 'size') if category_timing else 'size',
                            'selected_position_feature': category_timing.get('selected_position_feature', 'center') if category_timing else 'center'
                        }
                        # カテゴリの詳細情報を追加
                        if categories:
                            category_debug_info['category_details'] = []
                            for cat in categories:
                                cat_detail = {
                                    'category_id': cat.category_id,
                                    'total_objects': cat.total_objects,
                                    'object_count_per_grid': cat.object_count_per_grid,
                                    'grids_with_objects': sum(1 for count in (cat.object_count_per_grid or []) if count > 0),
                                    'is_valid': any(count > 0 for count in (cat.object_count_per_grid or [])) or (not cat.object_count_per_grid and cat.objects)
                                }
                                category_debug_info['category_details'].append(cat_detail)
                        # timing情報からnum_categoriesを取得
                        if category_timing:
                            category_debug_info['num_categories'] = category_timing.get('num_categories', len(categories))

                            # 部分プログラム生成失敗のデバッグ情報を追加
                            partial_program_failure = category_timing.get('partial_program_generation_failure')
                            if partial_program_failure:
                                category_debug_info['partial_program_generation_failure'] = partial_program_failure

                        debug_info['category_classification'][key] = category_debug_info

                        # 時間を累積
                        loop_times['category_classification'] += category_time
                        loop_times['partial_program_generation'] += partial_time

                        # カテゴリを集約（重複を避けるため、カテゴリIDを調整）
                        # カテゴリIDと変数名の対応関係も保存
                        pattern_category_var_mapping = {}  # このパターンのカテゴリIDと変数名の対応関係

                        # 部分プログラムとカテゴリ情報の対応関係を保存するために、元のカテゴリ情報をコピー
                        # 注意: categoriesは_process_single_patternから返されたもので、元のカテゴリIDのまま
                        pattern_categories_for_mapping = []  # 対応関係保存用（元のカテゴリIDを保持）

                        for cat in categories:
                            # 元のカテゴリIDを保存
                            original_category_id = cat.category_id

                            # 対応関係保存用にコピー（元のカテゴリIDを保持）
                            cat_copy_for_mapping = deepcopy(cat)
                            pattern_categories_for_mapping.append(cat_copy_for_mapping)

                            # カテゴリIDを一意にする（connectivityとpattern_idxを含める）
                            cat.category_id = f"{connectivity}_{pattern_idx}_{cat.category_id}"
                            all_categories.append(cat)

                            # カテゴリIDと変数名の対応関係を保存（新しいカテゴリIDを使用）
                            if isinstance(original_category_id, int) and original_category_id in category_var_mapping:
                                pattern_category_var_mapping[cat.category_id] = category_var_mapping[original_category_id]

                        # このパターンのカテゴリIDと変数名の対応関係を保存
                        # カテゴリ数が0または1の場合でも、部分プログラムが生成される可能性があるため、
                        # category_var_mappingが空でもall_category_var_mappingsに設定する
                        all_category_var_mappings[key] = pattern_category_var_mapping

                        # カテゴリ分けが失敗した場合（Noneが返された場合）、部分プログラムを追加しない
                        if partial_program is not None:
                            # 部分プログラムとカテゴリ情報の対応関係を保存
                            # 注意: pattern_categories_for_mappingは元のカテゴリIDを保持している
                            # カテゴリIDを一意にする（connectivityとpattern_idxを含める）
                            for cat_copy in pattern_categories_for_mapping:
                                original_cat_id = cat_copy.category_id
                                cat_copy.category_id = f"{connectivity}_{pattern_idx}_{original_cat_id}"

                            # 対応関係を保存（同じ部分プログラムが複数のパターンで生成された場合、最初のパターンの情報のみを保存）
                            # 理由: 同じ部分プログラムは同じ変数名（objects1, objects2など）を使用するため、
                            #       カテゴリ分けとマッピングはすべてのパターンで一致しているはず
                            if partial_program not in partial_program_to_categories:
                                partial_program_to_categories[partial_program] = pattern_categories_for_mapping.copy()

                            # 部分プログラムとcategory_var_mappingの対応関係を保存
                            # 注意: pattern_category_var_mappingは既に一意化されたカテゴリID（"{connectivity}_{pattern_idx}_{original_id}"形式）を使用
                            # 同じ部分プログラムが複数のパターンで生成された場合、最初のパターンのマッピングのみを保存
                            # 理由: 同じ部分プログラムは同じ変数名を使用するため、マッピングはすべてのパターンで一致しているはず
                            if partial_program not in partial_program_to_category_var_mapping:
                                partial_program_to_category_var_mapping[partial_program] = pattern_category_var_mapping.copy()

                            # 重複チェック: 既に同じ部分プログラムが存在する場合は追加しない
                            if partial_program not in partial_programs_set:
                                partial_programs.append(partial_program)
                                partial_programs_set.add(partial_program)

                    except Exception as e:
                        # エラーが発生した場合はスキップ
                        exception_count += 1
                        key = self._generate_pattern_key(connectivity, pattern_idx)
                        error_traceback = traceback.format_exc()
                        debug_info['filtering'][key] = {'error': str(e), 'traceback': error_traceback}
                        debug_info['category_classification'][key] = {'error': str(e), 'traceback': error_traceback}
                        debug_info.setdefault('_process_single_pattern_debug', {})[key] = {
                            'result': 'Exception',
                            'error': str(e),
                            'traceback': error_traceback
                        }

                # デバッグ情報を記録
                debug_info.setdefault('_process_single_pattern_stats', {})[f"{connectivity}"] = {
                    'total_patterns': num_patterns,
                    'success_count': success_count,
                    'none_count': none_count,
                    'exception_count': exception_count,
                    'empty_objects_count': empty_objects_count,
                    'total_input_objects': total_input_objects
                }

                # 連結性ごとの処理が完了したら、メモリをクリア
                # gc.collect()を一時的に無効化（パフォーマンステスト用）
                # import gc
                # gc.collect()

            # 条件①: すべての入力オブジェクトが完全一致の場合、8連結のパターン処理はスキップされ、
            # 8連結側は空のまま（複製しない）

            # マージ用の部分プログラムを追加（重複を除外）
            merge_start = time.time()
            merge_partial_programs = self._generate_merge_partial_programs(bg_strategy)
            for merge_program in merge_partial_programs:
                # 重複チェック: 既に同じ部分プログラムが存在する場合は追加しない
                if merge_program not in partial_programs_set:
                    partial_programs.append(merge_program)
                    partial_programs_set.add(merge_program)
            merge_time = time.time() - merge_start

            total_time = time.time() - total_start_time
            loop_times['total_loop'] = total_time - extract_time - bg_time - bg_strategy_time - pattern_time - merge_time

            # デバッグ情報に時間情報を追加
            debug_info['timing'] = {
                'object_extraction': extract_time,
                'background_color_inference': bg_time,
                'background_strategy_decision': bg_strategy_time,
                'pattern_analysis': pattern_time,
                'category_classification': loop_times['category_classification'],
                'partial_program_generation': loop_times['partial_program_generation'],
                'merge_program_generation': merge_time,
                'total': total_time
            }

            # 変換パターン分析の詳細タイミング情報を追加
            if 'pattern_analysis_details' in debug_info:
                debug_info['timing']['pattern_analysis_details'] = debug_info['pattern_analysis_details']

            # 部分プログラムの解析結果をキャッシュ（候補生成で繰り返し解析するのを避けるため）
            from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import parse_partial_program
            partial_program_parsed_cache = {}  # {partial_program: parsed_result}
            for partial_prog in partial_programs:
                if partial_prog and partial_prog not in partial_program_parsed_cache:
                    partial_program_parsed_cache[partial_prog] = parse_partial_program(partial_prog)

            # 注意: all_category_var_mappingsはループ内で累積されている
            return {
                'success': True,
                'partial_programs': partial_programs,
                # all_partial_programsは削除（partial_programsと同じため不要）
                'categories': all_categories,  # すべてのカテゴリを返す
                'partial_program_to_categories': partial_program_to_categories,  # 部分プログラムとカテゴリ情報の対応関係 {partial_program: List[CategoryInfo]}
                'partial_program_to_category_var_mapping': partial_program_to_category_var_mapping,  # 部分プログラムとcategory_var_mappingの対応関係 {partial_program: {category_id: variable_name}}
                'background_colors': background_colors,
                'bg_strategy': bg_strategy,  # 背景色戦略を追加
                'transformation_patterns': transformation_patterns,
                'category_var_mappings': all_category_var_mappings,  # カテゴリIDと変数名の対応関係 {pattern_key: {category_id: variable_name}}
                'grid_info_cache': grid_info_cache,  # グリッドサイズと背景色のキャッシュ {pair_idx: {...}}
                'partial_program_parsed_cache': partial_program_parsed_cache,  # 部分プログラムの解析結果キャッシュ {partial_program: parsed_result}
                'objects_data': objects_data,  # オブジェクトデータ（背景色オブジェクトも含む、インデックス設定済み）
                'debug_info': debug_info  # デバッグ情報を追加
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'partial_programs': []
            }

    def _extract_all_objects(self, task: Task) -> Dict[int, Dict[str, List[List[ObjectInfo]]]]:
        """オブジェクト抽出"""
        return self.object_extractor.extract_all_objects(task)

    def _are_objects_identical(self, obj1: ObjectInfo, obj2: ObjectInfo) -> bool:
        """
        2つのオブジェクトが完全に一致するかチェック

        Args:
            obj1: オブジェクト1
            obj2: オブジェクト2

        Returns:
            完全一致する場合True
        """
        # ピクセル座標と色で比較（順序は無視）
        if obj1.color != obj2.color:
            return False
        if len(obj1.pixels) != len(obj2.pixels):
            return False
        # ピクセル座標のセットで比較
        pixels1_set = get_pixels_set(obj1)
        pixels2_set = get_pixels_set(obj2)
        return pixels1_set == pixels2_set

    def _are_object_lists_identical(
        self, objects1: List[ObjectInfo], objects2: List[ObjectInfo]
    ) -> bool:
        """
        2つのオブジェクトリストが完全に一致するかチェック（順序は無視）

        Args:
            objects1: オブジェクトリスト1
            objects2: オブジェクトリスト2

        Returns:
            完全一致する場合True
        """
        if len(objects1) != len(objects2):
            return False

        # オブジェクトをシグネチャ（ピクセルセットと色）でソート
        sigs1 = sorted([sig for obj in objects1 if (sig := get_object_signature(obj)) is not None])
        sigs2 = sorted([sig for obj in objects2 if (sig := get_object_signature(obj)) is not None])

        return sigs1 == sigs2

    def _check_connectivity_optimization(
        self, objects_data: Dict[int, Dict[str, List[List[ObjectInfo]]]], task: Task
    ) -> Dict[str, Any]:
        """
        連結性最適化のチェック

        Args:
            objects_data: 抽出されたオブジェクトデータ
            task: Taskオブジェクト

        Returns:
            最適化情報
            {
                'input_objects_identical': bool,  # すべての入力オブジェクトが完全一致
                'train_io_objects_identical': bool,  # すべての訓練ペアの入出力オブジェクトが完全一致
                'skip_pattern_loop_8': bool,  # 8連結のパターン処理をスキップ
                'skip_correspondence_8': bool  # 8連結の対応関係分析をスキップ（4連結の結果を複製）
            }
        """
        if len(self.config.connectivities) < 2 or 4 not in self.config.connectivities or 8 not in self.config.connectivities:
            return {
                'input_objects_identical': False,
                'train_io_objects_identical': False,
                'skip_pattern_loop_8': False,
                'skip_correspondence_8': False
            }

        # ②すべての訓練ペアの入出力オブジェクトが4連結と8連結で完全一致かチェック
        # （条件②は条件①を含むため、先に条件②をチェック）
        train_io_objects_identical = True
        train_pairs = task.train

        for pair_idx, train_pair in enumerate(train_pairs):
            # 入力オブジェクトのチェック
            if (pair_idx >= len(objects_data[4]['input_grids']) or
                pair_idx >= len(objects_data[8]['input_grids'])):
                train_io_objects_identical = False
                break

            input_objs_4 = objects_data[4]['input_grids'][pair_idx]
            input_objs_8 = objects_data[8]['input_grids'][pair_idx]
            if not self._are_object_lists_identical(input_objs_4, input_objs_8):
                train_io_objects_identical = False
                break

            # 出力オブジェクトのチェック
            if (pair_idx >= len(objects_data[4]['output_grids']) or
                pair_idx >= len(objects_data[8]['output_grids'])):
                train_io_objects_identical = False
                break

            output_objs_4 = objects_data[4]['output_grids'][pair_idx]
            output_objs_8 = objects_data[8]['output_grids'][pair_idx]
            if not self._are_object_lists_identical(output_objs_4, output_objs_8):
                train_io_objects_identical = False
                break

        # ①すべての入力オブジェクト（訓練＋テスト）が4連結と8連結で完全一致かチェック
        # （条件②がFalseの場合のみチェック、条件②がTrueなら条件①もTrue）
        if train_io_objects_identical:
            input_objects_identical = True
        else:
            all_input_grids = task.get_all_inputs()
            input_objects_identical = True

            for grid_idx in range(len(all_input_grids)):
                if grid_idx >= len(objects_data[4]['input_grids']) or grid_idx >= len(objects_data[8]['input_grids']):
                    input_objects_identical = False
                    break

                objs_4 = objects_data[4]['input_grids'][grid_idx]
                objs_8 = objects_data[8]['input_grids'][grid_idx]

                if not self._are_object_lists_identical(objs_4, objs_8):
                    input_objects_identical = False
                    break

        return {
            'input_objects_identical': input_objects_identical,
            'train_io_objects_identical': train_io_objects_identical,
            'skip_pattern_loop_8': input_objects_identical,  # 条件①: すべての入力オブジェクトが完全一致
            'skip_correspondence_8': train_io_objects_identical  # 条件②: すべての訓練ペアの入出力オブジェクトが完全一致
        }

    def _analyze_transformation_patterns(
        self, task: Task, objects_data: Dict[int, Dict[str, List[List[ObjectInfo]]]],
        bg_strategy: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        対応関係分析（訓練ペアのみ、すべての連結性について）

        Args:
            task: Taskオブジェクト
            objects_data: 抽出されたオブジェクトデータ（背景色オブジェクトも含む）
            bg_strategy: 背景色戦略（カテゴリ分類時の背景色除外に使用）

        Returns:
            対応関係情報

        注意:
            - 対応関係検出時には背景色オブジェクトも含めて検出する
            - カテゴリ分類時には背景色オブジェクトを除外する（_process_single_pattern内で処理）
        """
        step_timings = {
            'find_correspondences': 0.0,
            'detect_merges': 0.0,
            'detect_splits': 0.0
        }

        transformation_patterns = {}

        # 各連結性について分析（単一連結性版を呼び出し）
        for connectivity in self.config.connectivities:
            patterns, timings = self._analyze_transformation_patterns_single_connectivity(
                task, objects_data, bg_strategy, connectivity
            )
            transformation_patterns.update(patterns)
            # タイミング情報を累積
            step_timings['find_correspondences'] += timings['find_correspondences']
            step_timings['detect_merges'] += timings['detect_merges']
            step_timings['detect_splits'] += timings['detect_splits']

        return transformation_patterns, step_timings

    def _analyze_transformation_patterns_single_connectivity(
        self, task: Task, objects_data: Dict[int, Dict[str, List[List[ObjectInfo]]]],
        bg_strategy: Dict[str, Any], connectivity: int
    ) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        単一連結性のみに対応関係分析を実行

        Args:
            task: Taskオブジェクト
            objects_data: 抽出されたオブジェクトデータ（背景色オブジェクトも含む）
            bg_strategy: 背景色戦略
            connectivity: 処理する連結性（4または8）

        Returns:
            (対応関係情報辞書, タイミング情報)
        """
        step_timings = {
            'find_correspondences': 0.0,
            'detect_merges': 0.0,
            'detect_splits': 0.0
        }

        transformation_patterns = {}

        input_objects_list = objects_data[connectivity]['input_grids']
        output_objects_list = objects_data[connectivity]['output_grids']

        # インデックスを設定
        for grid_idx, input_objects in enumerate(input_objects_list):
            if grid_idx >= len(input_objects_list):
                continue
            for i, obj in enumerate(input_objects):
                obj.index = i

        for grid_idx, output_objects in enumerate(output_objects_list):
            if grid_idx >= len(output_objects_list):
                continue
            for i, obj in enumerate(output_objects):
                obj.index = i

        # 訓練ペアごとに分析
        train_pairs = task.train
        pair_patterns = []

        for pair_idx, train_pair in enumerate(train_pairs):
            if pair_idx >= len(input_objects_list) or pair_idx >= len(output_objects_list):
                continue

            input_objects = input_objects_list[pair_idx]
            output_objects = output_objects_list[pair_idx]

            input_grid = train_pair['input']
            grid_size = (len(input_grid[0]) if input_grid and input_grid[0] else 0,
                        len(input_grid) if input_grid else 0)

            # 対応関係分析（設定で無効化可能）
            if self.config.enable_correspondence_detection:
                # 1対1対応の検出
                step_start = time.time()
                correspondences = self.correspondence_detector.find_correspondences(
                    input_objects, output_objects, grid_size=grid_size
                )
                step_timings['find_correspondences'] += time.time() - step_start

                # 対応関係をオブジェクトに設定
                for corr in correspondences:
                    corr['input_obj'].correspondence_type = 'one_to_one'
                    corr['input_obj'].matched_output_object = corr['output_obj']

                # 統合の検出
                step_start = time.time()
                merges = self.correspondence_detector.detect_merges(
                    input_objects, output_objects, correspondences, grid_size=grid_size
                )
                step_timings['detect_merges'] += time.time() - step_start

                # 統合の入力オブジェクトに対応関係を設定
                for merge in merges:
                    output_obj = merge['output_obj']
                    original_input_obj = None
                    for corr in correspondences:
                        if corr['output_obj'].index == output_obj.index:
                            original_input_obj = corr['input_obj']
                            break

                    if original_input_obj:
                        if original_input_obj not in merge['input_objects']:
                            merge['input_objects'].append(original_input_obj)
                            merge['merge_count'] = len(merge['input_objects'])
                            merge['transformation_patterns'].append({
                                'input_obj': original_input_obj
                            })

                    for input_obj in merge['input_objects']:
                        input_obj.correspondence_type = 'many_to_one'
                        input_obj.matched_output_objects = [merge['output_obj']]

                # 分割の検出
                step_start = time.time()
                splits = self.correspondence_detector.detect_splits(
                    input_objects, output_objects, correspondences, grid_size=grid_size
                )
                step_timings['detect_splits'] += time.time() - step_start

                # 分割の入力オブジェクトに対応関係を設定
                for split in splits:
                    input_obj = split['input_obj']
                    original_output_obj = None
                    for corr in correspondences:
                        if corr['input_obj'].index == input_obj.index:
                            original_output_obj = corr['output_obj']
                            break

                    if original_output_obj:
                        if original_output_obj not in split['output_objects']:
                            split['output_objects'].append(original_output_obj)
                            split['split_count'] = len(split['output_objects'])
                            split['transformation_patterns'].append({
                                'output_obj': original_output_obj
                            })

                    input_obj.correspondence_type = 'one_to_many'
                    input_obj.matched_output_objects = split['output_objects']

                # 消失・新規出現の検出
                merge_input_indices = {input_obj.index for merge in merges for input_obj in merge['input_objects']}
                merge_output_indices = {merge['output_obj'].index for merge in merges}
                split_input_indices = {split['input_obj'].index for split in splits}
                split_output_indices = {out.index for split in splits for out in split['output_objects']}

                matched_input_indices = (
                    {corr['input_obj'].index for corr in correspondences} |
                    merge_input_indices |
                    split_input_indices
                )
                matched_output_indices = (
                    {corr['output_obj'].index for corr in correspondences} |
                    merge_output_indices |
                    split_output_indices
                )

                disappeared_objects = [
                    obj for obj in input_objects
                    if obj.index not in matched_input_indices
                ]
                for obj in disappeared_objects:
                    obj.correspondence_type = 'one_to_zero'

                appeared_objects = [
                    obj for obj in output_objects
                    if obj.index not in matched_output_indices
                ]
                for obj in appeared_objects:
                    obj.correspondence_type = 'zero_to_one'
            else:
                # 対応関係分析を無効化した場合、空のリストを設定
                correspondences = []
                merges = []
                splits = []
                disappeared_objects = []
                appeared_objects = []
                # 対応関係タイプは設定しない（Noneのまま）

            pair_patterns.append({
                'pair_idx': pair_idx,
                'correspondences': correspondences,
                'splits': splits,
                'merges': merges,
                'disappeared_objects': disappeared_objects,
                'appeared_objects': appeared_objects
            })

        transformation_patterns[connectivity] = pair_patterns

        return transformation_patterns, step_timings

    def _copy_transformation_patterns_to_connectivity(
        self, transformation_patterns_4: Dict[int, List[Dict[str, Any]]],
        objects_data: Dict[int, Dict[str, List[List[ObjectInfo]]]],
        src_connectivity: int, dst_connectivity: int
    ) -> List[Dict[str, Any]]:
        """
        対応関係情報を別の連結性に複製（オブジェクトの対応関係情報も設定）

        Args:
            transformation_patterns_4: 4連結の対応関係情報
            objects_data: オブジェクトデータ
            src_connectivity: 元の連結性
            dst_connectivity: 複製先の連結性

        Returns:
            複製された対応関係情報のリスト
        """
        if src_connectivity not in transformation_patterns_4:
            return []

        src_patterns = transformation_patterns_4[src_connectivity]
        dst_patterns = []

        # オブジェクトシグネチャからオブジェクトへのマッピングを作成
        src_input_signature_map = {}  # {signature: obj}
        src_output_signature_map = {}  # {signature: obj}
        dst_input_signature_map = {}  # {signature: obj}
        dst_output_signature_map = {}  # {signature: obj}

        for grid_idx, src_objs in enumerate(objects_data[src_connectivity]['input_grids']):
            for obj in src_objs:
                sig = get_object_signature(obj)
                src_input_signature_map[(grid_idx, sig)] = obj

        for grid_idx, src_objs in enumerate(objects_data[src_connectivity]['output_grids']):
            for obj in src_objs:
                sig = get_object_signature(obj)
                src_output_signature_map[(grid_idx, sig)] = obj

        for grid_idx, dst_objs in enumerate(objects_data[dst_connectivity]['input_grids']):
            for obj in dst_objs:
                sig = get_object_signature(obj)
                dst_input_signature_map[(grid_idx, sig)] = obj

        for grid_idx, dst_objs in enumerate(objects_data[dst_connectivity]['output_grids']):
            for obj in dst_objs:
                sig = get_object_signature(obj)
                dst_output_signature_map[(grid_idx, sig)] = obj

        # 各訓練ペアの対応関係情報を複製
        for src_pair_pattern in src_patterns:
            pair_idx = src_pair_pattern['pair_idx']
            dst_pair_pattern = {
                'pair_idx': pair_idx,
                'correspondences': [],
                'splits': [],
                'merges': [],
                'disappeared_objects': [],
                'appeared_objects': []
            }

            # 1対1対応を複製
            for corr in src_pair_pattern['correspondences']:
                src_input_obj = corr['input_obj']
                src_output_obj = corr['output_obj']
                src_input_sig = get_object_signature(src_input_obj)
                src_output_sig = get_object_signature(src_output_obj)

                dst_input_obj = dst_input_signature_map.get((pair_idx, src_input_sig))
                dst_output_obj = dst_output_signature_map.get((pair_idx, src_output_sig))

                if dst_input_obj and dst_output_obj:
                    dst_input_obj.correspondence_type = 'one_to_one'
                    dst_input_obj.matched_output_object = dst_output_obj
                    dst_pair_pattern['correspondences'].append({
                        'input_obj': dst_input_obj,
                        'output_obj': dst_output_obj
                    })

            # 統合を複製
            for merge in src_pair_pattern['merges']:
                src_output_obj = merge['output_obj']
                src_output_sig = get_object_signature(src_output_obj)
                dst_output_obj = dst_output_signature_map.get((pair_idx, src_output_sig))

                if dst_output_obj:
                    dst_input_objs = []
                    for src_input_obj in merge['input_objects']:
                        src_input_sig = get_object_signature(src_input_obj)
                        dst_input_obj = dst_input_signature_map.get((pair_idx, src_input_sig))
                        if dst_input_obj:
                            dst_input_obj.correspondence_type = 'many_to_one'
                            dst_input_obj.matched_output_objects = [dst_output_obj]
                            dst_input_objs.append(dst_input_obj)

                    if dst_input_objs:
                        dst_pair_pattern['merges'].append({
                            'output_obj': dst_output_obj,
                            'input_objects': dst_input_objs,
                            'merge_count': len(dst_input_objs),
                            'transformation_patterns': [{'input_obj': obj} for obj in dst_input_objs]
                        })

            # 分割を複製
            for split in src_pair_pattern['splits']:
                src_input_obj = split['input_obj']
                src_input_sig = get_object_signature(src_input_obj)
                dst_input_obj = dst_input_signature_map.get((pair_idx, src_input_sig))

                if dst_input_obj:
                    dst_output_objs = []
                    for src_output_obj in split['output_objects']:
                        src_output_sig = get_object_signature(src_output_obj)
                        dst_output_obj = dst_output_signature_map.get((pair_idx, src_output_sig))
                        if dst_output_obj:
                            dst_output_objs.append(dst_output_obj)

                    if dst_output_objs:
                        dst_input_obj.correspondence_type = 'one_to_many'
                        dst_input_obj.matched_output_objects = dst_output_objs
                        dst_pair_pattern['splits'].append({
                            'input_obj': dst_input_obj,
                            'output_objects': dst_output_objs,
                            'split_count': len(dst_output_objs),
                            'transformation_patterns': [{'output_obj': obj} for obj in dst_output_objs]
                        })

            # 消失オブジェクトを複製
            for src_obj in src_pair_pattern['disappeared_objects']:
                src_sig = get_object_signature(src_obj)
                dst_obj = dst_input_signature_map.get((pair_idx, src_sig))
                if dst_obj:
                    dst_obj.correspondence_type = 'one_to_zero'
                    dst_pair_pattern['disappeared_objects'].append(dst_obj)

            # 新規出現オブジェクトを複製
            for src_obj in src_pair_pattern['appeared_objects']:
                src_sig = get_object_signature(src_obj)
                dst_obj = dst_output_signature_map.get((pair_idx, src_sig))
                if dst_obj:
                    dst_obj.correspondence_type = 'zero_to_one'
                    dst_pair_pattern['appeared_objects'].append(dst_obj)

            dst_patterns.append(dst_pair_pattern)

        return dst_patterns

    def _decide_background_color_strategy(
        self, background_colors: Dict[int, BackgroundColorInfo],
        task: Task
    ) -> Dict[str, Any]:
        """
        背景色戦略の決定（全パターン共通）

        Args:
            background_colors: 推論された背景色情報
            task: Taskオブジェクト

        Returns:
            背景色戦略（全パターンで同じ戦略を使用）

        優先順位:
        1. 最優先: 推論された背景色が完全一致 → 統一背景色戦略（推論された背景色）
        2. 第2優先: すべてのグリッドに0が含まれる → 統一背景色戦略（背景色0）
        3. 第3優先: 最頻出の背景色が1つ。かつすべてのグリッドにその色が含まれる。かつ、その色のピクセル数の割合がグリッド間で変動しすぎていない → 統一背景色戦略（最頻出の背景色）
        4. 第4優先: 最頻出かつすべてのグリッドにその色が含まれる背景色が複数。かつ、その色のピクセル数の割合がグリッド間で変動しすぎていない → 統一背景色戦略（最初の色）
        5. それ以外: グリッドごと戦略
        """
        # すべての入力グリッドを取得
        all_input_grids = task.get_all_inputs()

        if not background_colors:
            # 背景色情報が存在しない場合、警告を出力
            import warnings
            warnings.warn(
                "背景色情報が存在しません。グリッドから直接背景色を推論します。",
                UserWarning,
                stacklevel=2
            )
            # 背景色情報が存在しない場合、グリッドから直接背景色を推論
            # すべてのグリッドに0が含まれるかチェック
            if all_input_grids:
                all_grids_have_zero = self._check_all_grids_have_zero(all_input_grids)
                if all_grids_have_zero:
                    # すべてのグリッドに0が含まれる場合、背景色0を使用（統一背景色戦略）
                    return {
                        'type': 'unified',
                        'color': 0
                    }
                else:
                    # すべてのグリッドに0が含まれない場合、グリッドごと戦略（デフォルト値0を使用）
                    return {
                        'type': 'per_grid',
                        'colors': {idx: 0 for idx in range(len(all_input_grids))}
                    }
            else:
                # 入力グリッドが存在しない場合、デフォルト値0を使用
                return {'type': 'unified', 'color': 0}

        # 推論された背景色のリスト
        inferred_bg_colors = [bg.inferred_color for bg in background_colors.values()]

        # 最優先: 推論された背景色が完全一致
        if len(set(inferred_bg_colors)) == 1:
            # 推論された背景色が完全一致している場合、統一背景色戦略を使用
            unified_bg_color = inferred_bg_colors[0]
            return {
                'type': 'unified',
                'color': unified_bg_color
            }

        # 第2優先: すべてのグリッドに0が含まれる
        all_grids_have_zero = self._check_all_grids_have_zero(all_input_grids)
        if all_grids_have_zero:
            # すべてのグリッドに0が含まれる場合、背景色0を使用（統一背景色戦略）
            return {
                'type': 'unified',
                'color': 0
            }

        # 第3優先・第4優先: 最頻出の背景色をチェック
        bg_color_counts = {}
        for bg in background_colors.values():
            bg_color_counts[bg.inferred_color] = bg_color_counts.get(bg.inferred_color, 0) + 1

        max_count = max(bg_color_counts.values())
        most_common_colors = [
            color for color, count in bg_color_counts.items()
            if count == max_count
        ]

        # すべてのグリッドにその色が含まれるかチェック
        valid_colors = []
        for color in most_common_colors:
            # すべてのグリッドにその色が含まれるかチェック
            all_grids_have_color = all(any(color in row for row in grid) for grid in all_input_grids)
            if all_grids_have_color:
                # その色のピクセル数の割合がグリッド間で変動しすぎていないかチェック
                if self.background_color_inferencer.check_color_ratio_consistency(all_input_grids, color):
                    valid_colors.append(color)

        if not valid_colors:
            # それ以外: グリッドごと戦略
            # すべてのグリッド（訓練+テスト）の背景色を含める
            per_grid_colors = {idx: bg.inferred_color for idx, bg in background_colors.items()}
            # すべての入力グリッドのインデックスが含まれていることを確認
            all_input_grids = task.get_all_inputs()
            if len(per_grid_colors) != len(all_input_grids):
                import warnings
                warnings.warn(
                    f"per_grid戦略: background_colorsの数({len(per_grid_colors)})と"
                    f"all_input_gridsの数({len(all_input_grids)})が一致しません。"
                    f"背景色が含まれていないグリッドにはデフォルト値0を使用します。",
                    UserWarning,
                    stacklevel=2
                )
                # 不足しているグリッドの背景色を補完（デフォルト値0）
                for idx in range(len(all_input_grids)):
                    if idx not in per_grid_colors:
                        per_grid_colors[idx] = 0

            return {
                'type': 'per_grid',
                'colors': per_grid_colors
            }

        # 第3優先: 最頻出の背景色が1つ
        if len(valid_colors) == 1:
            return {
                'type': 'unified',
                'color': valid_colors[0]
            }

        # 第4優先: 最頻出かつすべてのグリッドにその色が含まれる背景色が複数
        # 最初の色を選択
        return {
            'type': 'unified',
            'color': valid_colors[0]
        }

    def _process_single_pattern(
        self, connectivity: int, pattern_idx: int,
        objects_data: Dict[str, List[List[ObjectInfo]]],
        bg_strategy: Dict[str, Any],
        transformation_patterns: List[Dict[str, Any]],
        grid_sizes: List[Tuple[int, int]],
        task: Task,
        total_patterns: Optional[int],
        max_colors_per_grid: int,
        is_debug_attempt: bool = False
    ) -> Optional[Tuple[str, List[CategoryInfo], Optional[str], float, float, Optional[Dict[str, Any]], Dict[str, Any], Dict[int, str]]]:
        """
        単一パターンの処理を実行

        Args:
            connectivity: 連結性（4または8）
            pattern_idx: パターンインデックス
            objects_data: オブジェクトデータ（連結性ごと、背景色オブジェクトも含む）
            bg_strategy: 事前に決定された背景色戦略（カテゴリ分類時の背景色除外に使用）
            transformation_patterns: 変換パターン情報
            grid_sizes: グリッドサイズのリスト
            task: Taskオブジェクト

        Returns:
            (key, categories, partial_program, category_time, partial_time, category_timing, filter_info, category_var_mapping)
            または None（エラー時）
        """
        try:
            # デバッグ: メソッドが呼び出されたことを記録
            # この情報は呼び出し元で集計される
            # オブジェクトの取得（変換パターン分析時には背景色オブジェクトも含まれる）
            import logging
            import sys
            logger = logging.getLogger(__name__)

            # デバッグ: objects_dataの内容を確認（環境変数で制御）
            if ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS:
                debug_msg = (
                    f"[DEBUG] Pattern Index {pattern_idx} (Connectivity {connectivity}): "
                    f"objects_data type: {type(objects_data)}, keys: {list(objects_data.keys()) if isinstance(objects_data, dict) else 'N/A'}, "
                    f"has input_grids: {'input_grids' in objects_data if isinstance(objects_data, dict) else False}"
                )
                logger.warning(debug_msg)
                print(debug_msg, file=sys.stderr)

                if isinstance(objects_data, dict) and 'input_grids' in objects_data:
                    input_grids_len = len(objects_data['input_grids'])
                    debug_msg2 = (
                        f"[DEBUG] Pattern Index {pattern_idx} (Connectivity {connectivity}): "
                        f"input_grids length: {input_grids_len}, "
                        f"first few grid sizes: {[len(objs) for objs in objects_data['input_grids'][:3]]}"
                    )
                    logger.warning(debug_msg2)
                    print(debug_msg2, file=sys.stderr)

            total_before_filter = sum(len(objs) for objs in objects_data['input_grids']) if isinstance(objects_data, dict) and 'input_grids' in objects_data else 0

            # デバッグ: total_before_filterの計算結果を確認（環境変数で制御）
            # 最初の数回の試行でデバッグ出力（is_debug_attemptフラグで制御）
            should_debug = (ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS) and is_debug_attempt
            if should_debug:
                debug_msg3 = (
                    f"[DEBUG] Pattern Index {pattern_idx} (Connectivity {connectivity}): "
                    f"total_before_filter: {total_before_filter}, "
                    f"input_grids length: {len(objects_data.get('input_grids', []))}, "
                    f"each grid object count: {[len(objs) for objs in objects_data.get('input_grids', [])]}"
                )
                logger.warning(debug_msg3)
                print(debug_msg3, file=sys.stderr)

                # オブジェクトが0個の場合、グリッドの内容を確認
                if total_before_filter == 0 and task:
                    try:
                        all_input_grids = task.get_all_inputs()
                        if all_input_grids:
                            first_grid = np.array(all_input_grids[0])
                            unique_colors = np.unique(first_grid)
                            grid_shape = first_grid.shape
                            debug_msg4 = (
                                f"[DEBUG] Pattern Index {pattern_idx} (Connectivity {connectivity}): "
                                f"グリッド内容確認 - shape: {grid_shape}, "
                                f"unique_colors: {unique_colors.tolist()}, "
                                f"is_single_color: {len(unique_colors) == 1}, "
                                f"grid_summary: min={first_grid.min()}, max={first_grid.max()}, mean={first_grid.mean():.2f}"
                            )
                            logger.warning(debug_msg4)
                            print(debug_msg4, file=sys.stderr)
                    except Exception as e:
                        error_msg = f"[DEBUG] グリッド内容確認エラー: {e}"
                        logger.error(error_msg)
                        print(error_msg, file=sys.stderr)

            # カテゴリ分類時のみ背景色オブジェクトを除外
            # 対応関係検出時には背景色オブジェクトも含めるが、カテゴリ分類時には除外する
            # タスク情報は呼び出し元で既に取得済みのため、引数として渡すことを検討
            # 現時点では、メソッド内で取得する（後で最適化可能）
            filter_start = time.time()
            task_all_input_grids = task.get_all_inputs()
            task_num_train_pairs = len(task.get_train_inputs())
            objects1 = []

            for grid_idx, input_objects in enumerate(objects_data['input_grids']):
                # 背景色を取得
                bg_color = self._get_background_color_for_grid(
                    bg_strategy, grid_idx, task_all_input_grids, task_num_train_pairs
                )

                # 背景色オブジェクトを除外してカテゴリ分類用のオブジェクトリストに追加
                filtered_input_objects = self._filter_objects_by_background_color(
                    input_objects, bg_color
                )

                # デバッグ: フィルタリングの詳細を記録（環境変数で制御、最初の3個のグリッドのみ）
                if (ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS) and pattern_idx < 3 and grid_idx < 3:
                    obj_colors = [obj.color for obj in input_objects if hasattr(obj, 'color')]
                    filtered_colors = [obj.color for obj in filtered_input_objects if hasattr(obj, 'color')]
                    debug_msg_filter = (
                        f"[DEBUG] Pattern Index {pattern_idx} (Connectivity {connectivity}): "
                        f"グリッド {grid_idx}: before={len(input_objects)}, after={len(filtered_input_objects)}, "
                        f"bg_color={bg_color}, obj_colors={obj_colors[:10]}, filtered_colors={filtered_colors[:10]}"
                    )
                    logger.warning(debug_msg_filter)
                    print(debug_msg_filter, file=sys.stderr)

                objects1.extend(filtered_input_objects)

            total_after_filter = len(objects1)
            filter_time = time.time() - filter_start
            filter_time = time.time() - filter_start

            # デバッグ: total_after_filterの値を確認（環境変数で制御、最初の3個のパターンのみ）
            if (ENABLE_DEBUG_OUTPUT or ENABLE_ALL_LOGS) and pattern_idx < 3:
                debug_msg_after = (
                    f"[DEBUG] Pattern Index {pattern_idx} (Connectivity {connectivity}): "
                    f"total_after_filter: {total_after_filter}, total_before_filter: {total_before_filter}"
                )
                logger.warning(debug_msg_after)
                print(debug_msg_after, file=sys.stderr)

            # デバッグ: オブジェクト数が0の場合でも、オブジェクト取得までの部分プログラムを返す
            if total_after_filter == 0:
                # オブジェクトが0個の場合、カテゴリ分類を実行できない
                # ただし、オブジェクト取得までの部分プログラムは返す
                # デバッグログ: 問題が発生しているパターンの詳細を記録（環境変数で制御、最初の3個のパターンのみ）
                if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and pattern_idx < 3:
                    debug_msg = (
                        f"Pattern Index {pattern_idx} (Connectivity {connectivity}): "
                        f"オブジェクトが0個になりました。bg_strategy: {bg_strategy}, "
                        f"before: {total_before_filter}, after: {total_after_filter}"
                    )
                    logger.warning(debug_msg)
                    print(debug_msg, file=sys.stderr)

                    # 各グリッドの詳細をログ出力（最初の3個のパターンのみ）
                    for grid_idx, input_objects in enumerate(objects_data['input_grids']):
                        try:
                            bg_color = self._get_background_color_for_grid(
                                bg_strategy, grid_idx, task_all_input_grids, task_num_train_pairs
                            )
                            filtered = self._filter_objects_by_background_color(input_objects, bg_color)
                            colors = [obj.color for obj in input_objects if hasattr(obj, 'color')]
                            grid_msg = (
                                f"  グリッド {grid_idx}: before={len(input_objects)}, "
                                f"after={len(filtered)}, bg_color={bg_color}, "
                                f"colors={colors[:10]}"
                            )
                            logger.warning(grid_msg)
                            print(grid_msg, file=sys.stderr)
                        except Exception as e:
                            error_msg = f"  グリッド {grid_idx}: エラーが発生しました: {e}"
                            logger.error(error_msg)
                            print(error_msg, file=sys.stderr)

                key = self._generate_pattern_key(connectivity, pattern_idx)
                filter_info = {
                    'before': total_before_filter,
                    'after': total_after_filter,
                    'bg_strategy': bg_strategy,
                    'reason': 'no_objects_after_filtering'
                }
                # オブジェクトが0個の場合でも、部分プログラム生成器を使用して背景色フィルタリングまでの部分プログラムを生成
                partial_program, category_var_mapping, retry_flag = self.partial_program_generator.generate_partial_program(
                    connectivity, bg_strategy, grid_sizes, valid_categories=[], pattern_idx=pattern_idx
                )

                # カテゴリとタイミング情報は空のまま返す
                categories = []
                category_time = 0.0
                partial_time = 0.0
                category_timing = None

                return (key, categories, partial_program, category_time, partial_time, category_timing, filter_info, category_var_mapping)

            # デバッグ情報
            key = self._generate_pattern_key(connectivity, pattern_idx)
            filter_info = {
                'before': total_before_filter,
                'after': total_after_filter,
                'bg_strategy': bg_strategy,
                'filter_time': filter_time
            }

            # ⑥ カテゴリ分けと⑦ 部分プログラム生成
            # 有効なカテゴリが1以下の場合、またはFILTER条件生成に失敗した場合、カテゴリ分けの再試行を行う
            # タスク情報を再利用（既に取得済み）
            num_input_grids = len(task_all_input_grids)
            # max_colors_per_gridは呼び出し元で計算済み

            # 対応関係情報は既に④でオブジェクトに設定されているため、
            # ここでは設定処理は不要（同じオブジェクトインスタンスが使用されている）
            # グリッドサイズ情報を渡す（位置分類に使用）
            retry_count = 0
            max_retry_count = self.config.max_category_retry_count
            category_time = 0.0
            partial_time = 0.0
            categories = None
            partial_program = None
            category_var_mapping = {}
            category_timing = None
            last_pattern_idx = pattern_idx

            # ループ処理全体の時間を計測
            loop_start = time.time()

            # ループ内の各処理時間を集計するための変数
            loop_timing_breakdown = {
                'categorize_objects': 0.0,
                'select_valid_categories': 0.0,
                'get_timing_info': 0.0,
                'generate_partial_program': 0.0,
                'other_loop_operations': 0.0
            }

            # すべての再試行のtiming_infoを保存（各再試行のpattern_idxに対応）
            all_retry_timing_infos = []

            while True:
                # カテゴリ分けの実行
                category_start = time.time()
                # 再試行時はpattern_idxを固定し、重みだけを変える
                # pattern_idxは変えずに、retry_countを渡して重みを変える
                categories = self.category_classifier.categorize_objects(
                    objects1, transformation_patterns, pattern_idx, num_input_grids, max_colors_per_grid, total_patterns, retry_count=retry_count
                )
                category_time += time.time() - category_start
                loop_timing_breakdown['categorize_objects'] += time.time() - category_start

                # 有効なカテゴリの選択（すべてのグリッドで0のカテゴリを除外）
                select_start = time.time()
                valid_categories = self._select_valid_categories(categories)
                select_time = time.time() - select_start
                loop_timing_breakdown['select_valid_categories'] += select_time

                # カテゴリ分けの詳細タイミング情報を取得
                timing_info_start = time.time()
                last_pattern_idx = pattern_idx  # pattern_idxは固定
                category_timing = self.category_classifier.get_last_timing_info(last_pattern_idx)
                loop_timing_breakdown['get_timing_info'] += time.time() - timing_info_start

                # すべての再試行のtiming_infoを保存（各再試行のpattern_idxに対応）
                if category_timing:
                    all_retry_timing_infos.append({
                        'pattern_idx': pattern_idx,  # pattern_idxは固定
                        'retry_count': retry_count,
                        'timing_info': category_timing.copy()
                    })

                # タイミング情報に追加の処理時間を記録
                if category_timing:
                    category_timing['filter_time'] = filter_time
                    category_timing['select_valid_categories_time'] = select_time

                # 有効なカテゴリが1以下の場合、またはFILTER条件生成失敗の場合、カテゴリ分けの再試行
                # 再試行が必要かどうかを判定
                should_retry = (len(valid_categories) <= 1)

                # 有効なカテゴリが2以上の場合、⑦ 部分プログラム生成
                if not should_retry:
                    # 注意: valid_categoriesは既に選択済みなので、部分プログラム生成に渡す
                    # 変数名には実際のpattern_idxを使用（current_pattern_idxではなく）
                    partial_start = time.time()
                    partial_program, category_var_mapping, retry_flag = self.partial_program_generator.generate_partial_program(
                        connectivity, bg_strategy, grid_sizes, valid_categories, pattern_idx=pattern_idx
                    )
                    partial_time += time.time() - partial_start
                    loop_timing_breakdown['generate_partial_program'] += time.time() - partial_start

                    # FILTER条件の生成に失敗した場合も再試行
                    if retry_flag == 'RETRY_CATEGORY':
                        should_retry = True
                        # 部分プログラム生成失敗のデバッグ情報を取得
                        failure_debug_info = getattr(self.partial_program_generator, 'last_failure_debug_info', [])
                        if failure_debug_info:
                            # category_timingに失敗情報を追加（既存の情報に追加）
                            if category_timing is None:
                                category_timing = {}
                            category_timing['partial_program_generation_failure'] = {
                                'failure_debug_info': failure_debug_info,
                                'num_failed_categories': len(failure_debug_info)
                            }

                # 再試行が必要な場合
                if should_retry:
                    # カテゴリ分けの再試行回数が上限に達した場合は終了
                    if retry_count >= max_retry_count:
                        break

                    retry_count += 1
                    # カテゴリ分けの再試行に戻る（ループの先頭に戻る）
                    continue

                # FILTER条件の生成に成功した、または他の理由で失敗した場合、ループを終了
                break

            loop_time = time.time() - loop_start

            # その他のループ処理時間を計算（ループ全体から測定済み時間を差し引く）
            measured_loop_time = (
                loop_timing_breakdown['categorize_objects'] +
                loop_timing_breakdown['select_valid_categories'] +
                loop_timing_breakdown['get_timing_info'] +
                loop_timing_breakdown['generate_partial_program']
            )
            loop_timing_breakdown['other_loop_operations'] = loop_time - measured_loop_time

            # タイミング情報にループ処理時間を記録
            if category_timing:
                category_timing['loop_time'] = loop_time
                category_timing['retry_count'] = retry_count
                category_timing['loop_timing_breakdown'] = loop_timing_breakdown
                # すべての再試行のtiming_infoを保存（集計用）
                category_timing['all_retry_timing_infos'] = all_retry_timing_infos

            # 部分プログラムが生成されていない場合（再試行が上限に達した場合など）、
            # 部分プログラム生成器を使用して背景色フィルタリングまでの部分プログラムを生成
            if partial_program is None:
                partial_program, category_var_mapping, retry_flag = self.partial_program_generator.generate_partial_program(
                    connectivity, bg_strategy, grid_sizes, valid_categories=[], pattern_idx=pattern_idx
                )
                # 再試行上限に達した場合、最後の試行のカテゴリ情報を保持（デバッグ用）
                # categoriesは最後の試行の結果を保持（有効でないカテゴリも含む）
                # category_timingには最後の試行の詳細情報が含まれる
                # 注意: categoriesがNoneの場合は空リストに設定
                if categories is None:
                    categories = []
                # 注意: 再試行上限に達した場合、valid_categoriesが1個以下なので、
                # categoriesにはすべてのカテゴリが含まれているが、有効なカテゴリは1個以下

                # 部分プログラム生成失敗のデバッグ情報を取得
                failure_debug_info = getattr(self.partial_program_generator, 'last_failure_debug_info', [])
                if failure_debug_info:
                    # category_timingに失敗情報を追加（既存の情報に追加）
                    if category_timing is None:
                        category_timing = {}
                    category_timing['partial_program_generation_failure'] = {
                        'failure_debug_info': failure_debug_info,
                        'num_failed_categories': len(failure_debug_info)
                    }

            return (key, categories, partial_program, category_time, partial_time, category_timing, filter_info, category_var_mapping)

        except Exception as e:
            # エラーが発生した場合はNoneを返す
            # 例外情報は呼び出し元で記録される
            # ここでは例外を再発生させて、呼び出し元でキャッチできるようにする
            raise
        # メモリクリアは呼び出し元で一括実行するため、ここでは削除

    def _filter_background_objects(
        self, objects_data: Dict[str, List[List[ObjectInfo]]],
        bg_strategy: Dict[str, Any], task: Optional[Task] = None
    ) -> List[ObjectInfo]:
        """
        背景色オブジェクトを除外

        Args:
            objects_data: オブジェクトデータ
            bg_strategy: 背景色戦略
            task: Taskオブジェクト（テスト入力の背景色推論に使用、Noneの場合は推論しない）

        Returns:
            フィルタリング後のオブジェクトリスト
        """
        all_objects = []
        input_objects_list = objects_data['input_grids']
        all_input_grids = task.get_all_inputs() if task else []
        num_train_pairs = len(task.train) if task else 0

        for grid_idx, input_objects in enumerate(input_objects_list):
            # 背景色を取得
            bg_color = self._get_background_color_for_grid(
                bg_strategy, grid_idx, all_input_grids, num_train_pairs
            )

            # 背景色のオブジェクトを除外
            filtered = self._filter_objects_by_background_color(input_objects, bg_color)
            all_objects.extend(filtered)

        return all_objects

    def _generate_merge_partial_programs(
        self, bg_strategy: Dict[str, Any]
    ) -> List[str]:
        """
        マージ用の部分プログラムを生成

        Args:
            bg_strategy: 背景色戦略

        Returns:
            マージ用の部分プログラムリスト
        """
        merge_programs = []
        num_merge_programs = self.config.num_merge_partial_programs

        # 生成可能なマージプログラムのテンプレート
        merge_templates = [
            # 1. 4連結のオブジェクトをマージ（背景色フィルタなし）
            "objects = GET_ALL_OBJECTS(4)\nobject = MERGE(objects)",
            # 2. 8連結のオブジェクトをマージ（背景色フィルタなし）
            "objects = GET_ALL_OBJECTS(8)\nobject = MERGE(objects)",
            # 3. 4連結のオブジェクトをマージ（背景色フィルタあり）
            None,  # 動的に生成
            # 4. 8連結のオブジェクトをマージ（背景色フィルタあり）
            None,  # 動的に生成
        ]

        # 背景色を取得（bg_strategyから取得、最初のグリッド（pair_idx=0）の背景色を使用）
        bg_color = 0
        if bg_strategy:
            if bg_strategy['type'] == 'unified':
                bg_color = bg_strategy['color']
            else:
                bg_color = bg_strategy['colors'].get(0, 0)

        # 動的に生成するテンプレートを埋める
        merge_templates[2] = (
            f"objects = GET_ALL_OBJECTS(4)\n"
            f"objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), {bg_color}))\n"
            f"object = MERGE(objects)"
        )
        merge_templates[3] = (
            f"objects = GET_ALL_OBJECTS(8)\n"
            f"objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), {bg_color}))\n"
            f"object = MERGE(objects)"
        )

        # 指定された数だけ生成（テンプレートを循環的に使用）
        for i in range(num_merge_programs):
            template_idx = i % len(merge_templates)
            merge_programs.append(merge_templates[template_idx])

        return merge_programs

    def match_objects_single_pattern(
        self,
        task: Task,
        connectivity: int = 4,
        max_attempts: int = None,
        original_background_color: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        プログラム生成フロー専用：1つのパターンだけを試行して部分プログラムを生成

        Args:
            task: Taskオブジェクト
            connectivity: 連結性（4または8、デフォルトは4）
            max_attempts: 最大試行回数（最初のパターンが失敗した場合、次のパターンを試行）
                          Noneの場合はconstants.pyのMAX_PARTIAL_PROGRAM_GENERATION_ATTEMPTSを使用
            original_background_color: データセット生成時限定：入力グリッド生成時に決定された背景色
                          Noneの場合: 推論パイプライン（従来通り）
                          指定された場合: 推論背景色と一致する場合はGET_BACKGROUND_COLOR()を使用、
                          不一致の場合はリテラル値を使用

        Returns:
            オブジェクトマッチング結果（部分プログラムが1つのみ）
        """
        # max_attemptsが指定されていない場合は定数から取得
        if max_attempts is None:
            max_attempts = MAX_PARTIAL_PROGRAM_GENERATION_ATTEMPTS
        total_start_time = time.time()
        try:
            # タスク情報を一度だけ取得
            all_input_grids = task.get_all_inputs()
            num_train_pairs = len(task.train)

            # ① オブジェクト抽出
            extract_start = time.time()
            objects_data = self._extract_all_objects(task)
            extract_time = time.time() - extract_start

            # 連結性が存在しない場合はエラー
            if connectivity not in objects_data:
                return {
                    'success': False,
                    'error': f'Connectivity {connectivity} not found in objects_data'
                }

            # ② 背景色推論
            bg_start = time.time()
            background_colors = self.background_color_inferencer.infer_background_colors(task)
            bg_time = time.time() - bg_start

            # ③ 背景色戦略の決定
            bg_strategy_start = time.time()
            bg_strategy = self._decide_background_color_strategy(background_colors, task)

            # データセット生成時限定：入力グリッド生成時の背景色と推論背景色を比較
            # 一致する場合はGET_BACKGROUND_COLOR()を使用、不一致の場合はリテラル値を使用
            if original_background_color is not None:
                # 推論された背景色を取得（データセット生成では入力グリッドが1つのみ）
                inferred_bg_color = None
                if background_colors and 0 in background_colors:
                    inferred_bg_color = background_colors[0].inferred_color

                # 背景色が一致する場合、GET_BACKGROUND_COLOR()を使用するためにper_grid戦略に変更
                if inferred_bg_color is not None and inferred_bg_color == original_background_color:
                    bg_strategy['type'] = 'per_grid'
                    # per_grid戦略の場合、colors辞書が必要（実際には1つのグリッドのみ）
                    bg_strategy['colors'] = {0: inferred_bg_color}
                    import sys
                    print(f"[BACKGROUND_COLOR] データセット生成: 背景色一致 (値={inferred_bg_color})、GET_BACKGROUND_COLOR()を使用", file=sys.stderr, flush=True)
                else:
                    # 不一致の場合はリテラル値を使用（unified戦略のまま）
                    bg_strategy['type'] = 'unified'
                    # 推論背景色が存在する場合はそれを使用、存在しない場合は元の背景色を使用
                    if inferred_bg_color is not None:
                        bg_strategy['color'] = inferred_bg_color
                    else:
                        bg_strategy['color'] = original_background_color
                    import sys
                    print(f"[BACKGROUND_COLOR] データセット生成: 背景色不一致 (元={original_background_color}, 推論={inferred_bg_color})、リテラル値を使用", file=sys.stderr, flush=True)

            bg_strategy_time = time.time() - bg_strategy_start

            # ③-1 データセット生成時限定：低確率でMERGEのみの部分プログラムを生成
            # 推論パイプラインとの一貫性を保つため、カテゴリ分けを試行する前にMERGEのみのパターンも選択
            if original_background_color is not None:  # データセット生成時のみ
                from src.data_systems.generator.program_generator.metadata.constants import MERGE_ONLY_PATTERN_PROBABILITY
                import random

                if random.random() < MERGE_ONLY_PATTERN_PROBABILITY:
                    # MERGEのみの部分プログラムを生成
                    merge_programs = self._generate_merge_partial_programs(bg_strategy)
                    if merge_programs:
                        # ランダムに1つ選択（または最初の1つを使用）
                        merge_program = random.choice(merge_programs) if len(merge_programs) > 1 else merge_programs[0]

                        # グリッドサイズと背景色を各訓練ペアごとに計算してキャッシュ
                        from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import get_grid_size, get_background_color
                        grid_info_cache = {}
                        for pair_idx, train_pair in enumerate(task.train):
                            input_grid = train_pair.get('input', [])
                            output_grid = train_pair.get('output', [])
                            if input_grid and output_grid:
                                input_h, input_w = grid_sizes[pair_idx] if pair_idx < len(grid_sizes) else (len(input_grid), len(input_grid[0]) if input_grid else 0)
                                output_h, output_w = get_grid_size(output_grid)
                                input_bg = get_background_color(input_grid)
                                output_bg = get_background_color(output_grid)
                                grid_info_cache[pair_idx] = {
                                    'input_size': (input_h, input_w),
                                    'output_size': (output_h, output_w),
                                    'input_bg': input_bg,
                                    'output_bg': output_bg
                                }

                        total_time = time.time() - total_start_time
                        import sys
                        print(f"[PROGRESS] 部分プログラム生成成功: MERGEのみパターン（低確率選択、確率={MERGE_ONLY_PATTERN_PROBABILITY*100:.1f}%）", file=sys.stderr, flush=True)

                        return {
                            'success': True,
                            'partial_programs': [merge_program],
                            'category_var_mappings': {},
                            'partial_program_to_category_var_mapping': {merge_program: {}},
                            'partial_program_to_categories': {merge_program: []},
                            'background_colors': background_colors,
                            'bg_strategy': bg_strategy,
                            'transformation_patterns': transformation_patterns,
                            'grid_info_cache': grid_info_cache,
                            'objects_data': objects_data,
                            'debug_info': {
                                'timing': {
                                    'object_extraction': extract_time,
                                    'background_color_inference': bg_time,
                                    'background_strategy_decision': bg_strategy_time,
                                    'pattern_analysis': 0.0,
                                    'loop_total': 0.0,
                                    'loop_attempts': 0,
                                    'loop_avg_per_attempt': 0.0,
                                    'loop_max_attempt': 0.0,
                                    'loop_min_attempt': 0.0,
                                    'total_category_time': 0.0,
                                    'total_partial_time': 0.0,
                                    'total': total_time
                                },
                                'merge_only_pattern': True
                            }
                        }

            # ④ 対応関係分析（簡略化版、必要に応じて）
            pattern_start = time.time()
            transformation_patterns, pattern_analysis_details = self._analyze_transformation_patterns(
                task, objects_data, bg_strategy
            )
            pattern_time = time.time() - pattern_start

            # ⑤ グリッドサイズ情報を取得
            grid_sizes = [(len(grid), len(grid[0]) if grid else 0) for grid in all_input_grids]

            # ⑥ 色数の最大値を計算
            max_colors_per_grid = self._calculate_max_colors_per_grid(
                objects_data[connectivity], bg_strategy, task, connectivity
            )

            # 【最適化1】早期成功判定: オブジェクト数を確認
            # フィルタリング前のオブジェクト数を計算（最初の入力グリッドのみ）
            input_grids = objects_data[connectivity].get('input_grids', [])
            objects_before_filter = 0
            if input_grids and len(input_grids) > 0:
                objects_before_filter = len(input_grids[0])

            # オブジェクト数が1個以下の場合: 背景色フィルタリングのみの部分プログラムを生成して成功として返す
            if objects_before_filter <= 1:
                total_time = time.time() - total_start_time
                import sys
                print(f"[PROGRESS] 部分プログラム生成を開始: 最大{max_attempts}回試行", file=sys.stderr, flush=True)

                # 背景色フィルタリングのみの部分プログラムを生成
                bg_only_partial_program = f"objects = GET_ALL_OBJECTS({connectivity})"
                if self.config.enable_background_filtering_in_partial_program:
                    if bg_strategy.get('type') == 'unified':
                        bg_color = bg_strategy.get('color', 0)
                        bg_only_partial_program += f"\nobjects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), {bg_color}))"
                    elif bg_strategy.get('type') == 'per_grid':
                        bg_only_partial_program += "\nobjects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))"

                print(f"[PROGRESS] 部分プログラム生成成功: 早期成功（オブジェクト数が1個以下: {objects_before_filter}個、背景色フィルタリングのみ）", file=sys.stderr, flush=True)

                # グリッドサイズと背景色を各訓練ペアごとに計算してキャッシュ
                from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import get_grid_size, get_background_color
                grid_info_cache = {}
                for pair_idx, train_pair in enumerate(task.train):
                    input_grid = train_pair.get('input', [])
                    output_grid = train_pair.get('output', [])
                    if input_grid and output_grid:
                        # grid_sizesは後で定義されるため、ここでは計算
                        input_h, input_w = len(input_grid), len(input_grid[0]) if input_grid else 0
                        output_h, output_w = get_grid_size(output_grid)
                        input_bg = get_background_color(input_grid)
                        output_bg = get_background_color(output_grid)
                        grid_info_cache[pair_idx] = {
                            'input_size': (input_h, input_w),
                            'output_size': (output_h, output_w),
                            'input_bg': input_bg,
                            'output_bg': output_bg
                        }

                return {
                    'success': True,
                    'partial_programs': [bg_only_partial_program],
                    'category_var_mappings': {},
                    'partial_program_to_category_var_mapping': {bg_only_partial_program: {}},
                    'partial_program_to_categories': {bg_only_partial_program: []},
                    'background_colors': background_colors,
                    'bg_strategy': bg_strategy,
                    'transformation_patterns': transformation_patterns,
                    'grid_info_cache': grid_info_cache,
                    'objects_data': objects_data,
                    'debug_info': {
                        'timing': {
                            'object_extraction': extract_time,
                            'background_color_inference': bg_time,
                            'background_strategy_decision': bg_strategy_time,
                            'pattern_analysis': pattern_time,
                            'loop_total': 0.0,
                            'loop_attempts': 0,
                            'loop_avg_per_attempt': 0.0,
                            'loop_max_attempt': 0.0,
                            'loop_min_attempt': 0.0,
                            'total_category_time': 0.0,
                            'total_partial_time': 0.0,
                            'total': total_time
                        },
                        'early_success_reason': 'insufficient_objects_for_category',
                        'objects_before_filter': objects_before_filter
                    }
                }

            # 【最適化2】背景色戦略がunknownの場合も早期失敗検出
            bg_strategy_type = bg_strategy.get('type', 'unknown')
            if bg_strategy_type == 'unknown':
                total_time = time.time() - total_start_time
                import sys
                print(f"[PROGRESS] 部分プログラム生成を開始: 最大{max_attempts}回試行", file=sys.stderr, flush=True)
                print(f"[PROGRESS] 部分プログラム生成失敗: 早期失敗検出（背景色戦略がunknown）", file=sys.stderr, flush=True)
                return {
                    'success': False,
                    'error': 'Background color strategy is unknown',
                    'debug_info': {
                        'timing': {
                            'total': total_time,
                            'object_extraction': extract_time,
                            'background_color_inference': bg_time,
                            'background_strategy_decision': bg_strategy_time,
                            'pattern_analysis': pattern_time,
                            'loop_total': 0.0,
                            'loop_attempts': 0,
                            'total_category_time': 0.0,
                            'total_partial_time': 0.0
                        },
                        'early_failure_reason': 'unknown_background_strategy',
                        'objects_before_filter': objects_before_filter
                    }
                }

            # 【最適化3】オブジェクト数ベースの試行回数調整
            # オブジェクト数に応じてmax_attemptsを動的に調整
            if objects_before_filter <= 2:
                # オブジェクト数1-2個: 10回（ただし、既に早期失敗検出で1個以下は除外済み）
                max_attempts = min(max_attempts, 10)
            elif objects_before_filter <= 5:
                # オブジェクト数3-5個: 30回
                max_attempts = min(max_attempts, 30)
            elif objects_before_filter <= 10:
                # オブジェクト数6-10個: 60回
                max_attempts = min(max_attempts, 60)
            # オブジェクト数11個以上: 元のmax_attemptsを維持（通常100回）

            # ⑦ 1つのパターンだけを処理（フォールバック処理付き）
            partial_program = None
            category_var_mapping = {}
            categories = []
            pattern_key = None
            category_timing = None
            filter_info = {}

            # 3071個の組み合わせから重み付け選択するため、total_patterns=3071を指定
            total_patterns_for_selection = 3071

            # 事前計算された組み合わせを取得（重み付け選択用）
            from .precomputed_combinations import PrecomputedCombinationsManager
            file_manager = PrecomputedCombinationsManager(self.config)
            precomputed = self.config.get_precomputed_combinations(total_patterns_for_selection)

            # 試行済みのpattern_idxを記録（重複を避けるため）
            tried_pattern_indices = set()

            # デバッグ用: 最初の数回の試行でグリッド内容を確認
            debug_attempt_count = 0
            max_debug_attempts = 3

            # ループ処理全体の時間を計測
            loop_start_time = time.time()

            # 各試行の時間を記録（成功・失敗に関わらず）
            attempt_times = []
            successful_attempt = None
            total_category_time = 0.0
            total_partial_time = 0.0

            # 失敗時のデバッグ情報を記録（最後の試行の情報を保存）
            last_attempt_debug_info = {
                'objects_before_filter': 0,
                'objects_after_filter': 0,
                'valid_categories_count': 0,
                'total_categories_count': 0,
                'last_pattern_idx': None,
                'last_result_type': None,  # 'None', 'prog_without_mapping', 'exception'
                'last_exception': None
            }

            # 進捗ログの出力間隔（10%ごと、または100回ごと、どちらか早い方）
            progress_log_interval = max(1, min(max_attempts // 10, 100))

            import sys
            # ログ出力制御（デフォルトで詳細ログを無効化）
            ENABLE_VERBOSE_OUTPUT_LOCAL = os.environ.get('ENABLE_VERBOSE_OUTPUT', 'false').lower() in ('true', '1', 'yes')
            ENABLE_ALL_LOGS_LOCAL = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')
            # 進捗ログは常に出力（タイミング情報の把握のため）
            print(f"[PROGRESS] 部分プログラム生成を開始: 最大{max_attempts}回試行", file=sys.stderr, flush=True)

            for attempt in range(max_attempts):
                # 各試行の開始時間を計測
                attempt_start_time = time.time()

                # 進捗ログを出力（10%ごと、または100回ごと、どちらか早い方）
                if (attempt + 1) % progress_log_interval == 0 or attempt == 0:
                    elapsed_loop_time = time.time() - loop_start_time
                    # 進捗ログは常に出力（タイミング情報の把握のため）
                    print(f"[PROGRESS] 試行 {attempt + 1}/{max_attempts} (経過時間: {elapsed_loop_time:.2f}秒, 平均: {elapsed_loop_time/(attempt+1):.3f}秒/試行)", file=sys.stderr, flush=True)

                # 重み付け選択でpattern_idxを取得（プログラム生成フロー用）
                # 組み合わせの選ばれやすさ = 組み合わせに含まれる特徴量の重みの平均
                # 試行済みのpattern_idxは除外して選択
                random.seed(attempt)  # attemptをシードとして使用（再現性のため）
                np.random.seed(attempt)  # numpyの乱数もシード設定
                pattern_idx = file_manager.get_weighted_combination_index(
                    total_patterns_for_selection, precomputed, excluded_indices=tried_pattern_indices
                )

                # 選択したpattern_idxを記録
                tried_pattern_indices.add(pattern_idx)
                try:
                    # デバッグ用: 最初の数回の試行でグリッド内容を確認
                    is_debug_attempt = debug_attempt_count < max_debug_attempts
                    if is_debug_attempt:
                        debug_attempt_count += 1

                    result = self._process_single_pattern(
                        connectivity, pattern_idx, objects_data[connectivity],
                        bg_strategy, transformation_patterns.get(connectivity, []),
                        grid_sizes, task, total_patterns_for_selection, max_colors_per_grid,  # total_patterns=3071: 3071個の組み合わせからランダム選択
                        is_debug_attempt=is_debug_attempt  # デバッグ用フラグ
                    )

                    # 試行の時間を記録（成功・失敗に関わらず）
                    attempt_elapsed = time.time() - attempt_start_time
                    attempt_times.append(attempt_elapsed)

                    if result is None:
                        # 失敗した試行の時間も記録
                        total_category_time += attempt_elapsed  # 概算（実際のcategory_timeは取得できない）
                        # 最後の試行の情報を記録（デバッグ用）
                        last_attempt_debug_info['last_pattern_idx'] = pattern_idx
                        last_attempt_debug_info['last_result_type'] = 'None'
                        continue  # 次のパターンを試行

                    # 成功した場合
                    key, cat_list, prog, category_time, partial_time, timing, info, mapping = result

                    # 成功した試行の時間を記録
                    total_category_time += category_time
                    total_partial_time += partial_time

                    # 最後の試行の情報を記録（デバッグ用）
                    if info:
                        last_attempt_debug_info['objects_before_filter'] = info.get('before', 0)
                        last_attempt_debug_info['objects_after_filter'] = info.get('after', 0)
                    if cat_list is not None:
                        valid_cats = self._select_valid_categories(cat_list) if cat_list else []
                        last_attempt_debug_info['total_categories_count'] = len(cat_list)
                        last_attempt_debug_info['valid_categories_count'] = len(valid_cats)
                    last_attempt_debug_info['last_pattern_idx'] = pattern_idx

                    if prog is not None:
                        # 部分プログラムが生成された場合
                        # category_var_mappingが空でない場合、カテゴリ分けが含まれている（カテゴリ数が2以上）
                        # category_var_mappingが空の場合、カテゴリ分けが含まれていない（カテゴリ数が1以下）
                        if mapping and len(mapping) > 0:
                            # カテゴリ分けが含まれている場合、成功として扱う
                            partial_program = prog
                            category_var_mapping = mapping
                            categories = cat_list
                            pattern_key = key
                            category_timing = timing
                            filter_info = info
                            successful_attempt = attempt + 1
                            # 成功ログは常に出力
                            print(f"[PROGRESS] 部分プログラム生成成功: 試行 {successful_attempt}/{max_attempts} (経過時間: {time.time() - loop_start_time:.2f}秒)", file=sys.stderr, flush=True)
                            break  # 成功したのでループを終了
                        else:
                            # カテゴリ分けが含まれていない場合
                            # 最後の試行でもカテゴリ数1以下の場合、成功として扱う（背景色フィルタリングのみの部分プログラム）
                            if attempt == max_attempts - 1:
                                # 最後の試行でもカテゴリ数1以下の場合、成功として扱う
                                partial_program = prog
                                category_var_mapping = {}
                                categories = cat_list if cat_list else []
                                pattern_key = key
                                category_timing = timing
                                filter_info = info
                                successful_attempt = attempt + 1
                                # 成功ログは常に出力
                                print(f"[PROGRESS] 部分プログラム生成成功: 試行 {successful_attempt}/{max_attempts} (経過時間: {time.time() - loop_start_time:.2f}秒, カテゴリ数1以下のため背景色フィルタリングのみ)", file=sys.stderr, flush=True)
                                break  # 成功したのでループを終了
                            else:
                                # 最後の試行でない場合、次のpattern_idxを試行する
                                last_attempt_debug_info['last_result_type'] = 'prog_without_mapping'
                                continue

                except Exception as e:
                    # エラーが発生した試行の時間も記録
                    attempt_elapsed = time.time() - attempt_start_time
                    attempt_times.append(attempt_elapsed)
                    total_category_time += attempt_elapsed  # 概算
                    # 最後の試行の情報を記録（デバッグ用）
                    last_attempt_debug_info['last_pattern_idx'] = pattern_idx
                    last_attempt_debug_info['last_result_type'] = 'exception'
                    last_attempt_debug_info['last_exception'] = str(e)
                    # エラーが発生した場合は次のパターンを試行
                    continue

            # ループ処理全体の時間を計測
            loop_time = time.time() - loop_start_time

            # 部分プログラムが生成されなかった場合
            if partial_program is None:
                # 失敗時もタイミング情報を返す
                total_time = time.time() - total_start_time
                # 失敗ログは常に出力
                avg_attempt_time = sum(attempt_times) / len(attempt_times) if attempt_times else 0.0
                print(f"[PROGRESS] 部分プログラム生成失敗: {max_attempts}回試行完了 (合計時間: {loop_time:.2f}秒, 平均: {avg_attempt_time:.3f}秒/試行)", file=sys.stderr, flush=True)

                # 詳細なデバッグ情報を収集
                # オブジェクト数の統計（フィルタリング前後）
                objects_before_filter_total = 0
                if connectivity in objects_data:
                    input_grids = objects_data[connectivity].get('input_grids', [])
                    if isinstance(input_grids, list):
                        objects_before_filter_total = sum(len(grid_objects) for grid_objects in input_grids)

                # グリッド情報を収集
                grid_debug_info = []
                for pair_idx, train_pair in enumerate(task.train):
                    input_grid = train_pair.get('input', [])
                    if input_grid:
                        try:
                            grid_array = np.array(input_grid)
                            unique_colors = np.unique(grid_array).tolist()
                            grid_debug_info.append({
                                'pair_idx': pair_idx,
                                'grid_size': (len(input_grid), len(input_grid[0]) if input_grid else 0),
                                'unique_colors': unique_colors,
                                'num_colors': len(unique_colors)
                            })
                        except Exception:
                            pass

                # 背景色情報を収集
                bg_debug_info = {
                    'strategy': bg_strategy.get('type', 'unknown'),
                    'inferred_colors': background_colors if background_colors else {}
                }

                # 詳細なデバッグ情報を出力
                print(f"[DEBUG] 部分プログラム生成失敗の詳細情報:", file=sys.stderr, flush=True)
                print(f"  試行回数: {max_attempts}回", file=sys.stderr, flush=True)
                print(f"  最後の試行パターンインデックス: {last_attempt_debug_info.get('last_pattern_idx', 'N/A')}", file=sys.stderr, flush=True)
                print(f"  最後の試行結果タイプ: {last_attempt_debug_info.get('last_result_type', 'N/A')}", file=sys.stderr, flush=True)
                if last_attempt_debug_info.get('last_exception'):
                    print(f"  最後の例外: {last_attempt_debug_info.get('last_exception')}", file=sys.stderr, flush=True)
                print(f"  オブジェクト数（フィルタリング前）: {objects_before_filter_total}", file=sys.stderr, flush=True)
                print(f"  オブジェクト数（フィルタリング後）: {last_attempt_debug_info.get('objects_after_filter', 'N/A')}", file=sys.stderr, flush=True)
                print(f"  カテゴリ数（総数）: {last_attempt_debug_info.get('total_categories_count', 'N/A')}", file=sys.stderr, flush=True)
                print(f"  有効なカテゴリ数: {last_attempt_debug_info.get('valid_categories_count', 'N/A')}", file=sys.stderr, flush=True)
                print(f"  背景色戦略: {bg_debug_info['strategy']}", file=sys.stderr, flush=True)
                if grid_debug_info:
                    print(f"  グリッド情報:", file=sys.stderr, flush=True)
                    for grid_info in grid_debug_info[:3]:  # 最初の3つのグリッドのみ出力
                        print(f"    グリッド{grid_info['pair_idx']}: サイズ={grid_info['grid_size']}, 色の種類数={grid_info['num_colors']}, 色={grid_info['unique_colors']}", file=sys.stderr, flush=True)

                return {
                    'success': False,
                    'error': f'Failed to generate partial program after {max_attempts} attempts',
                    'debug_info': {
                        'timing': {
                            'object_extraction': extract_time,
                            'background_color_inference': bg_time,
                            'background_strategy_decision': bg_strategy_time,
                            'pattern_analysis': pattern_time,
                            'loop_total': loop_time,
                            'loop_attempts': len(attempt_times),
                            'loop_avg_per_attempt': sum(attempt_times) / len(attempt_times) if attempt_times else 0.0,
                            'loop_max_attempt': max(attempt_times) if attempt_times else 0.0,
                            'loop_min_attempt': min(attempt_times) if attempt_times else 0.0,
                            'total': total_time
                        },
                        'failure_details': {
                            'objects_before_filter': objects_before_filter_total,
                            'objects_after_filter': last_attempt_debug_info.get('objects_after_filter', 0),
                            'valid_categories_count': last_attempt_debug_info.get('valid_categories_count', 0),
                            'total_categories_count': last_attempt_debug_info.get('total_categories_count', 0),
                            'last_pattern_idx': last_attempt_debug_info.get('last_pattern_idx'),
                            'last_result_type': last_attempt_debug_info.get('last_result_type'),
                            'last_exception': last_attempt_debug_info.get('last_exception'),
                            'background_strategy': bg_debug_info,
                            'grid_info': grid_debug_info
                        }
                    }
                }

            # ⑧ 結果を整形
            partial_programs = [partial_program]

            # カテゴリ情報を整形
            pattern_category_var_mapping = {}
            for cat in categories:
                original_category_id = cat.category_id
                if isinstance(original_category_id, int) and original_category_id in category_var_mapping:
                    pattern_category_var_mapping[cat.category_id] = category_var_mapping[original_category_id]

            # カテゴリ情報を整形（partial_program_to_categories用）
            pattern_categories_for_mapping = []
            for cat in categories:
                cat_copy = deepcopy(cat)
                pattern_categories_for_mapping.append(cat_copy)

            # グリッドサイズと背景色を各訓練ペアごとに計算してキャッシュ
            from src.hybrid_system.inference.program_synthesis.candidate_generators.common_helpers import get_grid_size, get_background_color
            grid_info_cache = {}
            for pair_idx, train_pair in enumerate(task.train):
                input_grid = train_pair.get('input', [])
                output_grid = train_pair.get('output', [])
                if input_grid and output_grid:
                    input_h, input_w = grid_sizes[pair_idx] if pair_idx < len(grid_sizes) else (len(input_grid), len(input_grid[0]) if input_grid else 0)
                    output_h, output_w = get_grid_size(output_grid)
                    input_bg = self._get_background_color_for_grid(
                        bg_strategy, pair_idx, all_input_grids, num_train_pairs
                    )
                    output_bg = get_background_color(output_grid)
                    grid_info_cache[pair_idx] = {
                        'input_size': (input_h, input_w),
                        'output_size': (output_h, output_w),
                        'input_bg': input_bg,
                        'output_bg': output_bg
                    }

            total_time = time.time() - total_start_time

            # 成功時のタイミング情報を拡張
            avg_attempt_time = sum(attempt_times) / len(attempt_times) if attempt_times else 0.0
            # 完了ログは常に出力
            print(f"[PROGRESS] 部分プログラム生成完了: {successful_attempt}回目で成功 (ループ処理時間: {loop_time:.2f}秒, 平均: {avg_attempt_time:.3f}秒/試行)", file=sys.stderr, flush=True)

            return {
                'success': True,
                'partial_programs': partial_programs,
                'category_var_mappings': {pattern_key: pattern_category_var_mapping} if pattern_key else {},
                'partial_program_to_category_var_mapping': {
                    partial_program: pattern_category_var_mapping
                },
                'partial_program_to_categories': {
                    partial_program: pattern_categories_for_mapping
                },
                'background_colors': background_colors,
                'bg_strategy': bg_strategy,
                'transformation_patterns': transformation_patterns,
                'grid_info_cache': grid_info_cache,
                'objects_data': objects_data,
                'debug_info': {
                    'timing': {
                        'object_extraction': extract_time,
                        'background_color_inference': bg_time,
                        'background_strategy_decision': bg_strategy_time,
                        'pattern_analysis': pattern_time,
                        'loop_total': loop_time,
                        'loop_attempts': successful_attempt,
                        'loop_avg_per_attempt': avg_attempt_time,
                        'loop_max_attempt': max(attempt_times) if attempt_times else 0.0,
                        'loop_min_attempt': min(attempt_times) if attempt_times else 0.0,
                        'total_category_time': total_category_time,
                        'total_partial_time': total_partial_time,
                        'total': total_time
                    }
                }
            }

        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            return {
                'success': False,
                'error': str(e),
                'traceback': error_traceback
            }

    def match_objects_batch(
        self, tasks: List[Task]
    ) -> List[Dict[str, Any]]:
        """
        複数のタスクを逐次処理してオブジェクトマッチングを実行

        Args:
            tasks: Taskオブジェクトのリスト

        Returns:
            マッチング結果のリスト（タスクの順序に対応）
        """
        if not tasks:
            return []

        # 逐次処理
        results = []
        for task in tasks:
            try:
                result = self.match_objects(task)
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__
                })

        return results

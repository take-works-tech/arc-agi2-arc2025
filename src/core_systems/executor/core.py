#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
プログラム実行器コア
プログラムの実行とメイン機能を提供
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from src.core_systems.executor.grid import grid_size_context, GridSizeCommandExecutor, OutputGridGenerator
from src.core_systems.executor.operations import Operation
from src.data_systems.data_models.core.object import Object
from src.data_systems.data_models.base import ObjectType
import logging

# s1_extractorをインポート
from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor, ObjectExtractionResult
from src.data_systems.config.config import ExtractionConfig

# モジュールレベルのloggerを定義
logger = logging.getLogger("ProgramExecutorCore")

# ログ出力制御（デフォルトで詳細ログを無効化）
ENABLE_VERBOSE_LOGGING = os.environ.get('ENABLE_VERBOSE_LOGGING', 'false').lower() in ('true', '1', 'yes')
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')

# オブジェクト数の上限設定（一括管理）
# ARC問題の特性を考慮した設定:
# - 通常のARC問題ではオブジェクト数は5-20個程度
# - 複雑な問題でも50個以下
# - プログラム実行中、ループなどで一時的に増える可能性があるが、通常は数百個程度
MAX_OBJECTS_DURING_EXECUTION = 10000  # プログラム実行中の総オブジェクト数上限（ARC問題では通常100個以下）
MAX_OBJECTS_IN_VARIABLE = 512  # 1つの変数（配列）に格納できる最大オブジェクト数（拡張: 大規模テンプレートに対応）
MAX_TOTAL_PIXELS_IN_VARIABLE = 4000  # 1つの変数（配列）に蓄積できるオブジェクト総ピクセル数の上限
MAX_OBJECTS_FOR_EXCLUDE = 512  # EXCLUDE操作で処理できる最大オブジェクト数（拡張: 大規模テンプレートに対応）

# スタックトレースを抑制するためのカスタム例外クラス
class SilentException(Exception):
    """スタックトレースを出力しない例外クラス"""
    def __init__(self, message):
        self.message = message
        super().__init__(message)
        # スタックトレースを完全に抑制（デフォルト設定）
        self.__traceback__ = None
        self.__suppress_context__ = True

# SilentExceptionをモジュールレベルで定義（他の関数からも参照可能にする）

# スタックトレースを抑制するためのexcepthook
_original_excepthook = sys.excepthook
def _silent_excepthook(exc_type, exc_value, exc_traceback):
    """特定の例外のスタックトレースを抑制する"""
    if isinstance(exc_value, SilentException):
        # SilentExceptionの場合は何も出力しない（完全に抑制）
        pass
    else:
        # それ以外の例外は通常通りスタックトレースを出力
        _original_excepthook(exc_type, exc_value, exc_traceback)

# excepthookを設定（グローバル）
sys.excepthook = _silent_excepthook

class ExecutorCore:
    """プログラム実行器コア"""

    def __init__(self):
        self.output_grid_generator = OutputGridGenerator()
        self.grid_size_executor = GridSizeCommandExecutor()
        self.grid_context = grid_size_context

        # s1_extractorを初期化
        self.object_extractor = IntegratedObjectExtractor(ExtractionConfig())

        # グローバルオブジェクトカウンター（一意のID生成用）
        self._object_counter = 0

        # プログラム実行環境
        self.execution_context = {
            'objects': {},    # オブジェクトの辞書として初期化
            'variables': {},  # 変数の永続性を確保
            'arrays': {},     # 配列の永続性を確保
            'results': [],    # 実行結果
            'input_image_index': 0,
            'background_color': 0,
            'object_type_filter': None,  # 設定されたオブジェクトタイプ
            'object_type_set': False     # 既に設定済みかどうか
        }

        #logger.info("ProgramExecutorCore初期化完了")

    def clear_persistent_data(self):
        """永続データをクリア（テスト用）"""
        self.execution_context['variables'] = {}
        self.execution_context['arrays'] = {}
        self.execution_context['results'] = []
        #logger.info("永続データ（変数・配列）をクリアしました")

    def _generate_unique_object_id(self, prefix: str = "obj") -> str:
        """一意のオブジェクトIDを生成

        Args:
            prefix: IDのプレフィックス（デフォルト: "obj"）

        Returns:
            一意のオブジェクトID
        """
        self._object_counter += 1
        return f"{prefix}_{self._object_counter}"

    def _get_all_objects_with_connectivity(self, connectivity: int) -> List[str]:
        """指定された連結性でオブジェクトを取得

        Args:
            connectivity: 連結性（4 or 8）

        Returns:
            オブジェクトIDのリスト（指定された連結性の全オブジェクト、背景色も含む）
        """
        # execution_contextからオブジェクトを取得
        all_objects = self.execution_context.get('objects', {})

        # デバッグ: 最初の10タスクのみ詳細を出力
        DEBUG_TASK_INDEX = int(os.environ.get('DEBUG_TASK_INDEX', '0'))
        DEBUG_GET_ALL_OBJECTS = DEBUG_TASK_INDEX > 0 and DEBUG_TASK_INDEX <= 10

        if DEBUG_GET_ALL_OBJECTS:
            print(f"  [デバッグ] GET_ALL_OBJECTS({connectivity}): execution_context['objects']の数={len(all_objects) if isinstance(all_objects, dict) else len(all_objects) if isinstance(all_objects, list) else 0}", flush=True)

        if isinstance(all_objects, dict) and len(all_objects) > 0:
            sample_obj_ids = list(all_objects.keys())[:3]
            if DEBUG_GET_ALL_OBJECTS:
                print(f"  [デバッグ] GET_ALL_OBJECTS({connectivity}): サンプルobject_id: {sample_obj_ids}", flush=True)
            # 各オブジェクトの色をログ出力
            for obj_id in sample_obj_ids[:3]:
                obj = all_objects.get(obj_id)
                if obj:
                    obj_color = obj.color if hasattr(obj, 'color') else 'N/A'
                    if DEBUG_GET_ALL_OBJECTS:
                        print(f"  [デバッグ] GET_ALL_OBJECTS({connectivity}):   - {obj_id}: color={obj_color}", flush=True)

        object_ids = []

        # 辞書形式とリスト形式の両方に対応
        if isinstance(all_objects, dict):
            for obj_id, obj in all_objects.items():
                # 連結性でフィルタリング（背景色は除外しない）
                if connectivity == 4:
                    # 4方向連結のみ
                    if 'single_color_4way' in obj_id.lower():
                        object_ids.append(obj_id)
                elif connectivity == 8:
                    # 8方向連結のみ
                    if 'single_color_8way' in obj_id.lower():
                        object_ids.append(obj_id)

            if DEBUG_GET_ALL_OBJECTS:
                print(f"  [デバッグ] GET_ALL_OBJECTS({connectivity}): フィルタリング後のオブジェクト数={len(object_ids)}", flush=True)
            if len(object_ids) > 0:
                # 抽出されたオブジェクトの色分布をログ出力
                color_counts = {}
                for obj_id in object_ids:
                    obj = all_objects.get(obj_id)
                    if obj:
                        obj_color = obj.color if hasattr(obj, 'color') else None
                        if obj_color is not None:
                            color_counts[obj_color] = color_counts.get(obj_color, 0) + 1
                if DEBUG_GET_ALL_OBJECTS:
                    print(f"  [デバッグ] GET_ALL_OBJECTS({connectivity})で抽出されたオブジェクトの色分布: {color_counts}", flush=True)
            if len(object_ids) == 0 and len(all_objects) > 0:
                if DEBUG_GET_ALL_OBJECTS:
                    print(f"  [警告] GET_ALL_OBJECTS({connectivity}): フィルタリングでオブジェクトが0個になりました。連結性{connectivity}に一致するobject_idが見つかりませんでした。", flush=True)
                    # 全オブジェクトのobject_idパターンをログ出力
                    all_obj_ids = list(all_objects.keys())[:10]
                    print(f"  [警告] GET_ALL_OBJECTS({connectivity}): 全オブジェクトのobject_idサンプル（最初の10個）: {all_obj_ids}", flush=True)
        else:
            # 注: リスト形式のフォールバックは削除（execution_context['objects']は常に辞書形式）
            # リスト形式が渡された場合は空のリストを返す
            if DEBUG_GET_ALL_OBJECTS:
                print(f"  [警告] GET_ALL_OBJECTS({connectivity}): all_objectsが辞書形式ではありません（型: {type(all_objects)}）", flush=True)
            return []

        return object_ids

    def _render_grid(self, object_ids: List[str], background_color: int,
                    width: int, height: int, x_offset: int = 0, y_offset: int = 0):
        """グリッドをレンダリング

        Args:
            object_ids: オブジェクトIDのリスト
            background_color: 背景色
            width: 幅
            height: 高さ
            x_offset: X方向のオフセット（オプション）
            y_offset: Y方向のオフセット（オプション）
        """
        # execution_contextからオブジェクトを取得
        all_objects = self.execution_context.get('objects', {})

        # オブジェクトIDに一致するObjectインスタンスを取得
        objects = []
        if isinstance(all_objects, dict):
            # 辞書形式
            for obj_id in object_ids:
                obj = all_objects.get(obj_id)
                if obj:
                    objects.append(obj)
        else:
            # 注: リスト形式のフォールバックは削除（execution_context['objects']は常に辞書形式）
            # リスト形式が渡された場合は空のリストを返す
            return []

        # output_grid_generatorを使用してグリッドを生成
        result = self.output_grid_generator.generate_output_grid_from_objects(
            objects=objects,
            input_grid_size=(width, height),
            background_color=background_color
        )

        if result.success:
            # execution_contextに保存
            self.execution_context['output_grid'] = result.output_grid
            self.execution_context['rendered_objects'] = objects  # 実際に描画したオブジェクト
            self.execution_context['program_terminated'] = True  # 出力グリッド生成完了フラグ
            #logger.info(f"RENDER_GRID: {width}x{height}グリッドを生成, {len(objects)}個のオブジェクトを描画")
            return result.output_grid
        else:
            #logger.error(f"RENDER_GRID失敗: {result.message}")
            raise RuntimeError(f"RENDER_GRID failed: {result.message}")

    def get_persistent_state(self):
        """永続状態を取得（テスト用）"""
        return {
            'variables': self.execution_context.get('variables', {}).copy(),
            'arrays': self.execution_context.get('arrays', {}).copy(),
            'results': self.execution_context.get('results', []).copy()
        }

    def execute_program(self, program_code: str, input_grid: np.ndarray,
                       input_objects: List[Object] = None,
                       input_image_index: int = 0,
                       background_color: int = None) -> Tuple[np.ndarray, List[Object], float]:
        """プログラムを実行（新構文のみ）"""
        # デバッグ: 最初の10タスクのみ詳細を出力
        DEBUG_TASK_INDEX = int(os.environ.get('DEBUG_TASK_INDEX', '0'))
        DEBUG_ENABLED = (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and DEBUG_TASK_INDEX > 0 and DEBUG_TASK_INDEX <= 10

        if DEBUG_ENABLED:
            print(f"  [デバッグ] ExecutorCore.execute_program: 開始 (タスク{DEBUG_TASK_INDEX})", flush=True)
            print(f"    input_grid.shape={input_grid.shape if hasattr(input_grid, 'shape') else 'N/A'}", flush=True)

        #logger.info("プログラム実行開始")
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[ExecutorCore.execute_program] _execute_new_syntax呼び出し前", flush=True)
        result = self._execute_new_syntax(program_code, input_grid,
                                       input_objects, input_image_index,
                                       background_color)
        if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
            print(f"[ExecutorCore.execute_program] _execute_new_syntax呼び出し完了", flush=True)

        if DEBUG_ENABLED:
            output_grid, final_objects, execution_time = result
            print(f"  [デバッグ] ExecutorCore.execute_program: 完了 (タスク{DEBUG_TASK_INDEX})", flush=True)
            print(f"    output_grid.shape={output_grid.shape if hasattr(output_grid, 'shape') else 'N/A'}", flush=True)
            print(f"    final_objects数={len(final_objects) if final_objects else 0}", flush=True)

        return result

    def execute(self, program_code: str, input_grid: np.ndarray) -> np.ndarray:
        """プログラムを実行

        execute_programの結果から出力グリッドのみを返す便利メソッド
        """
        result, _, _ = self.execute_program(program_code, input_grid)
        return result


    def _execute_new_syntax(self, program_code: str, input_grid: np.ndarray,
                           input_objects: List[Object] = None,
                           input_image_index: int = 0,
                           background_color: int = None) -> Tuple[np.ndarray, List[Object], float]:
        """
        新構文でプログラムを実行

        Args:
            program_code: プログラムコード（新構文）
            input_grid: 入力グリッド
            input_objects: 入力オブジェクト
            input_image_index: 入力画像インデックス
            background_color: 背景色

        Returns:
            (出力グリッド, オブジェクトリスト, 信頼度)
        """
        try:
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] 関数開始", flush=True)
            # デバッグ用の環境変数を取得（osはモジュールレベルで既にインポート済み）
            DEBUG_TASK_INDEX_FUNC = int(os.environ.get('DEBUG_TASK_INDEX', '0'))
            DEBUG_TASK_INDEX_FUNC_IN_TRY = DEBUG_TASK_INDEX_FUNC  # tryブロック内で使用する値を保持
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] 環境変数取得完了 (DEBUG_TASK_INDEX_FUNC={DEBUG_TASK_INDEX_FUNC})", flush=True)

            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] インポート開始", flush=True)
            from .parsing.tokenizer import Tokenizer
            from .parsing.parser import Parser
            from .parsing.interpreter import Interpreter
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] インポート完了", flush=True)

            # input_gridをnumpy配列に変換
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] input_grid変換開始", flush=True)
            if not isinstance(input_grid, np.ndarray):
                input_grid = np.array(input_grid)
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] input_grid変換完了 (shape={input_grid.shape})", flush=True)

            # グリッドサイズコンテキストを初期化
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] グリッドサイズコンテキスト初期化開始", flush=True)
            height, width = input_grid.shape
            self.grid_context.initialize((width, height))
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] グリッドサイズコンテキスト初期化完了", flush=True)

            # 実行コンテキストを設定
            if input_objects is None:
                # s1_extractorを使用してオブジェクトを抽出
                extraction_result = self.object_extractor.extract_objects_by_type(input_grid, input_image_index)
                if extraction_result.success:
                    # すべてのオブジェクトを統合（辞書形式）
                    all_objects = {}
                    for object_type, object_list in extraction_result.objects_by_type.items():
                        for obj in object_list:
                            all_objects[obj.object_id] = obj
                    self.execution_context['objects'] = all_objects
                    self.execution_context['background_color'] = extraction_result.background_color

                    # 大量のオブジェクトが抽出されている場合は警告
                    # 4連結と8連結の重複抽出を常にチェック（100個以上の条件を削除）
                    type_counts = {}
                    type_pixel_counts = {}  # 各タイプの総ピクセル数
                    pixels_by_type = {}  # 各タイプのピクセルセット（重複チェック用）

                    for obj_id, obj in all_objects.items():
                        obj_type = getattr(obj, 'object_type', None)
                        if obj_type:
                            type_name = obj_type.value if hasattr(obj_type, 'value') else str(obj_type)
                            type_counts[type_name] = type_counts.get(type_name, 0) + 1
                            # 各タイプの総ピクセル数をカウント
                            if type_name not in type_pixel_counts:
                                type_pixel_counts[type_name] = 0
                                pixels_by_type[type_name] = set()
                            if hasattr(obj, 'pixels') and obj.pixels:
                                pixel_set = set(obj.pixels)
                                type_pixel_counts[type_name] += len(obj.pixels)
                                pixels_by_type[type_name].update(pixel_set)

                    # グリッドサイズとオブジェクト数の関係を確認
                    grid_pixels = input_grid.shape[0] * input_grid.shape[1]

                    # 4連結と8連結の重複抽出は正常な動作（削除: 警告・エラーログは不要）

                    # デフォルトで無効化（詳細ログ）
                    if len(all_objects) > 100 and (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS):
                        logger.warning(f"オブジェクト抽出: 大量のオブジェクトが抽出されました（{len(all_objects)}個）")
                        logger.warning(f"オブジェクト抽出: オブジェクトタイプごとの数: {type_counts}")
                        logger.warning(f"オブジェクト抽出: オブジェクトタイプごとの総ピクセル数: {type_pixel_counts}")
                        logger.warning(f"オブジェクト抽出: グリッドサイズ={input_grid.shape}, 総ピクセル数={grid_pixels}, オブジェクト数={len(all_objects)}")

                        # 各ピクセルが個別のオブジェクトとして抽出されている可能性をチェック
                        if len(all_objects) > grid_pixels * 0.5:  # ピクセル数の50%以上
                            logger.warning(f"オブジェクト抽出: 各ピクセルが個別のオブジェクトとして抽出されている可能性があります！")

                        # 1ピクセルオブジェクトの数を確認
                        single_pixel_counts = {}
                        for obj_id, obj in all_objects.items():
                            obj_type = getattr(obj, 'object_type', None)
                            if obj_type:
                                type_name = obj_type.value if hasattr(obj_type, 'value') else str(obj_type)
                                if hasattr(obj, 'pixels') and obj.pixels and len(obj.pixels) == 1:
                                    single_pixel_counts[type_name] = single_pixel_counts.get(type_name, 0) + 1
                        if single_pixel_counts:
                            logger.warning(f"オブジェクト抽出: 1ピクセルオブジェクトの数: {single_pixel_counts}")

                    #logger.info(f"[DEBUG] オブジェクト抽出完了: {len(all_objects)}個のオブジェクト")
                    #logger.info(f"[DEBUG] 推論された背景色: {extraction_result.background_color}")

                    # 抽出されたオブジェクトの色分布をログ出力
                    if len(all_objects) > 0:
                        color_counts = {}
                        for obj_id, obj in all_objects.items():
                            obj_color = obj.color if hasattr(obj, 'color') else None
                            if obj_color is not None:
                                color_counts[obj_color] = color_counts.get(obj_color, 0) + 1
                        #logger.info(f"[DEBUG] 抽出されたオブジェクトの色分布: {color_counts}")

                    if len(all_objects) == 0:
                        #logger.warning(f"[DEBUG] 抽出されたオブジェクトが0個です。input_gridの形状: {input_grid.shape}, ユニークな色: {np.unique(input_grid)}")
                        pass
                else:
                    self.execution_context['objects'] = {}
                    # background_colorがNoneの場合はデフォルト値0を使用
                    self.execution_context['background_color'] = background_color if background_color is not None else 0
                    #logger.warning(f"[DEBUG] オブジェクト抽出に失敗しました")
            else:
                self.execution_context['objects'] = input_objects
                # background_colorがNoneの場合はデフォルト値0を使用
                self.execution_context['background_color'] = background_color if background_color is not None else 0

            self.execution_context['input_image_index'] = input_image_index
            self.execution_context['input_grid'] = input_grid
            self.execution_context['grid'] = input_grid  # GET_ALL_OBJECTS用

            # 変数と配列の初期化
            if 'variables' not in self.execution_context:
                self.execution_context['variables'] = {}
            if 'arrays' not in self.execution_context:
                self.execution_context['arrays'] = {}

            # オブジェクトタイプ設定フラグをリセット
            self.execution_context['object_type_set'] = False

            # トークナイズ
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] トークナイズ開始", flush=True)
            # DEBUG_TASK_INDEX_FUNC_IN_TRYは既に定義されている（オブジェクト抽出部分で定義済み）
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and DEBUG_TASK_INDEX_FUNC_IN_TRY > 0:
                print(f"[DEBUG] タスク{DEBUG_TASK_INDEX_FUNC_IN_TRY}: トークナイズ開始", flush=True)
            tokenizer = Tokenizer()
            tokens = tokenizer.tokenize(program_code)
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] トークナイズ完了（{len(tokens)}トークン）", flush=True)
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and DEBUG_TASK_INDEX_FUNC_IN_TRY > 0:
                print(f"[DEBUG] タスク{DEBUG_TASK_INDEX_FUNC_IN_TRY}: トークナイズ完了（{len(tokens)}トークン）", flush=True)
            #logger.info(f"トークン数: {len(tokens)}")

            # トークン数が異常に大きい場合は警告
            if len(tokens) > 10000:
                error_msg = f"トークン数が異常に大きいです（{len(tokens)}トークン）。パース処理が長時間かかる可能性があります。"
                print(f"  [警告] {error_msg}", flush=True)
                # SilentExceptionはモジュールレベルで定義済み
                raise SilentException(error_msg)

            # パース（タイムアウト付き）
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] パース開始（トークン数={len(tokens)}）", flush=True)
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and DEBUG_TASK_INDEX_FUNC_IN_TRY > 0:
                print(f"[DEBUG] タスク{DEBUG_TASK_INDEX_FUNC_IN_TRY}: パース開始（トークン数={len(tokens)}）", flush=True)
            parse_start_time = time.time()
            parser = Parser(tokens)
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] Parser作成完了、parser.parse呼び出し前", flush=True)

            # パース処理にタイムアウトを設定（最大1.0秒）
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            parse_timeout = 1.0  # パース処理の最大時間（秒）
            executor_pool = None
            future = None
            try:
                executor_pool = ThreadPoolExecutor(max_workers=1)
                future = executor_pool.submit(parser.parse)
                ast = future.result(timeout=parse_timeout)
                parse_elapsed = time.time() - parse_start_time
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    print(f"[_execute_new_syntax] パース完了（{len(ast)}ノード、処理時間={parse_elapsed:.3f}秒）", flush=True)
            except FuturesTimeoutError:
                error_msg = f"パース処理がタイムアウトしました（{parse_timeout:.2f}秒以内に完了しませんでした）。トークン数: {len(tokens)}"
                print(f"  [警告] {error_msg}", flush=True)
                raise SilentException(error_msg)
            except KeyboardInterrupt:
                if future is not None:
                    future.cancel()
                raise
            finally:
                if executor_pool is not None:
                    executor_pool.shutdown(wait=False)
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and DEBUG_TASK_INDEX_FUNC_IN_TRY > 0:
                print(f"[DEBUG] タスク{DEBUG_TASK_INDEX_FUNC_IN_TRY}: パース完了（{len(ast)}ノード）", flush=True)
            #logger.info(f"AST生成完了: {len(ast)}ノード")

            # 実行前のオブジェクト数を記録
            objects_before_exec = self.execution_context.get('objects', {})
            object_count_before = len(objects_before_exec) if isinstance(objects_before_exec, dict) else len(objects_before_exec) if isinstance(objects_before_exec, list) else 0

            # 実行
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] インタープリター実行開始（オブジェクト数={object_count_before}）", flush=True)
            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and DEBUG_TASK_INDEX_FUNC_IN_TRY > 0:
                print(f"[DEBUG] タスク{DEBUG_TASK_INDEX_FUNC_IN_TRY}: インタープリター実行開始（オブジェクト数={object_count_before}）", flush=True)
            interpreter = Interpreter(self)
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] Interpreter作成完了、interpreter.execute呼び出し前", flush=True)

            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and DEBUG_TASK_INDEX_FUNC_IN_TRY > 0:
                print(f"[DEBUG] タスク{DEBUG_TASK_INDEX_FUNC_IN_TRY}: interpreter.execute呼び出し", flush=True)

            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] interpreter.execute呼び出し開始", flush=True)
            interpreter.execute(ast)
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] interpreter.execute呼び出し完了", flush=True)

            # 実行後のオブジェクト数をチェック
            objects_after_exec = self.execution_context.get('objects', {})
            object_count_after = len(objects_after_exec) if isinstance(objects_after_exec, dict) else len(objects_after_exec) if isinstance(objects_after_exec, list) else 0
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] Inter1", flush=True)
            # オブジェクト数が異常に増加している場合はエラー
            if object_count_after > MAX_OBJECTS_DURING_EXECUTION:
                error_msg = f"プログラム実行中、オブジェクト数が上限を超えました（{object_count_after}個 > MAX_OBJECTS_DURING_EXECUTION個、実行前={object_count_before}個）。プログラムに問題がある可能性があります。"
                logger.error(error_msg)
                raise SilentException(error_msg)

            # トレース用にサマリ情報をresultsに記録（必要な場合のみ使用）
            try:
                results_list = self.execution_context.get('results')
                if not isinstance(results_list, list):
                    results_list = []
                    self.execution_context['results'] = results_list

                # 出力オブジェクト（描画に使われたオブジェクト）のサマリを作成
                rendered_objects = self.execution_context.get('rendered_objects', [])
                object_summaries = []
                try:
                    max_objects = 50  # JSONサイズ抑制のため、最大50個まで
                    if isinstance(rendered_objects, list):
                        for obj in rendered_objects[:max_objects]:
                            # Object型を想定した安全なアクセス（失敗してもスキップ）
                            try:
                                obj_id = getattr(obj, "object_id", "")
                                obj_type = getattr(obj, "object_type", None)
                                if hasattr(obj_type, "value"):
                                    obj_type = obj_type.value
                                color = getattr(obj, "color", None)
                                bbox = getattr(obj, "bbox", None)
                                width = getattr(obj, "bbox_width", None)
                                height = getattr(obj, "bbox_height", None)
                                object_summaries.append({
                                    "id": obj_id,
                                    "type": obj_type,
                                    "color": color,
                                    "bbox": list(bbox) if isinstance(bbox, tuple) else bbox,
                                    "width": width,
                                    "height": height,
                                })
                            except Exception:
                                continue
                except Exception:
                    object_summaries = []

                results_list.append({
                    'event': 'execution_summary',
                    'object_count_before': object_count_before,
                    'object_count_after': object_count_after,
                    'program_terminated': bool(self.execution_context.get('program_terminated', False)),
                    'grid_size': tuple(input_grid.shape) if hasattr(input_grid, 'shape') else None,
                    # 出力オブジェクト側の情報（描画に使われたオブジェクト）
                    'rendered_object_count': len(rendered_objects) if isinstance(rendered_objects, list) else 0,
                    'objects': object_summaries,
                })
            except Exception:
                # トレース記録で問題が起きても本処理には影響させない
                pass

            # デフォルトで無効化（詳細ログ）
            if (ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS) and DEBUG_TASK_INDEX_FUNC_IN_TRY > 0:
                print(f"[DEBUG] タスク{DEBUG_TASK_INDEX_FUNC_IN_TRY}: インタープリター実行完了（オブジェクト数={object_count_after}）", flush=True)
            #logger.info("プログラム実行完了")
            if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                print(f"[_execute_new_syntax] Inter2", flush=True)
            # RENDER_GRIDで出力が生成されたかチェック
            if self.execution_context.get('program_terminated'):
                output_grid = self.execution_context.get('output_grid')

                # RENDER_GRIDで実際に描画したオブジェクトを取得
                objects = self.execution_context.get('rendered_objects', [])

                #logger.info(f"RENDER_GRIDで生成された出力グリッド: {output_grid.shape}, {len(objects)}個のオブジェクト")

                # 最終結果のデバッグ情報をログ出力
                if len(objects) == 0:
                    #logger.warning(f"[DEBUG] 最終実行結果: RENDER_GRIDで描画されたオブジェクトが0個です")
                    # 実行コンテキストの状態を確認
                    all_ctx_objects = self.execution_context.get('objects', {})
                    #logger.warning(f"[DEBUG] 実行コンテキストの全オブジェクト数: {len(all_ctx_objects) if isinstance(all_ctx_objects, dict) else len(all_ctx_objects) if isinstance(all_ctx_objects, list) else 0}")
                    #logger.warning(f"[DEBUG] 実行コンテキストの背景色: {self.execution_context.get('background_color', 'N/A')}")

                return output_grid, objects, 1.0

            # 注: RENDER_GRIDがない場合は自動生成（プログラムにRENDER_GRIDコマンドが含まれていない場合のフォールバック）
            output_grid = self._generate_output_grid()
            objects_store = self.execution_context.get('objects', {})

            # dictをList[Object]に変換
            if isinstance(objects_store, dict):
                objects = list(objects_store.values())
            else:
                objects = objects_store

            #logger.info(f"新構文実行完了: グリッドサイズ={output_grid.shape}")

            return output_grid, objects, 1.0

        except Exception as e:
            # SilentExceptionの場合はそのまま再発生（スタックトレースは抑制される）
            if isinstance(e, SilentException):
                raise
            # ProgramExecutionErrorの場合はエラーメッセージを1回だけログ出力（重複を避ける）
            from src.core_systems.executor.parsing.interpreter import ProgramExecutionError
            if isinstance(e, ProgramExecutionError):
                # 同じエラーメッセージが複数回出力されるのを防ぐため、1回だけログ出力
                error_msg = str(e)
                if not hasattr(self, '_last_program_error') or self._last_program_error != error_msg:
                    logger.error(f"新構文実行エラー: {e}")
                    self._last_program_error = error_msg
                # ProgramExecutionErrorもSilentExceptionとして扱い、タスクを廃棄
                raise SilentException(f"プログラム実行エラー: {error_msg}")
            else:
                # 通常のエラーの場合はログ出力（デバッグ時のみ詳細なスタックトレースを出力）
                logger.error(f"新構文実行エラー: {e}")
                # デバッグモードでのみスタックトレースを出力（デフォルトでは出力しない）
                # モジュールレベルのENABLE_VERBOSE_LOGGINGとENABLE_ALL_LOGSを使用
                if ENABLE_VERBOSE_LOGGING or ENABLE_ALL_LOGS:
                    import traceback
                    traceback.print_exc()
                # エラー時はSilentExceptionを投げてタスクを廃棄（入力グリッドを返さない）
                raise SilentException(f"プログラム実行エラー: {type(e).__name__}: {str(e)}")

    def _check_single_object_only(self, operation_name: str, selected_objects: List[Object]) -> Object:
        """オブジェクト操作が単体のみ受け付けることをチェック

        Args:
            operation_name: 操作名
            selected_objects: 選択されたオブジェクトのリスト

        Returns:
            単体オブジェクト

        Raises:
            TypeError: 配列が渡された場合
        """
        if len(selected_objects) != 1:
            #logger.error(f"{operation_name}操作エラー: 単体オブジェクトのみ受け付けます（受け取ったオブジェクト数: {len(selected_objects)}）")
            #logger.error("配列に対して操作したい場合はFORループを使用してください")
            #logger.error(f"例: FOR i LEN(array) DO")
            #logger.error(f"        array[i] = {operation_name}(array[i], ...)")
            #logger.error(f"    END")
            raise TypeError(
                f"{operation_name}: 単体オブジェクトのみ受け付けます。"
                f"配列が渡されました（{len(selected_objects)}個のオブジェクト）。"
                f"配列に対して操作したい場合はFORループを使用してください。"
            )
        return selected_objects[0]

    def _merge_objects(self, object_ids: List[str]) -> str:
        """オブジェクトの統合（オブジェクトID配列を受け取る）

        Returns:
            str: 結合後のオブジェクトID
        """
        try:
            #logger.info(f"MERGE_OBJECTS: 受け取ったオブジェクトID配列: {object_ids} (型: {type(object_ids)})")

            if len(object_ids) < 1:
                #logger.warning("MERGE_OBJECTS: 少なくとも1つのオブジェクトIDを指定してください")
                return None

            selected_objects = []
            missing_object_ids = []
            for i, object_id in enumerate(object_ids):
                #logger.info(f"MERGE_OBJECTS: 処理中 [{i}]: {object_id} (型: {type(object_id)})")

                # オブジェクトIDが文字列であることを確認
                if not isinstance(object_id, str):
                    #logger.warning(f"MERGE_OBJECTS: オブジェクトIDは文字列である必要があります (型: {type(object_id).__name__})")
                    continue

                obj = self._find_object_by_id(object_id)
                if obj:
                    selected_objects.append(obj)
                    #logger.info(f"MERGE_OBJECTS: オブジェクト '{object_id}' を選択リストに追加")
                else:
                    #logger.warning(f"MERGE_OBJECTS: オブジェクト '{object_id}' が見つかりません")
                    # 見つからないオブジェクトを記録するが、処理は続行
                    missing_object_ids.append(object_id)

            if len(selected_objects) < 1:
                #logger.warning("MERGE_OBJECTS: 少なくとも1つの有効なオブジェクトが必要です")
                return ""

            # 見つからないオブジェクトがあった場合は警告（ただし処理は続行）
            if missing_object_ids:
                #logger.warning(f"MERGE_OBJECTS: {len(missing_object_ids)}個のオブジェクトが見つかりませんでした（スキップして続行）")
                pass

            # 1個だけの場合はそのまま返す
            if len(selected_objects) == 1:
                #logger.info(f"MERGE_OBJECTS: オブジェクトが1個のみのため、そのまま返します: {selected_objects[0].object_id}")
                return selected_objects[0].object_id

            # 最初のオブジェクトを基準に統合
            base_obj = selected_objects[0]
            merged_pixels = set(base_obj.pixels)

            # pixel_colorsを統合
            merged_pixel_colors = {}
            if hasattr(base_obj, 'pixel_colors') and base_obj.pixel_colors:
                merged_pixel_colors.update(base_obj.pixel_colors)
            else:
                # pixel_colorsがない場合はdominant_colorを使用
                for px, py in base_obj.pixels:
                    merged_pixel_colors[(px, py)] = base_obj.dominant_color

            # 他のオブジェクトのピクセルを統合
            for obj in selected_objects[1:]:
                merged_pixels.update(obj.pixels)

                # pixel_colorsを統合（後のオブジェクトが上書き）
                if hasattr(obj, 'pixel_colors') and obj.pixel_colors:
                    merged_pixel_colors.update(obj.pixel_colors)
                else:
                    for px, py in obj.pixels:
                        if (px, py) not in merged_pixel_colors:
                            merged_pixel_colors[(px, py)] = obj.dominant_color

            # バウンディングボックスを実際のピクセルの範囲に制限
            if merged_pixels:
                xs = [p[0] for p in merged_pixels]
                ys = [p[1] for p in merged_pixels]
                merged_bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                merged_bbox = base_obj.bbox

            # 統合されたオブジェクトを作成
            merged_obj = Object(
                object_id=self._generate_unique_object_id(f"{base_obj.object_id}_merged"),
                pixels=list(merged_pixels),
                pixel_colors=merged_pixel_colors,
                bbox=merged_bbox,  # バウンディングボックスを明示的に設定
                object_type=base_obj.object_type,
            )
            # color_ratioとdominant_colorはpixel_colorsから自動計算される

            # イミュータブル設計: 元のオブジェクトは削除せず、統合オブジェクトを追加
            self.execution_context['objects'][merged_obj.object_id] = merged_obj
            self.execution_context['selected_objects'] = [merged_obj]

            #logger.info(f"MERGE_OBJECTS完了: {len(selected_objects)}個のオブジェクトを統合 → {merged_obj.object_id}")
            return merged_obj.object_id

        except Exception as e:
            #logger.error(f"MERGE_OBJECTS実行エラー: {e}")
            import traceback
            # デバッグ用: エラーの詳細を出力（本番環境ではコメントアウト）
            # traceback.print_exc()
            return ""

    def _get_object_bounds(self, obj: Object) -> Dict[str, int]:
        """オブジェクトの境界を取得"""
        if not obj.pixels:
            return {'min_x': 0, 'max_x': 0, 'min_y': 0, 'max_y': 0}

        x_coords = [p[0] for p in obj.pixels]
        y_coords = [p[1] for p in obj.pixels]

        return {
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords)
        }




    def _expand(self, object_id: str, pixels: int) -> Optional[str]:
        """オブジェクトを指定したピクセル数だけ拡張して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            pixels: 拡張ピクセル数

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            current_pixels = set(obj.pixels)
            current_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

            # 元のピクセルの色を設定
            for px, py in obj.pixels:
                color = obj.pixel_colors.get((px, py), obj.dominant_color) if has_pixel_colors else obj.dominant_color
                current_pixel_colors[(px, py)] = color

            # 指定されたピクセル数だけ拡張
            for _ in range(pixels):
                expanded_pixels = set()
                for x, y in current_pixels:
                    # 元のピクセルの色を取得（拡張元の色を使用）
                    source_color = current_pixel_colors.get((x, y), obj.dominant_color)

                    # 8方向の隣接ピクセルを追加
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = (x + dx, y + dy)
                            expanded_pixels.add(new_pos)
                            # 新規ピクセルは隣接する元ピクセルの色を使用（既に色がある場合は上書きしない）
                            if new_pos not in current_pixel_colors:
                                current_pixel_colors[new_pos] = source_color

                # 新しいピクセルを現在のピクセルに追加
                current_pixels.update(expanded_pixels)

            # 新しいバウンディングボックスを計算
            if current_pixels:
                min_x = min(px for px, py in current_pixels)
                min_y = min(py for px, py in current_pixels)
                max_x = max(px for px, py in current_pixels)
                max_y = max(py for px, py in current_pixels)
                new_bbox = (min_x, min_y, max_x, max_y)
            else:
                new_bbox = obj.bbox

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_expand")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=list(current_pixels),
                pixel_colors=current_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"オブジェクト拡張: {new_id}, 拡張ピクセル数={pixels}, 新しいピクセル数={len(new_obj.pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"オブジェクト拡張エラー: {e}")
            return None

    def _get_position_x(self, object_source: str):
        """X座標を取得（単体オブジェクトのみ）"""
        try:
            # 単一オブジェクトIDとして処理（配列には対応しない）
            obj = self._get_object_by_id(object_source)
            if obj and obj.pixels:
                x_coords = [pixel[0] for pixel in obj.pixels]
                return min(x_coords) if x_coords else 0
                return 0
        except Exception as e:
            #logger.error(f"GET_Xエラー: {e}")
            return 0

    def _get_position_y(self, object_source: str):
        """Y座標を取得（単体オブジェクトのみ）"""
        try:
            # 単一オブジェクトIDとして処理（配列には対応しない）
            obj = self._get_object_by_id(object_source)
            if obj and obj.pixels:
                y_coords = [pixel[1] for pixel in obj.pixels]
                return min(y_coords) if y_coords else 0
                return 0
        except Exception as e:
            #logger.error(f"GET_Yエラー: {e}")
            return 0

    def _create_line(self, x: int, y: int, length: int, direction: str, color: int) -> str:
        """直線を生成（1点と長さと方向を指定）

        Args:
            direction: 方向（"C"の場合は点生成）

        Returns:
            str: 作成したオブジェクトのID
        """
        try:
            # "C"（中央）の場合は点生成（長さ1の点）
            if direction == "C":
                pixels = [(x, y)]
                pixel_colors = {(x, y): color}
                new_obj = Object(
                    object_id=self._generate_unique_object_id(f"point_{x}_{y}_c{color}"),
                    pixels=pixels,
                    pixel_colors=pixel_colors,
                    bbox=(x, y, x + 1, y + 1),
                    object_type="point"
                )
                self.execution_context['objects'][new_obj.object_id] = new_obj
                return new_obj.object_id

            # 方向に基づいて終点を計算（v3.1: XY8種統一）
            direction_map = {
                # 4方向（正負）
                'X': (1, 0),        # 右（0度、X+）
                '-X': (-1, 0),      # 左（180度、X-）
                'Y': (0, 1),        # 下（90度、Y+）
                '-Y': (0, -1),      # 上（270度、Y-）
                # 対角線4方向
                'XY': (1, 1),       # 右下（45度、X+Y+）
                'X-Y': (1, -1),     # 右上（-45度、315度、X+Y-）
                '-XY': (-1, 1),     # 左下（135度、X-Y+）
                '-X-Y': (-1, -1),   # 左上（225度、X-Y-）
            }

            if direction not in direction_map:
                #logger.error(f"無効な方向: {direction}。使用可能: X, Y, -X, -Y, XY, -XY, X-Y, -X-Y, C")
                return ""

            dx, dy = direction_map[direction]
            x2 = x + dx * (length - 1)
            y2 = y + dy * (length - 1)

            # ブレゼンハムの線描画アルゴリズム
            pixels = []
            abs_dx = abs(x2 - x)
            abs_dy = abs(y2 - y)
            sx = 1 if x < x2 else -1
            sy = 1 if y < y2 else -1
            err = abs_dx - abs_dy

            current_x, current_y = x, y
            while True:
                pixels.append((current_x, current_y))
                if current_x == x2 and current_y == y2:
                    break

                e2 = 2 * err
                if e2 > -abs_dy:
                    err -= abs_dy
                    current_x += sx
                if e2 < abs_dx:
                    err += abs_dx
                    current_y += sy

            # pixel_colors: すべて同じ色
            pixel_colors = {(px, py): color for px, py in pixels}

            new_obj = Object(
                object_id=self._generate_unique_object_id(f"line_{x}_{y}_{direction}_l{length}_c{color}"),
                pixels=pixels,
                pixel_colors=pixel_colors,
                object_type=ObjectType.CREATED,
                _dominant_color=color
            )
            self.execution_context['objects'][new_obj.object_id] = new_obj
            #logger.info(f"直線生成完了: ({x}, {y}) 方向={direction} 長さ={length}, 色={color}, ID={new_obj.object_id}")
            return new_obj.object_id

        except Exception as e:
            #logger.error(f"直線生成エラー: {e}")
            return ""


    def _create_rectangle(self, x: int, y: int, width: int, height: int, color: int) -> str:
        """矩形を生成

        Returns:
            str: 作成したオブジェクトのID
        """
        try:
            pixels = []
            for py in range(y, y + height):
                for px in range(x, x + width):
                    pixels.append((px, py))

            # pixel_colors: すべて同じ色
            pixel_colors = {(px, py): color for px, py in pixels}

            new_obj = Object(
                object_id=self._generate_unique_object_id(f"rectangle_{x}_{y}_{width}x{height}_c{color}"),
                pixels=pixels,
                pixel_colors=pixel_colors,
                object_type=ObjectType.CREATED,
                _dominant_color=color
            )
            self.execution_context['objects'][new_obj.object_id] = new_obj
            #logger.info(f"矩形生成完了: ({x}, {y}), サイズ={width}x{height}, 色={color}, ID={new_obj.object_id}")
            return new_obj.object_id

        except Exception as e:
            #logger.error(f"矩形生成エラー: {e}")
            return ""


    def _detect_outline_pixels(self, pixels: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """オブジェクトの輪郭ピクセルを検出"""
        if not pixels:
            return []

        pixel_set = set(pixels)
        outline_pixels = []

        # 4方向の近傍をチェック
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for x, y in pixels:
            is_outline = False
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor not in pixel_set:
                    is_outline = True
                    break

            if is_outline:
                outline_pixels.append((x, y))

        return outline_pixels


    def _split_by_connection(self, obj: Object, connectivity: int) -> List[str]:
        """連結成分でオブジェクトを分割（イミュータブル）

        Args:
            obj: 分割対象のオブジェクト
            connectivity: 連結性（4または8）

        Returns:
            List[str]: [元のオブジェクトID, 分割された新オブジェクトID...]
                      分割されなかった場合は [元のオブジェクトID]
        """
        try:
            #logger.info(f"連結分割開始: {obj.object_id}, 連結性={connectivity}")

            # 常に元のオブジェクトIDを先頭に含める
            result_ids = [obj.object_id]

            # ピクセルを座標セットに変換
            pixel_set = set(obj.pixels)
            visited = set()
            components = []

            # 各ピクセルから連結成分を抽出
            for pixel in obj.pixels:
                if pixel in visited:
                    continue

                # BFSで連結成分を抽出
                component = []
                queue = [pixel]
                visited.add(pixel)

                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    py, px = current

                    # 隣接ピクセルをチェック
                    if connectivity == 4:
                        neighbors = [(py-1, px), (py+1, px), (py, px-1), (py, px+1)]
                    else:  # 8連結
                        neighbors = [(py+dy, px+dx) for dy in [-1,0,1] for dx in [-1,0,1] if not (dy==0 and dx==0)]

                    for neighbor in neighbors:
                        if neighbor in pixel_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                components.append(component)

            # 連結成分が1つしかない場合は分割しない
            if len(components) <= 1:
                #logger.info(f"連結成分が1つのため分割しません: {obj.object_id}")
                return result_ids  # 元のオブジェクトIDのみ返す

            #logger.info(f"連結成分数: {len(components)}")

            # 各連結成分ごとに新しいオブジェクトを作成（元のオブジェクトは削除しない）
            for i, component_pixels in enumerate(components):
                # pixel_colorsを抽出
                new_pixel_colors = {}
                has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors
                for px, py in component_pixels:
                    if has_pixel_colors and (px, py) in obj.pixel_colors:
                        new_pixel_colors[(px, py)] = obj.pixel_colors[(px, py)]
                    else:
                        new_pixel_colors[(px, py)] = obj.dominant_color

                new_obj_id = self._generate_unique_object_id(f"{obj.object_id}_comp{i}")
                new_obj = Object(
                    object_id=new_obj_id,
                    object_type=obj.object_type,
                    pixels=component_pixels,
                    pixel_colors=new_pixel_colors,
                    source_image_id=obj.source_image_id,
                    source_image_type=obj.source_image_type,
                    source_image_index=obj.source_image_index,
                )
                # dominant_colorとcolor_ratioはpixel_colorsから自動計算される
                if hasattr(obj, '_grid'):
                    new_obj._grid = obj._grid

                self.execution_context['objects'][new_obj.object_id] = new_obj
                result_ids.append(new_obj.object_id)
                #logger.info(f"連結成分オブジェクトを作成: {new_obj.object_id}, {len(component_pixels)}ピクセル")

            #logger.info(f"連結分割完了: {obj.object_id} → [元 + {len(result_ids)-1}個の新オブジェクト] = 計{len(result_ids)}個")
            return result_ids

        except Exception as e:
            #logger.error(f"連結分割エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []



    def _crop(self, obj: Object, x: int, y: int, width: int, height: int) -> str:
        """指定した矩形範囲を切り取る

        Args:
            obj: 切り取り対象のオブジェクト
            x, y: 矩形の左上座標
            width, height: 矩形のサイズ

        Returns:
            str: 矩形範囲内のピクセルのみを含む新しいオブジェクトID
        """
        try:
            #logger.info(f"CROP開始: {obj.object_id}, rect=({x},{y},{width},{height})")

            if not obj.pixels:
                #logger.info("ピクセルが空のため切り取りできません")
                return obj.object_id

            # 矩形範囲を定義
            rect_x_min, rect_x_max = x, x + width - 1
            rect_y_min, rect_y_max = y, y + height - 1

            # 範囲内のピクセルのみを抽出
            inside_pixels = []
            pixel_colors = {}

            for pixel in obj.pixels:
                if len(pixel) == 3:
                    px, py, color = pixel
                else:
                    px, py = pixel
                    # pixel_colorsがある場合はそこから取得
                    if hasattr(obj, 'pixel_colors') and obj.pixel_colors and (px, py) in obj.pixel_colors:
                        color = obj.pixel_colors[(px, py)]
                    else:
                        color = obj.dominant_color

                # 矩形範囲内かチェック
                if rect_x_min <= px <= rect_x_max and rect_y_min <= py <= rect_y_max:
                    inside_pixels.append((px, py))
                    pixel_colors[(px, py)] = color

            # 切り取られたオブジェクトを作成
            if inside_pixels:
                cropped_id = self._generate_unique_object_id(f"{obj.object_id}_crop")
                cropped_obj = Object(
                    object_id=cropped_id,
                    object_type=obj.object_type,
                    pixels=inside_pixels,
                    pixel_colors=pixel_colors,
                    source_image_id=obj.source_image_id,
                    source_image_type=obj.source_image_type,
                    source_image_index=obj.source_image_index,
                )
                # dominant_colorはpixel_colorsから自動計算される
                if hasattr(obj, '_grid'):
                    cropped_obj._grid = obj._grid
                self.execution_context['objects'][cropped_obj.object_id] = cropped_obj
                #logger.info(f"CROPオブジェクト作成: {cropped_id} ({len(inside_pixels)}ピクセル)")
                return cropped_id
            else:
                # 範囲内にピクセルがない場合はNoneを返す
                #logger.warning(f"CROP: 範囲内にピクセルがありません")
                return None

        except Exception as e:
            #logger.error(f"CROPエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return obj.object_id

    def _arrange_grid(self, object_ids: List[str], columns: int, cell_width: int, cell_height: int) -> List[str]:
        """オブジェクトをグリッド状に配置

        Args:
            object_ids: 配置するオブジェクトIDの配列
            columns: 列数（横方向の個数）
            cell_width: 1つ当たりのセル幅
            cell_height: 1つ当たりのセル高さ

        Returns:
            List[str]: 配置後の新しいオブジェクトIDの配列（元の配列の順番を保持）
        """
        try:
            #logger.info(f"ARRANGE_GRID開始: {len(object_ids)}個のオブジェクト, columns={columns}, cell={cell_width}x{cell_height}")

            if not object_ids:
                #logger.warning("配列が空です")
                return []

            if columns <= 0:
                #logger.error(f"列数は1以上である必要があります: {columns}")
                return object_ids

            arranged_ids = []

            for i, obj_id in enumerate(object_ids):
                obj = self._find_object_by_id(obj_id)
                if not obj:
                    #logger.warning(f"オブジェクトが見つかりません: {obj_id}")
                    arranged_ids.append(obj_id)
                    continue

                # グリッド位置を計算
                # row = i // columns
                row = i // columns
                # col = i % columns
                col = i - row * columns

                # セル内の配置座標
                x = col * cell_width
                y = row * cell_height

                # オブジェクトを配置（TELEPORTを使用）
                new_obj_id = self._teleport(obj_id, x, y)
                arranged_ids.append(new_obj_id)

            #logger.info(f"ARRANGE_GRID完了: {len(arranged_ids)}個のオブジェクトを配置")
            return arranged_ids

        except Exception as e:
            #logger.error(f"ARRANGE_GRIDエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return object_ids

    def _tile(self, object_id: str, count_x: int, count_y: int) -> List[str]:
        """オブジェクトをタイル状に複製して配置

        Args:
            object_id: 複製元のオブジェクトID
            count_x: X方向の複製数
            count_y: Y方向の複製数

        Returns:
            List[str]: 配置後のオブジェクトIDの配列
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return []

            if count_x <= 0 or count_y <= 0:
                return []

            # オブジェクトのサイズを取得
            obj_x = self._get_position_x(object_id)
            obj_y = self._get_position_y(object_id)
            obj_width = self._get_bbox_width(object_id)
            obj_height = self._get_bbox_height(object_id)

            tiled_ids = []

            # タイル状に複製
            for j in range(count_y):
                for i in range(count_x):
                    # 配置位置を計算
                    offset_x = i * obj_width
                    offset_y = j * obj_height
                    new_x = obj_x + offset_x
                    new_y = obj_y + offset_y

                    # オブジェクトを複製して配置（TELEPORTを使用）
                    tiled_id = self._teleport(object_id, new_x, new_y)
                    if tiled_id:
                        tiled_ids.append(tiled_id)

            return tiled_ids

        except Exception as e:
            #logger.error(f"タイルエラー: {e}")
            return []

    def _match_pairs(self, objects1: List[str], objects2: List[str],
                     condition_func, context: dict = None) -> List[str]:
        """ペアマッチング（選択肢少ない順戦略）

        Args:
            objects1: マッチング元のオブジェクトID配列
            objects2: マッチング先のオブジェクトID配列
            condition_func: マッチング条件（AST式）
            context: 評価コンテキスト

        Returns:
            List[str]: ペアのフラット配列 [obj1_a, obj2_x, obj1_b, obj2_y, ...]
        """
        try:
            #logger.info(f"MATCH_PAIRS開始: objects1={len(objects1)}個, objects2={len(objects2)}個")

            if not objects1 or not objects2:
                #logger.info("配列が空のためマッチングできません")
                return []

            # インタープリターを取得（条件式の評価のため）
            from src.core_systems.executor.parsing.interpreter import Interpreter
            interpreter = Interpreter(self)

            # ステップ1: 各objects1のマッチ候補数をカウント
            match_counts = []
            for i, obj1_id in enumerate(objects1):
                count = 0
                for obj2_id in objects2:
                    # 条件式を評価（$obj1, $obj2を置き換え）
                    temp_vars = context.copy() if context else {}
                    temp_vars['obj1'] = obj1_id
                    temp_vars['obj2'] = obj2_id

                    # 条件式を評価
                    old_vars = interpreter.variables.copy()
                    interpreter.variables.update(temp_vars)

                    try:
                        result = interpreter.evaluate_expression(condition_func)
                        if result:
                            count += 1
                    except Exception as e:
                        #logger.warning(f"条件評価エラー: {e}")
                        pass
                    finally:
                        interpreter.variables = old_vars

                match_counts.append(count)
                #logger.debug(f"objects1[{i}]のマッチ候補数: {count}")

            # ステップ2: 選択肢少ない順でマッチング
            paired = []
            used2 = [False] * len(objects2)
            processed1 = [False] * len(objects1)

            # 最大選択肢数を取得
            max_choices = max(match_counts) if match_counts else 0

            # 選択肢が少ない順に処理（1個→2個→3個以上）
            for threshold in range(1, max_choices + 1):
                for i, obj1_id in enumerate(objects1):
                    if processed1[i]:
                        continue

                    # 閾値チェック
                    if threshold < 3:
                        # パス1-2: 選択肢が1個または2個
                        if match_counts[i] != threshold:
                            continue
                    else:
                        # パス3以降: 選択肢が3個以上
                        if match_counts[i] < 3:
                            continue

                    # 距離ベースの貪欲法でマッチング
                    best_j = -1
                    best_dist = float('inf')

                    for j, obj2_id in enumerate(objects2):
                        if used2[j]:
                            continue

                        # 条件チェック
                        temp_vars = context.copy() if context else {}
                        temp_vars['obj1'] = obj1_id
                        temp_vars['obj2'] = obj2_id

                        old_vars = interpreter.variables.copy()
                        interpreter.variables.update(temp_vars)

                        try:
                            result = interpreter.evaluate_expression(condition_func)
                            if not result:
                                continue
                        except Exception as e:
                            #logger.warning(f"条件評価エラー: {e}")
                            continue
                        finally:
                            interpreter.variables = old_vars

                        # 距離計算
                        dist = self._get_distance_between_objects(obj1_id, obj2_id)
                        if dist < best_dist:
                            best_dist = dist
                            best_j = j

                    # マッチング成功
                    if best_j >= 0:
                        paired.extend([obj1_id, objects2[best_j]])
                        used2[best_j] = True
                        processed1[i] = True
                        #logger.debug(f"ペア成立: {obj1_id} - {objects2[best_j]} (距離={best_dist:.2f})")

            #logger.info(f"MATCH_PAIRS完了: {len(paired) // 2}ペア作成")
            return paired

        except Exception as e:
            #logger.error(f"MATCH_PAIRSエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []

    def _extract_rectangles(self, obj: Object) -> List[str]:
        """オブジェクトから矩形パターンを自動抽出

        各色ごとに独立して矩形を検出します。

        Args:
            obj: 分割対象のオブジェクト

        Returns:
            List[str]: 抽出された矩形オブジェクトIDのリスト
        """
        try:
            #logger.info(f"矩形抽出開始: {obj.object_id}")

            if not obj.pixels:
                #logger.info("ピクセルが空のため矩形抽出できません")
                return []

            # ステップ1: ピクセルを色ごとにグループ化
            pixels_by_color = self._group_pixels_by_color(obj)
            #logger.info(f"色の数: {len(pixels_by_color)}色")

            result_ids = []

            # 背景色を取得
            background_color = self.execution_context.get('background_color', 0)

            # ステップ2: 各色ごとに矩形を検出（背景色は除外）
            for color, color_pixels in pixels_by_color.items():
                # 背景色の矩形はスキップ
                if color == background_color:
                    #logger.info(f"色{color}（背景色）の矩形検出はスキップします")
                    continue

                #logger.info(f"色{color}の矩形検出開始: {len(color_pixels)}ピクセル")
                rectangles = self._detect_rectangles_for_color(
                    color_pixels, color, obj
                )
                result_ids.extend(rectangles)
                #logger.info(f"色{color}の矩形: {len(rectangles)}個検出")

            #logger.info(f"矩形抽出完了: 合計{len(result_ids)}個の矩形を抽出")
            return result_ids

        except Exception as e:
            #logger.error(f"矩形抽出エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []

    def _detect_rectangles_for_color(self, color_pixels: list, color: int, obj: Object) -> List[str]:
        """特定の色のピクセルから矩形を検出

        Args:
            color_pixels: この色のピクセルのリスト
            color: 色の値
            obj: 元のオブジェクト（メタデータ用）

        Returns:
            List[str]: 抽出された矩形オブジェクトのIDリスト
        """
        try:
            # この色のピクセルセット
            pixel_set = set(color_pixels)

            if not pixel_set:
                return []

            # bboxを取得（この色のピクセルから計算）
            xs = [x for x, y in color_pixels]
            ys = [y for x, y in color_pixels]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # グリッドを作成（0: 空, 1: オブジェクトのピクセル）
            grid_width = max_x - min_x + 1
            grid_height = max_y - min_y + 1
            grid = np.zeros((grid_height, grid_width), dtype=int)

            for px, py in color_pixels:
                grid[py - min_y, px - min_x] = 1

            # 新アルゴリズム: 全候補から最適な組み合わせを選択

            # ステップ1: 全ての矩形候補を列挙
            all_candidates = []
            for y in range(grid_height):
                for x in range(grid_width):
                    if grid[y, x] == 1:
                        # この位置から可能な全ての矩形を探索
                        for h in range(1, grid_height - y + 1):
                            for w in range(1, grid_width - x + 1):
                                # この矩形が有効か確認
                                valid = True
                                for dy in range(h):
                                    for dx in range(w):
                                        if y + dy >= grid_height or x + dx >= grid_width or grid[y + dy, x + dx] != 1:
                                            valid = False
                                            break
                                    if not valid:
                                        break

                                if valid and w >= 2 and h >= 2:
                                    area = w * h
                                    # (x, y, width, height, area, pixels_set)
                                    pixels_set = set()
                                    for dy in range(h):
                                        for dx in range(w):
                                            pixels_set.add((x + dx, y + dy))
                                    all_candidates.append((x, y, w, h, area, pixels_set))

            # ステップ2: 面積順にソート（大きい順）
            all_candidates.sort(key=lambda r: r[4], reverse=True)

            #logger.info(f"矩形候補数: {len(all_candidates)}個")

            # ステップ3: 重複しない矩形の最適な組み合わせを選択（貪欲法）
            # ルール1: 同じ位置・形状の矩形は重複抽出禁止
            # ルール2: 抽出する矩形は、他の抽出する矩形と重複していないピクセルを1つ以上含む必要がある
            # ルール3（新規）: 矩形の横幅の両端または縦幅の両端に、その矩形と異なる色のピクセルが1行/列分でもある
            selected_rects = []
            usage_count = {}  # {(x, y): count} - 使用回数制限なし
            extracted_rects = set()  # (x, y, width, height) のセット

            for rect in all_candidates:
                rx, ry, rw, rh, area, pixels_set = rect
                rect_signature = (rx, ry, rw, rh)

                # ルール1チェック: 同じ矩形が既に抽出されていないか
                if rect_signature in extracted_rects:
                    continue

                # ルール2チェック: この矩形が、他の矩形と重複していないピクセルを1つ以上含むか確認
                has_unique_pixel = False
                for pixel in pixels_set:
                    if usage_count.get(pixel, 0) == 0:  # まだ使用されていないピクセル
                        has_unique_pixel = True
                        break

                if not has_unique_pixel:
                    #logger.debug(f"矩形スキップ(ルール2違反): pos=({rx},{ry}), size=({rw}x{rh}), 全ピクセルが既に使用済み")
                    continue

                # ルール3チェック: 横幅または縦幅の隣接する両端に1ピクセルずつ異なる色があるか
                # 異なる色の定義: 空（ピクセルなし） OR オブジェクトと異なる色のピクセル
                has_edge_contrast = False

                # 横幅の両端チェック（左端と右端の隣接ピクセル）
                # 1行分でも、左隣と右隣の両方に異なる色があればOK
                for y_check in range(ry, ry + rh):
                    # 左端の隣接ピクセル（矩形の外側）
                    left_neighbor_x = rx - 1
                    # 右端の隣接ピクセル（矩形の外側）
                    right_neighbor_x = rx + rw

                    # 左隣に異なる色がある（範囲外 OR 空 OR オブジェクトのピクセルではない）
                    left_has_different = (left_neighbor_x < 0 or left_neighbor_x >= grid_width or
                                         grid[y_check, left_neighbor_x] != 1)
                    # 右隣に異なる色がある
                    right_has_different = (right_neighbor_x < 0 or right_neighbor_x >= grid_width or
                                          grid[y_check, right_neighbor_x] != 1)

                    # 両方の隣接ピクセルが異なる色
                    if left_has_different and right_has_different:
                        has_edge_contrast = True
                        break

                # 縦幅の両端チェック（上端と下端の隣接ピクセル）
                # 1列分でも、上隣と下隣の両方に異なる色があればOK
                if not has_edge_contrast:
                    for x_check in range(rx, rx + rw):
                        # 上端の隣接ピクセル（矩形の外側）
                        top_neighbor_y = ry - 1
                        # 下端の隣接ピクセル（矩形の外側）
                        bottom_neighbor_y = ry + rh

                        # 上隣に異なる色がある（範囲外 OR 空 OR オブジェクトのピクセルではない）
                        top_has_different = (top_neighbor_y < 0 or top_neighbor_y >= grid_height or
                                            grid[top_neighbor_y, x_check] != 1)
                        # 下隣に異なる色がある
                        bottom_has_different = (bottom_neighbor_y < 0 or bottom_neighbor_y >= grid_height or
                                               grid[bottom_neighbor_y, x_check] != 1)

                        # 両方の隣接ピクセルが異なる色
                        if top_has_different and bottom_has_different:
                            has_edge_contrast = True
                            break

                if not has_edge_contrast:
                    #logger.debug(f"矩形スキップ(ルール3違反): pos=({rx},{ry}), size=({rw}x{rh}), 両端に異なる色がない")
                    continue

                # すべてのルールをパス
                selected_rects.append((rx, ry, rw, rh))
                # 使用カウントを更新（無制限）
                for pixel in pixels_set:
                    usage_count[pixel] = usage_count.get(pixel, 0) + 1
                extracted_rects.add(rect_signature)
                #logger.info(f"矩形選択: pos=({rx},{ry}), size=({rw}x{rh}), area={area}")

            #logger.info(f"選択された矩形: {len(selected_rects)}個")

            # ステップ4: 選択された矩形からオブジェクトを生成
            result_ids = []
            for idx, (rect_x, rect_y, rect_width, rect_height) in enumerate(selected_rects):
                # 矩形のピクセルを抽出
                rect_pixels = []
                for dy in range(rect_height):
                    for dx in range(rect_width):
                        gx, gy = rect_x + dx, rect_y + dy
                        if grid[gy, gx] == 1:
                            # 元の座標系に戻す
                            orig_x = gx + min_x
                            orig_y = gy + min_y
                            rect_pixels.append((orig_x, orig_y))

                # 新しい矩形オブジェクトを作成
                if rect_pixels:
                    new_obj_id = self._generate_unique_object_id(f"{obj.object_id}_rect")

                    # この色のピクセルなので、すべて同じ色
                    # pixel_colorsを設定（すべて同じ色）
                    new_pixel_colors = {}
                    for px, py in rect_pixels:
                        new_pixel_colors[(px, py)] = color

                    new_obj = Object(
                        object_id=new_obj_id,
                        object_type=obj.object_type,
                        pixels=rect_pixels,
                        pixel_colors=new_pixel_colors,
                        source_image_id=obj.source_image_id,
                        source_image_type=obj.source_image_type,
                        source_image_index=obj.source_image_index
                    )
                    # dominant_colorとcolor_ratioはpixel_colorsから自動計算される
                    if hasattr(obj, '_grid'):
                        new_obj._grid = obj._grid
                    self.execution_context['objects'][new_obj.object_id] = new_obj
                    result_ids.append(new_obj.object_id)
                    #logger.info(f"矩形オブジェクト作成: {new_obj_id} ({len(rect_pixels)}ピクセル)")

            #logger.info(f"色{color}の矩形抽出完了: {len(result_ids)}個")
            return result_ids

        except Exception as e:
            #logger.error(f"色{color}の矩形抽出エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []

    def _extract_hollow_rectangles(self, obj: Object) -> List[str]:
        """オブジェクトから中空矩形パターンを自動抽出

        中空矩形の定義:
        - 4辺すべてが存在（上下左右）
        - 各辺の厚さ1ピクセル
        - 内部は空（ピクセルがない）
        - 最小サイズ: 3x3

        各色ごとに独立して中空矩形を検出します。

        Returns:
            List[str]: 抽出された中空矩形オブジェクトのIDリスト
        """
        try:
            #logger.info(f"中空矩形抽出開始: {obj.object_id}")

            if not obj.pixels:
                #logger.info("ピクセルが空のため抽出できません")
                return []

            # ステップ1: ピクセルを色ごとにグループ化
            pixels_by_color = self._group_pixels_by_color(obj)
            #logger.info(f"色の数: {len(pixels_by_color)}色")

            result_ids = []

            # 背景色を取得
            background_color = self.execution_context.get('background_color', 0)

            # ステップ2: 各色ごとに中空矩形を検出（背景色は除外）
            for color, color_pixels in pixels_by_color.items():
                # 背景色の中空矩形はスキップ
                if color == background_color:
                    #logger.info(f"色{color}（背景色）の中空矩形検出はスキップします")
                    continue

                #logger.info(f"色{color}の中空矩形検出開始: {len(color_pixels)}ピクセル")
                hollow_rects = self._detect_hollow_rectangles_for_color(
                    color_pixels, color, obj
                )
                result_ids.extend(hollow_rects)
                #logger.info(f"色{color}の中空矩形: {len(hollow_rects)}個検出")

            #logger.info(f"中空矩形抽出完了: 合計{len(result_ids)}個の中空矩形を抽出")
            return result_ids

        except Exception as e:
            #logger.error(f"中空矩形抽出エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []

    def _group_pixels_by_color(self, obj: Object) -> dict:
        """オブジェクトのピクセルを色ごとにグループ化

        Args:
            obj: オブジェクト

        Returns:
            dict: {color: [(x, y), ...]}
        """
        pixels_by_color = {}
        has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

        for px, py in obj.pixels:
            if has_pixel_colors and (px, py) in obj.pixel_colors:
                color = obj.pixel_colors[(px, py)]
            else:
                color = obj.dominant_color

            if color not in pixels_by_color:
                pixels_by_color[color] = []
            pixels_by_color[color].append((px, py))

        return pixels_by_color

    def _detect_hollow_rectangles_for_color(self, color_pixels: list, color: int, obj: Object) -> List[str]:
        """特定の色のピクセルから中空矩形を検出

        Args:
            color_pixels: この色のピクセルのリスト
            color: 色の値
            obj: 元のオブジェクト（メタデータ用）

        Returns:
            List[str]: 抽出された中空矩形オブジェクトのIDリスト
        """
        try:
            # この色のピクセルセット
            pixel_set = set(color_pixels)

            if not pixel_set:
                return []

            # bboxを取得（この色のピクセルから計算）
            min_x = min(x for x, y in color_pixels)
            max_x = max(x for x, y in color_pixels)
            min_y = min(y for x, y in color_pixels)
            max_y = max(y for x, y in color_pixels)

            width = max_x - min_x + 1
            height = max_y - min_y + 1

            #logger.info(f"色{color}の範囲: ({min_x},{min_y}) - ({max_x},{max_y}), size=({width}x{height})")

            # ステップ1: 候補の中空矩形を列挙
            all_candidates = []

            for y in range(height):
                for x in range(width):
                    for h in range(3, height - y + 1):  # 最小3x3
                        for w in range(3, width - x + 1):
                            # 実際の座標
                            rect_x = min_x + x
                            rect_y = min_y + y

                            # 中空矩形かチェック
                            is_hollow = True

                            # 4辺がすべて存在するかチェック
                            # 上辺
                            for dx in range(w):
                                if (rect_x + dx, rect_y) not in pixel_set:
                                    is_hollow = False
                                    break

                            if not is_hollow:
                                continue

                            # 下辺
                            for dx in range(w):
                                if (rect_x + dx, rect_y + h - 1) not in pixel_set:
                                    is_hollow = False
                                    break

                            if not is_hollow:
                                continue

                            # 左辺
                            for dy in range(h):
                                if (rect_x, rect_y + dy) not in pixel_set:
                                    is_hollow = False
                                    break

                            if not is_hollow:
                                continue

                            # 右辺
                            for dy in range(h):
                                if (rect_x + w - 1, rect_y + dy) not in pixel_set:
                                    is_hollow = False
                                    break

                            if not is_hollow:
                                continue

                            # 内部チェック: 新ルール
                            # ルール①: 1層目の「異なる色」の割合が30%以上
                            # ルール②: 内部全体の「異なる色」の割合が30%以上
                            # ルール③: 1層目の各辺すべてに最低1ピクセルの異なる色
                            # すべてを満たす必要がある（AND条件）

                            # 内部のサイズ
                            inner_w = w - 2
                            inner_h = h - 2

                            if inner_w <= 0 or inner_h <= 0:
                                # 内部がない（3x3未満は既に除外済み）
                                continue

                            # 1層目の定義: 枠線のすぐ内側の1ピクセル分（枠線状の層）
                            # 各辺を個別に収集（ルール③で使用）
                            layer1_top = []     # 上辺の内側
                            layer1_bottom = []  # 下辺の内側
                            layer1_left = []    # 左辺の内側
                            layer1_right = []   # 右辺の内側

                            # 上辺の内側
                            for dx in range(1, w - 1):
                                layer1_top.append((rect_x + dx, rect_y + 1))
                            # 下辺の内側
                            for dx in range(1, w - 1):
                                layer1_bottom.append((rect_x + dx, rect_y + h - 2))
                            # 左辺の内側（角は除く）
                            for dy in range(2, h - 2):
                                layer1_left.append((rect_x + 1, rect_y + dy))
                            # 右辺の内側（角は除く）
                            for dy in range(2, h - 2):
                                layer1_right.append((rect_x + w - 2, rect_y + dy))

                            # 1層目の全ピクセル（setで重複除去）
                            layer1_pixels = set(layer1_top + layer1_bottom + layer1_left + layer1_right)

                            # ルール①: 1層目で「この色のpixel_setに存在しない」ピクセルの割合
                            # シンプルに「存在しない = 空」と判定
                            layer1_total = len(layer1_pixels)
                            layer1_different = sum(1 for p in layer1_pixels if p not in pixel_set)
                            layer1_ratio = layer1_different / layer1_total if layer1_total > 0 else 0

                            if layer1_ratio < 0.3:
                                # 1層目の異なる色が30%未満 → 失格
                                is_hollow = False
                                continue

                            # ルール②: 内部全体で「この色のpixel_setに存在しない」ピクセルの割合
                            # シンプルに「存在しない = 空」と判定
                            interior_total = inner_w * inner_h
                            interior_different = 0
                            for dy in range(1, h - 1):
                                for dx in range(1, w - 1):
                                    if (rect_x + dx, rect_y + dy) not in pixel_set:
                                        interior_different += 1

                            interior_ratio = interior_different / interior_total if interior_total > 0 else 0

                            if interior_ratio < 0.3:
                                # 内部全体の異なる色が30%未満 → 失格
                                is_hollow = False
                                continue

                            # ルール③: 1層目の各辺に最低1ピクセルの「この色のpixel_setに存在しない」ピクセル
                            # シンプルに「存在しない = 空」と判定
                            # 上辺
                            top_has_different = any(p not in pixel_set for p in layer1_top) if layer1_top else True
                            # 下辺
                            bottom_has_different = any(p not in pixel_set for p in layer1_bottom) if layer1_bottom else True
                            # 左辺
                            left_has_different = any(p not in pixel_set for p in layer1_left) if layer1_left else True
                            # 右辺
                            right_has_different = any(p not in pixel_set for p in layer1_right) if layer1_right else True

                            if not (top_has_different and bottom_has_different and left_has_different and right_has_different):
                                # いずれかの辺に異なる色がない → 失格
                                is_hollow = False

                            if is_hollow:
                                # 枠線のピクセルを収集
                                frame_pixels = set()
                                # 上辺
                                for dx in range(w):
                                    frame_pixels.add((rect_x + dx, rect_y))
                                # 下辺
                                for dx in range(w):
                                    frame_pixels.add((rect_x + dx, rect_y + h - 1))
                                # 左辺
                                for dy in range(1, h - 1):  # 角は既に追加済み
                                    frame_pixels.add((rect_x, rect_y + dy))
                                # 右辺
                                for dy in range(1, h - 1):  # 角は既に追加済み
                                    frame_pixels.add((rect_x + w - 1, rect_y + dy))

                                perimeter = 2 * (w + h) - 4  # 周囲長
                                all_candidates.append((rect_x, rect_y, w, h, perimeter, frame_pixels))

            # ステップ2: 周囲長順にソート（大きい順）
            all_candidates.sort(key=lambda r: r[4], reverse=True)

            #logger.info(f"中空矩形候補数: {len(all_candidates)}個")

            # ステップ3: 重複しない矩形の最適な組み合わせを選択（貪欲法）
            # ルール1: 同じ位置・形状の矩形は重複抽出禁止
            # ルール2: 抽出する矩形は、他の矩形と重複していないピクセルを1つ以上含む必要がある
            # ルール3（新規）: 横幅または縦幅の隣接両端に異なる色が1行/列分でもある
            selected_rects = []
            usage_count = {}  # {(x, y): count} - 使用回数制限なし
            extracted_rects = set()

            for rect in all_candidates:
                rx, ry, rw, rh, perimeter, frame_pixels = rect
                rect_signature = (rx, ry, rw, rh)

                # ルール1チェック: 同じ矩形が既に抽出されていないか
                if rect_signature in extracted_rects:
                    continue

                # ルール2チェック: この矩形が、他の矩形と重複していないピクセルを1つ以上含むか確認
                has_unique_pixel = False
                for pixel in frame_pixels:
                    if usage_count.get(pixel, 0) == 0:  # まだ使用されていないピクセル
                        has_unique_pixel = True
                        break

                if not has_unique_pixel:
                    #logger.debug(f"中空矩形スキップ(ルール2違反): pos=({rx},{ry}), size=({rw}x{rh}), 全ピクセルが既に使用済み")
                    continue

                # ルール3チェック: 横幅または縦幅の隣接両端に異なる色があるか
                # 異なる色の定義: 空（ピクセルなし） OR オブジェクトと異なる色のピクセル
                has_edge_contrast = False

                # 横幅の両端チェック（左端と右端の隣接ピクセル）
                for y_check in range(ry, ry + rh):
                    # 左端の隣接ピクセル
                    left_neighbor = (rx - 1, y_check)
                    # 右端の隣接ピクセル
                    right_neighbor = (rx + rw, y_check)

                    # 左隣が異なる色か（pixel_setに存在しない = 空 OR 異なる色）
                    left_is_different = left_neighbor not in pixel_set
                    # 右隣が異なる色か
                    right_is_different = right_neighbor not in pixel_set

                    if left_is_different and right_is_different:
                        has_edge_contrast = True
                        break

                # 縦幅の両端チェック（上端と下端の隣接ピクセル）
                if not has_edge_contrast:
                    for x_check in range(rx, rx + rw):
                        # 上端の隣接ピクセル
                        top_neighbor = (x_check, ry - 1)
                        # 下端の隣接ピクセル
                        bottom_neighbor = (x_check, ry + rh)

                        # 上隣が異なる色か
                        top_is_different = top_neighbor not in pixel_set
                        # 下隣が異なる色か
                        bottom_is_different = bottom_neighbor not in pixel_set

                        if top_is_different and bottom_is_different:
                            has_edge_contrast = True
                            break

                if not has_edge_contrast:
                    #logger.debug(f"中空矩形スキップ(ルール3違反): pos=({rx},{ry}), size=({rw}x{rh}), 両端に異なる色がない")
                    continue

                # すべてのルールをパス
                selected_rects.append((rx, ry, rw, rh, frame_pixels))
                # 使用カウントを更新（無制限）
                for pixel in frame_pixels:
                    usage_count[pixel] = usage_count.get(pixel, 0) + 1
                extracted_rects.add(rect_signature)
                #logger.info(f"中空矩形選択: pos=({rx},{ry}), size=({rw}x{rh}), perimeter={perimeter}")

            #logger.info(f"色{color}の選択された中空矩形: {len(selected_rects)}個")

            # ステップ4: 選択された中空矩形からオブジェクトを生成
            result_ids = []
            for idx, (rect_x, rect_y, rect_width, rect_height, frame_pixels) in enumerate(selected_rects):
                # 新しい中空矩形オブジェクトを作成
                if frame_pixels:
                    new_obj_id = self._generate_unique_object_id(f"{obj.object_id}_hollow_rect")

                    # この色のピクセルなので、すべて同じ色
                    # pixel_colorsを設定（すべて同じ色）
                    new_pixel_colors = {}
                    for px, py in frame_pixels:
                        new_pixel_colors[(px, py)] = color

                    # レイヤー調整: 重複するオブジェクトが同じレイヤーに存在しないようにする
                    new_obj = Object(
                        object_id=new_obj_id,
                        object_type=obj.object_type,
                        pixels=list(frame_pixels),
                        pixel_colors=new_pixel_colors,
                        source_image_id=obj.source_image_id,
                        source_image_type=obj.source_image_type,
                        source_image_index=obj.source_image_index,
                                            )
                    # dominant_colorとcolor_ratioはpixel_colorsから自動計算される
                    if hasattr(obj, '_grid'):
                        new_obj._grid = obj._grid

                    self.execution_context['objects'][new_obj.object_id] = new_obj
                    result_ids.append(new_obj_id)
                    #logger.info(f"中空矩形オブジェクト作成: {new_obj_id} ({len(frame_pixels)}ピクセル)")

            # イミュータブル設計: 元のオブジェクトは削除しない
            #logger.info(f"色{color}の中空矩形抽出完了: {len(result_ids)}個")
            return result_ids

        except Exception as e:
            #logger.error(f"色{color}の中空矩形抽出エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []

    def _split_by_line_detection(self, obj: Object) -> List[str]:
        """
        オブジェクトから直線を抽出する

        LINEの定義:
        - 厚さ1ピクセル
        - 長さ2ピクセル以上
        - 8方向（上下、左右、斜め4方向）

        各色ごとに独立して直線を検出します。

        優先順位:
        1. できるだけ長い直線
        2. 斜めより上下左右を優先

        Args:
            obj: 対象オブジェクト

        Returns:
            List[str]: 抽出された線オブジェクトIDのリスト
        """
        try:
            #logger.info(f"線抽出開始: {obj.object_id}")

            if not obj.pixels:
                #logger.info("ピクセルが空のため線抽出できません")
                return []

            # ステップ1: ピクセルを色ごとにグループ化
            pixels_by_color = self._group_pixels_by_color(obj)
            #logger.info(f"色の数: {len(pixels_by_color)}色")

            result_ids = []

            # 背景色を取得
            background_color = self.execution_context.get('background_color', 0)

            # ステップ2: 各色ごとに線を検出（背景色は除外）
            for color, color_pixels in pixels_by_color.items():
                # 背景色の線はスキップ
                if color == background_color:
                    #logger.info(f"色{color}（背景色）の線検出はスキップします")
                    continue

                #logger.info(f"色{color}の線検出開始: {len(color_pixels)}ピクセル")
                lines = self._detect_lines_for_color(
                    color_pixels, color, obj
                )
                result_ids.extend(lines)
                #logger.info(f"色{color}の線: {len(lines)}本検出")

            #logger.info(f"線抽出完了: 合計{len(result_ids)}本の線を抽出")
            return result_ids

        except Exception as e:
            #logger.error(f"線抽出エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []

    def _detect_lines_for_color(self, color_pixels: list, color: int, obj: Object) -> List[str]:
        """特定の色のピクセルから線を検出

        Args:
            color_pixels: この色のピクセルのリスト
            color: 色の値
            obj: 元のオブジェクト（メタデータ用）

        Returns:
            List[str]: 抽出された線オブジェクトのIDリスト
        """
        try:
            # この色のピクセルセット
            pixel_set = set(color_pixels)

            if not pixel_set:
                return []

            # bboxを取得（この色のピクセルから計算）
            xs = [x for x, y in color_pixels]
            ys = [y for x, y in color_pixels]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # グリッドを作成（0: 空, 1: オブジェクトのピクセル）
            grid_width = max_x - min_x + 1
            grid_height = max_y - min_y + 1
            grid = np.zeros((grid_height, grid_width), dtype=int)

            for px, py in color_pixels:
                grid[py - min_y, px - min_x] = 1

            # 8方向の定義
            # 優先順位: 上下左右 > 斜め
            directions = [
                # 上下左右（優先度高）
                (0, 1, 'vertical', 0),      # 下
                (1, 0, 'horizontal', 0),    # 右
                (0, -1, 'vertical', 0),     # 上
                (-1, 0, 'horizontal', 0),   # 左
                # 斜め（優先度低）
                (1, 1, 'diagonal_se', 1),   # 右下
                (1, -1, 'diagonal_ne', 1),  # 右上
                (-1, 1, 'diagonal_sw', 1),  # 左下
                (-1, -1, 'diagonal_nw', 1), # 左上
            ]

            # 全ての線候補を列挙
            # 重複を避けるため、各線をユニークなsetとして記録
            unique_lines = {}  # frozenset(pixels) -> (start_x, start_y, dx, dy, length, priority, pixels_set)

            for y in range(grid_height):
                for x in range(grid_width):
                    if grid[y, x] == 1:
                        # 各方向に線を探索
                        for dx, dy, dir_type, priority in directions:
                            line_pixels = set()
                            line_pixels.add((x, y))

                            # 正方向に伸ばす
                            cx, cy = x + dx, y + dy
                            while 0 <= cx < grid_width and 0 <= cy < grid_height and grid[cy, cx] == 1:
                                line_pixels.add((cx, cy))
                                cx += dx
                                cy += dy

                            # 逆方向に伸ばす
                            cx, cy = x - dx, y - dy
                            while 0 <= cx < grid_width and 0 <= cy < grid_height and grid[cy, cx] == 1:
                                line_pixels.add((cx, cy))
                                cx -= dx
                                cy -= dy

                            # 長さ2以上の線のみ候補に追加
                            if len(line_pixels) >= 2:
                                # 同じピクセルセットの線は1回のみ記録（最初に見つかった方向で）
                                pixels_key = frozenset(line_pixels)
                                if pixels_key not in unique_lines:
                                    unique_lines[pixels_key] = (x, y, dx, dy, len(line_pixels), priority, line_pixels)

            # リストに変換
            all_candidates = list(unique_lines.values())

            # ソート: 長さ降順 → 同じ長さなら優先度昇順（上下左右優先）
            all_candidates.sort(key=lambda c: (-c[4], c[5]))

            #logger.info(f"線候補数: {len(all_candidates)}個")

            # ルール1: 各ピクセルは無制限に使用可能（重複回数制限なし）
            # ルール2: 同じ位置・方向・長さの線は重複抽出禁止
            # ルール3: 抽出する線は、他の線と重複していないピクセルを1つ以上含む必要がある
            # ルール4: 抽出する線は、独立性を持つピクセルを1つ以上含む必要がある
            #          - 上下左右の線: 上下ペアOR左右ペアのどちらかで両サイドが異なる色
            #          - 斜めの線: 上下ペアAND左右ペアの両方で両サイドが異なる色
            selected_lines = []
            used_pixels = set()  # 使用されたピクセルの記録（回数は問わない）
            extracted_lines = set()

            for candidate in all_candidates:
                start_x, start_y, dx, dy, length, priority, pixels_set = candidate
                line_signature = (start_x, start_y, dx, dy, length)

                # ルール2チェック: 同じ線が既に抽出されていないか
                if line_signature in extracted_lines:
                    continue

                # ルール3チェック: この線が、他の線と重複していないピクセルを1つ以上含むか
                has_unique_pixel = False
                for pixel in pixels_set:
                    if pixel not in used_pixels:
                        has_unique_pixel = True
                        break

                if not has_unique_pixel:
                    #logger.debug(f"線スキップ(ルール3違反): pos=({start_x},{start_y}), dir=({dx},{dy}), 全ピクセルが既に使用済み")
                    continue

                # ルール4チェック: 独立性のチェック
                is_diagonal = dx != 0 and dy != 0  # 斜め線かどうか
                has_independent_pixel = False

                for px, py in pixels_set:
                    # 上下左右の隣接ピクセルを取得
                    up = grid[py-1, px] if py > 0 else -1
                    down = grid[py+1, px] if py < grid_height-1 else -1
                    left = grid[py, px-1] if px > 0 else -1
                    right = grid[py, px+1] if px < grid_width-1 else -1

                    # 上下ペア: 両方とも異なる色（0または存在しない=-1も含む）
                    up_down_ok = (up != 1) and (down != 1)

                    # 左右ペア: 両方とも異なる色
                    left_right_ok = (left != 1) and (right != 1)

                    if is_diagonal:
                        # 斜め線: 上下AND左右の両方が必要
                        if up_down_ok and left_right_ok:
                            has_independent_pixel = True
                            break
                    else:
                        # 上下左右の線: 上下OR左右のどちらか一方でOK
                        if up_down_ok or left_right_ok:
                            has_independent_pixel = True
                            break

                if not has_independent_pixel:
                    #logger.debug(f"線スキップ(ルール4違反): pos=({start_x},{start_y}), dir=({dx},{dy}), 独立性なし")
                    continue

                # すべてのルールをパス
                selected_lines.append((start_x, start_y, dx, dy, length, pixels_set))
                # すべてのピクセルを使用済みとして記録
                used_pixels.update(pixels_set)
                extracted_lines.add(line_signature)
                #logger.info(f"線選択: pos=({start_x},{start_y}), dir=({dx},{dy}), length={length}, priority={priority}")

            #logger.info(f"選択された線: {len(selected_lines)}本")

            # 選択された線からオブジェクトを生成
            result_ids = []
            for idx, (lx, ly, dx, dy, length, line_pixels) in enumerate(selected_lines):
                # 線のピクセルを抽出
                pixels_list = []
                for gx, gy in line_pixels:
                    # 元の座標系に戻す
                    orig_x = gx + min_x
                    orig_y = gy + min_y
                    pixels_list.append((orig_x, orig_y))

                # 新しい線オブジェクトを作成
                if pixels_list:
                    new_obj_id = self._generate_unique_object_id(f"{obj.object_id}_line")

                    # この色のピクセルなので、すべて同じ色
                    # pixel_colorsを設定（すべて同じ色）
                    new_pixel_colors = {}
                    for px, py in pixels_list:
                        new_pixel_colors[(px, py)] = color

                    # レイヤー調整: 重複するオブジェクトが同じレイヤーに存在しないようにする
                    new_obj = Object(
                        object_id=new_obj_id,
                        object_type=obj.object_type,
                        pixels=pixels_list,
                        pixel_colors=new_pixel_colors,
                        source_image_id=obj.source_image_id,
                        source_image_type=obj.source_image_type,
                        source_image_index=obj.source_image_index,
                                            )
                    # dominant_colorとcolor_ratioはpixel_colorsから自動計算される
                    if hasattr(obj, '_grid'):
                        new_obj._grid = obj._grid
                    self.execution_context['objects'][new_obj.object_id] = new_obj
                    result_ids.append(new_obj.object_id)
                    #logger.info(f"線オブジェクト作成: {new_obj_id} ({len(pixels_list)}ピクセル)")

            #logger.info(f"色{color}の線抽出完了: {len(result_ids)}本")
            return result_ids

        except Exception as e:
            #logger.error(f"色{color}の線抽出エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []


    def _generate_output_grid(self) -> np.ndarray:
        """出力グリッドを生成（フィルタリング対応）"""
        try:
            # すべてのオブジェクトを取得（SET_OBJECT_TYPE廃止により、フィルタリングなし）
            objects_store = self.execution_context.get('objects', {})
            # 注: execution_context['objects']は常に辞書形式
            if isinstance(objects_store, dict):
                filtered_objects = [obj for obj in objects_store.values()
                                  if obj.object_id != "background"]
            else:
                # 注: リスト形式のフォールバックは削除（execution_context['objects']は常に辞書形式）
                return None

            # 出力グリッド生成器を使用
            current_size = self.grid_context.get_input_grid_size()
            result = self.output_grid_generator.generate_output_grid_from_objects(
                filtered_objects,
                input_grid_size=current_size,
                background_color=self.execution_context['background_color']
            )
            output_grid = result.output_grid

            return output_grid

        except Exception as e:
            #logger.error(f"出力グリッド生成エラー: {e}")
            # フォールバック: デフォルト出力を生成
            return self._generate_default_output()

    def _generate_default_output(self) -> np.ndarray:
        """デフォルト出力を生成"""
        try:
            # 現在のグリッドサイズを取得
            current_size = self.grid_context.get_input_grid_size()
            if current_size:
                return np.full(current_size, self.execution_context['background_color'], dtype=np.int32)
            else:
                return np.array([[self.execution_context['background_color']]], dtype=np.int32)

        except Exception as e:
            #logger.error(f"デフォルト出力生成エラー: {e}")
            return np.array([[0]], dtype=np.int32)

    def get_execution_statistics(self) -> Dict[str, Any]:
        """実行統計を取得"""
        try:
            return {
                'total_objects': len(self.execution_context['objects']),
                'execution_time': 0.0,  # 実際の実行時間は呼び出し元で管理
                'grid_size': self.grid_context.get_current_size(),
                'background_color': self.execution_context['background_color']
            }

        except Exception as e:
            #logger.error(f"実行統計取得エラー: {e}")
            return {}

    def _build_current_grid_from_objects(self) -> Optional[np.ndarray]:
        """現在のオブジェクトから実際のグリッドを構築"""
        try:
            if not hasattr(self, 'execution_context') or 'objects' not in self.execution_context:
                return None

            objects = self.execution_context['objects']
            if not objects:
                return None

            # グリッドサイズを取得
            current_size = self.grid_context.get_input_grid_size()
            grid_height, grid_width = list(current_size) if current_size else [30, 30]
            current_grid = np.zeros((grid_height, grid_width), dtype=np.int32)

            # 各オブジェクトをグリッドに配置
            for obj in objects:
                if (hasattr(obj, 'pixels') and obj.pixels and
                    obj.object_id != "background"):
                    for x, y in obj.pixels:
                        if 0 <= y < grid_height and 0 <= x < grid_width:
                            # オブジェクトの色を取得
                            color = getattr(obj, 'dominant_color', 0)
                            current_grid[y, x] = color

            return current_grid

        except Exception as e:
            #logger.error(f"グリッド構築エラー: {e}")
            return None

    # =============================================================================
    # グローバル関数の実装（本格実装）
    # =============================================================================

    def _get_input_grid_size(self) -> List[int]:
        """入力グリッドサイズを取得（配列: [width, height]）"""
        size = self.grid_context.get_input_size()
        return list(size) if size else [30, 30]

    def _get_grid_size_difference(self) -> Tuple[int, int]:
        """グリッドサイズの差を取得"""
        current_size = self.grid_context.get_input_grid_size()
        current = list(current_size) if current_size else [30, 30]
        input_size = self._get_input_grid_size()
        return (current[0] - input_size[0], current[1] - input_size[1])

    def _get_grid_size_ratio(self) -> Tuple[int, int]:
        """グリッドサイズの比率を取得（整数ベース：100=1.0倍）"""
        current_size = self.grid_context.get_input_grid_size()
        current = list(current_size) if current_size else [30, 30]
        input_size = self._get_input_grid_size()
        height_ratio = int((current[0] / input_size[0]) * 100) if input_size[0] > 0 else 100
        width_ratio = int((current[1] / input_size[1]) * 100) if input_size[1] > 0 else 100
        return (height_ratio, width_ratio)

    def _get_grid_size_statistics(self) -> Dict[str, Any]:
        """グリッドサイズ統計を取得"""
        return {
            'input_size': self._get_input_grid_size(),
            'current_size': list(self.grid_context.get_input_grid_size()) if self.grid_context.get_input_grid_size() else [30, 30],
            'difference': self._get_grid_size_difference(),
            'ratio': self._get_grid_size_ratio()
        }

    def _get_objects_by_type(self, image_index: int, object_type: str) -> List[Object]:
        """オブジェクトタイプでオブジェクトを取得"""
        try:
            objects = []
            objects_store = self.execution_context.get('objects', {})

            # 注: execution_context['objects']は常に辞書形式
            if isinstance(objects_store, dict):
                for obj in objects_store.values():
                    if obj.object_type.value == object_type.lower():
                        objects.append(obj)
            else:
                # 注: リスト形式のフォールバックは削除（execution_context['objects']は常に辞書形式）
                return []

            return objects

        except Exception as e:
            #logger.error(f"オブジェクトタイプ取得エラー: {e}")
            return []

    def _get_objects_at_position(self, x: int, y: int) -> List[str]:
        """指定位置のオブジェクトIDを取得"""
        try:
            object_ids = []
            objects_store = self.execution_context.get('objects', {})

            # 注: execution_context['objects']は常に辞書形式
            if isinstance(objects_store, dict):
                for obj in objects_store.values():
                    if (x, y) in obj.pixels:
                        object_ids.append(obj.object_id)
            else:
                # 注: リスト形式のフォールバックは削除（execution_context['objects']は常に辞書形式）
                return []

            return object_ids

        except Exception as e:
            #logger.error(f"位置別オブジェクト取得エラー: {e}")
            return []

    def _get_unique_colors(self) -> List[int]:
        """ユニークな色のリストを取得"""
        objects_store = self.execution_context.get('objects', {})

        # 注: execution_context['objects']は常に辞書形式
        if isinstance(objects_store, dict):
            colors = set(obj.color for obj in objects_store.values())
        else:
            # 注: リスト形式のフォールバックは削除（execution_context['objects']は常に辞書形式）
            return [self.execution_context.get('background_color', 0)]

        colors.add(self.execution_context['background_color'])
        return list(colors)

    def _get_color_distribution(self) -> Dict[int, int]:
        """色の分布を取得"""
        distribution = {}
        objects_store = self.execution_context.get('objects', {})

        # 注: execution_context['objects']は常に辞書形式
        if isinstance(objects_store, dict):
            for obj in objects_store.values():
                distribution[obj.color] = distribution.get(obj.color, 0) + len(obj.pixels)
        else:
            # 注: リスト形式のフォールバックは削除（execution_context['objects']は常に辞書形式）
            pass

        # 背景色も追加
        distribution[self.execution_context['background_color']] = distribution.get(
            self.execution_context['background_color'], 0
        )

        return distribution

    def _get_grid_statistics(self) -> Dict[str, Any]:
        """グリッド統計を取得"""
        return {
            'size': list(self.grid_context.get_input_grid_size()) if self.grid_context.get_input_grid_size() else [30, 30],
            'object_count': len(self.execution_context['objects']),
            'color_distribution': self._get_color_distribution(),
            'unique_colors': self._get_unique_colors()
        }

    def _get_distance(self, obj1: Object, obj2: Object) -> float:
        """オブジェクト間の距離を取得"""
        try:
            center1_x = (obj1.bbox_left + obj1.bbox_right) / 2
            center1_y = (obj1.bbox_top + obj1.bbox_bottom) / 2
            center2_x = (obj2.bbox_left + obj2.bbox_right) / 2
            center2_y = (obj2.bbox_top + obj2.bbox_bottom) / 2

            return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

        except Exception as e:
            #logger.error(f"距離計算エラー: {e}")
            return 0.0

    def _get_angle(self, obj1: Object, obj2: Object) -> float:
        """オブジェクト間の角度を取得"""
        try:
            import math

            center1_x = (obj1.bbox_left + obj1.bbox_right) / 2
            center1_y = (obj1.bbox_top + obj1.bbox_bottom) / 2
            center2_x = (obj2.bbox_left + obj2.bbox_right) / 2
            center2_y = (obj2.bbox_top + obj2.bbox_bottom) / 2

            dx = center2_x - center1_x
            dy = center2_y - center1_y

            angle = math.atan2(dy, dx)
            return math.degrees(angle)

        except Exception as e:
            #logger.error(f"角度計算エラー: {e}")
            return 0.0



    def _get_spatial_relationships(self) -> Dict[str, List[Tuple[str, str]]]:
        """空間関係を取得"""
        try:
            relationships = {
                'overlapping': [],
                'adjacent': [],
                'distant': []
            }

            objects = self.execution_context['objects']
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects[i+1:], i+1):
                    if self._get_overlapping_pixel_count(obj1, obj2) > 0:
                        relationships['overlapping'].append((obj1.object_id, obj2.object_id))
                    elif self._get_adjacent_edge_count(obj1, obj2) > 0:
                        relationships['adjacent'].append((obj1.object_id, obj2.object_id))
                    else:
                        relationships['distant'].append((obj1.object_id, obj2.object_id))

            return relationships

        except Exception as e:
            #logger.error(f"空間関係取得エラー: {e}")
            return {'overlapping': [], 'adjacent': [], 'distant': []}

    # =============================================================================
    # グリッドサイズ変更関数
    # =============================================================================

    def _set_grid_size(self, height: int, width: int, padding_color: int = 0):
        """グリッドサイズを設定"""
        try:
            # グリッドサイズ変更コマンドを実行
            from src.core_systems.executor.grid.grid_manager import GridSizeCommand, SizeChangeCommand

            cmd = GridSizeCommand(
                command_type=SizeChangeCommand.SET_GRID_SIZE,
                target_size=(height, width),
                padding_color=padding_color
            )

            # 現在のグリッドを取得（本格実装）
            try:
                if hasattr(self, 'execution_context') and 'current_grid' in self.execution_context:
                    current_grid = self.execution_context['current_grid']
                else:
                    # 実際のグリッドを構築
                    current_grid = self._build_current_grid_from_objects()
                    if current_grid is None:
                        current_size = self.grid_context.get_input_grid_size()
                        grid_size = list(current_size) if current_size else [30, 30]
                        current_grid = np.zeros(grid_size, dtype=np.int32)
            except Exception as e:
                #logger.warning(f"グリッド取得エラー: {e}, フォールバックグリッドを使用")
                current_size = self.grid_context.get_input_grid_size()
                grid_size = list(current_size) if current_size else [30, 30]
                current_grid = np.zeros(grid_size, dtype=np.int32)

            # コマンドを実行
            new_grid, result = self.grid_size_executor.execute_command(cmd, current_grid)

            #if result.success:
                #logger.info(f"グリッドサイズ設定完了: {height}x{width}")
            # else:
            #     pass
                #logger.error(f"グリッドサイズ設定エラー: {result.message}")

        except Exception as e:
            pass
            #logger.error(f"グリッドサイズ設定エラー: {e}")

    def _pad_grid(self, pad_top: int, pad_bottom: int, pad_left: int, pad_right: int, padding_color: int = 0):
        """グリッドにパディングを追加"""
        try:
            # パディングコマンドを実行
            from src.core_systems.executor.grid.grid_manager import GridSizeCommand, SizeChangeCommand

            cmd = GridSizeCommand(
                command_type=SizeChangeCommand.PAD_GRID,
                padding={'top': pad_top, 'bottom': pad_bottom, 'left': pad_left, 'right': pad_right},
                padding_color=padding_color
            )

            # 現在のグリッドを取得
            current_size = self.grid_context.get_input_grid_size()
            grid_size = list(current_size) if current_size else [30, 30]
            current_grid = np.zeros(grid_size, dtype=np.int32)

            # コマンドを実行
            new_grid, result = self.grid_size_executor.execute_command(cmd, current_grid)

           # if result.success:
                #logger.info(f"グリッドパディング完了: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
            # else:
            #     pass
                #logger.error(f"グリッドパディングエラー: {result.message}")

        except Exception as e:
            pass
            #logger.error(f"グリッドパディングエラー: {e}")

    def _crop_grid(self, crop_top: int, crop_bottom: int, crop_left: int, crop_right: int):
        """グリッドをクロップ"""
        try:
            # クロップコマンドを実行
            from src.core_systems.executor.grid.grid_manager import GridSizeCommand, SizeChangeCommand

            cmd = GridSizeCommand(
                command_type=SizeChangeCommand.CROP_GRID,
                crop_bounds={'top': crop_top, 'bottom': crop_bottom, 'left': crop_left, 'right': crop_right}
            )

            # 現在のグリッドを取得
            current_size = self.grid_context.get_input_grid_size()
            grid_size = list(current_size) if current_size else [30, 30]
            current_grid = np.zeros(grid_size, dtype=np.int32)

            # コマンドを実行
            new_grid, result = self.grid_size_executor.execute_command(cmd, current_grid)

            # if result.success:
            #     #logger.info(f"グリッドクロップ完了: top={crop_top}, bottom={crop_bottom}, left={crop_left}, right={crop_right}")
            # else:
                #logger.error(f"グリッドクロップエラー: {result.message}")

        except Exception as e:
            pass
            #logger.error(f"グリッドクロップエラー: {e}")

    # =============================================================================
    # オブジェクト操作関数
    # =============================================================================

    def _move(self, object_id: str, dx: int, dy: int) -> Optional[str]:
        """オブジェクトを移動して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            dx, dy: 移動量

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"移動開始: {object_id} -> ({dx}, {dy})")

            # 新しいピクセル位置を計算
            new_pixels = [(x + dx, y + dy) for x, y in obj.pixels]

            # pixel_colorsを移動
            new_pixel_colors = {}
            if hasattr(obj, 'pixel_colors') and obj.pixel_colors:
                for (x, y), color in obj.pixel_colors.items():
                    new_pixel_colors[(x + dx, y + dy)] = color

            # 新しいバウンディングボックスを計算
            if new_pixels:
                xs = [p[0] for p in new_pixels]
                ys = [p[1] for p in new_pixels]
                new_bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                new_bbox = obj.bbox

            # 新しいオブジェクトIDを生成
            new_object_id = self._generate_unique_object_id(f"{obj.object_id}_m{dx}x{dy}")

            # 新しいオブジェクトを作成（元のオブジェクトは変更しない）
            new_obj = Object(
                object_id=new_object_id,
                pixels=new_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_type=obj.object_type,
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # オブジェクトを追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"移動完了: ({dx}, {dy}) -> {new_object_id}")
            return new_object_id

        except Exception as e:
            #logger.error(f"移動エラー: {e}")
            return None




    def _teleport(self, object_id: str, x: int, y: int) -> Optional[str]:
        """オブジェクトをテレポートして新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            x: テレポート先X座標（バウンディングボックスの左上）
            y: テレポート先Y座標（バウンディングボックスの左上）

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"テレポート開始: {object_id} -> ({x}, {y})")

            # オブジェクトの現在のバウンディングボックスを取得
            x1, y1, x2, y2 = obj.bbox

            # 移動量を計算
            dx = x - x1
            dy = y - y1

            # 新しいピクセル位置を計算
            new_pixels = [(px + dx, py + dy) for px, py in obj.pixels]

            # pixel_colorsもテレポート
            new_pixel_colors = {}
            if hasattr(obj, 'pixel_colors') and obj.pixel_colors:
                for (px, py), color in obj.pixel_colors.items():
                    new_pixel_colors[(px + dx, py + dy)] = color

            # 新しいバウンディングボックスを計算
            new_x1 = x
            new_y1 = y
            new_x2 = x + (x2 - x1)
            new_y2 = y + (y2 - y1)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_teleport")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=new_pixels,
                pixel_colors=new_pixel_colors,
                bbox=(new_x1, new_y1, new_x2, new_y2),
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"テレポート完了: {new_id} ({x1}, {y1}) -> ({new_x1}, {new_y1})")
            return new_id

        except Exception as e:
            #logger.error(f"テレポートエラー: {e}")
            return None

    def _align(self, object_id: str, mode: str) -> Optional[str]:
        """オブジェクトを整列して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            mode: 整列モード ("left", "right", "top", "bottom", "center_x", "center_y", "center")

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return None

            # グリッドサイズを取得 (width, height) 形式
            grid_size = self.grid_context.get_input_grid_size()
            grid_width, grid_height = grid_size if grid_size else (30, 30)

            # オブジェクトのサイズと位置を取得
            obj_x = self._get_position_x(object_id)  # オブジェクトの左上X座標
            obj_y = self._get_position_y(object_id)  # オブジェクトの左上Y座標
            obj_width = self._get_bbox_width(object_id)   # オブジェクトの幅
            obj_height = self._get_bbox_height(object_id) # オブジェクトの高さ

            # モードに応じて新しい位置を計算（初期値は元の位置）
            new_x = obj_x
            new_y = obj_y

            if mode == "left":
                # 左端に整列（X座標を0に、Y座標は元のまま）
                new_x = 0
            elif mode == "right":
                # 右端に整列（X座標を右端に、Y座標は元のまま）
                # 負の座標も許可（オブジェクトがグリッドより大きい場合）
                new_x = grid_width - obj_width
            elif mode == "top":
                # 上端に整列（Y座標を0に、X座標は元のまま）
                new_y = 0
            elif mode == "bottom":
                # 下端に整列（Y座標を下端に、X座標は元のまま）
                # 負の座標も許可（オブジェクトがグリッドより大きい場合）
                new_y = grid_height - obj_height
            elif mode == "center_x":
                # X方向中央に整列（X座標を中央に、Y座標は元のまま）
                # 負の座標も許可（オブジェクトがグリッドより大きい場合）
                new_x = (grid_width - obj_width) // 2
            elif mode == "center_y":
                # Y方向中央に整列（Y座標を中央に、X座標は元のまま）
                # 負の座標も許可（オブジェクトがグリッドより大きい場合）
                new_y = (grid_height - obj_height) // 2
            elif mode == "center":
                # 中央に整列（X座標とY座標の両方を中央に）
                # 負の座標も許可（オブジェクトがグリッドより大きい場合）
                new_x = (grid_width - obj_width) // 2
                new_y = (grid_height - obj_height) // 2
            else:
                # 無効なモードの場合は元の位置を維持
                pass

            # TELEPORTを使用して移動
            return self._teleport(object_id, new_x, new_y)

        except Exception as e:
            #logger.error(f"整列エラー: {e}")
            return None

    def _rotate(self, object_id: str, angle: int,
                center_x: float = None, center_y: float = None) -> Optional[str]:
        """オブジェクトを回転して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            angle: 回転角度
            center_x: 回転中心X座標（デフォルト: 左上ピクセルの左上角）
            center_y: 回転中心Y座標（デフォルト: 左上ピクセルの左上角）

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            # デフォルト: オブジェクトの左上ピクセルの左上角（整数座標）
            if center_x is None or center_y is None:
                x_coords = [p[0] for p in obj.pixels]
                y_coords = [p[1] for p in obj.pixels]
                center_x = float(min(x_coords))
                center_y = float(min(y_coords))
            else:
                # 指定された座標をそのまま使用（整数座標として扱う）
                center_x = float(center_x)
                center_y = float(center_y)

            # 回転前の左上角位置を記録（位置調整用）
            original_top_left_x = int(center_x)
            original_top_left_y = int(center_y)

            #logger.info(f"回転開始: {object_id} -> {angle}度, 中心=({center_x}, {center_y})")

            # 90度の倍数の回転は特殊ケースとして正確に処理
            angle_normalized = angle % 360

            if angle_normalized in [90, 180, 270]:
                # 特殊ケース: 90度単位の回転（整数演算、左上角基準）
                new_pixels = []
                new_pixel_colors = {}
                has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

                for px, py in obj.pixels:
                    # 左上角からの相対座標（整数）
                    rel_x = px - int(center_x)
                    rel_y = py - int(center_y)

                    # 90度単位の回転（整数演算）
                    if angle_normalized == 90:
                        # 90度回転: (x, y) → (-y, x)
                        new_rel_x = -rel_y
                        new_rel_y = rel_x
                    elif angle_normalized == 180:
                        # 180度回転: (x, y) → (-x, -y)
                        new_rel_x = -rel_x
                        new_rel_y = -rel_y
                    elif angle_normalized == 270:
                        # 270度回転: (x, y) → (y, -x)
                        new_rel_x = rel_y
                        new_rel_y = -rel_x

                    # 回転後の絶対座標
                    new_px = int(center_x) + new_rel_x
                    new_py = int(center_y) + new_rel_y

                    new_pixels.append((new_px, new_py))

                    # pixel_colorsも回転
                    if has_pixel_colors and (px, py) in obj.pixel_colors:
                        new_pixel_colors[(new_px, new_py)] = obj.pixel_colors[(px, py)]

                # 位置調整: 回転後のバウンディングボックスの左上を元の位置にそろえる
                if new_pixels:
                    new_min_x = min(p[0] for p in new_pixels)
                    new_min_y = min(p[1] for p in new_pixels)
                    offset_x = original_top_left_x - new_min_x
                    offset_y = original_top_left_y - new_min_y

                    # すべてのピクセルを調整
                    adjusted_pixels = []
                    adjusted_pixel_colors = {}
                    for px, py in new_pixels:
                        adjusted_px = px + offset_x
                        adjusted_py = py + offset_y
                        adjusted_pixels.append((adjusted_px, adjusted_py))
                        if has_pixel_colors and (px, py) in new_pixel_colors:
                            adjusted_pixel_colors[(adjusted_px, adjusted_py)] = new_pixel_colors[(px, py)]

                    new_pixels = adjusted_pixels
                    new_pixel_colors = adjusted_pixel_colors
            else:
                # 一般的な回転（任意角度、左上角基準）
                import math
                radians = math.radians(angle)
                cos_angle = math.cos(radians)
                sin_angle = math.sin(radians)

                # 各ピクセルを回転（重複を防ぐためsetを使用）
                new_pixels_set = set()
                new_pixel_colors = {}
                has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

                center_x_int = int(center_x)
                center_y_int = int(center_y)

                for px, py in obj.pixels:
                    # 左上角からの相対座標（整数）
                    rel_x = px - center_x_int
                    rel_y = py - center_y_int

                    # 回転後の相対座標
                    new_rel_x = rel_x * cos_angle - rel_y * sin_angle
                    new_rel_y = rel_x * sin_angle + rel_y * cos_angle

                    # 回転後の絶対座標（四捨五入）
                    new_px = int(round(center_x_int + new_rel_x))
                    new_py = int(round(center_y_int + new_rel_y))
                    new_pixels_set.add((new_px, new_py))

                    # pixel_colorsも回転
                    if has_pixel_colors and (px, py) in obj.pixel_colors:
                        new_pixel_colors[(new_px, new_py)] = obj.pixel_colors[(px, py)]

                # setをlistに変換
                new_pixels = list(new_pixels_set)

                # 位置調整: 回転後のバウンディングボックスの左上を元の位置にそろえる
                if new_pixels:
                    new_min_x = min(p[0] for p in new_pixels)
                    new_min_y = min(p[1] for p in new_pixels)
                    offset_x = original_top_left_x - new_min_x
                    offset_y = original_top_left_y - new_min_y

                    # すべてのピクセルを調整
                    adjusted_pixels = []
                    adjusted_pixel_colors = {}
                    for px, py in new_pixels:
                        adjusted_px = px + offset_x
                        adjusted_py = py + offset_y
                        adjusted_pixels.append((adjusted_px, adjusted_py))
                        if has_pixel_colors and (px, py) in new_pixel_colors:
                            adjusted_pixel_colors[(adjusted_px, adjusted_py)] = new_pixel_colors[(px, py)]

                    new_pixels = adjusted_pixels
                    new_pixel_colors = adjusted_pixel_colors

            # 新しいオブジェクトを作成
            new_object_id = self._generate_unique_object_id(f"{obj.object_id}_rot{angle}")

            # バウンディングボックスを計算
            if new_pixels:
                xs = [p[0] for p in new_pixels]
                ys = [p[1] for p in new_pixels]
                new_bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                new_bbox = obj.bbox

            # 新しいオブジェクトを作成（元のオブジェクトは変更しない）
            new_obj = Object(
                object_id=new_object_id,
                pixels=new_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_type=obj.object_type,
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # オブジェクトを追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"回転完了: {angle}度 -> {new_object_id}")
            return new_object_id

        except Exception as e:
            #logger.error(f"回転エラー: {e}")
            return None

    def _scale(self, object_id: str, scale_factor: int) -> Optional[str]:
        """オブジェクトをスケールして新しいオブジェクトを返す（左上角を基準、オブジェクト自体をn倍）

        Args:
            object_id: オブジェクトID
            scale_factor: スケール倍率（1以上の整数のみ）

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"スケール開始: {object_id} -> 倍率{scale_factor}")

            # スケールの基準点を計算（バウンディングボックスの左上角）
            x1, y1, x2, y2 = obj.bbox
            origin_x = x1  # 左上角のX座標
            origin_y = y1  # 左上角のY座標

            # オブジェクト自体をスケール（各ピクセルをn倍に拡張）
            new_pixels = set()
            new_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

            for px, py in obj.pixels:
                # 左上角を原点とした相対座標
                rel_x = px - origin_x
                rel_y = py - origin_y

                # 元のピクセルの色を取得
                original_color = obj.pixel_colors.get((px, py), obj.dominant_color) if has_pixel_colors else obj.dominant_color

                # スケール後の相対座標範囲を計算
                for dx in range(int(scale_factor)):
                    for dy in range(int(scale_factor)):
                        new_rel_x = rel_x * scale_factor + dx
                        new_rel_y = rel_y * scale_factor + dy

                        # 絶対座標に戻す
                        new_px = int(round(new_rel_x + origin_x))
                        new_py = int(round(new_rel_y + origin_y))

                        new_pixels.add((new_px, new_py))
                        # pixel_colors: スケールされたピクセルすべてに元の色を設定
                        new_pixel_colors[(new_px, new_py)] = original_color

            # 新しいバウンディングボックスを計算
            if new_pixels:
                min_x = min(px for px, py in new_pixels)
                min_y = min(py for px, py in new_pixels)
                max_x = max(px for px, py in new_pixels)
                max_y = max(py for px, py in new_pixels)
                new_bbox = (min_x, min_y, max_x, max_y)
            else:
                new_bbox = obj.bbox

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_scale{scale_factor}")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=list(new_pixels),
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"スケール完了: {new_id} -> 倍率{scale_factor}（オブジェクト自体をn倍）")
            return new_id

        except Exception as e:
            #logger.error(f"スケールエラー: {e}")
            return None

    def _scale_down(self, object_id: str, divisor: int) -> Optional[str]:
        """オブジェクトを1/divisor倍に縮小して新しいオブジェクトを返す

        バウンディングボックスをdivisor×divisorの正方形グリッドに分割し、
        各グリッド内のピクセル充填率が50%以上の場合のみ縮小後のピクセルとして残す。
        余り部分は空の領域として扱い、グリッドに含める（切り上げ）。
        これにより、穴（背景色）も相対的に縮小される。

        Args:
            object_id: オブジェクトID
            divisor: 除数（2以上の整数）

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"縮小開始: {object_id} -> 1/{divisor}倍")

            # バウンディングボックスを取得
            x1, y1, x2, y2 = obj.bbox
            width = x2 - x1 + 1
            height = y2 - y1 + 1

            # 縮小後のサイズ（切り上げ：余り部分も空の領域として含める）
            import math
            new_width = math.ceil(width / divisor)
            new_height = math.ceil(height / divisor)

            if new_width == 0 or new_height == 0:
                #logger.warning(f"縮小後のサイズが0: {object_id}, 元サイズ{width}x{height}")
                return None

            # オブジェクトのピクセルをセットに変換（高速検索用）
            pixel_set = set(obj.pixels)
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

            # 縮小後のピクセルを生成
            new_pixels = []
            new_pixel_colors = {}

            # divisor×divisorの正方形グリッドに分割
            for grid_y in range(new_height):
                for grid_x in range(new_width):
                    # この正方形グリッドの範囲
                    start_x = x1 + grid_x * divisor
                    start_y = y1 + grid_y * divisor
                    end_x = min(start_x + divisor, x2 + 1)  # バウンディングボックスの範囲内に制限
                    end_y = min(start_y + divisor, y2 + 1)

                    # この正方形グリッド内のピクセルを集計
                    filled_pixels = 0
                    colors_in_grid = []

                    for py in range(start_y, end_y):
                        for px in range(start_x, end_x):
                            if (px, py) in pixel_set:
                                filled_pixels += 1
                                color = obj.pixel_colors.get((px, py), obj.dominant_color) if has_pixel_colors else obj.dominant_color
                                colors_in_grid.append(color)

                    # 重要: 全ての正方形空間は同じサイズ（divisor×divisor）で計算
                    # 余り部分も、理論的な正方形サイズで充填率を計算
                    total_pixels = divisor * divisor

                    # 充填率が50%以上の場合のみピクセルとして残す
                    fill_rate = filled_pixels / total_pixels if total_pixels > 0 else 0
                    if fill_rate >= 0.5:
                        # 縮小後の座標
                        new_px = x1 + grid_x
                        new_py = y1 + grid_y

                        new_pixels.append((new_px, new_py))

                        # 最も多い色を選択
                        if colors_in_grid:
                            from collections import Counter
                            color_counts = Counter(colors_in_grid)
                            most_common_color = color_counts.most_common(1)[0][0]
                            new_pixel_colors[(new_px, new_py)] = most_common_color
                        else:
                            new_pixel_colors[(new_px, new_py)] = obj.dominant_color

            if not new_pixels:
                #logger.warning(f"縮小後のピクセルが0個: {object_id}")
                return None

            # 新しいバウンディングボックスを計算
            min_x = min(px for px, py in new_pixels)
            min_y = min(py for px, py in new_pixels)
            max_x = max(px for px, py in new_pixels)
            max_y = max(py for px, py in new_pixels)
            new_bbox = (min_x, min_y, max_x, max_y)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_scaledown{divisor}")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=new_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"縮小完了: {new_id} -> 1/{divisor}倍, ピクセル数: {len(obj.pixels)} → {len(new_obj.pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"縮小エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return None

    def _color_change(self, object_id: str, new_color: int) -> Optional[str]:
        """オブジェクトの色を変更して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            new_color: 新しい色

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj:
                #logger.warning(f"オブジェクトが見つかりません: {object_id}")
                return None

            #logger.info(f"色変更開始: {object_id} -> 色{new_color}")

            # pixel_colorsをすべて新しい色に更新
            new_pixel_colors = {}
            if obj.pixels:
                for px, py in obj.pixels:
                    new_pixel_colors[(px, py)] = new_color

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_setcolor{new_color}")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=obj.pixels.copy() if obj.pixels else [],
                pixel_colors=new_pixel_colors,
                bbox=obj.bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"色変更完了: {new_id} -> 色{new_color}")
            return new_id

        except Exception as e:
            #logger.error(f"色変更エラー: {e}")
            return None

    def _flip(self, object_id: str, axis: str, axis_pos: float = None) -> Optional[str]:
        """オブジェクトを反転して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            axis: "X", "Y"
            axis_pos: 軸の位置（デフォルト: オブジェクト中心）
                - X: X座標(垂直な軸の位置、左右反転)
                - Y: Y座標(水平な軸の位置、上下反転)

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            x_coords = [p[0] for p in obj.pixels]
            y_coords = [p[1] for p in obj.pixels]

            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            #logger.info(f"反転開始: {object_id} -> {axis}, bbox=({min_x},{min_y})-({max_x},{max_y}), size={width}x{height}")

            # ローカル座標系での反転（整数ベース）
            new_pixels = []
            new_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

            if axis.upper() == "X":
                # X軸反転（左右反転）: ローカル座標系を使用
                # ローカル座標に変換
                local_pixels = [(px - min_x, py - min_y) for px, py in obj.pixels]

                # ローカル座標系で左右反転
                flipped_local = [(width - 1 - lx, ly) for lx, ly in local_pixels]

                # グローバル座標系に戻す + pixel_colorsも反転
                for (lx, ly), (px, py) in zip(flipped_local, obj.pixels):
                    global_pos = (lx + min_x, ly + min_y)
                    new_pixels.append(global_pos)
                    if has_pixel_colors and (px, py) in obj.pixel_colors:
                        new_pixel_colors[global_pos] = obj.pixel_colors[(px, py)]

            elif axis.upper() == "Y":
                # Y軸反転（上下反転）: ローカル座標系を使用
                # ローカル座標に変換
                local_pixels = [(px - min_x, py - min_y) for px, py in obj.pixels]

                # ローカル座標系で上下反転
                flipped_local = [(lx, height - 1 - ly) for lx, ly in local_pixels]

                # グローバル座標系に戻す + pixel_colorsも反転
                for (lx, ly), (px, py) in zip(flipped_local, obj.pixels):
                    global_pos = (lx + min_x, ly + min_y)
                    new_pixels.append(global_pos)
                    if has_pixel_colors and (px, py) in obj.pixel_colors:
                        new_pixel_colors[global_pos] = obj.pixel_colors[(px, py)]

            else:
                #logger.error(f"無効な軸: {axis}。X, Y のいずれかを使用してください")
                return None

            # 新しいオブジェクトを作成
            new_object_id = self._generate_unique_object_id(f"{obj.object_id}_flip{axis}")

            # バウンディングボックスを計算
            if new_pixels:
                xs = [p[0] for p in new_pixels]
                ys = [p[1] for p in new_pixels]
                new_bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                new_bbox = obj.bbox

            # 新しいオブジェクトを作成（元のオブジェクトは変更しない）
            new_obj = Object(
                object_id=new_object_id,
                pixels=new_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_type=obj.object_type,
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # オブジェクトを追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"反転完了: {axis} -> {new_object_id}")
            return new_object_id

        except Exception as e:
            #logger.error(f"反転エラー: {e}")
            return None


    def _copy_paste(self, object_id: str, target_x: int, target_y: int) -> Optional[str]:
        """【非推奨】オブジェクトをコピー&ペーストして新しいオブジェクトのIDを返す

        注意: このメソッドは非推奨です。_teleport()と同じ機能です。
        代わりに_teleport(object_id, target_x, target_y)を使用してください。

        Args:
            object_id: コピー元のオブジェクトID
            target_x: コピー先X座標
            target_y: コピー先Y座標

        Returns:
            新しいオブジェクトのID
        """
        #logger.warning("_copy_pasteは非推奨です。_teleport()を使用してください。")
        return self._teleport(object_id, target_x, target_y)

    def _fill_hole(self, object_id: str, hole_color: int) -> Optional[str]:
        """オブジェクトの穴を埋めて新しいオブジェクトを返す

        バウンディングボックス内で、オブジェクトの外側に繋がっていない空間をすべて塗りつぶす。
        元のオブジェクトの色は変更せず、穴の部分だけをhole_colorで塗りつぶす。
        穴が複数ある場合はすべてを1つのオブジェクトとして返す。
        穴がない場合は元のオブジェクトのコピーを返す。

        Args:
            object_id: オブジェクトID
            hole_color: 穴埋めに使用する色

        Returns:
            穴埋め後のオブジェクトID（元のオブジェクト+穴）
            穴がない場合は元のオブジェクトのコピー
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"穴埋め開始: {object_id}, 色={hole_color}")

            # バウンディングボックスを取得
            x1, y1, x2, y2 = obj.bbox
            obj_pixels = set(obj.pixels)

            # バウンディングボックス内の空きピクセルを特定
            empty_pixels = set()
            for y in range(y1, y2 + 1):
                for x in range(x1, x2 + 1):
                    if (x, y) not in obj_pixels:
                        empty_pixels.add((x, y))

            if not empty_pixels:
                # 空きピクセルがない場合は元のオブジェクトをコピーして返す
                #logger.info(f"バウンディングボックス内に空きピクセルがありません。元のオブジェクトをコピーして返します: {object_id}")
                # そのまま処理を続行（hole_pixels=空セット）
                hole_pixels = set()
                filled_pixels = list(obj_pixels)
            else:
                # 外部から到達可能なピクセルを flood fill で特定
                # バウンディングボックスの境界（辺）の空きピクセルから開始
                external_pixels = set()
                visited = set()
                queue = []

                # バウンディングボックスの境界の空きピクセルをキューに追加
                # 上辺と下辺
                for x in range(x1, x2 + 1):
                    if (x, y1) in empty_pixels:
                        queue.append((x, y1))
                        visited.add((x, y1))
                    if (x, y2) in empty_pixels:
                        queue.append((x, y2))
                        visited.add((x, y2))

                # 左辺と右辺（角は上で処理済み）
                for y in range(y1 + 1, y2):
                    if (x1, y) in empty_pixels:
                        queue.append((x1, y))
                        visited.add((x1, y))
                    if (x2, y) in empty_pixels:
                        queue.append((x2, y))
                        visited.add((x2, y))

                # Flood fill: 外部から到達可能な空きピクセルを特定
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                while queue:
                    cx, cy = queue.pop(0)
                    external_pixels.add((cx, cy))

                    for dx, dy in directions:
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) in empty_pixels and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))

                # 穴 = 空きピクセル - 外部から到達可能なピクセル
                hole_pixels = empty_pixels - external_pixels

                if not hole_pixels:
                    # 穴がない場合は元のオブジェクトをコピーして返す
                    #logger.info(f"完全に囲まれた穴が見つかりません。元のオブジェクトをコピーして返します: {object_id}")
                    # 元のオブジェクトのコピーを作成
                    filled_pixels = list(obj_pixels)
                else:
                    #logger.info(f"検出された穴: {len(hole_pixels)}ピクセル")
                    # 元のオブジェクト + 穴を結合（1つのオブジェクト）
                    filled_pixels = list(obj_pixels | hole_pixels)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_fillhole")

            # 新しいバウンディングボックスを計算（変わらないはず）
            xs = [x for x, y in filled_pixels]
            ys = [y for x, y in filled_pixels]
            new_bbox = (min(xs), min(ys), max(xs), max(ys))

            # pixel_colors: 元のオブジェクトの色と穴の色を両方保持
            pixel_colors = {}
            for x, y in filled_pixels:
                if hole_pixels and (x, y) in hole_pixels:
                    # 穴の部分は指定色
                    pixel_colors[(x, y)] = hole_color
                else:
                    # 元のオブジェクト部分は元の色を保持
                    if hasattr(obj, 'pixel_colors') and (x, y) in obj.pixel_colors:
                        pixel_colors[(x, y)] = obj.pixel_colors[(x, y)]
                    else:
                        pixel_colors[(x, y)] = obj.dominant_color

            # 色比率を再計算
            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=filled_pixels,
                pixel_colors=pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_color、color_ratio、color_list、color_countsはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            # if hole_pixels:
            #     #logger.info(f"穴埋め完了: {new_id}, 総ピクセル数={len(filled_pixels)} (元: {len(obj.pixels)}, 穴: {len(hole_pixels)})")
            # else:
                #logger.info(f"穴埋め完了（穴なし、コピーのみ）: {new_id}, 総ピクセル数={len(filled_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"穴埋めエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return None

    def _slide(self, object_id: str, direction: str, obstacles: List[str]) -> Optional[str]:
        """オブジェクトを指定方向に衝突まで移動して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            direction: 移動方向 ("X", "Y", "-X", "-Y", "XY", "-XY", "X-Y", "-X-Y", "C")
            "C"の場合は変化なし（元のオブジェクトをそのまま返す）
            obstacles: 衝突判定するオブジェクトIDの配列

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"スライド開始: {object_id}, 方向={direction}")

            # "C"（中央）の場合は変化なし（元のオブジェクトをそのまま返す）
            if direction == "C":
                # 元のオブジェクトをコピーして新しいIDで返す
                new_id = self._generate_unique_object_id(f"{obj.object_id}_slide_copy")
                has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors
                new_obj = Object(
                    pixels=obj.pixels.copy(),
                    pixel_colors=obj.pixel_colors.copy() if has_pixel_colors else {},
                    bbox=obj.bbox,
                    object_id=new_id,
                    object_type=obj.object_type
                )
                self.execution_context['objects'][new_obj.object_id] = new_obj
                return new_id

            # 方向ベクトルを設定（8方向対応）
            direction_vectors = {
                # 4方向（正負）
                "X": (1, 0),        # 右（0度、X+）
                "-X": (-1, 0),      # 左（180度、X-）
                "Y": (0, 1),        # 下（90度、Y+）
                "-Y": (0, -1),      # 上（270度、Y-）
                # 対角線4方向
                "XY": (1, 1),       # 右下（45度、X+Y+）
                "X-Y": (1, -1),     # 右上（-45度、315度、X+Y-）
                "-XY": (-1, 1),     # 左下（135度、X-Y+）
                "-X-Y": (-1, -1),   # 左上（225度、X-Y-）
            }

            if direction not in direction_vectors:
                #logger.error(f"無効な方向: {direction}。使用可能: X, Y, -X, -Y, XY, -XY, X-Y, -X-Y, C")
                return None

            dx, dy = direction_vectors[direction]

            # 衝突対象オブジェクトのピクセルを取得
            obstacle_pixels = set()
            for obs_id in obstacles:
                obs_obj = self._find_object_by_id(obs_id)
                if obs_obj and obs_obj.object_id != object_id:
                    if hasattr(obs_obj, 'pixels') and obs_obj.pixels:
                        for pixel in obs_obj.pixels:
                            px_val, py_val = pixel[0], pixel[1]
                            obstacle_pixels.add((int(px_val), int(py_val)))

            # 新しいオブジェクトのピクセルを計算
            slide_pixels = []
            slide_pixel_colors = {}

            # ピクセル色情報の取得
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

            for px, py in obj.pixels:
                current_x, current_y = int(px), int(py)
                pixel_color = obj.pixel_colors.get((px, py), obj.dominant_color) if has_pixel_colors else obj.dominant_color

                # 衝突まで移動
                while True:
                    next_x = current_x + dx
                    next_y = current_y + dy

                    # グリッド境界チェック
                    if (next_x < 0 or next_y < 0 or
                        next_x >= self.grid_context.get_input_grid_size()[0] or
                        next_y >= self.grid_context.get_input_grid_size()[1]):
                        break

                    # 障害物との衝突チェック
                    # 斜め移動の場合、隣接ピクセルもチェック（角や中間ピクセルの衝突を検出）
                    is_diagonal = (dx != 0 and dy != 0)
                    collision_detected = False

                    if is_diagonal:
                        # 斜め移動: 次の位置と隣接する2つのピクセルをチェック
                        if ((next_x, next_y) in obstacle_pixels or
                            (next_x, current_y) in obstacle_pixels or
                            (current_x, next_y) in obstacle_pixels):
                            collision_detected = True
                    else:
                        # 直線移動: 次の位置のみをチェック
                        if (next_x, next_y) in obstacle_pixels:
                            collision_detected = True

                    if collision_detected:
                        break

                    # 移動
                    current_x, current_y = next_x, next_y

                # 最終位置を記録
                slide_pixels.append((current_x, current_y))
                slide_pixel_colors[(current_x, current_y)] = pixel_color

            if not slide_pixels:
                #logger.warning(f"スライド結果が空: {object_id}")
                return None

            # 新しいオブジェクトを作成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_slide")
            new_obj = Object(
                object_id=new_id,
                pixels=slide_pixels,
                dominant_color=obj.dominant_color,
                object_type=obj.object_type,
                pixel_colors=slide_pixel_colors if has_pixel_colors else {}
            )

            # オブジェクトを登録
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"スライド完了: {new_id}, ピクセル数={len(slide_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"スライドエラー: {e}")
            return None

    def _pathfind(self, object_id: str, target_x: int, target_y: int, obstacles: List[str]) -> Optional[str]:
        """経路探索移動: オブジェクトを目的地まで移動（障害物回避）

        Args:
            object_id: オブジェクトID
            target_x: 目的地X座標（左上のピクセルが到達すべき位置）
            target_y: 目的地Y座標（左上のピクセルが到達すべき位置）
            obstacles: 衝突判定するオブジェクトIDの配列

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return None

            # 左上のピクセルを取得（最小のx座標、最小のy座標）
            top_left_x = min(px for px, py in obj.pixels)
            top_left_y = min(py for px, py in obj.pixels if px == top_left_x)

            # 既に目的地に到達している場合
            if top_left_x == target_x and top_left_y == target_y:
                return object_id  # 移動不要

            # 障害物ピクセルの取得
            obstacle_pixels = set()
            for obs_id in obstacles:
                obs_obj = self._find_object_by_id(obs_id)
                if obs_obj and obs_obj.object_id != object_id:
                    if hasattr(obs_obj, 'pixels') and obs_obj.pixels:
                        for pixel in obs_obj.pixels:
                            px_val, py_val = pixel[0], pixel[1]
                            obstacle_pixels.add((int(px_val), int(py_val)))

            # 第1段階: 基本直線移動（8方向対応）
            path, final_direction = self._move_toward_target(
                obj,
                (top_left_x, top_left_y),
                (target_x, target_y),
                obstacle_pixels
            )

            # 目的地に到達した場合（左上のピクセルが目的地に到達）
            if path[-1] == (target_x, target_y):
                return self._create_object_from_path(obj, path)

            # 第2段階: 経路探索（左右後ろの順）
            final_path = self._explore_path(
                path[-1],  # 衝突位置
                (target_x, target_y),
                obstacle_pixels,
                final_direction,  # 最後の方向
                obj  # オブジェクト全体の移動に必要
            )

            # 新しいオブジェクトを作成
            return self._create_object_from_path(obj, final_path)

        except Exception as e:
            #logger.error(f"経路探索エラー: {e}")
            return None

    def _move_toward_target(self, obj: Object, top_left_pos: Tuple[int, int], target_pos: Tuple[int, int], obstacles: set) -> Tuple[List[Tuple[int, int]], str]:
        """目的地に向かって基本直線移動（オブジェクト全体を移動、SLIDEと同じ）

        Args:
            obj: 移動対象オブジェクト
            top_left_pos: 左上のピクセルの現在位置 (x, y)
            target_pos: 目的地 (x, y)
            obstacles: 障害物ピクセルのセット

        Returns:
            (path, final_direction): 左上のピクセルの移動経路と最後の方向
        """
        current_top_left_x, current_top_left_y = top_left_pos
        target_x, target_y = target_pos

        # 方向ベクトルの計算
        dx = target_x - current_top_left_x
        dy = target_y - current_top_left_y

        # 8方向に正規化（SLIDEと同じ）
        if dx > 0 and dy > 0:
            direction = "XY"
            dx_norm, dy_norm = 1, 1
        elif dx > 0 and dy < 0:
            direction = "X-Y"
            dx_norm, dy_norm = 1, -1
        elif dx < 0 and dy > 0:
            direction = "-XY"
            dx_norm, dy_norm = -1, 1
        elif dx < 0 and dy < 0:
            direction = "-X-Y"
            dx_norm, dy_norm = -1, -1
        elif dx > 0:
            direction = "X"
            dx_norm, dy_norm = 1, 0
        elif dx < 0:
            direction = "-X"
            dx_norm, dy_norm = -1, 0
        elif dy > 0:
            direction = "Y"
            dx_norm, dy_norm = 0, 1
        elif dy < 0:
            direction = "-Y"
            dx_norm, dy_norm = 0, -1
        else:
            # 既に目的地に到達
            return ([top_left_pos], direction)

        # SLIDEと同じ動作: オブジェクト全体を1ピクセルずつ移動
        path = [(current_top_left_x, current_top_left_y)]
        grid_width, grid_height = self.grid_context.get_input_grid_size()

        while True:
            # 次の左上のピクセル位置
            next_top_left_x = current_top_left_x + dx_norm
            next_top_left_y = current_top_left_y + dy_norm

            # 移動後のオブジェクト全体のピクセルを計算
            offset_x = next_top_left_x - top_left_pos[0]
            offset_y = next_top_left_y - top_left_pos[1]
            moved_pixels = [(int(px) + offset_x, int(py) + offset_y) for px, py in obj.pixels]

            # グリッド境界チェック（オブジェクト全体が範囲内か）
            if any(px < 0 or px >= grid_width or py < 0 or py >= grid_height
                   for px, py in moved_pixels):
                break

            # 障害物チェック（オブジェクト全体が衝突していないか）
            collision_detected = False
            is_diagonal = (dx_norm != 0 and dy_norm != 0)

            for px, py in moved_pixels:
                if is_diagonal:
                    # 斜め移動: 隣接ピクセルもチェック
                    if ((px, py) in obstacles or
                        (px - dx_norm, py) in obstacles or
                        (px, py - dy_norm) in obstacles):
                        collision_detected = True
                        break
                else:
                    # 直線移動: 次の位置のみをチェック
                    if (px, py) in obstacles:
                        collision_detected = True
                        break

            if collision_detected:
                break

            # 目的地到達チェック: 左上のピクセルが目的地に到達したか
            if next_top_left_x == target_x and next_top_left_y == target_y:
                path.append((next_top_left_x, next_top_left_y))
                return (path, direction)

            # 移動
            current_top_left_x, current_top_left_y = next_top_left_x, next_top_left_y
            path.append((current_top_left_x, current_top_left_y))

        return (path, direction)

    def _explore_path(self, start_pos: Tuple[int, int], target_pos: Tuple[int, int], obstacles: set, initial_direction: str, obj: Object, max_steps: int = 1000) -> List[Tuple[int, int]]:
        """経路探索（深さ優先探索、左右後ろの順）

        Args:
            start_pos: 開始位置（左上のピクセル位置）
            target_pos: 目的地（左上のピクセル位置）
            obstacles: 障害物ピクセルのセット
            initial_direction: 初期進行方向
            obj: 移動対象オブジェクト（オブジェクト全体の移動に必要）
            max_steps: 最大探索ステップ数

        Returns:
            path: 最終経路（左上のピクセルの移動経路）
        """
        visited = set()
        stack = [(start_pos, [start_pos], initial_direction)]
        best_position = start_pos
        best_distance = self._euclidean_distance(start_pos, target_pos)
        best_path = [start_pos]

        step_count = 0
        grid_width, grid_height = self.grid_context.get_input_grid_size()

        direction_vectors = {
            "X": (1, 0), "-X": (-1, 0), "Y": (0, 1), "-Y": (0, -1),
            "XY": (1, 1), "X-Y": (1, -1), "-XY": (-1, 1), "-X-Y": (-1, -1),
        }

        # 左上のピクセルの元の位置（オブジェクト全体の移動に必要）
        original_top_left = (min(px for px, py in obj.pixels),
                            min(py for px, py in obj.pixels if px == min(px for px, py in obj.pixels)))

        while stack and step_count < max_steps:
            step_count += 1
            current_pos, path, last_direction = stack.pop()

            # 目的地到達チェック: 左上のピクセルが目的地に到達したか
            if current_pos == target_pos:
                return path

            # 訪問済みチェック
            if current_pos in visited:
                continue
            visited.add(current_pos)

            # 最近位置の更新
            current_distance = self._euclidean_distance(current_pos, target_pos)
            if current_distance < best_distance:
                best_position = current_pos
                best_distance = current_distance
                best_path = path.copy()

            # 左右後ろの順で探索
            exploration_order = self._get_exploration_order(last_direction)

            # 深さ優先探索: 後ろから追加して、先に取り出す
            for direction in reversed(exploration_order):
                dx, dy = direction_vectors[direction]
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)

                # 移動後のオブジェクト全体のピクセルを計算
                offset_x = next_pos[0] - original_top_left[0]
                offset_y = next_pos[1] - original_top_left[1]
                moved_pixels = [(int(px) + offset_x, int(py) + offset_y) for px, py in obj.pixels]

                # グリッド境界チェック
                if any(px < 0 or px >= grid_width or py < 0 or py >= grid_height
                       for px, py in moved_pixels):
                    continue

                # 障害物チェック（オブジェクト全体が衝突していないか）
                is_diagonal = (dx != 0 and dy != 0)
                collision_detected = False

                for px, py in moved_pixels:
                    if is_diagonal:
                        if ((px, py) in obstacles or
                            (px - dx, py) in obstacles or
                            (px, py - dy) in obstacles):
                            collision_detected = True
                            break
                    else:
                        if (px, py) in obstacles:
                            collision_detected = True
                            break

                if collision_detected:
                    continue

                # 訪問済みチェック
                if next_pos in visited:
                    continue

                # スタックに追加
                stack.append((next_pos, path + [next_pos], direction))

        # 目的地に到達できなかった場合、最近位置を返す
        return best_path

    def _get_exploration_order(self, current_direction: str) -> List[str]:
        """現在の進行方向に対して、左右後ろの順で探索方向を返す

        Args:
            current_direction: 現在の進行方向

        Returns:
            探索方向のリスト（左、右、後ろの順）
        """
        direction_map = {
            # 4方向
            "X": ["-Y", "Y", "-X"],      # 右→左(上)、右(下)、後ろ(左)
            "-X": ["Y", "-Y", "X"],      # 左→左(下)、右(上)、後ろ(右)
            "Y": ["X", "-X", "-Y"],      # 下→左(右)、右(左)、後ろ(上)
            "-Y": ["-X", "X", "Y"],      # 上→左(左)、右(右)、後ろ(下)

            # 8方向（斜め）
            "XY": ["-Y", "Y", "-X-Y"],   # 右下→左(右上)、右(左下)、後ろ(左上)
            "X-Y": ["Y", "-Y", "-XY"],   # 右上→左(右下)、右(左上)、後ろ(左下)
            "-XY": ["-Y", "Y", "X-Y"],   # 左下→左(左上)、右(右下)、後ろ(右上)
            "-X-Y": ["Y", "-Y", "XY"],   # 左上→左(左下)、右(右上)、後ろ(右下)
        }
        return direction_map.get(current_direction, ["X", "Y", "-X", "-Y"])

    def _euclidean_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """ユークリッド距離を計算

        Args:
            pos1: 位置1 (x, y)
            pos2: 位置2 (x, y)

        Returns:
            ユークリッド距離
        """
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def _create_object_from_path(self, obj: Object, path: List[Tuple[int, int]]) -> Optional[str]:
        """経路に基づいて新しいオブジェクトを作成

        Args:
            obj: 元のオブジェクト
            path: 左上のピクセルの移動経路

        Returns:
            新しいオブジェクトのID
        """
        if not path:
            return None

        # 元の左上のピクセル位置
        original_top_left_x = min(px for px, py in obj.pixels)
        original_top_left_y = min(py for px, py in obj.pixels if px == original_top_left_x)

        # 最終位置
        final_top_left_x, final_top_left_y = path[-1]

        # オフセットを計算
        offset_x = final_top_left_x - original_top_left_x
        offset_y = final_top_left_y - original_top_left_y

        # ピクセル色情報の取得
        has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

        # 移動後のピクセルを計算
        moved_pixels = []
        moved_pixel_colors = {}
        for px, py in obj.pixels:
            new_x = int(px) + offset_x
            new_y = int(py) + offset_y
            moved_pixels.append((new_x, new_y))
            if has_pixel_colors:
                pixel_color = obj.pixel_colors.get((px, py), obj.dominant_color)
                moved_pixel_colors[(new_x, new_y)] = pixel_color

        if not moved_pixels:
            return None

        # 新しいオブジェクトを作成
        new_id = self._generate_unique_object_id(f"{obj.object_id}_pathfind")
        new_obj = Object(
            object_id=new_id,
            pixels=moved_pixels,
            dominant_color=obj.dominant_color,
            object_type=obj.object_type,
            pixel_colors=moved_pixel_colors if has_pixel_colors else {}
        )

        # オブジェクトを登録
        self.execution_context['objects'][new_obj.object_id] = new_obj

        return new_id

    def _outline(self, object_id: str, outline_color: int) -> Optional[str]:
        """オブジェクトの輪郭オブジェクトを作成して新しいオブジェクトを返す

        オブジェクトの外側1ピクセルに輪郭を作成し、
        内部に穴（空洞）がある場合は穴の内側1ピクセルにも輪郭を作成

        Args:
            object_id: オブジェクトID
            outline_color: 輪郭の色

        Returns:
            新しい輪郭オブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"輪郭作成開始: {object_id}, 色={outline_color}")

            pixel_set = set(obj.pixels)
            outline_pixels_set = set()

            # 4方向の隣接
            directions_4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # 8方向の隣接（対角線含む）
            directions_8 = [
                (0, 1), (0, -1), (1, 0), (-1, 0),      # 4方向
                (1, 1), (1, -1), (-1, 1), (-1, -1)     # 対角線4方向
            ]

            # 1. オブジェクトの外側に輪郭を作成（オブジェクトに接している）
            # オブジェクトに隣接する空ピクセルを輪郭とする
            for x, y in obj.pixels:
                for dx, dy in directions_8:
                    neighbor = (x + dx, y + dy)
                    if neighbor not in pixel_set:
                        outline_pixels_set.add(neighbor)

            #logger.info(f"外側輪郭ピクセル数: {len(outline_pixels_set)}")

            # 2. 内部の穴（空洞）を検出して、穴の境界の1ピクセル内側に輪郭を作成
            # バウンディングボックスを取得
            if obj.bbox:
                min_x, min_y, max_x, max_y = obj.bbox
            else:
                min_x = min(px for px, py in obj.pixels)
                min_y = min(py for px, py in obj.pixels)
                max_x = max(px for px, py in obj.pixels) + 1
                max_y = max(py for px, py in obj.pixels) + 1

            # バウンディングボックス内の空ピクセルで、完全にオブジェクトに囲まれているもの（穴）を検出
            # Flood fillを使って外側と繋がっていない空ピクセルを検出
            from collections import deque

            visited = set()
            outside_empty = set()

            # 探索範囲を拡張（バウンディングボックスの1ピクセル外側まで）
            search_min_x = min_x - 1
            search_max_x = max_x + 1
            search_min_y = min_y - 1
            search_max_y = max_y + 1

            # バウンディングボックスの外周から始めるFlood fill
            queue = deque()
            # 上下の外周
            for x in range(search_min_x, search_max_x):
                for y in [search_min_y, search_max_y - 1]:
                    if (x, y) not in pixel_set and (x, y) not in visited:
                        queue.append((x, y))
                        visited.add((x, y))
                        outside_empty.add((x, y))
            # 左右の外周
            for y in range(search_min_y, search_max_y):
                for x in [search_min_x, search_max_x - 1]:
                    if (x, y) not in pixel_set and (x, y) not in visited:
                        queue.append((x, y))
                        visited.add((x, y))
                        outside_empty.add((x, y))

            # Flood fillで外側と繋がっているすべての空ピクセルを検出（探索範囲内のみ）
            while queue:
                x, y = queue.popleft()
                for dx, dy in directions_4:
                    nx, ny = x + dx, y + dy
                    # 探索範囲内のみチェック
                    if (search_min_x <= nx < search_max_x and
                        search_min_y <= ny < search_max_y and
                        (nx, ny) not in visited and
                        (nx, ny) not in pixel_set):
                        visited.add((nx, ny))
                        outside_empty.add((nx, ny))
                        queue.append((nx, ny))

            # バウンディングボックス内で、外側と繋がっていない空ピクセル = 穴
            holes_pixels = set()
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    if (x, y) not in pixel_set and (x, y) not in outside_empty:
                        holes_pixels.add((x, y))

            # 穴の境界（穴ピクセルでオブジェクトに隣接しているもの）に輪郭を作成
            for x, y in holes_pixels:
                for dx, dy in directions_4:
                    neighbor = (x + dx, y + dy)
                    if neighbor in pixel_set:
                        # この穴ピクセルはオブジェクトに隣接している = 穴の境界
                        outline_pixels_set.add((x, y))
                        break

            if not outline_pixels_set:
                #logger.warning(f"輪郭ピクセルが見つかりません: {object_id}")
                return None

            # 新しいバウンディングボックスを計算
            outline_pixels = list(outline_pixels_set)
            min_x = min(px for px, py in outline_pixels)
            min_y = min(py for px, py in outline_pixels)
            max_x = max(px for px, py in outline_pixels)
            max_y = max(py for px, py in outline_pixels)
            new_bbox = (min_x, min_y, max_x, max_y)

            # pixel_colors: 輪郭はすべてoutline_color
            new_pixel_colors = {(px, py): outline_color for px, py in outline_pixels}

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_outline")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=outline_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"輪郭作成完了: {new_id}, ピクセル数={len(outline_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"輪郭作成エラー: {e}")
            return None

    def _hollow(self, object_id: str) -> Optional[str]:
        """オブジェクトを中空にして新しいオブジェクトを返す（内部を削除）

        Args:
            object_id: オブジェクトID

        Returns:
            新しい中空オブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"中空化開始: {object_id}")

            # 内部ピクセルを検出（すべての8方向に隣接ピクセルがあるもの）
            pixel_set = set(obj.pixels)
            edge_pixels = []

            # 8方向の隣接
            directions = [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

            # 境界ピクセルのみを残す + pixel_colors保持
            new_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

            for x, y in obj.pixels:
                # いずれかの方向に隣接ピクセルがない場合は境界
                is_edge = False
                for dx, dy in directions:
                    if (x + dx, y + dy) not in pixel_set:
                        is_edge = True
                        break

                if is_edge:
                    edge_pixels.append((x, y))
                    # pixel_colorsを保持
                    if has_pixel_colors and (x, y) in obj.pixel_colors:
                        new_pixel_colors[(x, y)] = obj.pixel_colors[(x, y)]

            if not edge_pixels:
                #logger.warning(f"境界ピクセルが見つかりません: {object_id}")
                return None

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_hollow")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=edge_pixels,
                pixel_colors=new_pixel_colors,
                object_id=new_id,
                object_type=obj.object_type,
                source_image_id=obj.source_image_id,
                source_image_type=obj.source_image_type,
                source_image_index=obj.source_image_index,
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"中空化完了: {new_id}, ピクセル数={len(edge_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"中空化エラー: {e}")
            return None

    def _BBOX(self, object_id: str, color: int) -> Optional[str]:
        """オブジェクトのバウンディングボックス矩形を作成して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            color: 矩形の色

        Returns:
            新しい矩形オブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            # 色が指定されていない場合は dominant_color を使用
            if color is None:
                color = obj.dominant_color

            #logger.info(f"バウンディングボックス抽出開始: {object_id}, color={color}")

            # バウンディングボックスを計算（obj.bboxが正しくない場合に備えて再計算）
            if obj.pixels:
                min_x = min(px for px, py in obj.pixels)
                max_x = max(px for px, py in obj.pixels)
                min_y = min(py for px, py in obj.pixels)
                max_y = max(py for px, py in obj.pixels)
            else:
                #logger.warning(f"オブジェクトにピクセルがありません: {object_id}")
                return None

            # バウンディングボックスの矩形ピクセルを生成（包含的範囲）
            bbox_pixels = []
            bbox_pixel_colors = {}
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    bbox_pixels.append((x, y))
                    # 指定された色、または dominant_color を使用
                    bbox_pixel_colors[(x, y)] = color

            # 新しいバウンディングボックスを計算（包括的形式）
            new_bbox = (min_x, min_y, max_x, max_y)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_bbox")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=bbox_pixels,
                pixel_colors=bbox_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"バウンディングボックス抽出完了: {new_id}, ピクセル数={len(bbox_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"バウンディングボックス抽出エラー: {e}")
            return None

    def _subtract(self, target_id: str, subtract_id: str) -> Optional[str]:
        """対象オブジェクトから減算オブジェクトのピクセルを除去して新しいオブジェクトを返す

        Args:
            target_id: 対象オブジェクトID
            subtract_id: 減算するオブジェクトID

        Returns:
            新しいオブジェクトのID
        """
        try:
            target_obj = self._find_object_by_id(target_id)
            subtract_obj = self._find_object_by_id(subtract_id)

            if not target_obj or not target_obj.pixels:
                #logger.warning(f"対象オブジェクトが見つからないか空: {target_id}")
                return None

            if not subtract_obj or not subtract_obj.pixels:
                #logger.warning(f"減算オブジェクトが見つからないか空: {subtract_id}")
                return None

            #logger.info(f"減算開始: {target_id} - {subtract_id}")

            # 減算: 対象オブジェクトから減算オブジェクトのピクセルを除去
            subtract_pixels_set = set(subtract_obj.pixels)
            result_pixels = [p for p in target_obj.pixels if p not in subtract_pixels_set]

            if not result_pixels:
                #logger.warning(f"減算結果が空になりました: {target_id} - {subtract_id}")
                return None

            # pixel_colors: 残ったピクセルの色を保持
            new_pixel_colors = {}
            has_target_pixel_colors = hasattr(target_obj, 'pixel_colors') and target_obj.pixel_colors

            for px, py in result_pixels:
                if has_target_pixel_colors and (px, py) in target_obj.pixel_colors:
                    new_pixel_colors[(px, py)] = target_obj.pixel_colors[(px, py)]
                else:
                    new_pixel_colors[(px, py)] = target_obj.dominant_color

            # 新しいバウンディングボックスを計算
            min_x = min(px for px, py in result_pixels)
            min_y = min(py for px, py in result_pixels)
            max_x = max(px for px, py in result_pixels)
            max_y = max(py for px, py in result_pixels)
            new_bbox = (min_x, min_y, max_x + 1, max_y + 1)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{target_obj.object_id}_subtract")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=result_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=target_obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"減算完了: {new_id}, ピクセル数={len(result_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"減算エラー: {e}")
            return None

    def _count_overlap(self, target_id: str, intersect_id: str) -> int:
        """2つのオブジェクトの重複ピクセル数を返す

        Args:
            target_id: 対象オブジェクトID
            intersect_id: 交差するオブジェクトID

        Returns:
            int: 重複ピクセル数（重複がない場合は0）
        """
        try:
            target_obj = self._find_object_by_id(target_id)
            intersect_obj = self._find_object_by_id(intersect_id)

            if not target_obj or not target_obj.pixels:
                #logger.info(f"対象オブジェクトが見つからないか空: {target_id}")
                return 0

            if not intersect_obj or not intersect_obj.pixels:
                #logger.info(f"交差オブジェクトが見つからないか空: {intersect_id}")
                return 0

            # 交差: 両方のオブジェクトに共通するピクセル
            target_pixels_set = set(target_obj.pixels)
            intersect_pixels_set = set(intersect_obj.pixels)
            overlap_count = len(target_pixels_set & intersect_pixels_set)

            #logger.info(f"重複ピクセル数: {target_id} ∩ {intersect_id} = {overlap_count}ピクセル")
            return overlap_count

        except Exception as e:
            #logger.error(f"重複ピクセル数の計算中にエラー: {e}")
            return 0

    def _intersection(self, target_id: str, intersect_id: str) -> Optional[str]:
        """2つのオブジェクトの共通ピクセルで新しいオブジェクトを返す

        Args:
            target_id: 対象オブジェクトID
            intersect_id: 交差するオブジェクトID

        Returns:
            新しいオブジェクトのID
        """
        try:
            target_obj = self._find_object_by_id(target_id)
            intersect_obj = self._find_object_by_id(intersect_id)

            if not target_obj or not target_obj.pixels:
                #logger.warning(f"対象オブジェクトが見つからないか空: {target_id}")
                return None

            if not intersect_obj or not intersect_obj.pixels:
                #logger.warning(f"交差オブジェクトが見つからないか空: {intersect_id}")
                return None

            #logger.info(f"交差開始: {target_id} ∩ {intersect_id}")

            # 交差: 両方のオブジェクトに共通するピクセル
            target_pixels_set = set(target_obj.pixels)
            intersect_pixels_set = set(intersect_obj.pixels)
            result_pixels = list(target_pixels_set & intersect_pixels_set)

            if not result_pixels:
                #logger.warning(f"交差結果が空になりました: {target_id} ∩ {intersect_id}")
                return None

            # pixel_colors: 交差部分はtarget_objの色を保持
            new_pixel_colors = {}
            has_target_pixel_colors = hasattr(target_obj, 'pixel_colors') and target_obj.pixel_colors

            for px, py in result_pixels:
                if has_target_pixel_colors and (px, py) in target_obj.pixel_colors:
                    new_pixel_colors[(px, py)] = target_obj.pixel_colors[(px, py)]
                else:
                    new_pixel_colors[(px, py)] = target_obj.dominant_color

            # 新しいバウンディングボックスを計算
            min_x = min(px for px, py in result_pixels)
            min_y = min(py for px, py in result_pixels)
            max_x = max(px for px, py in result_pixels)
            max_y = max(py for px, py in result_pixels)
            new_bbox = (min_x, min_y, max_x + 1, max_y + 1)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{target_obj.object_id}_intersection")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=result_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=target_obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"交差完了: {new_id}, ピクセル数={len(result_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"交差エラー: {e}")
            return None

    def _morph_open(self, object_id: str, iterations: int = 1) -> Optional[str]:
        """モルフォロジーオープニングして新しいオブジェクトを返す

        アルゴリズム:
        1. エロージョン（侵食）: 境界ピクセルを除去
        2. ディレーション（拡張）: 周囲に拡張

        Args:
            object_id: オブジェクトID
            iterations: 繰り返し回数

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"モルフォロジーオープニング開始: {object_id}, 繰り返し={iterations}")

            # 8方向の隣接
            neighbors = [
                (0, 1), (0, -1), (1, 0), (-1, 0),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]

            # エロージョン（侵食）: 境界ピクセルを除去
            eroded_pixels = list(obj.pixels)
            for _ in range(iterations):
                pixel_set = set(eroded_pixels)
                new_eroded = []
                for x, y in eroded_pixels:
                    # すべての方向に隣接ピクセルがあるかチェック
                    if all((x+dx, y+dy) in pixel_set for dx, dy in neighbors):
                        new_eroded.append((x, y))
                eroded_pixels = new_eroded

                if not eroded_pixels:
                    break

            # ピクセルがすべて消えた場合
            if not eroded_pixels:
                #logger.warning(f"オープニング: オブジェクトが小さすぎて削除: {object_id}")
                return None

            # ディレーション（拡張）: 8方向に拡張
            dilated_pixels = set(eroded_pixels)
            for _ in range(iterations):
                new_pixels = set()
                for x, y in dilated_pixels:
                    for dx, dy in neighbors:
                        new_pixels.add((x + dx, y + dy))
                dilated_pixels.update(new_pixels)

            result_pixels = list(dilated_pixels)

            # pixel_colors: 元のピクセルの色を保持、新規ピクセルはデフォルト色
            new_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors
            original_pixels_set = set(obj.pixels)

            for px, py in result_pixels:
                if (px, py) in original_pixels_set:
                    # 元のピクセルの色を保持
                    if has_pixel_colors and (px, py) in obj.pixel_colors:
                        new_pixel_colors[(px, py)] = obj.pixel_colors[(px, py)]
                    else:
                        new_pixel_colors[(px, py)] = obj.dominant_color
                else:
                    # 新規拡張ピクセルはデフォルト色
                    new_pixel_colors[(px, py)] = obj.dominant_color

            # 新しいバウンディングボックスを計算
            min_x = min(px for px, py in result_pixels)
            min_y = min(py for px, py in result_pixels)
            max_x = max(px for px, py in result_pixels)
            max_y = max(py for px, py in result_pixels)
            new_bbox = (min_x, min_y, max_x + 1, max_y + 1)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_morph_open")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=result_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"モルフォロジーオープニング完了: {new_id}, ピクセル数={len(result_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"モルフォロジーオープニングエラー: {e}")
            return None

    def _flow(self, object_id: str, direction: str, obstacles: List[str], max_iterations: int = 100) -> Optional[str]:
        """液体シミュレーション（完全版）して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            direction: 流れる方向 ("X", "Y", "-X", "-Y", "C")
            "C"の場合は変化なし（元のオブジェクトをそのまま返す）
            斜め方向（XY, -XY, X-Y, -X-Y）が指定された場合、自動的に "C" に置き換える
            obstacles: 衝突判定するオブジェクトIDの配列
            max_iterations: 最大反復回数

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"液体シミュレーション開始: {object_id}, 方向={direction}")

            # "C"（中央）の場合は変化なし（元のオブジェクトをそのまま返す）
            if direction == "C":
                # 元のオブジェクトをコピーして新しいIDで返す
                new_id = self._generate_unique_object_id(f"{obj.object_id}_flow_copy")
                has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors
                new_obj = Object(
                    pixels=obj.pixels.copy(),
                    pixel_colors=obj.pixel_colors.copy() if has_pixel_colors else {},
                    bbox=obj.bbox,
                    object_id=new_id,
                    object_type=obj.object_type
                )
                self.execution_context['objects'][new_obj.object_id] = new_obj
                return new_id

            # 液体シミュレータをインポート
            from src.core_systems.executor.execution.liquid_simulation import LiquidSimulator
            simulator = LiquidSimulator()

            # 衝突対象オブジェクトを取得
            obstacle_objects = []
            for obs_id in obstacles:
                obs_obj = self._find_object_by_id(obs_id)
                if obs_obj and obs_obj.object_id != object_id:
                    obstacle_objects.append(obs_obj)

            # 液体シミュレーション実行
            flow_pixels = simulator.simulate_liquid_behavior(
                obj, obstacle_objects, direction, max_iterations
            )

            if not flow_pixels:
                #logger.warning(f"液体シミュレーション結果が空: {object_id}")
                return None

            # 新しいバウンディングボックスを計算
            min_x = min(px for px, py in flow_pixels)
            min_y = min(py for px, py in flow_pixels)
            max_x = max(px for px, py in flow_pixels)
            max_y = max(py for px, py in flow_pixels)
            new_bbox = (min_x, min_y, max_x + 1, max_y + 1)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_flow")

            # pixel_colors: 元のピクセルの色を保持、新規ピクセルはデフォルト色
            flow_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors
            original_pixels_set = set(obj.pixels)

            for px, py in flow_pixels:
                if (px, py) in original_pixels_set:
                    # 元のピクセルの色を保持
                    if has_pixel_colors and (px, py) in obj.pixel_colors:
                        flow_pixel_colors[(px, py)] = obj.pixel_colors[(px, py)]
                    else:
                        flow_pixel_colors[(px, py)] = obj.dominant_color
                else:
                    # 新規流動ピクセルは元オブジェクトの色
                    flow_pixel_colors[(px, py)] = obj.dominant_color

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=flow_pixels,
                pixel_colors=flow_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # color_ratioは自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"液体シミュレーション完了: {new_id}, ピクセル数={len(flow_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"液体シミュレーションエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return None

    def _draw(self, object_id: str, dx: int, dy: int) -> Optional[str]:
        """オブジェクトを移動しながら軌跡を描画して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            dx: X方向の移動量
            dy: Y方向の移動量

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"描画開始: {object_id}, 移動=({dx}, {dy})")

            # 移動軌跡のすべてのピクセルを生成
            draw_pixels = set(obj.pixels)
            draw_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

            # 元のピクセルの色を設定
            for px, py in obj.pixels:
                color = obj.pixel_colors.get((px, py), obj.dominant_color) if has_pixel_colors else obj.dominant_color
                draw_pixel_colors[(px, py)] = color

            # 移動ステップ数を計算
            steps = max(abs(dx), abs(dy))
            if steps == 0:
                steps = 1

            # 各ステップで移動しながらピクセルを追加
            for step in range(steps + 1):
                offset_x = int(dx * step / steps)
                offset_y = int(dy * step / steps)

                for px, py in obj.pixels:
                    new_pos = (px + offset_x, py + offset_y)
                    draw_pixels.add(new_pos)
                    # 軌跡ピクセルも元の色を保持
                    color = obj.pixel_colors.get((px, py), obj.dominant_color) if has_pixel_colors else obj.dominant_color
                    draw_pixel_colors[new_pos] = color

            draw_pixels = list(draw_pixels)

            # 新しいバウンディングボックスを計算
            min_x = min(px for px, py in draw_pixels)
            min_y = min(py for px, py in draw_pixels)
            max_x = max(px for px, py in draw_pixels)
            max_y = max(py for px, py in draw_pixels)
            new_bbox = (min_x, min_y, max_x + 1, max_y + 1)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_draw")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=draw_pixels,
                pixel_colors=draw_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"描画完了: {new_id}, ピクセル数={len(draw_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"描画エラー: {e}")
            return None

    def _lay(self, object_id: str, direction: str, obstacles: List[str]) -> Optional[str]:
        """オブジェクトを指定方向に配置して新しいオブジェクトを返す

        Args:
            object_id: オブジェクトID
            direction: 配置方向 ("X", "Y", "-X", "-Y", "XY", "-XY", "X-Y", "-X-Y", "C")
            "C"の場合は変化なし（元のオブジェクトをそのまま返す）
            obstacles: 衝突判定するオブジェクトIDの配列

        Returns:
            新しいオブジェクトのID
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            #logger.info(f"配置開始: {object_id}, 方向={direction}")

            # "C"（中央）の場合は変化なし（元のオブジェクトをそのまま返す）
            if direction == "C":
                # 元のオブジェクトをコピーして新しいIDで返す
                new_id = self._generate_unique_object_id(f"{obj.object_id}_lay_copy")
                has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors
                new_obj = Object(
                    pixels=obj.pixels.copy(),
                    pixel_colors=obj.pixel_colors.copy() if has_pixel_colors else {},
                    bbox=obj.bbox,
                    object_id=new_id,
                    object_type=obj.object_type
                )
                self.execution_context['objects'][new_obj.object_id] = new_obj
                return new_id

            # 方向マッピング（8方向対応）
            direction_map = {
                # 4方向（正負）
                'X': (1, 0),        # 右（0度、X+）
                '-X': (-1, 0),      # 左（180度、X-）
                'Y': (0, 1),        # 下（90度、Y+）
                '-Y': (0, -1),      # 上（270度、Y-）
                # 対角線4方向
                'XY': (1, 1),       # 右下（45度、X+Y+）
                'X-Y': (1, -1),     # 右上（-45度、315度、X+Y-）
                '-XY': (-1, 1),     # 左下（135度、X-Y+）
                '-X-Y': (-1, -1),   # 左上（225度、X-Y-）
            }

            if direction not in direction_map:
                #logger.warning(f"無効な方向: {direction}。使用可能: X, Y, -X, -Y, XY, -XY, X-Y, -X-Y, C")
                return None

            dx, dy = direction_map[direction]
            dx, dy = int(dx), int(dy)

            #logger.info(f"移動方向: dx={dx}, dy={dy}")

            # 各ピクセルを指定方向に落下/配置
            lay_pixels = []
            lay_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors
            max_distance = 100

            # グリッドサイズを取得
            grid = self.execution_context.get('grid')
            if grid is None:
                grid = self.execution_context.get('input_grid')

            if grid is not None and hasattr(grid, 'shape'):
                grid_height, grid_width = int(grid.shape[0]), int(grid.shape[1])
            else:
                grid_height, grid_width = 30, 30  # デフォルト

            #logger.info(f"グリッドサイズ: {grid_width} x {grid_height}")

            # 衝突対象オブジェクトのピクセルを取得
            all_pixels = set()
            for obs_id in obstacles:
                obs_obj = self._find_object_by_id(obs_id)
                if obs_obj and obs_obj.object_id != object_id:
                    if hasattr(obs_obj, 'pixels') and obs_obj.pixels:
                        for pixel in obs_obj.pixels:
                            px_val, py_val = pixel[0], pixel[1]
                            all_pixels.add((int(px_val), int(py_val)))

            #logger.info(f"障害物ピクセル数: {len(all_pixels)}")

            for px, py in obj.pixels:
                # 各ピクセルを方向に沿って配置（軌跡も含める）
                try:
                    current_x, current_y = int(px), int(py)
                    moved_steps = 0

                    # 元のピクセルの色を取得
                    pixel_color = obj.pixel_colors.get((px, py), obj.dominant_color) if has_pixel_colors else obj.dominant_color

                    # 開始位置を追加
                    lay_pixels.append((current_x, current_y))
                    lay_pixel_colors[(current_x, current_y)] = pixel_color

                    for _ in range(max_distance):
                        next_x = int(current_x + dx)
                        next_y = int(current_y + dy)

                        # グリッド境界チェック
                        if next_x < 0 or next_x >= grid_width or next_y < 0 or next_y >= grid_height:
                            break

                        # 障害物チェック
                        # 斜め移動の場合、隣接ピクセルもチェック（角や中間ピクセルの衝突を検出）
                        is_diagonal = (dx != 0 and dy != 0)
                        collision_detected = False

                        if is_diagonal:
                            # 斜め移動: 次の位置と隣接する2つのピクセルをチェック
                            if ((next_x, next_y) in all_pixels or
                                (next_x, current_y) in all_pixels or
                                (current_x, next_y) in all_pixels):
                                collision_detected = True
                        else:
                            # 直線移動: 次の位置のみをチェック
                            if (next_x, next_y) in all_pixels:
                                collision_detected = True

                        if collision_detected:
                            break

                        current_x, current_y = next_x, next_y
                        moved_steps += 1

                        # 軌跡の各位置を追加
                        lay_pixels.append((current_x, current_y))
                        lay_pixel_colors[(current_x, current_y)] = pixel_color
                except Exception as pixel_error:
                    #logger.error(f"ピクセル処理エラー ({px}, {py}): {pixel_error}")
                    raise

            #logger.info(f"平均移動ステップ数: {sum(1 for _ in lay_pixels) / len(lay_pixels) if lay_pixels else 0}")

            # 新しいバウンディングボックスを計算
            min_x = min(px for px, py in lay_pixels)
            min_y = min(py for px, py in lay_pixels)
            max_x = max(px for px, py in lay_pixels)
            max_y = max(py for px, py in lay_pixels)
            new_bbox = (min_x, min_y, max_x + 1, max_y + 1)

            # 新しいオブジェクトIDを生成
            new_id = self._generate_unique_object_id(f"{obj.object_id}_lay")

            # 新しいオブジェクトを作成
            new_obj = Object(
                pixels=lay_pixels,
                pixel_colors=lay_pixel_colors,
                bbox=new_bbox,
                object_id=new_id,
                object_type=obj.object_type
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # 実行コンテキストに追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"配置完了: {new_id}, ピクセル数={len(lay_pixels)}")
            return new_id

        except Exception as e:
            #logger.error(f"配置エラー: {e}")
            return None

    # =============================================================================
    # オブジェクト情報取得関数
    # =============================================================================

    def _find_object_by_id(self, object_id: str) -> Optional[Object]:
        """object_idでオブジェクトを検索"""
        try:
            # 注: execution_context['objects']は常に辞書形式
            if isinstance(self.execution_context['objects'], dict):
                return self.execution_context['objects'].get(object_id)
            else:
                # 注: リスト形式のフォールバックは削除（execution_context['objects']は常に辞書形式）
                return None
        except Exception as e:
            #logger.error(f"オブジェクト検索エラー: {e}")
            return None


    def _is_line(self, object_id: str) -> bool:
        """オブジェクトが直線かどうか"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not hasattr(obj, 'pixels') or not obj.pixels or len(obj.pixels) <= 1:
                return False

            pixels = obj.pixels
            if len(pixels) == 2:
                return True

            # 3点以上の場合は直線性をチェック
            x_coords = [p[0] for p in pixels]
            y_coords = [p[1] for p in pixels]

            # すべての点が同じx座標（垂直線）または同じy座標（水平線）かチェック
            if len(set(x_coords)) == 1:  # 垂直線
                return True
            if len(set(y_coords)) == 1:  # 水平線
                return True

            # 斜めの直線チェック
            if len(pixels) >= 3:
                # 最初の2点から直線の方程式を計算
                x1, y1 = pixels[0]
                x2, y2 = pixels[1]

                if x2 - x1 == 0:  # 垂直線
                    return all(p[0] == x1 for p in pixels[2:])

                # 傾きを計算
                slope = (y2 - y1) / (x2 - x1)

                # 他のすべての点がこの直線上にあるかチェック
                for x, y in pixels[2:]:
                    expected_y = y1 + slope * (x - x1)
                    if abs(y - expected_y) > 0.1:  # 許容誤差
                        return False

                return True

            return False
        except Exception as e:
            #logger.error(f"直線判定エラー: {e}")
            return False

    def _get_line_direction(self, object_id: str) -> str:
        """線の方向を判定

        Returns:
            "X"    - 横線（X方向、すべてのピクセルが同じY座標）
            "Y"    - 縦線（Y方向、すべてのピクセルが同じX座標）
            "XY"   - 45度の対角線（slope = 1.0）
            "-XY"  - 135度の対角線（slope = -1.0）
            "C"    - 線ではない
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels or len(obj.pixels) < 2:
                return "C"

            # ピクセル座標から判定
            x_coords = [x for x, y in obj.pixels]
            y_coords = [y for x, y in obj.pixels]

            # 横線（X方向、すべてのピクセルが同じY座標）
            if len(set(y_coords)) == 1:
                return "X"

            # 縦線（Y方向、すべてのピクセルが同じX座標）
            if len(set(x_coords)) == 1:
                return "Y"

            # 斜め線のチェック
            if self._is_line(object_id):
                # 斜め線の傾きを計算
                # 最初の2点から傾きを計算
                pixels = obj.pixels
                x1, y1 = pixels[0][0], pixels[0][1]
                x2, y2 = pixels[1][0], pixels[1][1]

                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)

                    # ピッタリ45度判定（slope = 1.0、ピッタリのみ）
                    if abs(slope - 1.0) < 1e-10:  # 浮動小数点誤差のみ許容
                        return "XY"

                    # ピッタリ135度判定（slope = -1.0、ピッタリのみ）
                    elif abs(slope - (-1.0)) < 1e-10:  # 浮動小数点誤差のみ許容
                        return "-XY"

                    # その他の斜め線は線として認識しない
                    else:
                        return "C"
                else:
                    # x座標が同じ場合は縦線（既に判定されているはず）
                    return "Y"

            return "C"

        except Exception as e:
            #logger.error(f"線方向判定エラー: {e}")
            return "none"

    def _get_rectangle_type(self, object_id: str) -> str:
        """矩形の種類を判定

        Returns:
            "filled" - 塗りつぶされた矩形
            "hollow" - 中空矩形（枠線のみ、厚さ1ピクセル）
            "none"   - 矩形ではない
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels or len(obj.pixels) < 4:
                return "none"

            # bboxを取得
            x1, y1, x2, y2 = obj.bbox
            width = x2 - x1 + 1
            height = y2 - y1 + 1

            # 最小サイズチェック
            if width < 2 or height < 2:
                return "none"

            pixel_set = set(obj.pixels)

            # 塗りつぶし矩形かチェック（すべてのピクセルが存在）
            expected_pixels = width * height
            if len(obj.pixels) == expected_pixels:
                # すべてのピクセルが矩形範囲内に存在するか確認
                all_present = True
                for y in range(height):
                    for x in range(width):
                        if (x1 + x, y1 + y) not in pixel_set:
                            all_present = False
                            break
                    if not all_present:
                        break

                if all_present:
                    return "filled"

            # 中空矩形かチェック（3x3以上、厚さ1ピクセル）
            if width >= 3 and height >= 3:
                # 4辺がすべて存在するか
                has_all_edges = True

                # 上辺
                for x in range(width):
                    if (x1 + x, y1) not in pixel_set:
                        has_all_edges = False
                        break

                # 下辺
                if has_all_edges:
                    for x in range(width):
                        if (x1 + x, y2) not in pixel_set:
                            has_all_edges = False
                            break

                # 左辺
                if has_all_edges:
                    for y in range(height):
                        if (x1, y1 + y) not in pixel_set:
                            has_all_edges = False
                            break

                # 右辺
                if has_all_edges:
                    for y in range(height):
                        if (x2, y1 + y) not in pixel_set:
                            has_all_edges = False
                            break

                # 内部が空か
                if has_all_edges:
                    interior_empty = True
                    for y in range(1, height - 1):
                        for x in range(1, width - 1):
                            if (x1 + x, y1 + y) in pixel_set:
                                interior_empty = False
                                break
                        if not interior_empty:
                            break

                    if interior_empty:
                        return "hollow"

            return "none"

        except Exception as e:
            #logger.error(f"矩形種類判定エラー: {e}")
            return "none"

    def _get_hole_count(self, object_id: str) -> int:
        """オブジェクトの穴の数を取得（完全に囲まれた空白領域の数）"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return 0

            x_coords = [p[0] for p in obj.pixels]
            y_coords = [p[1] for p in obj.pixels]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            pixel_set = set((p[0], p[1]) for p in obj.pixels)

            # BFS/DFSで外側から到達可能な空白ピクセルをマーク
            visited = set()
            queue = []

            # 境界の空白ピクセルをキューに追加（外側に接している空白）
            for x in range(min_x, max_x + 1):
                for y in [min_y, max_y]:
                    if (x, y) not in pixel_set and (x, y) not in visited:
                        queue.append((x, y))
                        visited.add((x, y))
            for y in range(min_y + 1, max_y):
                for x in [min_x, max_x]:
                    if (x, y) not in pixel_set and (x, y) not in visited:
                        queue.append((x, y))
                        visited.add((x, y))

            # BFSで外側から到達可能な空白を全てマーク
            while queue:
                x, y = queue.pop(0)
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (min_x <= nx <= max_x and min_y <= ny <= max_y and
                        (nx, ny) not in pixel_set and (nx, ny) not in visited):
                        visited.add((nx, ny))
                        queue.append((nx, ny))

            # 残りの空白ピクセル（外側から到達不可能）が穴
            hole_pixels = set()
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    if (x, y) not in pixel_set and (x, y) not in visited:
                        hole_pixels.add((x, y))

            # 穴の連結成分の数をカウント
            hole_count = 0
            hole_visited = set()

            for hole_pixel in hole_pixels:
                if hole_pixel in hole_visited:
                    continue

                # 新しい穴を発見
                hole_count += 1
                queue = [hole_pixel]
                hole_visited.add(hole_pixel)

                # この穴に属するすべてのピクセルをマーク
                while queue:
                    x, y = queue.pop(0)
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in hole_pixels and (nx, ny) not in hole_visited:
                            hole_visited.add((nx, ny))
                            queue.append((nx, ny))

            return hole_count
        except Exception as e:
            #logger.error(f"穴数取得エラー: {e}")
            return 0

    def _get_center_position(self, object_id: str) -> List[int]:
        """オブジェクトの中心座標を取得（配列: [x, y]）- 内部使用"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return [0, 0]

            # ピクセルの平均位置を計算
            x_coords = [p[0] for p in obj.pixels]
            y_coords = [p[1] for p in obj.pixels]

            center_x = int(sum(x_coords) / len(x_coords))
            center_y = int(sum(y_coords) / len(y_coords))

            return [center_x, center_y]
        except Exception as e:
            #logger.error(f"中心座標取得エラー: {e}")
            return [0, 0]

    def _get_center_x(self, object_id: str) -> int:
        """オブジェクトの中心X座標を取得"""
        try:
            center_pos = self._get_center_position(object_id)
            return center_pos[0] if center_pos else 0
        except Exception as e:
            #logger.error(f"中心X座標取得エラー: {e}")
            return 0

    def _get_center_y(self, object_id: str) -> int:
        """オブジェクトの中心Y座標を取得"""
        try:
            center_pos = self._get_center_position(object_id)
            return center_pos[1] if center_pos else 0
        except Exception as e:
            #logger.error(f"中心Y座標取得エラー: {e}")
            return 0

    def _get_max_x(self, object_id: str) -> int:
        """オブジェクトの最大X座標（右端）を取得"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.bbox:
                return 0
            # bboxは(min_x, min_y, max_x, max_y)形式
            return obj.bbox[2]
        except Exception as e:
            #logger.error(f"最大X座標取得エラー: {e}")
            return 0

    def _get_max_y(self, object_id: str) -> int:
        """オブジェクトの最大Y座標（下端）を取得"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.bbox:
                return 0
            # bboxは(min_x, min_y, max_x, max_y)形式
            return obj.bbox[3]
        except Exception as e:
            #logger.error(f"最大Y座標取得エラー: {e}")
            return 0





    def _is_adjacent_to(self, object1_id: str, object2_id: str) -> bool:
        """object1とobject2が隣接しているか（ピクセルレベル）"""
        try:
            obj1 = self._find_object_by_id(object1_id)
            obj2 = self._find_object_by_id(object2_id)
            if not obj1 or not obj2 or not obj1.pixels or not obj2.pixels:
                return False

            pixel_set1 = set(obj1.pixels)
            pixel_set2 = set(obj2.pixels)

            # object1の各ピクセルの4方向隣接をチェック
            for x, y in obj1.pixels:
                neighbors = [(x, y-1), (x, y+1), (x-1, y), (x+1, y)]
                for nx, ny in neighbors:
                    if (nx, ny) in pixel_set2:
                        return True

            return False
        except Exception as e:
            #logger.error(f"隣接判定エラー: {e}")
            return False

    # 廃止されたBBOX版隣接判定メソッド

    def _is_overlapping_with(self, object1_id: str, object2_id: str) -> bool:
        """object1とobject2が重複しているか（ピクセルレベル）"""
        try:
            obj1 = self._find_object_by_id(object1_id)
            obj2 = self._find_object_by_id(object2_id)
            if not obj1 or not obj2 or not obj1.pixels or not obj2.pixels:
                return False

            pixel_set1 = set(obj1.pixels)
            pixel_set2 = set(obj2.pixels)

            # 共通ピクセルがあるかチェック
            return len(pixel_set1.intersection(pixel_set2)) > 0
        except Exception as e:
            #logger.error(f"重複判定エラー: {e}")
            return False

    # 廃止されたBBOX版重複判定メソッド


    def _get_position(self, object_id: str) -> List[int]:
        """オブジェクトの位置座標を取得"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return [0, 0]

            x_coords = [p[0] for p in obj.pixels]
            y_coords = [p[1] for p in obj.pixels]

            # 中心座標を返す
            center_x = (min(x_coords) + max(x_coords)) // 2
            center_y = (min(y_coords) + max(y_coords)) // 2

            return [center_x, center_y]
        except Exception as e:
            #logger.error(f"位置取得エラー: {e}")
            return [0, 0]


    # _is_line の重複実装を削除（行5166の実装が完全版で斜め線もサポート）

    def _is_rectangle(self, object_id: str) -> bool:
        """オブジェクトが矩形か判定"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return False

            if len(obj.pixels) < 4:  # 矩形は最低4ピクセル必要
                return False

            # バウンディングボックスを計算
            min_x = min(pixel[0] for pixel in obj.pixels)
            max_x = max(pixel[0] for pixel in obj.pixels)
            min_y = min(pixel[1] for pixel in obj.pixels)
            max_y = max(pixel[1] for pixel in obj.pixels)

            # バウンディングボックス内のすべてのピクセルが存在するかチェック
            expected_pixels = (max_x - min_x + 1) * (max_y - min_y + 1)
            actual_pixels = len(obj.pixels)

            # ピクセル数が一致し、かつ矩形の形状を満たすかチェック
            return actual_pixels == expected_pixels
        except Exception as e:
            #logger.error(f"矩形判定エラー: {e}")
            return False

    def _is_in_area(self, object_input, x1: int, y1: int, x2: int, y2: int) -> bool:
        """オブジェクトが矩形範囲内にあるか判定（IS_INSIDE）

        判定ルール:
        - オブジェクトのbboxが完全に矩形範囲内にある場合のみTrue
        - 境界（x1, y1, x2, y2）を含む
        - 一部でもはみ出している場合はFalse

        Args:
            object_input: オブジェクトIDまたはオブジェクトIDのリスト
            x1, y1: 矩形の左上座標
            x2, y2: 矩形の右下座標

        Returns:
            bool: 範囲内ならTrue、配列の場合はboolのリスト
        """
        try:
            # リスト入力の場合
            if isinstance(object_input, list):
                result = []
                for obj_id in object_input:
                    obj = self._find_object_by_id(obj_id)
                    if obj and obj.bbox:
                        obj_x1, obj_y1, obj_x2, obj_y2 = obj.bbox
                        # bboxが完全に範囲内にあるか（境界を含む）
                        is_inside = (obj_x1 >= x1 and obj_y1 >= y1 and
                                    obj_x2 <= x2 and obj_y2 <= y2)
                        result.append(is_inside)
                    else:
                        result.append(False)
                return result

            # 単一オブジェクトの場合
            obj = self._find_object_by_id(object_input)
            if not obj or not obj.bbox:
                return False

            obj_x1, obj_y1, obj_x2, obj_y2 = obj.bbox
            # bboxが完全に範囲内にあるか（境界を含む）
            return (obj_x1 >= x1 and obj_y1 >= y1 and
                   obj_x2 <= x2 and obj_y2 <= y2)

        except Exception as e:
            #logger.error(f"矩形範囲内判定エラー: {e}")
            return False

    def _get_distance_between_objects(self, obj1_id: str, obj2_id: str) -> int:
        """2つのオブジェクト間の最短距離を取得（GET_DISTANCE）

        2つのオブジェクトの最も近いピクセル間の距離を返す

        Args:
            obj1_id: オブジェクトID1
            obj2_id: オブジェクトID2

        Returns:
            int: 最短ピクセル間距離（整数、0=接触）
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2 or not obj1.pixels or not obj2.pixels:
                return 0

            min_distance = float('inf')

            # 全ピクセルペアの距離を計算し、最小距離を取得
            for px1, py1 in obj1.pixels:
                for px2, py2 in obj2.pixels:
                    distance = ((px1 - px2) ** 2 + (py1 - py2) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        # 距離0（接触）が見つかったら即座に終了
                        if min_distance == 0:
                            return 0

            return int(round(min_distance))

        except Exception as e:
            #logger.error(f"オブジェクト間距離取得エラー: {e}")
            return 0

    def _get_nearest(self, obj_id: str, object_ids: List[str]) -> Optional[str]:
        """最も近いオブジェクトを取得（GET_NEAREST）

        Args:
            obj_id: 基準オブジェクトID
            object_ids: 候補オブジェクトIDのリスト

        Returns:
            最も近いオブジェクトID、見つからない場合はNone
        """
        try:
            if not obj_id or not object_ids:
                return None

            # Noneを除外
            valid_candidates = [oid for oid in object_ids if oid is not None]
            if not valid_candidates:
                return None

            # 基準オブジェクト自体を除外
            valid_candidates = [oid for oid in valid_candidates if oid != obj_id]
            if not valid_candidates:
                return None

            nearest_id = None
            min_distance = float('inf')

            # 各候補オブジェクトとの距離を計算
            for candidate_id in valid_candidates:
                distance = self._get_distance_between_objects(obj_id, candidate_id)
                if distance < min_distance:
                    min_distance = distance
                    nearest_id = candidate_id
                    # 距離0（接触）が見つかったら即座に終了
                    if min_distance == 0:
                        break

            return nearest_id

        except Exception as e:
            #logger.error(f"最も近いオブジェクト取得エラー: {e}")
            return None

    def _sort_objects_by_key(self, object_ids: List[str], key_expr: str, order: str = "asc", *extra_args) -> List[str]:
        """オブジェクト配列をキー式でソート（SORT_BY）

        Args:
            object_ids: オブジェクトID配列
            key_expr: ソートキー式（コマンド名、例: "GET_SIZE", "GET_DISTANCE"）
            order: ソート順（"asc"=昇順、"desc"=降順）
            *extra_args: 複数引数コマンドの追加引数（例: GET_DISTANCEの基準オブジェクト）

        Returns:
            List[str]: ソート後のオブジェクトID配列
        """
        try:
            if not object_ids:
                return []

            # 各オブジェクトのキー値を計算
            obj_with_keys = []
            for obj_id in object_ids:
                # キー式を評価（obj_idを引数として渡す）
                try:
                    # 追加引数がある場合（複数引数コマンド）
                    if extra_args:
                        key_value = self._evaluate_get_command_with_args(key_expr, obj_id, *extra_args)
                    else:
                        # 単一引数コマンド
                        key_value = self._evaluate_get_command(key_expr, obj_id)

                    obj_with_keys.append((obj_id, key_value))
                except Exception as e:
                    #logger.warning(f"キー値評価エラー（obj={obj_id}, key={key_expr}）: {e}")
                    # エラー時は無限大（最後尾に配置）
                    obj_with_keys.append((obj_id, float('inf') if order == "asc" else float('-inf')))

            # ソート
            reverse = (order == "desc")
            sorted_objs = sorted(obj_with_keys, key=lambda x: x[1], reverse=reverse)

            return [obj_id for obj_id, _ in sorted_objs]

        except Exception as e:
            #logger.error(f"SORT_BYエラー: {e}")
            return object_ids

    def _evaluate_get_command(self, command_name: str, obj_id: str):
        """GETコマンドを評価してキー値を取得（単一引数）"""
        # コマンド名からメソッドを呼び出し
        if command_name == "GET_SIZE":
            return self._get_pixel_count(obj_id)
        elif command_name == "GET_WIDTH":
            return self._get_bbox_width(obj_id)
        elif command_name == "GET_HEIGHT":
            return self._get_bbox_height(obj_id)
        elif command_name == "GET_COLOR":
            colors = self._get_color_list(obj_id)
            return colors[0] if colors else 0
        elif command_name == "GET_X":
            return self._get_position_x(obj_id)
        elif command_name == "GET_Y":
            return self._get_position_y(obj_id)
        elif command_name == "COUNT_HOLES":
            return self._get_hole_count(obj_id)
        elif command_name == "GET_ASPECT_RATIO":
            return self._get_aspect_ratio(obj_id)
        elif command_name == "GET_DENSITY":
            return self._get_density(obj_id)
        elif command_name == "GET_CENTER_X":
            return self._get_center_x(obj_id)
        elif command_name == "GET_CENTER_Y":
            return self._get_center_y(obj_id)
        elif command_name == "GET_MAX_X":
            return self._get_max_x(obj_id)
        elif command_name == "GET_MAX_Y":
            return self._get_max_y(obj_id)
        else:
            #logger.warning(f"未知のソートキー: {command_name}")
            return 0

    def _evaluate_get_command_with_args(self, command_name: str, obj_id: str, *extra_args):
        """GETコマンドを評価してキー値を取得（複数引数）

        Args:
            command_name: コマンド名
            obj_id: 対象オブジェクトID
            *extra_args: 追加引数

        Returns:
            評価されたキー値
        """
        if command_name == "GET_DISTANCE":
            # GET_DISTANCE(obj, reference)
            if len(extra_args) >= 1:
                reference_id = extra_args[0]
                return self._get_distance_between_objects(obj_id, reference_id)
            else:
                raise ValueError("GET_DISTANCEには基準オブジェクトが必要です")

        elif command_name == "GET_SYMMETRY_SCORE":
            # GET_SYMMETRY_SCORE(obj, axis) - 引数必須
            if len(extra_args) >= 1:
                axis = extra_args[0]
                if axis not in ["X", "Y"]:
                    raise ValueError(f"GET_SYMMETRY_SCORE: axisは'X'または'Y'である必要があります（実際: {axis}）")
                return self._get_symmetry_score(obj_id, axis)
            else:
                raise ValueError("GET_SYMMETRY_SCORE: axis引数が必要です（'X'または'Y'）")

        elif command_name == "GET_DIRECTION":
            # GET_DIRECTION(obj1, obj2) - 2つのオブジェクト間の方向
            if len(extra_args) >= 1:
                obj2_id = extra_args[0]
                return self._get_direction(obj_id, obj2_id)
            else:
                raise ValueError("GET_DIRECTION: 基準オブジェクトが必要です")

        elif command_name == "GET_CENTROID":
            # GET_CENTROID(obj) - 単一引数だが、_evaluate_get_commandで処理可能
            return self._get_centroid(obj_id)

        # 将来の拡張用
        # elif command_name == "DISTANCE_SUM":
        #     if len(extra_args) >= 2:
        #         return self._calculate_distance_sum(obj_id, extra_args[0], extra_args[1])
        # elif command_name == "COLOR_SIMILARITY":
        #     if len(extra_args) >= 1:
        #         return self._calculate_color_similarity(obj_id, extra_args[0])

        else:
            #logger.warning(f"未知の複数引数ソートキー: {command_name}")
            return 0

    def _get_bbox_width(self, object_id: str) -> int:
        """バウンディングボックスの幅を取得"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return 0

            x_coords = [p[0] for p in obj.pixels]
            return max(x_coords) - min(x_coords) + 1
        except Exception as e:
            #logger.error(f"バウンディングボックス幅取得エラー: {e}")
            return 0

    def _get_bbox_height(self, object_id: str) -> int:
        """バウンディングボックスの高さを取得"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return 0

            y_coords = [p[1] for p in obj.pixels]
            return max(y_coords) - min(y_coords) + 1
        except Exception as e:
            #logger.error(f"バウンディングボックス高さ取得エラー: {e}")
            return 0


    def _get_color_list(self, object_id: str) -> List[int]:
        """オブジェクトの色リストを取得

        pixel_colorsがある場合は、ピクセル数の多い順に色を返す。
        ない場合は_dominant_colorを返す。
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj:
                return []

            # pixel_colorsがある場合は、それを使用
            if hasattr(obj, 'pixel_colors') and obj.pixel_colors:
                from collections import Counter
                color_counts = Counter(obj.pixel_colors.values())
                # ピクセル数の多い順にソート
                sorted_colors = [color for color, count in color_counts.most_common()]
                return sorted_colors

            # pixel_colorsがない場合はdominant_colorプロパティを返す
            return [obj.dominant_color]
        except Exception as e:
            #logger.error(f"色リスト取得エラー: {e}")
            return [0]

    def _get_density(self, object_id: str) -> int:
        """オブジェクトの密度を取得（100倍してintで返す）

        Returns:
            int: 密度（ピクセル数/バウンディングボックス面積 * 100）
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return 0

            pixel_count = len(obj.pixels)
            bbox_area = self._get_bbox_width(object_id) * self._get_bbox_height(object_id)

            if bbox_area == 0:
                return 0

            density = pixel_count / bbox_area
            return int(density * 100)
        except Exception as e:
            #logger.error(f"密度取得エラー: {e}")
            return 0

    def _get_aspect_ratio(self, object_id: str) -> int:
        """オブジェクトのアスペクト比を取得（幅/高さ、100倍してintで返す）

        Args:
            object_id: オブジェクトID

        Returns:
            int: アスペクト比（幅/高さ * 100）、高さが0の場合は100
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj:
                return 100

            width = self._get_bbox_width(object_id)
            height = self._get_bbox_height(object_id)

            if height == 0:
                return 100

            aspect_ratio = width / height
            return int(aspect_ratio * 100)
        except Exception as e:
            #logger.error(f"アスペクト比取得エラー: {e}")
            return 100

    def _get_direction(self, obj1_id: str, obj2_id: str) -> str:
        """2つのオブジェクト間の方向を取得

        Args:
            obj1_id: オブジェクト1のID（基準）
            obj2_id: オブジェクト2のID（目標）

        Returns:
            str: 方向文字列（"X", "Y", "-X", "-Y", "XY", "-XY", "X-Y", "-X-Y", "C"）
            "C"は同じオブジェクトまたは同じ位置の場合に返される
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                return "C"  # オブジェクトが見つからない場合は中央を返す

            # 同じオブジェクトの場合は中央を返す
            if obj1_id == obj2_id:
                return "C"

            # 中心座標を取得
            center1 = self._get_center_position(obj1_id)
            center2 = self._get_center_position(obj2_id)

            if not center1 or not center2:
                return "C"  # 中心座標が取得できない場合は中央を返す

            dx = center2[0] - center1[0]
            dy = center2[1] - center1[1]

            # 8方向に正規化
            if dx == 0 and dy == 0:
                return "C"  # 同じ位置の場合は中央を返す

            if dx > 0 and dy > 0:
                return "XY"  # 右下
            elif dx > 0 and dy < 0:
                return "X-Y"  # 右上
            elif dx < 0 and dy > 0:
                return "-XY"  # 左下
            elif dx < 0 and dy < 0:
                return "-X-Y"  # 左上
            elif dx > 0:
                return "X"  # 右
            elif dx < 0:
                return "-X"  # 左
            elif dy > 0:
                return "Y"  # 下
            elif dy < 0:
                return "-Y"  # 上
            else:
                return "C"  # 予期しない場合は中央を返す
        except Exception as e:
            #logger.error(f"方向取得エラー: {e}")
            return "C"

    def _get_centroid(self, object_id: str) -> str:
        """オブジェクトの重心位置を方向文字列で返す

        ARCタスクで頻繁に使用される「形状の重心位置」を方向で表現
        例: "C"（中央）、"X"（X方向に偏り）、"-XY"（左上に偏り）など

        重心をバウンディングボックス内で正規化（0.0-1.0）し、
        中心（0.5）に近い場合に"C"を返す（計算誤差を考慮）

        Args:
            object_id: オブジェクトID

        Returns:
            str: 重心位置の方向文字列
        """
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels or not obj.bbox:
                return "C"

            # バウンディングボックス内で重心を計算
            min_x, min_y, max_x, max_y = obj.bbox
            width = max_x - min_x + 1
            height = max_y - min_y + 1

            if width == 0 or height == 0:
                return "C"

            # 重心を計算（ピクセルの平均位置）
            sum_x = sum(p[0] for p in obj.pixels)
            sum_y = sum(p[1] for p in obj.pixels)
            pixel_count = len(obj.pixels)

            if pixel_count == 0:
                return "C"

            centroid_x = sum_x / pixel_count
            centroid_y = sum_y / pixel_count

            # バウンディングボックス内で正規化（0.0-1.0）
            local_x = (centroid_x - min_x) / width if width > 0 else 0.5
            local_y = (centroid_y - min_y) / height if height > 0 else 0.5

            # 中心（0.5）からの距離を計算
            # 計算誤差を考慮して、ぴったり中心（0.5）に近い場合のみ"C"と判定
            # 浮動小数点の計算誤差を考慮して、非常に小さな閾値（0.001）を使用
            # 0.499-0.501の範囲を中心と判定（約±0.1%、非常に厳密）
            center_threshold = 0.001
            center_min = 0.5 - center_threshold  # 0.499
            center_max = 0.5 + center_threshold  # 0.501

            # 方向を決定
            dir_x = ""
            dir_y = ""

            if local_x < center_min:
                dir_x = "-X"  # 左に偏り
            elif local_x > center_max:
                dir_x = "X"  # 右に偏り
            # center_min <= local_x <= center_max の場合は dir_x = ""（ぴったり中心）

            if local_y < center_min:
                dir_y = "-Y"  # 上に偏り
            elif local_y > center_max:
                dir_y = "Y"  # 下に偏り
            # center_min <= local_y <= center_max の場合は dir_y = ""（ぴったり中心）

            # 両方ともぴったり中心の場合のみ"C"を返す
            if dir_x == "" and dir_y == "":
                return "C"

            # 方向を結合
            if dir_x and dir_y:
                # 両方の方向がある場合（例: "X-Y", "-XY"）
                return f"{dir_x}{dir_y}" if dir_x.startswith("-") else f"{dir_x}-{dir_y}"
            elif dir_x:
                return dir_x
            elif dir_y:
                return dir_y
            else:
                return "C"
        except Exception as e:
            #logger.error(f"局所重心取得エラー: {e}")
            return "C"

    # =============================================================================
    # 新規コマンド: 形状・距離・重複判定
    # =============================================================================

    def _has_same_shape(self, obj1_id: str, obj2_id: str) -> bool:
        """2つのオブジェクトが同じ形状か判定（回転・反転は考慮しない）

        Args:
            obj1_id: オブジェクト1のID
            obj2_id: オブジェクト2のID

        Returns:
            bool: 同じ形状の場合True
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                return False

            if not obj1.pixels or not obj2.pixels:
                return False

            # ピクセル数が異なれば形状も異なる
            if len(obj1.pixels) != len(obj2.pixels):
                return False

            # ピクセル座標を正規化（左上を(0,0)に）
            def normalize_pixels(pixels):
                if not pixels:
                    return set()
                min_x = min(p[0] for p in pixels)
                min_y = min(p[1] for p in pixels)
                return set((p[0] - min_x, p[1] - min_y) for p in pixels)

            norm1 = normalize_pixels(obj1.pixels)
            norm2 = normalize_pixels(obj2.pixels)

            return norm1 == norm2

        except Exception as e:
            #logger.error(f"形状比較エラー: {e}")
            return False

    def _get_x_distance(self, obj1_id: str, obj2_id: str) -> int:
        """2つのオブジェクト間のX方向の最小距離を計算

        Args:
            obj1_id: オブジェクト1のID
            obj2_id: オブジェクト2のID

        Returns:
            int: X方向の最小距離（0以上）
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                return 0

            if not obj1.pixels or not obj2.pixels:
                return 0

            # X座標のみを抽出
            x_coords1 = set(p[0] for p in obj1.pixels)
            x_coords2 = set(p[0] for p in obj2.pixels)

            # 最小X距離を計算
            min_dist = float('inf')
            for x1 in x_coords1:
                for x2 in x_coords2:
                    dist = abs(x1 - x2)
                    min_dist = min(min_dist, dist)

            return int(min_dist) if min_dist != float('inf') else 0

        except Exception as e:
            #logger.error(f"X距離計算エラー: {e}")
            return 0

    def _get_y_distance(self, obj1_id: str, obj2_id: str) -> int:
        """2つのオブジェクト間のY方向の最小距離を計算

        Args:
            obj1_id: オブジェクト1のID
            obj2_id: オブジェクト2のID

        Returns:
            int: Y方向の最小距離（0以上）
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                return 0

            if not obj1.pixels or not obj2.pixels:
                return 0

            # Y座標のみを抽出
            y_coords1 = set(p[1] for p in obj1.pixels)
            y_coords2 = set(p[1] for p in obj2.pixels)

            # 最小Y距離を計算
            min_dist = float('inf')
            for y1 in y_coords1:
                for y2 in y_coords2:
                    dist = abs(y1 - y2)
                    min_dist = min(min_dist, dist)

            return int(min_dist) if min_dist != float('inf') else 0

        except Exception as e:
            #logger.error(f"Y距離計算エラー: {e}")
            return 0

    def _has_same_color_structure(self, obj1_id: str, obj2_id: str) -> bool:
        """2つのオブジェクトが同じ形状かつ色構造か判定（色の値は異なってもよい）

        Args:
            obj1_id: オブジェクト1のID
            obj2_id: オブジェクト2のID

        Returns:
            bool: 同じ形状かつ色構造の場合True
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                return False

            if not obj1.pixels or not obj2.pixels:
                return False

            # ピクセル数が異なれば形状も異なる
            if len(obj1.pixels) != len(obj2.pixels):
                return False

            # ピクセル座標と色を正規化
            def normalize_with_colors(pixels):
                if not pixels:
                    return {}
                min_x = min(p[0] for p in pixels)
                min_y = min(p[1] for p in pixels)
                result = {}
                for p in pixels:
                    pos = (p[0] - min_x, p[1] - min_y)
                    color = p[2] if len(p) > 2 else 0
                    result[pos] = color
                return result

            norm1 = normalize_with_colors(obj1.pixels)
            norm2 = normalize_with_colors(obj2.pixels)

            # 位置が一致するか確認
            if set(norm1.keys()) != set(norm2.keys()):
                return False

            # 色構造を確認（一対一の色のマッピングを見つける）
            color_map = {}  # obj1の色 → obj2の色
            reverse_map = {}  # obj2の色 → obj1の色（一対一を保証）

            for pos in norm1.keys():
                c1 = norm1[pos]
                c2 = norm2[pos]

                # obj1の色c1が既にマッピング済みの場合
                if c1 in color_map:
                    if color_map[c1] != c2:
                        return False  # 矛盾（c1が複数の色にマップされる）
                else:
                    color_map[c1] = c2

                # obj2の色c2が既にマッピング済みの場合（一対一チェック）
                if c2 in reverse_map:
                    if reverse_map[c2] != c1:
                        return False  # 多対一マッピング（複数の色が同じ色にマップされる）
                else:
                    reverse_map[c2] = c1

            return True

        except Exception as e:
            #logger.error(f"色構造比較エラー: {e}")
            return False

    def _has_same_shape_and_color(self, obj1_id: str, obj2_id: str) -> bool:
        """2つのオブジェクトが完全に一致するか判定（形状と色）

        Args:
            obj1_id: オブジェクト1のID
            obj2_id: オブジェクト2のID

        Returns:
            bool: 完全に一致する場合True
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                return False

            if not obj1.pixels or not obj2.pixels:
                return False

            # ピクセル数が異なれば一致しない
            if len(obj1.pixels) != len(obj2.pixels):
                return False

            # ピクセル座標と色を正規化
            def normalize_with_colors(pixels):
                if not pixels:
                    return set()
                min_x = min(p[0] for p in pixels)
                min_y = min(p[1] for p in pixels)
                result = set()
                for p in pixels:
                    pos_x = p[0] - min_x
                    pos_y = p[1] - min_y
                    color = p[2] if len(p) > 2 else 0
                    result.add((pos_x, pos_y, color))
                return result

            norm1 = normalize_with_colors(obj1.pixels)
            norm2 = normalize_with_colors(obj2.pixels)

            return norm1 == norm2

        except Exception as e:
            #logger.error(f"完全一致判定エラー: {e}")
            return False

    def _get_adjacent_edge_count(self, obj1_id: str, obj2_id: str) -> int:
        """2つのオブジェクトの外形が隣接するセグメント数を取得

        定義: 外形の輪郭同士が接している部分を、一直線になっているセグメントごとにカウント

        Args:
            obj1_id: オブジェクト1のID
            obj2_id: オブジェクト2のID

        Returns:
            int: 隣接するセグメント数（一直線の隣接を1とする）
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                return 0

            if not obj1.pixels or not obj2.pixels:
                return 0

            # ピクセル座標のセット
            pixels1 = set((p[0], p[1]) for p in obj1.pixels)
            pixels2 = set((p[0], p[1]) for p in obj2.pixels)

            # 外形同士が隣接している辺を列挙
            adjacent_edges = []

            for x, y in pixels1:
                for dx, dy, direction in [(1, 0, 'R'), (-1, 0, 'L'), (0, 1, 'D'), (0, -1, 'U')]:
                    nx, ny = x + dx, y + dy

                    # 重複している場合は隣接ではない
                    if (nx, ny) in pixels1 and (nx, ny) in pixels2:
                        continue

                    # obj1のこの方向が外形かチェック（内側にピクセルがない）
                    is_obj1_outline = (nx, ny) not in pixels1

                    # obj2のこの方向が外形かチェック（obj1から見た隣の位置にobj2があり、その逆方向が外形）
                    if (nx, ny) in pixels2:
                        # (nx, ny)がobj2のピクセル
                        # その逆方向(-dx, -dy)が外形かチェック
                        reverse_x, reverse_y = nx - dx, ny - dy
                        is_obj2_outline = (reverse_x, reverse_y) not in pixels2

                        # 両方の外形が接している場合のみカウント
                        if is_obj1_outline and is_obj2_outline:
                            adjacent_edges.append(((x, y), direction))

            if not adjacent_edges:
                return 0

            # 連続する辺をグループ化してセグメント数をカウント
            visited = set()
            segment_count = 0

            for edge in adjacent_edges:
                if edge in visited:
                    continue

                # 新しいセグメントを開始
                segment_count += 1
                queue = [edge]
                visited.add(edge)

                # 連続する辺を探す
                while queue:
                    (cx, cy), cdir = queue.pop(0)

                    # 同じ方向で連続する辺を探す
                    if cdir in ['U', 'D']:  # 上下方向
                        # 左右に連続する辺を探す
                        for next_edge in [((cx-1, cy), cdir), ((cx+1, cy), cdir)]:
                            if next_edge in adjacent_edges and next_edge not in visited:
                                visited.add(next_edge)
                                queue.append(next_edge)

                    elif cdir in ['L', 'R']:  # 左右方向
                        # 上下に連続する辺を探す
                        for next_edge in [((cx, cy-1), cdir), ((cx, cy+1), cdir)]:
                            if next_edge in adjacent_edges and next_edge not in visited:
                                visited.add(next_edge)
                                queue.append(next_edge)

            return segment_count

        except Exception as e:
            #logger.error(f"隣接セグメント数計算エラー: {e}")
            return 0



    def _get_line_endpoints(self, object_id: str) -> List[List[int]]:
        """線の端点座標を取得"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return [[0, 0], [0, 0]]

            if len(obj.pixels) < 2:
                return [list(obj.pixels[0][:2]), list(obj.pixels[0][:2])]

            # 線の端点を特定
            x_coords = [p[0] for p in obj.pixels]
            y_coords = [p[1] for p in obj.pixels]

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            # 端点を探す
            endpoints = []
            for pixel in obj.pixels:
                x, y = pixel[0], pixel[1]
                if x == min_x or x == max_x or y == min_y or y == max_y:
                    endpoints.append([x, y])

            if len(endpoints) >= 2:
                return endpoints[:2]
            else:
                return [[min_x, min_y], [max_x, max_y]]
        except Exception as e:
            #logger.error(f"線端点取得エラー: {e}")
            return [[0, 0], [0, 0]]


    def _get_symmetry_score(self, object_id: str, axis: str) -> float:
        """オブジェクトの対称性スコアを取得"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                return 0.0

            # 対称性計算（X軸またはY軸基準）
            x_coords = [p[0] for p in obj.pixels]
            y_coords = [p[1] for p in obj.pixels]

            if axis.upper() == "X":
                # X軸対称性
                center_x = (min(x_coords) + max(x_coords)) / 2
                symmetric_pixels = 0
                for pixel in obj.pixels:
                    x, y = pixel[0], pixel[1]
                    mirror_x = int(2 * center_x - x)
                    if (mirror_x, y) in obj.pixels:
                        symmetric_pixels += 1
                return symmetric_pixels / len(obj.pixels)

            elif axis.upper() == "Y":
                # Y軸対称性
                center_y = (min(y_coords) + max(y_coords)) / 2
                symmetric_pixels = 0
                for pixel in obj.pixels:
                    x, y = pixel[0], pixel[1]
                    mirror_y = int(2 * center_y - y)
                    if (x, mirror_y) in obj.pixels:
                        symmetric_pixels += 1
                return symmetric_pixels / len(obj.pixels)

            return 0.0
        except Exception as e:
            #logger.error(f"対称性スコア取得エラー: {e}")
            return 0.0

    def _create_symmetry(self, object_id: str, axis: str) -> Optional[str]:
        """対称オブジェクトを生成して新しいオブジェクトIDを返す"""
        try:
            obj = self._find_object_by_id(object_id)
            if not obj or not obj.pixels:
                #logger.warning(f"オブジェクトが見つからないか空: {object_id}")
                return None

            x_coords = [p[0] for p in obj.pixels]
            y_coords = [p[1] for p in obj.pixels]

            new_pixels = []
            new_pixel_colors = {}
            has_pixel_colors = hasattr(obj, 'pixel_colors') and obj.pixel_colors

            if axis.upper() == "X" or axis.lower() == "horizontal":
                # X軸対称（水平軸）
                center_x = (min(x_coords) + max(x_coords)) / 2
                for pixel in obj.pixels:
                    x, y = pixel[0], pixel[1]
                    mirror_x = int(2 * center_x - x)
                    new_pos = (mirror_x, y)
                    new_pixels.append(new_pos)
                    # pixel_colors保持
                    if has_pixel_colors and (x, y) in obj.pixel_colors:
                        new_pixel_colors[new_pos] = obj.pixel_colors[(x, y)]

            elif axis.upper() == "Y" or axis.lower() == "vertical":
                # Y軸対称（垂直軸）
                center_y = (min(y_coords) + max(y_coords)) / 2
                for pixel in obj.pixels:
                    x, y = pixel[0], pixel[1]
                    mirror_y = int(2 * center_y - y)
                    new_pos = (x, mirror_y)
                    new_pixels.append(new_pos)
                    # pixel_colors保持
                    if has_pixel_colors and (x, y) in obj.pixel_colors:
                        new_pixel_colors[new_pos] = obj.pixel_colors[(x, y)]

            elif axis.lower() == "diagonal":
                # 対角線対称
                for pixel in obj.pixels:
                    x, y = pixel[0], pixel[1]
                    new_pos = (y, x)
                    new_pixels.append(new_pos)
                    # pixel_colors保持
                    if has_pixel_colors and (x, y) in obj.pixel_colors:
                        new_pixel_colors[new_pos] = obj.pixel_colors[(x, y)]

            else:
                #logger.error(f"無効な軸: {axis}")
                return None

            # 新しいオブジェクトを作成
            new_object_id = self._generate_unique_object_id(f"{obj.object_id}_symmetry_{axis}")

            # バウンディングボックスを計算
            if new_pixels:
                xs = [p[0] for p in new_pixels]
                ys = [p[1] for p in new_pixels]
                new_bbox = (min(xs), min(ys), max(xs), max(ys))
            else:
                new_bbox = obj.bbox

            # 新しいオブジェクトを作成（元のオブジェクトは変更しない）
            new_obj = Object(
                object_id=new_object_id,
                pixels=new_pixels,
                pixel_colors=new_pixel_colors,
                bbox=new_bbox,
                object_type=obj.object_type,   # 元のオブジェクトと同じレイヤー
            )
            # dominant_colorとcolor_ratioはpixel_colorsから自動計算される

            # オブジェクトを追加
            self.execution_context['objects'][new_obj.object_id] = new_obj

            #logger.info(f"対称オブジェクト生成完了: {axis} -> {new_object_id}")
            return new_object_id

        except Exception as e:
            #logger.error(f"対称オブジェクト生成エラー: {e}")
            return None


    def _concat(self, array1: List[str], array2: List[str]) -> List[str]:
        """2つの配列を結合

        Args:
            array1: 1つ目の配列
            array2: 2つ目の配列

        Returns:
            List[str]: 結合された配列
        """
        try:
            # 両方が配列であることを確認
            if not isinstance(array1, list):
                array1 = [array1] if array1 else []
            if not isinstance(array2, list):
                array2 = [array2] if array2 else []

            # 配列を結合（重複を許可）
            result = array1 + array2

            #logger.info(f"CONCAT: {len(array1)}個 + {len(array2)}個 = {len(result)}個")
            return result

        except Exception as e:
            #logger.error(f"CONCAT エラー: {e}")
            return array1 if isinstance(array1, list) else []

    def _exclude(self, objects: List[str], targets: List[str]) -> List[str]:
        """オブジェクト配列から特定のオブジェクトを除外

        Args:
            objects: 元のオブジェクト配列
            targets: 除外するオブジェクト配列

        Returns:
            List[str]: 除外後のオブジェクト配列

        Note:
            イミュータブル対応：IDではなく、オブジェクトの内容で比較
            - 位置（x, y）
            - サイズ（width, height）
            - すべてのピクセル（座標+色）

            効率化: 位置・サイズ・ピクセル数でインデックスを作成し、
            同じインデックスのターゲットのみ詳細比較することで、
            計算量をO(n×m)からO(n×k)に削減（k << m）
        """
        try:
            import json

            # 配列の正規化
            if not isinstance(objects, list):
                objects = [objects] if objects else []
            if not isinstance(targets, list):
                targets = [targets] if targets else []

            # オブジェクト数が多い場合は警告
            if len(objects) > 1000 or len(targets) > 1000:
                logger.warning(f"_exclude: 大量のオブジェクトを処理します（objects={len(objects)}, targets={len(targets)}）")

            # ターゲットが空の場合は即座に返す
            if not targets:
                return objects

            # targetsを位置・サイズ・ピクセル数でインデックス化（効率化）
            target_index = {}
            for target in targets:
                try:
                    target_data = json.loads(target) if isinstance(target, str) else target
                    # 位置・サイズ・ピクセル数の組み合わせでキーを作成
                    index_key = (
                        target_data.get('x'),
                        target_data.get('y'),
                        target_data.get('width'),
                        target_data.get('height'),
                        len(target_data.get('pixels', []))
                    )
                    if index_key not in target_index:
                        target_index[index_key] = []
                    target_index[index_key].append(target)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    # パースエラーは無視（後で詳細比較で除外される）
                    continue

            result = []
            excluded_count = 0

            for obj in objects:
                try:
                    obj_data = json.loads(obj) if isinstance(obj, str) else obj
                    # 同じ位置・サイズ・ピクセル数のターゲットのみを候補とする
                    index_key = (
                        obj_data.get('x'),
                        obj_data.get('y'),
                        obj_data.get('width'),
                        obj_data.get('height'),
                        len(obj_data.get('pixels', []))
                    )

                    is_excluded = False

                    # 同じインデックスのターゲットのみ詳細比較
                    if index_key in target_index:
                        for target in target_index[index_key]:
                            if self._objects_are_equal(obj, target):
                                is_excluded = True
                                excluded_count += 1
                                break

                    if not is_excluded:
                        result.append(obj)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    # パースエラーは除外対象として扱わない（元のオブジェクトを保持）
                    result.append(obj)

            #logger.info(f"EXCLUDE: {len(objects)}個から{excluded_count}個を除外 → {len(result)}個")
            return result

        except Exception as e:
            #logger.error(f"EXCLUDE エラー: {e}")
            return objects if isinstance(objects, list) else []

    def _objects_are_equal(self, obj1: str, obj2: str) -> bool:
        """2つのオブジェクトが完全に一致するかチェック

        Args:
            obj1: オブジェクト1（JSON文字列）
            obj2: オブジェクト2（JSON文字列）

        Returns:
            bool: 完全一致する場合True

        Note:
            以下をすべてチェック：
            1. 位置（x, y）
            2. サイズ（width, height）
            3. すべてのピクセル（座標+色）
        """
        try:
            import json

            # Noneまたは空文字列のチェック
            if obj1 is None or obj2 is None:
                return False
            if isinstance(obj1, str) and (not obj1 or obj1.strip() == ''):
                return False
            if isinstance(obj2, str) and (not obj2 or obj2.strip() == ''):
                return False

            # 文字列が非常に長い場合は早期終了（無限ループ防止）
            if isinstance(obj1, str) and len(obj1) > 1000000:  # 1MB以上
                return False
            if isinstance(obj2, str) and len(obj2) > 1000000:  # 1MB以上
                return False

            # JSON文字列をパース
            if isinstance(obj1, str):
                try:
                    obj1_data = json.loads(obj1)
                except json.JSONDecodeError:
                    return False
            else:
                obj1_data = obj1

            if isinstance(obj2, str):
                try:
                    obj2_data = json.loads(obj2)
                except json.JSONDecodeError:
                    return False
            else:
                obj2_data = obj2

            # 1. 位置が同じか
            if obj1_data.get('x') != obj2_data.get('x') or obj1_data.get('y') != obj2_data.get('y'):
                return False

            # 2. サイズが同じか
            if obj1_data.get('width') != obj2_data.get('width') or obj1_data.get('height') != obj2_data.get('height'):
                return False

            # 3. すべてのピクセルが同じか
            pixels1 = obj1_data.get('pixels', [])
            pixels2 = obj2_data.get('pixels', [])

            if len(pixels1) != len(pixels2):
                return False

            # ピクセル数が多すぎる場合は早期終了（無限ループ防止）
            if len(pixels1) > 100000:  # 10万ピクセル以上
                return False

            # ピクセルをセットに変換して比較（順序に依存しない）
            # 大きな配列の場合は段階的に比較
            if len(pixels1) > 10000:  # 1万ピクセル以上の場合はハッシュで比較
                pixels1_hash = hash(tuple(sorted(tuple(p) for p in pixels1)))
                pixels2_hash = hash(tuple(sorted(tuple(p) for p in pixels2)))
                return pixels1_hash == pixels2_hash
            else:
                pixels1_set = set(tuple(p) for p in pixels1)
                pixels2_set = set(tuple(p) for p in pixels2)
                return pixels1_set == pixels2_set

        except (json.JSONDecodeError, ValueError) as e:
            # JSONパースエラーは頻繁に発生する可能性があるため、デバッグレベルに変更
            # 空文字列や不正なJSON形式の場合はFalseを返す
            return False
        except Exception as e:
            # その他のエラーは警告レベルで記録
            #logger.warning(f"_objects_are_equal 予期しないエラー: {type(e).__name__}")
            return False

    def _extend_pattern(self, objects_input, side: str, count: int) -> List[str]:
        """パターンを検出して延長（イミュータブル）

        Args:
            objects_input: 単一オブジェクトID(str)または配列(List[str])
            side: 延長側 "start"(始点), "end"(終点), "both"(両端)
            count: 延長個数（bothの場合は各側の個数）

        Returns:
            List[str]: 延長後のオブジェクトID配列（元のオブジェクト + 新オブジェクト）
        """
        try:
            # 入力を配列に正規化
            if isinstance(objects_input, str):
                object_ids = [objects_input]
            else:
                object_ids = objects_input

            if not object_ids or len(object_ids) == 0:
                #logger.warning("EXTEND_PATTERN: 空の配列")
                return []

            # オブジェクトを取得
            objects = []
            for obj_id in object_ids:
                obj = self._find_object_by_id(obj_id)
                if obj:
                    objects.append(obj)

            if len(objects) < 2:
                #logger.warning("EXTEND_PATTERN: パターン検出には最低2個のオブジェクトが必要")
                return object_ids

            #logger.info(f"パターン延長開始: {len(objects)}個のオブジェクト, side={side}, count={count}")

            # 位置パターンを検出
            position_pattern = self._detect_position_pattern(objects)

            # 色パターンを検出
            color_pattern = self._detect_color_pattern(objects)

            #logger.info(f"検出パターン - 位置: {position_pattern}, 色: {color_pattern}")

            # 延長するオブジェクトを生成
            result_ids = list(object_ids)  # 元の配列をコピー

            # 位置パターン + 色パターンで延長
            if side in ["start", "both"]:
                # 始点側に延長
                start_objects = self._generate_extended_objects(
                    objects, position_pattern, color_pattern, "start", count
                )
                # 始点側のオブジェクトIDを先頭に追加
                result_ids = start_objects + result_ids

            if side in ["end", "both"]:
                # 終点側に延長
                end_objects = self._generate_extended_objects(
                    objects, position_pattern, color_pattern, "end", count
                )
                # 終点側のオブジェクトIDを末尾に追加
                result_ids = result_ids + end_objects

            #logger.info(f"パターン延長完了: {len(object_ids)}個 → {len(result_ids)}個")
            return result_ids

        except Exception as e:
            #logger.error(f"パターン延長エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return object_ids if isinstance(objects_input, list) else [objects_input]

    def _estimate_rotation_angle(self, obj1: Object, obj2: Object) -> Optional[int]:
        """2つのオブジェクト間の回転角度を推定（90度単位）

        Args:
            obj1: 1つ目のオブジェクト
            obj2: 2つ目のオブジェクト

        Returns:
            Optional[int]: 回転角度（90, 180, 270）、検出できない場合はNone
        """
        try:
            # ピクセル数が異なる場合は回転ではない
            if len(obj1.pixels) != len(obj2.pixels):
                return None

            # 正規化された座標に変換（bbox左上を原点に）
            def normalize_pixels(obj):
                if not obj.pixels:
                    return set()
                min_x = min(p[1] for p in obj.pixels)
                min_y = min(p[0] for p in obj.pixels)
                return set((p[0] - min_y, p[1] - min_x) for p in obj.pixels)

            pixels1 = normalize_pixels(obj1)
            pixels2 = normalize_pixels(obj2)

            if not pixels1 or not pixels2:
                return None

            # 90度回転の変換関数
            def rotate_90(pixels):
                # (y, x) -> (-x, y) -> 正規化
                rotated = [(-p[1], p[0]) for p in pixels]
                min_y = min(p[0] for p in rotated)
                min_x = min(p[1] for p in rotated)
                return set((p[0] - min_y, p[1] - min_x) for p in rotated)

            # 各角度で回転して比較
            test_pixels = pixels1
            for angle in [90, 180, 270]:
                test_pixels = rotate_90(test_pixels)
                if test_pixels == pixels2:
                    return angle

            return None

        except Exception as e:
            #logger.debug(f"回転角度推定エラー: {e}")
            return None

    def _detect_rotation_pattern(self, objects: List) -> dict:
        """回転パターンを検出（90度単位）

        Args:
            objects: オブジェクトのリスト

        Returns:
            dict: パターン情報
                - type: "rotation" または "none"
                - angle_delta: 回転角度の差分（90, 180, 270）
                - angles: 各オブジェクトの累積角度のリスト
        """
        try:
            if len(objects) < 2:
                return {"type": "none"}

            # ピクセル数が全て同じかチェック
            pixel_counts = [len(obj.pixels) for obj in objects]
            if len(set(pixel_counts)) > 1:
                return {"type": "none"}

            # 1x1ピクセルのオブジェクト（または全てのオブジェクトが1ピクセル）の場合は回転パターンを検出しない
            # 1x1ピクセルは回転しても同じ形状のため、位置パターンで延長すべき
            if all(count == 1 for count in pixel_counts):
                return {"type": "none"}

            # 連続するオブジェクト間の回転角度を推定
            angle_deltas = []
            for i in range(len(objects) - 1):
                angle = self._estimate_rotation_angle(objects[i], objects[i + 1])
                if angle is None:
                    return {"type": "none"}
                angle_deltas.append(angle)

            # 全ての角度差が同じかチェック（等間隔回転）
            if len(set(angle_deltas)) != 1:
                return {"type": "none"}

            angle_delta = angle_deltas[0]

            # 累積角度を計算
            cumulative_angles = [0]
            for delta in angle_deltas:
                cumulative_angles.append((cumulative_angles[-1] + delta) % 360)

            #logger.info(f"回転パターン検出: {angle_delta}度ずつ回転, 累積角度={cumulative_angles}")

            return {
                "type": "rotation",
                "angle_delta": angle_delta,
                "angles": cumulative_angles
            }

        except Exception as e:
            #logger.error(f"回転パターン検出エラー: {e}")
            return {"type": "none"}

    def _detect_position_pattern(self, objects: List) -> dict:
        """位置パターンを検出"""
        try:
            positions = [(obj.bbox_left, obj.bbox_top) for obj in objects]

            if len(positions) < 2:
                return {"type": "none"}

            # 差分ベクトルを計算
            deltas = []
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                deltas.append((dx, dy))

            # 等間隔パターン（差分が一定）
            if len(set(deltas)) == 1:
                return {
                    "type": "uniform",
                    "delta": deltas[0],
                    "positions": positions
                }

            # 加速度パターン（差分の差分が一定）
            if len(deltas) >= 2:
                delta_deltas = []
                for i in range(1, len(deltas)):
                    ddx = deltas[i][0] - deltas[i-1][0]
                    ddy = deltas[i][1] - deltas[i-1][1]
                    delta_deltas.append((ddx, ddy))

                if len(set(delta_deltas)) == 1:
                    return {
                        "type": "acceleration",
                        "delta": deltas[-1],
                        "acceleration": delta_deltas[0],
                        "positions": positions
                    }

            # その他（平均差分を使用）
            avg_dx = sum(d[0] for d in deltas) / len(deltas)
            avg_dy = sum(d[1] for d in deltas) / len(deltas)
            return {
                "type": "average",
                "delta": (int(round(avg_dx)), int(round(avg_dy))),
                "positions": positions
            }

        except Exception as e:
            #logger.error(f"位置パターン検出エラー: {e}")
            return {"type": "none"}

    def _detect_color_pattern(self, objects: List) -> dict:
        """色パターンを検出"""
        try:
            colors = [obj.dominant_color for obj in objects]

            if len(colors) < 2:
                return {"type": "none"}

            # 全て同じ色
            if len(set(colors)) == 1:
                return {"type": "constant", "value": colors[0]}

            # 2個の場合は周期2として扱う（特別処理）
            if len(colors) == 2:
                return {"type": "periodic", "pattern": colors, "period": 2}

            # 周期パターンを検出（3個以上）
            for period in range(2, len(colors) // 2 + 1):
                pattern = colors[:period]
                is_periodic = True
                for i in range(period, len(colors)):
                    if colors[i] != pattern[i % period]:
                        is_periodic = False
                        break
                if is_periodic:
                    return {"type": "periodic", "pattern": pattern, "period": period}

            # 等差数列
            if len(colors) >= 3:
                diffs = [colors[i+1] - colors[i] for i in range(len(colors)-1)]
                if len(set(diffs)) == 1:
                    return {"type": "arithmetic", "diff": diffs[0], "last": colors[-1]}

            # その他（繰り返し）
            return {"type": "repeat", "sequence": colors}

        except Exception as e:
            #logger.error(f"色パターン検出エラー: {e}")
            return {"type": "none"}

    def _detect_size_pattern(self, objects: List) -> dict:
        """サイズパターンを検出"""
        try:
            sizes = [(obj.bbox_width, obj.bbox_height) for obj in objects]

            if len(sizes) < 2:
                return {"type": "none"}

            # 全て同じサイズ
            if len(set(sizes)) == 1:
                return {"type": "constant", "value": sizes[0]}

            # 幅の等差数列
            widths = [s[0] for s in sizes]
            heights = [s[1] for s in sizes]

            width_diffs = [widths[i+1] - widths[i] for i in range(len(widths)-1)]
            height_diffs = [heights[i+1] - heights[i] for i in range(len(heights)-1)]

            if len(set(width_diffs)) == 1 and len(set(height_diffs)) == 1:
                return {
                    "type": "arithmetic",
                    "width_diff": width_diffs[0],
                    "height_diff": height_diffs[0],
                    "last": sizes[-1]
                }

            # その他（最後のサイズを維持）
            return {"type": "repeat", "value": sizes[-1]}

        except Exception as e:
            #logger.error(f"サイズパターン検出エラー: {e}")
            return {"type": "none"}

    def _generate_rotation_extended_objects(self, objects: List, rotation_pattern: dict,
                                           position_pattern: dict, direction: str, count: int) -> List[str]:
        """回転パターンで延長オブジェクトを生成

        Args:
            objects: 元のオブジェクトリスト
            rotation_pattern: 回転パターン情報
            position_pattern: 位置パターン情報（複合パターン用）
            direction: "start" or "end"
            count: 生成個数

        Returns:
            List[str]: 新しいオブジェクトIDのリスト
        """
        try:
            new_object_ids = []
            angle_delta = rotation_pattern["angle_delta"]
            base_obj = objects[-1] if direction == "end" else objects[0]
            current_angle = rotation_pattern["angles"][-1] if direction == "end" else rotation_pattern["angles"][0]

            for i in range(count):
                # 回転角度を計算
                if direction == "end":
                    new_angle = (current_angle + angle_delta * (i + 1)) % 360
                else:
                    new_angle = (current_angle - angle_delta * (i + 1)) % 360

                # 回転オブジェクトを生成
                rotated_id = self._rotate(base_obj.object_id, new_angle)

                if rotated_id:
                    # 位置パターンがある場合は移動も適用
                    if position_pattern["type"] == "uniform":
                        delta = position_pattern["delta"]
                        if direction == "start":
                            offset_x = -delta[0] * (i + 1)
                            offset_y = -delta[1] * (i + 1)
                            new_x = objects[0].bbox_left + offset_x
                            new_y = objects[0].bbox_top + offset_y
                        else:
                            offset_x = delta[0] * (i + 1)
                            offset_y = delta[1] * (i + 1)
                            new_x = objects[-1].bbox_left + offset_x
                            new_y = objects[-1].bbox_top + offset_y

                        # テレポートで位置を調整
                        rotated_id = self._teleport(rotated_id, new_x, new_y)

                    new_object_ids.append(rotated_id)
                    #logger.info(f"回転オブジェクト生成: 角度={new_angle}度")

            return new_object_ids

        except Exception as e:
            #logger.error(f"回転オブジェクト生成エラー: {e}")
            return []

    def _generate_extended_objects(self, objects: List, position_pattern: dict,
                                   color_pattern: dict,
                                   direction: str, count: int) -> List[str]:
        """延長オブジェクトを生成"""
        try:
            new_object_ids = []
            base_obj = objects[-1] if direction == "end" else objects[0]

            for i in range(count):
                # 位置を計算
                if position_pattern["type"] == "uniform":
                    delta = position_pattern["delta"]
                    if direction == "start":
                        # 逆方向
                        offset_x = -delta[0] * (i + 1)
                        offset_y = -delta[1] * (i + 1)
                        new_x = objects[0].bbox_left + offset_x
                        new_y = objects[0].bbox_top + offset_y
                    else:
                        # 順方向
                        offset_x = delta[0] * (i + 1)
                        offset_y = delta[1] * (i + 1)
                        new_x = objects[-1].bbox_left + offset_x
                        new_y = objects[-1].bbox_top + offset_y

                elif position_pattern["type"] == "acceleration":
                    delta = position_pattern["delta"]
                    accel = position_pattern["acceleration"]
                    if direction == "start":
                        # 加速度を逆方向に適用
                        offset_x = -delta[0] - accel[0] * i
                        offset_y = -delta[1] - accel[1] * i
                        new_x = objects[0].bbox_left + offset_x
                        new_y = objects[0].bbox_top + offset_y
                    else:
                        # 加速度を順方向に適用
                        offset_x = delta[0] + accel[0] * i
                        offset_y = delta[1] + accel[1] * i
                        new_x = objects[-1].bbox_left + offset_x
                        new_y = objects[-1].bbox_top + offset_y

                else:  # average or none
                    delta = position_pattern.get("delta", (5, 0))
                    if direction == "start":
                        new_x = objects[0].bbox_left - delta[0] * (i + 1)
                        new_y = objects[0].bbox_top - delta[1] * (i + 1)
                    else:
                        new_x = objects[-1].bbox_left + delta[0] * (i + 1)
                        new_y = objects[-1].bbox_top + delta[1] * (i + 1)

                # 色を計算
                if color_pattern["type"] == "constant":
                    new_color = color_pattern["value"]
                elif color_pattern["type"] == "periodic":
                    pattern = color_pattern["pattern"]
                    if direction == "start":
                        # 逆方向のインデックス
                        idx = (len(objects) - 1 - i) % len(pattern)
                    else:
                        idx = (len(objects) + i) % len(pattern)
                    new_color = pattern[idx]
                elif color_pattern["type"] == "arithmetic":
                    diff = color_pattern["diff"]
                    if direction == "start":
                        new_color = objects[0].dominant_color - diff * (i + 1)
                    else:
                        new_color = color_pattern["last"] + diff * (i + 1)
                elif color_pattern["type"] == "repeat":
                    # sequenceを繰り返す
                    sequence = color_pattern["sequence"]
                    if direction == "start":
                        idx = (len(objects) - 1 - i) % len(sequence)
                    else:
                        idx = (len(objects) + i) % len(sequence)
                    new_color = sequence[idx]
                else:
                    new_color = base_obj.dominant_color

                # 元のオブジェクトの形状を保持（サイズパターンは廃止）
                # 元のオブジェクトのピクセルをコピーして新しい位置に移動
                base_x = base_obj.bbox_left
                base_y = base_obj.bbox_top

                # 元のオブジェクトのピクセルを新しい位置に移動
                new_pixels = []
                new_pixel_colors = {}
                has_pixel_colors = hasattr(base_obj, 'pixel_colors') and base_obj.pixel_colors

                for px, py in base_obj.pixels:
                    # 相対位置を計算
                    rel_x = px - base_x
                    rel_y = py - base_y
                    # 新しい位置に移動
                    new_px = new_x + rel_x
                    new_py = new_y + rel_y
                    new_pixels.append((new_px, new_py))
                    # 色を設定
                    if has_pixel_colors and (px, py) in base_obj.pixel_colors:
                        new_pixel_colors[(new_px, new_py)] = new_color
                    else:
                        new_pixel_colors[(new_px, new_py)] = new_color

                # 新しいオブジェクトを作成
                new_obj_id = self._generate_unique_object_id(f"{base_obj.object_id}_extend_{i}")

                # バウンディングボックスを計算
                if new_pixels:
                    xs = [p[0] for p in new_pixels]
                    ys = [p[1] for p in new_pixels]
                    new_bbox = (min(xs), min(ys), max(xs), max(ys))
                else:
                    new_bbox = base_obj.bbox

                new_obj = Object(
                    object_id=new_obj_id,
                    pixels=new_pixels,
                    pixel_colors=new_pixel_colors,
                    bbox=new_bbox,
                    object_type=base_obj.object_type,
                )
                self.execution_context['objects'][new_obj_id] = new_obj

                if new_obj_id:
                    new_object_ids.append(new_obj_id)
                    #logger.info(f"延長オブジェクト生成: {new_obj_id} at ({new_x},{new_y}), color={new_color}")

            # start方向の場合は逆順にする
            if direction == "start":
                new_object_ids.reverse()

            return new_object_ids

        except Exception as e:
            #logger.error(f"延長オブジェクト生成エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return []

    # ==================== 高階関数（Higher-Order Functions） ====================

    def _evaluate_condition(self, condition_expr: str, obj_id: str, *extra_args) -> bool:
        """条件式を評価（変数展開サポート、複数引数コマンド対応）

        Args:
            condition_expr: 条件式文字列（例: "GREATER(GET_SIZE, {threshold})"）
            obj_id: 評価対象のオブジェクトID
            *extra_args: 複数引数コマンド用の追加引数

        Returns:
            bool: 条件の真偽値
        """
        try:
            # 条件式を解析して評価
            import re

            # ステップ1: 変数展開（{var_name}を実際の値に置換）
            expanded_expr = condition_expr
            for match in re.finditer(r'\{(\w+)\}', condition_expr):
                var_name = match.group(1)
                if var_name in self.execution_context.get('variables', {}):
                    value = self.execution_context['variables'][var_name]
                    expanded_expr = expanded_expr.replace(f'{{{var_name}}}', str(value))
                #else:
                    #logger.warning(f"変数展開: 変数 '{var_name}' が見つかりません")

            condition_expr = expanded_expr
            #logger.info(f"変数展開後の条件式: {condition_expr}")

            # ステップ2: オブジェクトを取得
            obj = self._find_object_by_id(obj_id)
            if not obj:
                return False

            # ステップ3: 複数引数コマンドの評価（GET_DISTANCE, GET_SYMMETRY_SCORE）
            # GET_DISTANCE を実際の値に置換
            if "GET_DISTANCE" in condition_expr and len(extra_args) > 0:
                ref_obj_id = extra_args[0]
                distance = self._get_distance_between_objects(obj_id, ref_obj_id)
                condition_expr = condition_expr.replace("GET_DISTANCE", str(distance))

            # GET_SYMMETRY_SCORE を実際の値に置換（axisが指定されている場合）
            if "GET_SYMMETRY_SCORE" in condition_expr and len(extra_args) > 0:
                axis = extra_args[0] if isinstance(extra_args[0], str) else 'vertical'
                symmetry_score = self._get_symmetry_score(obj_id, axis)
                condition_expr = condition_expr.replace("GET_SYMMETRY_SCORE", str(symmetry_score))

            # ステップ4: GET_コマンドを実際の値に置換

            # GET_SIZE を実際の値に置換
            if "GET_SIZE" in condition_expr:
                pixel_count = len(obj.pixels) if obj.pixels else 0
                condition_expr = condition_expr.replace("GET_SIZE", str(pixel_count))


            # GET_COLOR を実際の値に置換
            if "GET_COLOR" in condition_expr:
                colors = self._get_color_list(obj_id)
                color = colors[0] if colors else 0
                condition_expr = condition_expr.replace("GET_COLOR", str(color))

            # GET_WIDTH を実際の値に置換
            if "GET_WIDTH" in condition_expr:
                width = obj.bbox_width
                condition_expr = condition_expr.replace("GET_WIDTH", str(width))

            # GET_HEIGHT を実際の値に置換
            if "GET_HEIGHT" in condition_expr:
                height = obj.bbox_height
                condition_expr = condition_expr.replace("GET_HEIGHT", str(height))

            # GET_X を実際の値に置換
            if "GET_X" in condition_expr:
                x = obj.bbox_left
                condition_expr = condition_expr.replace("GET_X", str(x))

            # GET_Y を実際の値に置換
            if "GET_Y" in condition_expr:
                y = obj.bbox_top
                condition_expr = condition_expr.replace("GET_Y", str(y))

            # COUNT_HOLES を実際の値に置換
            if "COUNT_HOLES" in condition_expr:
                hole_count = self._get_hole_count(obj_id)
                condition_expr = condition_expr.replace("COUNT_HOLES", str(hole_count))

            # GET_SYMMETRY_SCOREは複数引数コマンドとして上で処理済み（extra_argsがない場合のみデフォルト処理）
            if "GET_SYMMETRY_SCORE" in condition_expr and len(extra_args) == 0:
                symmetry_score = self._get_symmetry_score(obj_id, 'vertical')
                condition_expr = condition_expr.replace("GET_SYMMETRY_SCORE", str(symmetry_score))






            # IS_INSIDE を実際の値に置換（本格実装：オブジェクトが矩形範囲内にあるか）
            if "IS_INSIDE" in condition_expr:
                is_in_rectangle = False
                if len(extra_args) >= 4:
                    # 矩形の座標が指定されている場合
                    x1, y1, x2, y2 = extra_args[0], extra_args[1], extra_args[2], extra_args[3]
                    # オブジェクトのバウンディングボックスが矩形範囲内にあるかチェック
                    obj_x1, obj_y1 = obj.bbox_left, obj.bbox_top
                    obj_x2, obj_y2 = obj.bbox_left + obj.bbox_width, obj.bbox_top + obj.bbox_height

                    # 矩形の範囲を正規化
                    rect_x1, rect_x2 = min(x1, x2), max(x1, x2)
                    rect_y1, rect_y2 = min(y1, y2), max(y1, y2)

                    # オブジェクトが矩形範囲内にあるかチェック
                    is_in_rectangle = (obj_x1 >= rect_x1 and obj_x2 <= rect_x2 and
                                     obj_y1 >= rect_y1 and obj_y2 <= rect_y2)
                elif obj.pixels:
                    # パラメータが指定されていない場合は中心座標を使用
                    center_x = sum(p[0] for p in obj.pixels) // len(obj.pixels)
                    center_y = sum(p[1] for p in obj.pixels) // len(obj.pixels)
                    # デフォルトの矩形範囲（0,0から10,10）でチェック
                    is_in_rectangle = (0 <= center_x <= 10 and 0 <= center_y <= 10)
                condition_expr = condition_expr.replace("IS_INSIDE", str(is_in_rectangle).lower())

            # 比較演算子を評価
            # GREATER(a, b) → a > b
            if "GREATER(" in condition_expr:
                match = re.search(r'GREATER\(([^,]+),\s*([^)]+)\)', condition_expr)
                if match:
                    a, b = match.group(1).strip(), match.group(2).strip()
                    result = float(a) > float(b)
                    return result

            # LESS(a, b) → a < b
            if "LESS(" in condition_expr:
                match = re.search(r'LESS\(([^,]+),\s*([^)]+)\)', condition_expr)
                if match:
                    a, b = match.group(1).strip(), match.group(2).strip()
                    result = float(a) < float(b)
                    return result

            # EQUAL(a, b) → a == b
            if "EQUAL(" in condition_expr:
                match = re.search(r'EQUAL\(([^,]+),\s*([^)]+)\)', condition_expr)
                if match:
                    a, b = match.group(1).strip(), match.group(2).strip()
                    # 数値比較を試みる
                    try:
                        # NumPy配列の場合に備えて、スカラー値に変換
                        import numpy as np
                        if isinstance(a, np.ndarray):
                            a = a.item() if a.size == 1 else str(a)
                        if isinstance(b, np.ndarray):
                            b = b.item() if b.size == 1 else str(b)
                        result = float(a) == float(b)
                    except:
                        result = a == b
                    return result

            # NOT_EQUAL(a, b) → a != b
            if "NOT_EQUAL(" in condition_expr:
                match = re.search(r'NOT_EQUAL\(([^,]+),\s*([^)]+)\)', condition_expr)
                if match:
                    a, b = match.group(1).strip(), match.group(2).strip()
                    try:
                        result = float(a) != float(b)
                    except:
                        result = a != b
                    return result


            # AND(a, b)
            if "AND(" in condition_expr:
                # 再帰的に評価
                match = re.search(r'AND\(([^,]+),\s*(.+)\)', condition_expr)
                if match:
                    cond1 = match.group(1).strip()
                    cond2 = match.group(2).strip()
                    result1 = self._evaluate_condition(cond1, obj_id)
                    result2 = self._evaluate_condition(cond2, obj_id)
                    return result1 and result2

            # OR(a, b)
            if "OR(" in condition_expr:
                match = re.search(r'OR\(([^,]+),\s*(.+)\)', condition_expr)
                if match:
                    cond1 = match.group(1).strip()
                    cond2 = match.group(2).strip()
                    result1 = self._evaluate_condition(cond1, obj_id)
                    result2 = self._evaluate_condition(cond2, obj_id)
                    return result1 or result2

            #logger.warning(f"条件式評価: 未対応の形式 - {condition_expr}")
            return False

        except Exception as e:
            #logger.error(f"条件式評価エラー: {e} (条件式: {condition_expr})")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return False

    def _filter(self, object_ids: List[str], condition_expr: str, *extra_args) -> List[str]:
        """条件式で配列をフィルタ（Filter関数）

        Args:
            object_ids: オブジェクトID配列
            condition_expr: 条件式文字列
            *extra_args: 複数引数コマンド用の追加引数（GET_DISTANCE用のref_obj、GET_SYMMETRY_SCORE用のaxisなど）

        Returns:
            List[str]: フィルタ後のオブジェクトID配列
        """
        try:
            #logger.info(f"FILTER開始: {len(object_ids)}個のオブジェクトを条件「{condition_expr}」でフィルタ（追加引数: {extra_args}）")
            result_ids = []

            for obj_id in object_ids:
                if self._evaluate_condition(condition_expr, obj_id, *extra_args):
                    result_ids.append(obj_id)

            #logger.info(f"FILTER完了: {len(result_ids)}個のオブジェクトが条件を満たす")
            return result_ids

        except Exception as e:
            #logger.error(f"FILTERエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return object_ids

    # ==================== 統一構文用メソッド（新実装） ====================

    def _sort_by_unified(self, object_ids: List[str], key_command, order: str, interpreter) -> List[str]:
        """統一構文でのソート

        Args:
            object_ids: オブジェクトID配列
            key_command: キーコマンド（FunctionCallのみ、$objを含む）
            order: "asc" または "desc"
            interpreter: Interpreterインスタンス

        Returns:
            List[str]: ソート後のオブジェクトID配列

        注意: v3.0より$objが必須になりました。
              単一引数: GET_SIZE($obj)
              複数引数: GET_DISTANCE($obj, ref_obj)
        """
        try:
            from .parsing.parser import Identifier, FunctionCall, Placeholder

            # key_commandから評価関数を作成
            if isinstance(key_command, Identifier):
                # 注: 単一引数コマンド（例: GET_SIZE）は$objなしでも動作する
                command_name = key_command.name
                #logger.warning(f"SORT_BY: $objなしの構文は非推奨です。GET_{command_name}($obj)を使用してください。")

                def key_func(obj_id):
                    return interpreter.execute_command(command_name, [obj_id])

            elif isinstance(key_command, FunctionCall):
                # 推奨: FunctionCall形式（$objを含む）
                command_name = key_command.name

                def key_func(obj_id):
                    # 引数を評価（Placeholderを現在のobj_idに置き換え）
                    evaluated_args = []
                    for arg in key_command.arguments:
                        if isinstance(arg, Placeholder):
                            # $obj -> 現在のオブジェクトID
                            evaluated_args.append(obj_id)
                        else:
                            evaluated_args.append(interpreter.evaluate_expression(arg))

                    return interpreter.execute_command(command_name, evaluated_args)
            else:
                #logger.error(f"不正なkey_command型: {type(key_command)}")
                return object_ids

            # ソート実行
            sorted_ids = sorted(object_ids, key=key_func, reverse=(order == "desc"))
            #logger.info(f"SORT_BY完了: {len(sorted_ids)}個のオブジェクトをソート")
            return sorted_ids

        except Exception as e:
            #logger.error(f"SORT_BYエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return object_ids

    def _filter_unified(self, object_ids: List[str], condition_expr, interpreter) -> List[str]:
        """統一構文でのフィルタ

        Args:
            object_ids: オブジェクトID配列
            condition_expr: 条件式（関数呼び出し、例: GREATER(GET_SIZE, 10)）
            interpreter: Interpreterインスタンス

        Returns:
            List[str]: フィルタ後のオブジェクトID配列
        """
        try:
            from .parsing.parser import FunctionCall

            #logger.info(f"[DEBUG] FILTER開始: 入力オブジェクト数={len(object_ids)}")

            # 入力オブジェクトの色分布をログ出力
            if len(object_ids) > 0:
                input_color_counts = {}
                for obj_id in object_ids:
                    obj = self._find_object_by_id(obj_id)
                    if obj:
                        obj_color = obj.color if hasattr(obj, 'color') else None
                        if obj_color is not None:
                            input_color_counts[obj_color] = input_color_counts.get(obj_color, 0) + 1
                #logger.info(f"[DEBUG] FILTER入力オブジェクトの色分布: {input_color_counts}")

            if not isinstance(condition_expr, FunctionCall):
                #logger.error(f"不正なcondition_expr型: {type(condition_expr)}")
                return object_ids

            # 条件式の情報をログ出力
            condition_str = str(condition_expr) if hasattr(condition_expr, '__str__') else str(type(condition_expr))
            #logger.info(f"[DEBUG] FILTER条件式: {condition_str}")

            result_ids = []
            passed_count = 0
            failed_count = 0

            for obj_id in object_ids:
                # 条件式を評価（obj_idをコンテキストに設定）
                result = self._evaluate_unified_condition(condition_expr, obj_id, interpreter)
                if result:
                    result_ids.append(obj_id)
                    passed_count += 1
                else:
                    failed_count += 1

            #logger.info(f"[DEBUG] FILTER完了: 条件を満たした={passed_count}個, 条件を満たさなかった={failed_count}個, 結果={len(result_ids)}個")

            # 結果オブジェクトの色分布をログ出力
            if len(result_ids) > 0:
                result_color_counts = {}
                for obj_id in result_ids:
                    obj = self._find_object_by_id(obj_id)
                    if obj:
                        obj_color = obj.color if hasattr(obj, 'color') else None
                        if obj_color is not None:
                            result_color_counts[obj_color] = result_color_counts.get(obj_color, 0) + 1
                #logger.info(f"[DEBUG] FILTER結果オブジェクトの色分布: {result_color_counts}")
            #else:
                #logger.warning(f"[DEBUG] FILTER結果: すべてのオブジェクトが除外されました（結果数=0）")

            return result_ids

        except Exception as e:
            #logger.error(f"FILTERエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return object_ids

    def _evaluate_unified_condition(self, condition_expr, obj_id: str, interpreter) -> bool:
        """統一構文での条件評価

        Args:
            condition_expr: 条件式（FunctionCall）
            obj_id: 評価対象のオブジェクトID
            interpreter: Interpreterインスタンス

        Returns:
            bool: 条件の真偽値
        """
        try:
            from .parsing.parser import FunctionCall, Identifier, Literal, Placeholder

            if not isinstance(condition_expr, FunctionCall):
                return False

            # $objプレースホルダーを設定
            interpreter.variables['$obj'] = obj_id
            interpreter.current_object_id = obj_id

            command_name = condition_expr.name

            # 比較演算子の処理
            if command_name in ["GREATER", "LESS", "EQUAL", "NOT_EQUAL"]:
                # 第1引数: GETコマンド、第2引数: 比較値
                if len(condition_expr.arguments) < 2:
                    return False

                left_expr = condition_expr.arguments[0]
                right_expr = condition_expr.arguments[1]

                # 左辺を評価（GETコマンド）
                left_value = self._evaluate_get_command_unified(left_expr, obj_id, interpreter)

                # 右辺を評価（GETコマンド、リテラル、または変数）
                from .parsing.parser import Identifier, FunctionCall
                if isinstance(right_expr, Identifier) and right_expr.name.startswith('GET_'):
                    # 右辺もGETコマンド
                    # 引数なしコマンドは直接実行、それ以外はobj_idを渡す
                    if right_expr.name in ['GET_BACKGROUND_COLOR', 'GET_INPUT_GRID_SIZE']:
                        right_value = interpreter.execute_command(right_expr.name, [])
                    else:
                        right_value = self._evaluate_get_command_unified(right_expr, obj_id, interpreter)
                elif isinstance(right_expr, FunctionCall):
                    # 関数呼び出しの場合は通常の評価
                    # $objが含まれている可能性があるため、再度設定
                    interpreter.variables['$obj'] = obj_id
                    interpreter.current_object_id = obj_id
                    right_value = interpreter.evaluate_expression(right_expr)
                else:
                    # リテラルまたは変数
                    # $objが含まれている可能性があるため、設定
                    interpreter.variables['$obj'] = obj_id
                    interpreter.current_object_id = obj_id
                    right_value = interpreter.evaluate_expression(right_expr)

                # NOT_EQUALの場合は詳細ログを出力
                if command_name == "NOT_EQUAL":
                    obj = self._find_object_by_id(obj_id)
                    obj_color = obj.color if obj and hasattr(obj, 'color') else 'N/A'
                    #logger.info(f"[DEBUG] NOT_EQUAL評価: obj_id={obj_id}, obj_color={obj_color}, left_value={left_value}, right_value={right_value}, 結果={left_value != right_value}")

                # 比較実行
                if command_name == "GREATER":
                    try:
                        return float(left_value) > float(right_value)
                    except (ValueError, TypeError):
                        # 文字列やNoneが比較できない場合はFalseを返す（エラーを発生させない）
                        return False
                elif command_name == "LESS":
                    try:
                        return float(left_value) < float(right_value)
                    except (ValueError, TypeError):
                        # 文字列やNoneが比較できない場合はFalseを返す（エラーを発生させない）
                        return False
                elif command_name == "EQUAL":
                    try:
                        return float(left_value) == float(right_value)
                    except (ValueError, TypeError):
                        return left_value == right_value
                elif command_name == "NOT_EQUAL":
                    try:
                        return float(left_value) != float(right_value)
                    except:
                        return left_value != right_value

            # 論理演算子の処理
            elif command_name == "AND":
                if len(condition_expr.arguments) < 2:
                    return False
                result1 = self._evaluate_unified_condition(condition_expr.arguments[0], obj_id, interpreter)
                result2 = self._evaluate_unified_condition(condition_expr.arguments[1], obj_id, interpreter)
                return result1 and result2

            elif command_name == "OR":
                if len(condition_expr.arguments) < 2:
                    return False
                result1 = self._evaluate_unified_condition(condition_expr.arguments[0], obj_id, interpreter)
                result2 = self._evaluate_unified_condition(condition_expr.arguments[1], obj_id, interpreter)
                return result1 or result2

            elif command_name == "NOT":
                if len(condition_expr.arguments) < 1:
                    return False
                result = self._evaluate_unified_condition(condition_expr.arguments[0], obj_id, interpreter)
                return not result

            # 真偽値を返すコマンド
            else:
                result = self._evaluate_get_command_unified(condition_expr, obj_id, interpreter)
                return bool(result)

            return False

        except Exception as e:
            #logger.error(f"条件評価エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return False

    def _evaluate_get_command_unified(self, expr, obj_id: str, interpreter):
        """統一構文でのGETコマンド評価

        Args:
            expr: 式（IdentifierまたはFunctionCall）
            obj_id: オブジェクトID
            interpreter: Interpreterインスタンス

        Returns:
            Any: コマンドの実行結果
        """
        try:
            from .parsing.parser import Identifier, FunctionCall, Placeholder

            if isinstance(expr, Identifier):
                # 引数なしコマンドのチェック
                if expr.name in ['GET_BACKGROUND_COLOR', 'GET_INPUT_GRID_SIZE']:
                    # 引数なしのコマンド（括弧なしでも動作するが非推奨）
                    # $objを設定（他のコマンドで使用される可能性があるため）
                    interpreter.variables['$obj'] = obj_id
                    interpreter.current_object_id = obj_id
                    return interpreter.execute_command(expr.name, [])
                else:
                    # 単一引数コマンド（例: GET_SIZE）
                    # $objを設定
                    interpreter.variables['$obj'] = obj_id
                    interpreter.current_object_id = obj_id
                    return interpreter.execute_command(expr.name, [obj_id])

            elif isinstance(expr, FunctionCall):
                # 複数引数コマンド（例: GET_DISTANCE($obj, ref_obj)）
                command_name = expr.name

                # 引数を評価（Placeholderを現在のobj_idに置き換え）
                evaluated_args = []
                for arg in expr.arguments:
                    if isinstance(arg, Placeholder):
                        # $obj -> 現在のオブジェクトID
                        evaluated_args.append(obj_id)
                    else:
                        # $objが含まれている可能性があるため、設定
                        interpreter.variables['$obj'] = obj_id
                        interpreter.current_object_id = obj_id
                        evaluated_args.append(interpreter.evaluate_expression(arg))

                # コマンドを実行
                return interpreter.execute_command(command_name, evaluated_args)

            else:
                #logger.error(f"不正なexpr型: {type(expr)}")
                return 0

        except Exception as e:
            #logger.error(f"GETコマンド評価エラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return 0

    def _fit_shape(self, obj1_id: str, obj2_id: str) -> str:
        """obj1をobj2に対して形状（位置）をフィット

        位置の重複を最大化（色は無視）
        回転あり（0/90/180/270度）
        スコアリング改良版

        Args:
            obj1_id: 移動・回転するオブジェクトのID
            obj2_id: 基準となるオブジェクトのID（固定）

        Returns:
            新しいオブジェクトID
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                #logger.error(f"FIT_SHAPE: オブジェクトが見つかりません")
                return obj1_id

            best_score = -1
            best_pixels = None
            best_rotation = 0
            best_dx = 0
            best_dy = 0
            best_flip = None

            # 探索範囲を設定（グリッドサイズに基づく、最大30に制限）
            grid_size = self.grid_context.get_input_grid_size() or (30, 30)
            search_range = min(max(grid_size[0], grid_size[1]), 30)

            # obj2の左上角を計算（ROTATE/SCALEコマンドと同じ基準）
            obj2_pixels = [(p[0], p[1]) for p in obj2.pixels]
            obj2_top_left_x = min(p[0] for p in obj2_pixels) if obj2_pixels else 0
            obj2_top_left_y = min(p[1] for p in obj2_pixels) if obj2_pixels else 0

            # すべての反転パターンを試す（None, "X", "Y"）
            for flip_type in [None, "X", "Y"]:
                # 反転処理
                if flip_type is None:
                    # 反転なし
                    flipped_pixels = [(p[0], p[1], obj1.dominant_color) for p in obj1.pixels]
                else:
                    # 反転あり
                    temp_obj_id = self._flip(obj1_id, flip_type)
                    if temp_obj_id:
                        temp_obj = self._find_object_by_id(temp_obj_id)
                        if temp_obj:
                            flipped_pixels = [(p[0], p[1], obj1.dominant_color) for p in temp_obj.pixels]
                        else:
                            continue
                    else:
                        continue

                # すべての回転を試す
                for rotation_index, angle in enumerate([0, 90, 180, 270]):
                    # 反転後のピクセルを回転
                    if angle == 0:
                        # 0度の場合
                        rotated_pixels = flipped_pixels
                    else:
                        # 回転前の左上角位置を記録（位置調整用）
                        original_top_left_x = min(p[0] for p in flipped_pixels)
                        original_top_left_y = min(p[1] for p in flipped_pixels)

                        # 回転処理（整数演算、最小座標基準）
                        base_x = original_top_left_x
                        base_y = original_top_left_y

                        rotated_pixels = []
                        for p in flipped_pixels:
                            px, py = p[0], p[1]
                            color = p[2] if len(p) > 2 else obj1.dominant_color

                            # 最小座標からの相対座標
                            rx = px - base_x
                            ry = py - base_y

                            # 回転（整数演算）
                            if rotation_index == 1:  # 90度
                                new_rx = -ry
                                new_ry = rx
                            elif rotation_index == 2:  # 180度
                                new_rx = -rx
                                new_ry = -ry
                            else:  # 270度
                                new_rx = ry
                                new_ry = -rx

                            new_px = base_x + new_rx
                            new_py = base_y + new_ry

                            rotated_pixels.append((new_px, new_py, color))

                        # 位置調整: 回転後のバウンディングボックスの左上を元の位置にそろえる
                        if rotated_pixels:
                            new_min_x = min(p[0] for p in rotated_pixels)
                            new_min_y = min(p[1] for p in rotated_pixels)
                            offset_x = original_top_left_x - new_min_x
                            offset_y = original_top_left_y - new_min_y

                            # すべてのピクセルを調整
                            adjusted_pixels = []
                            for px, py, color in rotated_pixels:
                                adjusted_pixels.append((px + offset_x, py + offset_y, color))

                            rotated_pixels = adjusted_pixels

                    # 回転後の左上角を計算（ROTATE/SCALEコマンドと同じ基準）
                    if rotated_pixels:
                        rotated_top_left_x = min(p[0] for p in rotated_pixels)
                        rotated_top_left_y = min(p[1] for p in rotated_pixels)
                    else:
                        continue

                    # obj2の周囲を探索（左上角基準）
                    for dx in range(-search_range, search_range + 1):
                        for dy in range(-search_range, search_range + 1):
                            # 移動後の位置を計算（左上角を基準に配置）
                            offset_x = obj2_top_left_x + dx - rotated_top_left_x
                            offset_y = obj2_top_left_y + dy - rotated_top_left_y

                            # ピクセルを移動
                            moved_pixels = []
                            for p in rotated_pixels:
                                new_x = p[0] + offset_x
                                new_y = p[1] + offset_y
                                if len(p) > 2:
                                    moved_pixels.append((new_x, new_y, p[2]))
                                else:
                                    moved_pixels.append((new_x, new_y))

                            # スコア計算（位置のみ、flip_typeを渡す）
                            score = self._calculate_fit_shape_score(
                                moved_pixels, obj2_pixels, rotation_index, offset_x, offset_y, flip_type
                            )

                            # 最高スコアを更新
                            if score > best_score:
                                best_score = score
                                best_pixels = moved_pixels
                                best_rotation = angle
                                best_dx = offset_x
                                best_dy = offset_y
                                best_flip = flip_type

            # 最適な配置でオブジェクトを作成
            new_obj_id = self._generate_unique_object_id(f"{obj1.object_id}_fit_shape")

            # ピクセルを(x, y)形式に変換 + pixel_colors抽出
            final_pixels = []
            final_pixel_colors = {}
            for p in best_pixels:
                px, py = p[0], p[1]
                color = p[2] if len(p) > 2 else obj1.dominant_color
                final_pixels.append((px, py))
                final_pixel_colors[(px, py)] = color

            new_obj = Object(
                object_id=new_obj_id,
                object_type=obj1.object_type,
                pixels=final_pixels,
                pixel_colors=final_pixel_colors,
            )

            self.execution_context['objects'][new_obj_id] = new_obj

            #logger.info(f"FIT_SHAPE: obj1={obj1_id}, obj2={obj2_id}, "
                        # f"flip={best_flip}, rotation={best_rotation}, dx={best_dx}, dy={best_dy}, "
                        # f"score={best_score}")

            return new_obj_id

        except Exception as e:
            #logger.error(f"FIT_SHAPEエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return obj1_id

    def _fit_shape_color(self, obj1_id: str, obj2_id: str) -> str:
        """obj1をobj2に対して形状と色をフィット

        位置+色の重複を最大化
        回転あり（0/90/180/270度）
        スコアリング改良版

        Args:
            obj1_id: 移動・回転するオブジェクトのID
            obj2_id: 基準となるオブジェクトのID（固定）

        Returns:
            新しいオブジェクトID
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                #logger.error(f"FIT_SHAPE_COLOR: オブジェクトが見つかりません")
                return obj1_id

            best_score = -1
            best_pixels = None
            best_rotation = 0
            best_dx = 0
            best_dy = 0
            best_flip = None

            # 探索範囲を設定（グリッドサイズに基づく、最大30に制限）
            grid_size = self.grid_context.get_input_grid_size() or (30, 30)
            search_range = min(max(grid_size[0], grid_size[1]), 30)

            # obj2の左上角を計算（ROTATE/SCALEコマンドと同じ基準）
            obj2_pixels = [(p[0], p[1]) for p in obj2.pixels]
            obj2_top_left_x = min(p[0] for p in obj2_pixels) if obj2_pixels else 0
            obj2_top_left_y = min(p[1] for p in obj2_pixels) if obj2_pixels else 0

            # obj2の色情報を準備
            obj2_color_map = {}
            if hasattr(obj2, 'pixel_colors') and obj2.pixel_colors:
                obj2_color_map = obj2.pixel_colors.copy()
            else:
                # pixel_colorsがない場合はdominant_colorを使用
                for p in obj2.pixels:
                    pos = (p[0], p[1])
                    obj2_color_map[pos] = obj2.dominant_color

            # obj1の色情報を取得
            obj1_color_map = {}
            if hasattr(obj1, 'pixel_colors') and obj1.pixel_colors:
                obj1_color_map = obj1.pixel_colors.copy()
            else:
                # pixel_colorsがない場合はdominant_colorを使用
                for p in obj1.pixels:
                    pos = (p[0], p[1])
                    obj1_color_map[pos] = obj1.dominant_color

            # すべての反転パターンを試す（None, "X", "Y"）
            for flip_type in [None, "X", "Y"]:
                # 反転処理
                if flip_type is None:
                    # 反転なし
                    flipped_pixels_with_colors = [(p[0], p[1], obj1_color_map.get((p[0], p[1]), obj1.dominant_color)) for p in obj1.pixels]
                else:
                    # 反転あり
                    temp_obj_id = self._flip(obj1_id, flip_type)
                    if temp_obj_id:
                        temp_obj = self._find_object_by_id(temp_obj_id)
                        if temp_obj and hasattr(temp_obj, 'pixel_colors') and temp_obj.pixel_colors:
                            flipped_pixels_with_colors = [(p[0], p[1], temp_obj.pixel_colors.get((p[0], p[1]), obj1.dominant_color)) for p in temp_obj.pixels]
                        else:
                            continue
                    else:
                        continue

                # すべての回転を試す
                for rotation_index, angle in enumerate([0, 90, 180, 270]):
                    # 反転後のピクセルを回転
                    if angle == 0:
                        # 0度の場合
                        rotated_pixels = flipped_pixels_with_colors
                    else:
                        # 回転前の左上角位置を記録（位置調整用）
                        original_top_left_x = min(p[0] for p in flipped_pixels_with_colors)
                        original_top_left_y = min(p[1] for p in flipped_pixels_with_colors)

                        # 回転処理（整数演算、最小座標基準）
                        base_x = original_top_left_x
                        base_y = original_top_left_y

                        rotated_pixels = []
                        for p in flipped_pixels_with_colors:
                            px, py = p[0], p[1]
                            color = p[2] if len(p) > 2 else obj1.dominant_color

                            # 最小座標からの相対座標
                            rx = px - base_x
                            ry = py - base_y

                            # 回転（整数演算）
                            if rotation_index == 1:  # 90度
                                new_rx = -ry
                                new_ry = rx
                            elif rotation_index == 2:  # 180度
                                new_rx = -rx
                                new_ry = -ry
                            else:  # 270度
                                new_rx = ry
                                new_ry = -rx

                            new_px = base_x + new_rx
                            new_py = base_y + new_ry

                            rotated_pixels.append((new_px, new_py, color))

                        # 位置調整: 回転後のバウンディングボックスの左上を元の位置にそろえる
                        if rotated_pixels:
                            new_min_x = min(p[0] for p in rotated_pixels)
                            new_min_y = min(p[1] for p in rotated_pixels)
                            offset_x = original_top_left_x - new_min_x
                            offset_y = original_top_left_y - new_min_y

                            # すべてのピクセルを調整
                            adjusted_pixels = []
                            for px, py, color in rotated_pixels:
                                adjusted_pixels.append((px + offset_x, py + offset_y, color))

                            rotated_pixels = adjusted_pixels

                    # 回転後の左上角を計算（ROTATE/SCALEコマンドと同じ基準）
                    if rotated_pixels:
                        rotated_top_left_x = min(p[0] for p in rotated_pixels)
                        rotated_top_left_y = min(p[1] for p in rotated_pixels)
                    else:
                        continue

                    # obj2の周囲を探索（左上角基準）
                    for dx in range(-search_range, search_range + 1):
                        for dy in range(-search_range, search_range + 1):
                            # 移動後の位置を計算（左上角を基準に配置）
                            offset_x = obj2_top_left_x + dx - rotated_top_left_x
                            offset_y = obj2_top_left_y + dy - rotated_top_left_y

                            # ピクセルを移動（色情報を保持）
                            moved_pixels = []
                            for p in rotated_pixels:
                                new_x = p[0] + offset_x
                                new_y = p[1] + offset_y
                                # 色情報を保持（元のピクセルに色があればそれを、なければobj1の支配的な色を使用）
                                if len(p) > 2:
                                    moved_pixels.append((new_x, new_y, p[2]))
                                else:
                                    moved_pixels.append((new_x, new_y, obj1.dominant_color))

                            # スコア計算（位置+色、flip_typeを渡す）
                            score = self._calculate_fit_shape_color_score(
                                moved_pixels, obj2_color_map, rotation_index, offset_x, offset_y, flip_type
                            )

                            # 最高スコアを更新
                            if score > best_score:
                                best_score = score
                                best_pixels = moved_pixels
                                best_rotation = angle
                                best_dx = offset_x
                                best_dy = offset_y
                                best_flip = flip_type

            # 最適な配置でオブジェクトを作成
            new_obj_id = self._generate_unique_object_id(f"{obj1.object_id}_fit_shape")

            # ピクセルを(x, y)形式に変換 + pixel_colors抽出
            final_pixels = []
            final_pixel_colors = {}
            for p in best_pixels:
                px, py = p[0], p[1]
                color = p[2] if len(p) > 2 else obj1.dominant_color
                final_pixels.append((px, py))
                final_pixel_colors[(px, py)] = color

            new_obj = Object(
                object_id=new_obj_id,
                object_type=obj1.object_type,
                pixels=final_pixels,
                pixel_colors=final_pixel_colors,
            )

            self.execution_context['objects'][new_obj_id] = new_obj

            #logger.info(f"FIT_SHAPE_COLOR: obj1={obj1_id}, obj2={obj2_id}, "
                        # f"flip={best_flip}, rotation={best_rotation}, dx={best_dx}, dy={best_dy}, "
                        # f"score={best_score}")

            return new_obj_id

        except Exception as e:
            #logger.error(f"FIT_SHAPE_COLORエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return obj1_id

    def _fit_adjacent(self, obj1_id: str, obj2_id: str) -> str:
        """obj1をobj2に対して隣接辺数が最大になるように配置

        定義: 外形の輪郭同士が接する辺の数を最大化（重複は避ける）

        Args:
            obj1_id: 移動するオブジェクトのID
            obj2_id: 基準となるオブジェクトのID（固定）

        Returns:
            str: 新しいオブジェクトのID（移動・回転後のobj1）
        """
        try:
            obj1 = self._find_object_by_id(obj1_id)
            obj2 = self._find_object_by_id(obj2_id)

            if not obj1 or not obj2:
                #logger.error(f"FIT_ADJACENT: オブジェクトが見つかりません")
                return obj1_id

            if not obj1.pixels or not obj2.pixels:
                #logger.error(f"FIT_ADJACENT: ピクセルが空です")
                return obj1_id

            # グリッドサイズを取得
            grid_size = self.grid_context.get_input_grid_size() or (30, 30)

            # bbox計算
            obj1_x = [p[0] for p in obj1.pixels]
            obj1_y = [p[1] for p in obj1.pixels]
            obj2_x = [p[0] for p in obj2.pixels]
            obj2_y = [p[1] for p in obj2.pixels]

            obj1_min_x, obj1_max_x = min(obj1_x), max(obj1_x)
            obj1_min_y, obj1_max_y = min(obj1_y), max(obj1_y)
            obj2_min_x, obj2_max_x = min(obj2_x), max(obj2_x)
            obj2_min_y, obj2_max_y = min(obj2_y), max(obj2_y)

            # 探索範囲を計算（グリッドサイズに基づく、最大30に制限）
            search_range = min(max(grid_size[0], grid_size[1]), 30)
            min_dx = obj2_min_x - obj1_max_x - search_range
            max_dx = obj2_max_x - obj1_min_x + search_range
            min_dy = obj2_min_y - obj1_max_y - search_range
            max_dy = obj2_max_y - obj1_min_y + search_range

            best_score = -float('inf')
            best_pixels = None
            best_rotation = 0
            best_dx = 0
            best_dy = 0
            best_flip = None

            # obj1の色情報を取得
            obj1_color_map = {}
            if hasattr(obj1, 'pixel_colors') and obj1.pixel_colors:
                obj1_color_map = obj1.pixel_colors.copy()
            else:
                for p in obj1.pixels:
                    pos = (p[0], p[1])
                    obj1_color_map[pos] = obj1.dominant_color

            # すべての反転パターンを試す（None, "X", "Y"）
            for flip_type in [None, "X", "Y"]:
                # 反転処理
                if flip_type is None:
                    flipped_pixels = [(p[0], p[1], obj1_color_map.get((p[0], p[1]), obj1.dominant_color)) for p in obj1.pixels]
                else:
                    temp_obj_id = self._flip(obj1_id, flip_type)
                    if temp_obj_id:
                        temp_obj = self._find_object_by_id(temp_obj_id)
                        if temp_obj:
                            temp_color_map = temp_obj.pixel_colors if hasattr(temp_obj, 'pixel_colors') and temp_obj.pixel_colors else {}
                            flipped_pixels = [(p[0], p[1], temp_color_map.get((p[0], p[1]), obj1.dominant_color)) for p in temp_obj.pixels]
                        else:
                            continue
                    else:
                        continue

                # すべての回転・移動を試す
                for rotation_index in [0, 1, 2, 3]:
                    # 回転（整数演算、最小座標基準）
                    if rotation_index == 0:
                        rotated_pixels = flipped_pixels
                    else:
                        # 回転前の左上角位置を記録（位置調整用）
                        original_top_left_x = min(p[0] for p in flipped_pixels)
                        original_top_left_y = min(p[1] for p in flipped_pixels)

                        # 最小座標を基準に回転（整数演算）
                        base_x = original_top_left_x
                        base_y = original_top_left_y

                        rotated_pixels = []
                        for p in flipped_pixels:
                            x, y = p[0], p[1]
                            color = p[2] if len(p) > 2 else obj1.dominant_color

                            # 最小座標からの相対座標
                            rx = x - base_x
                            ry = y - base_y

                            # 回転（整数演算）
                            if rotation_index == 1:  # 90度
                                new_rx = -ry
                                new_ry = rx
                            elif rotation_index == 2:  # 180度
                                new_rx = -rx
                                new_ry = -ry
                            else:  # 270度
                                new_rx = ry
                                new_ry = -rx

                            new_x = base_x + new_rx
                            new_y = base_y + new_ry

                            rotated_pixels.append((new_x, new_y, color))

                        # 位置調整: 回転後のバウンディングボックスの左上を元の位置にそろえる
                        if rotated_pixels:
                            new_min_x = min(p[0] for p in rotated_pixels)
                            new_min_y = min(p[1] for p in rotated_pixels)
                            offset_x = original_top_left_x - new_min_x
                            offset_y = original_top_left_y - new_min_y

                            # すべてのピクセルを調整
                            adjusted_pixels = []
                            for px, py, color in rotated_pixels:
                                adjusted_pixels.append((px + offset_x, py + offset_y, color))

                            rotated_pixels = adjusted_pixels

                    # 移動とスコア計算
                    for dx in range(min_dx, max_dx + 1):
                        for dy in range(min_dy, max_dy + 1):
                            moved_pixels = [(p[0] + dx, p[1] + dy, p[2]) for p in rotated_pixels]

                            # グリッド範囲内かチェック
                            if not all(0 <= p[0] < grid_size[0] and 0 <= p[1] < grid_size[1]
                                      for p in moved_pixels):
                                continue

                            # 重複チェック（重複がある場合はスキップ）
                            moved_coords = set((p[0], p[1]) for p in moved_pixels)
                            obj2_coords = set((p[0], p[1]) for p in obj2.pixels)
                            overlap = moved_coords & obj2_coords

                            if len(overlap) > 0:
                                continue  # 重複禁止

                            # スコア計算（重複なしの配置のみ、flip_typeを渡す）
                            score = self._calculate_fit_adjacent_score(
                                moved_pixels, obj2.pixels, rotation_index, dx, dy, flip_type
                            )

                            if score > best_score:
                                best_score = score
                                best_pixels = moved_pixels
                                best_rotation = rotation_index * 90
                                best_dx = dx
                                best_dy = dy
                                best_flip = flip_type

            if best_pixels is None:
                #logger.warning(f"FIT_ADJACENT: 有効な配置が見つかりません")
                return obj1_id

            # 新しいオブジェクトを作成
            new_obj_id = self._generate_unique_object_id(f"{obj1.object_id}_fit_adjacent")
            final_pixels = [(p[0], p[1]) for p in best_pixels]

            # 色情報を保持（best_pixelsは(x, y, color)のタプル）
            pixel_colors = {}
            for p in best_pixels:
                pixel_colors[(p[0], p[1])] = p[2]

            new_obj = Object(
                object_id=new_obj_id,
                object_type=obj1.object_type,
                pixels=final_pixels,
                pixel_colors=pixel_colors,
            )

            self.execution_context['objects'][new_obj_id] = new_obj

            #logger.info(f"FIT_ADJACENT: obj1={obj1_id}, obj2={obj2_id}, "
                        # f"flip={best_flip}, rotation={best_rotation}, dx={best_dx}, dy={best_dy}, "
                        # f"score={best_score}")

            return new_obj_id

        except Exception as e:
            #logger.error(f"FIT_ADJACENTエラー: {e}")
            import traceback
            # traceback.print_exc() をコメントアウト（スタックトレース出力を抑制）
            # traceback.print_exc()
            return obj1_id

    def _calculate_fit_adjacent_score(self, moved_pixels, obj2_pixels, rotation_index, dx, dy, flip_type=None):
        """FIT_ADJACENT用のスコア計算（重複禁止版）

        優先順位:
        1. 隣接辺数（最優先）× 10000
        2. 反転しない（第2優先）× 100
        3. 回転の少なさ（第3優先）× 10
        4. 移動量の少なさ（第4優先）× 1

        注: 重複がある配置は事前に除外されているため、重複ペナルティは不要
        """
        # 1. 隣接辺数を計算
        moved_coords = set((p[0], p[1]) for p in moved_pixels)
        obj2_coords = set((p[0], p[1]) for p in obj2_pixels)

        adjacent_edges = 0
        for x, y in moved_coords:
            for dx_dir, dy_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx_dir, y + dy_dir

                # 外形チェック
                is_obj1_outline = (nx, ny) not in moved_coords

                if (nx, ny) in obj2_coords:
                    reverse_x, reverse_y = nx - dx_dir, ny - dy_dir
                    is_obj2_outline = (reverse_x, reverse_y) not in obj2_coords

                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 2. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 3. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 4. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア（重複ペナルティなし）
        score = (adjacent_edges * 10000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score

    def _calculate_fit_shape_score(self, moved_pixels, obj2_pixels, rotation_index, dx, dy, flip_type=None):
        """FIT_SHAPE用のスコア計算（位置のみ）

        優先順位:
        1. 重複ピクセル数（最優先）× 100000
        2. 隣接辺数（輪郭同士、次優先）× 1000
        3. 反転しない（第3優先）× 100
        4. 回転の少なさ（第4優先）× 10
        5. 移動量の少なさ（第5優先）× 1
        """
        # 1. 重複ピクセル数（位置のみ）
        moved_coords = set((p[0], p[1]) for p in moved_pixels)
        obj2_coords = set((p[0], p[1]) for p in obj2_pixels)
        overlap_coords = moved_coords & obj2_coords
        overlap_count = len(overlap_coords)

        # 2. 隣接辺数（位置のみ、両方の外形が内側から接している部分）
        # 両方のオブジェクトの外形が内側から接している辺のみカウント
        adjacent_edges = 0
        for x, y in overlap_coords:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が重複領域の外側かチェック
                if (nx, ny) not in overlap_coords:
                    # obj1とobj2の両方でこの方向が外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_coords
                    is_obj2_outline = (nx, ny) not in obj2_coords

                    # 両方の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score

    def _extract_outline_pixels(self, pixels_set):
        """外形輪郭を抽出

        Args:
            pixels_set: ピクセル座標のセット

        Returns:
            外形輪郭のピクセル座標のセット
        """
        if not pixels_set:
            return set()

        outline = set()

        for x, y in pixels_set:
            # 4方向の隣接をチェック
            neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
            for nx, ny in neighbors:
                # 隣接ピクセルがオブジェクトに含まれていない場合、このピクセルは輪郭
                if (nx, ny) not in pixels_set:
                    outline.add((x, y))
                    break

        return outline

    def _calculate_fit_shape_color_score(self, moved_pixels, obj2_color_map, rotation_index, dx, dy, flip_type=None):
        """FIT_SHAPE_COLOR用のスコア計算（位置+色）

        優先順位:
        1. 重複ピクセル数（位置+色）× 100000
        2. 隣接辺数（色一致、輪郭同士）× 1000
        3. 反転しない（第3優先）× 100
        4. 回転の少なさ（第4優先）× 10
        5. 移動量の少なさ（第5優先）× 1
        """
        # 1. 重複ピクセル数（位置+色）
        overlap_count = 0
        moved_color_map = {}
        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score

        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score
        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score

        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score
        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score

        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score
        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score

        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score
        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score

        color_overlap_coords = set()

        for p in moved_pixels:
            pos = (p[0], p[1])
            color = p[2] if len(p) > 2 else 0
            moved_color_map[pos] = color

            if pos in obj2_color_map and color == obj2_color_map[pos]:
                overlap_count += 1
                color_overlap_coords.add(pos)

        # 2. 隣接辺数（色一致、両方の色ごとの外形が内側から接している部分）
        # 両方のオブジェクトの色ごとの外形が内側から接している辺のみカウント
        adjacent_edges = 0

        for pos in color_overlap_coords:
            x, y = pos
            color = moved_color_map.get(pos, -1)

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy

                # この方向が色一致重複領域の外側かチェック
                if (nx, ny) not in color_overlap_coords:
                    # obj1とobj2の両方でこの方向が同じ色の外形かチェック
                    is_obj1_outline = (nx, ny) not in moved_color_map or moved_color_map.get((nx, ny), -1) != color
                    is_obj2_outline = (nx, ny) not in obj2_color_map or obj2_color_map.get((nx, ny), -1) != color

                    # 両方の色の外形が内側から接している場合のみカウント
                    if is_obj1_outline and is_obj2_outline:
                        adjacent_edges += 1

        # 3. 反転ペナルティ（反転しない方が高スコア）
        flip_score = 10 if flip_type is None else 0

        # 4. 回転ペナルティ
        rotation_score = 4 - rotation_index

        # 5. 移動量ペナルティ
        distance_moved = abs(dx) + abs(dy)
        movement_score = max(0, 100 - distance_moved)

        # 総合スコア
        score = (overlap_count * 100000 +
                 adjacent_edges * 1000 +
                 flip_score * 100 +
                 rotation_score * 10 +
                 movement_score)

        return score
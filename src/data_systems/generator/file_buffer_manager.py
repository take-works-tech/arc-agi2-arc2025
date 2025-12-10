"""
ファイルI/Oバッファマネージャー

ファイル書き込みを一括化してパフォーマンスを向上させる
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

# ログ出力制御（パフォーマンス最適化：デフォルトですべてのログを無効化）
ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')

# グローバルバッファマネージャーインスタンス（モジュールレベル）
_global_buffer_manager: Optional['FileBufferManager'] = None

def get_global_buffer_manager() -> Optional['FileBufferManager']:
    """グローバルバッファマネージャーを取得"""
    return _global_buffer_manager

def set_global_buffer_manager(manager: 'FileBufferManager'):
    """グローバルバッファマネージャーを設定"""
    global _global_buffer_manager
    _global_buffer_manager = manager

try:
    from PIL import Image
except ImportError:
    Image = None


class FileBufferManager:
    """ファイルI/Oをバッファして一括処理するマネージャー"""

    BATCH_SIZE = 1000  # 1000タスク単位でまとめる（デフォルト、main.pyで上書き可能）

    def __init__(self, base_output_dir: str, auto_flush: bool = False):
        """初期化

        Args:
            base_output_dir: 出力ディレクトリのベースパス
            auto_flush: Trueの場合、バッファサイズに達したら自動的にフラッシュ。Falseの場合、手動フラッシュのみ
        """
        self.base_output_dir = base_output_dir
        self.task_counter = 0  # 全タスクの通し番号
        self.auto_flush = auto_flush  # 自動フラッシュフラグ

        # バッファ
        self.debug_programs = []  # デバッグプログラムファイル（original + fallback）

        # JSONファイルのバッファ（種類ごと）
        self.program_json_buffer: List[Dict[str, Any]] = []
        self.tokens_json_buffer: List[Dict[str, Any]] = []
        self.stats_json_buffer: List[Dict[str, Any]] = []
        self.grid_json_buffer: List[Dict[str, Any]] = []

        # PNGファイルのバッファ（タスク情報を保存）
        self.png_buffer: List[Dict[str, Any]] = []  # {"task_index": int, "input_grid": np.ndarray, "output_grid": np.ndarray, "timestamp": str}

        # 仮のインプットグリッドのバッファ（部分プログラムフロー用）
        self.temporary_input_grid_buffer: List[Dict[str, Any]] = []  # {"task_index": int, "temporary_input_grid": np.ndarray, "timestamp": str}

    def _upsert_json_buffer(self, buffer_name: str, task_id: str, data: Dict[str, Any]):
        """
        指定されたJSONバッファ内でtask_idが一致する項目を最新データで置き換え、存在しなければ追加する。
        生成中断→再実行時の重複保存を防ぐ。
        """
        buffer: List[Dict[str, Any]] = getattr(self, buffer_name)
        for idx, item in enumerate(buffer):
            if item.get("task_id") == task_id:
                buffer[idx] = data
                break
        else:
            buffer.append(data)

    def _replace_png_entry(self, task_index: int, data: Dict[str, Any]):
        """
        PNGバッファ内で同じtask_indexの既存データを置き換える。
        """
        for idx, item in enumerate(self.png_buffer):
            if item.get("task_index") == task_index:
                self.png_buffer[idx] = data
                return
        self.png_buffer.append(data)

    def add_debug_program(self, cmd_name: str, node_index: int, loop_count: int,
                         program_code: str, is_fallback: bool = False,
                         fallback_code: Optional[str] = None):
        """デバッグプログラムをバッファに追加（書き込みは後で一括実行）

        Args:
            cmd_name: コマンド名
            node_index: ノードインデックス
            loop_count: ループカウント
            program_code: プログラムコード
            is_fallback: fallback版かどうか
            fallback_code: fallbackコード名（fallback版の場合）
        """
        self.debug_programs.append({
            "cmd_name": cmd_name,
            "node_index": node_index,
            "loop_count": loop_count,
            "program_code": program_code,
            "is_fallback": is_fallback,
            "fallback_code": fallback_code,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        })

    def add_program_json(self, task_index: int, program_data: Dict[str, Any]):
        """プログラムJSONデータをバッファに追加

        Args:
            task_index: タスクインデックス（1始まり）
            program_data: プログラムデータ（JSON形式）
        """
        self.task_counter = max(self.task_counter, task_index)  # 最大のタスクインデックスを記録
        task_id = program_data.get("task_id", f"task_{task_index:03d}")
        self._upsert_json_buffer("program_json_buffer", task_id, program_data)
        self._check_and_flush_json("program", task_index)

    def update_program_json(self, task_index: int, updated_program_code: str):
        """プログラムJSONバッファ内のprogram_codeを更新

        Args:
            task_index: タスクインデックス（1始まり）
            updated_program_code: 更新後のプログラムコード
        """
        task_id = f"task_{task_index:03d}"
        # バッファ内から該当するタスクを探して更新
        for item in self.program_json_buffer:
            if item.get("task_id") == task_id:
                item["program_code"] = updated_program_code
                item["program_length"] = len(updated_program_code)
                # statisticsも更新
                if "statistics" in item:
                    item["statistics"]["character_count"] = len(updated_program_code)
                    item["statistics"]["line_count"] = updated_program_code.count('\n') + 1
                    item["statistics"]["word_count"] = len(updated_program_code.split())
                break
        # tokens_jsonとstats_jsonも更新（必要な場合）
        # 注: tokens_jsonとstats_jsonはprogram_codeから生成されるため、更新が必要な場合は
        # 呼び出し側で再生成して更新する必要がある

    def add_tokens_json(self, task_index: int, tokens_data: Dict[str, Any]):
        """トークンJSONデータをバッファに追加

        Args:
            task_index: タスクインデックス（1始まり）
            tokens_data: トークンデータ（JSON形式）
        """
        self.task_counter = max(self.task_counter, task_index)
        task_id = tokens_data.get("task_id", f"task_{task_index:03d}")
        self._upsert_json_buffer("tokens_json_buffer", task_id, tokens_data)
        self._check_and_flush_json("tokens", task_index)

    def add_stats_json(self, task_index: int, stats_data: Dict[str, Any]):
        """統計JSONデータをバッファに追加

        Args:
            task_index: タスクインデックス（1始まり）
            stats_data: 統計データ（JSON形式）
        """
        self.task_counter = max(self.task_counter, task_index)
        task_id = stats_data.get("task_id", f"task_{task_index:03d}")
        self._upsert_json_buffer("stats_json_buffer", task_id, stats_data)
        self._check_and_flush_json("stats", task_index)

    def add_grid_json(self, task_index: int, input_grid: np.ndarray, output_grid: np.ndarray, trace_results: Optional[List[Any]] = None):
        """グリッドデータをバッファに追加（複数ペア対応）

        Args:
            task_index: タスクインデックス（1始まり）
            input_grid: 入力グリッド
            output_grid: 出力グリッド
        """
        self.task_counter = max(self.task_counter, task_index)
        task_id = f"task_{task_index:03d}"

        # 既存のアイテムを探す
        existing_item = None
        for item in self.grid_json_buffer:
            if item.get("task_id") == task_id:
                existing_item = item
                break

        # 新しいペアを作成
        new_pair = {
            "input": input_grid.tolist() if isinstance(input_grid, np.ndarray) else input_grid,
            "output": output_grid.tolist() if isinstance(output_grid, np.ndarray) else output_grid
        }
        # トレース（任意）
        if trace_results is not None:
            new_pair["trace_results"] = trace_results

        if existing_item:
            # 既存のアイテムにペアを追加
            if "train" not in existing_item:
                existing_item["train"] = []
            existing_item["train"].append(new_pair)
        else:
            # 新しいアイテムを作成
            grid_data = {
                "task_id": task_id,
                "train": [new_pair],
                "test": []
            }
            self.grid_json_buffer.append(grid_data)

        self._check_and_flush_json("grid", task_index)

    def add_png_data(self, task_index: int, input_grid: np.ndarray, output_grid: np.ndarray, timestamp: str, pairs_data: List[Dict]):
        """PNGデータをバッファに追加（複数ペア対応）

        Args:
            task_index: タスクインデックス（1始まり）
            input_grid: 入力グリッド（最初のペア）
            output_grid: 出力グリッド（最初のペア）
            timestamp: タイムスタンプ
            pairs_data: すべてのペアのデータ（複数ペア対応、空でないリストを期待）
        """
        self.task_counter = max(self.task_counter, task_index)

        if not pairs_data or len(pairs_data) == 0:
            raise ValueError(f"pairs_data must be a non-empty list for task {task_index}")

        png_entry = {
            "task_index": task_index,
            "input_grid": input_grid,  # 最初のペア
            "output_grid": output_grid,  # 最初のペア
            "timestamp": timestamp,
            "pairs": []  # すべてのペアを保存
        }
        # すべてのペアを追加
        for pair_data in pairs_data:
            png_entry["pairs"].append({
                "pair_index": pair_data.get("pair_index", 0),
                "input_grid": np.asarray(pair_data.get("input_grid", input_grid), dtype=int),
                "output_grid": np.asarray(pair_data.get("output_grid", output_grid), dtype=int)
            })

        self._replace_png_entry(task_index, png_entry)
        self._check_and_flush_png(task_index)

    def add_temporary_input_grid(self, task_index: int, temporary_input_grid: np.ndarray, timestamp: str):
        """仮のインプットグリッドをバッファに追加（部分プログラムフロー用）

        Args:
            task_index: タスクインデックス（1始まり）
            temporary_input_grid: 仮のインプットグリッド（2D numpy配列）
            timestamp: タイムスタンプ
        """
        self.task_counter = max(self.task_counter, task_index)

        grid_entry = {
            "task_index": task_index,
            "temporary_input_grid": np.asarray(temporary_input_grid, dtype=int),
            "timestamp": timestamp
        }

        # 同じtask_indexの既存データを置き換え
        self._replace_temporary_input_grid_entry(task_index, grid_entry)

    def _replace_temporary_input_grid_entry(self, task_index: int, data: Dict[str, Any]):
        """仮のインプットグリッドバッファ内で同じtask_indexの既存データを置き換える。

        Args:
            task_index: タスクインデックス
            data: 新しいデータ
        """
        for idx, item in enumerate(self.temporary_input_grid_buffer):
            if item.get("task_index") == task_index:
                self.temporary_input_grid_buffer[idx] = data
                return
        self.temporary_input_grid_buffer.append(data)

    def _check_and_flush_json(self, json_type: str, current_task_index: int):
        """JSONバッファをチェックして、1000件に達したらフラッシュ（auto_flushがTrueの場合のみ）

        Args:
            json_type: JSONタイプ（"program", "tokens", "stats", "grid"）
            current_task_index: 現在のタスクインデックス（1始まり）
        """
        if not self.auto_flush:
            return  # 自動フラッシュが無効の場合は何もしない

        buffer = getattr(self, f"{json_type}_json_buffer")
        if len(buffer) >= self.BATCH_SIZE:
            # バッチインデックス = (タスクインデックス - 1) // BATCH_SIZE（1始まりなので-1）
            batch_index = (current_task_index - 1) // self.BATCH_SIZE
            self._flush_json(json_type, batch_index)
            setattr(self, f"{json_type}_json_buffer", [])

    def _flush_json(self, json_type: str, batch_index: int):
        """JSONバッファをファイルに書き込む

        Args:
            json_type: JSONタイプ
            batch_index: バッチインデックス
        """
        buffer = getattr(self, f"{json_type}_json_buffer")
        if not buffer:
            return

        # task_idでソート（task_001, task_002, ...の順序）
        def get_task_number(item):
            task_id = item.get("task_id", "task_000")
            try:
                return int(task_id.split("_")[-1])
            except (ValueError, IndexError):
                return 0

        sorted_buffer = sorted(buffer, key=get_task_number)

        # ファイル名を生成（ソート後の最初と最後のタスク番号を使用）
        start_task_num = get_task_number(sorted_buffer[0])
        end_task_num = get_task_number(sorted_buffer[-1])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{json_type}_batch_{batch_index:04d}_tasks_{start_task_num:03d}_to_{end_task_num:03d}_{timestamp}.json"

        # バッチディレクトリを作成
        batch_dir = os.path.join(self.base_output_dir, f"batch_{batch_index:04d}")
        os.makedirs(batch_dir, exist_ok=True)

        # JSONファイルに保存
        filepath = os.path.join(batch_dir, filename)
        buffer_size = len(sorted_buffer)

        try:
            if ENABLE_ALL_LOGS:
                print(f"  [{json_type.upper()}] バッチ {batch_index} を保存中: {filepath} ({buffer_size}件)...")

            with open(filepath, "w", encoding="utf-8") as f:
                if json_type == "grid":
                    # gridは特殊な形式（challenges形式、ソート順で保存）
                    challenges_dict = {}
                    for item in sorted_buffer:
                        # 複数ペア対応：既にtrain配列がある場合はそのまま使用
                        # 注: add_grid_jsonでは常にtrain配列を作成するため、通常はこの分岐が実行される
                        if "train" in item and isinstance(item["train"], list) and len(item["train"]) > 0:
                            # 既にtrain配列がある場合（複数ペア対応）
                            challenges_dict[item["task_id"]] = {
                                "train": item["train"],
                                "test": item.get("test", [])
                            }
                        else:
                            # 旧形式の場合（単一ペア、既存データファイルとの互換性のため）
                            # 注: 現在のコードでは使用されないが、既存のJSONファイルを読み込む際に必要
                            challenges_dict[item["task_id"]] = {
                                "train": [{"input": item["input"], "output": item["output"]}],
                                "test": []
                            }
                    json.dump(challenges_dict, f, ensure_ascii=False, indent=2)
                else:
                    # その他は配列形式（ソート順で保存）
                    json.dump(sorted_buffer, f, ensure_ascii=False, indent=2)

            if ENABLE_ALL_LOGS:
                print(f"  [{json_type.upper()}] バッチ {batch_index} を保存完了: {filepath} ({buffer_size}件)")
        except KeyboardInterrupt:
            # ユーザーが中断した場合、部分的なファイルが残る可能性があるため削除
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            print(f"\n  [{json_type.upper()}] バッチ {batch_index} の保存が中断されました。")
            raise
        except Exception as e:
            # その他のエラーが発生した場合もファイルを削除
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            print(f"  [{json_type.upper()}] バッチ {batch_index} の保存中にエラーが発生しました: {e}")
            raise

    def _check_and_flush_png(self, current_task_index: int):
        """PNGバッファをチェックして、1000件に達したらフラッシュ（auto_flushがTrueの場合のみ）

        Args:
            current_task_index: 現在のタスクインデックス（1始まり）
        """
        if not self.auto_flush:
            return  # 自動フラッシュが無効の場合は何もしない

        if len(self.png_buffer) >= self.BATCH_SIZE:
            # バッチインデックス = (タスクインデックス - 1) // BATCH_SIZE（1始まりなので-1）
            batch_index = (current_task_index - 1) // self.BATCH_SIZE
            self._flush_png(batch_index)
            self.png_buffer = []

    def _flush_png(self, batch_index: int):
        """PNGバッファをファイルに書き込む（冒頭10タスク分のみ）

        Args:
            batch_index: バッチインデックス
        """
        if not self.png_buffer:
            return

        # task_indexでソート（1, 2, 3, ...の順序）
        sorted_buffer = sorted(self.png_buffer, key=lambda x: x.get("task_index", 0))

        # 冒頭10タスク分のみを取得（ソート後）
        tasks_to_save = sorted_buffer[:10]

        # バッチディレクトリを作成
        batch_dir = os.path.join(self.base_output_dir, f"batch_{batch_index:04d}")
        os.makedirs(batch_dir, exist_ok=True)

        # 1つのPNGファイルに結合（グリッドを並べて配置）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ファイル名に実際のタスク範囲を反映（1始まりのタスク番号）
        if tasks_to_save:
            # task_indexからタスク番号を取得（1始まり）
            def get_task_num_from_item(item):
                task_index = item.get("task_index", 1)
                return task_index  # task_indexは既に1始まり

            start_task_num = get_task_num_from_item(tasks_to_save[0])
            end_task_num = get_task_num_from_item(tasks_to_save[-1])
        else:
            start_task_num = 1
            end_task_num = 1
        filename = f"grids_batch_{batch_index:04d}_tasks_{start_task_num:03d}_to_{end_task_num:03d}_{timestamp}.png"
        filepath = os.path.join(batch_dir, filename)

        # グリッドを結合してPNGに保存
        self._save_grids_to_png(tasks_to_save, filepath)

        if ENABLE_ALL_LOGS:
            print(f"  [PNG] バッチ {batch_index} を保存: {filepath} ({len(tasks_to_save)}タスク)")

    def _flush_temporary_input_grid_png(self, batch_index: int):
        """仮のインプットグリッドPNGバッファをファイルに書き込む（冒頭10タスク分のみ）

        Args:
            batch_index: バッチインデックス
        """
        if not self.temporary_input_grid_buffer:
            return

        # task_indexでソート（1, 2, 3, ...の順序）
        sorted_buffer = sorted(self.temporary_input_grid_buffer, key=lambda x: x.get("task_index", 0))

        # 冒頭10タスク分のみを取得（ソート後）
        tasks_to_save = sorted_buffer[:10]

        if not tasks_to_save:
            return

        # バッチディレクトリを作成
        batch_dir = os.path.join(self.base_output_dir, f"batch_{batch_index:04d}")
        os.makedirs(batch_dir, exist_ok=True)

        # 1つのPNGファイルに結合（グリッドを並べて配置）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # ファイル名に実際のタスク範囲を反映（1始まりのタスク番号）
        start_task_num = tasks_to_save[0].get("task_index", 1)
        end_task_num = tasks_to_save[-1].get("task_index", 1)
        filename = f"temporary_input_grids_batch_{batch_index:04d}_tasks_{start_task_num:03d}_to_{end_task_num:03d}_{timestamp}.png"
        filepath = os.path.join(batch_dir, filename)

        # グリッドを結合してPNGに保存
        self._save_temporary_input_grids_to_png(tasks_to_save, filepath)

        if ENABLE_ALL_LOGS:
            print(f"  [PNG] 仮のインプットグリッド バッチ {batch_index} を保存: {filepath} ({len(tasks_to_save)}タスク)")

    def _save_temporary_input_grids_to_png(self, tasks: List[Dict[str, Any]], output_path: str):
        """複数の仮のインプットグリッドを1つのPNGファイルに結合

        Args:
            tasks: タスクデータのリスト
            output_path: 出力ファイルパス
        """
        try:
            if Image is None:
                print(f"  [WARNING] PIL/Pillowがインストールされていないため、PNG結合をスキップします")
                return
            from src.data_systems.generator.grid_visualizer import save_single_grid_to_png

            # 各タスクのグリッド画像を生成して結合
            images = []
            import tempfile
            import time

            for task in tasks:
                task_index = task.get('task_index', 0)
                temporary_input_grid = task.get('temporary_input_grid', None)

                if temporary_input_grid is None:
                    continue

                # 個別の画像を生成（一時的に）
                tmp_path = None
                try:
                    # 一時ファイルを作成（delete=Falseで後で削除）
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=tempfile.gettempdir()) as tmp:
                        tmp_path = tmp.name

                    success = save_single_grid_to_png(
                        grid=np.asarray(temporary_input_grid, dtype=int),
                        output_path=tmp_path,
                        title=f"Task {task_index} - Temporary Input Grid"
                    )

                    if success and tmp_path and os.path.exists(tmp_path):
                        # ファイルを開いて読み込む（確実に閉じる）
                        img = None
                        try:
                            img = Image.open(tmp_path)
                            # 画像データをメモリに読み込む（ファイルを閉じられるように）
                            img.load()
                            images.append(img.copy())  # コピーを作成して元の画像を閉じられるようにする
                            img.close()
                            img = None
                        except Exception as e:
                            print(f"  [WARNING] 画像読み込みエラー: {e}")
                            if img:
                                try:
                                    img.close()
                                except:
                                    pass

                    # ファイルを削除（少し待ってから削除）
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            # Windowsではファイルロックのため、少し待つ
                            time.sleep(0.01)
                            os.unlink(tmp_path)
                        except PermissionError:
                            # 削除に失敗した場合は後で削除を試みる（ただし警告は出さない）
                            try:
                                time.sleep(0.1)
                                os.unlink(tmp_path)
                            except:
                                pass  # 削除に失敗しても続行
                        except Exception as e:
                            print(f"  [WARNING] 一時ファイル削除エラー: {e}")

                except Exception as e:
                    print(f"  [WARNING] タスク{task_index}の仮のインプットグリッドPNG生成エラー: {e}")
                    # クリーンアップ
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            time.sleep(0.1)
                            os.unlink(tmp_path)
                        except:
                            pass

            # 画像を結合（グリッドレイアウト: 2列に配置して見やすくする）
            if images:
                # 2列レイアウト（より見やすくするため）
                cols = 2
                rows = (len(images) + cols - 1) // cols  # 切り上げ

                # 各画像のサイズを取得
                max_width = max(img.width for img in images)
                max_height = max(img.height for img in images)

                # パディングとスペーシング
                padding = 10  # 画像間のスペース
                title_height = 30  # タイトル用のスペース（各タスク用）

                # キャンバスサイズを計算
                canvas_width = cols * max_width + (cols + 1) * padding
                canvas_height = rows * (max_height + title_height) + (rows + 1) * padding

                combined = Image.new('RGB', (canvas_width, canvas_height), color='white')

                # 画像を配置
                for idx, img in enumerate(images):
                    row = idx // cols
                    col = idx % cols

                    x_offset = padding + col * (max_width + padding)
                    y_offset = padding + row * (max_height + title_height + padding)

                    # 画像を中央揃えで配置
                    x_center = x_offset + (max_width - img.width) // 2
                    y_center = y_offset + title_height + (max_height - img.height) // 2

                    combined.paste(img, (x_center, y_center))

                combined.save(output_path)

                # 画像を確実に閉じる
                combined.close()
                for img in images:
                    try:
                        if hasattr(img, 'close'):
                            img.close()
                    except:
                        pass
                images.clear()  # リストもクリア
        except Exception as e:
            print(f"  [WARNING] 仮のインプットグリッドPNG保存エラー: {e}")

    def _save_grids_to_png(self, tasks: List[Dict[str, Any]], output_path: str):
        """複数のグリッドを1つのPNGファイルに結合

        Args:
            tasks: タスクデータのリスト
            output_path: 出力ファイルパス
        """
        try:
            if Image is None:
                print(f"  [WARNING] PIL/Pillowがインストールされていないため、PNG結合をスキップします")
                return
            from src.data_systems.generator.grid_visualizer import save_grids_to_png as save_single

            # 各タスクのグリッド画像を生成して結合
            images = []
            import tempfile
            import time

            for task in tasks:
                task_index = task.get('task_index', 0)

                # 複数ペア対応: pairsがある場合はすべてのペアを表示
                # 注: add_png_dataでは常にpairs配列を作成するため、通常はこの分岐が実行される
                pairs = task.get('pairs', None)
                if pairs is not None and len(pairs) > 0:
                    # 複数ペアがある場合: 各ペアを表示
                    for pair in pairs:
                        pair_index = pair.get('pair_index', 0)
                        input_grid = pair['input_grid']
                        output_grid = pair['output_grid']

                        # 個別の画像を生成（一時的に）
                        tmp_path = None
                        try:
                            # 一時ファイルを作成（delete=Falseで後で削除）
                            with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=tempfile.gettempdir()) as tmp:
                                tmp_path = tmp.name

                            success = save_single(
                                input_grid=np.asarray(input_grid, dtype=int),
                                output_grid=np.asarray(output_grid, dtype=int),
                                output_path=tmp_path,
                                title=f"Task {task_index} - Pair {pair_index+1}/{len(pairs)}"
                            )

                            if success and tmp_path and os.path.exists(tmp_path):
                                # ファイルを開いて読み込む（確実に閉じる）
                                img = None
                                try:
                                    img = Image.open(tmp_path)
                                    # 画像データをメモリに読み込む（ファイルを閉じられるように）
                                    img.load()
                                    images.append(img.copy())  # コピーを作成して元の画像を閉じられるようにする
                                    img.close()
                                    img = None
                                except Exception as e:
                                    print(f"  [WARNING] 画像読み込みエラー: {e}")
                                    if img:
                                        try:
                                            img.close()
                                        except:
                                            pass

                            # ファイルを削除（少し待ってから削除）
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    # Windowsではファイルロックのため、少し待つ
                                    time.sleep(0.01)
                                    os.unlink(tmp_path)
                                except PermissionError:
                                    # 削除に失敗した場合は後で削除を試みる（ただし警告は出さない）
                                    try:
                                        time.sleep(0.1)
                                        os.unlink(tmp_path)
                                    except:
                                        pass  # 削除に失敗しても続行
                                except Exception as e:
                                    print(f"  [WARNING] 一時ファイル削除エラー: {e}")

                        except Exception as e:
                            print(f"  [WARNING] タスク{task_index}ペア{pair_index+1}のPNG生成エラー: {e}")
                            # クリーンアップ
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    time.sleep(0.1)
                                    os.unlink(tmp_path)
                                except:
                                    pass
                else:
                    # 単一ペア（既存データファイルとの互換性のため）
                    # 注: 現在のコードでは使用されないが、既存のPNGバッファファイルを読み込む際に必要
                    input_grid = task["input_grid"]
                    output_grid = task["output_grid"]

                    # 個別の画像を生成（一時的に）
                    tmp_path = None
                    try:
                        # 一時ファイルを作成（delete=Falseで後で削除）
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False, dir=tempfile.gettempdir()) as tmp:
                            tmp_path = tmp.name

                        success = save_single(
                            input_grid=np.asarray(input_grid, dtype=int),
                            output_grid=np.asarray(output_grid, dtype=int),
                            output_path=tmp_path,
                            title=f"Task {task_index}"
                        )

                        if success and tmp_path and os.path.exists(tmp_path):
                            # ファイルを開いて読み込む（確実に閉じる）
                            img = None
                            try:
                                img = Image.open(tmp_path)
                                # 画像データをメモリに読み込む（ファイルを閉じられるように）
                                img.load()
                                images.append(img.copy())  # コピーを作成して元の画像を閉じられるようにする
                                img.close()
                                img = None
                            except Exception as e:
                                print(f"  [WARNING] 画像読み込みエラー: {e}")
                                if img:
                                    try:
                                        img.close()
                                    except:
                                        pass

                            # ファイルを削除（少し待ってから削除）
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    # Windowsではファイルロックのため、少し待つ
                                    time.sleep(0.01)
                                    os.unlink(tmp_path)
                                except PermissionError:
                                    # 削除に失敗した場合は後で削除を試みる（ただし警告は出さない）
                                    try:
                                        time.sleep(0.1)
                                        os.unlink(tmp_path)
                                    except:
                                        pass  # 削除に失敗しても続行
                                except Exception as e:
                                    print(f"  [WARNING] 一時ファイル削除エラー: {e}")

                    except Exception as e:
                        print(f"  [WARNING] タスク{task_index}のPNG生成エラー: {e}")
                        # クリーンアップ
                        if tmp_path and os.path.exists(tmp_path):
                            try:
                                time.sleep(0.1)
                                os.unlink(tmp_path)
                            except:
                                pass

            # 画像を結合（グリッドレイアウト: 2列に配置して見やすくする）
            if images:
                # 2列レイアウト（より見やすくするため）
                cols = 2
                rows = (len(images) + cols - 1) // cols  # 切り上げ

                # 各画像のサイズを取得
                max_width = max(img.width for img in images)
                max_height = max(img.height for img in images)

                # パディングとスペーシング
                padding = 10  # 画像間のスペース
                title_height = 30  # タイトル用のスペース（各タスク/ペア用）

                # キャンバスサイズを計算
                canvas_width = cols * max_width + (cols + 1) * padding
                canvas_height = rows * (max_height + title_height) + (rows + 1) * padding

                combined = Image.new('RGB', (canvas_width, canvas_height), color='white')

                # 画像を配置
                for idx, img in enumerate(images):
                    row = idx // cols
                    col = idx % cols

                    x_offset = padding + col * (max_width + padding)
                    y_offset = padding + row * (max_height + title_height + padding)

                    # 画像を中央揃えで配置
                    x_center = x_offset + (max_width - img.width) // 2
                    y_center = y_offset + title_height + (max_height - img.height) // 2

                    combined.paste(img, (x_center, y_center))

                combined.save(output_path)

                # 画像を確実に閉じる
                combined.close()
                for img in images:
                    try:
                        if hasattr(img, 'close'):
                            img.close()
                    except:
                        pass
                images.clear()  # リストもクリア
        except Exception as e:
            print(f"  [WARNING] PNG保存エラー: {e}")

    def flush_all(self):
        """すべてのバッファをフラッシュ（最後に呼び出す）"""
        # 残りのJSONバッファをフラッシュ（1000に満たない場合も処理）
        for json_type in ["program", "tokens", "stats", "grid"]:
            buffer = getattr(self, f"{json_type}_json_buffer")
            if buffer:
                # task_idでソート（task_001, task_002, ...の順序）
                def get_task_number(item):
                    task_id = item.get("task_id", "task_000")
                    try:
                        return int(task_id.split("_")[-1])
                    except (ValueError, IndexError):
                        return 0

                sorted_buffer = sorted(buffer, key=get_task_number)

                # バッファ内の最後のタスクインデックスからバッチインデックスを計算
                if sorted_buffer:
                    # 最後のタスクのtask_idからインデックスを取得
                    last_task_id = sorted_buffer[-1].get("task_id", "task_000")
                    try:
                        last_task_index = int(last_task_id.split("_")[-1])
                        batch_index = (last_task_index - 1) // self.BATCH_SIZE
                    except (ValueError, IndexError):
                        # フォールバック: task_counterから計算
                        batch_index = (self.task_counter - 1) // self.BATCH_SIZE if self.task_counter > 0 else 0
                else:
                    batch_index = 0

                # ソート済みバッファを設定してフラッシュ
                setattr(self, f"{json_type}_json_buffer", sorted_buffer)
                self._flush_json(json_type, batch_index)
                setattr(self, f"{json_type}_json_buffer", [])

        # PNGバッファをフラッシュ（残りがあれば）
        # バッチごとにPNGを保存するため、タスクインデックスごとにバッチを判定して保存
        if self.png_buffer:
            # タスクをソート（1, 2, 3, ...の順序）
            sorted_png_buffer = sorted(self.png_buffer, key=lambda x: x.get("task_index", 0))

            # バッファ内のタスクをバッチごとにグループ化
            tasks_by_batch = {}
            for task_data in sorted_png_buffer:
                task_index = task_data.get("task_index", 1)
                batch_index = (task_index - 1) // self.BATCH_SIZE
                if batch_index not in tasks_by_batch:
                    tasks_by_batch[batch_index] = []
                tasks_by_batch[batch_index].append(task_data)

            # 各バッチごとにPNGを保存（ソート済み）
            for batch_index in sorted(tasks_by_batch.keys()):
                tasks_in_batch = tasks_by_batch[batch_index]
                # 各バッチ内でも再度ソート（念のため）
                tasks_in_batch = sorted(tasks_in_batch, key=lambda x: x.get("task_index", 0))
                # 一時的にバッファを置き換えてフラッシュ
                original_buffer = self.png_buffer
                self.png_buffer = tasks_in_batch
                self._flush_png(batch_index)
                self.png_buffer = original_buffer

            self.png_buffer = []

        # デバッグプログラムファイルを一括書き込み
        self._flush_debug_programs()

    def _flush_debug_programs(self):
        """デバッグプログラムファイルを一括書き込み"""
        if not self.debug_programs:
            return

        debug_dir = os.path.join(self.base_output_dir, "debug_programs")
        os.makedirs(debug_dir, exist_ok=True)

        # タイムスタンプでソート
        self.debug_programs.sort(key=lambda x: x["timestamp"])

        # 1つのファイルにまとめて書き込み
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_filename = os.path.join(debug_dir, f"debug_programs_batch_{timestamp}.txt")

        with open(batch_filename, "w", encoding="utf-8") as f:
            for item in self.debug_programs:
                prefix = "FALLBACK" if item["is_fallback"] else "ORIGINAL"
                fallback_info = f" → {item['fallback_code']}" if item["is_fallback"] and item["fallback_code"] else ""
                f.write(f"# ========================================\n")
                f.write(f"# {prefix}: {item['cmd_name']}{fallback_info}\n")
                f.write(f"# Node {item['node_index']}, Loop {item['loop_count']}\n")
                f.write(f"# Timestamp: {item['timestamp']}\n")
                f.write(f"# ========================================\n")
                f.write(item['program_code'])
                f.write("\n\n")

        print(f"  [DEBUG] デバッグプログラムファイル一括保存: {batch_filename} ({len(self.debug_programs)}件)")
        self.debug_programs = []

    def generate_summary_txt_files(self):
        """JSONからすべてのタスクをTXT化して保存（バッチごとに生成）

        バッチディレクトリ内の各program_batch_*.jsonファイルごとに
        対応するTXTファイルを生成する（効率的なバッチ処理）
        """
        # batch_*ディレクトリを探す
        batch_dirs = [d for d in os.listdir(self.base_output_dir)
                     if os.path.isdir(os.path.join(self.base_output_dir, d)) and d.startswith("batch_")]
        batch_dirs.sort()

        for batch_dir_name in batch_dirs:
            batch_dir = os.path.join(self.base_output_dir, batch_dir_name)

            # program JSONファイルを探す
            program_json_files = [f for f in os.listdir(batch_dir)
                                if f.startswith("program_batch_") and f.endswith(".json")]
            program_json_files.sort()  # ファイル名でソート

            if not program_json_files:
                continue

            # 各JSONファイルごとにTXTファイルを生成（バッチごとの効率的な処理）
            for json_file in program_json_files:
                json_path = os.path.join(batch_dir, json_file)
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        programs = json.load(f)

                    if not programs:
                        continue

                    # task_idでソート（task_001, task_002, ...の順）
                    sorted_programs = sorted(programs, key=lambda x: int(x.get('task_id', 'task_000').split('_')[-1]) if '_' in x.get('task_id', '') else 0)

                    # JSONファイル名から情報を抽出してTXTファイル名を生成
                    # 例: program_batch_0000_tasks_001_to_100_20251105_013829.json
                    # → program_batch_0000_tasks_001_to_100_20251105_013829.txt
                    txt_filename = json_file.replace('.json', '.txt')
                    txt_path = os.path.join(batch_dir, txt_filename)

                    # 既にTXTファイルが存在する場合はスキップ（重複防止）
                    if os.path.exists(txt_path):
                        if ENABLE_ALL_LOGS:
                            print(f"  [TXT] スキップ（既存）: {txt_path}")
                        continue

                    with open(txt_path, "w", encoding="utf-8") as f:
                        for program in sorted_programs:
                            f.write(f"# Task {program.get('task_id', 'unknown')}\n")
                            f.write(f"# Complexity: {program.get('complexity', 'N/A')}\n")
                            f.write(f"# Timestamp: {program.get('timestamp', 'N/A')}\n")
                            f.write("# ========================================\n")
                            f.write(program.get("program_code", ""))
                            f.write("\n\n")

                    print(f"  [TXT] バッチファイル生成: {txt_path} ({len(sorted_programs)}タスク)")

                except Exception as e:
                    print(f"  [WARNING] JSONファイル読み込みエラー: {json_path} - {e}")
                    continue

            # PNG画像は_flush_pngで既に保存されているため、ここでは保存しない（重複防止）

    # _generate_summary_png_filesメソッドは削除（_flush_pngで既に保存されているため重複防止）

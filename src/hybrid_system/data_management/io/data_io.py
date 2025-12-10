"""
データI/O処理

データの読み込み、書き込み、変換処理
"""

from typing import List, Dict, Any, Optional, Union, Iterator
import json
import csv
import pickle
import gzip
import os
from pathlib import Path
import pandas as pd

from src.hybrid_system.core.data_structures import DataPair, Task


class DataIO:
    """データI/O処理クラス"""

    def __init__(self, base_dir: str = "data"):
        """初期化"""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_data_pairs(
        self,
        data_pairs: List[DataPair],
        filename: str,
        format: str = "jsonl",
        compress: bool = False
    ) -> str:
        """DataPairリストを保存

        Args:
            data_pairs: 保存するDataPairのリスト
            filename: ファイル名
            format: 保存形式 (jsonl, json, csv, pickle)
            compress: 圧縮するかどうか

        Returns:
            保存されたファイルパス
        """
        file_path = self._get_file_path(filename, format, compress)

        if format == "jsonl":
            self._save_jsonl(data_pairs, file_path, compress)
        elif format == "json":
            self._save_json(data_pairs, file_path, compress)
        elif format == "csv":
            self._save_csv(data_pairs, file_path, compress)
        elif format == "pickle":
            self._save_pickle(data_pairs, file_path, compress)
        else:
            raise ValueError(f"サポートされていない形式: {format}")

        return str(file_path)

    def load_data_pairs(
        self,
        file_path: str,
        format: Optional[str] = None
    ) -> List[DataPair]:
        """DataPairリストを読み込み

        Args:
            file_path: ファイルパス（相対パスの場合、base_dirを基準にする）
            format: ファイル形式（Noneの場合は自動判定）

        Returns:
            DataPairのリスト
        """
        # パスを解決（絶対パスでない場合、base_dirを基準にする）
        if not os.path.isabs(file_path):
            resolved_path = str(self.base_dir / file_path)
        else:
            resolved_path = file_path

        if format is None:
            format = self._detect_format(resolved_path)

        if format == "jsonl":
            return self._load_jsonl(resolved_path)
        elif format == "json":
            return self._load_json(resolved_path)
        elif format == "csv":
            return self._load_csv(resolved_path)
        elif format == "pickle":
            return self._load_pickle(resolved_path)
        else:
            raise ValueError(f"サポートされていない形式: {format}")

    def save_tasks(
        self,
        tasks: List[Task],
        filename: str,
        format: str = "json",
        compress: bool = False
    ) -> str:
        """Taskリストを保存

        Args:
            tasks: 保存するTaskのリスト
            filename: ファイル名
            format: 保存形式 (json, pickle)
            compress: 圧縮するかどうか

        Returns:
            保存されたファイルパス
        """
        file_path = self._get_file_path(filename, format, compress)

        if format == "json":
            self._save_tasks_json(tasks, file_path, compress)
        elif format == "pickle":
            self._save_pickle(tasks, file_path, compress)
        else:
            raise ValueError(f"サポートされていない形式: {format}")

        return str(file_path)

    def load_tasks(
        self,
        file_path: str,
        format: Optional[str] = None
    ) -> List[Task]:
        """Taskリストを読み込み

        Args:
            file_path: ファイルパス
            format: ファイル形式（Noneの場合は自動判定）

        Returns:
            Taskのリスト
        """
        if format is None:
            format = self._detect_format(file_path)

        if format == "json":
            return self._load_tasks_json(file_path)
        elif format == "pickle":
            return self._load_pickle(file_path)
        else:
            raise ValueError(f"サポートされていない形式: {format}")

    def save_single_data_pair(
        self,
        data_pair: DataPair,
        filename: str,
        format: str = "json",
        compress: bool = False
    ) -> str:
        """単一のDataPairを保存

        Args:
            data_pair: 保存するDataPair
            filename: ファイル名
            format: 保存形式
            compress: 圧縮するかどうか

        Returns:
            保存されたファイルパス
        """
        return self.save_data_pairs([data_pair], filename, format, compress)

    def save_single_task(
        self,
        task: Task,
        filename: str,
        format: str = "json",
        compress: bool = False
    ) -> str:
        """単一のTaskを保存

        Args:
            task: 保存するTask
            filename: ファイル名
            format: 保存形式
            compress: 圧縮するかどうか

        Returns:
            保存されたファイルパス
        """
        return self.save_tasks([task], filename, format, compress)

    def stream_data_pairs(
        self,
        file_path: str,
        format: Optional[str] = None
    ) -> Iterator[DataPair]:
        """DataPairをストリーミング読み込み

        Args:
            file_path: ファイルパス
            format: ファイル形式

        Yields:
            DataPair
        """
        if format is None:
            format = self._detect_format(file_path)

        if format == "jsonl":
            yield from self._stream_jsonl(file_path)
        else:
            # 他の形式は一度に読み込んでからストリーミング
            data_pairs = self.load_data_pairs(file_path, format)
            for pair in data_pairs:
                yield pair

    def stream_tasks(
        self,
        file_path: str,
        format: Optional[str] = None
    ) -> Iterator[Task]:
        """Taskをストリーミング読み込み

        Args:
            file_path: ファイルパス
            format: ファイル形式

        Yields:
            Task
        """
        if format is None:
            format = self._detect_format(file_path)

        if format == "json":
            yield from self._stream_tasks_json(file_path)
        else:
            # 他の形式は一度に読み込んでからストリーミング
            tasks = self.load_tasks(file_path, format)
            for task in tasks:
                yield task

    def convert_format(
        self,
        input_path: str,
        output_path: str,
        input_format: Optional[str] = None,
        output_format: Optional[str] = None
    ) -> str:
        """ファイル形式を変換

        Args:
            input_path: 入力ファイルパス
            output_path: 出力ファイルパス
            input_format: 入力形式
            output_format: 出力形式

        Returns:
            出力ファイルパス
        """
        if input_format is None:
            input_format = self._detect_format(input_path)

        if output_format is None:
            output_format = self._detect_format(output_path)

        # データを読み込み
        if input_path.endswith('.json') and 'task' in input_path:
            data = self.load_tasks(input_path, input_format)
        else:
            data = self.load_data_pairs(input_path, input_format)

        # データを保存
        if output_path.endswith('.json') and 'task' in output_path:
            return self.save_tasks(data, output_path, output_format)
        else:
            return self.save_data_pairs(data, output_path, output_format)

    def _get_file_path(self, filename: str, format: str, compress: bool) -> Path:
        """ファイルパスを生成"""
        if not filename.endswith(f".{format}"):
            filename += f".{format}"

        if compress:
            filename += ".gz"

        return self.base_dir / filename

    def _detect_format(self, file_path: str) -> str:
        """ファイル形式を自動判定"""
        file_path = file_path.lower()

        if file_path.endswith('.jsonl') or file_path.endswith('.jsonl.gz'):
            return "jsonl"
        elif file_path.endswith('.json') or file_path.endswith('.json.gz'):
            return "json"
        elif file_path.endswith('.csv') or file_path.endswith('.csv.gz'):
            return "csv"
        elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            return "pickle"
        else:
            raise ValueError(f"ファイル形式を判定できません: {file_path}")

    def _save_jsonl(self, data_pairs: List[DataPair], file_path: Path, compress: bool):
        """JSONL形式で保存"""
        open_func = gzip.open if compress else open
        mode = 'wt' if compress else 'w'

        with open_func(file_path, mode, encoding='utf-8') as f:
            for pair in data_pairs:
                json.dump(pair.to_dict(), f, ensure_ascii=False)
                f.write('\n')

    def _load_jsonl(self, file_path: str) -> List[DataPair]:
        """JSONL形式で読み込み"""
        data_pairs = []
        open_func = gzip.open if file_path.endswith('.gz') else open
        mode = 'rt' if file_path.endswith('.gz') else 'r'

        with open_func(file_path, mode, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    data_pairs.append(DataPair.from_dict(data))

        return data_pairs

    def _stream_jsonl(self, file_path: str) -> Iterator[DataPair]:
        """JSONL形式でストリーミング読み込み"""
        open_func = gzip.open if file_path.endswith('.gz') else open
        mode = 'rt' if file_path.endswith('.gz') else 'r'

        with open_func(file_path, mode, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    yield DataPair.from_dict(data)

    def _save_json(self, data_pairs: List[DataPair], file_path: Path, compress: bool):
        """JSON形式で保存"""
        data = [pair.to_dict() for pair in data_pairs]

        open_func = gzip.open if compress else open
        mode = 'wt' if compress else 'w'

        with open_func(file_path, mode, encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_json(self, file_path: str) -> List[DataPair]:
        """JSON形式で読み込み"""
        open_func = gzip.open if file_path.endswith('.gz') else open
        mode = 'rt' if file_path.endswith('.gz') else 'r'

        with open_func(file_path, mode, encoding='utf-8') as f:
            data = json.load(f)

        return [DataPair.from_dict(item) for item in data]

    def _save_tasks_json(self, tasks: List[Task], file_path: Path, compress: bool):
        """TaskをJSON形式で保存"""
        data = [task.to_dict() for task in tasks]

        open_func = gzip.open if compress else open
        mode = 'wt' if compress else 'w'

        with open_func(file_path, mode, encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _load_tasks_json(self, file_path: str) -> List[Task]:
        """TaskをJSON形式で読み込み"""
        open_func = gzip.open if file_path.endswith('.gz') else open
        mode = 'rt' if file_path.endswith('.gz') else 'r'

        with open_func(file_path, mode, encoding='utf-8') as f:
            data = json.load(f)

        return [Task.from_dict(item) for item in data]

    def _stream_tasks_json(self, file_path: str) -> Iterator[Task]:
        """TaskをJSON形式でストリーミング読み込み"""
        # JSON形式はストリーミングが困難なため、一度に読み込んでからストリーミング
        tasks = self._load_tasks_json(file_path)
        for task in tasks:
            yield task

    def _save_csv(self, data_pairs: List[DataPair], file_path: Path, compress: bool):
        """CSV形式で保存"""
        # DataPairをCSV形式に変換
        rows = []
        for pair in data_pairs:
            row = {
                'pair_id': pair.pair_id,
                'input_grid': json.dumps(pair.input_grid),
                'output_grid': json.dumps(pair.output_grid),
                'program': pair.program,
                'complexity': pair.complexity,
                'command_sequence': json.dumps(pair.command_sequence),
                'metadata': json.dumps(pair.metadata)
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        if compress:
            df.to_csv(file_path, index=False, compression='gzip')
        else:
            df.to_csv(file_path, index=False)

    def _load_csv(self, file_path: str) -> List[DataPair]:
        """CSV形式で読み込み"""
        if file_path.endswith('.gz'):
            df = pd.read_csv(file_path, compression='gzip')
        else:
            df = pd.read_csv(file_path)

        data_pairs = []
        for _, row in df.iterrows():
            data = {
                'input_grid': json.loads(row['input_grid']),
                'output_grid': json.loads(row['output_grid']),
                'program': row['program'],
                'metadata': json.loads(row['metadata']),
                'pair_id': row['pair_id'],
                'complexity': row['complexity'],
                'command_sequence': json.loads(row['command_sequence'])
            }
            data_pairs.append(DataPair.from_dict(data))

        return data_pairs

    def _save_pickle(self, data: List[Union[DataPair, Task]], file_path: Path, compress: bool):
        """Pickle形式で保存"""
        open_func = gzip.open if compress else open
        mode = 'wb' if compress else 'wb'

        with open_func(file_path, mode) as f:
            pickle.dump(data, f)

    def _load_pickle(self, file_path: str) -> List[Union[DataPair, Task]]:
        """Pickle形式で読み込み"""
        open_func = gzip.open if file_path.endswith('.gz') else open
        mode = 'rb' if file_path.endswith('.gz') else 'rb'

        with open_func(file_path, mode) as f:
            return pickle.load(f)

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """ファイル情報を取得"""
        file_path = Path(file_path)

        if not file_path.exists():
            return {'error': 'ファイルが存在しません'}

        stat = file_path.stat()

        info = {
            'file_path': str(file_path),
            'file_size': stat.st_size,
            'format': self._detect_format(str(file_path)),
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'is_compressed': str(file_path).endswith('.gz')
        }

        # データ数を推定
        try:
            if info['format'] == 'jsonl':
                with open(file_path, 'r', encoding='utf-8') as f:
                    info['estimated_count'] = sum(1 for _ in f)
            elif info['format'] == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info['estimated_count'] = len(data) if isinstance(data, list) else 1
            else:
                info['estimated_count'] = 'unknown'
        except Exception as e:
            info['estimated_count'] = f'error: {e}'

        return info

"""
訓練ロガー

訓練プロセスのロギングとメトリクスの記録
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict
import csv


class TrainingLogger:
    """
    訓練ロガー
    
    訓練中の損失、メトリクス、学習率などを記録
    """
    
    def __init__(self, log_dir: str, experiment_name: str):
        """初期化
        
        Args:
            log_dir: ログディレクトリ
            experiment_name: 実験名
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        
        # ログディレクトリを作成
        os.makedirs(log_dir, exist_ok=True)
        
        # ログファイルのパス
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.json")
        self.csv_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.csv")
        
        # ログデータ
        self.logs = {
            'experiment_name': experiment_name,
            'start_time': timestamp,
            'config': {},
            'epochs': []
        }
        
        # メトリクスの履歴
        self.metrics_history = defaultdict(list)
        
        # CSVヘッダー
        self.csv_headers = ['epoch', 'step', 'loss', 'learning_rate']
        self.csv_initialized = False
    
    def log_config(self, config: Dict[str, Any]):
        """設定を記録
        
        Args:
            config: 設定辞書
        """
        self.logs['config'] = config
        self._save_json()
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], mode: str = 'train'):
        """エポックのメトリクスを記録
        
        Args:
            epoch: エポック番号
            metrics: メトリクス辞書
            mode: 'train'または'val'
        """
        epoch_log = {
            'epoch': epoch,
            'mode': mode,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logs['epochs'].append(epoch_log)
        
        # メトリクスの履歴を更新
        for key, value in metrics.items():
            self.metrics_history[f"{mode}_{key}"].append(value)
        
        self._save_json()
    
    def log_step(
        self,
        epoch: int,
        step: int,
        loss: float,
        learning_rate: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """ステップのメトリクスを記録
        
        Args:
            epoch: エポック番号
            step: ステップ番号
            loss: 損失値
            learning_rate: 学習率
            additional_metrics: 追加メトリクス
        """
        # CSV に記録
        row = {
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'learning_rate': learning_rate if learning_rate is not None else 0.0
        }
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                if key not in self.csv_headers:
                    self.csv_headers.append(key)
                row[key] = value
        
        self._append_to_csv(row)
    
    def log_best_model(self, epoch: int, metrics: Dict[str, float]):
        """最良モデルの情報を記録
        
        Args:
            epoch: エポック番号
            metrics: メトリクス
        """
        self.logs['best_model'] = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        self._save_json()
    
    def get_metrics_history(self, metric_name: str) -> List[float]:
        """メトリクスの履歴を取得
        
        Args:
            metric_name: メトリクス名
        
        Returns:
            メトリクスの履歴
        """
        return self.metrics_history.get(metric_name, [])
    
    def get_best_epoch(self, metric_name: str, mode: str = 'min') -> int:
        """最良エポックを取得
        
        Args:
            metric_name: メトリクス名
            mode: 'min'または'max'
        
        Returns:
            最良エポック番号
        """
        history = self.get_metrics_history(metric_name)
        if not history:
            return -1
        
        if mode == 'min':
            best_epoch = history.index(min(history))
        else:
            best_epoch = history.index(max(history))
        
        return best_epoch
    
    def summary(self) -> Dict[str, Any]:
        """訓練のサマリーを取得
        
        Returns:
            サマリー辞書
        """
        return {
            'experiment_name': self.experiment_name,
            'total_epochs': len([e for e in self.logs['epochs'] if e['mode'] == 'train']),
            'best_model': self.logs.get('best_model', {}),
            'final_metrics': self.logs['epochs'][-1] if self.logs['epochs'] else {}
        }
    
    def _save_json(self):
        """JSONファイルに保存"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.logs, f, indent=2, ensure_ascii=False)
    
    def _append_to_csv(self, row: Dict[str, Any]):
        """CSVファイルに追記"""
        file_exists = os.path.exists(self.csv_file)
        
        with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            
            if not file_exists or not self.csv_initialized:
                writer.writeheader()
                self.csv_initialized = True
            
            writer.writerow(row)
    
    def close(self):
        """ロガーを閉じる"""
        # 最終保存
        self._save_json()
        
        # サマリーを出力
        summary = self.summary()
        summary_file = os.path.join(self.log_dir, f"{self.experiment_name}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

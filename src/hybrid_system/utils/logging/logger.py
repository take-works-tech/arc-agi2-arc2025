"""
基本ロガー

アプリケーション全体のロギングを管理
"""

import logging
import os
from typing import Optional
from datetime import datetime


class Logger:
    """
    基本ロガー

    標準的なPythonロギングをラップし、統一されたインターフェースを提供
    """

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str, log_dir: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
        """ロガーを取得

        Args:
            name: ロガー名
            log_dir: ログディレクトリ
            level: ログレベル

        Returns:
            ロガー
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(level)

        # ハンドラーが既に設定されている場合はスキップ
        if logger.handlers:
            return logger

        # フォーマッタ
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # ファイルハンドラー
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger

        return logger

    @classmethod
    def set_level(cls, name: str, level: int):
        """ログレベルを設定

        Args:
            name: ロガー名
            level: ログレベル
        """
        if name in cls._loggers:
            cls._loggers[name].setLevel(level)

    @classmethod
    def close_logger(cls, name: str):
        """ロガーを閉じる

        Args:
            name: ロガー名
        """
        if name in cls._loggers:
            logger = cls._loggers[name]
            for handler in logger.handlers:
                handler.close()
                logger.removeHandler(handler)
            del cls._loggers[name]

    @classmethod
    def close_all(cls):
        """すべてのロガーを閉じる"""
        for name in list(cls._loggers.keys()):
            cls.close_logger(name)

"""
プログラム合成エンジン専用の設定管理システム

YAML形式の設定ファイルと環境変数による設定上書きをサポート
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field, asdict

# 循環インポートを回避するため、遅延インポートを使用
# from src.hybrid_system.inference.program_synthesis.synthesis_engine import SynthesisConfig


class SynthesisConfigManager:
    """プログラム合成エンジンの設定管理クラス"""

    def __init__(self, config_file: Optional[str] = None):
        """
        初期化

        Args:
            config_file: 設定ファイルのパス（YAML形式）
                        指定しない場合、デフォルトパスを使用
        """
        self.config_file = config_file or self._get_default_config_file()
        self.config = self._load_config()

    def _get_default_config_file(self) -> str:
        """デフォルトの設定ファイルパスを取得"""
        # 1. 環境変数から取得
        env_config = os.getenv("SYNTHESIS_CONFIG_FILE")
        if env_config:
            return env_config

        # 2. デフォルトパス
        default_path = Path("configs/synthesis_config.yaml")
        if default_path.exists():
            return str(default_path)

        # 3. 存在しない場合はNoneを返す（デフォルト設定を使用）
        return None

    def _load_config(self):
        """設定を読み込み"""
        # 遅延インポート
        from src.hybrid_system.inference.program_synthesis.synthesis_engine import SynthesisConfig
        # 1. デフォルト設定を作成
        default_config = SynthesisConfig()

        # 2. YAMLファイルから読み込み（存在する場合）
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                if config_data:
                    # YAMLデータをSynthesisConfigにマージ
                    default_config = self._merge_config(default_config, config_data)
            except Exception as e:
                print(f"警告: 設定ファイルの読み込みに失敗: {e}")
                print(f"デフォルト設定を使用します")

        # 3. 環境変数で上書き
        default_config = self._apply_environment_overrides(default_config)

        return default_config

    def _merge_config(self, default, yaml_data: Dict[str, Any]):
        """YAMLデータをSynthesisConfigにマージ"""
        config_dict = asdict(default)

        # YAMLデータをマージ
        for key, value in yaml_data.items():
            if key in config_dict:
                # 型チェックと変換
                if isinstance(value, type(config_dict[key])):
                    config_dict[key] = value
                elif isinstance(config_dict[key], bool) and isinstance(value, str):
                    # 文字列のbool値を変換
                    config_dict[key] = value.lower() in ('true', '1', 'yes', 'on')
                elif isinstance(config_dict[key], (int, float)) and isinstance(value, (int, float, str)):
                    try:
                        config_dict[key] = type(config_dict[key])(value)
                    except (ValueError, TypeError):
                        pass  # 変換失敗時はスキップ

        return SynthesisConfig(**config_dict)

    def _apply_environment_overrides(self, config):
        """環境変数で設定を上書き"""
        config_dict = asdict(config)

        # 環境変数のマッピング
        env_mappings = {
            'SYNTHESIS_MAX_CANDIDATES': ('max_candidates_per_pair', int),
            'SYNTHESIS_MAX_ATTEMPTS': ('max_synthesis_attempts', int),
            'SYNTHESIS_CONSISTENCY_THRESHOLD': ('consistency_threshold', float),
            'SYNTHESIS_COMPLEXITY_WEIGHT': ('complexity_weight', float),
            'SYNTHESIS_ENABLE_PARALLEL': ('enable_parallel_processing', bool),
            'SYNTHESIS_TIMEOUT': ('timeout_seconds', float),
            'SYNTHESIS_ENABLE_PROGRAM_MERGING': ('enable_program_merging', bool),
            'SYNTHESIS_MIN_CONSISTENCY_THRESHOLD': ('min_consistency_threshold', float),
            'SYNTHESIS_THRESHOLD_DECAY': ('threshold_decay', float),
            'SYNTHESIS_ENABLE_LCS': ('enable_lcs_extraction', bool),
            'SYNTHESIS_ENABLE_CLUSTERING': ('enable_clustering', bool),
            'SYNTHESIS_ENABLE_PROGRAM_SCORER': ('enable_program_scorer', bool),
            'SYNTHESIS_PROGRAM_SCORER_PATH': ('program_scorer_model_path', str),
            'SYNTHESIS_ENABLE_OBJECT_MATCHING': ('enable_object_matching', bool),
            'SYNTHESIS_OBJECT_MATCHING_THRESHOLD': ('object_matching_confidence_threshold', float),
            'SYNTHESIS_ENABLE_DEBUG': ('enable_debug_mode', bool),
            'SYNTHESIS_ENABLE_LOGGING': ('enable_logging', bool),
            'SYNTHESIS_LOG_DIR': ('log_dir', str),
            'SYNTHESIS_ENABLE_VISUALIZATION': ('enable_visualization', bool),
            'SYNTHESIS_VISUALIZATION_DIR': ('visualization_dir', str),
        }

        for env_var, (config_key, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if value_type == bool:
                        # 文字列のbool値を変換
                        config_dict[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type == str:
                        config_dict[config_key] = env_value
                    else:
                        config_dict[config_key] = value_type(env_value)
                except (ValueError, TypeError) as e:
                    print(f"警告: 環境変数 {env_var} の値 '{env_value}' を {value_type.__name__} に変換できませんでした: {e}")

        # 遅延インポート
        from src.hybrid_system.inference.program_synthesis.synthesis_engine import SynthesisConfig
        return SynthesisConfig(**config_dict)

    def get_config(self):
        """設定を取得"""
        return self.config

    def save_config(self, output_path: Optional[str] = None) -> None:
        """現在の設定をYAMLファイルに保存"""
        output_path = output_path or self.config_file or "configs/synthesis_config.yaml"
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self.config)
        # None値を除外
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"設定を保存しました: {output_path}")

    def validate_config(self) -> tuple[bool, list[str]]:
        """
        設定の妥当性を検証

        Returns:
            (is_valid, issues): 妥当性と問題点のリスト
        """
        issues = []
        config_dict = asdict(self.config)

        # 閾値の検証
        if not (0.0 <= config_dict['consistency_threshold'] <= 1.0):
            issues.append(f"consistency_threshold は 0.0-1.0 の範囲である必要があります: {config_dict['consistency_threshold']}")

        if not (0.0 <= config_dict['min_consistency_threshold'] <= 1.0):
            issues.append(f"min_consistency_threshold は 0.0-1.0 の範囲である必要があります: {config_dict['min_consistency_threshold']}")

        if config_dict['min_consistency_threshold'] > config_dict['consistency_threshold']:
            issues.append(f"min_consistency_threshold ({config_dict['min_consistency_threshold']}) は consistency_threshold ({config_dict['consistency_threshold']}) 以下である必要があります")

        if not (0.0 <= config_dict['complexity_weight'] <= 1.0):
            issues.append(f"complexity_weight は 0.0-1.0 の範囲である必要があります: {config_dict['complexity_weight']}")

        if not (0.0 <= config_dict['object_matching_confidence_threshold'] <= 1.0):
            issues.append(f"object_matching_confidence_threshold は 0.0-1.0 の範囲である必要があります: {config_dict['object_matching_confidence_threshold']}")

        # 正の値の検証
        if config_dict['max_candidates_per_pair'] <= 0:
            issues.append(f"max_candidates_per_pair は正の値である必要があります: {config_dict['max_candidates_per_pair']}")

        if config_dict['max_synthesis_attempts'] <= 0:
            issues.append(f"max_synthesis_attempts は正の値である必要があります: {config_dict['max_synthesis_attempts']}")

        if config_dict['timeout_seconds'] <= 0:
            issues.append(f"timeout_seconds は正の値である必要があります: {config_dict['timeout_seconds']}")

        # パスの検証
        if config_dict['program_scorer_model_path']:
            if not os.path.exists(config_dict['program_scorer_model_path']):
                issues.append(f"program_scorer_model_path が存在しません: {config_dict['program_scorer_model_path']}")

        is_valid = len(issues) == 0
        return is_valid, issues


def load_synthesis_config(config_file: Optional[str] = None):
    """
    設定を読み込む便利関数

    Args:
        config_file: 設定ファイルのパス（オプション）

    Returns:
        SynthesisConfig: 読み込まれた設定
    """
    manager = SynthesisConfigManager(config_file)
    return manager.get_config()

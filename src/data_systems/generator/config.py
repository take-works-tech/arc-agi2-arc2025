"""
Generator設定モジュール

generatorフォルダ内のすべての設定値をまとめる
"""
import os
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class GeneratorConfig:
    """Generator全体の設定"""

    # ========================================
    # ファイル・ディレクトリ設定
    # ========================================
    weight_adjustment_dir: str = "outputs/weight_adjustments"
    weight_adjustment_filename: str = "command_weight_adjustments.json"
    batch_progress_filename: str = "batch_progress.json"

    # ========================================
    # データ生成設定
    # ========================================
    batch_size: int = 1000  # 1バッチあたりのタスク数（50万タスク生成用: 500バッチ）
    task_count: int = field(default_factory=lambda: int(os.environ.get('TASK_COUNT', '500000')))  # 本番用: 500000、テスト用: 環境変数TASK_COUNT=100で指定

    # ========================================
    # ログ出力制御
    # ========================================
    enable_debug_output: bool = False
    enable_verbose_output: bool = False
    enable_verbose_logging: bool = field(default_factory=lambda: os.environ.get('ENABLE_VERBOSE_LOGGING', 'false').lower() in ('true', '1', 'yes'))
    enable_all_logs: bool = field(default_factory=lambda: os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes'))
    enable_trace_for_dataset: bool = field(default_factory=lambda: os.environ.get('ENABLE_TRACE_FOR_DATASET', 'false').lower() in ('true', '1', 'yes'))

    # ========================================
    # 条件チェック制御
    # ========================================
    enable_output_condition_check: bool = field(default_factory=lambda: os.environ.get('ENABLE_OUTPUT_CONDITION_CHECK', 'true').lower() in ('true', '1', 'yes'))
    enable_replacement_verification: bool = field(default_factory=lambda: os.environ.get('ENABLE_REPLACEMENT_VERIFICATION', 'true').lower() in ('true', '1', 'yes'))

    # ========================================
    # 詳細タイミングログ制御
    # ========================================
    enable_detailed_timing_logs: bool = field(default_factory=lambda: os.environ.get('ENABLE_DETAILED_TIMING_LOGS', 'false').lower() in ('true', '1', 'yes'))
    enable_debug_logs: bool = field(default_factory=lambda: os.environ.get('ENABLE_DEBUG_LOGS', 'false').lower() in ('true', '1', 'yes'))

    # ========================================
    # 条件④と条件⑥の確率的な破棄設定
    # ========================================
    # 公式データセットの割合: 条件④=4.01%, 条件⑥=1.68%
    # 該当するタスクのうち、一定確率で残すことで分布を調整
    condition4_keep_probability: float = field(default_factory=lambda: float(os.environ.get('CONDITION4_KEEP_PROBABILITY', '0.25')))
    condition6_keep_probability: float = field(default_factory=lambda: float(os.environ.get('CONDITION6_KEEP_PROBABILITY', '0.10')))

    # ========================================
    # 複数ペア生成設定
    # ========================================
    # 1つのプログラムから生成する入出力ペア数（3-10の乱数）
    min_pairs_per_program: int = field(default_factory=lambda: int(os.environ.get('MIN_PAIRS_PER_PROGRAM', '3')))
    max_pairs_per_program: int = field(default_factory=lambda: int(os.environ.get('MAX_PAIRS_PER_PROGRAM', '10')))

    # ========================================
    # ペア生成リトライ設定
    # ========================================
    # 最初以外のペアでエラーが発生した場合の最大リトライ回数
    # プログラム実行エラーは即座にスキップされるため、主に重複やタイムアウトなどのエラーに適用される
    # 50→20に削減: プログラム実行エラーの無駄なリトライを削減したため、20回で十分と判断
    max_pair_retries: int = field(default_factory=lambda: int(os.environ.get('MAX_PAIR_RETRIES', '20')))

    # ========================================
    # グリッド再生成設定
    # ========================================
    # validate_nodes_and_adjust_objects内のグリッド再生成の最大試行回数
    max_consecutive_continues: int = field(default_factory=lambda: int(os.environ.get('MAX_CONSECUTIVE_CONTINUES', '5')))  # 3→5に増加（入力グリッド検証の成功率向上のため）
    max_grid_regeneration_attempts: int = field(default_factory=lambda: None)  # 後で計算（デフォルト: max(MAX_CONSECUTIVE_CONTINUES, 7)に削減）
    max_replacement_commands: int = field(default_factory=lambda: int(os.environ.get('MAX_REPLACEMENT_COMMANDS', '0')))  # 0=無制限、N=最初のN個のみ検証

    # ========================================
    # 実行時間制限
    # ========================================
    max_execution_time: float = field(default_factory=lambda: float(os.environ.get('MAX_EXECUTION_TIME', '3.0')))  # 秒
    max_loop_time: float = 30.0  # ループ全体の最大実行時間（秒）
    max_attempt_time: float = 0.8  # 各試行の最大実行時間（秒）

    # ========================================
    # 複雑度分布
    # ========================================
    complexity_ratios: Dict[int, float] = field(default_factory=lambda: {
        # 10万データセット用の最適比率（手動作成プログラム分析に基づく）
        # 実問題では複雑度3と5が多く存在（合計60%）ため、それに合わせて調整
        # 複雑度1-5: 主要な学習データ（実問題で確認された範囲）
        # 複雑度6-8: 非常に複雑な問題（多様性確保のため少量含める）
        1: 21.0,  # 基礎パターン（実問題: 20%）
        2: 31.0,  # 基本構造（実問題: 8%、学習用に増加）
        3: 17.0,  # 中程度（実問題: 36%、最重要）← 大幅増加
        4: 11.0,  # 複雑な構造（実問題: 12%、学習用に増加）
        5: 9.5,   # 高度なロジック（実問題: 24%、学習用に調整）← 増加
        # 複雑度6-8: 超複雑な問題（手動作成プログラムでは未確認、多様性のために少量含める）
        6: 2.5,   # 超複雑（max_nodes:10, FOR:7, IF:6, ネスト:3）
        7: 1.0,   # 超複雑（max_nodes:16, FOR:8, IF:7, ネスト:4）
        8: 1.0,   # 超複雑（max_nodes:30, FOR:9, IF:8, ネスト:4）
    })

    # ========================================
    # デフォルト値
    # ========================================
    default_complexity: int = 5
    default_grid_size: tuple = (30, 30)  # フォールバック用

    # ========================================
    # 関係性生成コマンドの初期重み調整
    # ========================================
    # 関係性の密度向上のため、関係性生成コマンドの優先度を上げる
    # 重み調整は倍率で指定（1.0がデフォルト、1.5で1.5倍）
    # コマンドクイックリファレンスに基づいて、実際に存在するコマンドのみを指定
    relationship_command_weights: Dict[str, float] = field(default_factory=lambda: {
        # 高優先度（関係性密度向上に重要）
        # 配列操作（オブジェクト間の関係性を生成）
        'MATCH_PAIRS': 1.5,      # ペアマッチング（高優先度）
        'EXTEND_PATTERN': 1.5,   # パターン延長（高優先度）
        'ARRANGE_GRID': 1.4,     # グリッド配置（高優先度）
        'SORT_BY': 1.4,          # ソート（高優先度）

        # 変換操作（オブジェクト間の関係性を生成）
        'INTERSECTION': 1.4,     # 共通部分（高優先度）
        'SUBTRACT': 1.4,          # オブジェクト差分（高優先度）
        'FIT_SHAPE': 1.4,        # 形状フィット（高優先度）
        'FIT_SHAPE_COLOR': 1.4,  # 形状・色フィット（高優先度）
        'FIT_ADJACENT': 1.4,     # 隣接フィット（高優先度）

        # 中優先度
        # 変換操作（対称性・回転）
        'FLIP': 1.3,              # 反転（X/Y、中優先度）- MIRROR_X/Yの代わり
        'ROTATE': 1.2,            # 回転（中優先度）

        # 情報取得（オブジェクト間の距離・関係性を測定）
        'GET_DISTANCE': 1.3,      # ユークリッド距離（中優先度）
        'GET_X_DISTANCE': 1.3,    # X軸距離（中優先度）
        'GET_Y_DISTANCE': 1.3,    # Y軸距離（中優先度）
        'GET_DIRECTION': 1.3,     # オブジェクト間の方向（中優先度）
        'GET_NEAREST': 1.3,       # 最も近いオブジェクト（中優先度）
        'COUNT_ADJACENT': 1.3,    # 隣接セグメント数（中優先度）
        'COUNT_OVERLAP': 1.3,     # 重複ピクセル数（中優先度）

        # 判定関数（オブジェクト間の関係性を判定）
        'IS_SAME_SHAPE': 1.3,     # 形状一致（中優先度）
        'IS_SAME_STRUCT': 1.3,    # 色構造一致（中優先度）
        'IS_IDENTICAL': 1.3,      # 形状・色完全一致（中優先度）
        'IS_INSIDE': 1.2,         # 矩形内判定（包含関係、中優先度）

        # 低優先度（関係性生成に寄与するが、優先度は低め）
        'SPLIT_CONNECTED': 1.1,   # 連結成分分割（低優先度）
    })

    # 関係性生成コマンドの重み調整を有効にするか
    enable_relationship_command_weights: bool = field(default_factory=lambda: os.environ.get('ENABLE_RELATIONSHIP_COMMAND_WEIGHTS', 'true').lower() in ('true', '1', 'yes'))

    # ========================================
    # オブジェクト配置パターンの優先度設定
    # ========================================
    # クラスタ配置、グリッド配置、構造化配置を優先的に使用
    # 重みが高いほど選択される確率が高い
    placement_pattern_weights: Dict[str, float] = field(default_factory=lambda: {
        # 高優先度（関係性密度向上に重要）
        'cluster': 3.0,      # クラスタ配置（高優先度）
        'grid': 3.0,         # グリッド配置（高優先度）
        'structured': 3.0,   # 構造化配置（高優先度）

        # 中優先度
        'spiral': 1.5,       # 螺旋配置（中優先度）
        'symmetry': 1.5,     # 対称配置（中優先度）
        'arc_pattern': 1.5,  # ARCパターン配置（中優先度）

        # 低優先度
        'border': 1.0,       # 境界配置（低優先度）
        'center': 1.0,       # 中央配置（低優先度）
        'random': 0.5,       # ランダム配置（低優先度）
    })

    # 配置パターンの優先度を有効にするか
    enable_placement_pattern_weights: bool = field(default_factory=lambda: os.environ.get('ENABLE_PLACEMENT_PATTERN_WEIGHTS', 'true').lower() in ('true', '1', 'yes'))

    # ========================================
    # オブジェクト間の最小スペース設定
    # ========================================
    # min_spacingを0に設定して、隣接を許可
    default_min_spacing: int = 0  # デフォルト値: 0（隣接を許可）
    force_min_spacing_zero: bool = field(default_factory=lambda: os.environ.get('FORCE_MIN_SPACING_ZERO', 'true').lower() in ('true', '1', 'yes'))  # 常に0に強制するか

    def __post_init__(self):
        """初期化後の処理"""
        # MAX_GRID_REGENERATION_ATTEMPTSのデフォルト値はMAX_CONSECUTIVE_CONTINUESと7の大きい方
        # 10→7に削減: 処理時間短縮のため（validate_nodes_output_v3が全体の57%を占める）
        if self.max_grid_regeneration_attempts is None:
            default_max_attempts = max(self.max_consecutive_continues, 7)
            env_value = os.environ.get('MAX_GRID_REGENERATION_ATTEMPTS', str(default_max_attempts))
            self.max_grid_regeneration_attempts = int(env_value)

        # MAX_CONSECUTIVE_CONTINUESがMAX_GRID_REGENERATION_ATTEMPTSを超えないようにする（安全性のため）
        if self.max_consecutive_continues > self.max_grid_regeneration_attempts:
            self.max_grid_regeneration_attempts = self.max_consecutive_continues


# グローバル設定インスタンス（遅延読み込み）
_global_config: GeneratorConfig = None


def get_config() -> GeneratorConfig:
    """グローバルなGenerator設定を取得（シングルトン）

    Returns:
        GeneratorConfigオブジェクト
    """
    global _global_config
    if _global_config is None:
        _global_config = GeneratorConfig()
    return _global_config


def reload_config() -> GeneratorConfig:
    """Generator設定を再読み込み

    Returns:
        GeneratorConfigオブジェクト
    """
    global _global_config
    _global_config = GeneratorConfig()
    return _global_config


def set_config(config: GeneratorConfig) -> None:
    """グローバルなGenerator設定を設定

    Args:
        config: GeneratorConfigオブジェクト
    """
    global _global_config
    _global_config = config

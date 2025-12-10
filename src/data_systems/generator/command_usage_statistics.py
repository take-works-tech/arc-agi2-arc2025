"""
コマンド使用統計を収集し、極端に少ないコマンドの重みを調整する機能
"""
import os
from typing import Dict, List, Set, Optional
from collections import Counter, defaultdict


class CommandUsageStatistics:
    """コマンド使用統計を収集し、重み調整を提案するクラス"""

    def __init__(self, threshold_ratio: float = 0.01, adjustment_factor: float = 1.5):
        """
        Args:
            threshold_ratio: 極端に少ないと判定する閾値（全体の割合、デフォルト1%）
            adjustment_factor: 重み調整の倍率（デフォルト1.5倍）
        """
        self.command_counts: Counter = Counter()
        self.total_programs: int = 0
        self.threshold_ratio = threshold_ratio
        self.adjustment_factor = adjustment_factor

        # すべての利用可能なコマンドのリスト（統計用）
        self.all_available_commands: Set[str] = set()

        # プログラム生成失敗統計
        self.generation_failures: Counter = Counter()
        self.other_errors: List[Dict[str, str]] = []  # その他のエラーの詳細

    def add_program_commands(self, commands: Set[str]):
        """プログラムで使用されたコマンドを統計に追加

        Args:
            commands: プログラムで使用されたコマンド名のセット
        """
        if commands:
            for cmd in commands:
                self.command_counts[cmd] += 1
                self.all_available_commands.add(cmd)
        self.total_programs += 1

    def get_underused_commands(self) -> List[tuple]:
        """極端に使用頻度が低いコマンドを検出

        Returns:
            (コマンド名, 使用回数, 使用率) のタプルのリスト（使用率の昇順）
        """
        if self.total_programs == 0:
            return []

        threshold_count = max(1, int(self.total_programs * self.threshold_ratio))

        underused = []
        for cmd, count in self.command_counts.items():
            usage_ratio = count / self.total_programs
            if count < threshold_count:
                underused.append((cmd, count, usage_ratio))

        # 使用率でソート
        underused.sort(key=lambda x: x[2])

        return underused

    def get_weight_adjustments(self) -> Dict[str, float]:
        """重み調整の提案を取得

        Returns:
            コマンド名をキー、調整倍率を値とする辞書
        """
        underused_commands = self.get_underused_commands()

        adjustments = {}
        for cmd, count, ratio in underused_commands:
            # 使用率に応じて調整倍率を決定（使用率が低いほど倍率を上げる）
            if ratio == 0:
                multiplier = self.adjustment_factor * 2.0  # 全く使われていない場合は2倍
            elif ratio < 0.005:  # 0.5%未満
                multiplier = self.adjustment_factor * 1.5
            else:
                multiplier = self.adjustment_factor

            adjustments[cmd] = multiplier

        return adjustments

    def get_statistics_summary(self) -> Dict[str, any]:
        """統計サマリーを取得

        Returns:
            統計情報の辞書
        """
        underused = self.get_underused_commands()
        adjustments = self.get_weight_adjustments()

        return {
            'total_programs': self.total_programs,
            'unique_commands': len(self.command_counts),
            'underused_count': len(underused),
            'underused_commands': underused[:10],  # 上位10個
            'adjustments_count': len(adjustments),
            'top_commands': self.command_counts.most_common(10),
        }

    def print_statistics(self):
        """統計情報を出力（ログ抑制対応）"""
        import os
        ENABLE_ALL_LOGS = os.environ.get('ENABLE_ALL_LOGS', 'false').lower() in ('true', '1', 'yes')
        if not ENABLE_ALL_LOGS:
            return  # ログ抑制時は何も出力しない

        stats = self.get_statistics_summary()

        print(f"\n{'='*80}")
        print(f"コマンド使用統計 (総プログラム数: {stats['total_programs']})")
        print(f"{'='*80}")
        print(f"ユニークコマンド数: {stats['unique_commands']}")
        print(f"極端に少ないコマンド数: {stats['underused_count']}")

        if stats['underused_commands']:
            print(f"\n【極端に少ないコマンド（上位10個）】")
            for cmd, count, ratio in stats['underused_commands']:
                print(f"  {cmd:20s}: {count:4d}回 ({ratio*100:.2f}%)")

        if stats['adjustments_count'] > 0:
            adjustments = self.get_weight_adjustments()
            print(f"\n【重み調整提案】")
            for cmd, multiplier in sorted(adjustments.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {cmd:20s}: {multiplier:.2f}倍")

        print(f"\n【最も使用されているコマンド（上位10個）】")
        for cmd, count in stats['top_commands']:
            ratio = count / stats['total_programs'] if stats['total_programs'] > 0 else 0
            print(f"  {cmd:20s}: {count:4d}回 ({ratio*100:.2f}%)")

        print(f"{'='*80}\n")

    def record_generation_failure(self, error_type: str, error_message: str = ""):
        """プログラム生成失敗を記録

        Args:
            error_type: エラーの種類（'runtime_error', 'empty_nodes', 'empty_code', 'missing_get_all_objects', 'condition_arg_failed', 'literal_generation_failed', 'other'）
            error_message: エラーメッセージ（その他のエラーの場合）
        """
        self.generation_failures[error_type] += 1
        if error_type == 'other' and error_message:
            self.other_errors.append({
                'type': error_type,
                'message': error_message[:200]  # メッセージを200文字に制限
            })

    def get_failure_statistics(self) -> Dict:
        """失敗統計を取得

        Returns:
            失敗統計の辞書
        """
        return {
            'total_failures': sum(self.generation_failures.values()),
            'failure_counts': dict(self.generation_failures),
            'other_errors': self.other_errors[:50]  # 最大50件まで
        }


# グローバル統計インスタンス
_global_statistics: Optional[CommandUsageStatistics] = None


def get_global_statistics() -> CommandUsageStatistics:
    """グローバル統計インスタンスを取得（シングルトン）"""
    global _global_statistics
    if _global_statistics is None:
        _global_statistics = CommandUsageStatistics()
    return _global_statistics


def reset_global_statistics():
    """グローバル統計をリセット"""
    global _global_statistics
    _global_statistics = None

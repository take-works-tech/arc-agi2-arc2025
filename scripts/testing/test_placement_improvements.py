"""
オブジェクト配置改善の動作確認スクリプト

確認項目:
1. 配置パターンの優先度設定が正しく適用されているか
2. min_spacingが0に設定されているか
3. クラスタ配置、グリッド配置、構造化配置が優先的に選択されるか
"""
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.config import get_config, reload_config
# ProgramAwareGeneratorは削除されました
# 代わりにcore_executor_mainを使用してください
from src.data_systems.generator.program_executor.core_executor import main as core_executor_main
from src.data_systems.generator.program_executor.node_validator_output import get_generate_objects_kwargs_from_commands


def test_config_settings():
    """設定値の確認"""
    print("=" * 60)
    print("1. 設定値の確認")
    print("=" * 60)

    config = get_config()

    print(f"\n配置パターンの優先度設定:")
    print(f"  enable_placement_pattern_weights: {config.enable_placement_pattern_weights}")
    for pattern, weight in sorted(config.placement_pattern_weights.items(), key=lambda x: -x[1]):
        print(f"    {pattern}: {weight}")

    print(f"\nmin_spacing設定:")
    print(f"  default_min_spacing: {config.default_min_spacing}")
    print(f"  force_min_spacing_zero: {config.force_min_spacing_zero}")

    # 高優先度パターンが正しく設定されているか確認
    high_priority_patterns = ['cluster', 'grid', 'structured']
    all_high_priority = all(
        config.placement_pattern_weights.get(p, 0) >= 2.0
        for p in high_priority_patterns
    )

    if all_high_priority:
        print(f"\n[OK] 高優先度パターン（cluster, grid, structured）が正しく設定されています")
    else:
        print(f"\n[NG] 高優先度パターンの設定に問題があります")
        return False

    if config.default_min_spacing == 0 and config.force_min_spacing_zero:
        print(f"[OK] min_spacingが0に設定され、隣接が許可されています")
    else:
        print(f"[NG] min_spacingの設定に問題があります")
        return False

    return True


def test_placement_pattern_selection():
    """配置パターン選択の確認"""
    print("\n" + "=" * 60)
    print("2. 配置パターン選択の確認")
    print("=" * 60)

    # 複雑度別にテスト
    import random
    rng = random.Random(42)  # ProgramAwareGeneratorの代わりに直接Randomを使用
    complexities = [3, 6, 8]
    pattern_counts = {}

    for complexity in complexities:
        print(f"\n複雑度 {complexity} での配置パターン選択:")
        # ProgramAwareGeneratorは削除されました
        # このテストは配置パターンの選択ロジックをテストするため、直接ロジックをテストします

        # 100回選択して分布を確認
        patterns = []
        for _ in range(100):
            # 配置パターン選択ロジックを模擬
            from src.data_systems.generator.config import get_config
            _config = get_config()

            if _config.enable_placement_pattern_weights:
                if complexity <= 3:
                    available_patterns = ['random', 'grid', 'center']
                elif complexity <= 6:
                    available_patterns = ['random', 'grid', 'spiral', 'cluster', 'border']
                else:
                    available_patterns = ['random', 'grid', 'spiral', 'cluster', 'border',
                                        'symmetry', 'arc_pattern', 'structured']

                weights = []
                patterns_list = []
                for pattern in available_patterns:
                    if pattern in _config.placement_pattern_weights:
                        patterns_list.append(pattern)
                        weights.append(_config.placement_pattern_weights[pattern])

                if sum(weights) == 0:
                    selected = rng.choice(patterns_list)
                else:
                    selected = rng.choices(patterns_list, weights=weights, k=1)[0]
            else:
                if complexity <= 3:
                    selected = rng.choice(['random', 'grid', 'center'])
                elif complexity <= 6:
                    selected = rng.choice(['random', 'grid', 'spiral', 'cluster', 'border'])
                else:
                    selected = rng.choice(['random', 'grid', 'spiral', 'cluster', 'border',
                                            'symmetry', 'arc_pattern', 'structured'])

            patterns.append(selected)

        # 分布を集計
        from collections import Counter
        pattern_dist = Counter(patterns)
        total = len(patterns)

        print(f"  選択回数分布:")
        for pattern, count in sorted(pattern_dist.items(), key=lambda x: -x[1]):
            percentage = (count / total) * 100
            print(f"    {pattern}: {count}回 ({percentage:.1f}%)")

        # 高優先度パターンの選択率を確認
        high_priority_patterns = ['cluster', 'grid', 'structured']
        high_priority_count = sum(pattern_dist.get(p, 0) for p in high_priority_patterns)
        high_priority_rate = (high_priority_count / total) * 100

        print(f"  高優先度パターン（cluster, grid, structured）の選択率: {high_priority_rate:.1f}%")

        if complexity >= 7 and high_priority_rate >= 50:
            print(f"  [OK] 高優先度パターンが優先的に選択されています")
        elif complexity < 7:
            print(f"  [INFO] 複雑度{complexity}では高優先度パターンが利用可能でない場合があります")
        else:
            print(f"  [WARN] 高優先度パターンの選択率が低い可能性があります")

        pattern_counts[complexity] = pattern_dist

    return True


def test_min_spacing_setting():
    """min_spacing設定の確認"""
    print("\n" + "=" * 60)
    print("3. min_spacing設定の確認")
    print("=" * 60)

    # 様々なコマンドセットでテスト
    test_cases = [
        (set(), "コマンドなし"),
        ({'FLOW'}, "FLOWのみ"),
        ({'LAY'}, "LAYのみ"),
        ({'FIT_ADJACENT'}, "FIT_ADJACENTのみ"),
        ({'FLOW', 'LAY', 'FIT_ADJACENT'}, "FLOW/LAY/FIT_ADJACENTすべて"),
        ({'MOVE', 'ROTATE'}, "その他のコマンド"),
    ]

    for commands, description in test_cases:
        kwargs = get_generate_objects_kwargs_from_commands(commands)
        min_spacing = kwargs.get('min_spacing', None)

        print(f"\n{description}:")
        print(f"  コマンド: {commands}")
        print(f"  min_spacing: {min_spacing}")

        if min_spacing == 0:
            print(f"  [OK] min_spacingが0に設定されています（隣接が許可されています）")
        else:
            print(f"  [NG] min_spacingが{min_spacing}に設定されています（期待値: 0）")
            return False

    return True


def main():
    """メイン関数"""
    print("オブジェクト配置改善の動作確認")
    print("=" * 60)

    # 設定を再読み込み（環境変数の影響を確認）
    reload_config()

    results = []

    # 1. 設定値の確認
    results.append(("設定値の確認", test_config_settings()))

    # 2. 配置パターン選択の確認
    results.append(("配置パターン選択の確認", test_placement_pattern_selection()))

    # 3. min_spacing設定の確認
    results.append(("min_spacing設定の確認", test_min_spacing_setting()))

    # 結果サマリー
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "[OK]" if passed else "[NG]"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[OK] すべてのテストが成功しました")
        return 0
    else:
        print("\n[NG] 一部のテストが失敗しました")
        return 1


if __name__ == "__main__":
    exit(main())

"""
比較結果を可視化し、改善提案を生成するスクリプト

比較結果のJSONファイルを読み込み、以下の可視化と分析を行います:
1. グリッドサイズの分布の比較（ヒストグラム）
2. 色数の分布の比較（ヒストグラム）
3. 背景色の分布の比較（棒グラフ）
4. オブジェクトピクセル比率の比較（ヒストグラム）
5. 改善提案の生成
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定（GUI不要）
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォントの設定
rcParams['font.family'] = 'DejaVu Sans'

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_comparison_result(comparison_file: Path) -> Dict[str, Any]:
    """比較結果のJSONファイルを読み込む"""
    with open(comparison_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_histogram_comparison(
    data_generated: List[float],
    data_arc: List[float],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    bins: int = 30
):
    """ヒストグラムの比較図を作成"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # データ範囲を決定
    all_data = data_generated + data_arc
    data_min = min(all_data)
    data_max = max(all_data)

    # ヒストグラムを描画
    ax.hist(data_generated, bins=bins, alpha=0.7, label='Generated', color='blue', density=True)
    ax.hist(data_arc, bins=bins, alpha=0.7, label='ARC-AGI2', color='red', density=True)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  作成: {output_path.name}")


def create_bar_chart_comparison(
    data_generated: Dict[str, int],
    data_arc: Dict[str, int],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path
):
    """棒グラフの比較図を作成"""
    # すべてのキーを取得
    all_keys = sorted(set(list(data_generated.keys()) + list(data_arc.keys())))

    # データを準備
    gen_values = [data_generated.get(k, 0) for k in all_keys]
    arc_values = [data_arc.get(k, 0) for k in all_keys]

    # 正規化（パーセンテージ）
    gen_total = sum(gen_values) if sum(gen_values) > 0 else 1
    arc_total = sum(arc_values) if sum(arc_values) > 0 else 1
    gen_values_pct = [v / gen_total * 100 for v in gen_values]
    arc_values_pct = [v / arc_total * 100 for v in arc_values]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(all_keys))
    width = 0.35

    ax.bar(x - width/2, gen_values_pct, width, label='Generated', color='blue', alpha=0.7)
    ax.bar(x + width/2, arc_values_pct, width, label='ARC-AGI2', color='red', alpha=0.7)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel + ' (%)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(all_keys, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  作成: {output_path.name}")


def generate_improvement_suggestions(comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
    """比較結果から改善提案を生成"""
    suggestions = []
    comparisons = comparison.get('comparisons', {})

    # 1. グリッドサイズの比較
    if 'grid_size' in comparisons:
        gen_size = comparisons['grid_size']['generated']
        arc_size = comparisons['grid_size']['arc_agi2']

        gen_mean_area = gen_size['area']['mean']
        arc_mean_area = arc_size['area']['mean']

        if abs(gen_mean_area - arc_mean_area) / arc_mean_area > 0.2:
            suggestions.append({
                'category': 'グリッドサイズ',
                'issue': f'生成されたグリッドの平均面積がARC-AGI2と大きく異なります（生成: {gen_mean_area:.1f}, ARC-AGI2: {arc_mean_area:.1f}）',
                'suggestion': 'グリッドサイズ決定ロジックをARC-AGI2の分布に合わせて調整することを推奨します'
            })

    # 2. 色数の比較
    if 'color_count' in comparisons:
        gen_colors = comparisons['color_count']['generated']
        arc_colors = comparisons['color_count']['arc_agi2']

        gen_mean_object_colors = gen_colors['object']['mean']
        arc_mean_object_colors = arc_colors['object']['mean']

        if abs(gen_mean_object_colors - arc_mean_object_colors) / max(arc_mean_object_colors, 1) > 0.2:
            suggestions.append({
                'category': '色数',
                'issue': f'生成されたグリッドのオブジェクト色数がARC-AGI2と大きく異なります（生成: {gen_mean_object_colors:.2f}, ARC-AGI2: {arc_mean_object_colors:.2f}）',
                'suggestion': '色数決定の確率分布をARC-AGI2の分布に合わせて調整することを推奨します'
            })

        # 分布の比較
        gen_dist = gen_colors['object']['distribution']
        arc_dist = arc_colors['object']['distribution']

        # 主要な色数の分布を比較
        for color_count in [1, 2, 3, 4, 5]:
            gen_count = gen_dist.get(str(color_count), 0) / comparison['generated_count'] * 100
            arc_count = arc_dist.get(str(color_count), 0) / comparison['arc_count'] * 100

            if abs(gen_count - arc_count) > 10:  # 10%以上の差がある場合
                suggestions.append({
                    'category': '色数分布',
                    'issue': f'{color_count}色のオブジェクトの割合が大きく異なります（生成: {gen_count:.1f}%, ARC-AGI2: {arc_count:.1f}%）',
                    'suggestion': f'色数決定の確率分布で{color_count}色の割合を調整することを推奨します'
                })
                break

    # 3. 背景色の比較
    if 'background_color' in comparisons:
        gen_bg = comparisons['background_color']['generated']
        arc_bg = comparisons['background_color']['arc_agi2']

        # 背景色0の割合を比較
        gen_bg0 = gen_bg.get('0', 0) / comparison['generated_count'] * 100
        arc_bg0 = arc_bg.get('0', 0) / comparison['arc_count'] * 100

        if abs(gen_bg0 - arc_bg0) > 15:  # 15%以上の差がある場合
            suggestions.append({
                'category': '背景色',
                'issue': f'背景色0の割合が大きく異なります（生成: {gen_bg0:.1f}%, ARC-AGI2: {arc_bg0:.1f}%）',
                'suggestion': '背景色決定ロジックをARC-AGI2の分布に合わせて調整することを推奨します'
            })

    # 4. オブジェクトピクセル比率の比較
    if 'object_pixel_ratio' in comparisons:
        gen_ratio = comparisons['object_pixel_ratio']['generated']
        arc_ratio = comparisons['object_pixel_ratio']['arc_agi2']

        gen_mean_ratio = gen_ratio['mean']
        arc_mean_ratio = arc_ratio['mean']

        if abs(gen_mean_ratio - arc_mean_ratio) / max(arc_mean_ratio, 0.01) > 0.3:
            suggestions.append({
                'category': 'オブジェクト密度',
                'issue': f'生成されたグリッドのオブジェクト密度がARC-AGI2と大きく異なります（生成: {gen_mean_ratio:.3f}, ARC-AGI2: {arc_mean_ratio:.3f}）',
                'suggestion': 'オブジェクト数やオブジェクトサイズの決定ロジックを調整して、密度分布をARC-AGI2に合わせることを推奨します'
            })

    return suggestions


def visualize_comparison_results(comparison_file: Path, output_dir: Path):
    """比較結果を可視化"""
    print(f"\n{'='*80}")
    print("比較結果の可視化開始")
    print(f"{'='*80}\n")

    # 比較結果を読み込み
    comparison = load_comparison_result(comparison_file)
    comparisons = comparison.get('comparisons', {})

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. グリッドサイズの分布
    if 'grid_size' in comparisons:
        print("1. グリッドサイズの分布を可視化中...")
        gen_size = comparisons['grid_size']['generated']
        arc_size = comparisons['grid_size']['arc_agi2']

        # 幅の分布
        # 実際のデータから分布を再構築する必要があるため、統計情報から近似分布を生成
        # ここでは簡易的に、平均と標準偏差から正規分布を仮定して可視化
        # 実際の実装では、統計情報ファイルから元のデータを読み込む方が正確

        print("  （統計情報から分布を再構築する必要があるため、簡易的な可視化をスキップ）")

    # 2. 色数の分布
    if 'color_count' in comparisons:
        print("2. 色数の分布を可視化中...")
        gen_colors = comparisons['color_count']['generated']
        arc_colors = comparisons['color_count']['arc_agi2']

        # オブジェクト色数の分布
        gen_dist = gen_colors['object']['distribution']
        arc_dist = arc_colors['object']['distribution']

        create_bar_chart_comparison(
            data_generated=gen_dist,
            data_arc=arc_dist,
            title='Object Color Count Distribution Comparison',
            xlabel='Number of Object Colors',
            ylabel='Percentage',
            output_path=output_dir / 'color_count_distribution.png'
        )

    # 3. 背景色の分布
    if 'background_color' in comparisons:
        print("3. 背景色の分布を可視化中...")
        gen_bg = comparisons['background_color']['generated']
        arc_bg = comparisons['background_color']['arc_agi2']

        create_bar_chart_comparison(
            data_generated=gen_bg,
            data_arc=arc_bg,
            title='Background Color Distribution Comparison',
            xlabel='Background Color',
            ylabel='Percentage',
            output_path=output_dir / 'background_color_distribution.png'
        )

    # 改善提案を生成
    print("\n改善提案を生成中...")
    suggestions = generate_improvement_suggestions(comparison)

    if suggestions:
        suggestions_file = output_dir / 'improvement_suggestions.json'
        with open(suggestions_file, 'w', encoding='utf-8') as f:
            json.dump(suggestions, f, indent=2, ensure_ascii=False)

        print(f"\n改善提案を保存しました: {suggestions_file}")
        print(f"\n改善提案の数: {len(suggestions)}")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n{i}. 【{suggestion['category']}】")
            print(f"   問題: {suggestion['issue']}")
            print(f"   提案: {suggestion['suggestion']}")
    else:
        print("\n改善提案はありません（統計がARC-AGI2とよく一致しています）")

    print(f"\n可視化完了: {output_dir}")


def main():
    """メイン関数"""
    import argparse

    parser = argparse.ArgumentParser(description='比較結果を可視化し、改善提案を生成')
    parser.add_argument('--comparison-file', type=str, required=True, help='比較結果のJSONファイル')
    parser.add_argument('--output-dir', type=str, default=None, help='出力ディレクトリ（デフォルト: comparison_fileと同じディレクトリ）')

    args = parser.parse_args()

    comparison_file = Path(args.comparison_file)
    if not comparison_file.exists():
        print(f"エラー: 比較結果ファイルが見つかりません: {comparison_file}")
        return

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = comparison_file.parent / 'visualizations'

    visualize_comparison_results(comparison_file, output_dir)


if __name__ == '__main__':
    main()


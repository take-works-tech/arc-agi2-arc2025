"""
プログラムありの場合の失敗原因を分析するスクリプト
"""
import json
from pathlib import Path
from collections import Counter

def analyze_failure_reasons():
    """失敗原因を分析"""

    # エラーログを読み込み
    log_file = Path("generate_with_programs_output.txt")
    if not log_file.exists():
        print(f"エラーログファイルが見つかりません: {log_file}")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        log_lines = f.readlines()

    # 統計を集計
    stats = {
        'total_attempts': 0,
        'max_retries_reached': 0,
        'all_objects_empty': 0,
        'program_execution_errors': 0,
        'condition_failures': 0,
        'successful': 0
    }

    error_patterns = {
        'max_retries': '最大試行回数に達しました',
        'all_objects_empty': 'all_objectsが空です',
        'program_error': 'プログラム実行エラー',
        'condition': '条件',
        'success': '成功'
    }

    i = 0
    while i < len(log_lines):
        line = log_lines[i]

        if '最大試行回数に達しました' in line:
            stats['max_retries_reached'] += 1
            stats['total_attempts'] += 1

            # 次の行をチェック
            if i + 1 < len(log_lines):
                next_line = log_lines[i + 1]
                if 'best_objectsなし' in next_line:
                    stats['all_objects_empty'] += 1

        elif 'all_objectsが空です' in line:
            stats['all_objects_empty'] += 1
            stats['total_attempts'] += 1

        elif 'プログラム実行エラー' in line:
            stats['program_execution_errors'] += 1

        elif '成功' in line and '失敗' in line:
            # 進捗行を解析
            if '成功:' in line:
                try:
                    parts = line.split('成功:')[1].split(',')[0].strip()
                    success_count = int(parts)
                    stats['successful'] = success_count
                except:
                    pass
        i += 1

    # 結果を表示
    print("=" * 80)
    print("プログラムありの場合の失敗原因分析")
    print("=" * 80)
    print(f"\n総試行数: {stats['total_attempts']}")
    print(f"成功数: {stats['successful']}")
    print(f"失敗数: {stats['total_attempts'] - stats['successful']}")
    print(f"\n失敗原因:")
    print(f"  最大試行回数に達した: {stats['max_retries_reached']}")
    print(f"  そのうちall_objectsが空だった: {stats['all_objects_empty']}")
    print(f"  プログラム実行エラー: {stats['program_execution_errors']}")

    # 成功した統計を確認
    stats_file = Path("outputs/input_grid_comparison_with_programs/generated_statistics.json")
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            generated_stats = json.load(f)

        print(f"\n成功したタスクの統計:")
        print(f"  総数: {len(generated_stats)}")

        if generated_stats:
            # 色数の分布
            color_counts = [s['num_colors_total'] for s in generated_stats]
            color_dist = Counter(color_counts)
            print(f"\n  色数の分布:")
            for color_count in sorted(color_dist.keys()):
                print(f"    {color_count}色: {color_dist[color_count]}個 ({color_dist[color_count]/len(generated_stats)*100:.1f}%)")

            # オブジェクトピクセル比率
            ratios = [s['object_pixel_ratio'] for s in generated_stats]
            print(f"\n  オブジェクトピクセル比率:")
            print(f"    最小: {min(ratios):.4f}")
            print(f"    最大: {max(ratios):.4f}")
            print(f"    平均: {sum(ratios)/len(ratios):.4f}")
            print(f"    中央値: {sorted(ratios)[len(ratios)//2]:.4f}")

if __name__ == "__main__":
    analyze_failure_reasons()


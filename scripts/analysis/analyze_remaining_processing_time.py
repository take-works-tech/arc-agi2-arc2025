"""
残りの処理時間を分析するスクリプト
"""
import re
from pathlib import Path

def analyze_remaining_time():
    """残りの処理時間を分析"""
    file_path = Path(__file__).parent.parent.parent / 'output_100tasks_generation_optimized.txt'

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 全体実行時間を取得
    match = re.search(r'全体実行時間: ([\d.]+)秒', content)
    total_time = float(match.group(1)) if match else 140.623

    # 各処理の統計を抽出
    # パターン: "処理名:\n  合計: 時間秒"
    # より柔軟なパターンで抽出
    lines = content.split('\n')
    processes = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # 処理名の行を探す（:で終わる行）
        if line and line.endswith(':'):
            name = line[:-1].strip()
            # 次の数行を確認して合計時間を探す
            for j in range(i+1, min(i+10, len(lines))):
                next_line = lines[j].strip()
                match = re.search(r'合計:\s*([\d.]+)秒', next_line)
                if match:
                    time = float(match.group(1))
                    if name not in processes:
                        processes[name] = 0
                    processes[name] += time
                    break
        i += 1

    # カテゴリ別に集計
    categories = {
        'プログラム検証': [],
        'プログラム実行': [],
        'オブジェクト形状生成': [],
        '位置決め': [],
        'その他': []
    }

    for name, time in processes.items():
        if 'validate_nodes_output_v3' in name:
            categories['プログラム検証'].append((name, time))
        elif 'execute_program_string' in name:
            categories['プログラム実行'].append((name, time))
        elif 'generate_objects_from_conditions' in name or 'generate_objects' in name:
            categories['オブジェクト形状生成'].append((name, time))
        elif 'build_grid' in name or 'set_position' in name or 'find_best_position' in name:
            categories['位置決め'].append((name, time))
        else:
            categories['その他'].append((name, time))

    # 結果を出力
    output_file = Path(__file__).parent.parent.parent / 'output_remaining_processing_time_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("残りの処理時間分析レポート\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"全体実行時間: {total_time:.3f}秒\n\n")

        total_categorized = 0
        for category, items in categories.items():
            if items:
                category_total = sum(time for _, time in items)
                total_categorized += category_total
                pct = category_total / total_time * 100
                f.write(f"【{category}】\n")
                f.write(f"  合計: {category_total:.3f}秒 ({pct:.1f}%)\n")
                # 上位5件を表示
                sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:5]
                for name, time in sorted_items:
                    item_pct = time / total_time * 100
                    f.write(f"    - {name}: {time:.3f}秒 ({item_pct:.1f}%)\n")
                f.write("\n")

        # 未分類の時間
        unaccounted = total_time - total_categorized
        unaccounted_pct = unaccounted / total_time * 100
        f.write(f"【未計測・その他の処理】\n")
        f.write(f"  合計: {unaccounted:.3f}秒 ({unaccounted_pct:.1f}%)\n")
        f.write("  （プログラム生成、データ保存、ログ出力、その他のオーバーヘッドなど）\n\n")

        # 主要な処理の内訳
        f.write("=" * 80 + "\n")
        f.write("主要な処理の内訳\n")
        f.write("=" * 80 + "\n\n")

        # プログラム検証
        validate_total = sum(time for _, time in categories['プログラム検証'])
        f.write(f"1. プログラム検証 (validate_nodes_output_v3): {validate_total:.3f}秒 ({validate_total/total_time*100:.1f}%)\n")

        # プログラム実行
        execute_total = sum(time for _, time in categories['プログラム実行'])
        f.write(f"2. プログラム実行 (execute_program_string): {execute_total:.3f}秒 ({execute_total/total_time*100:.1f}%)\n")

        # オブジェクト形状生成
        obj_gen_total = sum(time for _, time in categories['オブジェクト形状生成'])
        f.write(f"3. オブジェクト形状生成: {obj_gen_total:.3f}秒 ({obj_gen_total/total_time*100:.1f}%)\n")

        # 位置決め
        pos_total = sum(time for _, time in categories['位置決め'])
        f.write(f"4. 位置決め: {pos_total:.3f}秒 ({pos_total/total_time*100:.1f}%)\n")

        # その他
        other_total = sum(time for _, time in categories['その他'])
        f.write(f"5. その他（計測済み）: {other_total:.3f}秒 ({other_total/total_time*100:.1f}%)\n")

        # 未計測
        f.write(f"6. 未計測・その他: {unaccounted:.3f}秒 ({unaccounted_pct:.1f}%)\n\n")

        # 合計確認
        f.write(f"合計: {total_categorized + unaccounted:.3f}秒\n")
        f.write(f"（全体実行時間との差分: {abs(total_time - (total_categorized + unaccounted)):.3f}秒）\n")

    print(f"分析結果を {output_file} に出力しました")

if __name__ == '__main__':
    analyze_remaining_time()

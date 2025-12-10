"""
処理時間比較分析スクリプト

前回の結果と最適化後の結果を比較し、位置決めとオブジェクト形状生成の時間を分析
"""
import re
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def extract_timing_stats(file_path):
    """処理時間統計を抽出"""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    stats = {}

    # 全体実行時間
    match = re.search(r'全体実行時間: ([\d.]+)秒', content)
    if match:
        stats['total_time'] = float(match.group(1))

    # DataPair数
    match = re.search(r'DataPair: (\d+)個', content)
    if match:
        stats['data_pairs'] = int(match.group(1))

    # 1タスクあたりの平均時間
    match = re.search(r'1タスクあたりの平均時間: ([\d.]+)秒', content)
    if match:
        stats['avg_time_per_task'] = float(match.group(1))

    # 各処理の統計を抽出
    pattern = r'^([^:]+):\s*合計: ([\d.]+)秒.*?平均: ([\d.]+)秒.*?最大: ([\d.]+)秒.*?回数: (\d+)'
    matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

    for match in matches:
        name = match.group(1).strip()
        total = float(match.group(2))
        avg = float(match.group(3))
        max_time = float(match.group(4))
        count = int(match.group(5))

        stats[name] = {
            'total': total,
            'average': avg,
            'max': max_time,
            'count': count
        }

    return stats


def analyze_object_generation_time(stats):
    """オブジェクト形状生成の時間を集計"""
    total_time = 0.0
    total_count = 0

    for name, data in stats.items():
        if isinstance(data, dict) and 'generate_objects_from_conditions' in name:
            total_time += data['total']
            total_count += data['count']

    return total_time, total_count


def analyze_positioning_time(stats):
    """位置決めの時間を集計"""
    total_time = 0.0
    total_count = 0

    for name, data in stats.items():
        if isinstance(data, dict) and 'build_grid' in name:
            total_time += data['total']
            total_count += data['count']

    return total_time, total_count


def main():
    # 前回の結果
    old_file = project_root / 'output_100tasks_generation.txt'
    # 最適化後の結果
    new_file = project_root / 'output_100tasks_generation_optimized.txt'

    old_stats = extract_timing_stats(old_file)
    new_stats = extract_timing_stats(new_file)

    # オブジェクト形状生成の時間
    old_obj_time, old_obj_count = analyze_object_generation_time(old_stats)
    new_obj_time, new_obj_count = analyze_object_generation_time(new_stats)

    # 位置決めの時間
    old_pos_time, old_pos_count = analyze_positioning_time(old_stats)
    new_pos_time, new_pos_count = analyze_positioning_time(new_stats)

    # 結果を出力
    output_file = project_root / 'output_performance_comparison.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("処理時間比較分析レポート\n")
        f.write("=" * 80 + "\n\n")

        # 全体実行時間の比較
        f.write("【全体実行時間の比較】\n")
        f.write(f"前回: {old_stats.get('total_time', 0):.3f}秒\n")
        f.write(f"最適化後: {new_stats.get('total_time', 0):.3f}秒\n")
        diff = new_stats.get('total_time', 0) - old_stats.get('total_time', 0)
        f.write(f"差分: {diff:+.3f}秒 ({diff/old_stats.get('total_time', 1)*100:+.1f}%)\n\n")

        # オブジェクト形状生成の時間
        f.write("【オブジェクト形状生成の時間】\n")
        f.write(f"前回: {old_obj_time:.3f}秒 (回数: {old_obj_count})\n")
        f.write(f"最適化後: {new_obj_time:.3f}秒 (回数: {new_obj_count})\n")
        obj_diff = new_obj_time - old_obj_time
        f.write(f"差分: {obj_diff:+.3f}秒 ({obj_diff/old_obj_time*100 if old_obj_time > 0 else 0:+.1f}%)\n")
        f.write(f"前回の全体に対する割合: {old_obj_time/old_stats.get('total_time', 1)*100:.1f}%\n")
        f.write(f"最適化後の全体に対する割合: {new_obj_time/new_stats.get('total_time', 1)*100:.1f}%\n\n")

        # 位置決めの時間
        f.write("【位置決めの時間】\n")
        f.write(f"前回: {old_pos_time:.3f}秒 (回数: {old_pos_count})\n")
        f.write(f"最適化後: {new_pos_time:.3f}秒 (回数: {new_pos_count})\n")
        pos_diff = new_pos_time - old_pos_time
        f.write(f"差分: {pos_diff:+.3f}秒 ({pos_diff/old_pos_time*100 if old_pos_time > 0 else 0:+.1f}%)\n")
        f.write(f"前回の全体に対する割合: {old_pos_time/old_stats.get('total_time', 1)*100:.1f}%\n")
        f.write(f"最適化後の全体に対する割合: {new_pos_time/new_stats.get('total_time', 1)*100:.1f}%\n\n")

        # 主要な処理の比較
        f.write("【主要な処理の比較】\n")
        key_processes = [
            'validate_nodes_output_v3 (core_executor)',
            'generate_objects_from_conditions_per_object (max)',
            'execute_program_string (attempt_1)',
        ]

        for process in key_processes:
            if process in old_stats and process in new_stats:
                old_data = old_stats[process]
                new_data = new_stats[process]
                f.write(f"\n{process}:\n")
                f.write(f"  前回: 合計={old_data['total']:.3f}秒, 平均={old_data['average']:.3f}秒, 最大={old_data['max']:.3f}秒, 回数={old_data['count']}\n")
                f.write(f"  最適化後: 合計={new_data['total']:.3f}秒, 平均={new_data['average']:.3f}秒, 最大={new_data['max']:.3f}秒, 回数={new_data['count']}\n")
                total_diff = new_data['total'] - old_data['total']
                f.write(f"  差分: {total_diff:+.3f}秒 ({total_diff/old_data['total']*100 if old_data['total'] > 0 else 0:+.1f}%)\n")

    print(f"分析結果を {output_file} に出力しました")


if __name__ == '__main__':
    main()


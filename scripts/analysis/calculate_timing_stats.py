"""
タイミング情報の統計を計算するスクリプト
"""
import re
import sys
from pathlib import Path
from typing import List, Dict

def extract_all_timing_info(file_path: str) -> List[Dict]:
    """すべてのタイミング情報を抽出"""
    encodings = ['utf-8', 'utf-16', 'shift_jis', 'cp932', 'latin-1']
    lines = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
            break
        except Exception:
            continue

    if lines is None:
        return []

    results = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '[TIMING]' in line or '合計時間:' in line:
            timing_data = {}

            # 合計時間を探す
            if '合計時間:' in line:
                match = re.search(r'合計時間: ([0-9.]+)秒', line)
                if match:
                    timing_data['total'] = float(match.group(1))
                    i += 1
            elif i + 1 < len(lines) and '合計時間:' in lines[i + 1]:
                match = re.search(r'合計時間: ([0-9.]+)秒', lines[i + 1])
                if match:
                    timing_data['total'] = float(match.group(1))
                    i += 2
            else:
                i += 1
                continue

            # 各処理の時間と割合を探す
            patterns = [
                (r'① オブジェクト抽出: ([0-9.]+)秒 \(([0-9.]+)%\)', 'extract'),
                (r'② 背景色推論: ([0-9.]+)秒 \(([0-9.]+)%\)', 'bg_inference'),
                (r'③ 背景色戦略決定: ([0-9.]+)秒 \(([0-9.]+)%\)', 'bg_strategy'),
                (r'④ 変換パターン分析: ([0-9.]+)秒 \(([0-9.]+)%\)', 'pattern_analysis'),
                (r'⑤ ループ処理.*?: ([0-9.]+)秒 \(([0-9.]+)%\)', 'loop')
            ]

            for pattern, key in patterns:
                found = False
                # 次の10行を検索
                for j in range(i, min(i + 10, len(lines))):
                    match = re.search(pattern, lines[j])
                    if match:
                        timing_data[key] = {
                            'time': float(match.group(1)),
                            'ratio': float(match.group(2))
                        }
                        found = True
                        break
                if not found:
                    timing_data[key] = {'time': 0.0, 'ratio': 0.0}

            if 'total' in timing_data:
                results.append(timing_data)
                # 次のタイミング情報までスキップ
                while i < len(lines) and '⑤ ループ処理' not in lines[i]:
                    i += 1
                i += 1
            else:
                i += 1
        else:
            i += 1

    return results

def calculate_stats(values: List[float]) -> Dict:
    """統計値を計算"""
    if not values:
        return {'avg': 0.0, 'max': 0.0, 'min': 0.0, 'count': 0}
    return {
        'avg': sum(values) / len(values),
        'max': max(values),
        'min': min(values),
        'count': len(values)
    }

def main():
    if len(sys.argv) < 2:
        print("使用方法: python calculate_timing_stats.py <出力ファイルパス>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"ファイルが見つかりません: {file_path}")
        sys.exit(1)

    print(f"タイミング情報を抽出中: {file_path}...")
    timing_data = extract_all_timing_info(file_path)

    if not timing_data:
        print("タイミング情報が見つかりませんでした。")
        sys.exit(1)

    print(f"見つかったタイミング情報: {len(timing_data)}件\n")

    # 各処理の時間を抽出
    total_times = [d['total'] for d in timing_data]
    extract_times = [d.get('extract', {}).get('time', 0.0) for d in timing_data]
    loop_times = [d.get('loop', {}).get('time', 0.0) for d in timing_data]

    # 統計を計算
    total_stats = calculate_stats(total_times)
    extract_stats = calculate_stats(extract_times)
    loop_stats = calculate_stats(loop_times)

    print("="*80)
    print("1タスクあたりの時間統計")
    print("="*80)
    print(f"\nタスク数: {len(timing_data)}件")

    print(f"\n【合計時間】")
    print(f"  平均: {total_stats['avg']:.3f}秒")
    print(f"  最大: {total_stats['max']:.3f}秒")
    print(f"  最小: {total_stats['min']:.3f}秒")

    print(f"\n【① オブジェクト抽出】")
    print(f"  平均: {extract_stats['avg']:.3f}秒")
    print(f"  最大: {extract_stats['max']:.3f}秒")
    print(f"  最小: {extract_stats['min']:.3f}秒")

    print(f"\n【⑤ ループ処理（カテゴリ分類+部分プログラム生成）】")
    print(f"  平均: {loop_stats['avg']:.3f}秒")
    print(f"  最大: {loop_stats['max']:.3f}秒")
    print(f"  最小: {loop_stats['min']:.3f}秒")

    # 詳細なデータも表示
    print("\n" + "="*80)
    print("詳細データ")
    print("="*80)
    for i, data in enumerate(timing_data, 1):
        print(f"\nタスク{i}:")
        print(f"  合計時間: {data['total']:.3f}秒")
        print(f"  ① オブジェクト抽出: {data.get('extract', {}).get('time', 0.0):.3f}秒 ({data.get('extract', {}).get('ratio', 0.0):.1f}%)")
        print(f"  ⑤ ループ処理: {data.get('loop', {}).get('time', 0.0):.3f}秒 ({data.get('loop', {}).get('ratio', 0.0):.1f}%)")

if __name__ == '__main__':
    main()
タイミング情報の統計を計算するスクリプト
"""
import re
import sys
from pathlib import Path
from typing import List, Dict

def extract_all_timing_info(file_path: str) -> List[Dict]:
    """すべてのタイミング情報を抽出"""
    encodings = ['utf-8', 'utf-16', 'shift_jis', 'cp932', 'latin-1']
    lines = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                lines = f.readlines()
            break
        except Exception:
            continue

    if lines is None:
        return []

    results = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if '[TIMING]' in line or '合計時間:' in line:
            timing_data = {}

            # 合計時間を探す
            if '合計時間:' in line:
                match = re.search(r'合計時間: ([0-9.]+)秒', line)
                if match:
                    timing_data['total'] = float(match.group(1))
                    i += 1
            elif i + 1 < len(lines) and '合計時間:' in lines[i + 1]:
                match = re.search(r'合計時間: ([0-9.]+)秒', lines[i + 1])
                if match:
                    timing_data['total'] = float(match.group(1))
                    i += 2
            else:
                i += 1
                continue

            # 各処理の時間と割合を探す
            patterns = [
                (r'① オブジェクト抽出: ([0-9.]+)秒 \(([0-9.]+)%\)', 'extract'),
                (r'② 背景色推論: ([0-9.]+)秒 \(([0-9.]+)%\)', 'bg_inference'),
                (r'③ 背景色戦略決定: ([0-9.]+)秒 \(([0-9.]+)%\)', 'bg_strategy'),
                (r'④ 変換パターン分析: ([0-9.]+)秒 \(([0-9.]+)%\)', 'pattern_analysis'),
                (r'⑤ ループ処理.*?: ([0-9.]+)秒 \(([0-9.]+)%\)', 'loop')
            ]

            for pattern, key in patterns:
                found = False
                # 次の10行を検索
                for j in range(i, min(i + 10, len(lines))):
                    match = re.search(pattern, lines[j])
                    if match:
                        timing_data[key] = {
                            'time': float(match.group(1)),
                            'ratio': float(match.group(2))
                        }
                        found = True
                        break
                if not found:
                    timing_data[key] = {'time': 0.0, 'ratio': 0.0}

            if 'total' in timing_data:
                results.append(timing_data)
                # 次のタイミング情報までスキップ
                while i < len(lines) and '⑤ ループ処理' not in lines[i]:
                    i += 1
                i += 1
            else:
                i += 1
        else:
            i += 1

    return results

def calculate_stats(values: List[float]) -> Dict:
    """統計値を計算"""
    if not values:
        return {'avg': 0.0, 'max': 0.0, 'min': 0.0, 'count': 0}
    return {
        'avg': sum(values) / len(values),
        'max': max(values),
        'min': min(values),
        'count': len(values)
    }

def main():
    if len(sys.argv) < 2:
        print("使用方法: python calculate_timing_stats.py <出力ファイルパス>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"ファイルが見つかりません: {file_path}")
        sys.exit(1)

    print(f"タイミング情報を抽出中: {file_path}...")
    timing_data = extract_all_timing_info(file_path)

    if not timing_data:
        print("タイミング情報が見つかりませんでした。")
        sys.exit(1)

    print(f"見つかったタイミング情報: {len(timing_data)}件\n")

    # 各処理の時間を抽出
    total_times = [d['total'] for d in timing_data]
    extract_times = [d.get('extract', {}).get('time', 0.0) for d in timing_data]
    loop_times = [d.get('loop', {}).get('time', 0.0) for d in timing_data]

    # 統計を計算
    total_stats = calculate_stats(total_times)
    extract_stats = calculate_stats(extract_times)
    loop_stats = calculate_stats(loop_times)

    print("="*80)
    print("1タスクあたりの時間統計")
    print("="*80)
    print(f"\nタスク数: {len(timing_data)}件")

    print(f"\n【合計時間】")
    print(f"  平均: {total_stats['avg']:.3f}秒")
    print(f"  最大: {total_stats['max']:.3f}秒")
    print(f"  最小: {total_stats['min']:.3f}秒")

    print(f"\n【① オブジェクト抽出】")
    print(f"  平均: {extract_stats['avg']:.3f}秒")
    print(f"  最大: {extract_stats['max']:.3f}秒")
    print(f"  最小: {extract_stats['min']:.3f}秒")

    print(f"\n【⑤ ループ処理（カテゴリ分類+部分プログラム生成）】")
    print(f"  平均: {loop_stats['avg']:.3f}秒")
    print(f"  最大: {loop_stats['max']:.3f}秒")
    print(f"  最小: {loop_stats['min']:.3f}秒")

    # 詳細なデータも表示
    print("\n" + "="*80)
    print("詳細データ")
    print("="*80)
    for i, data in enumerate(timing_data, 1):
        print(f"\nタスク{i}:")
        print(f"  合計時間: {data['total']:.3f}秒")
        print(f"  ① オブジェクト抽出: {data.get('extract', {}).get('time', 0.0):.3f}秒 ({data.get('extract', {}).get('ratio', 0.0):.1f}%)")
        print(f"  ⑤ ループ処理: {data.get('loop', {}).get('time', 0.0):.3f}秒 ({data.get('loop', {}).get('ratio', 0.0):.1f}%)")

if __name__ == '__main__':
    main()

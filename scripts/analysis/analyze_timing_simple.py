"""
タイミング情報を分析するスクリプト（シンプル版）
"""
import re
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

def extract_timing_info(file_path: str) -> List[Dict]:
    """タイミング情報を抽出"""
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
        if '[TIMING] 部分プログラム生成の時間配分:' in line:
            timing_data = {}
            i += 1

            # 合計時間
            if i < len(lines) and '合計時間:' in lines[i]:
                match = re.search(r'合計時間: ([0-9.]+)秒', lines[i])
                if match:
                    timing_data['total'] = float(match.group(1))
                i += 1

            # 各処理の時間と割合
            patterns = [
                (r'① オブジェクト抽出: ([0-9.]+)秒 \(([0-9.]+)%\)', 'extract'),
                (r'② 背景色推論: ([0-9.]+)秒 \(([0-9.]+)%\)', 'bg_inference'),
                (r'③ 背景色戦略決定: ([0-9.]+)秒 \(([0-9.]+)%\)', 'bg_strategy'),
                (r'④ 変換パターン分析: ([0-9.]+)秒 \(([0-9.]+)%\)', 'pattern_analysis'),
                (r'⑤ ループ処理.*?: ([0-9.]+)秒 \(([0-9.]+)%\)', 'loop')
            ]

            for pattern, key in patterns:
                if i < len(lines):
                    match = re.search(pattern, lines[i])
                    if match:
                        timing_data[key] = {
                            'time': float(match.group(1)),
                            'ratio': float(match.group(2))
                        }
                        i += 1
                    else:
                        timing_data[key] = {'time': 0.0, 'ratio': 0.0}

            if 'total' in timing_data:
                results.append(timing_data)
        else:
            i += 1

    return results

def analyze_timing(timing_data: List[Dict]) -> Dict:
    """タイミング情報を分析"""
    if not timing_data:
        return {}

    # 統計情報を計算
    stats = {
        'count': len(timing_data),
        'total': [],
        'extract': {'times': [], 'ratios': []},
        'bg_inference': {'times': [], 'ratios': []},
        'bg_strategy': {'times': [], 'ratios': []},
        'pattern_analysis': {'times': [], 'ratios': []},
        'loop': {'times': [], 'ratios': []}
    }

    for data in timing_data:
        stats['total'].append(data['total'])
        for key in ['extract', 'bg_inference', 'bg_strategy', 'pattern_analysis', 'loop']:
            if key in data:
                stats[key]['times'].append(data[key]['time'])
                stats[key]['ratios'].append(data[key]['ratio'])

    # 平均、最大、最小を計算
    def calc_stats(values):
        if not values:
            return {'avg': 0, 'max': 0, 'min': 0, 'total': 0}
        return {
            'avg': sum(values) / len(values),
            'max': max(values),
            'min': min(values),
            'total': sum(values)
        }

    analysis = {
        'count': stats['count'],
        'total': calc_stats(stats['total']),
        'extract': {
            'time': calc_stats(stats['extract']['times']),
            'ratio': calc_stats(stats['extract']['ratios'])
        },
        'bg_inference': {
            'time': calc_stats(stats['bg_inference']['times']),
            'ratio': calc_stats(stats['bg_inference']['ratios'])
        },
        'bg_strategy': {
            'time': calc_stats(stats['bg_strategy']['times']),
            'ratio': calc_stats(stats['bg_strategy']['ratios'])
        },
        'pattern_analysis': {
            'time': calc_stats(stats['pattern_analysis']['times']),
            'ratio': calc_stats(stats['pattern_analysis']['ratios'])
        },
        'loop': {
            'time': calc_stats(stats['loop']['times']),
            'ratio': calc_stats(stats['loop']['ratios'])
        }
    }

    return analysis

def print_analysis(analysis: Dict):
    """分析結果を出力"""
    if not analysis:
        print("タイミング情報が見つかりませんでした。")
        return

    print("="*80)
    print("タイミング分析結果")
    print("="*80)
    print(f"\n対象タスク数: {analysis['count']}")
    print(f"\n合計時間: 平均 {analysis['total']['avg']:.3f}秒, 最大 {analysis['total']['max']:.3f}秒, 最小 {analysis['total']['min']:.3f}秒")

    print("\n各処理の時間配分:")
    print(f"  ① オブジェクト抽出:")
    print(f"    時間: 平均 {analysis['extract']['time']['avg']:.3f}秒, 最大 {analysis['extract']['time']['max']:.3f}秒, 最小 {analysis['extract']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['extract']['ratio']['avg']:.1f}%, 最大 {analysis['extract']['ratio']['max']:.1f}%, 最小 {analysis['extract']['ratio']['min']:.1f}%")

    print(f"\n  ② 背景色推論:")
    print(f"    時間: 平均 {analysis['bg_inference']['time']['avg']:.3f}秒, 最大 {analysis['bg_inference']['time']['max']:.3f}秒, 最小 {analysis['bg_inference']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['bg_inference']['ratio']['avg']:.1f}%, 最大 {analysis['bg_inference']['ratio']['max']:.1f}%, 最小 {analysis['bg_inference']['ratio']['min']:.1f}%")

    print(f"\n  ③ 背景色戦略決定:")
    print(f"    時間: 平均 {analysis['bg_strategy']['time']['avg']:.3f}秒, 最大 {analysis['bg_strategy']['time']['max']:.3f}秒, 最小 {analysis['bg_strategy']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['bg_strategy']['ratio']['avg']:.1f}%, 最大 {analysis['bg_strategy']['ratio']['max']:.1f}%, 最小 {analysis['bg_strategy']['ratio']['min']:.1f}%")

    print(f"\n  ④ 変換パターン分析:")
    print(f"    時間: 平均 {analysis['pattern_analysis']['time']['avg']:.3f}秒, 最大 {analysis['pattern_analysis']['time']['max']:.3f}秒, 最小 {analysis['pattern_analysis']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['pattern_analysis']['ratio']['avg']:.1f}%, 最大 {analysis['pattern_analysis']['ratio']['max']:.1f}%, 最小 {analysis['pattern_analysis']['ratio']['min']:.1f}%")

    print(f"\n  ⑤ ループ処理（カテゴリ分類+部分プログラム生成）:")
    print(f"    時間: 平均 {analysis['loop']['time']['avg']:.3f}秒, 最大 {analysis['loop']['time']['max']:.3f}秒, 最小 {analysis['loop']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['loop']['ratio']['avg']:.1f}%, 最大 {analysis['loop']['ratio']['max']:.1f}%, 最小 {analysis['loop']['ratio']['min']:.1f}%")

    # 最も時間がかかる処理を特定
    avg_ratios = {
        'オブジェクト抽出': analysis['extract']['ratio']['avg'],
        '背景色推論': analysis['bg_inference']['ratio']['avg'],
        '背景色戦略決定': analysis['bg_strategy']['ratio']['avg'],
        '変換パターン分析': analysis['pattern_analysis']['ratio']['avg'],
        'ループ処理': analysis['loop']['ratio']['avg']
    }

    max_ratio_process = max(avg_ratios.items(), key=lambda x: x[1])

    print("\n" + "="*80)
    print("ボトルネック分析")
    print("="*80)
    print(f"\n最も時間がかかる処理: {max_ratio_process[0]} ({max_ratio_process[1]:.1f}%)")
    print("\n時間配分の順位:")
    for process, ratio in sorted(avg_ratios.items(), key=lambda x: x[1], reverse=True):
        print(f"  {process}: {ratio:.1f}%")

def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_timing_simple.py <出力ファイルパス>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"ファイルが見つかりません: {file_path}")
        sys.exit(1)

    print(f"タイミング情報を抽出中: {file_path}...")
    timing_data = extract_timing_info(file_path)

    if not timing_data:
        print("タイミング情報が見つかりませんでした。")
        sys.exit(1)

    print(f"見つかったタイミング情報: {len(timing_data)}件\n")

    analysis = analyze_timing(timing_data)
    print_analysis(analysis)

if __name__ == '__main__':
    main()
タイミング情報を分析するスクリプト（シンプル版）
"""
import re
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

def extract_timing_info(file_path: str) -> List[Dict]:
    """タイミング情報を抽出"""
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
        if '[TIMING] 部分プログラム生成の時間配分:' in line:
            timing_data = {}
            i += 1

            # 合計時間
            if i < len(lines) and '合計時間:' in lines[i]:
                match = re.search(r'合計時間: ([0-9.]+)秒', lines[i])
                if match:
                    timing_data['total'] = float(match.group(1))
                i += 1

            # 各処理の時間と割合
            patterns = [
                (r'① オブジェクト抽出: ([0-9.]+)秒 \(([0-9.]+)%\)', 'extract'),
                (r'② 背景色推論: ([0-9.]+)秒 \(([0-9.]+)%\)', 'bg_inference'),
                (r'③ 背景色戦略決定: ([0-9.]+)秒 \(([0-9.]+)%\)', 'bg_strategy'),
                (r'④ 変換パターン分析: ([0-9.]+)秒 \(([0-9.]+)%\)', 'pattern_analysis'),
                (r'⑤ ループ処理.*?: ([0-9.]+)秒 \(([0-9.]+)%\)', 'loop')
            ]

            for pattern, key in patterns:
                if i < len(lines):
                    match = re.search(pattern, lines[i])
                    if match:
                        timing_data[key] = {
                            'time': float(match.group(1)),
                            'ratio': float(match.group(2))
                        }
                        i += 1
                    else:
                        timing_data[key] = {'time': 0.0, 'ratio': 0.0}

            if 'total' in timing_data:
                results.append(timing_data)
        else:
            i += 1

    return results

def analyze_timing(timing_data: List[Dict]) -> Dict:
    """タイミング情報を分析"""
    if not timing_data:
        return {}

    # 統計情報を計算
    stats = {
        'count': len(timing_data),
        'total': [],
        'extract': {'times': [], 'ratios': []},
        'bg_inference': {'times': [], 'ratios': []},
        'bg_strategy': {'times': [], 'ratios': []},
        'pattern_analysis': {'times': [], 'ratios': []},
        'loop': {'times': [], 'ratios': []}
    }

    for data in timing_data:
        stats['total'].append(data['total'])
        for key in ['extract', 'bg_inference', 'bg_strategy', 'pattern_analysis', 'loop']:
            if key in data:
                stats[key]['times'].append(data[key]['time'])
                stats[key]['ratios'].append(data[key]['ratio'])

    # 平均、最大、最小を計算
    def calc_stats(values):
        if not values:
            return {'avg': 0, 'max': 0, 'min': 0, 'total': 0}
        return {
            'avg': sum(values) / len(values),
            'max': max(values),
            'min': min(values),
            'total': sum(values)
        }

    analysis = {
        'count': stats['count'],
        'total': calc_stats(stats['total']),
        'extract': {
            'time': calc_stats(stats['extract']['times']),
            'ratio': calc_stats(stats['extract']['ratios'])
        },
        'bg_inference': {
            'time': calc_stats(stats['bg_inference']['times']),
            'ratio': calc_stats(stats['bg_inference']['ratios'])
        },
        'bg_strategy': {
            'time': calc_stats(stats['bg_strategy']['times']),
            'ratio': calc_stats(stats['bg_strategy']['ratios'])
        },
        'pattern_analysis': {
            'time': calc_stats(stats['pattern_analysis']['times']),
            'ratio': calc_stats(stats['pattern_analysis']['ratios'])
        },
        'loop': {
            'time': calc_stats(stats['loop']['times']),
            'ratio': calc_stats(stats['loop']['ratios'])
        }
    }

    return analysis

def print_analysis(analysis: Dict):
    """分析結果を出力"""
    if not analysis:
        print("タイミング情報が見つかりませんでした。")
        return

    print("="*80)
    print("タイミング分析結果")
    print("="*80)
    print(f"\n対象タスク数: {analysis['count']}")
    print(f"\n合計時間: 平均 {analysis['total']['avg']:.3f}秒, 最大 {analysis['total']['max']:.3f}秒, 最小 {analysis['total']['min']:.3f}秒")

    print("\n各処理の時間配分:")
    print(f"  ① オブジェクト抽出:")
    print(f"    時間: 平均 {analysis['extract']['time']['avg']:.3f}秒, 最大 {analysis['extract']['time']['max']:.3f}秒, 最小 {analysis['extract']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['extract']['ratio']['avg']:.1f}%, 最大 {analysis['extract']['ratio']['max']:.1f}%, 最小 {analysis['extract']['ratio']['min']:.1f}%")

    print(f"\n  ② 背景色推論:")
    print(f"    時間: 平均 {analysis['bg_inference']['time']['avg']:.3f}秒, 最大 {analysis['bg_inference']['time']['max']:.3f}秒, 最小 {analysis['bg_inference']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['bg_inference']['ratio']['avg']:.1f}%, 最大 {analysis['bg_inference']['ratio']['max']:.1f}%, 最小 {analysis['bg_inference']['ratio']['min']:.1f}%")

    print(f"\n  ③ 背景色戦略決定:")
    print(f"    時間: 平均 {analysis['bg_strategy']['time']['avg']:.3f}秒, 最大 {analysis['bg_strategy']['time']['max']:.3f}秒, 最小 {analysis['bg_strategy']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['bg_strategy']['ratio']['avg']:.1f}%, 最大 {analysis['bg_strategy']['ratio']['max']:.1f}%, 最小 {analysis['bg_strategy']['ratio']['min']:.1f}%")

    print(f"\n  ④ 変換パターン分析:")
    print(f"    時間: 平均 {analysis['pattern_analysis']['time']['avg']:.3f}秒, 最大 {analysis['pattern_analysis']['time']['max']:.3f}秒, 最小 {analysis['pattern_analysis']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['pattern_analysis']['ratio']['avg']:.1f}%, 最大 {analysis['pattern_analysis']['ratio']['max']:.1f}%, 最小 {analysis['pattern_analysis']['ratio']['min']:.1f}%")

    print(f"\n  ⑤ ループ処理（カテゴリ分類+部分プログラム生成）:")
    print(f"    時間: 平均 {analysis['loop']['time']['avg']:.3f}秒, 最大 {analysis['loop']['time']['max']:.3f}秒, 最小 {analysis['loop']['time']['min']:.3f}秒")
    print(f"    割合: 平均 {analysis['loop']['ratio']['avg']:.1f}%, 最大 {analysis['loop']['ratio']['max']:.1f}%, 最小 {analysis['loop']['ratio']['min']:.1f}%")

    # 最も時間がかかる処理を特定
    avg_ratios = {
        'オブジェクト抽出': analysis['extract']['ratio']['avg'],
        '背景色推論': analysis['bg_inference']['ratio']['avg'],
        '背景色戦略決定': analysis['bg_strategy']['ratio']['avg'],
        '変換パターン分析': analysis['pattern_analysis']['ratio']['avg'],
        'ループ処理': analysis['loop']['ratio']['avg']
    }

    max_ratio_process = max(avg_ratios.items(), key=lambda x: x[1])

    print("\n" + "="*80)
    print("ボトルネック分析")
    print("="*80)
    print(f"\n最も時間がかかる処理: {max_ratio_process[0]} ({max_ratio_process[1]:.1f}%)")
    print("\n時間配分の順位:")
    for process, ratio in sorted(avg_ratios.items(), key=lambda x: x[1], reverse=True):
        print(f"  {process}: {ratio:.1f}%")

def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_timing_simple.py <出力ファイルパス>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"ファイルが見つかりません: {file_path}")
        sys.exit(1)

    print(f"タイミング情報を抽出中: {file_path}...")
    timing_data = extract_timing_info(file_path)

    if not timing_data:
        print("タイミング情報が見つかりませんでした。")
        sys.exit(1)

    print(f"見つかったタイミング情報: {len(timing_data)}件\n")

    analysis = analyze_timing(timing_data)
    print_analysis(analysis)

if __name__ == '__main__':
    main()

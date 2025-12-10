"""
タイミング情報を分析するスクリプト
"""
import re
import sys
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

def extract_timing_info(file_path: str) -> List[Dict]:
    """タイミング情報を抽出"""
    # エンコーディングを自動検出
    encodings = ['utf-8', 'utf-16', 'shift_jis', 'cp932', 'latin-1']
    content = None
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        # バイナリモードで読み込んでデコードを試みる
        with open(file_path, 'rb') as f:
            raw_content = f.read()
        content = raw_content.decode('utf-8', errors='ignore')

    # タイミング情報のパターン（改行を考慮）
    pattern = r'\[TIMING\] 部分プログラム生成の時間配分:\s*\n\s*合計時間: ([0-9.]+)秒\s*\n\s*① オブジェクト抽出: ([0-9.]+)秒 \(([0-9.]+)%\)\s*\n\s*② 背景色推論: ([0-9.]+)秒 \(([0-9.]+)%\)\s*\n\s*③ 背景色戦略決定: ([0-9.]+)秒 \(([0-9.]+)%\)\s*\n\s*④ 変換パターン分析: ([0-9.]+)秒 \(([0-9.]+)%\)\s*\n\s*⑤ ループ処理[^:]*: ([0-9.]+)秒 \(([0-9.]+)%\)'

    matches = re.findall(pattern, content)

    results = []
    for match in matches:
        results.append({
            'total': float(match[0]),
            'extract': {'time': float(match[1]), 'ratio': float(match[2])},
            'bg_inference': {'time': float(match[3]), 'ratio': float(match[4])},
            'bg_strategy': {'time': float(match[5]), 'ratio': float(match[6])},
            'pattern_analysis': {'time': float(match[7]), 'ratio': float(match[8])},
            'loop': {'time': float(match[9]), 'ratio': float(match[10])}
        })

    return results

def analyze_timing(timing_data: List[Dict]) -> Dict:
    """タイミング情報を分析"""
    if not timing_data:
        return {}

    # 統計情報を計算
    stats = {
        'count': len(timing_data),
        'total': defaultdict(list),
        'extract': defaultdict(list),
        'bg_inference': defaultdict(list),
        'bg_strategy': defaultdict(list),
        'pattern_analysis': defaultdict(list),
        'loop': defaultdict(list)
    }

    for data in timing_data:
        stats['total']['times'].append(data['total'])
        stats['extract']['times'].append(data['extract']['time'])
        stats['extract']['ratios'].append(data['extract']['ratio'])
        stats['bg_inference']['times'].append(data['bg_inference']['time'])
        stats['bg_inference']['ratios'].append(data['bg_inference']['ratio'])
        stats['bg_strategy']['times'].append(data['bg_strategy']['time'])
        stats['bg_strategy']['ratios'].append(data['bg_strategy']['ratio'])
        stats['pattern_analysis']['times'].append(data['pattern_analysis']['time'])
        stats['pattern_analysis']['ratios'].append(data['pattern_analysis']['ratio'])
        stats['loop']['times'].append(data['loop']['time'])
        stats['loop']['ratios'].append(data['loop']['ratio'])

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
        'total': calc_stats(stats['total']['times']),
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
        print("使用方法: python analyze_timing.py <出力ファイルパス>")
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

    print(f"見つかったタイミング情報: {len(timing_data)}件")

    analysis = analyze_timing(timing_data)
    print_analysis(analysis)

if __name__ == '__main__':
    main()

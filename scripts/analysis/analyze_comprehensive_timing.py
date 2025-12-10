"""
包括的なタイミング分析スクリプト

試行タスク数、全体時間、計測できていない時間を分析
"""
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

def extract_timing_info(file_path: str) -> List[Dict]:
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
                while i < len(lines) and '⑤ ループ処理' not in lines[i]:
                    i += 1
                i += 1
            else:
                i += 1
        else:
            i += 1

    return results

def extract_task_info(file_path: str) -> Dict:
    """タスク情報を抽出"""
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
        return {}

    info = {
        'total_tasks': 0,
        'success_count': 0,
        'error_count': 0,
        'start_time': None,
        'end_time': None,
        'total_seconds': 0.0
    }

    # TASK_COUNTを探す
    for line in lines:
        if 'TASK_COUNT:' in line:
            match = re.search(r'TASK_COUNT:\s*(\d+)', line)
            if match:
                info['total_tasks'] = int(match.group(1))

        # 開始時刻を探す
        if '開始時刻:' in line:
            match = re.search(r'開始時刻:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                info['start_time'] = match.group(1)

        # 終了時刻を探す
        if '終了時刻:' in line:
            match = re.search(r'終了時刻:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                info['end_time'] = match.group(1)

        # 総実行時間を探す
        if '総実行時間:' in line:
            match = re.search(r'総実行時間:\s*([0-9.]+)秒', line)
            if match:
                info['total_seconds'] = float(match.group(1))

        # 処理タスク数を探す
        if '処理タスク数:' in line:
            match = re.search(r'処理タスク数:\s*(\d+)', line)
            if match:
                info['success_count'] = int(match.group(1))

    # エラー数をカウント
    error_pattern = r'\[ERROR\]\s*タスク\d+の生成に失敗'
    error_count = len([line for line in lines if re.search(error_pattern, line)])
    info['error_count'] = error_count

    return info

def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_comprehensive_timing.py <出力ファイルパス>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"ファイルが見つかりません: {file_path}")
        sys.exit(1)

    print(f"タイミング情報を分析中: {file_path}...")

    # タイミング情報を抽出
    timing_data = extract_timing_info(file_path)
    print(f"見つかったタイミング情報: {len(timing_data)}件")

    # タスク情報を抽出
    task_info = extract_task_info(file_path)

    print("\n" + "="*80)
    print("包括的タイミング分析結果")
    print("="*80)

    print(f"\n【試行タスク数】")
    print(f"  設定タスク数: {task_info.get('total_tasks', 0)}")
    print(f"  成功タスク数: {task_info.get('success_count', 0)}")
    print(f"  失敗タスク数: {task_info.get('error_count', 0)}")

    if task_info.get('start_time'):
        print(f"\n【実行時間】")
        print(f"  開始時刻: {task_info['start_time']}")
        if task_info.get('end_time'):
            print(f"  終了時刻: {task_info['end_time']}")
        if task_info.get('total_seconds', 0) > 0:
            total_seconds = task_info['total_seconds']
            total_minutes = total_seconds / 60
            print(f"  総実行時間: {total_seconds:.2f}秒 ({total_minutes:.2f}分)")

    if timing_data:
        # 部分プログラム生成の統計
        total_times = [d['total'] for d in timing_data]
        avg_partial_time = sum(total_times) / len(total_times) if total_times else 0

        print(f"\n【部分プログラム生成の時間（{len(timing_data)}タスク分）】")
        print(f"  平均: {avg_partial_time:.3f}秒")
        print(f"  最大: {max(total_times):.3f}秒")
        print(f"  最小: {min(total_times):.3f}秒")
        print(f"  合計: {sum(total_times):.3f}秒")

        # 計測できていない時間を計算
        if task_info.get('total_seconds', 0) > 0:
            measured_time = sum(total_times)
            total_time = task_info['total_seconds']
            unmeasured_time = total_time - measured_time

            print(f"\n【計測できている時間】")
            print(f"  部分プログラム生成: {measured_time:.3f}秒")
            print(f"  総実行時間: {total_time:.2f}秒")
            print(f"  計測できていない時間: {unmeasured_time:.2f}秒 ({unmeasured_time/total_time*100:.1f}%)")

            print(f"\n【計測できていない時間の内訳（推定）】")
            print(f"  - グリッドサイズ決定")
            print(f"  - インプットグリッド生成")
            print(f"  - 部分プログラムパース")
            print(f"  - プログラムノード生成")
            print(f"  - プログラムコード生成")
            print(f"  - 失敗したタスクの処理時間")
            print(f"  - バッファフラッシュ")
            print(f"  - その他のオーバーヘッド")

if __name__ == '__main__':
    main()
包括的なタイミング分析スクリプト

試行タスク数、全体時間、計測できていない時間を分析
"""
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

def extract_timing_info(file_path: str) -> List[Dict]:
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
                while i < len(lines) and '⑤ ループ処理' not in lines[i]:
                    i += 1
                i += 1
            else:
                i += 1
        else:
            i += 1

    return results

def extract_task_info(file_path: str) -> Dict:
    """タスク情報を抽出"""
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
        return {}

    info = {
        'total_tasks': 0,
        'success_count': 0,
        'error_count': 0,
        'start_time': None,
        'end_time': None,
        'total_seconds': 0.0
    }

    # TASK_COUNTを探す
    for line in lines:
        if 'TASK_COUNT:' in line:
            match = re.search(r'TASK_COUNT:\s*(\d+)', line)
            if match:
                info['total_tasks'] = int(match.group(1))

        # 開始時刻を探す
        if '開始時刻:' in line:
            match = re.search(r'開始時刻:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                info['start_time'] = match.group(1)

        # 終了時刻を探す
        if '終了時刻:' in line:
            match = re.search(r'終了時刻:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                info['end_time'] = match.group(1)

        # 総実行時間を探す
        if '総実行時間:' in line:
            match = re.search(r'総実行時間:\s*([0-9.]+)秒', line)
            if match:
                info['total_seconds'] = float(match.group(1))

        # 処理タスク数を探す
        if '処理タスク数:' in line:
            match = re.search(r'処理タスク数:\s*(\d+)', line)
            if match:
                info['success_count'] = int(match.group(1))

    # エラー数をカウント
    error_pattern = r'\[ERROR\]\s*タスク\d+の生成に失敗'
    error_count = len([line for line in lines if re.search(error_pattern, line)])
    info['error_count'] = error_count

    return info

def main():
    if len(sys.argv) < 2:
        print("使用方法: python analyze_comprehensive_timing.py <出力ファイルパス>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"ファイルが見つかりません: {file_path}")
        sys.exit(1)

    print(f"タイミング情報を分析中: {file_path}...")

    # タイミング情報を抽出
    timing_data = extract_timing_info(file_path)
    print(f"見つかったタイミング情報: {len(timing_data)}件")

    # タスク情報を抽出
    task_info = extract_task_info(file_path)

    print("\n" + "="*80)
    print("包括的タイミング分析結果")
    print("="*80)

    print(f"\n【試行タスク数】")
    print(f"  設定タスク数: {task_info.get('total_tasks', 0)}")
    print(f"  成功タスク数: {task_info.get('success_count', 0)}")
    print(f"  失敗タスク数: {task_info.get('error_count', 0)}")

    if task_info.get('start_time'):
        print(f"\n【実行時間】")
        print(f"  開始時刻: {task_info['start_time']}")
        if task_info.get('end_time'):
            print(f"  終了時刻: {task_info['end_time']}")
        if task_info.get('total_seconds', 0) > 0:
            total_seconds = task_info['total_seconds']
            total_minutes = total_seconds / 60
            print(f"  総実行時間: {total_seconds:.2f}秒 ({total_minutes:.2f}分)")

    if timing_data:
        # 部分プログラム生成の統計
        total_times = [d['total'] for d in timing_data]
        avg_partial_time = sum(total_times) / len(total_times) if total_times else 0

        print(f"\n【部分プログラム生成の時間（{len(timing_data)}タスク分）】")
        print(f"  平均: {avg_partial_time:.3f}秒")
        print(f"  最大: {max(total_times):.3f}秒")
        print(f"  最小: {min(total_times):.3f}秒")
        print(f"  合計: {sum(total_times):.3f}秒")

        # 計測できていない時間を計算
        if task_info.get('total_seconds', 0) > 0:
            measured_time = sum(total_times)
            total_time = task_info['total_seconds']
            unmeasured_time = total_time - measured_time

            print(f"\n【計測できている時間】")
            print(f"  部分プログラム生成: {measured_time:.3f}秒")
            print(f"  総実行時間: {total_time:.2f}秒")
            print(f"  計測できていない時間: {unmeasured_time:.2f}秒 ({unmeasured_time/total_time*100:.1f}%)")

            print(f"\n【計測できていない時間の内訳（推定）】")
            print(f"  - グリッドサイズ決定")
            print(f"  - インプットグリッド生成")
            print(f"  - 部分プログラムパース")
            print(f"  - プログラムノード生成")
            print(f"  - プログラムコード生成")
            print(f"  - 失敗したタスクの処理時間")
            print(f"  - バッファフラッシュ")
            print(f"  - その他のオーバーヘッド")

if __name__ == '__main__':
    main()

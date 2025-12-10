"""
出力ファイルからタイミング情報と実行時間を抽出するスクリプト
"""
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional

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
        if '[TIMING]' in line and '部分プログラム生成' in line:
            timing_data = {}

            # 次の数行を確認
            for j in range(i + 1, min(i + 10, len(lines))):
                if '合計時間:' in lines[j]:
                    match = re.search(r'合計時間:\s*([0-9.]+)秒', lines[j])
                    if match:
                        timing_data['total'] = float(match.group(1))

                patterns = [
                    (r'① オブジェクト抽出:\s*([0-9.]+)秒\s*\(([0-9.]+)%\)', 'extract'),
                    (r'② 背景色推論:\s*([0-9.]+)秒\s*\(([0-9.]+)%\)', 'bg_inference'),
                    (r'③ 背景色戦略決定:\s*([0-9.]+)秒\s*\(([0-9.]+)%\)', 'bg_strategy'),
                    (r'④ 変換パターン分析:\s*([0-9.]+)秒\s*\(([0-9.]+)%\)', 'pattern_analysis'),
                    (r'⑤ ループ処理.*?:\s*([0-9.]+)秒\s*\(([0-9.]+)%\)', 'loop')
                ]

                for pattern, key in patterns:
                    match = re.search(pattern, lines[j])
                    if match:
                        timing_data[key] = {
                            'time': float(match.group(1)),
                            'ratio': float(match.group(2))
                        }

            if 'total' in timing_data:
                results.append(timing_data)

            i += 10
        else:
            i += 1

    return results

def extract_execution_info(file_path: str) -> Dict:
    """実行情報を抽出"""
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
        'start_time': None,
        'end_time': None,
        'total_seconds': None,
        'task_count': 10,  # デフォルト値
        'success_count': 0,
        'error_count': 0,
        'total_lines': len(lines)
    }

    for line in lines:
        if '開始時刻:' in line:
            match = re.search(r'開始時刻:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                info['start_time'] = match.group(1)

        if '終了時刻:' in line:
            match = re.search(r'終了時刻:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                info['end_time'] = match.group(1)

        if '総実行時間:' in line:
            match = re.search(r'総実行時間:\s*([0-9.]+)秒', line)
            if match:
                info['total_seconds'] = float(match.group(1))

        if 'TASK_COUNT:' in line:
            match = re.search(r'TASK_COUNT:\s*(\d+)', line)
            if match:
                info['task_count'] = int(match.group(1))

        if '処理タスク数:' in line:
            match = re.search(r'処理タスク数:\s*(\d+)', line)
            if match:
                info['success_count'] = int(match.group(1))

        if re.search(r'\[ERROR\]\s*タスク\d+の生成に失敗', line):
            info['error_count'] += 1

    return info

def main():
    if len(sys.argv) < 2:
        print("使用方法: python extract_timing_from_output.py <出力ファイルパス>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"ファイルが見つかりません: {file_path}")
        sys.exit(1)

    print(f"出力ファイルを分析中: {file_path}...")

    # タイミング情報を抽出
    timing_data = extract_timing_info(file_path)
    print(f"見つかったタイミング情報: {len(timing_data)}件")

    # 実行情報を抽出
    exec_info = extract_execution_info(file_path)

    print("\n" + "="*80)
    print("タイミング分析結果")
    print("="*80)

    print(f"\n【ファイル情報】")
    print(f"  総行数: {exec_info['total_lines']}")
    print(f"  開始時刻: {exec_info['start_time'] or '未記録'}")
    print(f"  終了時刻: {exec_info['end_time'] or '実行中'}")

    if exec_info['start_time']:
        from datetime import datetime
        try:
            start = datetime.strptime(exec_info['start_time'], '%Y-%m-%d %H:%M:%S')
            if exec_info['end_time']:
                end = datetime.strptime(exec_info['end_time'], '%Y-%m-%d %H:%M:%S')
                elapsed = (end - start).total_seconds()
                print(f"  経過時間: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")
            else:
                from datetime import datetime as dt
                now = dt.now()
                elapsed = (now - start).total_seconds()
                print(f"  経過時間（現在）: {elapsed:.2f}秒 ({elapsed/60:.2f}分) - 実行中")
        except Exception:
            pass

    if exec_info['total_seconds']:
        print(f"  総実行時間（記録済み）: {exec_info['total_seconds']:.2f}秒")

    print(f"\n【タスク情報】")
    print(f"  設定タスク数: {exec_info['task_count']}")
    print(f"  成功タスク数: {exec_info['success_count']}")
    print(f"  失敗タスク数: {exec_info['error_count']}")

    if timing_data:
        print(f"\n【部分プログラム生成の時間（{len(timing_data)}タスク分）】")
        total_times = [d['total'] for d in timing_data]
        print(f"  合計: {sum(total_times):.3f}秒")
        print(f"  平均: {sum(total_times)/len(total_times):.3f}秒")
        print(f"  最大: {max(total_times):.3f}秒")
        print(f"  最小: {min(total_times):.3f}秒")

        # 各処理の平均時間
        if all('extract' in d for d in timing_data):
            extract_times = [d['extract']['time'] for d in timing_data]
            loop_times = [d.get('loop', {}).get('time', 0.0) for d in timing_data]
            print(f"\n  ① オブジェクト抽出: 平均 {sum(extract_times)/len(extract_times):.3f}秒")
            print(f"  ⑤ ループ処理: 平均 {sum(loop_times)/len(loop_times):.3f}秒")

        # 計測できていない時間を計算
        if exec_info['start_time']:
            from datetime import datetime
            try:
                start = datetime.strptime(exec_info['start_time'], '%Y-%m-%d %H:%M:%S')
                from datetime import datetime as dt
                now = dt.now()
                elapsed = (now - start).total_seconds()
                measured_time = sum(total_times)
                unmeasured_time = elapsed - measured_time

                print(f"\n【計測できている時間】")
                print(f"  部分プログラム生成: {measured_time:.3f}秒")
                print(f"\n【計測できていない時間】")
                print(f"  経過時間: {elapsed:.2f}秒")
                print(f"  部分プログラム生成以外: {unmeasured_time:.2f}秒 ({unmeasured_time/elapsed*100:.1f}%)")
                print(f"  - プログラム生成処理（部分プログラムパース以降）")
                print(f"  - 失敗したタスクの処理時間")
                print(f"  - その他のオーバーヘッド")
            except Exception:
                pass

if __name__ == '__main__':
    main()

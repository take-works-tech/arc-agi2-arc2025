"""
計測できていない時間を計算するスクリプト
"""
import re
import sys
from pathlib import Path
from datetime import datetime

def extract_all_info(file_path: str) -> dict:
    """すべての情報を抽出"""
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
        'total_lines': len(lines),
        'timing_entries': [],
        'errors': [],
        'task_count': 10
    }

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 開始時刻
        if '開始時刻:' in line and not info['start_time']:
            match = re.search(r'開始時刻:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
            if match:
                info['start_time'] = match.group(1)

        # TASK_COUNT
        if 'TASK_COUNT:' in line:
            match = re.search(r'TASK_COUNT:\s*(\d+)', line)
            if match:
                info['task_count'] = int(match.group(1))

        # タイミング情報
        if '[TIMING]' in line and '部分プログラム生成' in line:
            timing_data = {}
            for j in range(i + 1, min(i + 10, len(lines))):
                if '合計時間:' in lines[j]:
                    match = re.search(r'合計時間:\s*([0-9.]+)秒', lines[j])
                    if match:
                        timing_data['total'] = float(match.group(1))

                patterns = [
                    (r'① オブジェクト抽出:\s*([0-9.]+)秒', 'extract'),
                    (r'⑤ ループ処理.*?:\s*([0-9.]+)秒', 'loop')
                ]

                for pattern, key in patterns:
                    match = re.search(pattern, lines[j])
                    if match:
                        timing_data[key] = float(match.group(1))

            if 'total' in timing_data:
                info['timing_entries'].append(timing_data)

            i += 10
        else:
            i += 1

        # エラー
        if '[ERROR]' in line and 'タスク' in line and 'の生成に失敗' in line:
            match = re.search(r'タスク(\d+)の生成に失敗', line)
            if match:
                info['errors'].append(int(match.group(1)))

    return info

def main():
    if len(sys.argv) < 2:
        print("使用方法: python calculate_unmeasured_time.py <出力ファイルパス>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"ファイルが見つかりません: {file_path}")
        sys.exit(1)

    print(f"出力ファイルを分析中: {file_path}...")

    info = extract_all_info(file_path)

    print("\n" + "="*80)
    print("計測できていない時間の分析")
    print("="*80)

    print(f"\n【基本情報】")
    print(f"  ファイル総行数: {info['total_lines']}")
    print(f"  設定タスク数: {info['task_count']}")
    print(f"  タイミング情報: {len(info['timing_entries'])}件")
    print(f"  エラー数: {len(info['errors'])}件")

    if info['start_time']:
        try:
            start = datetime.strptime(info['start_time'], '%Y-%m-%d %H:%M:%S')
            now = datetime.now()
            elapsed = (now - start).total_seconds()

            print(f"\n【実行時間】")
            print(f"  開始時刻: {info['start_time']}")
            print(f"  現在時刻: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  経過時間: {elapsed:.2f}秒 ({elapsed/60:.2f}分)")

            # 計測できている時間
            if info['timing_entries']:
                measured_partial_time = sum(d['total'] for d in info['timing_entries'])
                print(f"\n【計測できている時間】")
                print(f"  部分プログラム生成: {measured_partial_time:.3f}秒 ({len(info['timing_entries'])}タスク分)")

                # 計測できていない時間
                unmeasured_time = elapsed - measured_partial_time
                print(f"\n【計測できていない時間】")
                print(f"  合計: {unmeasured_time:.2f}秒 ({unmeasured_time/elapsed*100:.1f}%)")
                print(f"  内訳:")
                print(f"    - プログラム生成処理（部分プログラムパース以降）")
                print(f"    - 失敗したタスクの処理時間（{len(info['errors'])}タスク）")
                print(f"    - その他のオーバーヘッド")

        except Exception as e:
            print(f"  時刻解析エラー: {e}")

    if info['timing_entries']:
        print(f"\n【部分プログラム生成の詳細】")
        for i, entry in enumerate(info['timing_entries'], 1):
            print(f"  タスク{i}: {entry['total']:.3f}秒")
            if 'extract' in entry:
                print(f"    - オブジェクト抽出: {entry['extract']:.3f}秒")
            if 'loop' in entry:
                print(f"    - ループ処理: {entry['loop']:.3f}秒")

if __name__ == '__main__':
    main()

"""
テスト出力からオブジェクト数の統計を抽出
"""
import re
from collections import Counter
from pathlib import Path

def extract_object_count_statistics(log_file: str):
    """ログファイルからオブジェクト数の統計を抽出"""

    # エンコーディングを自動検出
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']
    content = None
    for enc in encodings:
        try:
            with open(log_file, 'r', encoding=enc) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        print(f"エラー: ファイル {log_file} を読み込めませんでした")
        return

    # 決定されたオブジェクト数を抽出（num_objects=のパターン）
    decided_object_counts = []
    # 一括生成開始時のnum_objectsを抽出
    pattern_decided = r'一括生成開始.*num_objects=(\d+)'
    matches = re.findall(pattern_decided, content)
    for match in matches:
        decided_object_counts.append(int(match))

    # 実際に生成されたオブジェクト数を抽出
    actual_object_counts = []
    # 「生成オブジェクト数=X」または「要求数=X, 生成数=Y, 有効オブジェクト数=Z」のパターン
    pattern_actual = r'生成オブジェクト数=(\d+)|要求数=\d+,\s*生成数=(\d+),\s*有効オブジェクト数=(\d+)'
    matches = re.findall(pattern_actual, content)
    for match in matches:
        # 複数のグループのうち、最初の非空の値を取得
        for val in match:
            if val:
                actual_object_counts.append(int(val))
                break

    # より正確なパターン: 「ループ終了 - 要求数=X, 生成数=Y, 有効オブジェクト数=Z」
    pattern_loop_end = r'ループ終了\s*-\s*要求数=(\d+),\s*生成数=(\d+),\s*有効オブジェクト数=(\d+)'
    matches = re.findall(pattern_loop_end, content)
    loop_end_requested = []
    loop_end_generated = []
    loop_end_valid = []
    for match in matches:
        loop_end_requested.append(int(match[0]))
        loop_end_generated.append(int(match[1]))
        loop_end_valid.append(int(match[2]))

    print("="*80)
    print("オブジェクト数統計")
    print("="*80)

    if loop_end_valid:
        print(f"\n【一括生成時のオブジェクト数】（{len(loop_end_valid)}ケース）")
        print(f"\n要求されたオブジェクト数の分布:")
        requested_dist = Counter(loop_end_requested)
        for obj_count in sorted(requested_dist.keys()):
            count = requested_dist[obj_count]
            percentage = (count / len(loop_end_requested)) * 100
            print(f"  {obj_count}個: {count}回 ({percentage:.2f}%)")

        print(f"\n要求されたオブジェクト数の統計:")
        if loop_end_requested:
            print(f"  最小: {min(loop_end_requested)}")
            print(f"  最大: {max(loop_end_requested)}")
            print(f"  平均: {sum(loop_end_requested) / len(loop_end_requested):.2f}")
            print(f"  中央値: {sorted(loop_end_requested)[len(loop_end_requested)//2]}")

        print(f"\n実際に生成されたオブジェクト数の分布:")
        valid_dist = Counter(loop_end_valid)
        for obj_count in sorted(valid_dist.keys()):
            count = valid_dist[obj_count]
            percentage = (count / len(loop_end_valid)) * 100
            print(f"  {obj_count}個: {count}回 ({percentage:.2f}%)")

        print(f"\n実際に生成されたオブジェクト数の統計:")
        if loop_end_valid:
            print(f"  最小: {min(loop_end_valid)}")
            print(f"  最大: {max(loop_end_valid)}")
            print(f"  平均: {sum(loop_end_valid) / len(loop_end_valid):.2f}")
            print(f"  中央値: {sorted(loop_end_valid)[len(loop_end_valid)//2]}")

        # 一致率
        matches = sum(1 for req, valid in zip(loop_end_requested, loop_end_valid) if req == valid)
        match_rate = (matches / len(loop_end_requested)) * 100 if loop_end_requested else 0
        print(f"\nオブジェクト数の一致率:")
        print(f"  一致: {matches}/{len(loop_end_requested)} ({match_rate:.2f}%)")

        # 不一致のケース
        mismatches = [(req, valid) for req, valid in zip(loop_end_requested, loop_end_valid) if req != valid]
        if mismatches:
            print(f"\n不一致のケース（最初の10ケース）:")
            for req, valid in mismatches[:10]:
                diff = valid - req
                print(f"  要求数={req}, 実際={valid} (差分={diff:+d})")
            if len(mismatches) > 10:
                print(f"  ... 他{len(mismatches) - 10}ケース")

    if decided_object_counts:
        print(f"\n【決定されたオブジェクト数】（{len(decided_object_counts)}ケース）")
        decided_dist = Counter(decided_object_counts)
        print(f"\n決定されたオブジェクト数の分布:")
        for obj_count in sorted(decided_dist.keys()):
            count = decided_dist[obj_count]
            percentage = (count / len(decided_object_counts)) * 100
            print(f"  {obj_count}個: {count}回 ({percentage:.2f}%)")

        print(f"\n決定されたオブジェクト数の統計:")
        if decided_object_counts:
            print(f"  最小: {min(decided_object_counts)}")
            print(f"  最大: {max(decided_object_counts)}")
            print(f"  平均: {sum(decided_object_counts) / len(decided_object_counts):.2f}")
            print(f"  中央値: {sorted(decided_object_counts)[len(decided_object_counts)//2]}")

if __name__ == '__main__':
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'test_new_probability_output.txt'
    extract_object_count_statistics(log_file)

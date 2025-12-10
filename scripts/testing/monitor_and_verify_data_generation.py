"""
データ生成の監視と検証

データ生成の完了を待ち、新しいデータで検証を実行
"""
import sys
import time
import os
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.data_systems.data_models.base import ObjectType
import json
import gzip
import numpy as np
from collections import Counter


def extract_objects_from_grid(grid, extractor):
    """グリッドからオブジェクトを抽出"""
    try:
        grid_np = np.array(grid)
        result = extractor.extract_objects_by_type(grid_np, input_image_index=0)
        if not result.success:
            return []
        objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
        return objects
    except Exception:
        return []


def analyze_data_file(data_file: Path, extractor, limit: int = 100):
    """データファイルを分析"""
    object_counts = []

    if not data_file.exists():
        return None

    try:
        with gzip.open(data_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                if not line.strip():
                    continue
                try:
                    pair = json.loads(line)
                    if 'input' in pair:
                        input_grid = pair['input']
                    elif 'input_grid' in pair:
                        input_grid = pair['input_grid']
                    else:
                        continue

                    objects = extract_objects_from_grid(input_grid, extractor)
                    object_counts.append(len(objects))
                except Exception:
                    continue
    except Exception:
        return None

    if not object_counts:
        return None

    return {
        'object_counts': object_counts,
        'total_samples': len(object_counts),
        'avg_count': np.mean(object_counts),
        'min_count': min(object_counts),
        'max_count': max(object_counts),
        'median_count': np.median(object_counts)
    }


def print_verification_result(analysis):
    """検証結果を出力"""
    if not analysis:
        print("[NG] データが読み込めませんでした")
        return

    counts = analysis['object_counts']
    range_2_10 = sum(1 for c in counts if 2 <= c <= 10)
    range_10_30 = sum(1 for c in counts if 10 < c <= 30)
    range_30_plus = sum(1 for c in counts if c > 30)

    print("\n" + "="*80)
    print("改善後のデータ生成 - 検証結果")
    print("="*80)
    print(f"\n【基本統計】")
    print(f"  サンプル数: {analysis['total_samples']}")
    print(f"  平均オブジェクト数: {analysis['avg_count']:.2f}")
    print(f"  中央値: {analysis['median_count']:.1f}")
    print(f"  最小: {analysis['min_count']}, 最大: {analysis['max_count']}")

    print(f"\n【範囲別の分布】")
    print(f"  2-10個: {range_2_10}個 ({range_2_10/len(counts)*100:.1f}%)")
    print(f"  10-30個: {range_10_30}個 ({range_10_30/len(counts)*100:.1f}%)")
    print(f"  30個以上: {range_30_plus}個 ({range_30_plus/len(counts)*100:.1f}%)")

    print(f"\n【ARC-AGI2との比較】")
    print(f"  ARC-AGI2（100タスク）:")
    print(f"    2-10個: 約60%")
    print(f"    10-30個: 約30%")
    print(f"    30個以上: 約10%")
    print(f"  生成データ（{analysis['total_samples']}タスク）:")
    print(f"    2-10個: {range_2_10/len(counts)*100:.1f}%")
    print(f"    10-30個: {range_10_30/len(counts)*100:.1f}%")
    print(f"    30個以上: {range_30_plus/len(counts)*100:.1f}%")

    print(f"\n【改善の評価】")
    if range_10_30 / len(counts) >= 0.20:
        print(f"  [OK] 10-30個のオブジェクトを含むタスクが十分に生成されています（{range_10_30/len(counts)*100:.1f}%）")
    else:
        print(f"  [WARN] 10-30個のオブジェクトを含むタスクが不足しています（{range_10_30/len(counts)*100:.1f}%）")

    if range_30_plus / len(counts) >= 0.05:
        print(f"  [OK] 30個以上のオブジェクトを含むタスクが生成されています（{range_30_plus/len(counts)*100:.1f}%）")
    else:
        print(f"  [WARN] 30個以上のオブジェクトを含むタスクが不足しています（{range_30_plus/len(counts)*100:.1f}%）")

    if analysis['avg_count'] >= 10:
        print(f"  [OK] 平均オブジェクト数が改善されています（{analysis['avg_count']:.2f}個）")
    else:
        print(f"  [WARN] 平均オブジェクト数がまだ低いです（{analysis['avg_count']:.2f}個）")

    print("\n" + "="*80)


def main():
    """メイン処理"""
    print("="*80)
    print("データ生成の監視と検証")
    print("="*80)

    data_file = project_root / "data" / "generated" / "phase1_pairs" / "data_pairs.jsonl.gz"

    print(f"\n[1/3] データ生成の完了を待機中...")
    print(f"[INFO] 監視対象: {data_file.name}")

    # 初期のファイルサイズを記録
    initial_size = data_file.stat().st_size if data_file.exists() else 0
    initial_mtime = data_file.stat().st_mtime if data_file.exists() else 0

    print(f"[INFO] 初期状態: サイズ={initial_size} bytes, 更新時刻={datetime.fromtimestamp(initial_mtime).strftime('%Y-%m-%d %H:%M:%S') if initial_mtime > 0 else 'N/A'}")

    # ファイルの更新を監視（最大10分待機）
    max_wait_time = 600  # 10分
    check_interval = 10  # 10秒ごとにチェック
    waited_time = 0

    while waited_time < max_wait_time:
        if data_file.exists():
            current_size = data_file.stat().st_size
            current_mtime = data_file.stat().st_mtime

            # ファイルが更新されたか確認（サイズが増加、または更新時刻が変化）
            if current_size > initial_size or current_mtime > initial_mtime:
                # さらに10秒待って、ファイルが安定するのを待つ
                time.sleep(10)
                print(f"[OK] データファイルが更新されました")
                print(f"[INFO] サイズ: {initial_size} → {current_size} bytes")
                break

        time.sleep(check_interval)
        waited_time += check_interval
        if waited_time % 30 == 0:
            print(f"[INFO] 待機中... ({waited_time}秒経過)")

    if waited_time >= max_wait_time:
        print(f"[WARN] タイムアウト: データ生成が完了しませんでした（{max_wait_time}秒経過）")
        print(f"[INFO] 既存のデータで検証を続行します")

    # データを分析
    print(f"\n[2/3] データを分析中...")
    extractor = IntegratedObjectExtractor(ExtractionConfig())
    analysis = analyze_data_file(data_file, extractor, limit=100)

    if not analysis:
        print(f"[NG] データが読み込めませんでした")
        return 1

    # 結果を出力
    print_verification_result(analysis)

    # 結果を保存
    output_file = project_root / "test_outputs" / "new_data_verification.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'data_file': str(data_file),
            'analysis': analysis
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[3/3] 結果を保存しました: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

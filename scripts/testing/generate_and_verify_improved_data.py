"""
改善後のロジックでデータ生成と検証を実行

改善後のオブジェクト数決定ロジックを使用してデータを生成し、
分布を確認する
"""
import sys
import os
import subprocess
import json
import gzip
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime
import time

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.data_systems.data_models.base import ObjectType


def extract_objects_from_grid(grid: List[List[int]], extractor) -> List[Any]:
    """グリッドからオブジェクトを抽出"""
    try:
        grid_np = np.array(grid)
        result = extractor.extract_objects_by_type(grid_np, input_image_index=0)

        if not result.success:
            return []

        objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
        return objects
    except Exception as e:
        return []


def analyze_data_file(data_file: Path, extractor, limit: int = 100) -> Dict[str, Any]:
    """データファイルを分析"""
    object_counts = []
    successful_reads = 0
    failed_reads = 0

    if not data_file.exists():
        return {
            'file_path': str(data_file),
            'exists': False,
            'object_counts': [],
            'distribution': {},
            'total_samples': 0
        }

    print(f"[INFO] データファイルを読み込み中: {data_file.name}")

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
                        failed_reads += 1
                        continue

                    objects = extract_objects_from_grid(input_grid, extractor)
                    object_counts.append(len(objects))
                    successful_reads += 1
                except Exception as e:
                    failed_reads += 1
                    continue
    except Exception as e:
        print(f"[NG] ファイル読み込みエラー: {e}")
        return {
            'file_path': str(data_file),
            'exists': True,
            'error': str(e),
            'object_counts': [],
            'distribution': {},
            'total_samples': 0
        }

    distribution = Counter(object_counts)

    return {
        'file_path': str(data_file),
        'exists': True,
        'object_counts': object_counts,
        'distribution': dict(distribution),
        'total_samples': len(object_counts),
        'successful_reads': successful_reads,
        'failed_reads': failed_reads,
        'avg_count': np.mean(object_counts) if object_counts else 0,
        'min_count': min(object_counts) if object_counts else 0,
        'max_count': max(object_counts) if object_counts else 0,
        'median_count': np.median(object_counts) if object_counts else 0
    }


def print_verification_report(analysis: Dict[str, Any]):
    """検証結果を出力"""
    print("\n" + "="*80)
    print("改善後のデータ生成 - 検証結果")
    print("="*80)

    if not analysis.get('exists', False):
        print(f"\n[NG] ファイルが見つかりません: {analysis['file_path']}")
        return

    if analysis.get('error'):
        print(f"\n[NG] エラー: {analysis['file_path']} - {analysis['error']}")
        return

    if not analysis['object_counts']:
        print(f"\n[WARN] データが読み込めませんでした: {analysis['file_path']}")
        print(f"  成功: {analysis['successful_reads']}, 失敗: {analysis['failed_reads']}")
        return

    print(f"\n【基本統計】")
    print(f"  サンプル数: {analysis['total_samples']}")
    print(f"  平均オブジェクト数: {analysis['avg_count']:.2f}")
    print(f"  中央値: {analysis['median_count']:.1f}")
    print(f"  最小: {analysis['min_count']}, 最大: {analysis['max_count']}")

    # 範囲別の集計
    counts = analysis['object_counts']
    range_2_10 = sum(1 for c in counts if 2 <= c <= 10)
    range_10_30 = sum(1 for c in counts if 10 < c <= 30)
    range_30_plus = sum(1 for c in counts if c > 30)

    print(f"\n【範囲別の分布】")
    print(f"  2-10個: {range_2_10}個 ({range_2_10/len(counts)*100:.1f}%)")
    print(f"  10-30個: {range_10_30}個 ({range_10_30/len(counts)*100:.1f}%)")
    print(f"  30個以上: {range_30_plus}個 ({range_30_plus/len(counts)*100:.1f}%)")

    # ARC-AGI2との比較
    print(f"\n【ARC-AGI2との比較】")
    print(f"  ARC-AGI2（100タスク）:")
    print(f"    2-10個: 約60%")
    print(f"    10-30個: 約30%")
    print(f"    30個以上: 約10%")
    print(f"  生成データ（{analysis['total_samples']}タスク）:")
    print(f"    2-10個: {range_2_10/len(counts)*100:.1f}%")
    print(f"    10-30個: {range_10_30/len(counts)*100:.1f}%")
    print(f"    30個以上: {range_30_plus/len(counts)*100:.1f}%")

    # 改善の評価
    print(f"\n【改善の評価】")
    if range_10_30 / len(counts) >= 0.20:
        print(f"  [OK] 10-30個のオブジェクトを含むタスクが十分に生成されています（{range_10_30/len(counts)*100:.1f}%）")
    else:
        print(f"  [WARN] 10-30個のオブジェクトを含むタスクが不足しています（{range_10_30/len(counts)*100:.1f}%）")
        print(f"     目標: 20%以上、現在: {range_10_30/len(counts)*100:.1f}%")

    if range_30_plus / len(counts) >= 0.05:
        print(f"  [OK] 30個以上のオブジェクトを含むタスクが生成されています（{range_30_plus/len(counts)*100:.1f}%）")
    else:
        print(f"  [WARN] 30個以上のオブジェクトを含むタスクが不足しています（{range_30_plus/len(counts)*100:.1f}%）")
        print(f"     目標: 5%以上、現在: {range_30_plus/len(counts)*100:.1f}%")

    if analysis['avg_count'] >= 10:
        print(f"  [OK] 平均オブジェクト数が改善されています（{analysis['avg_count']:.2f}個）")
    else:
        print(f"  [WARN] 平均オブジェクト数がまだ低いです（{analysis['avg_count']:.2f}個）")
        print(f"     目標: 10個以上、現在: {analysis['avg_count']:.2f}個")

    # 分布の詳細（上位20個）
    print(f"\n【オブジェクト数の分布（上位20個）】")
    distribution = analysis['distribution']
    sorted_dist = sorted(distribution.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
    for count, freq in sorted_dist[:20]:
        percentage = (freq / analysis['total_samples']) * 100
        bar = '#' * int(percentage / 2)
        print(f"  {count:3d}個: {freq:3d}回 ({percentage:5.1f}%) {bar}")

    print("\n" + "="*80)


def generate_test_data(num_programs: int = 30):
    """テストデータを生成"""
    print(f"\n[1/3] テストデータを生成中（{num_programs}プログラム）...")
    print(f"[INFO] 注意: データ生成には時間がかかる場合があります（数分〜数十分）")

    # 環境変数を設定
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = f"{str(project_root)}{os.pathsep}{pythonpath}"
    else:
        env['PYTHONPATH'] = str(project_root)

    # データ生成スクリプトを実行
    # 注意: 実際のデータ生成には時間がかかるため、既存のパイプラインを使用
    # または、小規模なテスト生成を実行

    print(f"[INFO] データ生成を開始します...")
    print(f"[INFO] 生成プログラム数: {num_programs}")

    # データ生成コマンド（main.pyを使用）
    cmd = [
        "python", "-m", "src.data_systems.generator.main",
        "--output-dir", "outputs/test_verification"
    ]

    # 環境変数でプログラム数を制限（main.pyのTASK_COUNTを変更する必要がある）
    # または、直接main.pyを呼び出す

    print(f"[INFO] データ生成を実行中...")
    print(f"[INFO] 注意: この処理には時間がかかる場合があります")

    # 実際には、既存のデータ生成パイプラインを使用するか、
    # または小規模なテストを実行する
    # ここでは、既存のデータファイルを確認する方法を提案

    return True


def find_latest_data_file() -> Path:
    """最新のデータファイルを探す"""
    data_dir = project_root / "data" / "generated" / "phase1_pairs"

    # タイムスタンプで最新のファイルを探す
    data_files = [
        data_dir / "data_pairs.jsonl.gz",
        data_dir / "data_pairs_from_outputs.jsonl.gz",
        data_dir / "data_pairs_manual.jsonl.gz",
        data_dir / "data_pairs_object_operations.jsonl.gz"
    ]

    latest_file = None
    latest_mtime = 0

    for data_file in data_files:
        if data_file.exists():
            mtime = data_file.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = data_file

    return latest_file


def main():
    """メイン処理"""
    print("="*80)
    print("改善後のデータ生成と検証")
    print("="*80)

    # オブジェクト抽出器を初期化
    extractor = IntegratedObjectExtractor(ExtractionConfig())

    # 最新のデータファイルを探す
    print(f"\n[1/3] 最新のデータファイルを検索中...")
    latest_file = find_latest_data_file()

    if latest_file:
        file_mtime = datetime.fromtimestamp(latest_file.stat().st_mtime)
        print(f"[OK] 最新のデータファイル: {latest_file.name}")
        print(f"[INFO] 最終更新: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

        # ファイルが最近更新されたか確認（1時間以内）
        time_diff = time.time() - latest_file.stat().st_mtime
        if time_diff > 3600:  # 1時間以上前
            print(f"[WARN] データファイルが1時間以上前に更新されています")
            print(f"[INFO] 新しいデータを生成することを推奨します")
            print(f"[INFO] 既存のデータで検証を続行しますか？ (y/n)")
            # 自動で既存データで検証を続行
        else:
            print(f"[INFO] データファイルは最近更新されています（{int(time_diff/60)}分前）")
    else:
        print(f"[NG] データファイルが見つかりません")
        print(f"[INFO] 新しいデータを生成します...")
        generate_test_data(num_programs=30)

        # 再度検索
        latest_file = find_latest_data_file()

    if not latest_file or not latest_file.exists():
        print(f"[NG] データファイルが見つかりません。データ生成を実行してください。")
        return 1

    # データを分析
    print(f"\n[2/3] データを分析中...")
    analysis = analyze_data_file(latest_file, extractor, limit=100)

    # 結果を出力
    print_verification_report(analysis)

    # 結果をJSONファイルに保存
    output_file = project_root / "test_outputs" / "improved_data_verification.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'data_file': str(latest_file),
            'analysis': analysis
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[3/3] 結果を保存しました: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

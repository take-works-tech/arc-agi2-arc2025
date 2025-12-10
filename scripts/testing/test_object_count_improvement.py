"""
オブジェクト数改善の検証スクリプト

改善後のデータ生成を実行し、オブジェクト数の分布を確認
"""
import sys
import json
import gzip
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
import subprocess
import os

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.data_systems.data_models.base import ObjectType


def extract_objects_from_grid(grid: List[List[int]], extractor) -> List[Any]:
    """グリッドからオブジェクトを抽出"""
    grid_np = np.array(grid)
    result = extractor.extract_objects_by_type(grid_np, input_image_index=0)

    if not result.success:
        return []

    objects = result.objects_by_type.get(ObjectType.SINGLE_COLOR_4WAY, [])
    return objects


def analyze_generated_data(data_file: Path, extractor, limit: int = 50) -> Dict[str, Any]:
    """生成されたデータを分析"""
    object_counts = []

    if not data_file.exists():
        print(f"[NG] データファイルが見つかりません: {data_file}")
        return {'object_counts': [], 'distribution': {}}

    print(f"[INFO] データファイルを読み込み中: {data_file}")

    try:
        with gzip.open(data_file, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                try:
                    pair = json.loads(line)

                    # DataPairの構造に合わせてフィールド名を確認
                    if 'input' in pair:
                        input_grid = pair['input']
                    elif 'input_grid' in pair:
                        input_grid = pair['input_grid']
                    else:
                        continue

                    objects = extract_objects_from_grid(input_grid, extractor)
                    object_counts.append(len(objects))
                except Exception as e:
                    print(f"[WARN] データ読み込みエラー (line {i}): {e}")
                    continue
    except Exception as e:
        print(f"[NG] ファイル読み込みエラー: {e}")
        return {'object_counts': [], 'distribution': {}}

    distribution = Counter(object_counts)

    return {
        'object_counts': object_counts,
        'distribution': dict(distribution),
        'total_samples': len(object_counts),
        'avg_count': np.mean(object_counts) if object_counts else 0,
        'min_count': min(object_counts) if object_counts else 0,
        'max_count': max(object_counts) if object_counts else 0
    }


def print_analysis_report(analysis: Dict[str, Any]):
    """分析結果を出力"""
    print("\n" + "="*80)
    print("オブジェクト数改善の検証結果")
    print("="*80)

    if not analysis['object_counts']:
        print("[NG] データが読み込めませんでした")
        return

    print(f"\n【基本統計】")
    print(f"  サンプル数: {analysis['total_samples']}")
    print(f"  平均オブジェクト数: {analysis['avg_count']:.2f}")
    print(f"  最小オブジェクト数: {analysis['min_count']}")
    print(f"  最大オブジェクト数: {analysis['max_count']}")

    print(f"\n【オブジェクト数の分布】")
    distribution = analysis['distribution']
    sorted_dist = sorted(distribution.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)

    for count, freq in sorted_dist[:30]:  # 最初の30個を表示
        percentage = (freq / analysis['total_samples']) * 100
        print(f"  {count}個: {freq}回 ({percentage:.1f}%)")

    # 範囲別の集計
    print(f"\n【範囲別の集計】")
    counts = analysis['object_counts']

    range_2_10 = sum(1 for c in counts if 2 <= c <= 10)
    range_10_30 = sum(1 for c in counts if 10 < c <= 30)
    range_30_plus = sum(1 for c in counts if c > 30)

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
    if range_10_30 / len(counts) >= 0.20:  # 20%以上
        print(f"  ✅ 10-30個のオブジェクトを含むタスクが十分に生成されています（{range_10_30/len(counts)*100:.1f}%）")
    else:
        print(f"  ⚠️  10-30個のオブジェクトを含むタスクが不足しています（{range_10_30/len(counts)*100:.1f}%）")

    if range_30_plus / len(counts) >= 0.05:  # 5%以上
        print(f"  ✅ 30個以上のオブジェクトを含むタスクが生成されています（{range_30_plus/len(counts)*100:.1f}%）")
    else:
        print(f"  ⚠️  30個以上のオブジェクトを含むタスクが不足しています（{range_30_plus/len(counts)*100:.1f}%）")

    if analysis['avg_count'] >= 10:
        print(f"  ✅ 平均オブジェクト数が改善されています（{analysis['avg_count']:.2f}個）")
    else:
        print(f"  ⚠️  平均オブジェクト数がまだ低いです（{analysis['avg_count']:.2f}個）")

    print("\n" + "="*80)


def generate_test_data(num_tasks: int = 20):
    """テストデータを生成"""
    print(f"\n[1/2] テストデータを生成中（{num_tasks}タスク）...")

    # 設定ファイルを一時的に変更して小規模データを生成
    # または、直接スクリプトを実行

    # データ生成スクリプトを実行
    cmd = [
        "python", "scripts/production/data_generation/generate_data.py"
    ]

    # 環境変数でPYTHONPATHを設定
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = f"{str(project_root)}{os.pathsep}{pythonpath}"
    else:
        env['PYTHONPATH'] = str(project_root)

    # データ生成を実行（バックグラウンドで実行し、完了を待つ）
    # 注意: 実際のデータ生成には時間がかかるため、既存のデータを使用することも可能
    print(f"[INFO] データ生成を実行中...")
    print(f"[INFO] 注意: データ生成には時間がかかる場合があります")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=300  # 5分のタイムアウト
        )

        if result.returncode == 0:
            print(f"[OK] データ生成が完了しました")
            return True
        else:
            print(f"[WARN] データ生成でエラーが発生しました（終了コード: {result.returncode}）")
            print(f"[INFO] 既存のデータを使用して検証を続行します")
            return False
    except subprocess.TimeoutExpired:
        print(f"[WARN] データ生成がタイムアウトしました")
        print(f"[INFO] 既存のデータを使用して検証を続行します")
        return False
    except Exception as e:
        print(f"[WARN] データ生成でエラーが発生しました: {e}")
        print(f"[INFO] 既存のデータを使用して検証を続行します")
        return False


def main():
    """メイン処理"""
    print("="*80)
    print("オブジェクト数改善の検証")
    print("="*80)

    # オブジェクト抽出器を初期化
    extractor = IntegratedObjectExtractor(ExtractionConfig())

    # データファイルのパス
    data_files = [
        project_root / "data" / "generated" / "phase1_pairs" / "data_pairs.jsonl.gz",
        project_root / "data" / "generated" / "phase1_pairs" / "data_pairs_from_outputs.jsonl.gz",
        project_root / "data" / "generated" / "phase1_pairs" / "data_pairs_manual.jsonl.gz"
    ]

    # 最新のデータファイルを探す
    latest_file = None
    latest_mtime = 0

    for data_file in data_files:
        if data_file.exists():
            mtime = data_file.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_file = data_file

    if latest_file:
        print(f"\n[2/2] 既存のデータを分析中: {latest_file}")
        print(f"[INFO] 最終更新: {latest_file.stat().st_mtime}")
    else:
        print(f"\n[INFO] 既存のデータファイルが見つかりません")
        print(f"[INFO] 新しいデータを生成しますか？ (y/n)")
        # 自動でデータ生成を試行
        generate_test_data(num_tasks=20)

        # 再度データファイルを探す
        for data_file in data_files:
            if data_file.exists():
                mtime = data_file.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_file = data_file

    if not latest_file or not latest_file.exists():
        print(f"[NG] データファイルが見つかりません。データ生成を実行してください。")
        return 1

    # データを分析
    analysis = analyze_generated_data(latest_file, extractor, limit=100)

    # 結果を出力
    print_analysis_report(analysis)

    # 結果をJSONファイルに保存
    output_file = project_root / "test_outputs" / "object_count_verification.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 結果を保存しました: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
既存データを使用したオブジェクト数改善の検証スクリプト

既存のデータファイルから最新のデータを読み込み、オブジェクト数の分布を確認
改善前後の比較が可能なように、複数のデータファイルを分析
"""
import sys
import json
import gzip
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from datetime import datetime

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

                    # DataPairの構造に合わせてフィールド名を確認
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
                except json.JSONDecodeError as e:
                    failed_reads += 1
                    continue
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


def print_analysis_report(analyses: List[Dict[str, Any]]):
    """分析結果を出力"""
    print("\n" + "="*80)
    print("オブジェクト数改善の検証結果（既存データ分析）")
    print("="*80)

    for analysis in analyses:
        if not analysis.get('exists', False):
            print(f"\n[NG] ファイルが見つかりません: {analysis['file_path']}")
            continue

        if analysis.get('error'):
            print(f"\n[NG] エラー: {analysis['file_path']} - {analysis['error']}")
            continue

        if not analysis['object_counts']:
            print(f"\n[WARN] データが読み込めませんでした: {analysis['file_path']}")
            print(f"  成功: {analysis['successful_reads']}, 失敗: {analysis['failed_reads']}")
            continue

        file_name = Path(analysis['file_path']).name
        print(f"\n【{file_name}】")
        print(f"  サンプル数: {analysis['total_samples']}")
        print(f"  平均オブジェクト数: {analysis['avg_count']:.2f}")
        print(f"  中央値: {analysis['median_count']:.1f}")
        print(f"  最小: {analysis['min_count']}, 最大: {analysis['max_count']}")

        # 範囲別の集計
        counts = analysis['object_counts']
        range_2_10 = sum(1 for c in counts if 2 <= c <= 10)
        range_10_30 = sum(1 for c in counts if 10 < c <= 30)
        range_30_plus = sum(1 for c in counts if c > 30)

        print(f"\n  範囲別の分布:")
        print(f"    2-10個: {range_2_10}個 ({range_2_10/len(counts)*100:.1f}%)")
        print(f"    10-30個: {range_10_30}個 ({range_10_30/len(counts)*100:.1f}%)")
        print(f"    30個以上: {range_30_plus}個 ({range_30_plus/len(counts)*100:.1f}%)")

        # 分布の詳細（上位20個）
        print(f"\n  オブジェクト数の分布（上位20個）:")
        distribution = analysis['distribution']
        sorted_dist = sorted(distribution.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
        for count, freq in sorted_dist[:20]:
            percentage = (freq / analysis['total_samples']) * 100
            bar = '#' * int(percentage / 2)
            print(f"    {count:3d}個: {freq:3d}回 ({percentage:5.1f}%) {bar}")

    # 全体のサマリー
    print(f"\n【全体サマリー】")
    all_counts = []
    for analysis in analyses:
        if analysis.get('exists') and analysis.get('object_counts'):
            all_counts.extend(analysis['object_counts'])

    if all_counts:
        range_2_10 = sum(1 for c in all_counts if 2 <= c <= 10)
        range_10_30 = sum(1 for c in all_counts if 10 < c <= 30)
        range_30_plus = sum(1 for c in all_counts if c > 30)

        print(f"  総サンプル数: {len(all_counts)}")
        print(f"  平均オブジェクト数: {np.mean(all_counts):.2f}")
        print(f"  中央値: {np.median(all_counts):.1f}")
        print(f"  最小: {min(all_counts)}, 最大: {max(all_counts)}")
        print(f"\n  範囲別の分布:")
        print(f"    2-10個: {range_2_10}個 ({range_2_10/len(all_counts)*100:.1f}%)")
        print(f"    10-30個: {range_10_30}個 ({range_10_30/len(all_counts)*100:.1f}%)")
        print(f"    30個以上: {range_30_plus}個 ({range_30_plus/len(all_counts)*100:.1f}%)")

        # ARC-AGI2との比較
        print(f"\n【ARC-AGI2との比較】")
        print(f"  ARC-AGI2（100タスク）:")
        print(f"    2-10個: 約60%")
        print(f"    10-30個: 約30%")
        print(f"    30個以上: 約10%")
        print(f"  生成データ（{len(all_counts)}タスク）:")
        print(f"    2-10個: {range_2_10/len(all_counts)*100:.1f}%")
        print(f"    10-30個: {range_10_30/len(all_counts)*100:.1f}%")
        print(f"    30個以上: {range_30_plus/len(all_counts)*100:.1f}%")

        # 改善の評価
        print(f"\n【改善の評価】")
        if range_10_30 / len(all_counts) >= 0.20:
            print(f"  [OK] 10-30個のオブジェクトを含むタスクが十分に生成されています（{range_10_30/len(all_counts)*100:.1f}%）")
        else:
            print(f"  [WARN] 10-30個のオブジェクトを含むタスクが不足しています（{range_10_30/len(all_counts)*100:.1f}%）")
            print(f"     目標: 20%以上、現在: {range_10_30/len(all_counts)*100:.1f}%")

        if range_30_plus / len(all_counts) >= 0.05:
            print(f"  [OK] 30個以上のオブジェクトを含むタスクが生成されています（{range_30_plus/len(all_counts)*100:.1f}%）")
        else:
            print(f"  [WARN] 30個以上のオブジェクトを含むタスクが不足しています（{range_30_plus/len(all_counts)*100:.1f}%）")
            print(f"     目標: 5%以上、現在: {range_30_plus/len(all_counts)*100:.1f}%")

        if np.mean(all_counts) >= 10:
            print(f"  [OK] 平均オブジェクト数が改善されています（{np.mean(all_counts):.2f}個）")
        else:
            print(f"  [WARN] 平均オブジェクト数がまだ低いです（{np.mean(all_counts):.2f}個）")
            print(f"     目標: 10個以上、現在: {np.mean(all_counts):.2f}個")

    print("\n" + "="*80)


def main():
    """メイン処理"""
    print("="*80)
    print("オブジェクト数改善の検証（既存データ分析）")
    print("="*80)

    # オブジェクト抽出器を初期化
    extractor = IntegratedObjectExtractor(ExtractionConfig())

    # データファイルのパス
    data_dir = project_root / "data" / "generated" / "phase1_pairs"
    data_files = [
        data_dir / "data_pairs.jsonl.gz",
        data_dir / "data_pairs_from_outputs.jsonl.gz",
        data_dir / "data_pairs_manual.jsonl.gz",
        data_dir / "data_pairs_object_operations.jsonl.gz"
    ]

    print(f"\n[1/2] データファイルを検索中...")
    existing_files = [f for f in data_files if f.exists()]

    if not existing_files:
        print(f"[NG] データファイルが見つかりません")
        print(f"[INFO] データ生成を実行してください")
        return 1

    print(f"[OK] {len(existing_files)}個のデータファイルが見つかりました")

    # 各ファイルを分析
    print(f"\n[2/2] データを分析中...")
    analyses = []
    for data_file in existing_files:
        analysis = analyze_data_file(data_file, extractor, limit=100)
        analyses.append(analysis)

    # 結果を出力
    print_analysis_report(analyses)

    # 結果をJSONファイルに保存
    output_file = project_root / "test_outputs" / "object_count_verification.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'analyses': analyses
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 結果を保存しました: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

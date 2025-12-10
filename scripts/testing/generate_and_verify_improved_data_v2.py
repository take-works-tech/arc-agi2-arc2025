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
import yaml
import shutil

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.data_systems.data_models.base import ObjectType


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
    successful_reads = 0
    failed_reads = 0

    if not data_file.exists():
        return None

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
                except Exception:
                    failed_reads += 1
                    continue
    except Exception as e:
        print(f"[NG] ファイル読み込みエラー: {e}")
        return None

    if not object_counts:
        return None

    return {
        'object_counts': object_counts,
        'total_samples': len(object_counts),
        'successful_reads': successful_reads,
        'failed_reads': failed_reads,
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
    if range_10_30 / len(counts) >= 0.25:
        print(f"  [OK] 10-30個のオブジェクトを含むタスクが十分に生成されています（{range_10_30/len(counts)*100:.1f}%）")
    elif range_10_30 / len(counts) >= 0.20:
        print(f"  [OK] 10-30個のオブジェクトを含むタスクが生成されています（{range_10_30/len(counts)*100:.1f}%）")
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

    # 分布の詳細
    distribution = Counter(counts)
    print(f"\n【オブジェクト数の分布（上位20個）】")
    sorted_dist = sorted(distribution.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
    for count, freq in sorted_dist[:20]:
        percentage = (freq / analysis['total_samples']) * 100
        bar = '#' * int(percentage / 2)
        print(f"  {count:3d}個: {freq:3d}回 ({percentage:5.1f}%) {bar}")

    print("\n" + "="*80)


def generate_test_data(num_programs: int = 50):
    """テストデータを生成"""
    print(f"\n[1/3] テストデータを生成中（{num_programs}プログラム）...")
    print(f"[INFO] 改善後のオブジェクト数決定ロジックを使用します")
    print(f"[INFO] 注意: データ生成には時間がかかる場合があります（10-20分）")

    # 環境変数を設定
    env = os.environ.copy()
    pythonpath = env.get('PYTHONPATH', '')
    if pythonpath:
        env['PYTHONPATH'] = f"{str(project_root)}{os.pathsep}{pythonpath}"
    else:
        env['PYTHONPATH'] = str(project_root)

    # 設定ファイルのバックアップと一時的な変更
    config_path = project_root / "configs" / "default_config.yaml"
    backup_path = project_root / "configs" / "default_config.yaml.backup_v2"

    if backup_path.exists():
        shutil.copy(backup_path, config_path)
    else:
        shutil.copy(config_path, backup_path)

    # 設定を読み込み
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 小規模データ生成用に設定を変更
    original_num_programs = config['data']['generation']['num_programs']
    config['data']['generation']['num_programs'] = num_programs

    # 設定を保存
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"[INFO] 設定を変更しました: {original_num_programs} → {num_programs}プログラム")
    print(f"[INFO] データ生成を実行中...\n")

    try:
        # データ生成スクリプトを実行
        result = subprocess.run(
            ["python", "scripts/production/data_generation/generate_data.py"],
            cwd=str(project_root),
            env=env,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=1800  # 30分のタイムアウト
        )

        if result.returncode == 0:
            print(f"\n[OK] データ生成が完了しました")
            return True
        else:
            print(f"\n[WARN] データ生成でエラーが発生しました（終了コード: {result.returncode}）")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n[WARN] データ生成がタイムアウトしました")
        return False
    except Exception as e:
        print(f"\n[WARN] データ生成でエラーが発生しました: {e}")
        return False
    finally:
        # 設定を復元
        if backup_path.exists():
            shutil.copy(backup_path, config_path)
            print(f"[INFO] 設定を復元しました")


def find_latest_data_file() -> Path:
    """最新のデータファイルを探す"""
    data_dir = project_root / "data" / "generated" / "phase1_pairs"
    data_file = data_dir / "data_pairs.jsonl.gz"
    return data_file


def main():
    """メイン処理"""
    print("="*80)
    print("改善後のロジックでデータ生成と検証")
    print("="*80)

    # データ生成
    success = generate_test_data(num_programs=50)

    if not success:
        print(f"\n[WARN] データ生成が完了しませんでした")
        print(f"[INFO] 既存のデータで検証を続行します")

    # 最新のデータファイルを探す
    print(f"\n[2/3] 最新のデータファイルを検索中...")
    data_file = find_latest_data_file()

    if not data_file.exists():
        print(f"[NG] データファイルが見つかりません: {data_file}")
        return 1

    file_mtime = datetime.fromtimestamp(data_file.stat().st_mtime)
    print(f"[OK] データファイル: {data_file.name}")
    print(f"[INFO] 最終更新: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # データを分析
    print(f"\n[3/3] データを分析中...")
    extractor = IntegratedObjectExtractor(ExtractionConfig())
    analysis = analyze_data_file(data_file, extractor, limit=100)

    if not analysis:
        print(f"[NG] データが読み込めませんでした")
        return 1

    # 結果を出力
    print_verification_result(analysis)

    # 結果を保存
    output_file = project_root / "test_outputs" / "improved_data_verification_v2.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'data_file': str(data_file),
            'analysis': analysis
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 結果を保存しました: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

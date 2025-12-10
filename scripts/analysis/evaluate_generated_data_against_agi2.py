"""
生成されたデータセットをARC-AGI2と比較評価するスクリプト

評価項目:
1. インプットグリッドの評価
   - グリッドサイズ分布
   - 色数分布
   - オブジェクト数分布
   - オブジェクトサイズ分布
   - 背景色分布

2. プログラムの評価
   - プログラム長分布
   - ノード数分布
   - 複雑度分布
   - 使用コマンド分布
   - ループの深さ

3. 生成時間の評価
   - タスクあたりの平均生成時間
   - 各フェーズの時間分布
   - スループット

4. 学習データとしての品質評価
   - 多様性
   - 分布の類似性（KLダイバージェンスなど）
"""

import json
import sys
import gzip
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core_systems.extractor.object_extractor import IntegratedObjectExtractor
from src.data_systems.config.config import ExtractionConfig
from src.data_systems.data_models.base import ObjectType


def load_arc_agi2_tasks(max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
    """ARC-AGI2の訓練データを読み込む"""
    possible_paths = [
        project_root / "data" / "arc-agi_training_challenges.json",
        project_root / "data" / "core_arc_agi2" / "arc-agi_training_challenges.json",
        project_root / "data" / "arc-agi2" / "arc-agi_training_challenges.json",
    ]

    for task_file in possible_paths:
        if task_file.exists():
            print(f"ARC-AGI2タスクファイルを読み込み: {task_file}", flush=True)
            with open(task_file, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)

            tasks = []
            for task_id, task_data in tasks_data.items():
                if max_tasks and len(tasks) >= max_tasks:
                    break
                tasks.append({
                    'task_id': task_id,
                    'train': task_data.get('train', []),
                    'test': task_data.get('test', [])
                })
            return tasks

    print("ARC-AGI2タスクファイルが見つかりませんでした", flush=True)
    return []


def load_generated_data(output_dir: Path) -> Tuple[List[Dict], Dict]:
    """生成されたデータを読み込む"""
    # JSONLファイルを読み込み
    jsonl_path = output_dir / "hybrid" / "phase1_pairs" / "data_pairs.jsonl.gz"
    if not jsonl_path.exists():
        jsonl_path = output_dir / "hybrid" / "phase1_pairs" / "data_pairs.jsonl"

    pairs = []
    if jsonl_path.exists():
        print(f"生成データを読み込み中: {jsonl_path}", flush=True)
        open_func = gzip.open if jsonl_path.suffix == '.gz' else open
        with open_func(jsonl_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
        print(f"  {len(pairs)}個のペアを読み込みました", flush=True)
    else:
        print(f"警告: データファイルが見つかりません: {jsonl_path}", flush=True)

    # 実行時間ログを読み込み
    time_log_path = output_dir / "execution_time_log.json"
    time_stats = {}
    if time_log_path.exists():
        with open(time_log_path, 'r', encoding='utf-8') as f:
            time_stats = json.load(f)

    return pairs, time_stats


def analyze_input_grids(tasks: List[Dict], is_agi2: bool = True) -> Dict[str, Any]:
    """インプットグリッドを分析"""
    extractor = IntegratedObjectExtractor(config=ExtractionConfig())

    widths = []
    heights = []
    areas = []
    color_counts = []
    object_counts = []
    object_sizes = []
    background_colors = []

    print(f"インプットグリッドを分析中... ({len(tasks)}タスク)", flush=True)

    for task_idx, task in enumerate(tasks):
        if task_idx % 100 == 0:
            print(f"  進捗: {task_idx}/{len(tasks)}", flush=True)

        train_pairs = task.get('train', [])
        if not train_pairs:
            continue

        # 最初のペアのインプットグリッドを使用
        input_grid_data = train_pairs[0].get('input', [])
        if not input_grid_data:
            continue

        grid = np.array(input_grid_data, dtype=int)
        h, w = grid.shape
        widths.append(w)
        heights.append(h)
        areas.append(w * h)

        # 色数
        unique_colors = np.unique(grid)
        color_counts.append(len(unique_colors))

        # 背景色（最頻値）
        bg_color = np.bincount(grid.flatten()).argmax()
        background_colors.append(int(bg_color))

        # オブジェクト抽出
        try:
            objects = extractor.extract_objects_by_type(grid, object_type=ObjectType.CONNECTED_COMPONENT)
            object_counts.append(len(objects))

            # オブジェクトサイズ
            for obj in objects:
                if 'pixels' in obj:
                    object_sizes.append(len(obj['pixels']))
        except Exception as e:
            # エラーが発生した場合はスキップ
            continue

    stats = {
        'count': len(widths),
        'width': {
            'mean': float(np.mean(widths)) if widths else 0.0,
            'std': float(np.std(widths)) if widths else 0.0,
            'min': int(np.min(widths)) if widths else 0,
            'max': int(np.max(widths)) if widths else 0,
            'median': float(np.median(widths)) if widths else 0.0,
            'distribution': dict(Counter(widths))
        },
        'height': {
            'mean': float(np.mean(heights)) if heights else 0.0,
            'std': float(np.std(heights)) if heights else 0.0,
            'min': int(np.min(heights)) if heights else 0,
            'max': int(np.max(heights)) if heights else 0,
            'median': float(np.median(heights)) if heights else 0.0,
            'distribution': dict(Counter(heights))
        },
        'area': {
            'mean': float(np.mean(areas)) if areas else 0.0,
            'std': float(np.std(areas)) if areas else 0.0,
            'min': int(np.min(areas)) if areas else 0,
            'max': int(np.max(areas)) if areas else 0,
            'median': float(np.median(areas)) if areas else 0.0,
        },
        'color_count': {
            'mean': float(np.mean(color_counts)) if color_counts else 0.0,
            'std': float(np.std(color_counts)) if color_counts else 0.0,
            'min': int(np.min(color_counts)) if color_counts else 0,
            'max': int(np.max(color_counts)) if color_counts else 0,
            'median': float(np.median(color_counts)) if color_counts else 0.0,
            'distribution': dict(Counter(color_counts))
        },
        'object_count': {
            'mean': float(np.mean(object_counts)) if object_counts else 0.0,
            'std': float(np.std(object_counts)) if object_counts else 0.0,
            'min': int(np.min(object_counts)) if object_counts else 0,
            'max': int(np.max(object_counts)) if object_counts else 0,
            'median': float(np.median(object_counts)) if object_counts else 0.0,
            'distribution': dict(Counter(object_counts))
        },
        'object_size': {
            'mean': float(np.mean(object_sizes)) if object_sizes else 0.0,
            'std': float(np.std(object_sizes)) if object_sizes else 0.0,
            'min': int(np.min(object_sizes)) if object_sizes else 0,
            'max': int(np.max(object_sizes)) if object_sizes else 0,
            'median': float(np.median(object_sizes)) if object_sizes else 0.0,
        },
        'background_color': {
            'distribution': dict(Counter(background_colors))
        }
    }

    return stats


def analyze_programs(pairs: List[Dict]) -> Dict[str, Any]:
    """プログラムを分析"""
    program_lengths = []
    node_counts = []
    complexities = []
    command_usage = Counter()

    print(f"プログラムを分析中... ({len(pairs)}ペア)", flush=True)

    for pair in pairs:
        program = pair.get('program', '')
        if not program:
            continue

        program_lengths.append(len(program))

        # 複雑度
        metadata = pair.get('metadata', {})
        complexity = metadata.get('complexity', None)
        if complexity is not None:
            complexities.append(complexity)

        # コマンド使用頻度（簡易版：キーワードマッチング）
        for cmd in ['GET_ALL_OBJECTS', 'FILTER', 'MOVE', 'COLOR_CHANGE', 'MERGE',
                   'CONCAT', 'SORT', 'REVERSE', 'FOR', 'WHILE', 'IF', 'DEFINE']:
            if cmd in program:
                command_usage[cmd] += 1

    stats = {
        'count': len(program_lengths),
        'program_length': {
            'mean': float(np.mean(program_lengths)) if program_lengths else 0.0,
            'std': float(np.std(program_lengths)) if program_lengths else 0.0,
            'min': int(np.min(program_lengths)) if program_lengths else 0,
            'max': int(np.max(program_lengths)) if program_lengths else 0,
            'median': float(np.median(program_lengths)) if program_lengths else 0.0,
        },
        'complexity': {
            'mean': float(np.mean(complexities)) if complexities else 0.0,
            'std': float(np.std(complexities)) if complexities else 0.0,
            'min': int(np.min(complexities)) if complexities else 0,
            'max': int(np.max(complexities)) if complexities else 0,
            'median': float(np.median(complexities)) if complexities else 0.0,
            'distribution': dict(Counter(complexities))
        },
        'command_usage': dict(command_usage)
    }

    return stats


def analyze_generation_time(time_stats: Dict) -> Dict[str, Any]:
    """生成時間を分析"""
    if not time_stats:
        return {}

    timing_stats = time_stats.get('timing_statistics', {})
    if not timing_stats:
        return {}

    # 各ステップの時間を集計
    step_times = defaultdict(list)
    for step_name, step_data in timing_stats.items():
        if isinstance(step_data, dict) and 'times' in step_data:
            step_times[step_name] = step_data['times']

    stats = {}
    for step_name, times in step_times.items():
        if times:
            stats[step_name] = {
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
                'max': float(np.max(times)),
                'median': float(np.median(times)),
                'total': float(np.sum(times))
            }

    # 全体の実行時間
    overall_time = time_stats.get('overall_execution_time', 0.0)
    task_count = time_stats.get('task_count', 0)

    stats['overall'] = {
        'total_time': overall_time,
        'task_count': task_count,
        'avg_time_per_task': overall_time / task_count if task_count > 0 else 0.0,
        'throughput': task_count / overall_time if overall_time > 0 else 0.0  # タスク/秒
    }

    return stats


def calculate_kl_divergence(p: Dict[int, int], q: Dict[int, int]) -> float:
    """KLダイバージェンスを計算（離散分布）"""
    # 両方の分布のキーを統合
    all_keys = set(p.keys()) | set(q.keys())

    # 正規化
    p_sum = sum(p.values())
    q_sum = sum(q.values())

    if p_sum == 0 or q_sum == 0:
        return float('inf')

    kl = 0.0
    for key in all_keys:
        p_prob = p.get(key, 0) / p_sum
        q_prob = q.get(key, 0) / q_sum

        if p_prob > 0:
            if q_prob == 0:
                return float('inf')
            kl += p_prob * np.log(p_prob / q_prob)

    return kl


def compare_distributions(gen_stats: Dict, agi2_stats: Dict) -> Dict[str, Any]:
    """分布を比較"""
    comparison = {}

    # グリッドサイズ
    if 'width' in gen_stats and 'width' in agi2_stats:
        gen_width_dist = gen_stats['width'].get('distribution', {})
        agi2_width_dist = agi2_stats['width'].get('distribution', {})
        if gen_width_dist and agi2_width_dist:
            comparison['width_kl_divergence'] = calculate_kl_divergence(
                gen_width_dist, agi2_width_dist
            )

    if 'height' in gen_stats and 'height' in agi2_stats:
        gen_height_dist = gen_stats['height'].get('distribution', {})
        agi2_height_dist = agi2_stats['height'].get('distribution', {})
        if gen_height_dist and agi2_height_dist:
            comparison['height_kl_divergence'] = calculate_kl_divergence(
                gen_height_dist, agi2_height_dist
            )

    # 色数
    if 'color_count' in gen_stats and 'color_count' in agi2_stats:
        gen_color_dist = gen_stats['color_count'].get('distribution', {})
        agi2_color_dist = agi2_stats['color_count'].get('distribution', {})
        if gen_color_dist and agi2_color_dist:
            comparison['color_count_kl_divergence'] = calculate_kl_divergence(
                gen_color_dist, agi2_color_dist
            )

    # オブジェクト数
    if 'object_count' in gen_stats and 'object_count' in agi2_stats:
        gen_obj_dist = gen_stats['object_count'].get('distribution', {})
        agi2_obj_dist = agi2_stats['object_count'].get('distribution', {})
        if gen_obj_dist and agi2_obj_dist:
            comparison['object_count_kl_divergence'] = calculate_kl_divergence(
                gen_obj_dist, agi2_obj_dist
            )

    return comparison


def generate_comparison_report(gen_stats: Dict, agi2_stats: Dict,
                                gen_program_stats: Dict, time_stats: Dict,
                                comparison: Dict, output_dir: Path):
    """比較レポートを生成"""
    report_path = output_dir / "evaluation_report.md"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 生成データセット vs ARC-AGI2 評価レポート\n\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. インプットグリッドの評価\n\n")

        # グリッドサイズ
        f.write("### 1.1 グリッドサイズ\n\n")
        f.write("| 項目 | 生成データ | ARC-AGI2 | 差分 |\n")
        f.write("|------|-----------|----------|------|\n")

        for metric in ['mean', 'std', 'min', 'max', 'median']:
            gen_w = gen_stats['width'].get(metric, 0)
            agi2_w = agi2_stats['width'].get(metric, 0)
            diff = gen_w - agi2_w
            f.write(f"| 幅 {metric} | {gen_w:.2f} | {agi2_w:.2f} | {diff:+.2f} |\n")

        f.write("\n")

        for metric in ['mean', 'std', 'min', 'max', 'median']:
            gen_h = gen_stats['height'].get(metric, 0)
            agi2_h = agi2_stats['height'].get(metric, 0)
            diff = gen_h - agi2_h
            f.write(f"| 高さ {metric} | {gen_h:.2f} | {agi2_h:.2f} | {diff:+.2f} |\n")

        f.write("\n")

        # 色数・オブジェクト数
        f.write("### 1.2 色数とオブジェクト数\n\n")
        f.write("| 項目 | 生成データ | ARC-AGI2 | 差分 |\n")
        f.write("|------|-----------|----------|------|\n")

        for metric in ['mean', 'std', 'min', 'max', 'median']:
            gen_c = gen_stats['color_count'].get(metric, 0)
            agi2_c = agi2_stats['color_count'].get(metric, 0)
            diff = gen_c - agi2_c
            f.write(f"| 色数 {metric} | {gen_c:.2f} | {agi2_c:.2f} | {diff:+.2f} |\n")

        f.write("\n")

        for metric in ['mean', 'std', 'min', 'max', 'median']:
            gen_o = gen_stats['object_count'].get(metric, 0)
            agi2_o = agi2_stats['object_count'].get(metric, 0)
            diff = gen_o - agi2_o
            f.write(f"| オブジェクト数 {metric} | {gen_o:.2f} | {agi2_o:.2f} | {diff:+.2f} |\n")

        f.write("\n")

        # 分布の類似性
        f.write("### 1.3 分布の類似性（KLダイバージェンス）\n\n")
        f.write("| 項目 | KLダイバージェンス |\n")
        f.write("|------|-------------------|\n")
        for key, value in comparison.items():
            if isinstance(value, float):
                if value == float('inf'):
                    f.write(f"| {key} | ∞ (分布が完全に異なる) |\n")
                else:
                    f.write(f"| {key} | {value:.4f} |\n")

        f.write("\n")

        # プログラム評価
        f.write("## 2. プログラムの評価\n\n")
        f.write(f"### 2.1 基本統計\n\n")
        f.write(f"- プログラム数: {gen_program_stats['count']}\n")
        f.write(f"- 平均プログラム長: {gen_program_stats['program_length']['mean']:.2f}文字\n")
        f.write(f"- プログラム長範囲: {gen_program_stats['program_length']['min']} - {gen_program_stats['program_length']['max']}文字\n")

        if 'complexity' in gen_program_stats:
            f.write(f"- 平均複雑度: {gen_program_stats['complexity']['mean']:.2f}\n")
            f.write(f"- 複雑度範囲: {gen_program_stats['complexity']['min']} - {gen_program_stats['complexity']['max']}\n")

        f.write("\n### 2.2 コマンド使用頻度\n\n")
        f.write("| コマンド | 使用回数 |\n")
        f.write("|---------|----------|\n")
        for cmd, count in sorted(gen_program_stats.get('command_usage', {}).items(),
                                 key=lambda x: x[1], reverse=True):
            f.write(f"| {cmd} | {count} |\n")

        f.write("\n")

        # 生成時間評価
        f.write("## 3. 生成時間の評価\n\n")
        if time_stats and 'overall' in time_stats:
            overall = time_stats['overall']
            f.write(f"- 総実行時間: {overall['total_time']:.2f}秒\n")
            f.write(f"- タスク数: {overall['task_count']}\n")
            f.write(f"- タスクあたりの平均時間: {overall['avg_time_per_task']:.3f}秒\n")
            f.write(f"- スループット: {overall['throughput']:.2f}タスク/秒\n")

        f.write("\n### 3.1 各フェーズの時間分布\n\n")
        f.write("| フェーズ | 平均時間(秒) | 最大時間(秒) | 合計時間(秒) |\n")
        f.write("|---------|-------------|-------------|-------------|\n")
        for phase, stats in time_stats.items():
            if phase != 'overall' and isinstance(stats, dict):
                f.write(f"| {phase} | {stats.get('mean', 0):.4f} | {stats.get('max', 0):.4f} | {stats.get('total', 0):.4f} |\n")

        f.write("\n")

        # 総合評価
        f.write("## 4. 総合評価\n\n")
        f.write("### 4.1 学習データとしての品質\n\n")
        f.write("- **多様性**: 生成データの分布がARC-AGI2とどの程度類似しているか\n")
        f.write("- **カバレッジ**: 様々なグリッドサイズ、色数、オブジェクト数のカバー状況\n")
        f.write("- **実用性**: 生成時間とスループット\n\n")

        # 評価サマリー
        f.write("### 4.2 評価サマリー\n\n")

        # KLダイバージェンスが小さいほど良い（< 0.5: 非常に類似, < 1.0: 類似, < 2.0: やや異なる, >= 2.0: 大きく異なる）
        good_kl_threshold = 1.0
        moderate_kl_threshold = 2.0

        f.write("**分布の類似性**:\n")
        for key, value in comparison.items():
            if isinstance(value, float):
                if value == float('inf'):
                    f.write(f"- {key}: ⚠️ 分布が完全に異なる\n")
                elif value < good_kl_threshold:
                    f.write(f"- {key}: ✅ 非常によく類似 ({value:.4f})\n")
                elif value < moderate_kl_threshold:
                    f.write(f"- {key}: ⚠️ やや異なる ({value:.4f})\n")
                else:
                    f.write(f"- {key}: ❌ 大きく異なる ({value:.4f})\n")

        f.write("\n**生成効率**:\n")
        if time_stats and 'overall' in time_stats:
            throughput = time_stats['overall']['throughput']
            if throughput > 10:
                f.write(f"- ✅ 高速 ({throughput:.2f}タスク/秒)\n")
            elif throughput > 5:
                f.write(f"- ⚠️ やや遅い ({throughput:.2f}タスク/秒)\n")
            else:
                f.write(f"- ❌ 遅い ({throughput:.2f}タスク/秒)\n")

    print(f"評価レポートを保存: {report_path}", flush=True)


def main():
    """メイン関数"""
    # 最新の出力ディレクトリを検索
    output_base = project_root / "outputs"
    if not output_base.exists():
        print("出力ディレクトリが見つかりません", flush=True)
        return

    # 最新のディレクトリを取得
    output_dirs = sorted([d for d in output_base.iterdir() if d.is_dir()],
                        key=lambda x: x.name, reverse=True)
    if not output_dirs:
        print("出力ディレクトリが見つかりません", flush=True)
        return

    output_dir = output_dirs[0]
    print(f"最新の出力ディレクトリを使用: {output_dir}", flush=True)

    # ARC-AGI2データを読み込み
    print("\n=== ARC-AGI2データの分析 ===", flush=True)
    agi2_tasks = load_arc_agi2_tasks(max_tasks=1000)
    if not agi2_tasks:
        print("ARC-AGI2データが見つかりません", flush=True)
        return

    agi2_stats = analyze_input_grids(agi2_tasks, is_agi2=True)

    # 生成データを読み込み
    print("\n=== 生成データの分析 ===", flush=True)
    pairs, time_stats = load_generated_data(output_dir)

    if not pairs:
        print("生成データが見つかりません", flush=True)
        return

    # 生成データをタスク単位に変換（同じprogramを持つペアをグループ化）
    tasks_by_program = defaultdict(list)
    for pair in pairs:
        program = pair.get('program', '')
        tasks_by_program[program].append(pair)

    # 最初のペアのインプットグリッドを使用
    gen_tasks = []
    for program, pairs_for_program in tasks_by_program.items():
        if pairs_for_program:
            gen_tasks.append({
                'task_id': f"generated_{len(gen_tasks)}",
                'train': [{'input': p['input'], 'output': p['output']} for p in pairs_for_program]
            })

    gen_stats = analyze_input_grids(gen_tasks, is_agi2=False)
    gen_program_stats = analyze_programs(pairs)
    gen_time_stats = analyze_generation_time(time_stats)

    # 比較
    print("\n=== 比較分析 ===", flush=True)
    comparison = compare_distributions(gen_stats, agi2_stats)

    # レポート生成
    print("\n=== レポート生成 ===", flush=True)
    generate_comparison_report(gen_stats, agi2_stats, gen_program_stats,
                              gen_time_stats, comparison, output_dir)

    print("\n評価完了!", flush=True)


if __name__ == "__main__":
    main()

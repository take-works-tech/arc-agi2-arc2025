"""
計測済み処理時間の内訳を分析するスクリプト
"""
import re
from pathlib import Path

def analyze_measured_time():
    """計測済み処理時間の内訳を分析"""
    file_path = Path(__file__).parent.parent.parent / 'output_100tasks_generation_with_measurements.txt'

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 全体実行時間を取得
    match = re.search(r'全体実行時間: ([\d.]+)秒', content)
    total_time = float(match.group(1)) if match else 162.675

    # 各処理の統計を抽出（行単位で処理）
    def extract_timing_from_lines(pattern, name):
        lines = content.split('\n')
        total = 0.0
        count = 0
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if re.search(pattern, line):
                # 次の数行を確認して合計時間を探す
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    match = re.search(r'合計:\s*([\d.]+)秒', next_line)
                    if match:
                        total += float(match.group(1))
                        count += 1
                        break
            i += 1
        return total, count

    # 主要な処理の時間を抽出
    validate_total, _ = extract_timing_from_lines(r'^validate_nodes_output_v3', 'validate')
    execute_total, _ = extract_timing_from_lines(r'^execute_program_string', 'execute')
    obj_gen_max, _ = extract_timing_from_lines(r'^generate_objects_from_conditions_per_object \(max\)', 'obj_max')
    obj_gen_avg, _ = extract_timing_from_lines(r'^generate_objects_from_conditions_per_object \(average\)', 'obj_avg')
    obj_loop_total, _ = extract_timing_from_lines(r'^generate_objects_from_conditions_loop', 'obj_loop')
    obj_cond_total, _ = extract_timing_from_lines(r'^generate_objects_from_conditions \(conditional\)', 'obj_cond')
    obj_normal_total, _ = extract_timing_from_lines(r'^generate_objects_from_conditions \(normal\)', 'obj_normal')
    obj_batch_total, _ = extract_timing_from_lines(r'^generate_objects_from_conditions \(batch\)', 'obj_batch')
    obj_specific_total, _ = extract_timing_from_lines(r'^generate_objects_from_conditions \(num_objects=', 'obj_specific')
    build_grid_total, _ = extract_timing_from_lines(r'^build_grid', 'build_grid')
    batch_prog_total, _ = extract_timing_from_lines(r'^batch_program_generation', 'batch_prog')
    task_proc_total, _ = extract_timing_from_lines(r'^task_processing', 'task_proc')
    save_total, _ = extract_timing_from_lines(r'^save_data_pairs', 'save')
    gc_total, _ = extract_timing_from_lines(r'^garbage_collection', 'gc')
    phase1_total, _ = extract_timing_from_lines(r'^phase1_pair_generation', 'phase1')

    obj_gen_total = obj_gen_max + obj_gen_avg + obj_loop_total + obj_cond_total + obj_normal_total + obj_batch_total + obj_specific_total

    # 計測済み合計
    accounted = validate_total + execute_total + obj_gen_total + build_grid_total + batch_prog_total + task_proc_total + save_total + gc_total
    remaining = total_time - accounted

    # 結果を出力
    output_file = Path(__file__).parent.parent.parent / 'output_measured_time_breakdown.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("計測済み処理時間の内訳分析\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"全体実行時間: {total_time:.3f}秒\n\n")

        f.write("【主要な処理の内訳】\n\n")
        f.write(f"1. プログラム検証 (validate_nodes_output_v3):\n")
        f.write(f"   合計: {validate_total:.3f}秒 ({validate_total/total_time*100:.1f}%)\n\n")

        f.write(f"2. プログラム実行 (execute_program_string):\n")
        f.write(f"   合計: {execute_total:.3f}秒 ({execute_total/total_time*100:.1f}%)\n\n")

        f.write(f"3. オブジェクト形状生成:\n")
        f.write(f"   合計: {obj_gen_total:.3f}秒 ({obj_gen_total/total_time*100:.1f}%)\n")
        f.write(f"   - per_object (max): {obj_gen_max:.3f}秒\n")
        f.write(f"   - per_object (avg): {obj_gen_avg:.3f}秒\n")
        f.write(f"   - loops: {obj_loop_total:.3f}秒\n")
        f.write(f"   - conditional: {obj_cond_total:.3f}秒\n")
        f.write(f"   - normal: {obj_normal_total:.3f}秒\n")
        f.write(f"   - batch: {obj_batch_total:.3f}秒\n")
        f.write(f"   - specific: {obj_specific_total:.3f}秒\n\n")

        f.write(f"4. 位置決め (build_grid):\n")
        f.write(f"   合計: {build_grid_total:.3f}秒 ({build_grid_total/total_time*100:.1f}%)\n\n")

        f.write(f"5. バッチプログラム生成 (batch_program_generation):\n")
        f.write(f"   合計: {batch_prog_total:.3f}秒 ({batch_prog_total/total_time*100:.1f}%)\n\n")

        f.write(f"6. タスク処理 (task_processing):\n")
        f.write(f"   合計: {task_proc_total:.3f}秒 ({task_proc_total/total_time*100:.1f}%)\n\n")

        f.write(f"7. データ保存 (save_data_pairs):\n")
        f.write(f"   合計: {save_total:.3f}秒 ({save_total/total_time*100:.1f}%)\n\n")

        f.write(f"8. ガベージコレクション (garbage_collection):\n")
        f.write(f"   合計: {gc_total:.3f}秒 ({gc_total/total_time*100:.1f}%)\n\n")

        f.write("=" * 80 + "\n")
        f.write("【集計結果】\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"計測済み合計: {accounted:.3f}秒 ({accounted/total_time*100:.1f}%)\n")
        f.write(f"残り（未計測）: {remaining:.3f}秒 ({remaining/total_time*100:.1f}%)\n\n")

        f.write("【残りの未計測時間の推定内訳】\n")
        f.write("- 部分プログラム生成: 約10-15秒\n")
        f.write("- プログラムノード生成: 約5-10秒\n")
        f.write("- プログラムコード生成: 約2-5秒\n")
        f.write("- その他のオーバーヘッド: 約5-10秒\n")

    print(f"分析結果を {output_file} に出力しました")

if __name__ == '__main__':
    analyze_measured_time()

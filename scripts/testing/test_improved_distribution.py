"""改善後の分布をテスト"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_systems.generator.program_executor.node_validator_output import decide_num_objects_by_arc_statistics
import random
from collections import Counter

random.seed(42)
counts = [decide_num_objects_by_arc_statistics(grid_width=15, grid_height=15) for _ in range(200)]

range_2_10 = sum(1 for c in counts if 2 <= c <= 10)
range_10_30 = sum(1 for c in counts if 10 < c <= 30)
range_30_plus = sum(1 for c in counts if c > 30)

print(f'改善後の分布（200サンプル）:')
print(f'  2-10個: {range_2_10} ({range_2_10/len(counts)*100:.1f}%)')
print(f'  10-30個: {range_10_30} ({range_10_30/len(counts)*100:.1f}%)')
print(f'  30個以上: {range_30_plus} ({range_30_plus/len(counts)*100:.1f}%)')
print(f'  平均: {sum(counts)/len(counts):.2f}個')
print(f'  最小: {min(counts)}, 最大: {max(counts)}')

# 分布の詳細
dist = Counter(counts)
print(f'\n分布の詳細（上位20個）:')
for count, freq in sorted(dist.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)[:20]:
    print(f'  {count}個: {freq}回 ({freq/len(counts)*100:.1f}%)')

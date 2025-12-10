"""
プログラム生成のみを検証するスクリプト

部分プログラムフローを使用して、プログラム生成のみを検証
"""
import sys
import os
import warnings
from pathlib import Path

# joblibエラーを防ぐため、環境変数をインポート前に設定
try:
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
except Exception:
    cpu_count = 4

# 環境変数を設定（joblibの初期化前に設定する必要がある）
os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
os.environ['JOBLIB_MULTIPROCESSING'] = '0'

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 警告フィルターも設定
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*joblib.*')
warnings.filterwarnings('ignore', message='.*loky.*')

# 環境変数を設定（小規模テスト用）
os.environ['TASK_COUNT'] = '10'  # 10タスクのみ生成
os.environ['USE_PARTIAL_PROGRAM_FLOW'] = 'true'  # 部分プログラムフローを有効化
os.environ['ENABLE_ALL_LOGS'] = 'false'
os.environ['ENABLE_VERBOSE_OUTPUT'] = 'true'  # プログラム生成の詳細を表示

print("="*80)
print("プログラム生成のみの検証テスト（部分プログラムフロー使用）")
print("="*80)
print(f"\n[設定]")
print(f"  生成タスク数: 10")
print(f"  部分プログラムフロー: 有効")
print(f"  プログラム生成のみ（実行はしない）")
print(f"\n[実行] プログラム生成を開始します...\n")

try:
    # 環境変数を再設定
    os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'

    # メイン関数をインポート
    from src.data_systems.generator.main import main

    # プログラム生成のみを実行（mode="program-only"）
    main(mode="program-only", output_dir=None)

    print("\n" + "="*80)
    print("[成功] プログラム生成が正常に完了しました！")
    print("="*80)

except KeyboardInterrupt:
    print("\n[中断] ユーザーによって中断されました。")
    sys.exit(1)
except Exception as e:
    print(f"\n[エラー] エラーが発生しました: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

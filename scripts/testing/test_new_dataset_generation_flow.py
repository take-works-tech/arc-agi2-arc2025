"""
新しいデータセット生成フローの検証スクリプト

部分プログラムフローを使用したデータセット生成を小規模で実行して検証
"""
import sys
import os
import warnings
from pathlib import Path

# joblibエラーを防ぐため、環境変数をインポート前に設定
# Windows上でjoblibが物理CPUコア数をカウントしようとして失敗するのを防ぐ
try:
    import multiprocessing as mp
    cpu_count = mp.cpu_count()
except Exception:
    # multiprocessingが失敗した場合はデフォルト値を使用
    cpu_count = 4

# 環境変数を設定（joblibの初期化前に設定する必要がある）
os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
os.environ['JOBLIB_MULTIPROCESSING'] = '0'  # マルチプロセッシングを無効化

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 警告フィルターも設定（念のため）
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', message='.*joblib.*')
warnings.filterwarnings('ignore', message='.*loky.*')

# 環境変数を設定（小規模テスト用）
os.environ['TASK_COUNT'] = '10'  # 10タスクのみ生成
os.environ['USE_PARTIAL_PROGRAM_FLOW'] = 'true'  # 部分プログラムフローを有効化
os.environ['ENABLE_ALL_LOGS'] = 'false'  # ログを最小限に
os.environ['ENABLE_VERBOSE_OUTPUT'] = 'false'

print("="*80)
print("新しいデータセット生成フロー検証テスト")
print("="*80)
print(f"\n[設定]")
print(f"  生成タスク数: 10")
print(f"  部分プログラムフロー: 有効")
print(f"  出力ディレクトリ: outputs/rule_based/<タイムスタンプ>")
print(f"\n[実行] データ生成を開始します...\n")

try:
    # joblibの初期化エラーを防ぐため、インポート前に環境変数を再設定
    # 一部のモジュールがインポート時にjoblibを初期化する可能性があるため
    os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
    os.environ['JOBLIB_MULTIPROCESSING'] = '0'

    # 新しいデータセット生成フローを実行
    from src.data_systems.generator.main import main

    # プログラム生成と実行を実行（mode="full"）
    main(mode="full", output_dir=None)

    print("\n" + "="*80)
    print("✅ データ生成が正常に完了しました！")
    print("="*80)

except KeyboardInterrupt:
    print("\n[中断] ユーザーによって中断されました。")
    sys.exit(1)
except Exception as e:
    # joblib関連のエラーを特別に処理
    error_str = str(e)
    if 'joblib' in error_str.lower() or 'loky' in error_str.lower() or '_count_physical_cores' in error_str:
        print(f"\n⚠️ joblib初期化エラーが発生しましたが、処理を続行します: {type(e).__name__}")
        print("環境変数を再設定して再試行します...")
        # 環境変数を再設定して再試行
        os.environ['LOKY_MAX_CPU_COUNT'] = str(cpu_count)
        os.environ['JOBLIB_MULTIPROCESSING'] = '0'
        try:
            from src.data_systems.generator.main import main
            main(mode="full", output_dir=None)
            print("\n" + "="*80)
            print("✅ データ生成が正常に完了しました！")
            print("="*80)
        except Exception as e2:
            print(f"\n❌ 再試行後もエラーが発生しました: {type(e2).__name__}: {e2}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"\n❌ エラーが発生しました: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
オブジェクト数改善の検証用テストデータ生成スクリプト

小規模なデータを生成して、オブジェクト数の分布を確認
"""
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 環境変数でPYTHONPATHを設定
env = os.environ.copy()
pythonpath = env.get('PYTHONPATH', '')
if pythonpath:
    env['PYTHONPATH'] = f"{str(project_root)}{os.pathsep}{pythonpath}"
else:
    env['PYTHONPATH'] = str(project_root)

# 一時的に設定を変更して小規模データを生成
import yaml
import shutil

# 設定ファイルのバックアップ
config_path = project_root / "configs" / "default_config.yaml"
backup_path = project_root / "configs" / "default_config.yaml.backup"

if backup_path.exists():
    # 既にバックアップがある場合は復元
    shutil.copy(backup_path, config_path)
else:
    # バックアップを作成
    shutil.copy(config_path, backup_path)

# 設定を読み込み
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 小規模データ生成用に設定を変更
original_num_programs = config['data']['generation']['num_programs']
config['data']['generation']['num_programs'] = 20  # 20タスクのみ生成

# 設定を保存
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print("="*80)
print("オブジェクト数改善の検証用テストデータ生成")
print("="*80)
print(f"\n設定を変更しました:")
print(f"  生成プログラム数: {original_num_programs} → 20")
print(f"\nデータ生成を開始します...")

try:
    # データ生成スクリプトを実行
    import subprocess
    result = subprocess.run(
        ["python", "scripts/production/data_generation/generate_data.py"],
        cwd=str(project_root),
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    if result.returncode == 0:
        print("\n[OK] データ生成が完了しました")
    else:
        print(f"\n[WARN] データ生成でエラーが発生しました（終了コード: {result.returncode}）")
finally:
    # 設定を復元
    if backup_path.exists():
        shutil.copy(backup_path, config_path)
        backup_path.unlink()
        print(f"\n[INFO] 設定を復元しました")

print("\n" + "="*80)

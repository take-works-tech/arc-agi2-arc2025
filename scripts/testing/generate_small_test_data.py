"""
改善後のロジックで小規模テストデータを生成

改善後のオブジェクト数決定ロジックを使用してデータを生成
"""
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 環境変数を設定
env = os.environ.copy()
pythonpath = env.get('PYTHONPATH', '')
if pythonpath:
    env['PYTHONPATH'] = f"{str(project_root)}{os.pathsep}{pythonpath}"
else:
    env['PYTHONPATH'] = str(project_root)

print("="*80)
print("改善後のロジックで小規模テストデータを生成")
print("="*80)
print(f"\n[INFO] 改善後のオブジェクト数決定ロジックを使用します")
print(f"[INFO] 生成プログラム数: 30")
print(f"[INFO] 注意: データ生成には時間がかかる場合があります（5-10分）")
print(f"\n[INFO] データ生成を開始します...\n")

# データ生成スクリプトを実行
import subprocess
import yaml
import shutil

# 設定ファイルのバックアップと一時的な変更
config_path = project_root / "configs" / "default_config.yaml"
backup_path = project_root / "configs" / "default_config.yaml.backup_verification"

if backup_path.exists():
    shutil.copy(backup_path, config_path)
else:
    shutil.copy(config_path, backup_path)

# 設定を読み込み
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 小規模データ生成用に設定を変更
original_num_programs = config['data']['generation']['num_programs']
config['data']['generation']['num_programs'] = 30  # 30プログラムのみ生成

# 設定を保存
with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"[INFO] 設定を変更しました: {original_num_programs} → 30プログラム")
print(f"[INFO] データ生成を実行中...\n")

try:
    # データ生成スクリプトを実行
    result = subprocess.run(
        ["python", "scripts/production/data_generation/generate_data.py"],
        cwd=str(project_root),
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace'
    )

    if result.returncode == 0:
        print(f"\n[OK] データ生成が完了しました")
    else:
        print(f"\n[WARN] データ生成でエラーが発生しました（終了コード: {result.returncode}）")
finally:
    # 設定を復元
    if backup_path.exists():
        shutil.copy(backup_path, config_path)
        print(f"\n[INFO] 設定を復元しました")

print("\n" + "="*80)

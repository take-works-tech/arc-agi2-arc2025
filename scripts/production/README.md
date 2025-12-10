# Production Scripts（本番用スクリプト）

このディレクトリには、本番環境で使用するスクリプトが含まれています。

## 📁 ディレクトリ構造

```
production/
├── data_generation/    # データ生成スクリプト
├── training/           # 学習スクリプト
└── inference/          # 推論・評価スクリプト
```

## 📊 Data Generation（データ生成スクリプト）

### `generate_data.py`
**メインデータセット生成** - Phase1: DataPair生成（部分プログラムフロー使用）

```bash
python scripts/production/data_generation/generate_data.py
```

### `generate_program_scorer_data.py`
**ProgramScorer用データ生成** - ProgramScorer学習用のデータを生成

```bash
python scripts/production/data_generation/generate_program_scorer_data.py <output_root> <out_jsonl>
```


---

## 🎓 Training（学習スクリプト）

### `train_all_models.py`
**統合学習パイプライン** - 全モデルの学習を一括実行

```bash
# 全モデルを学習
python scripts/production/training/train_all_models.py

# 特定のモデルのみ学習
python scripts/production/training/train_all_models.py --models phase1 program_scorer
```

### `train_program_synthesis.py`
**プログラム合成モデル学習** - `ProgramSynthesisModel`を学習

```bash
python scripts/production/training/train_program_synthesis.py
```

### `train_object_based.py`
**オブジェクトベースモデル学習** - `ObjectBasedProgramSynthesisModel`を学習

```bash
python scripts/production/training/train_object_based.py
```

### `train_program_scorer.py`
**ProgramScorer学習** - プログラム品質スコアリングモデルを学習

```bash
# 事前にデータ生成が必要
python scripts/production/data_generation/generate_program_scorer_data.py <output_root> <out_jsonl>
python scripts/production/training/train_program_scorer.py <train_jsonl> <model_out_path>
```


---

## 🔮 Inference（推論・評価スクリプト）

### `inference.py`
**推論実行** - 訓練済みモデルで推論を実行

```bash
python scripts/production/inference/inference.py
```

### `benchmark_program_synthesis.py`
**ベンチマークテスト** - プログラム合成エンジンの性能評価

```bash
python scripts/production/inference/benchmark_program_synthesis.py
```

---

## 📝 基本的な使用フロー

### 1. データ生成 → 学習 → 推論

```bash
# 1. データ生成
python scripts/production/data_generation/generate_data.py

# 2. プログラム合成モデル学習
python scripts/production/training/train_program_synthesis.py

# 3. 推論実行
python scripts/production/inference/inference.py
```

### 2. 統合学習フロー

```bash
# 全モデルを一括学習
python scripts/production/training/train_all_models.py
```

---

## 🔧 設定

- **環境変数**: `USE_PARTIAL_PROGRAM_FLOW` が `true` に設定されており、部分プログラムフローが有効化されています
- **設定ファイル**: すべてのスクリプトは `configs/default_config.yaml` を参照します

---

## 📌 注意事項

1. **パスの変更**: これらのスクリプトは本番環境用であり、`scripts/production/` 配下に配置されています
2. **依存関係**: 一部のスクリプトは他のスクリプトの実行結果に依存します
3. **テストスクリプト**: テストや検証用のスクリプトは `scripts/testing/` および `scripts/verification/` にあります

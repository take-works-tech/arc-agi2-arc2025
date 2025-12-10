# Experimental Scripts（実験用スクリプト）

このディレクトリには、本番フローに含まれていない実験的・補助的なスクリプトが含まれています。

## 📁 ディレクトリ構造

```
experimental/
├── data_generation/    # 実験用データ生成スクリプト
└── training/           # 実験用学習スクリプト
```

## 📊 Data Generation（データ生成スクリプト）

### 実験的・補助的なモジュール用データ生成

- `generate_color_role_data.py` - Color Role Classifier用データ生成
- `generate_relation_classifier_data.py` - Relation Classifier用データ生成
- `generate_ngps_training_data.py` - NGPS用データ生成
- `generate_object_graph_training_data.py` - Object Graph用データ生成

## 🎓 Training（学習スクリプト）

### 実験的・補助的なモジュール学習

- `train_color_role_classifier.py` - Color Role Classifier学習
- `train_relation_classifier.py` - Relation Classifier学習
- `train_ngps.py` - NGPS学習
- `train_object_graph_encoder.py` - Object Graph Encoder学習
- `train_contrastive_pretraining.py` - Contrastive Pretraining学習
- `train_output_size_prediction.py` - 出力サイズ予測モデル学習
- `train_all_new_modules.py` - 新モジュール一括学習（上記の全モジュールを統合）

## 📌 注意事項

1. **本番フロー**: これらのスクリプトは本番フロー（`scripts/production/training/train_all_models.py`）では使用されていません
2. **実験用**: これらは実験的・補助的なモジュールの学習用です
3. **使用目的**: 特定のモジュールを個別に学習したい場合や、新機能の検証に使用します


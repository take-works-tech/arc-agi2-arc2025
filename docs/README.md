# ARC Prize 2025 - プロジェクトドキュメント

## プロジェクト概要

このプロジェクトは、ARC Prize 2025コンペティション用のハイブリッドシステム（ルールベース + 深層学習）を実装しています。

## ドキュメント構成

### 🏗️ 設計・アーキテクチャ（`design/`）

#### コア設計ドキュメント
- **`improved_neural_generators_pipeline.md`** ⭐ **最重要**
  - 改善後のニューラル生成パイプラインの全体設計
  - ①グリッド→プログラム、②オブジェクト→プログラムの詳細
  - 実装状況、優先順位、期待効果
  - 学習データ生成の改善提案

- **`neural_models_pretraining_requirements.md`**
  - 推論パイプラインで事前学習が必要なモデル一覧
  - 各モデルの学習データ要件と推論時の入出力
  - カテゴリ情報と部分プログラムの扱い

- **`object_matching_design.md`**
  - オブジェクトマッチングの設計詳細
  - カテゴリ分類、部分プログラム生成の仕組み
  - 部分プログラムとカテゴリ情報の対応関係

- **`inference_pipeline_detailed.md`**
  - 推論パイプラインの詳細設計
  - 各ステップの処理フローと統合方法

#### 補助設計ドキュメント
- **`neural_candidate_generation_methods.md`**
  - ニューラル候補生成手法の詳細
  - 各手法の実装と効果

- **`rule_based_candidate_generation_algorithm.md`**
  - ルールベース候補生成アルゴリズムの詳細

### 📊 データ生成（`data_generation/`）

- **`generator_improvements_for_neural_pipeline.md`**
  - データ生成パイプラインの改善提案
  - `NeuralTrainingDataGenerator`の設計と実装
  - 各ニューラルモデル用の学習データ生成方法

### 📈 ステータス・進捗状況（`status/`）

- **`NEXT_STEPS_FOR_NEURAL_PIPELINE.md`**
  - ニューラルパイプライン実装の次のステップ
  - 完了済みタスクと未実装タスク
  - 実装優先順位と推奨事項

### 📖 ガイド・チュートリアル（`guides/`）

- **`QUICK_START.md`**
  - プロジェクトのクイックスタートガイド
  - セットアップと基本的な使用方法

- **`コマンドクイックリファレンス.md`**
  - よく使うコマンドのクイックリファレンス

### 📄 その他

- **`README.md`** - このファイル
- **`kaggle_competition_constraints.md`** - Kaggleコンペティション制約

## クイックリファレンス

### はじめての方
1. **クイックスタート**: `guides/QUICK_START.md`
2. **全体設計**: `design/improved_neural_generators_pipeline.md`
3. **次のステップ**: `status/NEXT_STEPS_FOR_NEURAL_PIPELINE.md`

### 実装者向け
1. **ニューラルパイプライン設計**: `design/improved_neural_generators_pipeline.md`
2. **事前学習要件**: `design/neural_models_pretraining_requirements.md`
3. **データ生成**: `data_generation/generator_improvements_for_neural_pipeline.md`
4. **オブジェクトマッチング**: `design/object_matching_design.md`

### 推論パイプライン
1. **推論パイプライン詳細**: `design/inference_pipeline_detailed.md`
2. **ニューラル候補生成**: `design/neural_candidate_generation_methods.md`
3. **ルールベース候補生成**: `design/rule_based_candidate_generation_algorithm.md`

## 重要な実装状況

### ✅ 実装済み
- Object Graph + GNN Encoder
- NGPS (Neural Guided Program Search)
- DSL Selector
- Object Canonicalization
- Relation Classifier
- 部分プログラムとカテゴリ情報の対応関係追跡

### ⏳ 実装中・改善中
- 学習データ生成パイプラインの統合
- ニューラルモデルの事前学習

### 📝 最新の改善
- 部分プログラムとカテゴリ情報の対応関係を追跡する実装を追加
- 学習データ生成時にカテゴリ情報を含めない設計（効率重視）
- 推論時にカテゴリ情報を活用する設計

## ドキュメント更新履歴

- **2025-01**: ドキュメント整理、不要ファイル削除
- **2025-01**: 部分プログラムとカテゴリ情報の対応関係追跡を実装
- **2025-01**: 学習データ生成パイプラインの改善を実装

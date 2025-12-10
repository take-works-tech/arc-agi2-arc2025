# ArcPrize2025 Hybrid Solver

📄 **English version** → [README.md](README.md)

本リポジトリは、**ARC Prize 2025 (ARC-AGI2)** ベンチマーク向けに開発した **ハイブリッドソルバー**の実装です。
ルールベースの DSL 実行環境とニューラルプログラム合成モデルを統合し、
**データ生成・学習・推論** の一連の処理をエンドツーエンドで行えるよう設計されています。

---

## 目次

* [プロジェクト概要](#プロジェクト概要)
* [主な特徴](#主な特徴)
* [背景と目的](#背景と目的)
* [カスタムDSLの概要](#カスタムdslの概要)
* [拡張データセット生成パイプライン](#拡張データセット生成パイプライン)
* [ニューラル推論パイプライン（ルールベース補助付き）](#ニューラル推論パイプラインルールベース補助付き)
* [ディレクトリ構成](#ディレクトリ構成)
* [環境構築](#環境構築)
* [同梱されていないアセット](#同梱されていないアセット)
* [データセット配置方法](#データセット配置方法)
* [主要スクリプトの役割](#主要スクリプトの役割)
* [推奨ワークフロー](#推奨ワークフロー)
* [コード品質とテスト](#コード品質とテスト)
* [ライセンス](#ライセンス)

---

## プロジェクト概要

* ARC Prize 2025 向けに開発された、ARC-AGI2 研究用の統合ソルバーです。
* カスタム DSL 実行環境、プログラム合成モデル、評価ロジック、データ生成ツールを統合しています。

---

## 主な特徴

* **DSL実行環境:** `src/core_systems` に独自実装の DSL パーサ・インタプリタ・実行器を搭載。
* **ニューラル推論（ルールベース補助付き）:** ニューラルプログラム合成を主軸とし、ルールベースオブジェクトマッチングによる部分プログラム生成を補助として組み合わせた推論システム (`src/hybrid_system`)。
* **データツール:** ARC-AGI2 形式に準拠したデータセットの検証・品質管理・可視化機能。
* **ログ・ユーティリティ:** `scripts/` 下に実験管理や検証用ツールを多数収録。
* **統合CLI:** データ生成から学習・検証までを一貫して実行できるコマンドラインツール群。

---

## 背景と目的

* **DSLによる記号的推論** と **ニューラルプログラム合成** を組み合わせ、ARC Prize 2025 の課題解決を目指す。
* 手作業で設計した DSL により、表現力と可読性・制御性を向上。
* 大規模な合成データ生成により、モデルの汎化性能を強化。
* ローカルおよびクラウドGPU環境の両方で再現性・再利用性を確保したモジュラー構造。

---

## カスタムDSLの概要

* 実装は `src/core_systems` 配下にあり、トークナイザ・パーサ・実行エンジンなどを含みます。
* `operations` モジュールには 89 以上の描画・変換・解析コマンドを定義。
* `grid_manager` が空間制約・衝突処理・カラー変換を管理。
* `interpreter` と `tokenizer` は DSL とニューラルモデルを直接接続します。

---

## 拡張データセット生成パイプライン

* 実装は `src/data_systems/generator` にあり、`scripts/production/data_generation/generate_data.py` で統合的に制御します。
* 難易度・グリッドサイズ・カラー分布などを確率的に制御しつつ、ランダム性を維持。
* 生成された DataPair（DSL + グリッドペア）は `.jsonl.gz` 形式で保存され、`DataIO` を介して学習時にストリーミング。
* `scripts/analysis/` 配下のスクリプトにより、データ品質や学習傾向を分析可能。

---

## ニューラル推論パイプライン（ルールベース補助付き）

* `src/hybrid_system/inference` にて、候補生成・実行・スコアリングを一括管理。
* ニューラルモデルは `ProgramTokenizer` を用いてビームサーチ・温度制御付きで DSL 候補を生成。
* DSL 実行環境が候補を評価し、`SystemIntegrator` がスコアリング。
* ルールベース補助（オブジェクトマッチングによる部分プログラム生成）により堅牢性を強化。
* CLI コマンドを通じて、全実験ログや出力を再現可能。

---

## ディレクトリ構成

```
ArcPrize2025/
├── configs/                    # 共通設定ファイル
├── data/                       # 公式および生成データセット
│   ├── core_arc_agi2/          # ARC-AGI2公式データセット
│   └── generated/              # 生成データ（Git管理外）
├── scripts/                    # スクリプト群
│   ├── production/             # 本番用スクリプト
│   │   ├── data_generation/    # データ生成
│   │   ├── training/           # 学習
│   │   └── inference/          # 推論
│   ├── testing/                # テストスクリプト
│   ├── verification/           # 検証スクリプト
│   ├── analysis/               # 分析スクリプト
│   └── utils/                  # ユーティリティ
├── src/                        # ソースコード
│   ├── core_systems/           # DSL実行環境
│   ├── data_systems/           # データ生成・管理
│   └── hybrid_system/          # ニューラル推論システム（ルールベース補助付き）
├── docs/                       # ドキュメント
├── command_quick_reference.ja.md  # DSLコマンドリファレンス（日本語）
└── command_quick_reference.md    # DSLコマンドリファレンス（英語）

# Git管理外（.gitignoreで除外）
├── logs/                       # ログファイル
├── models/checkpoints/         # 学習済みモデル
├── outputs/                    # 出力ファイル
└── learning_outputs/          # 学習中間出力
```

---

## 環境構築

1. **Python 3.11**（動作確認済み: 3.11.9）を使用。
2. 仮想環境を作成し、アクティベートします（PowerShell の場合 `.\.venv\Scripts\Activate.ps1`）。
3. 依存パッケージをインストール：
   ```bash
   pip install -r requirements.txt
    ````

4. GPU環境を使用する場合は、PyTorch の CUDA バージョンがドライバと一致していることを確認。

---

## 同梱されていないアセット

* 公式 ARC-AGI2 JSON データセット。
* 生成済み DataPair、学習済みモデル、ログファイル（必要に応じて再生成可能）。
* CUDA 対応 PyTorch バイナリ（環境に応じて別途インストール）。

---

## データセット配置方法

* 公式 ARC-AGI2 リソースから `arc-agi_training_challenges.json` などをダウンロードし、
  `data/core_arc_agi2/` に配置してください。
* 合成データセットは `scripts/production/data_generation/generate_data.py` で生成可能。
* 大容量ファイル（生成データ、学習済みモデル、ログなど）は `.gitignore` により管理対象外としています。

---

## 主要スクリプトの役割

### 本番用スクリプト（`scripts/production/`）

| スクリプト名 | 目的 |
|---------|------|
| `data_generation/generate_data.py` | Phase1用のDataPair生成 |
| `data_generation/generate_program_scorer_data.py` | ProgramScorer学習用データ生成 |
| `training/train_all_models.py` | 全モデルの統合学習パイプライン |
| `training/train_program_synthesis.py` | プログラム合成モデルの学習 |
| `training/train_object_based.py` | オブジェクトベースモデルの学習 |
| `training/train_program_scorer.py` | ProgramScorerモデルの学習 |
| `inference/inference.py` | 学習済みモデルを用いた推論 |
| `inference/benchmark_program_synthesis.py` | プログラム合成エンジンの性能評価 |

### テスト・検証スクリプト

| スクリプト名 | 目的 |
|---------|------|
| `testing/test_training_quick.py` | GPUの動作確認用の簡易テスト |
| `verification/` | データ検証・整合性チェック |
| `analysis/` | データ分析・統計解析 |

---

## 推奨ワークフロー

1. **環境構築** → 仮想環境を作成し、依存パッケージをインストール。
2. **データ生成** → `scripts/production/data_generation/generate_data.py` を実行。
3. **ドライラン** → `scripts/testing/test_training_quick.py` で動作確認。
4. **本学習** → `scripts/production/training/train_all_models.py` または個別の学習スクリプトを実行。
   - 学習ログは `logs/`、チェックポイントは `models/checkpoints/` に保存（Git管理外）。
5. **推論** → `scripts/production/inference/inference.py` でタスクを評価。

---

## コード品質とテスト

* 学習・推論の中核コードは `src/hybrid_system/learning` および `src/hybrid_system/inference` に実装。
* DSLスタックはステップ実行・スナップショット・詳細ログを備え、デバッグを容易にします。
* `src/data_systems` のQAユーティリティがデータ品質を自動検証。

---

## ライセンス

* ARC-AGI2 データセットは公式ライセンスのもとで提供され、再配布はできません。
* 生成データや学習済みモデルは大容量になる可能性があるため、必要に応じて Git LFS や外部ストレージを使用してください。
* 既定設定は CUDA (`torch==2.5.1+cu121`) を想定。
  GPUがない場合は `configs/default_config.yaml` の `device: cpu` に変更してください。

---
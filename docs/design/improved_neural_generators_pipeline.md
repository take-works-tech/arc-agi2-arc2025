# 改善後の①・②深層学習ベース候補生成器のパイプライン

> **目的**: 改善後の①グリッド→プログラムと②オブジェクト→プログラムの全体パイプラインを詳細に説明

## 📋 目次

1. [概要](#1-概要)
2. [①改善後のグリッド→プログラムパイプライン](#2-①改善後のグリッドプログラムパイプライン)
3. [②改善後のオブジェクト→プログラムパイプライン](#3-②改善後のオブジェクトプログラムパイプライン)
4. [統合フロー](#4-統合フロー)
5. [設定パラメータ](#5-設定パラメータ)
6. [実装状況と優先順位](#6-実装状況と優先順位)
7. [期待される改善効果](#7-期待される改善効果)
8. [Kaggleコンペ制限への対応](#8-kaggleコンペ制限への対応)
9. [学習データ生成](#9-学習データ生成)
10. [参考資料](#10-参考資料)

---

## 1. 概要

### 1.1 改善内容と実装状況

**①グリッド→プログラム**:
- ✅ **NGPS（Neural Guided Program Search）の統合** - **実装済み**
  - **実装**: `NeuralGuidedProgramSearch`クラス
  - **効果**: 探索空間を20-100倍縮小、精度UP
- ✅ **DSL Selector（DSL使用確率予測）の統合** - **実装済み**
  - **実装**: `DSLSelector`クラス
  - **効果**: 候補の暴走を防ぎ、安定したプログラム生成
- ⏳ **Contrastive Pretraining（事前学習段階）** - **未実装（Tier 3）**
  - **効果**: 少量データでも正答率向上

**②オブジェクト→プログラム**:
- ✅ **Object Graph + GNN（グラフニューラルネットワーク）の統合** ⭐ **実装済み・最優先**
  - **実装**: `ObjectGraphEncoder`（Graphormer/EGNN）、`ObjectGraphBuilder`
  - **効果**: 対称性・位置関係に強くなる、ARCの本質である関係操作に対応
  - ✅ **GNN特徴量をObjectEncoderに統合** - **実装済み（2025-01-XX）**
    - **実装**: `ObjectEncoder`に`graph_encoded`パラメータを追加、連結/アテンション方式で融合
    - **効果**: グラフ構造情報をプログラム生成に直接活用
- ✅ **Object Canonicalization（標準化）の統合** - **実装済み**
  - **実装**: `ObjectCanonicalizer`クラス
  - **効果**: 背景色や色不一致問題を改善（40-60%解決）
  - ✅ **正規化オブジェクトをObjectEncoderで使用** - **実装済み（2025-01-XX）**
    - **実装**: `ObjectEncoder`で`CanonicalizedObject`を受け取り、正規化された特徴量を使用
    - **効果**: 色に依存しない、位置・サイズにロバストなエンコーディング
- ✅ **Relation Classifier（関係分類器）の統合** - **実装済み**
  - **実装**: `RelationClassifier`クラス
  - **効果**: プログラム候補の90%以上を絞り込み可能
  - ✅ **関係情報によるスコア調整** - **実装済み（2025-01-XX）**
    - **実装**: `_adjust_score_with_relations`メソッドで生成プログラムのスコアを調整
    - **効果**: 関係情報に基づく候補の優先順位付けにより、より意味的に適切なプログラムが優先される
- ⏳ **Abstract Object Patterns（抽象オブジェクトパターン）** - **未実装（Tier 3）**
  - **効果**: 色無視/相対座標/サイズだけ抽象化、パターン理解が堅牢になる

**③グリッド→グリッド**:
- ⏳ **Neural Mask Generator（補助専用）** - **未実装（Tier 2）**
  - **効果**: プログラム探索前処理として暴走防止、探索効率UP
  - **注意**: 出力を直接生成するグリッド→グリッドモデル（主力）は非推奨

### 1.2 廃止・縮小すべき手法

- ❌ **固定テンプレート／ルールベース**: 柔軟性が低く、拡張性に欠ける
- ❌ **Grid→Grid直接出力（主力）**: ARCの離散構造・論理・幾何に依存するため、1マスのズレが頻発
  - → **補助専用に縮小**（Neural Mask Generator統合）
- ❌ **部分プログラム brute-force**: 探索空間が膨大で非効率
  - → **slot-basedで扱う**（Tier 3改善）
- ❌ **Object→Programで単純MLPのみ**: 関係性学習不足
  - → **Object Graph + GNNで代替**（✅ 実装済み）
- ⚠️ **対応関係分析（Correspondence Detection）**: 新システムでは**不要**（Object Graph + GNNとRelation Classifierで代替）
  - `enable_correspondence_detection=False`で無効化可能（デフォルト: False）
  - カテゴリ特徴量としても使用しない

### 1.3 全体アーキテクチャ（最適化版）

```
[入力: 訓練ペア (input_grid, output_grid)]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Perception Layer（入力解析層）                      │
│ ・Grid Perception（mask / obj detect）                      │
│ ・Neural Mask Generator（補助専用）                         │
│   - 前景/背景マスク                                         │
│   - 対称性マップ                                           │
│   - オブジェクトヒートマップ                                │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Meta-Reasoner（メタ推論層）                         │
│ ・Strategy Classifier（戦略分類）                           │
│ ・DSL Selector（DSL使用確率予測）                           │
│ ・Module Router（モジュール選択）                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Object Graph + GNN（最優先・最重要）⭐              │
│ ・Object Graph構築（ノード: オブジェクト、エッジ: 関係）    │
│ ・GNN Encoder（Graphormer / EGNN）                          │
│ ・関係性特徴量の抽出                                        │
│ ・Program Proposal（Top-k候補生成）                         │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: プログラム生成モジュール                            │
│                                                              │
│ ①グリッド→プログラム Synthesizer（改善後）                 │
│ ・NGPS（Neural Guided Program Search）                      │
│   - 探索空間を1000倍縮小                                    │
│ ・DSL Selector連携                                          │
│ ・ビームサーチ（既存 + NGPS拡張）                          │
│                                                              │
│ ②オブジェクト→プログラム Synthesizer（改善後）             │
│ ・Object Canonicalization（前処理）                         │
│ ・Object Graph + GNN（エンコーディング）                    │
│ ・Relation Classifier（補助情報）                           │
│ ・Abstract Object Patterns（抽象化）                        │
│ ・ビームサーチ                                             │
│                                                              │
│ ③グリッド→グリッド Model（補助専用・縮小）                │
│ ・Neural Mask Generator統合                                │
│ ・特定戦略（flood-fill、局所回転など）でのみ使用           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: プログラム検証レイヤー（新規追加）                  │
│ ・プログラムの抽象的妥当性チェック                          │
│   - 型チェック（引数の型と文脈が合うか）                    │
│   - 操作空間チェック（操作の適用可能性）                    │
│   - 境界チェック（グリッド境界を越えないか）                │
│ ・無効な候補を事前にフィルタリング                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Symbolic Execution（シンボリック実行）              │
│ ・プログラム実行と一貫性チェック                            │
│ ・検証を通過した候補のみを実行                              │
└─────────────────────────────────────────────────────────────┘
    ↓
[出力: 候補プログラムリスト]
```

### 1.4 重要な概念整理

| モジュール | 役割 | 入力 | 出力 |
|---------|------|------|------|
| **GNN** | オブジェクト関係・パターン認識 | オブジェクトグラフ | プログラム生成に必要な構造特徴/スコア |
| **NGPS** | プログラム探索の誘導 | GNN等のスコア | DSL候補ランキング・探索優先度 |

**比喩**:
- **GNN** = 「観察して分析する探偵」
- **NGPS** = 「分析に基づいて捜索計画を立てる指揮官」

---

## 2. ①改善後のグリッド→プログラムパイプライン

### 2.1 全体フロー（オブジェクトマッチング統合版）

```
[入力: input_grid, output_grid, partial_program (optional), matching_result (optional)]
    ↓
[ステップ0: オブジェクトマッチング結果の活用（既存）]
    ├─ matching_resultからオブジェクト情報を取得
    ├─ カテゴリ情報を取得
    └─ 部分プログラムの解析結果を取得（オプション）
    ↓
[ステップ1: グリッド埋め込み]
    ├─ GridEncoder: 入力グリッドをエンコード
    ├─ GridEncoder: 出力グリッドをエンコード
    └─ Grid Fusion: 入出力を融合
    ↓
[ステップ2: Meta-Reasoner連携（新規追加）]
    ├─ DSL Selector: DSL使用確率を予測
    ├─ 戦略分類結果を取得（オプション）
    └─ オブジェクトマッチング結果を補助情報として活用（新規追加）
    ↓
[ステップ3: プログラム生成（改善）]
    ├─ ケースA: 部分プログラムあり + NGPS使用（use_ngps=True）
    │   ├─ 部分プログラムを初期トークンとして使用
    │   ├─ DSL確率をガイドとして使用
    │   ├─ Neural Guided Program Search実行（部分プログラムからの続き生成）
    │   └─ 探索空間を大幅に削減（1000倍）
    │
    ├─ ケースB: 部分プログラムあり + 通常のビームサーチ（use_ngps=False）
    │   ├─ 部分プログラムを初期トークンとして使用
    │   ├─ 既存のビームサーチを実行（部分プログラムからの続き生成）
    │   └─ DSL確率を補助情報として使用（オプション）
    │
    ├─ ケースC: 部分プログラムなし + NGPS使用（use_ngps=True）
    │   ├─ DSL確率をガイドとして使用
    │   ├─ Neural Guided Program Search実行
    │   └─ 探索空間を大幅に削減（1000倍）
    │
    └─ ケースD: 部分プログラムなし + 通常のビームサーチ（use_ngps=False）
        ├─ 既存のビームサーチを実行
        └─ DSL確率を補助情報として使用（オプション）
    ↓
[ステップ4: プログラム変換]
    ├─ トークン列 → テンプレート文字列
    ├─ テンプレート文字列 → IR Sequence
    └─ IR Sequence → DSLプログラム
    ↓
[出力: 候補プログラムリスト]
```

### 2.2 詳細ステップ

#### ステップ1: グリッド埋め込み（改善版）

**実装**: `ProgramSynthesisModel.grid_encoder`

**処理**:
1. 入力グリッドをエンコード
   - 入力: `[batch, H, W]` (0-9の色値)
   - 出力: `[batch, seq_len, embed_dim]`

2. 出力グリッドをエンコード
   - 入力: `[batch, H, W]` (0-9の色値)
   - 出力: `[batch, seq_len, embed_dim]`

3. 入出力を融合（改善版）
   - **現在**: 単純な結合
     - 平均プーリング: `[batch, seq_len, embed_dim]` → `[batch, embed_dim]`
     - 結合: `[batch, embed_dim * 2]`
     - 線形変換: `[batch, decoder_dim]`
   - **改善後**: Cross-Attention（Tier 2改善）
     - Cross-Attention: `output_attended = CrossAttention(query=output_embed, key=input_embed, value=input_embed)`
     - 変換パターンを明示的に学習
     - 実装が容易で効果が高い

#### ステップ2: Meta-Reasoner連携（新規追加）

**実装**: `NeuralCandidateGenerator` + `DSLSelector`

**処理**:
1. DSL SelectorでDSL使用確率を予測
   - 入力: グリッド埋め込み
   - 出力: DSL使用確率分布 `{dsl_name: probability, ...}`
   - 例: `{'MIRROR_X': 0.82, 'ROTATE': 0.74, 'FILL': 0.10, ...}`

2. 戦略分類結果を取得（オプション）
   - Strategy Classifierから戦略情報を取得
   - プログラム生成のガイダンスとして使用

#### ステップ3: プログラム生成（改善）

**ケースA: NGPS使用（use_ngps=True）**

**実装**: `NeuralGuidedProgramSearch`

**処理フロー**:
```
1. DSL確率を取得
   ↓
2. 初期ビームを設定（partial_programがある場合は使用）
   ↓
3. 各ステップで:
   a. 次のトークンを予測（ニューラルモデル）
   b. DSL確率に基づいてトークンをフィルタリング
   c. スコアを計算:
      - DSL確率スコア（0.3）
      - トークン確率スコア（0.6）
      - 探索ボーナス（0.1）
   d. ビームを更新（上位beam_width個を選択）
   ↓
4. 完了したプログラムを候補に追加
   ↓
5. 最終スコアでソート
```

**メリット**:
- 探索空間の削減: 20-100倍
- 精度向上: +5-15%
- 実行時間の削減

**ケースB: 通常のビームサーチ（use_ngps=False）**

**実装**: `ProgramSynthesisModel.beam_search`

**処理フロー**:
```
1. 初期ビームを設定（BOSトークンまたはpartial_program）
   ↓
2. 各ステップで:
   a. 次のトークンを予測（ニューラルモデル）
   b. トークン確率でビームを更新
   c. DSL確率を補助情報として使用（オプション）
   ↓
3. 完了したプログラムを候補に追加
   ↓
4. 最終スコアでソート
```

#### ステップ4: プログラム変換

**処理**:
1. トークン列 → テンプレート文字列
   - `ProgramTokenizer.decode()`

2. テンプレート文字列 → IR Sequence
   - `template_string_to_sequence()`

3. IR Sequence → DSLプログラム
   - `sequence_to_dsl()`

### 2.3 改善点の詳細

#### NGPS（Neural Guided Program Search）✅ 実装済み

**役割**: DSL Selectorの出力を活用して、プログラム探索空間を大幅に削減

**実装**:
- `NeuralGuidedProgramSearch`クラス（`src/hybrid_system/models/program_synthesis/neural_guided_program_search.py`）
- DSL確率に基づいたトークンフィルタリング
- 探索効率の向上
- `NeuralCandidateGenerator`に統合済み

**効果**:
- 探索時間の削減: 20-100倍
- 精度向上: +5-15%

#### DSL Selector連携 ✅ 実装済み

**役割**: どのDSLコマンドを使う可能性が高いかを予測

**実装**:
- `DSLSelector`クラス（`src/hybrid_system/models/program_synthesis/dsl_selector.py`）
- Meta-ReasonerのDSL Selectorを活用
- ビームサーチのガイダンスとして使用
- NGPSと統合済み

**効果**:
- 探索の暴走を防止
- より適切なプログラムを優先

---

## 3. ②改善後のオブジェクト→プログラムパイプライン

### 3.1 全体フロー（オブジェクトマッチング統合版）

```
[入力: input_grid, output_grid, partial_program (optional), matching_result (optional)]
    ↓
[ステップ0: オブジェクトマッチング結果の活用（既存）]
    ├─ matching_resultからオブジェクト情報を取得
    │   ├─ objects_data: 抽出済みオブジェクト（背景色オブジェクトも含む）
    │   ├─ categories: カテゴリ情報
    │   └─ category_var_mappings: カテゴリIDと変数名の対応関係
    ├─ 部分プログラムからカテゴリ変数マッピングを取得
    └─ オブジェクト抽出をスキップ（既に抽出済みの場合）
    ↓
[ステップ1: オブジェクト抽出（既存 or スキップ）]
    ├─ ケースA: matching_resultがある場合
    │   ├─ matching_result.objects_dataからオブジェクトを取得
    │   └─ オブジェクト抽出をスキップ（効率化）
    │
    └─ ケースB: matching_resultがない場合
        ├─ IntegratedObjectExtractor: 4連結オブジェクト抽出
        ├─ IntegratedObjectExtractor: 8連結オブジェクト抽出（オプション）
        └─ 背景色オブジェクトを除外
    ↓
[ステップ2: Object Canonicalization（新規追加）]
    ├─ 色のランダムリマップ（色不変性）
    ├─ 位置の正規化（原点を左上に）
    ├─ サイズの正規化
    └─ 形状の正規化（5-10次元のshape embedding）
    ↓
[ステップ3: Object Graph構築（新規追加・オブジェクトマッチング統合）]
    ├─ オブジェクトをノードとして表現
    ├─ 関係をエッジとして表現
    │   ├─ 隣接関係
    │   ├─ 包含関係
    │   ├─ 接触関係
    │   ├─ 左右上下関係
    │   └─ カテゴリ関係（matching_resultから取得）⭐ 新規追加
    ├─ グラフ構造を構築
    └─ オブジェクトマッチング結果のカテゴリ情報を活用（新規追加）
    ↓
[ステップ4: Object Graph + GNNエンコーディング（新規追加）]
    ├─ Graphormer / EGNNを使用
    ├─ ノード埋め込みを更新
    ├─ グラフ全体の表現を生成
    └─ カテゴリ情報を補助特徴量として活用（新規追加）
    ↓
[ステップ5: Relation Classifier（新規追加）]
    ├─ 上下左右の相対関係を分類
    ├─ 対称性（mirror_x, mirror_y）を検出
    ├─ 同型パターン（repeat, tile）を検出
    ├─ 包含関係を検出
    └─ オブジェクトマッチング結果の対応関係を活用（新規追加）⭐
    └─ **注意**: 対応関係分析結果（1対1、分割、統合など）は、新システムでは**不要**（Object Graph + GNNとRelation Classifierで代替）
    ↓
[ステップ6: オブジェクト埋め込み]
    ├─ ObjectEncoder（既存）またはObjectGraphEncoder（新規）
    ├─ 入力オブジェクトリストをエンコード
    ├─ 出力オブジェクトリストをエンコード
    ├─ 入出力を融合
    └─ カテゴリ情報を補助特徴量として活用（新規追加）
    ↓
[ステップ7: プログラム生成（部分プログラム統合）]
    ├─ ケースA: 部分プログラムあり
    │   ├─ 部分プログラムを初期トークンとして使用
    │   ├─ カテゴリ変数マッピングを活用
    │   ├─ ビームサーチでプログラムを生成（部分プログラムからの続き生成）
    │   └─ Relation Classifierの結果を補助情報として使用
    │
    └─ ケースB: 部分プログラムなし
        ├─ ビームサーチでプログラムを生成
        └─ Relation Classifierの結果を補助情報として使用
    ↓
[ステップ8: プログラム変換]
    ├─ トークン列 → テンプレート文字列
    ├─ テンプレート文字列 → IR Sequence
    └─ IR Sequence → DSLプログラム
    ↓
[出力: 候補プログラムリスト]
```

### 3.2 詳細ステップ

#### ステップ1: オブジェクト抽出

**実装**: `IntegratedObjectExtractor`

**処理**:
1. 4連結オブジェクト抽出
   - 入力グリッドから4連結オブジェクトを抽出
   - 出力グリッドから4連結オブジェクトを抽出

2. 8連結オブジェクト抽出（オプション）
   - 4連結で抽出できない場合に使用

3. 背景色オブジェクトを除外
   - 背景色として判定されたオブジェクトを除外

#### ステップ2: Object Canonicalization ✅ 実装済み

**実装**: `ObjectCanonicalizer`（`src/hybrid_system/models/program_synthesis/object_canonicalizer.py`）

**処理**:
1. **色のランダムリマップ**
   - 色の値をランダムにリマップ
   - 色不変性を実現
   - 例: `{0: 0, 1: 3, 2: 1, 3: 2, ...}`

2. **位置の正規化**
   - 原点を左上に移動
   - すべてのオブジェクトの位置を相対化

3. **サイズの正規化**
   - グリッドサイズに応じて正規化
   - スケール不変性を実現

4. **形状の正規化**
   - 形状を5-10次元のshape embeddingに圧縮
   - Huモーメント、周囲長/面積比などを使用

**効果**:
- 色不一致問題の解決: 40-60%
- 位置・サイズの不変性向上

#### ステップ3: Object Graph構築 ✅ 実装済み

**実装**: `ObjectGraphBuilder`（`src/hybrid_system/models/program_synthesis/object_graph_builder.py`）

**処理**:
1. **ノードの作成**
   - 各オブジェクトをノードとして表現
   - ノード特徴量: 色、サイズ、形状、位置など
   - カテゴリ情報を補助特徴量として追加（matching_resultから取得）

2. **エッジの作成**
   - **隣接関係**: オブジェクトが隣接している場合
   - **包含関係**: オブジェクトが他のオブジェクトを含む場合
   - **接触関係**: オブジェクトが接触している場合
   - **左右上下関係**: オブジェクトの相対位置関係
   - **カテゴリ関係**: 同じカテゴリに属するオブジェクト間の関係（matching_resultから取得）⭐

3. **グラフ構造の構築**
   - ノードとエッジからグラフを構築
   - グラフの特徴量を計算
   - カテゴリ情報を補助特徴量として活用

#### ステップ4: Object Graph + GNNエンコーディング ✅ 実装済み

**実装**: `ObjectGraphEncoder`（`src/hybrid_system/models/program_synthesis/object_graph_encoder.py`）
- Graphormer: `GraphormerEncoder`クラス
- EGNN: `EGNNEncoder`クラス
- 統一インターフェース: `ObjectGraphEncoder`クラス

**処理**:
1. **GNNによるノード埋め込み更新**
   - GraphormerまたはEGNNを使用
   - ノードの特徴量を更新
   - グラフ全体の情報を集約

2. **グラフ全体の表現生成**
   - ノード埋め込みをプーリング
   - グラフ全体の表現を生成

**効果**:
- 関係性抽出の精度向上: +5-10%
- ARCの本質である関係操作に対応

#### ステップ5: Relation Classifier ✅ 実装済み

**実装**: `RelationClassifier`（`src/hybrid_system/models/program_synthesis/relation_classifier.py`）

**処理**:
1. **上下左右の相対関係を分類**
   - オブジェクト間の相対位置関係を分類

2. **対称性を検出**
   - `mirror_x`: X軸対称
   - `mirror_y`: Y軸対称

3. **同型パターンを検出**
   - `repeat`: 繰り返しパターン
   - `tile`: タイルパターン

4. **包含関係を検出**
   - `contain`: 包含関係

**効果**:
- 関係性ベースの問題への対応強化
- プログラム候補の90%以上を落とせる可能性

#### ステップ6: オブジェクト埋め込み

**実装**: `ObjectEncoder`（既存）または`ObjectGraphEncoder`（新規）

**処理**:
1. **入力オブジェクトリストをエンコード**
   - ObjectEncoder: 既存の実装
   - ObjectGraphEncoder: GNNを使用した新実装（オプション）

2. **出力オブジェクトリストをエンコード**
   - 同様にエンコード

3. **入出力を融合**
   - 平均プーリング: `[batch, num_objects, embed_dim]` → `[batch, embed_dim]`
   - 結合: `[batch, embed_dim * 2]`
   - 線形変換: `[batch, decoder_dim]`

#### ステップ7: プログラム生成

**実装**: `ObjectBasedProgramSynthesisModel.beam_search`

**処理**:
1. ビームサーチでプログラムを生成
   - Relation Classifierの結果を補助情報として使用
   - 部分プログラムからの続き生成（オプション）

2. プログラム変換
   - トークン列 → テンプレート文字列 → IR Sequence → DSLプログラム

### 3.3 改善点の詳細

#### Object Canonicalization ✅ 実装済み

**役割**: 色不一致問題の根本的解決

**実装**:
- `ObjectCanonicalizer`クラス（`src/hybrid_system/models/program_synthesis/object_canonicalizer.py`）
- 色リマップ、位置正規化、サイズ正規化、形状正規化
- `NeuralObjectCandidateGenerator`に統合済み

**効果**:
- 色不一致問題の解決: 40-60%
- 位置・サイズの不変性向上

#### Object Graph + GNN ✅ 実装済み

**役割**: オブジェクト間の関係性を抽出

**実装**:
- `ObjectGraphBuilder`: グラフ構築（`src/hybrid_system/models/program_synthesis/object_graph_builder.py`）
- `ObjectGraphEncoder`: GNNによるエンコーディング（Graphormer / EGNN）
  - `src/hybrid_system/models/program_synthesis/object_graph_encoder.py`
- `NeuralObjectCandidateGenerator`に統合済み

**効果**:
- 関係性抽出の精度向上: +5-10%
- ARCの本質である関係操作に対応

#### Relation Classifier ✅ 実装済み

**役割**: 関係性を分類してプログラム生成をガイド

**実装**:
- `RelationClassifier`クラス（`src/hybrid_system/models/program_synthesis/relation_classifier.py`）
- 上下左右、対称性、同型パターン、包含関係の分類
- 訓練スクリプト: `scripts/training/train_relation_classifier.py`

**効果**:
- 関係性ベースの問題への対応強化
- プログラム候補の絞り込み

---

## 4. 統合フロー（オブジェクトマッチング統合版）

### 4.1 全体パイプライン（ProgramSynthesisEngine統合版）

```
[入力: Task (train_pairs, test_pairs)]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 0: オブジェクトマッチング（既存・必須）                │
│ ・RuleBasedObjectMatcher.match_objects()                    │
│ ・オブジェクト抽出（4連結/8連結）                           │
│ ・対応関係検出（新システムでは無効化可能）                  │
│   - enable_correspondence_detection=False で無効化          │
│   - 新システムではObject Graph + GNNとRelation Classifierで代替│
│ ・カテゴリ分類（最大926パターン）                           │
│ ・部分プログラム生成（最大930個、重複排除後200-300個）      │
│ ・出力: matching_result                                     │
│   - partial_programs: 部分プログラムリスト                  │
│   - categories: カテゴリ情報                                │
│   - objects_data: オブジェクトデータ                        │
│   - category_var_mappings: カテゴリIDと変数名の対応        │
└─────────────────────────────────────────────────────────────┘
    ↓
[各部分プログラム × 各訓練ペアの組み合わせでループ]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Meta-Reasoner（メタ推論層）                        │
│ ・Strategy Classifier: 戦略分類                             │
│ ・DSL Selector: DSL使用確率予測                             │
│ ・Module Router: モジュール選択                             │
│ ・オブジェクトマッチング結果を補助情報として活用（新規）    │
└─────────────────────────────────────────────────────────────┘
    ↓
[モジュール重みに基づいて候補生成数を調整]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: ①グリッド→プログラム Synthesizer（改善後）        │
│ ・入力: input_grid, output_grid, partial_program,           │
│        matching_result                                      │
│ ・NGPS使用（use_ngps=True）                                 │
│ ・DSL Selector連携                                          │
│ ・部分プログラムからの続き生成（既存）                      │
│ ・候補数: num_neural_candidates_with_partial / without     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: ②オブジェクト→プログラム Synthesizer（改善後）    │
│ ・入力: input_grid, output_grid, partial_program,           │
│        matching_result                                      │
│ ・Object Canonicalization（enable_canonicalization=True）   │
│ ・Object Graph + GNN（use_graph_encoder=True）              │
│ ・Relation Classifier（use_relation_classifier=True）       │
│ ・オブジェクトマッチング結果の活用（新規）                  │
│   - objects_dataからオブジェクトを取得（抽出スキップ）      │
│   - categories情報を補助特徴量として活用                    │
│   - category_var_mappingsを活用                             │
│ ・部分プログラムからの続き生成（既存）                      │
│ ・候補数: num_neural_object_candidates_with_partial / without│
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: ③グリッド→グリッド Synthesizer（補助専用）        │
│ ・Neural Mask Generator統合                                 │
│ ・特定戦略（flood-fill、局所回転など）でのみ使用           │
│ ・候補数: num_grid_to_grid_candidates                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: 候補統合（既存）                                    │
│ ・重複除去                                                  │
│ ・複雑度ランキング                                          │
│ ・上位max_candidates_per_pair個を選出                       │
└─────────────────────────────────────────────────────────────┘
    ↓
[すべての訓練ペア × 部分プログラムの組み合わせで統合]
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: プログラム検証レイヤー（新規追加）                  │
│ ・プログラムの抽象的妥当性チェック                          │
│   - 型チェック（引数の型と文脈が合うか）                    │
│   - 操作空間チェック（操作の適用可能性）                    │
│   - 境界チェック（グリッド境界を越えないか）                │
│ ・無効な候補を事前にフィルタリング                          │
│ ・`ProgramValidator`を拡張して実装                          │
│ ・`ProgramScorer`内のペナルティ項として組み込む             │
│ ・**効果**: シンボリック実行にかかる総時間を削減            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: 一貫性チェック（既存）                              │
│ ・ConsistencyChecker.check_consistency()                    │
│ ・すべての訓練ペアで一貫性を確認                            │
│ ・検証を通過した候補のみを実行                              │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 8: 最良プログラム選択（既存・改善済み）                │
│ ・一貫性スコアが1.0が1つの場合: そのまま採用                │
│ ・一貫性スコアが1.0が複数の場合: ProgramScorerで優先順位付け│
│ ・一貫性スコアが1.0がなしの場合: 統合スコアで選択          │
└─────────────────────────────────────────────────────────────┘
    ↓
[出力: 最良プログラム]
```

### 4.2 オブジェクトマッチング結果の活用方法

**matching_resultの主要フィールド**:
- `partial_programs`: 部分プログラムリスト（最大930個、重複排除後200-300個）
- `categories`: カテゴリ情報（各カテゴリのオブジェクト、特徴量など）
- `objects_data`: オブジェクトデータ（背景色オブジェクトも含む、インデックス設定済み）
- `category_var_mappings`: カテゴリIDと変数名の対応関係
- `grid_info_cache`: グリッドサイズと背景色のキャッシュ
- `partial_program_parsed_cache`: 部分プログラムの解析結果キャッシュ

**活用方法**:

1. **①グリッド→プログラム**:
   - `partial_programs`を初期トークンとして使用
   - `grid_info_cache`から背景色情報を取得
   - `partial_program_parsed_cache`から解析結果を取得

2. **②オブジェクト→プログラム**:
   - `objects_data`からオブジェクトを取得（抽出スキップ）
   - `categories`情報を補助特徴量として活用
   - `category_var_mappings`を活用して変数名を決定
   - Object Graph構築時にカテゴリ関係をエッジとして追加
   - Relation Classifierで関係性を分類（対応関係分析結果は不要）
   - **注意**: 対応関係分析結果（1対1、分割、統合など）は新システムでは**不要**
     - Object Graph + GNNが関係性を自動的に学習
     - Relation Classifierが関係性を分類
     - 対応関係分析は`enable_correspondence_detection=False`で無効化可能

### 4.2 モジュール重みに基づく調整

**実装**: `CandidateGenerator.generate_candidates()`

**処理**:
1. Meta-Reasonerからモジュール重みを取得
   - `module_weights = {'grid2program': 0.7, 'object2program': 0.2, 'grid2grid': 0.1}`

2. 各モジュールの候補生成数を調整
   - `num_neural = int(max_candidates * module_weights['grid2program'])`
   - `num_neural_object = int(max_candidates * module_weights['object2program'])`

3. 各モジュールで候補生成を実行

---

## 5. 設定パラメータ

### 5.1 ①グリッド→プログラムの設定

**ファイル**: `src/hybrid_system/inference/program_synthesis/config.py`

```python
@dataclass
class CandidateConfig:
    # 既存の設定
    max_candidates: int = 30
    enable_neural_generation: bool = True
    num_neural_candidates_with_partial: int = 20
    num_neural_candidates_without_partial: int = 20

    # 新規追加
    enable_ngps: bool = True  # NGPSを有効化
    enable_dsl_selector: bool = True  # DSL Selectorを有効化
    ngps_exploration_weight: float = 0.1  # NGPSの探索重み
    ngps_dsl_selector_weight: float = 0.3  # NGPSのDSL Selector重み
    ngps_max_search_depth: int = 10  # NGPSの最大探索深度
```

### 5.2 ②オブジェクト→プログラムの設定

```python
@dataclass
class CandidateConfig:
    # 既存の設定
    enable_neural_object_generation: bool = True
    num_neural_object_candidates_with_partial: int = 20
    num_neural_object_candidates_without_partial: int = 20

    # 新規追加
    enable_object_canonicalization: bool = True  # Object Canonicalizationを有効化
    enable_object_graph: bool = True  # Object Graph + GNNを有効化
    enable_relation_classifier: bool = True  # Relation Classifierを有効化
    graph_encoder_type: str = "graphormer"  # "graphormer" or "egnn"
    relation_classifier_threshold: float = 0.7  # Relation Classifierの閾値
```

### 5.3 Meta-Reasonerの設定

**ファイル**: `src/hybrid_system/inference/meta_reasoner/config.py`

```python
@dataclass
class MetaReasonerConfig:
    enable_strategy_classifier: bool = True
    enable_dsl_selector: bool = True
    enable_module_router: bool = True
    num_strategies: int = 40
    dsl_vocab_size: int = 100
```

---

## 6. 実装状況と優先順位

### 6.1 実装済み機能 ✅

1. **Object Graph + GNN Encoder**（②オブジェクト→プログラム）⭐
   - ✅ `ObjectGraphBuilder`クラス実装済み
   - ✅ `ObjectGraphEncoder`クラス実装済み（Graphormer / EGNN）
   - ✅ `NeuralObjectCandidateGenerator`に統合済み
   - **効果**: 対称性・位置関係に強くなる、ARCの本質である関係操作に対応

2. **Neural Guided Program Search（NGPS）**（①グリッド→プログラム）
   - ✅ `NeuralGuidedProgramSearch`クラス実装済み
   - ✅ `NeuralCandidateGenerator`に統合済み
   - **効果**: 探索空間を20-100倍縮小、精度UP

3. **DSL Selector**（①グリッド→プログラム）
   - ✅ `DSLSelector`クラス実装済み
   - ✅ NGPSと統合済み
   - **効果**: 候補の暴走を防ぎ、安定したプログラム生成

4. **Object Canonicalization**（②オブジェクト→プログラム）
   - ✅ `ObjectCanonicalizer`クラス実装済み
   - ✅ `NeuralObjectCandidateGenerator`に統合済み
   - **効果**: 背景色や色不一致問題を改善（40-60%解決）

5. **Relation Classifier**（②オブジェクト→プログラム）
   - ✅ `RelationClassifier`クラス実装済み
   - ✅ 訓練スクリプト実装済み
   - **効果**: プログラム候補の90%以上を絞り込み可能

### 6.2 Tier 1: 最重要改善ポイント（最優先・未実装）

1. **背景色・色役割分類の導入**
   - 背景色推定の改善
   - 色の役割分類（前景/背景/構造色など）
   - Neural Mask Generatorと統合

2. **Object relation特徴量の追加**
   - オブジェクト間の関係性特徴量
   - 位置関係、包含関係、接触関係など
   - Object Graph構築時に活用（一部実装済み）

3. **Program Decoderを構文木ベースにする**
   - 構文制約を考慮したプログラム生成
   - より正確なプログラム生成
   - ビームサーチと統合

### 6.3 Tier 2: 中優先度改善（未実装）

1. **プログラム検証レイヤーの強化**（全モジュール）
   - プログラムの抽象的妥当性チェックの導入
   - **型チェック**: 引数の型と文脈が合うか
   - **操作空間チェック**: 操作の適用可能性
   - **境界チェック**: 移動・回転操作がグリッドの境界を越えないか
   - `ProgramScorer`内のペナルティ項として組み込む
   - **効果**: シンボリック実行にかかる総時間を削減

2. **Neural Mask Generatorの統合**（③グリッド→グリッド、補助専用）
   - `NeuralMaskGenerator`クラスの実装
   - `GridToGridCandidateGenerator`への統合
   - **効果**: プログラム探索前処理として暴走防止、探索効率UP

3. **Cross-Attention between Input/Outputの強化**
   - Transformerのcross-attentionを使用して変換パターンを明示的に学習
   - ①グリッド→プログラム、②オブジェクト→プログラム
   - 実装が容易で効果が高い

### 6.4 Tier 3: 補助改善（将来的に実装）

1. **Contrastive Pretraining**（全モジュール）
   - モデルの事前学習段階で実装
   - **効果**: 少量データでも正答率向上

2. **Abstract Object Patterns**（②オブジェクト→プログラム）
   - 色無視/相対座標/サイズだけ抽象化
   - **効果**: パターン理解が堅牢になる

3. **ViT EncoderでGrid理解を強化**（①グリッド→プログラム）
   - Vision Transformerを使用したグリッドエンコーディング

4. **Symmetry-Aware Augmentation**
   - 回転不変性、反転不変性、スケール不変性
   - Object Canonicalizationと相補的

5. **その他の補助改善**
   - データAugmentationで学習データを増強
   - 部分プログラムをslot-basedで扱う
   - Beam Search + Syntax-guided completion
   - object-level pretraining
   - color permutation augmentation
   - 深層モデルアンサンブル

---

## 7. 期待される改善効果

### 7.1 ①グリッド→プログラム

**現在の性能**:
- 基本的なビームサーチ
- DSL Selectorなし
- 探索空間が膨大

**改善後の期待性能**:
- **探索時間の削減: 1000倍**（NGPS）⭐
- **精度向上: +10-20%**（NGPS + DSL Selector）
- 探索の暴走を防止（DSL Selector）
- 少量データでも正答率向上（Contrastive Pretraining）

### 7.2 ②オブジェクト→プログラム

**現在の性能**:
- 基本的なオブジェクトエンコーディング
- 関係性抽出が弱い
- 色不一致問題が頻発

**改善後の期待性能**:
- **関係性抽出の精度向上: +10-20%**（Object Graph + GNN）⭐ **最重要**
- **色不一致問題の解決: 40-60%**（Object Canonicalization）
- **プログラム候補の90%以上を絞り込み可能**（Relation Classifier）
- パターン理解が堅牢になる（Abstract Object Patterns）

### 7.3 ③グリッド→グリッド（補助専用）

**現在の性能**:
- 出力を直接生成（主力として使用）
- 1マスのズレが頻発

**改善後の期待性能**:
- **補助専用として使用**（Neural Mask Generator）
- プログラム探索前処理として暴走防止
- 探索効率UP
- 特定戦略（flood-fill、局所回転など）でのみ使用

### 7.4 全体の期待効果

**現在の性能**: 3.00% (3/100タスク)

**改善後の期待性能**:
- **Tier 1完了後: 10-15%** (+7-12%)
- **Tier 2完了後: 15-25%** (+12-22%)
- **Tier 3完了後: 25-35%** (+22-32%)

---

## 8. Kaggleコンペ制限への対応

### 8.1 制限事項の確認

**メモリ使用量**:
- 現在: 2-4GB（GPU）+ 1-2GB（CPU）
- 改善後: 2.3-5GB（GPU）+ 1-2GB（CPU）
- **Kaggle制限**: 16-32GB（GPU）+ 8-16GB（CPU）→ ✅ **問題なし**

**実行時間**:
- 現在: 20-55秒/タスク
- 改善後: 16-43秒/タスク（NGPSで短縮）
- **Kaggle制限**: 数秒〜数十秒/タスク → ⚠️ **やや長い（最適化が必要）**

**ファイルサイズ**:
- 現在: 300-1500MB
- 改善後: 570-2500MB
- **Kaggle制限**: 数百MB〜数GB → ⚠️ **やや大きい（最適化が必要）**

### 8.2 推奨される最適化

1. **メモリ使用量の最適化**
   - モデルの遅延ロード（必要な時だけロード）
   - 部分プログラム数の制限（最大50個）
   - 並列処理の無効化（メモリ節約）

2. **実行時間の最適化**
   - 早期終了（一貫性スコア1.0の候補が見つかったら終了）
   - キャッシュの活用
   - 候補生成数の削減（10 → 5）

3. **ファイルサイズの最適化**
   - モデルの量子化（INT8）
   - モデルの統合（マルチタスク学習）
   - 選択的ロード（必要なモデルのみ）

### 8.3 追加提案の評価

**採用すべき提案**:
- ✅ **Cross-Attention between Input/Outputの強化**（Tier 2）
  - 実装が容易で効果が高い
  - 計算コストもそれほど高くない
  - Transformerの標準的な機能

- ✅ **Symmetry-Aware Augmentation**（Tier 3）
  - Object Canonicalizationと相補的
  - 実装も比較的容易
  - 効果も期待できる

**検討が必要な提案**:
- ⚠️ **Diffusion-based Program Synthesis**
  - 最新研究では効果があるとされている
  - ただし、計算コストが高い（Kaggleの実行時間制限を考慮）
  - 実装の複雑さも高い
  - **判断**: 現時点では導入を見送り（将来的に検討）

- ⚠️ **Program Sketch + Neural Completion**
  - 2段階アプローチは有効
  - ただし、既存の部分プログラム生成と重複する可能性
  - **判断**: 既存の部分プログラム生成を改善する方向で検討

- ❌ **Few-shot In-Context Learning**
  - 大規模言語モデルが必要
  - Kaggleではモデルサイズの制限がある
  - 実装の複雑さが高い
  - **判断**: 導入を見送り

- ❌ **Curriculum Learning**
  - これは学習段階での改善
  - 推論時には影響しない
  - **判断**: 推論パイプラインのドキュメントには含めない

- ✅ **Program Verifier**
  - 既に「プログラム検証レイヤー」として提案済み
  - 重複しているため、追加不要

詳細は [Kaggleコンペ制限への対応](./kaggle_competition_constraints.md) を参照してください。

---

## 9. 学習データ生成

### 9.1 必要な学習データ

パイプラインの各モジュールを学習するために、以下の学習データが必要です：

#### ①グリッド→プログラムパイプライン

**NGPS/DSL Selector用**:
- **形式**: JSONL
- **内容**: グリッド特徴量 + DSL使用確率分布
- **生成スクリプト**: `scripts/data_generation/generate_ngps_training_data.py`

**プログラム生成モデル用**:
- **形式**: JSONL
- **内容**: グリッドペア + プログラムコード + トークン列
- **生成方法**: generatorの出力から直接生成可能

#### ②オブジェクト→プログラムパイプライン

**Object Graph + GNN用**:
- **形式**: JSONL
- **内容**: オブジェクトグラフ特徴量 + プログラムコード
- **生成スクリプト**: `scripts/data_generation/generate_object_graph_training_data.py`

**Relation Classifier用**:
- **形式**: JSONL
- **内容**: オブジェクトペア特徴量 + 関係性ラベル
- **生成スクリプト**: `scripts/data_generation/generate_relation_classifier_data.py`

**Object Canonicalization用**:
- **形式**: JSONL
- **内容**: 正規化前後のオブジェクト特徴量
- **生成方法**: generatorの出力から直接生成可能

### 9.2 generatorからの学習データ生成

**現状**:
- generatorはプログラムコード、グリッドデータ、統計情報を生成
- ニューラルモデル用の学習データ（JSONL形式）は別途生成が必要
- 既存の学習データ生成スクリプトはARC-AGI2の訓練データを読み込む必要がある

**改善提案**:
- generatorの出力から直接学習データを生成する機能を追加
- 2段階処理（generator → 学習データ生成スクリプト）を1ステップに統合
- バッチ処理に対応し、メモリ効率を向上
- 詳細は [拡張データセット生成パイプラインの改善提案](../data_generation/generator_improvements_for_neural_pipeline.md) を参照

**実装優先順位**:
1. **Tier 1（最優先）**:
   - NGPS/DSL Selector用データ生成
   - Object Graph + GNN用データ生成
   - Relation Classifier用データ生成
2. **Tier 2（中優先度）**:
   - Object Canonicalization用データ生成
   - プログラム生成モデル用データ生成

**期待される効果**:
- **効率化**: 2段階処理の統合により処理時間を短縮
- **データ品質**: generatorで生成したデータと学習データの一貫性が保証
- **メンテナンス性**: コードの統合によりメンテナンスが容易に

---

## 10. 参考資料

- [改善提案：アーキテクチャ設計](./improvement_proposal_architecture.md)
- [改善提案：具体的実装設計](./improvement_proposal_implementation.md)
- [改善提案：Neural Generators](./improvement_proposal_neural_generators.md)
- [深層学習ベース候補生成手法の詳細仕様](./neural_candidate_generation_methods.md)
- [Kaggleコンペ制限への対応](./kaggle_competition_constraints.md)
- [拡張データセット生成パイプラインの改善提案](../data_generation/generator_improvements_for_neural_pipeline.md)

# 将来実装モジュール

このフォルダーには、実装済みだが現在使用されていない機能が含まれています。

## 含まれるモジュール

### 1. AppearancePatternPredictor（新規出現パターン予測）

**場所**: `appearance_pattern_predictor/appearance_pattern_predictor.py`

**機能**: 新規出現オブジェクトの特徴を分析し、予測する機能

**状態**: 実装済みだが、`RuleBasedObjectMatcher`や`ProgramSynthesisEngine`に統合されていない

**将来の統合方法**:
- `RuleBasedObjectMatcher`で新規出現パターンを分析
- `CandidateGenerator`で予測結果を活用してCREATE系コマンドの優先探索を実装

---

### 2. ObjectLevelPretraining（オブジェクトレベル事前学習）

**場所**: `object_level_pretraining/`

**機能**: オブジェクトエンコーダー、グラフエンコーダー、関係分類器の事前学習

**状態**: 実装済みだが、学習スクリプトが存在しない

**将来の統合方法**:
- `scripts/training/train_object_level_pretraining.py`を作成
- オブジェクトベースモデルの精度向上に活用

---

### 3. ViTGridEncoder（ViT Encoder）

**場所**: `vit_grid_encoder/vit_grid_encoder.py`

**機能**: Vision Transformerアーキテクチャを使用したグリッドエンコーダー

**状態**: 実装済みだが、`ProgramSynthesisModel`で使用されていない（現在は`GridEncoder`を使用）

**将来の統合方法**:
- `ProgramSynthesisModel`に`use_vit_encoder`フラグを追加（オプション）
- パッチベースの理解が有効なタスクで使用

---

### 4. SlotBasedPartialProgramHandler（Slot-based部分プログラム）

**場所**: `slot_based_partial_program/slot_based_partial_program.py`

**機能**: 部分プログラムをスロット（プレースホルダー）として扱う機能

**状態**: 実装済みだが、`CandidateGenerator`や`ProgramSynthesisEngine`に統合されていない

**将来の統合方法**:
- `CandidateGenerator`で部分プログラムのスロット化と値の埋め込みを実装
- 部分プログラムの柔軟性向上に活用

---

## 使用方法

これらのモジュールを使用する場合は、以下のようにインポートしてください：

```python
# AppearancePatternPredictor
from src.future_implementations.appearance_pattern_predictor.appearance_pattern_predictor import (
    AppearancePatternPredictor,
    AppearancePattern
)

# ObjectLevelPretraining
from src.future_implementations.object_level_pretraining import (
    ObjectLevelPretrainer,
    ObjectPretrainingConfig
)

# ViTGridEncoder
from src.future_implementations.vit_grid_encoder.vit_grid_encoder import (
    ViTGridEncoder,
    PatchEmbedding
)

# SlotBasedPartialProgramHandler
from src.future_implementations.slot_based_partial_program.slot_based_partial_program import (
    SlotBasedPartialProgramHandler,
    Slot,
    SlotType
)
```

---

## 注意事項

- これらのモジュールは実装済みですが、メインパイプラインに統合されていません
- 統合する場合は、適切なテストと検証が必要です
- インポートパスが変更されているため、既存のコードからは使用できません

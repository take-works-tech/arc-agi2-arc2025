# クイックスタートガイド

## 実装モジュールの動作確認

### 1. インポート検証

すべてのモジュールが正常にインポートできることを確認：

```bash
python scripts/verify_implementations.py
```

**期待結果**: 20/20 (100.0%) すべてのモジュールが正常にインポートできました

### 2. 統合テスト実行

各モジュールの基本的な動作を確認：

```bash
python -m pytest tests/test_integration_new_modules.py -v
```

**期待結果**: 12 passed

## 主要モジュールの使用

### Object Graph + GNN

```python
from src.hybrid_system.models.program_synthesis.object_graph_builder import ObjectGraphBuilder
from src.hybrid_system.models.program_synthesis.object_graph_encoder import GraphormerEncoder

builder = ObjectGraphBuilder()
graph = builder.build_graph(objects)

encoder = GraphormerEncoder(node_feature_dim=12, embed_dim=256)
encoded, _ = encoder(graph)
```

### NGPS（Neural Guided Program Search）

```python
from src.hybrid_system.models.program_synthesis.neural_guided_program_search import NeuralGuidedProgramSearch

ngps = NeuralGuidedProgramSearch(dsl_selector=dsl_selector)
candidates = ngps.search(context=context, max_depth=10, beam_width=5)
```

### プログラム検証レイヤー

```python
from src.hybrid_system.utils.validation.enhanced_program_validator import EnhancedProgramValidator

validator = EnhancedProgramValidator(grid_size=(10, 10))
result = validator.validate_program_enhanced(program, task)
# result.validation_penalty が検証ペナルティ（0.0-1.0）
```

## 設定

### Cross-Attention融合の有効化

`configs/synthesis_config.yaml`またはコード内で：

```python
from src.hybrid_system.models.base.model_config import ProgramSynthesisConfig

config = ProgramSynthesisConfig(
    use_cross_attention_fusion=True,  # Cross-Attention融合を有効化
    cross_attention_layers=2
)
```

## 詳細情報

- **パイプライン詳細**: `design/improved_neural_generators_pipeline.md`
- **使用例**: `guides/usage_examples.md`
- **実装状況**: `status/IMPLEMENTATION_STATUS.md`

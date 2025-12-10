# ルールベース候補生成アルゴリズム

## 概要

ルールベース候補生成器（`RuleBasedCandidateGenerator`）は、入力グリッドと出力グリッドの比較に基づいて、ヒューリスティックなルールを使用して候補プログラムを生成します。**部分プログラムが存在することを前提**としており、部分プログラムがない場合は候補を生成しません。オブジェクトマッチング結果や部分プログラムを活用し、ARCタスクで頻出する変換パターンを検出してプログラムを生成します。

## 重要な前提条件

- **部分プログラムが必須**: 部分プログラムがない場合、ルールベース候補生成は使用できません（空のリストを返す）
- **オブジェクトマッチング結果の活用**: `matching_result`からカテゴリ情報やカテゴリIDと変数名の対応関係（`category_var_mappings`）を取得可能

## アルゴリズムの全体フロー

```
1. 部分プログラムの存在確認
   └─ 部分プログラムがない場合 → 空のリストを返して終了
   ↓
2. 部分プログラムの優先処理
   ↓
3. オブジェクトマッチング結果の活用
   ↓
4. オブジェクトベースのパターン検出
   ↓
5. 基本的な変換パターンの検出
   ↓
6. ARC頻出パターンの検出
   ↓
7. 候補プログラムの生成
```

## 詳細な処理フロー

### 1. 部分プログラムの存在確認

**重要**: ルールベース候補生成は、部分プログラムが存在することを前提としています。

```python
# ① 部分プログラムが存在することを前提とする
# 部分プログラムがない場合は、ルールベースは使用できない
if not partial_program:
    return []
```

部分プログラムがない場合、空のリストを返して処理を終了します。

### 2. 部分プログラムの優先処理

部分プログラムが提供されている場合、それを優先的に使用して候補を生成します。

#### 1.1 部分プログラムの解析

- `parse_partial_program()`を使用して部分プログラムを解析
- カテゴリ分けを含むかどうかを確認（`has_categories`）
- パターンタイプ（`color_change`, `position_change`, `rotation`など）を識別

#### 2.2 カテゴリ分けを含む部分プログラムからの候補生成

カテゴリ分けを含む部分プログラムがある場合：

1. **色変更パターン**の検出
   - `_is_color_change_transformation()`で色変更を検出
   - `infer_target_color_candidates_from_output()`で複数の色候補を生成
   - 各部分プログラムに対して`extend_partial_program()`で変換コードを追加

2. **移動パターン**の検出
   - `_is_movement_transformation()`で移動を検出
   - `estimate_movement_parameters()`で移動量候補を推定
   - 各部分プログラムに対して移動コードを追加

3. **回転パターン**の検出
   - `_is_rotation_transformation()`で回転を検出
   - 90度、180度、270度の回転角度を試行
   - 各部分プログラムに対して回転コードを追加

カテゴリ分けを含む部分プログラムから候補が生成された場合、他のパターン検出をスキップして早期リターンします。

### 3. オブジェクトマッチング結果の活用

オブジェクトマッチング結果（`matching_result`）が提供されている場合、カテゴリ情報から候補を生成します。

#### 3.1 カテゴリ情報からの候補生成

- `matching_result.get('categories')`からカテゴリリストを取得
- 各カテゴリの`transformation_patterns`を確認
- 各変換パターンに対して`_generate_program_from_category_pattern()`でプログラムを生成

#### 3.2 カテゴリIDと変数名の対応関係

オブジェクトマッチング結果には、`category_var_mappings`が含まれています：

```python
category_var_mappings = matching_result.get('category_var_mappings', {})
# 形式: {pattern_key: {category_id: variable_name}}
# 例: {"4_0": {"4_0_0": "objects1", "4_0_1": "objects2"}}
```

この情報により、どのカテゴリがどの変数名に代入されたかを把握できます。これにより、カテゴリ固有の変換を正確に適用できます。

### 4. オブジェクトベースのパターン検出

オブジェクトレベルでの対応関係を分析し、変換パターンを検出します。

#### 4.1 オブジェクト抽出

- 4連結と8連結の両方でオブジェクトを抽出
- `GET_ALL_OBJECTS(4)`と`GET_ALL_OBJECTS(8)`を実行

#### 4.2 オブジェクト間の対応関係検出

`_find_object_correspondences()`で入力オブジェクトと出力オブジェクトの対応関係を検出：

1. **類似度行列の計算**
   - 各入力オブジェクトと出力オブジェクトのペアについて、以下の特徴量で類似度を計算：
     - **位置の類似度**: 中心点間のユークリッド距離
     - **サイズの類似度**: サイズの差分
     - **色の類似度**: 色の一致/不一致
     - **形状の類似度**: アスペクト比の差分
   - 重み付き総合類似度スコアを計算：
     - 位置: 0.3
     - サイズ: 0.3
     - 色: 0.2
     - 形状: 0.2

2. **貪欲法によるマッチング**
   - 類似度が高い順にペアをマッチング
   - 閾値（0.3）以上の類似度を持つペアのみを対応関係として記録

#### 4.3 対応関係からの変換パターン分析

`_analyze_transformation_from_correspondence()`で対応関係から変換パターンを分析：

1. **パラメータ推定**
   - 移動量（dx, dy）: 中心点の差分
   - 回転角度: `_estimate_rotation_angle()`で推定（90度、180度、270度）
   - スケール倍率: `estimate_scale_factor()`で推定

2. **変換パターンの決定**
   - **色変更**: 色が異なり、位置・サイズがほぼ同じ
   - **位置変更**: 位置が異なり、サイズがほぼ同じ
   - **サイズ変更**: スケール倍率が1.0から大きく異なる
   - **回転**: 回転角度が検出された
   - **色変更+位置変更**: 色と位置の両方が異なる

#### 3.4 オブジェクトパターンからのプログラム生成

検出されたパターンから`_generate_program_from_object_pattern()`でプログラムを生成：

- 部分プログラムがある場合: 部分プログラムを拡張
- 部分プログラムがない場合: 完全なプログラムを生成
  - `GET_ALL_OBJECTS(4)`でオブジェクト取得
  - 変換コード（`FOR`ループ内で変換を適用）
  - `RENDER_GRID()`でグリッドにレンダリング

### 5. 基本的な変換パターンの検出

グリッド全体の比較に基づいて基本的な変換パターンを検出します。

#### 5.1 恒等変換

- `_is_identity_transformation()`: 入力グリッドと出力グリッドが完全に一致
- プログラム: `GET_ALL_OBJECTS(4)`

#### 4.2 色変更変換

- `_is_color_change_transformation()`: パターンアナライザーを使用
- 複数の色候補を生成: `infer_target_color_candidates_from_output()`
- 部分プログラムがある場合: 部分プログラムを拡張
- 部分プログラムがない場合: 完全なプログラムを生成
  ```
  objects = GET_ALL_OBJECTS(4)
  FOR i LEN(objects) DO
      objects[i] = SET_COLOR(objects[i], {target_color})
  END
  ```

#### 4.3 移動変換

- `_is_movement_transformation()`: パターンアナライザーを使用
- 移動量を推定: `estimate_movement_parameters()`（グリッド比較による推定）
- 移動量候補がない場合: 固定値の候補（-2～2）を試行
- 部分プログラムがある場合: 部分プログラムを拡張
- 部分プログラムがない場合: 完全なプログラムを生成
  ```
  objects = GET_ALL_OBJECTS(4)
  FOR i LEN(objects) DO
      objects[i] = MOVE(objects[i], {dx}, {dy})
  END
  ```

#### 4.4 回転変換

- `_is_rotation_transformation()`: グリッドの形状変化を確認
- 回転角度: 90度、180度、270度を試行
- 部分プログラムがある場合: 部分プログラムを拡張
- 部分プログラムがない場合: 完全なプログラムを生成
  ```
  objects = GET_ALL_OBJECTS(4)
  FOR i LEN(objects) DO
      objects[i] = ROTATE(objects[i], {angle})
  END
  ```

### 6. ARC頻出パターンの検出

ARCタスクで頻出する特殊なパターンを検出します。

#### 6.1 サイズフィルタリングパターン

- `_has_size_filtering_pattern()`: 出力の非ゼロピクセル数が入力より少ない
- 最大オブジェクトのみを抽出するプログラムを生成
  ```
  objects = GET_ALL_OBJECTS(4)
  max_obj = None
  max_size = 0
  FOR i LEN(objects) DO
      size = MULTIPLY(GET_WIDTH(objects[i]), GET_HEIGHT(objects[i]))
      IF GREATER(size, max_size) THEN
          max_size = size
          max_obj = objects[i]
      END
  END
  IF NOT_EQUAL(max_obj, None) THEN
      objects = [max_obj]
  ELSE
      objects = []
  END
  ```

#### 5.2 枠線描画パターン

- `_is_border_pattern()`: 出力の非ゼロピクセル数が入力より多い
- 複数の枠線色候補を生成: `_infer_border_color_candidates()`
- 各色候補に対して`OUTLINE()`を使用したプログラムを生成
  ```
  objects = GET_ALL_OBJECTS(4)
  FOR i LEN(objects) DO
      objects[i] = OUTLINE(objects[i], {border_color})
  END
  ```

#### 5.3 特定色のオブジェクトのみ抽出

- `_has_color_filtering_pattern()`: 出力の色数が入力より少ない
- `infer_target_color_from_output()`で対象色を推定
- フィルタリングプログラムを生成
  ```
  objects = GET_ALL_OBJECTS(4)
  filtered = []
  FOR i LEN(objects) DO
      IF EQUAL(GET_COLOR(objects[i]), {target_color}) THEN
          filtered = APPEND(filtered, objects[i])
      END
  END
  objects = filtered
  ```

#### 5.4 最大オブジェクトのコピー＆中央配置

- `_has_copy_center_pattern()`: 出力が単一オブジェクトで中央配置
- 最大オブジェクトを検出し、中央に配置するプログラムを生成

#### 5.5 その他の頻出パターン

- **整列・並べ替えパターン**: `_has_arrangement_pattern()`
- **水平反射パターン**: `_has_horizontal_flip_pattern()`
- **垂直反射パターン**: `_has_vertical_flip_pattern()`
- **オブジェクト統合パターン**: `_has_merge_pattern()`
- **オブジェクト分割パターン**: `_has_split_pattern()`
- **スケールダウンパターン**: `_has_scale_down_pattern()`
- **バウンディングボックス抽出パターン**: `_has_bounding_box_extraction_pattern()`
- **色数減少パターン**: `_has_color_reduction_pattern()`
- **オブジェクトサイズフィルタリングパターン**: `_has_object_size_filtering_pattern()`

### 7. 候補プログラムの生成

検出されたパターンからDSLプログラムを生成します。

#### 7.1 プログラム構造

基本的なプログラム構造：

```
1. オブジェクト取得: GET_ALL_OBJECTS(4) または 部分プログラム
2. 変換処理: FORループ内で各オブジェクトに変換を適用
3. グリッドレンダリング: RENDER_GRID(objects, bg_color, width, height)
```

#### 6.2 パラメータ推定

- **色**: `infer_target_color_candidates_from_output()`で複数候補を生成
- **移動量**: `estimate_movement_parameters()`でグリッド比較により推定
- **スケール倍率**: `estimate_scale_factor()`でサイズ比較により推定
- **背景色**: `get_background_color()`で出力グリッドから推定

#### 6.3 部分プログラムの拡張

部分プログラムがある場合、`extend_partial_program()`を使用：

1. 部分プログラムを解析
2. 変換コードを生成
3. パラメータを改善（推定値を使用）
4. 部分プログラムに変換コードを追加
5. `RENDER_GRID()`を追加

## 主要な特徴

### 優先順位

1. **部分プログラムの存在確認**: 部分プログラムがない場合は処理を終了
2. **部分プログラム（カテゴリ分け含む）**: 最優先
3. **オブジェクトマッチング結果**: 次に優先（カテゴリ情報と`category_var_mappings`を活用）
4. **オブジェクトベースパターン**: オブジェクト対応関係から検出
5. **基本的な変換パターン**: グリッド全体の比較
6. **ARC頻出パターン**: 特殊なパターン

### パラメータ推定の改善

- **複数候補の生成**: 色や移動量など、複数の候補を生成して試行
- **グリッド比較による推定**: 単純な差分ではなく、グリッド全体の比較により精度向上
- **共通実装の活用**: `common_helpers.py`の共通関数を使用

### 部分プログラムの活用

- オブジェクトマッチングで生成された部分プログラムを優先的に使用
- カテゴリ分けを含む部分プログラムがある場合、他のパターン検出をスキップ
- 部分プログラムのパラメータを改善して候補を生成

## 制限事項と改善の余地

### 現在の制限

1. **部分プログラムが必須**: 部分プログラムがない場合は候補を生成できない
2. **パターン検出の順序**: 最初に検出されたパターンが優先される
3. **パラメータ推定の精度**: 複雑な変換では推定が困難
4. **組み合わせパターン**: 複数の変換の組み合わせに対応が限定的
5. **候補数の上限**: `config.num_rule_based_candidates`で制限

### 改善の余地

1. **パターン優先度の動的調整**: 検出されたパターンの信頼度に基づいて優先度を調整
2. **パラメータ推定の改善**: より高度な推定手法の導入
3. **組み合わせパターンの拡充**: 複数の変換を組み合わせたパターンの検出
4. **候補生成の最適化**: より効率的な候補生成アルゴリズム

## ルールベース②候補生成アルゴリズム

### 概要

ルールベース②候補生成器（`RuleBased2CandidateGenerator`）は、オブジェクトマッチング結果から変換パターン情報を活用して部分プログラムを生成します。ルールベース①とは異なり、**変換パターン情報の統計的分析**に基づいて、カテゴリ内で80%以上のオブジェクトが同じ変換パラメータを持つ場合にコマンドを生成します。

### 重要な前提条件

- **部分プログラムが必須**: 部分プログラムがない場合、ルールベース②は使用できません（空のリストを返す）
- **オブジェクトマッチング結果が必須**: `matching_result`が提供され、`success`が`True`である必要がある
- **カテゴリ情報が必要**: `matching_result`に`categories`が含まれている必要がある
- **変換パターン情報が必要**: 各オブジェクトに`transformation_pattern`が設定されている必要がある

### アルゴリズムの全体フロー

```
1. 部分プログラムの存在確認
   └─ 部分プログラムがない場合 → 空のリストを返して終了
   ↓
2. オブジェクトマッチング結果の確認
   └─ matching_resultがない、またはsuccess=False → 空のリストを返して終了
   ↓
3. カテゴリ情報の取得
   └─ categoriesがない → 空のリストを返して終了
   ↓
4. 部分プログラムの解析
   └─ category_vars（カテゴリIDと変数名の対応関係）を取得
   ↓
5. 各カテゴリについてループ
   ├─ 対応関係にあるオブジェクト情報を持つオブジェクトを取得
   ├─ 変換パターン情報を取得
   ├─ disappearanceが80%以上Trueの場合 → continue
   ├─ 各変換パラメータについて80%以上の一致率をチェック
   │  ├─ output_color → SET_COLOR
   │  ├─ rotation_index → ROTATE
   │  ├─ flip_type → FLIP
   │  ├─ scale_factor → SCALE/SCALE_DOWN
   │  ├─ has_outline → OUTLINE
   │  ├─ has_expand → EXPAND
   │  ├─ has_fill_holes → FILL_HOLES
   │  ├─ has_hollow → HOLLOW
   │  ├─ has_BBOX → BBOX
   │  └─ dx/dy → MOVE/SLIDE（特殊な場合分け）
   ├─ FORループを生成
   └─ result_objectsに追加
   ↓
6. RENDER_GRIDを追加
   ↓
7. 候補プログラムを返す
```

### 詳細な処理フロー

#### 1. 変換パラメータの抽出

各カテゴリ内のオブジェクトの変換パターン情報から、以下のパラメータを抽出します：

- **output_color**: 変換先の色（0-9の範囲）
- **rotation_index**: 回転インデックス（0: 0度, 1: 90度, 2: 180度, 3: 270度）
- **flip_type**: 反転タイプ（None, "X", "Y"）
- **scale_factor**: スケール倍率（1.0: スケールなし、2.0: 2倍、0.5: 1/2倍など）
- **has_outline**: OUTLINEが適用されているか
- **has_expand**: EXPANDが適用されているか
- **expand_pixels**: EXPANDのピクセル数
- **has_fill_holes**: FILL_HOLESが適用されているか
- **has_hollow**: HOLLOWが適用されているか
- **has_BBOX**: BBOXが適用されているか
- **dx**: X方向の移動量
- **dy**: Y方向の移動量

#### 2. 80%一致率チェック

各パラメータについて、カテゴリ内のオブジェクトの80%以上が同じ値を持つ場合、その変換コマンドを生成します。

```python
def _is_80_percent_same(self, values: List[Any]) -> bool:
    """値のリストが80%以上同じ値かチェック"""
    if not values:
        return False
    counter = Counter(values)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count >= len(values) * 0.8
```

#### 3. 変換コマンドの生成

80%以上の一致率が確認されたパラメータについて、対応するコマンドを生成します：

- **output_color**: `SET_COLOR(objects[i], {color})`
- **rotation_index**: `ROTATE(objects[i], {angle})`（angle = rotation_index * 90）
- **flip_type**: `FLIP(objects[i], "{flip_type}")`
- **scale_factor**:
  - `scale_factor > 1.0`: `SCALE(objects[i], {scale_int})`
  - `scale_factor < 1.0`: `SCALE_DOWN(objects[i], {scale_int})`
- **has_outline**: `OUTLINE(objects[i], {outline_color})`
- **has_expand**: `EXPAND(objects[i], {expand_pixels})`
- **has_fill_holes**: `FILL_HOLES(objects[i])`
- **has_hollow**: `HOLLOW(objects[i])`
- **has_BBOX**: `BBOX(objects[i])`

#### 4. dx/dyの特殊な場合分け

`dx`と`dy`は特殊な場合分けで処理されます：

1. **①dxとdyが80%以上同じ値で、かつ0でない場合**
   - `MOVE(objects[i], {dx}, {dy})`

2. **②dxが80%以上同じ値で、かつ0でない場合**
   - `MOVE(objects[i], {dx}, 0)`
   - `dy`の値が80%以上が同じ符号の場合、`SLIDE(objects[i], "{direction}", obstacles)`も追加

3. **③dyが80%以上同じ値で、かつ0でない場合**
   - `MOVE(objects[i], 0, {dy})`
   - `dx`の値が80%以上が同じ符号の場合、`SLIDE(objects[i], "{direction}", obstacles)`も追加

4. **④それ以外**
   - `dx`の値が80%以上が同じ符号の場合、`SLIDE(objects[i], "{direction}", obstacles)`
   - `dy`の値が80%以上が同じ符号の場合、`SLIDE(objects[i], "{direction}", obstacles)`

#### 5. 符号の一致チェック

`dx`/`dy`の符号（正、負、0）が80%以上一致する場合、`SLIDE`コマンドを生成します。

```python
def _is_80_percent_same_sign(self, values: List[Any]) -> bool:
    """値のリストが80%以上が同じ符号（正負（0を含める））かチェック"""
    if not values:
        return False
    signs = [1 if v > 0 else (-1 if v < 0 else 0) for v in values]
    counter = Counter(signs)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count >= len(values) * 0.8
```

#### 6. プログラム構造

生成されるプログラムの構造：

```
result_objects = []
FOR i LEN({category_var_name}) DO
    {category_var_name}[i] = SET_COLOR({category_var_name}[i], {color})
    {category_var_name}[i] = ROTATE({category_var_name}[i], {angle})
    ...（他の変換コマンド）
END
result_objects = CONCAT(result_objects, {category_var_name})
...（他のカテゴリ）
RENDER_GRID(result_objects, {bg_color}, {output_w}, {output_h})
```

### 主要な特徴

#### 統計的分析に基づく変換抽出

- カテゴリ内のオブジェクトの80%以上が同じ変換パラメータを持つ場合のみコマンドを生成
- ノイズに強い（一部のオブジェクトが異なる変換を受けていても、大多数が同じ変換なら適用）

#### オブジェクトマッチング結果の活用

- オブジェクトマッチングで検出された変換パターン情報を直接活用
- カテゴリIDと変数名の対応関係（`category_var_mappings`）を使用して、正確な変数名でコマンドを生成

#### 複数の変換の組み合わせ

- 複数の変換パラメータが80%以上の一致率を持つ場合、すべての変換コマンドを生成
- 例: 色変更 + 回転 + 移動を同時に適用

### 制限事項と改善の余地

#### 現在の制限

1. **80%の閾値が固定**: すべてのパラメータで80%の閾値を使用
2. **disappearanceの処理**: disappearanceが80%以上Trueの場合、そのカテゴリをスキップ
3. **obstaclesの選択**: `SLIDE`コマンドの`obstacles`パラメータは、他のカテゴリのオブジェクト配列から選択（ランダム）

#### 改善の余地

1. **閾値の動的調整**: パラメータごとに異なる閾値を使用
2. **obstaclesの最適化**: `SLIDE`コマンドの`obstacles`をより適切に選択
3. **変換の順序**: 複数の変換を適用する場合の順序の最適化

## 関連ファイル

- `src/hybrid_system/inference/program_synthesis/candidate_generators/rule_based_generator.py`: ルールベース①のメイン実装
- `src/hybrid_system/inference/program_synthesis/candidate_generators/rule_based_2_generator.py`: ルールベース②のメイン実装
- `src/hybrid_system/inference/program_synthesis/candidate_generators/common_helpers.py`: 共通ヘルパー関数
- `src/hybrid_system/inference/program_synthesis/candidate_generators/pattern_analyzer.py`: パターンアナライザー
- `src/hybrid_system/inference/program_synthesis/config.py`: 設定（`CandidateConfig`）

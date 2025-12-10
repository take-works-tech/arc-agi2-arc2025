# オブジェクトマッチング設計書（深層学習用部分プログラム生成）

## 📋 概要

オブジェクトマッチングは、ルールベースでオブジェクトの役割分類を行い、深層学習モデルに渡す部分プログラムを生成する機能です。現在は**ルールベースオブジェクトマッチング**（`RuleBasedObjectMatcher`）のみが実装されています。

### 目的

- **オブジェクトの役割分類**: 色、形状、位置などの特徴に基づいてオブジェクトをカテゴリ分け
- **部分プログラム生成**: カテゴリ分けに基づいて、複数のパターンの部分プログラムを生成
- **深層学習への活用**: 生成された部分プログラムを深層学習モデルの初期状態として利用

### 実装状況（最新）

- ✅ **オブジェクトマッチング統合完了**: `ProgramSynthesisEngine`に統合済み
- ✅ **オブジェクトベースのパターン検出のバグ修正完了**: 早期リターンを削除し、オブジェクトマッチング結果があってもオブジェクトベースのパターン検出を実行
- ✅ **オブジェクト対応関係検出の本格実装完了**: 位置、サイズ、色、形状の多特徴量マッチングを実装
- ✅ **カテゴリIDと変数名の対応関係の保存**: `category_var_mappings`を`matching_result`に含める
- ✅ **部分プログラム生成の改善**: カテゴリIDと変数名の対応関係を返すように変更
- ✅ **3段階構造の廃止と1段階構造への変更**: すべての特徴量をフラットなリストとして扱い、`available_features`で使用する特徴量を指定可能に
- ✅ **効果**: 候補数が増加、部分プログラムの活用が改善、効率的な候補生成が可能に

---

## 🎯 設計方針

### 基本方針

1. **ルールベース**: 深層学習モデルではなく、ルールベースでオブジェクトマッチングを実施
2. **タスク全体の視点**: タスク内のすべての入力グリッド（訓練+テスト）を考慮
3. **複数パターン生成**: カテゴリ分けのパターンを複数作成し、それぞれから部分プログラムを生成
4. **背景色の考慮**: 背景色の推論と統一を適切に処理

### 重要な概念

- **オブジェクトの役割**: タスク内で同じ変換パターンを持つオブジェクト群
- **カテゴリ**: 色、形状、位置などの特徴に基づいたオブジェクトの分類
- **部分プログラム**: オブジェクト取得とフィルタリングまでを含む不完全なプログラム

---

## 📐 処理フロー

### 全体フロー

```
タスク（訓練ペア + テストペア）
    ↓
① オブジェクト抽出（4連結、8連結）
    ↓
② 変換パターン分析（訓練ペアのみ）
    ↓
③ 背景色推論（全入力グリッド）
    ↓
ループ1: 連結性（4連結、8連結）ごと
    ↓
    ループ2: カテゴリ分けパターン数分
        ↓
        ④ 背景色決定
        ↓
        ⑤ 背景色オブジェクト除外
        ↓
        ⑥ カテゴリ分け
        ↓
        ⑦ 部分プログラム生成
        ↓
        部分プログラムリストに追加
    ↓
ループ1終了
    ↓
部分プログラムリストを返す
```

---

## 🔧 詳細仕様

### ① オブジェクト抽出

**処理内容**:
- タスク内のすべての入力グリッドと出力グリッドからオブジェクトを抽出
- 4連結（connectivity=4）と8連結（connectivity=8）の両方で抽出

**実装方法**:
- `ExecutorCore.execute_program("GET_ALL_OBJECTS(4)", grid)` を使用
- `ExecutorCore.execute_program("GET_ALL_OBJECTS(8)", grid)` を使用

**出力**:
```python
{
    'connectivity_4': {
        'input_grids': [
            [Object, Object, ...],  # 各入力グリッドのオブジェクトリスト
            ...
        ],
        'output_grids': [
            [Object, Object, ...],  # 各出力グリッドのオブジェクトリスト（訓練ペアのみ）
            ...
        ]
    },
    'connectivity_8': {
        'input_grids': [...],
        'output_grids': [...]
    }
}
```

**オブジェクトの情報**:
- 色（color）
- 形状（size, width, height, aspect_ratio）
- 位置（center, bbox）
- ピクセル情報（pixels）

---

### ② 変換パターン分析（訓練ペアのみ）

**処理内容**:
- 各訓練ペアの入力オブジェクトと出力オブジェクトの類似度を計算
- 対応関係を特定し、変換パターンを分析
- 対応関係がないオブジェクト（消失・新規出現）を検出
- 複数対応（分割・統合）を検出

**対応関係の特定アルゴリズム**:

1. **1対1対応の検出**:
   ```python
   # 各入力オブジェクトに対して、最も類似度の高い出力オブジェクトを探す
   # 注意: 1つの入力オブジェクトは最大1つの出力オブジェクトと対応付けられる
   # 注意: 1つの出力オブジェクトは最大1つの入力オブジェクトと対応付けられる
   # 注意: 類似度が閾値（similarity_threshold）以上でないと対応付けられない

   for input_obj in input_objects:
       best_match = None
       best_similarity = 0.0

       for output_obj in output_objects:
           similarity = calculate_similarity(input_obj, output_obj)
           # 類似度が閾値以上で、かつ現在の最良マッチより高い場合のみ更新
           if similarity > best_similarity and similarity >= similarity_threshold:
               best_similarity = similarity
               best_match = output_obj

       # 最良マッチが見つかった場合のみ対応関係を追加
       if best_match:
           correspondences.append({
               'input_obj': input_obj,
               'output_obj': best_match,
               'similarity': best_similarity
           })
       # 注意: best_matchがNoneの場合（類似度が閾値未満）、この入力オブジェクトは対応付けられない
   ```

   **閾値の役割**:

   - **`similarity_threshold`**: 1対1対応を確立するための最小類似度（デフォルト: 0.7）
   - 類似度がこの閾値未満の場合、対応関係は確立されない
   - 閾値未満のオブジェクトは、後続の分割・統合・消失・新規出現の検出で分析される

   **閾値の影響**:

   - 閾値が高い場合: より確実な対応関係のみが検出されるが、分割・統合の可能性があるオブジェクトも未対応として残る
   - 閾値が低い場合: より多くの対応関係が検出されるが、誤対応のリスクが増える
   - 分割・統合の検出では、より低い閾値（`split_similarity_threshold`, `merge_similarity_threshold`）を使用する

   **1対1対応の検出後にオブジェクトが余る理由**:

   1対1対応の検出では、各オブジェクトは最大1つの相手としか対応付けられません。そのため、以下の理由でオブジェクトが余ることがあります：

   a. **統合（Merge）の可能性**:
      - 複数の入力オブジェクトが1つの出力オブジェクトに対応する可能性があるが、1対1対応では1つしか対応付けられない
      - 例: 3つの小さなオブジェクトが1つの大きなオブジェクトに統合される場合、1対1対応では1つしか対応付けられず、残り2つの入力オブジェクトが余る

   b. **分割（Split）の可能性**:
      - 1つの入力オブジェクトが複数の出力オブジェクトに対応する可能性があるが、1対1対応では1つしか対応付けられない
      - 例: 1つの大きなオブジェクトが3つの小さなオブジェクトに分割される場合、1対1対応では1つしか対応付けられず、残り2つの出力オブジェクトが余る

   c. **消失（Disappearance）**:
      - 入力オブジェクトに対応する出力オブジェクトが見つからない（類似度が閾値未満、または統合・分割の検出後も残った場合）
      - 例: 入力に存在するが、出力に存在しないオブジェクト

   d. **新規出現（Appearance）**:
      - 出力オブジェクトに対応する入力オブジェクトが見つからない（類似度が閾値未満、または統合・分割の検出後も残った場合）
      - 例: 出力に新しく出現したオブジェクト

   **検出の順序**:

   残ったオブジェクトは、以下の順序で調査されます：

   1. **統合（Merge）の検出**: 複数の入力オブジェクトが1つの出力オブジェクトに統合される可能性を調査
   2. **分割（Split）の検出**: 1つの入力オブジェクトが複数の出力オブジェクトに分割される可能性を調査
   3. **消失（Disappearance）の検出**: 統合・分割の検出後も残った入力オブジェクトを消失と判定
   4. **新規出現（Appearance）の検出**: 統合・分割の検出後も残った出力オブジェクトを新規出現と判定

   **注意**:
   - 統合と分割は、1対1対応の検出後に残ったオブジェクトに対して調査されます
   - 統合・分割で使用されなかったオブジェクトが、消失・新規出現として判定されます
   - この順序により、統合・分割を優先的に検出し、それでも説明できないオブジェクトを消失・新規出現として扱います

2. **消失オブジェクトの検出**:
   ```python
   # 対応関係が見つからなかった入力オブジェクト
   # 注意: これらは消失の可能性があるが、分割・統合の可能性もある
   matched_input_indices = {corr['input_obj'].index for corr in correspondences}
   unmatched_input_objects = [
       obj for i, obj in enumerate(input_objects)
       if i not in matched_input_indices
   ]

   # 消失の可能性を判定（分割・統合の検出後に確定）
   disappeared_objects = []  # 分割・統合の検出後に、残った入力オブジェクトを消失と判定
   ```

3. **新規出現オブジェクトの検出**:
   ```python
   # 対応関係が見つからなかった出力オブジェクト
   # 注意: これらは新規出現の可能性があるが、分割・統合の可能性もある
   matched_output_indices = {corr['output_obj'].index for corr in correspondences}
   unmatched_output_objects = [
       obj for i, obj in enumerate(output_objects)
       if i not in matched_output_indices
   ]

   # 新規出現の可能性を判定（分割・統合の検出後に確定）
   appeared_objects = []  # 分割・統合の検出後に、残った出力オブジェクトを新規出現と判定
   ```

4. **分割・統合の検出**:

   1対1対応の検出後、残ったオブジェクト間の関係を分析します。

   **前提条件**:
   - 1対1対応の検出が完了している
   - 対応関係が見つからなかった入力オブジェクトと出力オブジェクトが存在する

   **検出の目的**:
   - 残ったオブジェクトが、消失・新規出現・分割・統合のいずれかを判定する
   - 分割・統合が検出された場合、それらのオブジェクトは消失・新規出現から除外する

   **分割（Split）の検出**:

   分割の検出は、FIT_SHAPE_COLORベースの正規化スコアを使用した繰り返し探索方式を採用しています。

   ```python
   def detect_splits(self, input_objects: List[ObjectInfo],
                     output_objects: List[ObjectInfo],
                     correspondences: List[Dict]) -> List[Dict]:
       """分割パターンを検出（FIT_SHAPE_COLORベース）"""
       splits = []

       # 1対1対応で使用されていないオブジェクトを特定
       matched_input_indices = {corr['input_obj'].index for corr in correspondences}
       matched_output_indices = {corr['output_obj'].index for corr in correspondences}

       unmatched_input_objects = [
           obj for i, obj in enumerate(input_objects)
           if obj.index not in matched_input_indices
       ]
       unmatched_output_objects = [
           obj for i, obj in enumerate(output_objects)
           if obj.index not in matched_output_indices
       ]

       if not unmatched_input_objects or not unmatched_output_objects:
           return splits

       # バウンディングボックスサイズの大きい順にソート
       unmatched_input_objects.sort(key=lambda obj: self._calculate_bbox_size(obj), reverse=True)

       # 各未対応の入力オブジェクトに対して、分割先を探索
       for input_obj in unmatched_input_objects:
           # 分割先候補
           selected_outputs = []
           remaining_outputs = unmatched_output_objects.copy()

           # 探索範囲を設定（入力オブジェクトのサイズに基づく）
           search_range = min(10, max(input_obj.width, input_obj.height) // 2)

           # 繰り返し探索（形状の減算なし）
           while remaining_outputs:
               best_output = None
               best_score = -1
               best_result = None

               # 残りの出力オブジェクトから最適な分割先を探す
               for output_obj in remaining_outputs:
                   # FIT_SHAPE_COLORで最適な配置を探索
                   # obj1=出力オブジェクト, obj2=入力オブジェクト（元の形状のまま）
                   result = self._find_best_fit_shape_color(output_obj, input_obj, search_range)
                   if result and result['score'] > best_score:
                       best_score = result['score']
                       best_output = output_obj
                       best_result = result

               # スコアが一定以上の場合、分割先として選ぶ
               # 正規化されたスコアなので、重複ピクセル率に基づいて閾値を設定
               # 例: 重複ピクセル率が0.1以上（10%以上）の場合: 0.1 × 100000 = 10000
               score_threshold = 10000.0  # 重複ピクセル率が0.1以上の場合の最小スコア（要調整）
               if best_output and best_result and best_score >= score_threshold:
                   selected_outputs.append({
                       'output_obj': best_output,
                       'fit_result': best_result
                   })

                   # 選んだ出力オブジェクトを候補から除外
                   remaining_outputs.remove(best_output)
               else:
                   # 一定以上高いスコアのオブジェクトが存在しない場合、終了
                   break

           # 分割先が2つ以上ある場合、分割と判定
           if len(selected_outputs) >= 2:
               output_objs = [item['output_obj'] for item in selected_outputs]

               # 各出力オブジェクトに対する変換パターンを分析
               transformation_patterns = []
               for item in selected_outputs:
                   output_obj = item['output_obj']
                   pattern = self._identify_transformation_pattern(input_obj, output_obj)
                   transformation_patterns.append({
                       'output_obj': output_obj,
                       'pattern': pattern
                   })

               # 信頼度を計算（選んだオブジェクトの数とスコアに基づく）
               avg_score = sum(item['fit_result']['score'] for item in selected_outputs) / len(selected_outputs) if selected_outputs else 0.0
               # スコアを正規化（10000-100000の範囲を0-1にマッピング、簡易版）
               # 重複ピクセル率が0.1の場合10000、1.0の場合100000を想定
               normalized_score = min(1.0, (avg_score - 10000.0) / 90000.0) if avg_score >= 10000.0 else 0.0
               # オブジェクト数の影響（2個で0.5、3個以上で0.8、4個以上で1.0）
               count_factor = min(1.0, 0.5 + (len(selected_outputs) - 2) * 0.15)
               confidence = (normalized_score * 0.6 + count_factor * 0.4)

               splits.append({
                   'input_obj': input_obj,
                   'output_objects': output_objs,
                   'correspondence_type': 'one_to_many',
                   'transformation_patterns': transformation_patterns,
                   'split_count': len(output_objs),
                   'confidence': confidence
               })

       return splits
   ```

   **FIT_SHAPE_COLOR正規化スコア計算**:

   分割・統合の検出では、`_calculate_fit_shape_color_score()`メソッドを使用して、位置と色の両方を考慮した正規化スコアを計算します。このメソッドは、元のFIT_SHAPE_COLORと同じロジックだが、重複ピクセル数と隣接辺数をobj2のピクセル数と辺数で正規化してからスコアに換算します。これにより、オブジェクトサイズに依存しない評価が可能になります。

   ```python
   def _calculate_fit_shape_color_score(
       self, moved_pixels, obj2_pixels, obj2_color_map, rotation_index, dx, dy, flip_type=None
   ):
       """統合・分割調査用のFIT_SHAPE_COLORスコア計算（位置+色、正規化版）

       FIT_SHAPE_COLORと同じロジックだが、重複ピクセル数と隣接辺数を
       obj2のピクセル数と辺数で正規化してからスコアに換算する。
       これにより、オブジェクトサイズに依存しない評価が可能になる。

       優先順位:
       1. 重複ピクセル率（位置+色、正規化）× 100000
       2. 重複ピクセル率（位置、正規化）× 10000
       3. 隣接辺率（輪郭同士、正規化）× 1000
       4. 反転しない（第4優先）× 100
       5. 回転の少なさ（第5優先）× 10
       6. 移動量の少なさ（第6優先）× 1
       """
       # obj2の基本情報を計算
       obj2_pixels_set = set((p[0], p[1]) for p in obj2_pixels)
       obj2_pixel_count = len(obj2_pixels_set)
       obj2_edge_count = self._calculate_edge_count(obj2_pixels_set)

       # 1. 重複ピクセル数（位置+色）と重複ピクセル数（位置のみ）
       overlap_count_color = 0  # 位置+色の両方が一致
       overlap_count_position = 0  # 位置のみが一致（色は異なっても良い）
       position_overlap_coords = set()

       for p in moved_pixels:
           pos = (p[0], p[1])
           color = p[2] if len(p) > 2 else 0

           # 位置が一致するかチェック
           if pos in obj2_pixels_set:
               position_overlap_coords.add(pos)
               # 位置+色の両方が一致するかチェック
               if pos in obj2_color_map and color == obj2_color_map[pos]:
                   overlap_count_color += 1

       # 位置のみの重複ピクセル数（位置+色の重複を含む）
       overlap_count_position = len(position_overlap_coords)

       # 正規化: obj2のピクセル数で割る
       normalized_overlap_color = overlap_count_color / obj2_pixel_count
       normalized_overlap_position = overlap_count_position / obj2_pixel_count

       # 2. 隣接辺数（両方のオブジェクトの外形が内側から接している部分、色は問わない）
       # 位置一致領域の外側で、両方のオブジェクトの外形が内側から接している辺をカウント
       adjacent_edges = 0
       moved_pixels_set = set((p[0], p[1]) for p in moved_pixels)

       for pos in position_overlap_coords:
           x, y = pos
           for dx_dir, dy_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
               nx, ny = x + dx_dir, y + dy_dir
               # この方向が位置一致領域の外側かチェック
               if (nx, ny) not in position_overlap_coords:
                   # obj1とobj2の両方でこの方向が外形かチェック（色は問わない）
                   is_obj1_outline = (nx, ny) not in moved_pixels_set
                   is_obj2_outline = (nx, ny) not in obj2_pixels_set
                   # 両方の外形が内側から接している場合のみカウント
                   if is_obj1_outline and is_obj2_outline:
                       adjacent_edges += 1

       # 正規化: obj2の辺数で割る
       normalized_adjacent = adjacent_edges / obj2_edge_count

       # 3-6. 反転、回転、移動量のペナルティ
       # ...

       # 総合スコア（正規化後の値を使用）
       score = (normalized_overlap_color * 100000 +
                normalized_overlap_position * 10000 +
                normalized_adjacent * 1000 +
                flip_score * 100 +
                rotation_score * 10 +
                movement_score)

       return score
   ```

   **重要なポイント**:
   - **処理順序**: バウンディングボックスサイズの大きい順に未対応の入力オブジェクトを処理
   - **FIT_SHAPE_COLORベース**: 位置と色の両方を考慮した最適配置を探索
   - **正規化スコア**: 重複ピクセル数と隣接辺数をobj2のピクセル数と辺数で正規化（オブジェクトサイズに依存しない評価）
   - **重複ピクセル率の2分割**: 位置+色の重複と位置のみの重複を別々に評価
   - **隣接辺率**: 色の一致条件を除外し、位置のみで輪郭同士の接続を評価
   - **繰り返し探索**: 最適な分割先を1つずつ選び、候補から除外（形状の減算なし）
   - **終了条件**: 適切な分割先が見つからなくなるまで繰り返す
   - **信頼度**: 選んだオブジェクトの数と平均スコアに基づいて計算

   **統合（Merge）の検出**:

   統合の検出は、分割と同様にFIT_SHAPE_COLORベースの正規化スコアを使用した繰り返し探索方式を採用しています。

   ```python
   def detect_merges(self, input_objects: List[ObjectInfo],
                     output_objects: List[ObjectInfo],
                     correspondences: List[Dict]) -> List[Dict]:
       """統合パターンを検出（FIT_SHAPE_COLORベース）"""
       merges = []

       # 1対1対応で使用されていないオブジェクトを特定
       matched_input_indices = {corr['input_obj'].index for corr in correspondences}
       matched_output_indices = {corr['output_obj'].index for corr in correspondences}

       unmatched_input_objects = [
           obj for i, obj in enumerate(input_objects)
           if obj.index not in matched_input_indices
       ]
       unmatched_output_objects = [
           obj for i, obj in enumerate(output_objects)
           if obj.index not in matched_output_indices
       ]

       if not unmatched_input_objects or not unmatched_output_objects:
           return merges

       # バウンディングボックスサイズの大きい順にソート
       unmatched_output_objects.sort(key=lambda obj: self._calculate_bbox_size(obj), reverse=True)

       # 各未対応の出力オブジェクトに対して、統合元を探索
       for output_obj in unmatched_output_objects:
           # 統合元候補
           selected_inputs = []
           remaining_inputs = unmatched_input_objects.copy()

           # 探索範囲を設定（出力オブジェクトのサイズに基づく）
           search_range = min(10, max(output_obj.width, output_obj.height) // 2)

           # 繰り返し探索（形状の減算なし）
           while remaining_inputs:
               best_input = None
               best_score = -1
               best_result = None

               # 残りの入力オブジェクトから最適な統合元を探す
               for input_obj in remaining_inputs:
                   # FIT_SHAPE_COLORで最適な配置を探索
                   # obj1=入力オブジェクト, obj2=出力オブジェクト（元の形状のまま）
                   result = self._find_best_fit_shape_color(input_obj, output_obj, search_range)
                   if result and result['score'] > best_score:
                       best_score = result['score']
                       best_input = input_obj
                       best_result = result

               # スコアが一定以上の場合、統合元として選ぶ
               # 正規化されたスコアなので、重複ピクセル率に基づいて閾値を設定
               # 例: 重複ピクセル率が0.1以上（10%以上）の場合: 0.1 × 100000 = 10000
               score_threshold = 10000.0  # 重複ピクセル率が0.1以上の場合の最小スコア（要調整）
               if best_input and best_result and best_score >= score_threshold:
                   selected_inputs.append({
                       'input_obj': best_input,
                       'fit_result': best_result
                   })

                   # 選んだ入力オブジェクトを候補から除外
                   remaining_inputs.remove(best_input)
               else:
                   # 一定以上高いスコアのオブジェクトが存在しない場合、終了
                   break

           # 統合元が2つ以上ある場合、統合と判定
           if len(selected_inputs) >= 2:
               input_objs = [item['input_obj'] for item in selected_inputs]

               # 各入力オブジェクトに対する変換パターンを分析
               transformation_patterns = []
               for item in selected_inputs:
                   input_obj = item['input_obj']
                   pattern = self._identify_transformation_pattern(input_obj, output_obj)
                   transformation_patterns.append({
                       'input_obj': input_obj,
                       'pattern': pattern
                   })

               # 信頼度を計算（選んだオブジェクトの数とスコアに基づく）
               avg_score = sum(item['fit_result']['score'] for item in selected_inputs) / len(selected_inputs) if selected_inputs else 0.0
               # スコアを正規化（10000-100000の範囲を0-1にマッピング、簡易版）
               # 重複ピクセル率が0.1の場合10000、1.0の場合100000を想定
               normalized_score = min(1.0, (avg_score - 10000.0) / 90000.0) if avg_score >= 10000.0 else 0.0
               # オブジェクト数の影響（2個で0.5、3個以上で0.8、4個以上で1.0）
               count_factor = min(1.0, 0.5 + (len(selected_inputs) - 2) * 0.15)
               confidence = (normalized_score * 0.6 + count_factor * 0.4)

               merges.append({
                   'input_objects': input_objs,
                   'output_obj': output_obj,
                   'correspondence_type': 'many_to_one',
                   'transformation_patterns': transformation_patterns,
                   'merge_count': len(input_objs),
                   'confidence': confidence
               })

       return merges
   ```

   **重要なポイント**:
   - **処理順序**: バウンディングボックスサイズの大きい順に未対応の出力オブジェクトを処理
   - **FIT_SHAPE_COLORベース**: 位置と色の両方を考慮した最適配置を探索
   - **正規化スコア**: 重複ピクセル数と隣接辺数をobj2のピクセル数と辺数で正規化（オブジェクトサイズに依存しない評価）
   - **重複ピクセル率の2分割**: 位置+色の重複と位置のみの重複を別々に評価
   - **隣接辺率**: 色の一致条件を除外し、位置のみで輪郭同士の接続を評価
   - **繰り返し探索**: 最適な統合元を1つずつ選び、候補から除外（形状の減算なし）
   - **終了条件**: 適切な統合元が見つからなくなるまで繰り返す
   - **信頼度**: 選んだオブジェクトの数と平均スコアに基づいて計算

   **分割・統合検出の実行順序**:

   ```python
   # 1. 1対1対応の検出
   correspondences = self.find_correspondences(input_objects, output_objects)

   # 2. 統合の検出（1対1対応の後）
   merges = self.detect_merges(input_objects, output_objects, correspondences)

   # 3. 分割の検出（1対1対応と統合の後）
   splits = self.detect_splits(input_objects, output_objects, correspondences)

   # 4. 消失・新規出現の検出（統合・分割の検出後）
   # 統合・分割で使用されていないオブジェクトを消失・新規出現と判定
   ```

   **検出の優先順位**:

   - 1対1対応が最優先
   - 統合が分割より優先（統合を先に検出）
   - 統合・分割で使用されなかったオブジェクトが消失・新規出現として判定される

   **検出後の処理**:

   ```python
   # 1. 1対1対応の検出
   correspondences = self.find_correspondences(input_objects, output_objects)

   # 2. 統合の検出（1対1対応の後）
   merges = self.detect_merges(input_objects, output_objects, correspondences)

   # 3. 分割の検出（1対1対応と統合の後）
   splits = self.detect_splits(input_objects, output_objects, correspondences)

   # 4. 消失・新規出現の検出（統合・分割の検出後）
   # 統合・分割で使用されていないオブジェクトを消失・新規出現と判定
   merge_input_indices = {inp.index for merge in merges for inp in merge['input_objects']}
   merge_output_indices = {merge['output_obj'].index for merge in merges}
   split_input_indices = {split['input_obj'].index for split in splits}
   split_output_indices = {out.index for split in splits for out in split['output_objects']}

   matched_input_indices = (
       {corr['input_obj'].index for corr in correspondences} |
       merge_input_indices |
       split_input_indices
   )
   matched_output_indices = (
       {corr['output_obj'].index for corr in correspondences} |
       merge_output_indices |
       split_output_indices
   )

   # 消失オブジェクト（統合・分割で使用されていない入力オブジェクト）
   disappeared_objects = [
       obj for obj in input_objects
       if obj.index not in matched_input_indices
   ]

   # 新規出現オブジェクト（統合・分割で使用されていない出力オブジェクト）
   appeared_objects = [
       obj for obj in output_objects
       if obj.index not in matched_output_indices
   ]
   ```

**類似度計算**:

類似度は以下の要素から計算されます：

1. **色の一致度**:
   - オブジェクトの色が一致しているか（1.0）または不一致（0.0）

2. **形状の類似度**:
   - サイズの類似度（ピクセル数の比率）
   - アスペクト比の類似度（幅/高さの比率）
   - ピクセル配置の類似度
   - **注意**: 現在の実装では、回転や反転は考慮されていません（形状は元の向きのまま比較）

3. **位置の類似度**:
   - 重心間距離に基づく類似度
   - 距離が近いほど類似度が高い

**回転・反転の考慮**:

回転・反転を考慮した類似度計算を実装します。段階的なアプローチを採用します：

1. **第1段階: 通常の類似度計算で候補を絞り込む**
   - 回転・反転を考慮しない通常の類似度計算を実行
   - 閾値以上の候補を抽出（候補数は制限可能）

2. **第2段階: 候補に対して回転・反転を考慮した詳細な比較**
   - 候補オブジェクトに対して、すべての回転・反転パターンを試す
   - 最も高い類似度を採用

**回転・反転を考慮した形状類似度計算**:

```python
def _calculate_shape_similarity_with_rotation_flip(
    self, obj1: Dict[str, Any], obj2: Dict[str, Any]
) -> Tuple[float, Optional[int], Optional[str]]:
    """
    回転・反転を考慮した形状類似度を計算

    Returns:
        (similarity, best_rotation, best_flip):
        - similarity: 最高類似度
        - best_rotation: 最良の回転角度（0, 90, 180, 270）
        - best_flip: 最良の反転タイプ（None, "X", "Y"）
    """
    pixels1 = set(obj1.get('pixels', []))
    pixels2 = set(obj2.get('pixels', []))

    if not pixels1 or not pixels2:
        return 0.0, None, None

    # 重心基準に正規化
    pixels1_norm = self._normalize_pixels(pixels1)

    best_similarity = 0.0
    best_rotation = None
    best_flip = None

    # すべての回転・反転パターンを試す
    rotations = [0, 90, 180, 270]
    flips = [None, "X", "Y"]

    for rotation in rotations:
        for flip in flips:
            # 回転・反転を適用
            pixels2_transformed = self._transform_pixels(pixels2, rotation, flip)
            pixels2_norm = self._normalize_pixels(pixels2_transformed)

            # Jaccard係数を計算
            intersection = len(pixels1_norm & pixels2_norm)
            union = len(pixels1_norm | pixels2_norm)
            similarity = intersection / union if union > 0 else 0.0

            if similarity > best_similarity:
                best_similarity = similarity
                best_rotation = rotation
                best_flip = flip

    return best_similarity, best_rotation, best_flip

def _transform_pixels(
    self, pixels: set, rotation: int, flip: Optional[str]
) -> set:
    """
    ピクセル座標を回転・反転変換

    Args:
        pixels: ピクセル座標のセット
        rotation: 回転角度（0, 90, 180, 270）
        flip: 反転タイプ（None, "X", "Y"）

    Returns:
        変換後のピクセル座標のセット
    """
    transformed = set()

    for y, x in pixels:
        # 回転
        if rotation == 90:
            new_y, new_x = x, -y
        elif rotation == 180:
            new_y, new_x = -y, -x
        elif rotation == 270:
            new_y, new_x = -x, y
        else:  # rotation == 0
            new_y, new_x = y, x

        # 反転
        if flip == "X":
            new_y = -new_y
        elif flip == "Y":
            new_x = -new_x

        transformed.add((new_y, new_x))

    return transformed
```

**段階的な類似度計算アルゴリズム**:

```python
def _calculate_object_similarity_with_rotation_flip(
    self, obj1: Dict[str, Any], obj2: Dict[str, Any],
    use_rotation_flip: bool = True
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    回転・反転を考慮したオブジェクト類似度を計算

    Args:
        obj1: 入力オブジェクト
        obj2: 出力オブジェクト
        use_rotation_flip: 回転・反転を考慮するか

    Returns:
        (similarity, transformation_info):
        - similarity: 類似度
        - transformation_info: 変換情報（回転角度、反転タイプなど）
    """
    similarities = []
    transformation_info = None

    # 色の類似度
    if self.config.enable_color_matching:
        color_similarity = 1.0 if obj1['color'] == obj2['color'] else 0.0
        similarities.append(color_similarity)

    # 形状の類似度
    if self.config.enable_shape_matching:
        if use_rotation_flip:
            shape_similarity, rotation, flip = self._calculate_shape_similarity_with_rotation_flip(obj1, obj2)
            transformation_info = {
                'rotation': rotation,
                'flip': flip
            }
        else:
            shape_similarity = self._calculate_shape_similarity(obj1, obj2)
        similarities.append(shape_similarity)

    # 位置の類似度（回転・反転を考慮しない）
    if self.config.enable_position_matching:
        position_similarity = self._calculate_position_similarity(obj1, obj2)
        similarities.append(position_similarity)

    # サイズの類似度
    size_similarity = self._calculate_size_similarity(obj1, obj2)
    similarities.append(size_similarity)

    # 重み付き平均
    weights = [0.3, 0.3, 0.2, 0.2]  # 色、形状、位置、サイズの重み
    weighted_similarity = sum(s * w for s, w in zip(similarities, weights))

    return weighted_similarity, transformation_info
```

**1対1対応の検出（回転・反転考慮版）**:

```python
def _find_object_correspondences_with_rotation_flip(
    self, input_objects: List[ObjectInfo],
    output_objects: List[ObjectInfo]
) -> List[Dict]:
    """
    回転・反転を考慮した1対1対応の検出
    """
    correspondences = []
    used_output_indices = set()

    # 第1段階: 通常の類似度計算で候補を絞り込む
    candidates = []
    for i, input_obj in enumerate(input_objects):
        for j, output_obj in enumerate(output_objects):
            if j in used_output_indices:
                continue

            # 通常の類似度計算（回転・反転を考慮しない）
            similarity = self._calculate_object_similarity_with_rotation_flip(
                input_obj, output_obj, use_rotation_flip=False
            )[0]

            # 候補閾値（通常の閾値より低め）
            if similarity >= self.config.candidate_similarity_threshold:
                candidates.append({
                    'input_idx': i,
                    'output_idx': j,
                    'input_obj': input_obj,
                    'output_obj': output_obj,
                    'preliminary_similarity': similarity
                })

    # 候補を類似度順にソート
    candidates.sort(key=lambda x: x['preliminary_similarity'], reverse=True)

    # 第2段階: 候補に対して回転・反転を考慮した詳細な比較
    for candidate in candidates:
        input_idx = candidate['input_idx']
        output_idx = candidate['output_idx']

        if output_idx in used_output_indices:
            continue

        # 回転・反転を考慮した詳細な類似度計算
        similarity, transformation_info = self._calculate_object_similarity_with_rotation_flip(
            candidate['input_obj'], candidate['output_obj'], use_rotation_flip=True
        )

        # 最終閾値チェック
        if similarity >= self.config.similarity_threshold:
            correspondences.append({
                'input_obj': candidate['input_obj'],
                'output_obj': candidate['output_obj'],
                'similarity': similarity,
                'transformation_info': transformation_info  # 回転・反転情報
            })
            used_output_indices.add(output_idx)

    return correspondences
```

**設定パラメータの追加**:

```python
@dataclass
class ObjectMatchingConfig:
    # ... 既存のパラメータ ...

    # 回転・反転の考慮
    enable_rotation_flip_matching: bool = True  # 回転・反転を考慮するか
    candidate_similarity_threshold: float = 0.5  # 候補を絞り込むための閾値（通常の閾値より低め）
    max_rotation_flip_candidates: int = 10  # 回転・反転を考慮する候補の最大数（計算コスト制御）
```

**対応関係の種類**:

対応関係の種類と変換パターンは、独立したパラメータとして扱います。

1. **1対1対応**:
   - 1つの入力オブジェクトが1つの出力オブジェクトに対応

2. **1対多対応（分割）**:
   - 1つの入力オブジェクトが複数の出力オブジェクトに対応
   - 例: 1つの大きなオブジェクトが3つの小さなオブジェクトに分割

3. **多対1対応（統合）**:
   - 複数の入力オブジェクトが1つの出力オブジェクトに対応
   - 例: 3つの小さなオブジェクトが1つの大きなオブジェクトに統合

4. **1対0対応（消失）**:
   - 入力オブジェクトに対応する出力オブジェクトが見つからない

5. **0対1対応（新規出現）**:
   - 出力オブジェクトに対応する入力オブジェクトが見つからない

**変換パターン**:

対応関係の種類とは独立して、オブジェクトの属性変化を表すパターンです。

変換パターンは、4種類の変化（`color_change`, `position_change`, `shape_change`, `disappearance`）を個別のパラメータとして保持します。組み合わせは個別パラメータから自動的に表現されます。

**個別パラメータ**:

1. **`color_change`**: 色変化の有無（bool）
   - `True`: 入力オブジェクトと出力オブジェクトの色が異なる
   - `False`: 色が同じ

2. **`position_change`**: 位置変化の有無（bool）
   - `True`: 入力オブジェクトと出力オブジェクトの中心位置が1.0ピクセル以上異なる
   - `False`: 位置がほぼ同じ

3. **`shape_change`**: ピクセル形状変化の有無（bool）
   - `True`: 入力オブジェクトと出力オブジェクトのピクセル形状（ピクセル数、幅、または高さ）が異なる
   - `False`: ピクセル形状（ピクセル数、幅、高さ）がすべて同じ

4. **`disappearance`**: 消失の有無（bool）
   - `True`: 入力オブジェクトに対応する出力オブジェクトが存在しない（消失）
   - `False`: 入力オブジェクトに対応する出力オブジェクトが存在する

**組み合わせの表現**:

個別パラメータの組み合わせにより、以下のような変換パターンを表現できます：

- **変化なし**: `color_change=False, position_change=False, shape_change=False, disappearance=False`
- **消失**: `color_change=False, position_change=False, shape_change=False, disappearance=True`
- **色のみ変化**: `color_change=True, position_change=False, shape_change=False, disappearance=False`
- **位置のみ変化**: `color_change=False, position_change=True, shape_change=False, disappearance=False`
- **ピクセル形状のみ変化**: `color_change=False, position_change=False, shape_change=True, disappearance=False`
- **色と位置が変化**: `color_change=True, position_change=True, shape_change=False, disappearance=False`
- **色とピクセル形状が変化**: `color_change=True, position_change=False, shape_change=True, disappearance=False`
- **位置とピクセル形状が変化**: `color_change=False, position_change=True, shape_change=True, disappearance=False`
- **すべてが変化**: `color_change=True, position_change=True, shape_change=True, disappearance=False`

**データ構造**:

```python
{
    'color_change': bool,  # 色変化の有無（個別パラメータ）
    'position_change': bool,  # 位置変化の有無（個別パラメータ）
    'shape_change': bool,  # ピクセル形状変化の有無（個別パラメータ）
    'disappearance': bool  # 消失の有無（個別パラメータ）
}
```

**注意**: 個別パラメータのみを保持し、組み合わせは個別パラメータから自動的に表現されます。`type`フィールドは削除されました。

**利点**:

- **柔軟性**: 個別パラメータ（`color_change`, `position_change`, `shape_change`, `disappearance`）により、組み合わせの違いを正確に表現
- **拡張性**: 新しい組み合わせパターンが出現しても、個別パラメータを組み合わせることで対応可能
- **類似度計算**: 個別パラメータを使用することで、部分的な一致も考慮した類似度計算が可能
- **多次元ベクトル**: 個別パラメータを直接多次元ベクトル（4次元）として特徴量に使用
  - 4次元ベクトル `[color_change, position_change, shape_change, disappearance]`を使用
  - **利点**: コンパクト（4次元）、柔軟性が高い、部分的な一致も考慮可能

**注意**:
- 対応関係の種類（1対1、分割、統合など）と変換パターン（色変更、位置変更など）は独立したパラメータです
- 例えば、`position_change`（位置変更）を行った結果、複数のオブジェクトが統合される場合、対応関係は「統合」、変換パターンは「位置変更」となります

**出力**:
```python
{
    'train_pair_0': {
        'input_objects': [Object, ...],
        'output_objects': [Object, ...],
        'correspondences': [
            {
                'input_obj': Object,
                'output_obj': Object,  # 1対1対応
                'similarity': float,
                'correspondence_type': 'one_to_one',  # 対応関係の種類
                'transformation_pattern': {  # 変換パターン（独立したパラメータ）
                    'color_change': bool,  # 色変化の有無（個別パラメータ）
                    'position_change': bool,  # 位置変化の有無（個別パラメータ）
                    'shape_change': bool,  # ピクセル形状変化の有無（個別パラメータ）
                    'disappearance': bool  # 消失の有無（個別パラメータ）
                }
            },
            ...
        ],
        'disappeared_objects': [
            {
                'input_obj': Object,
                'correspondence_type': 'one_to_zero',  # 対応関係の種類
                'transformation_pattern': {  # 変換パターン（消失）
                    'color_change': False,
                    'position_change': False,
                    'shape_change': False,
                    'disappearance': True  # 消失の有無（個別パラメータ）
                },
                'confidence': float
            },
            ...
        ],
        'appeared_objects': [
            {
                'output_obj': Object,
                'correspondence_type': 'zero_to_one',  # 対応関係の種類
                'transformation_pattern': None,  # 新規出現の場合は変換パターンなし
                'confidence': float
            },
            ...
        ],
        'split_objects': [
            {
                'input_obj': Object,
                'output_objects': [Object, ...],  # 複数の出力オブジェクト
                'correspondence_type': 'one_to_many',  # 対応関係の種類（分割）
                'transformation_patterns': [  # 変換パターン（各出力オブジェクトごと、独立したパラメータ）
                    {
                        'output_obj': Object,
                        'pattern': {  # この出力オブジェクトに対する変換パターン
                            'color_change': bool,  # 色変化の有無（個別パラメータ）
                            'position_change': bool,  # 位置変化の有無（個別パラメータ）
                            'shape_change': bool,  # ピクセル形状変化の有無（個別パラメータ）
                            'disappearance': bool  # 消失の有無（個別パラメータ）
                        }
                    },
                    ...
                ],
                'split_count': int,
                'confidence': float
            },
            ...
        ],
        'merged_objects': [
            {
                'input_objects': [Object, ...],  # 複数の入力オブジェクト
                'output_obj': Object,
                'correspondence_type': 'many_to_one',  # 対応関係の種類（統合）
                'transformation_patterns': [  # 変換パターン（各入力オブジェクトごと、独立したパラメータ）
                    {
                        'input_obj': Object,
                        'pattern': {  # この入力オブジェクトに対する変換パターン
                            'color_change': bool,  # 色変化の有無（個別パラメータ）
                            'position_change': bool,  # 位置変化の有無（個別パラメータ）
                            'shape_change': bool,  # ピクセル形状変化の有無（個別パラメータ）
                            'disappearance': bool  # 消失の有無（個別パラメータ）
                        }
                    },
                    ...
                ],
                'merge_count': int,
                'confidence': float
            },
            ...
        ]
    },
    ...
}
```

---

### ③ 背景色推論（全入力グリッド）

**処理内容**:
- タスク内のすべての入力グリッド（訓練+テスト）に対して背景色を推論

**推論方法**:
- 最頻出色を基本候補とする
- エッジ色を考慮
- グリッドサイズを考慮した重み付け

**出力**:
```python
{
    'input_grid_0': {
        'inferred_bg_color': int,
        'confidence': float,
        'method': str  # 'frequency', 'edge', etc.
    },
    ...
}
```

---

### ループ1: 連結性ごとの処理

**ループ対象**: `[4, 8]` (connectivity)

**処理内容**:
- 4連結と8連結それぞれについて、独立して部分プログラムを生成

---

### ループ2: カテゴリ分けパターンごとの処理

**ループ回数**: 設定可能（デフォルト: 3-5パターン）

**処理内容**:
- 各ループで異なるカテゴリ分けパターンを生成
- 乱数やループインデックスを使用して、カテゴリ分けの優先度や閾値を変更

---

### ④ 背景色決定

**処理内容**:
- 入力グリッドごとの背景色を決定
- ループ2の各ループで、異なる背景色戦略を選択

#### ケース1: 完全一致の場合

**条件**: すべての入力グリッドで推論した背景色が完全一致

**処理**:
- すべての入力グリッドで、推論した背景色を使用
- 部分プログラムに以下を追記:
  ```python
  "objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))"
  ```

#### ケース2: 完全一致でない場合（確率的な処理）

**処理**:
1. **一致度の計算**:
   ```python
   # 入力グリッドごとに使われている色の一致度
   color_consistency = self._calculate_color_consistency(all_input_grids)
   # 0.0 ~ 1.0: すべての入力グリッドで使われている色が一致している度合い

   # 入力グリッドごとの推論した背景色の一致度
   bg_color_consistency = self._calculate_bg_color_consistency(inferred_bg_colors)
   # 0.0 ~ 1.0: 推論した背景色が一致している度合い
   ```

2. **確率的な選択**:
   ```python
   # 一致度が大きい場合 → ケース2-1（統一背景色）の確率を高く
   # 一致度が小さい場合 → ケース2-2（グリッドごと）の確率を高く

   # 確率の計算
   consistency_score = (color_consistency + bg_color_consistency) / 2.0
   probability_unified = consistency_score  # 一致度が高いほど統一背景色の確率が高い

   # ループ2のインデックスも考慮（多様性を確保）
   pattern_bias = pattern_idx / num_patterns  # 0.0 ~ 1.0
   final_probability = probability_unified * (1.0 - pattern_bias * 0.3)  # パターンが進むほど確率を下げる

   if random.random() < final_probability:
       # ケース2-1: 統一背景色
   else:
       # ケース2-2: グリッドごと
   ```

**ケース2-1: 統一背景色を指定**

**処理**:
```python
# 推論した背景色の中で最も多い色を選択
bg_color_counts = Counter([bg['inferred_color'] for bg in inferred_bg_colors])
most_common_colors = [color for color, count in bg_color_counts.most_common()
                      if count == bg_color_counts.most_common(1)[0][1]]

if len(most_common_colors) == 1:
    # 最も多い色が1つ
    unified_bg_color = most_common_colors[0]
else:
    # 最も多い色が複数ある場合
    # すべての入力グリッドに0が含まれていて、かつ推論した背景色に1つでも0が含まれていたら
    all_grids_have_zero = all(0 in grid for grid in all_input_grids)
    inferred_has_zero = 0 in most_common_colors

    if all_grids_have_zero and inferred_has_zero:
        unified_bg_color = 0
    else:
        # 最も多い色からランダム選択
        unified_bg_color = random.choice(most_common_colors)
```

**部分プログラムに追記**:
```python
"objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), <unified_bg_color>))"
```

**ケース2-2: グリッドごとの背景色**

**処理**:
- 各入力グリッドで推論した背景色を使用

**部分プログラムに追記**:
```python
"objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))"
```

**一致度計算の詳細**:

```python
def _calculate_color_consistency(self, input_grids: List[List[List[int]]]) -> float:
    """入力グリッドごとに使われている色の一致度を計算"""
    if not input_grids:
        return 0.0

    # 各グリッドで使われている色の集合
    color_sets = [set(grid.flatten()) for grid in input_grids]

    # すべてのグリッドで共通の色
    common_colors = set.intersection(*color_sets) if color_sets else set()

    # すべてのグリッドで使われている色の集合
    all_colors = set.union(*color_sets) if color_sets else set()

    if not all_colors:
        return 0.0

    # 一致度 = 共通色の数 / 全色の数
    consistency = len(common_colors) / len(all_colors)
    return consistency

def _calculate_bg_color_consistency(self, inferred_bg_colors: List[int]) -> float:
    """
    推論した背景色の一致度を計算

    Args:
        inferred_bg_colors: 各入力グリッドから推論した背景色のリスト
                           （例: [0, 0, 1, 0] → 4つの入力グリッドの背景色）

    Returns:
        一致度（0.0～1.0）
    """
    if not inferred_bg_colors:
        return 0.0

    # 最頻出背景色
    most_common_bg = Counter(inferred_bg_colors).most_common(1)[0][0]
    most_common_count = Counter(inferred_bg_colors).most_common(1)[0][1]

    # 一致度 = 最頻出背景色の出現回数 / 総入力グリッド数
    # inferred_bg_colorsの長さは総入力グリッド数に等しい
    consistency = most_common_count / len(inferred_bg_colors)
    return consistency
```

---

### ⑤ 背景色オブジェクト除外

**処理内容**:
- 各入力グリッドのオブジェクトリストから、背景色のオブジェクトを除外

**実装**:
```python
filtered_objects = [
    obj for obj in objects
    if obj.color != background_color_for_grid
]
```

---

### ⑥ カテゴリ分け

**処理内容**:
- 各入力オブジェクトを、タスク全体に共通するカテゴリに分類

**オブジェクトの情報**:
- 色（color）
- 形状（size, width, height, aspect_ratio）
- 位置（center, bbox）
- 変換パターン（訓練ペアのみ）

**カテゴリ分けの基準**:

1. **色による分類**:
   - 色は独立した値なので、完全一致か不一致かのみ
   - ループ2の各ループで、色の一致を優先するかどうかを変更

2. **形状による分類**:
   - サイズの類似度: `min(size1, size2) / max(size1, size2)`
   - アスペクト比の類似度: `min(aspect1, aspect2) / max(aspect1, aspect2)`
   - ループ2の各ループで、形状の類似度閾値を変更

3. **位置による分類**:
   - 重心間距離に基づく類似度
   - ループ2の各ループで、位置の重みを変更

4. **変換パターンによる分類**:
   - 訓練ペアでの変換パターンが同じオブジェクトを同じカテゴリに
   - ループ2の各ループで、変換パターンの重みを変更
   - **消失パターン**: 消失するオブジェクトを同じカテゴリに分類
   - **新規出現パターン**: 新規出現するオブジェクトを同じカテゴリに分類（ただし、入力オブジェクトには存在しないため、カテゴリ分けには直接使用しない）

**カテゴリ分けの戦略**:

ループ2の各ループで、以下のパラメータを変更:

```python
category_params = {
    'color_weight': random.uniform(0.2, 0.5),  # 色の重み
    'shape_weight': random.uniform(0.2, 0.5),  # 形状の重み
    'position_weight': random.uniform(0.1, 0.3),  # 位置の重み
    'transformation_weight': random.uniform(0.1, 0.3),  # 変換パターンの重み
    'shape_similarity_threshold': random.uniform(0.6, 0.9),  # 形状類似度閾値
    'position_similarity_threshold': random.uniform(0.5, 0.8),  # 位置類似度閾値
    'min_objects_per_category': random.choice([1, 2]),  # カテゴリあたりの最小オブジェクト数
}
```

**カテゴリ分けアルゴリズム**:

1. **初期化**: 各オブジェクトを個別のカテゴリとして初期化

2. **類似度計算**: すべてのオブジェクトペアの類似度を計算
   ```python
   def calculate_similarity(obj1: ObjectInfo, obj2: ObjectInfo, params: Dict) -> float:
       # 色の類似度（完全一致か不一致か）
       color_sim = 1.0 if obj1.color == obj2.color else 0.0

       # 形状の類似度
       size_ratio = min(obj1.size, obj2.size) / max(obj1.size, obj2.size)
       aspect_ratio1 = obj1.width / obj1.height if obj1.height > 0 else 1.0
       aspect_ratio2 = obj2.width / obj2.height if obj2.height > 0 else 1.0
       aspect_sim = min(aspect_ratio1, aspect_ratio2) / max(aspect_ratio1, aspect_ratio2)
       shape_sim = (size_ratio + aspect_sim) / 2.0

       # 位置の類似度（正規化された距離）
       distance = np.sqrt((obj1.center[0] - obj2.center[0])**2 +
                         (obj1.center[1] - obj2.center[1])**2)
       max_distance = 10.0  # 仮定
       position_sim = max(0.0, 1.0 - distance / max_distance)

       # 対応関係の種類の類似度（訓練ペアのみ）
       if obj1.correspondence_type and obj2.correspondence_type:
           if obj1.correspondence_type == obj2.correspondence_type:
               correspondence_sim = 1.0  # 同じ対応関係の種類
           else:
               correspondence_sim = 0.0  # 異なる対応関係の種類
       else:
           correspondence_sim = 0.5  # 対応関係の種類がない場合は中立

       # 変換パターンの類似度（訓練ペアのみ）
       # 個別パラメータ（color_change, position_change, shape_change, disappearance）を使用して類似度を計算
       if obj1.transformation_pattern and obj2.transformation_pattern:
           pattern1 = obj1.transformation_pattern
           pattern2 = obj2.transformation_pattern

           matches = 0
           total = 0
           if pattern1.get('color_change') == pattern2.get('color_change'):
               matches += 1
           total += 1
           if pattern1.get('position_change') == pattern2.get('position_change'):
               matches += 1
           total += 1
           if pattern1.get('shape_change') == pattern2.get('shape_change'):
               matches += 1
           total += 1
           if pattern1.get('disappearance', False) == pattern2.get('disappearance', False):
               matches += 1
           total += 1

           trans_sim = matches / total if total > 0 else 0.0  # 完全一致: 1.0、3つ一致: 0.75、2つ一致: 0.5、1つ一致: 0.25、0つ一致: 0.0
       else:
           trans_sim = 0.5  # 変換パターンがない場合は中立

       # 対応関係の種類と変換パターンの両方を考慮
       # 注意: 対応関係の種類と変換パターンは独立したパラメータなので、別々に重み付け
       combined_sim = (
           params.get('correspondence_weight', 0.2) * correspondence_sim +
           params.get('transformation_weight', 0.2) * trans_sim
       )

       # 重み付き平均
       # 注意: 対応関係の種類と変換パターンは独立したパラメータなので、別々に重み付け
       similarity = (
           params['color_weight'] * color_sim +
           params['shape_weight'] * shape_sim +
           params['position_weight'] * position_sim +
           combined_sim  # 対応関係の種類と変換パターンの組み合わせ
       )

       return similarity
   ```

3. **クラスタリング**: 類似度が閾値以上のオブジェクトを同じカテゴリに統合
   ```python
   # 類似度マトリックスを作成
   similarity_pairs = []
   for i, obj1 in enumerate(objects):
       for j, obj2 in enumerate(objects[i+1:], start=i+1):
           similarity = calculate_similarity(obj1, obj2, params)
           if similarity >= params['similarity_threshold']:
               similarity_pairs.append((i, j, similarity))

   # 類似度が高い順にソート
   similarity_pairs.sort(key=lambda x: x[2], reverse=True)

   # クラスタリング（Union-Find風のアプローチ）
   category_map = {i: i for i in range(len(objects))}  # 各オブジェクトのカテゴリID

   for i, j, similarity in similarity_pairs:
       # 既に同じカテゴリに属している場合はスキップ
       if category_map[i] == category_map[j]:
           continue

       # カテゴリを統合
       category_id_i = category_map[i]
       category_id_j = category_map[j]
       # より小さいカテゴリIDに統合
       target_category = min(category_id_i, category_id_j)
       source_category = max(category_id_i, category_id_j)

       # すべてのオブジェクトのカテゴリIDを更新
       for k in range(len(objects)):
           if category_map[k] == source_category:
               category_map[k] = target_category
   ```

4. **最小オブジェクト数チェック**: 各カテゴリに最低1つのオブジェクトが属していることを確認
   ```python
   # カテゴリごとのオブジェクト数をカウント
   category_counts = Counter(category_map.values())

   # 最小オブジェクト数未満のカテゴリを統合
   for category_id, count in category_counts.items():
       if count < params['min_objects_per_category']:
           # 最も近いカテゴリに統合
           # （実装詳細は省略）
           pass
   ```

5. **カテゴリの特徴抽出**: 各カテゴリの代表的な特徴（色、形状、位置など）を抽出
   ```python
   for category_id in set(category_map.values()):
       category_objects = [obj for i, obj in enumerate(objects)
                          if category_map[i] == category_id]

       # 代表色（最頻出色）
       color_counts = Counter([obj.color for obj in category_objects])
       representative_color = color_counts.most_common(1)[0][0]

       # 代表形状（平均）
       avg_size = np.mean([obj.size for obj in category_objects])
       avg_aspect_ratio = np.mean([obj.width / obj.height if obj.height > 0 else 1.0
                                   for obj in category_objects])

       # 代表対応関係の種類（最頻出）
       correspondence_types = [obj.correspondence_type
                              for obj in category_objects
                              if obj.correspondence_type]
       if correspondence_types:
           representative_correspondence_type = Counter(correspondence_types).most_common(1)[0][0]
       else:
           representative_correspondence_type = None

       # 注意: 代表変換パターン（representative_transformation_pattern）は生成されていません
       # 変換パターンは個別オブジェクト（ObjectInfo）のtransformation_pattern属性に設定されます
       # 候補生成器では、個別オブジェクトのtransformation_patternから集約して変換パターンを抽出します
       #
       # 以下のコードは参考用です（実際には実行されていません）:
       # color_change_count = 0
       # position_change_count = 0
       # shape_change_count = 0
       # disappearance_count = 0
       # total_count = 0
       #
       # for obj in category_objects:
       #     if obj.transformation_pattern:
       #         if obj.transformation_pattern.get('color_change', False):
       #             color_change_count += 1
       #         if obj.transformation_pattern.get('position_change', False):
       #             position_change_count += 1
       #         if obj.transformation_pattern.get('shape_change', False):
       #             shape_change_count += 1
       #         if obj.transformation_pattern.get('disappearance', False):
       #             disappearance_count += 1
       #         total_count += 1
       #
       # if total_count == 0:
       #     representative_transformation = None
       # else:
       #     # 過半数で決定
       #     threshold = total_count / 2.0
       #     representative_transformation = {
       #         'color_change': color_change_count > threshold,
       #         'position_change': position_change_count > threshold,
       #         'shape_change': shape_change_count > threshold,
       #         'disappearance': disappearance_count > threshold
       #     }

       # 消失オブジェクトの割合（対応関係の種類から計算）
       disappearance_count = sum(1 for obj in category_objects
                                if obj.correspondence_type == 'one_to_zero')
       disappearance_ratio = disappearance_count / len(category_objects) if category_objects else 0.0

       # 分割オブジェクトの割合
       split_count = sum(1 for obj in category_objects
                        if obj.correspondence_type == 'one_to_many')
       split_ratio = split_count / len(category_objects) if category_objects else 0.0

       # 統合オブジェクトの割合
       merge_count = sum(1 for obj in category_objects
                        if obj.correspondence_type == 'many_to_one')
       merge_ratio = merge_count / len(category_objects) if category_objects else 0.0

       # 各入力グリッドでのオブジェクト数
       object_count_per_grid = [
           sum(1 for obj in category_objects if obj.grid_index == grid_idx)
           for grid_idx in range(num_input_grids)
       ]
   ```

**出力**:
```python
{
    'category_0': {
        'objects': [Object, Object, ...],  # 各オブジェクトにはtransformation_pattern属性が設定されている
        'representative_color': int,  # 代表色（最頻出色）
        'representative_shape': {...},  # 代表形状
        # 注意: transformation_pattern属性は存在しません
        # 変換パターンは個別オブジェクト（ObjectInfo）のtransformation_pattern属性から集約します
        'object_count_per_grid': [int, ...]  # 各入力グリッドでのオブジェクト数
    },
    'category_1': {...},
    ...
}
```

**部分プログラムへの反映**:

各カテゴリに対して、FILTER条件を生成:

```python
def generate_filter_condition(category: CategoryInfo, params: Dict) -> str:
    """カテゴリからFILTER条件を生成"""
    conditions = []

    # 色によるフィルタリング
    if category.representative_color is not None:
        conditions.append(f"EQUAL(GET_COLOR($obj), {category.representative_color})")

    # 形状によるフィルタリング（将来的に実装）
    # if category.representative_shape is not None:
    #     size_condition = f"AND(GREATER_EQUAL(GET_SIZE($obj), {min_size}), LESS_EQUAL(GET_SIZE($obj), {max_size}))"
    #     conditions.append(size_condition)

    # 複数条件の組み合わせ
    if len(conditions) > 1:
        filter_condition = " AND ".join([f"({c})" for c in conditions])
    elif len(conditions) == 1:
        filter_condition = conditions[0]
    else:
        # 条件がない場合はTrue（すべてのオブジェクト）
        filter_condition = "TRUE"

    return filter_condition

# 部分プログラムに追記
# 拡張データセット生成器の命名規則に合わせてobjects1, objects2, ...を使用
for category_id, category in enumerate(categories):
    filter_condition = generate_filter_condition(category, params)
    partial_program += f"\nobjects{category_id + 1} = FILTER(objects, {filter_condition})"
```

**注意**:
- カテゴリ分けは、タスク全体で共通の変換プログラムのための分類であるため、各入力グリッドでそのカテゴリに属するオブジェクトが存在するかを確認する必要があります。存在しないグリッドがある場合、そのカテゴリは無効とみなすか、条件を緩和する必要があります。
- **消失パターン**: 消失するオブジェクトは、カテゴリ分けに含めることができます。消失パターンが高いカテゴリは、部分プログラムで特別に扱う必要がある場合があります（例: そのカテゴリのオブジェクトを除外する、または条件付きで処理する）。消失オブジェクトは、色や形状などの特徴が類似している場合、同じカテゴリに分類されます。
- **新規出現パターン**: 新規出現するオブジェクトは、入力オブジェクトには存在しないため、カテゴリ分けには直接使用できません。ただし、出力オブジェクトの特徴を分析することで、将来的に新規出現を予測するカテゴリを作成できる可能性があります。
- **分割・統合パターン**: 分割や統合が発生するオブジェクトも、カテゴリ分けに含めることができます。ただし、これらのパターンは複雑なため、カテゴリ分けの際に特別な考慮が必要な場合があります。

---

### ⑦ 部分プログラム生成

**処理内容**:
- カテゴリ分けの結果に基づいて、部分プログラムを生成
- **カテゴリIDと変数名の対応関係を返す**: どのカテゴリがどの変数名に代入されたかを記録

**部分プログラムの構造**:

```python
# 基本構造
partial_program = "objects = GET_ALL_OBJECTS(4)"  # または 8

# 背景色フィルタリング
if background_color_strategy == "unified":
    partial_program += "\nobjects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), <指定色>))"
elif background_color_strategy == "per_grid":
    partial_program += "\nobjects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))"

# カテゴリ分け
category_var_names = {}  # カテゴリIDと変数名のマッピング
for category_id, category in enumerate(categories):
    filter_condition = generate_filter_condition(category)
    # 拡張データセット生成器の命名規則に合わせてobjects1, objects2, ...を使用
    category_var_name = variable_naming_system.get_next_variable_name(...)
    partial_program += f"\n{category_var_name} = FILTER(objects, {filter_condition})"
    category_var_names[category_id] = category_var_name

    # 消失パターンが高いカテゴリの場合、特別な処理を追加（将来的に実装）
    # if category.disappearance_ratio > 0.5:
    #     # このカテゴリのオブジェクトは消失する可能性が高い
    #     # 部分プログラムに特別な処理を追加（例: 条件付きで除外）
    #     pass

# カテゴリIDと変数名の対応関係を返す
return partial_program, category_var_names
```

**出力例**:
```python
# 部分プログラム
"objects = GET_ALL_OBJECTS(4)
objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))
objects1 = FILTER(objects, EQUAL(GET_COLOR($obj), 1))
objects2 = FILTER(objects, EQUAL(GET_COLOR($obj), 2))"

# カテゴリIDと変数名の対応関係
{0: "objects1", 1: "objects2"}
```

**カテゴリIDと変数名の対応関係の保存**:

オブジェクトマッチング結果（`matching_result`）には、`category_var_mappings`が含まれます：

```python
{
    'success': True,
    'partial_programs': [...],
    'categories': [...],
    'category_var_mappings': {
        '4_0': {'4_0_0': 'objects1', '4_0_1': 'objects2'},  # パターン4_0のカテゴリIDと変数名
        '4_1': {'4_1_0': 'objects1', '4_1_1': 'objects2'},  # パターン4_1のカテゴリIDと変数名
        ...
    }
}
```

- **`pattern_key`**: `"{connectivity}_{pattern_idx}"`の形式（例: `"4_0"`）
- **`category_id`**: 一意化されたカテゴリID（`"{connectivity}_{pattern_idx}_{original_category_id}"`の形式）
- **`variable_name`**: 部分プログラムで使用される変数名（例: `"objects1"`, `"objects2"`）

この情報により、候補生成時にどのカテゴリがどの変数名に代入されたかを把握し、カテゴリ固有の変換を正確に適用できます。

---

## 📊 データ構造

### オブジェクト情報

```python
@dataclass
class ObjectInfo:
    """オブジェクト情報"""
    # 基本情報
    pixels: List[Tuple[int, int]]  # ピクセル座標
    color: int  # 色
    size: int  # ピクセル数

    # 形状情報
    bbox: Tuple[int, int, int, int]  # (min_i, min_j, max_i, max_j)
    center: Tuple[float, float]  # (center_y, center_x)
    width: int  # 幅
    height: int  # 高さ
    aspect_ratio: float  # アスペクト比

    # グリッド情報
    grid_index: int  # どの入力グリッドから抽出されたか
    is_train: bool  # 訓練ペアかどうか

    # 変換パターン（訓練ペアのみ、対応関係の種類とは独立）
    transformation_pattern: Optional[Dict[str, Any]] = None  # {'color_change': bool, 'position_change': bool, 'shape_change': bool, 'disappearance': bool}
    matched_output_object: Optional['ObjectInfo'] = None  # 1対1対応の場合
    matched_output_objects: Optional[List['ObjectInfo']] = None  # 分割の場合（複数対応）

    # 対応関係の種類
    correspondence_type: str = 'matched'  # 'matched', 'disappeared', 'appeared', 'split', 'merged'
```

### カテゴリ情報

```python
@dataclass
class CategoryInfo:
    """カテゴリ情報"""
    category_id: int
    objects: List[ObjectInfo]

    # 代表特徴
    representative_color: Optional[int] = None
    representative_shape: Optional[Dict[str, Any]] = None
    representative_correspondence_type: Optional[str] = None  # 代表対応関係の種類
    # 注意: representative_transformation_pattern属性は存在しません
    # 変換パターンは個別オブジェクト（ObjectInfo）のtransformation_pattern属性から集約します

    # 統計情報
    object_count_per_grid: List[int]  # 各入力グリッドでのオブジェクト数
    total_objects: int  # 総オブジェクト数

    # 対応関係の種類の統計
    disappearance_ratio: float = 0.0  # 消失オブジェクトの割合（correspondence_type == 'one_to_zero'）
    appearance_ratio: float = 0.0  # 新規出現オブジェクトの割合（将来的に使用）
    split_ratio: float = 0.0  # 分割オブジェクトの割合（correspondence_type == 'one_to_many'）
    merge_ratio: float = 0.0  # 統合オブジェクトの割合（correspondence_type == 'many_to_one'）
```

### 背景色情報

```python
@dataclass
class BackgroundColorInfo:
    """背景色情報"""
    grid_index: int
    inferred_color: int
    confidence: float
    method: str  # 'frequency', 'edge', etc.

    # 統計情報
    color_frequency: Dict[int, int]  # 色の頻度
    edge_colors: List[int]  # エッジ色
```

---

## 🔧 実装詳細

### クラス設計

```python
class RuleBasedObjectMatcher:
    """ルールベースオブジェクトマッチャー"""

    def __init__(self, config: ObjectMatchingConfig):
        self.config = config
        self.executor = ExecutorCore()

    def match_objects(self, task: Task) -> Dict[str, Any]:
        """オブジェクトマッチングを実行"""
        # ① オブジェクト抽出
        objects_data = self._extract_all_objects(task)

        # ② 変換パターン分析
        transformation_patterns = self._analyze_transformation_patterns(task, objects_data)

        # ③ 背景色推論
        background_colors = self._infer_background_colors(task)

        # 部分プログラムリスト
        partial_programs = []

        # ループ1: 連結性ごと
        for connectivity in [4, 8]:
            # ループ2: カテゴリ分けパターン数分
            num_patterns = self.config.num_category_patterns_4 if connectivity == 4 else self.config.num_category_patterns_8
            for pattern_idx in range(num_patterns):
                # ④ 背景色決定
                bg_strategy = self._decide_background_color_strategy(
                    background_colors, pattern_idx
                )

                # ⑤ 背景色オブジェクト除外
                filtered_objects = self._filter_background_objects(
                    objects_data[connectivity], bg_strategy
                )

                # ⑥ カテゴリ分け
                categories = self._categorize_objects(
                    filtered_objects, transformation_patterns, pattern_idx
                )

                # ⑦ 部分プログラム生成
                partial_program = self._generate_partial_program(
                    connectivity, bg_strategy, categories
                )

                partial_programs.append(partial_program)

            return {
                'success': True,
                'partial_programs': partial_programs,
                'all_partial_programs': partial_programs,  # すべての部分プログラム（成功したもののみ）
                'categories': all_categories,  # すべてのカテゴリを返す
                'background_colors': background_colors,
                'transformation_patterns': transformation_patterns,
                'category_var_mappings': all_category_var_mappings,  # カテゴリIDと変数名の対応関係 {pattern_key: {category_id: variable_name}}
                'debug_info': debug_info  # デバッグ情報を追加
            }
```

---

## ⚙️ 設定パラメータ

```python
@dataclass
class ObjectMatchingConfig:
    """オブジェクトマッチング設定"""
    # 連結性
    connectivities: List[int] = field(default_factory=lambda: [4, 8])

    # カテゴリ分けパターン数（連結性ごとに個別設定）
    # 463パターン: 1段階構造で全組み合わせを網羅するために必要
    num_category_patterns_4: int = 463  # 4連結のカテゴリ分けパターン数
    num_category_patterns_8: int = 463  # 8連結のカテゴリ分けパターン数

    # 類似度閾値
    similarity_threshold: float = 0.7  # 1対1対応を確立するための最小類似度
                                       # この閾値未満の場合、対応関係は確立されない
    shape_similarity_threshold_min: float = 0.6  # 形状類似度の最小閾値（カテゴリ分け用）
    shape_similarity_threshold_max: float = 0.9  # 形状類似度の最大閾値（カテゴリ分け用）
    position_similarity_threshold_min: float = 0.5  # 位置類似度の最小閾値（カテゴリ分け用）
    position_similarity_threshold_max: float = 0.8  # 位置類似度の最大閾値（カテゴリ分け用）

    # 回転・反転の考慮
    enable_rotation_flip_matching: bool = True  # 回転・反転を考慮するか
    candidate_similarity_threshold: float = 0.5  # 候補を絞り込むための閾値（通常の閾値より低め）
                                                 # この閾値以上の候補に対して回転・反転を考慮した詳細な比較を行う
    max_rotation_flip_candidates: int = 10  # 回転・反転を考慮する候補の最大数（計算コスト制御）
                                            # 候補が多い場合、上位N個のみ回転・反転を考慮

    # 分割・統合の検出閾値（FIT_SHAPE_COLORベース、正規化スコア）
    split_score_threshold: float = 10000.0  # 分割の正規化スコア閾値（重複ピクセル率が0.1以上の場合の最小スコア）
    merge_score_threshold: float = 10000.0  # 統合の正規化スコア閾値（重複ピクセル率が0.1以上の場合の最小スコア）
    # 正規化スコアの範囲（信頼度計算用）
    fit_shape_color_score_min: float = 10000.0  # 正規化スコアの最小値（重複ピクセル率0.1に対応）
    fit_shape_color_score_max: float = 100000.0  # 正規化スコアの最大値（重複ピクセル率1.0に対応）
    fit_shape_color_score_range: float = 90000.0  # 正規化スコアの範囲（max - min）
    # 信頼度計算の重み
    confidence_score_weight: float = 0.6  # スコアの重み
    confidence_count_weight: float = 0.4  # オブジェクト数の重み
    # オブジェクト数の影響係数
    confidence_count_base: float = 0.5  # 2個の場合の基本値
    confidence_count_increment: float = 0.15  # 1個増えるごとの増分

    # 重みの範囲
    color_weight_min: float = 0.2
    color_weight_max: float = 0.5
    shape_weight_min: float = 0.2
    shape_weight_max: float = 0.5
    position_weight_min: float = 0.1
    position_weight_max: float = 0.3
    correspondence_weight_min: float = 0.1  # 対応関係の種類の重み
    correspondence_weight_max: float = 0.3
    transformation_weight_min: float = 0.1  # 変換パターンの重み
    transformation_weight_max: float = 0.3

    # 背景色決定
    color_consistency_threshold: float = 0.8  # 色の一致度閾値
    bg_color_consistency_threshold: float = 0.8  # 背景色の一致度閾値

    # カテゴリ分け
    min_objects_per_category: int = 1
    max_categories: int = 10

    # パフォーマンス最適化設定
    max_objects_for_vector_distance: int = 100  # この数以上のオブジェクトの場合、K-meansクラスタリングを使用

    # K-meansクラスタリング設定
    kmeans_n_clusters_ratio: float = 0.5  # クラスタ数 = sqrt(n / 2) * ratio（デフォルト: sqrt(n/2)）
    kmeans_random_state: int = 42  # 再現性のための乱数シード

    # 使用可能な特徴量のリスト（すべての特徴量を1段階でフラットに定義）
    # このリストから、指定された要素の組み合わせを生成する
    # 例: available_features=['color', 'x']と指定すると、[color], [x], [color, x]の組み合わせを生成
    #
    # 注意: グループ要素（x, y, width, heightなど）は個別には含まれない
    # - グループ化が有効な場合: グループ名（dimensions, symmetry, patch_hash）と単独特徴量のみが選択可能
    # - グループ化が無効な場合: グループ要素を個別に選択したい場合は、available_featuresで明示的に指定する必要がある
    #   ただし、通常はグループ化を有効にして、グループ名を使用することを推奨
    all_available_features: List[str] = field(default_factory=lambda: [
        # グループ名（グループ化が有効な場合に使用）
        'symmetry',  # symmetry_x, symmetry_yを含む
        'patch_hash',  # patch_hash_3x3, patch_hash_2x2を含む
        'dimensions',  # width, heightを含む（グループ名）
        # 単独特徴量
        'color',
        'x',  # 位置情報（x座標）
        'y',  # 位置情報（y座標）
        'size',  # 単独特徴量（width * height、dimensionsグループとは別）
        'hole_count',
        'downscaled_bitmap',
        'contour_direction',
        'skeleton',
        'local_centroid',
    ])

    # 使用する特徴量のリスト（Noneの場合はall_available_featuresを使用）
    # このリストから、すべての組み合わせを生成する
    # 例: ['color', 'x']と指定すると、[color], [x], [color, x]の組み合わせを生成
    available_features: Optional[List[str]] = None  # Noneの場合はall_available_featuresを使用

    # 特徴量のグループ化設定（一緒に選択されるべき特徴量をグループ化）
    # グループ化により、組み合わせ数を削減できる
    # 例: {'dimensions': ['width', 'height']} → 'dimensions'を選択すると、'width'と'height'の両方が含まれる
    feature_groups: Dict[str, List[str]] = field(default_factory=lambda: {
        'symmetry': ['symmetry_x', 'symmetry_y'],  # 対称性は常に一緒に使用
        'patch_hash': ['patch_hash_3x3', 'patch_hash_2x2'],  # パッチハッシュは常に一緒に使用
        'dimensions': ['width', 'height'],  # 幅と高さは常に一緒に使用（size単独特徴量とは別）
    })

    # グループ単位で選択するか、個別に選択するか
    # True: グループ単位で選択（組み合わせ数を削減）
    # False: 個別に選択（従来通り、すべての組み合わせを生成）
    use_feature_groups: bool = True  # デフォルト: True（グループ単位で選択）

    # 排除推奨の組み合わせ（一緒に使用すべきでない特徴量の組み合わせ）
    # これらの組み合わせは生成されない
    excluded_combinations: List[List[str]] = field(default_factory=lambda: [
        ['width', 'height', 'size'],  # 3つすべては冗長（size ≈ width * height）
    ])

    # hole_count特徴量の条件付き使用設定
    # 入力グリッドの一定割合以上のオブジェクトに穴がある場合のみ、hole_count特徴量を使用する
    # Noneの場合は常に使用（条件なし）
    hole_count_min_ratio: Optional[float] = 0.3  # デフォルト: 30%以上のオブジェクトに穴がある場合のみ使用
```

---

## 🎯 カテゴリ分けアルゴリズムの詳細

### アルゴリズム: 階層的クラスタリング風アプローチ

1. **初期化**:
   - 各オブジェクトを個別のカテゴリとして初期化

2. **類似度マトリックスの計算**:
   ```python
   similarity_matrix = []
   for obj1 in objects:
       for obj2 in objects:
           if obj1 != obj2:
               similarity = calculate_similarity(obj1, obj2, params)
               similarity_matrix.append((obj1, obj2, similarity))
   ```

3. **クラスタリング**:
   - 類似度が高い順にソート
   - 類似度が閾値以上のオブジェクトペアを同じカテゴリに統合
   - 最小オブジェクト数制約を満たすまで繰り返し

4. **カテゴリの特徴抽出**:
   - 各カテゴリの代表色（最頻出色）
   - 代表形状（平均サイズ、平均アスペクト比）
   - 注意: 代表変換パターン（representative_transformation_pattern）は生成されていません
   - 変換パターンは個別オブジェクト（ObjectInfo）のtransformation_pattern属性に設定され、
     候補生成器で集約して使用されます

5. **カテゴリの検証**:
   - 各カテゴリに最低1つのオブジェクトが属していることを確認
   - 各入力グリッドで、そのカテゴリに属するオブジェクトが存在するかを確認

---

## 📝 部分プログラムの例

### 例1: 色による分類

```python
"objects = GET_ALL_OBJECTS(4)
objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))
objects1 = FILTER(objects, EQUAL(GET_COLOR($obj), 1))
objects2 = FILTER(objects, EQUAL(GET_COLOR($obj), 2))"
```

### 例2: 形状による分類（将来的に実装）

```python
"objects = GET_ALL_OBJECTS(4)
objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))
objects1 = FILTER(objects, LESS(GET_SIZE($obj), 5))
objects2 = FILTER(objects, GREATER(GET_SIZE($obj), 10))"
```

### 例3: 複合条件（将来的に実装）

```python
"objects = GET_ALL_OBJECTS(4)
objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))
objects1 = FILTER(objects, AND(EQUAL(GET_COLOR($obj), 1), LESS(GET_SIZE($obj), 5)))"
```

### 例4: 消失パターンを含む分類

```python
"objects = GET_ALL_OBJECTS(4)
objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))
# 消失するオブジェクトのカテゴリ（将来的に実装）
objects1 = FILTER(objects, EQUAL(GET_COLOR($obj), 3))"
```

---

## 🔄 今後の拡張

### Phase 1: 基本実装（現在）

- ✅ 色による分類
- ✅ 背景色の処理
- ✅ 基本的な部分プログラム生成

### Phase 2: 形状・位置による分類（実装完了）

- ✅ 形状による分類（サイズ、アスペクト比）
- ✅ 位置による分類（重心位置、グリッド内の位置）

### Phase 3: 高度な分類（実装完了）

- ✅ 変換パターンによる分類（変換パターンの重みを強化）
- ✅ 複合条件による分類（AND/OR条件の組み合わせ）
- ✅ MERGE、EXTRACTなどの操作の追加（統合・分割パターンに基づく操作追加）
- ✅ 新規出現パターンの予測（新規出現オブジェクトの特徴分析）
- ✅ パフォーマンス最適化（類似度計算のキャッシュ機能）

---

## 📊 評価指標

### 部分プログラムの品質指標

1. **カテゴリの一貫性**: 各カテゴリがタスク全体で一貫しているか
2. **オブジェクトの網羅性**: すべてのオブジェクトがカテゴリに分類されているか
3. **カテゴリ数の適切性**: カテゴリ数が適切か（多すぎず、少なすぎず）
4. **部分プログラムの有効性**: 生成された部分プログラムが深層学習モデルに有効か

---

## 🚀 実装計画

### Step 1: 基本構造の実装

- [ ] `RuleBasedObjectMatcher`クラスの作成
- [ ] オブジェクト抽出の実装
- [ ] 1対1対応の検出実装

### Step 2: 変換パターン分析の実装

- [x] 消失・新規出現の検出実装
- [x] 分割（Split）パターンの検出実装（FIT_SHAPE_COLORベース）
  - [x] `detect_splits()`メソッドの実装
  - [x] `_calculate_fit_shape_color_score()`メソッドの実装（FIT_SHAPE_COLORと同じロジック、obj2による正規化あり）
  - [x] `_find_best_fit_shape_color()`メソッドの実装
  - [x] `_calculate_edge_count()`メソッドの実装
  - [x] `_calculate_bbox_size()`メソッドの実装
  - 注: `_subtract_shapes()`メソッドは実装されているが、形状減算ロジックが廃止されたため、現在は使用されていない
- [x] 統合（Merge）パターンの検出実装（FIT_SHAPE_COLORベース）
  - [x] `detect_merges()`メソッドの実装
  - [x] 分割と同じヘルパーメソッドを使用
- [x] 変換パターンの信頼度計算

### Step 4: 背景色処理の実装

- [ ] 背景色推論の実装
- [ ] 背景色決定ロジックの実装
- [ ] 背景色オブジェクト除外の実装

### Step 5: カテゴリ分けの実装

- [ ] カテゴリ分けアルゴリズムの実装
- [ ] 複数パターン生成の実装
- [ ] カテゴリ特徴抽出の実装
- [ ] 消失パターンを含むカテゴリ分け

### Step 6: 部分プログラム生成の実装

- [ ] 部分プログラム生成ロジックの実装
- [ ] FILTER条件生成の実装
- [ ] 消失パターンを含む部分プログラム生成
- [ ] 部分プログラムの検証

### Step 7: 統合とテスト

- [ ] `ProgramSynthesisEngine`への統合
- [ ] テストケースの作成（分割・統合を含む）
- [ ] パフォーマンス評価

---

## 📝 更新履歴

- **2025-01-XX**: 分割・統合の検出方法をFIT_SHAPE_COLORベースの正規化スコア方式に変更。繰り返し探索による検証を実装（形状減算は廃止、obj2による正規化は必要）。重複ピクセル率を「位置+色」と「位置のみ」の2つに分割。隣接辺率から色の一致条件を除外。
- **2025-11-18**: オブジェクトマッチングのバグ修正完了、オブジェクト対応関係検出の本格実装完了（多特徴量マッチング）、実装改善完了を反映
- **2025-01-XX**: 初版作成

# DSLコマンド クイックリファレンス（実装済みのみ）

**総コマンド数**: 89コマンド + 制御構造3個 + グローバル関数1個（2025-12-10時点）⭐v4.1更新

> 出典: `src/core_systems/executor/parsing/interpreter.py` の `execute_command` 実装を直接確認して作成。表内の名称・引数は同ファイルの実装と一致しています。

## カテゴリサマリー

| カテゴリ | コマンド数 | 概要 |
|---------|-----------|------|
| オブジェクト変換 | 22 | 位置・形状・色を変更する操作 |
| 生成・抽出・分割 | 10 | オブジェクトの生成、合成、抽出、分解 |
| 関係・距離計算 | 5 | オブジェクト間の関係量を計測 |
| 情報取得 | 26 | サイズ・座標・属性などの取得 |
| 配列・集合操作 | 11 | List 操作とパターン整形 |
| 判定関数 | 4 | 真偽値判定 |
| 数値演算 | 5 | 整数向け算術演算 |
| 比較演算 | 4 | 整数比較 |
| 論理演算 | 2 | ブール演算 |
| グローバル関数 | 1 | `RENDER_GRID` |
| 制御構造 | 3 | `FOR` / `WHILE` / `IF` |

以下、各カテゴリの詳細です。引数はコマンド実装で想定される順序を明示し、戻り値は DSL 実行系が返す型を記載しています。

---

## オブジェクト変換（22）

| コマンド | 引数 (型) | 戻り値 (型) | 概要 |
|----------|-----------|-------------|------|
| `MOVE` | obj (str); dx (int); dy (int) | str (object_id) | オブジェクトを相対移動 |
| `TELEPORT` | obj (str); x (int); y (int) | str (object_id) | 絶対座標に移動 |
| `SLIDE` | obj (str); direction (str); obstacles (List[str]) | str (object_id) | 障害物にぶつかるまで平行移動 |
| `ROTATE` | obj (str); angle (int); cx (int, optional); cy (int, optional) | str (object_id) | 90度刻みで回転（中心指定可） |
| `FLIP` | obj (str); axis (str) | str (object_id) | 水平または垂直に反転 |
| `SCALE` | obj (str); factor (int) | str (object_id) | 座標を整数倍に拡大 |
| `SCALE_DOWN` | obj (str); divisor (int) | str (object_id) | 均等縮小（1/divisor） |
| `EXPAND` | obj (str); pixels (int) | str (object_id) | 指定ピクセル厚で拡張 |
| `FLOW` | obj (str); direction (str); obstacles (List[str]) | str (object_id) | 液体シミュレーション的に流下 |
| `DRAW` | obj (str); x (int); y (int) | str (object_id) | 指定座標へ軌跡を描画 |
| `LAY` | obj (str); direction (str); obstacles (List[str]) | str (object_id) | 重力落下のように配置 |
| `CROP` | obj (str); x (int); y (int); w (int); h (int) | str (object_id) | 指定矩形で切り取り |
| `SET_COLOR` | obj (str); color (int) | str (object_id) | 単一色へ置換 |
| `FILL_HOLES` | obj (str); color (int) | str (object_id) | 内部の穴を塗りつぶし |
| `OUTLINE` | obj (str); color (int) | str (object_id) | 外周を抽出し塗りつぶし |
| `HOLLOW` | obj (str) | str (object_id) | 中空化（縁のみ残す） |
| `FIT_SHAPE` | obj1 (str); obj2 (str) | str (object_id) | obj2 へ形状フィット |
| `FIT_SHAPE_COLOR` | obj1 (str); obj2 (str) | str (object_id) | 形状と色を同時フィット |
| `FIT_ADJACENT` | obj1 (str); obj2 (str) | str (object_id) | 隣接配置を最適化 |
| `ALIGN` | obj (str); mode (str) | str (object_id) | オブジェクトを整列（left/right/top/bottom/center_x/center_y/center） |
| `PATHFIND` | obj (str); target_x (int); target_y (int); obstacles (List[str]) | str (object_id) | 経路探索で移動 |

方向を指定するコマンド (`SLIDE` / `FLOW` / `LAY`) は `"X"`, `"Y"`, `"-X"`, `"-Y"` の 4 方位を受け付けます。`ROTATE` は 0/90/180/270 度、`SCALE` と `SCALE_DOWN` は整数のみ受理。

---

## 生成・抽出・分割（10）

| コマンド | 引数 (型) | 戻り値 (型) | 概要 |
|----------|-----------|-------------|------|
| `MERGE` | objects (List[str]) | str (object_id) | 複数オブジェクトを連結して新規オブジェクトを生成 |
| `CREATE_LINE` | x (int); y (int); length (int); direction (str); color (int) | str (object_id) | 8 方向対応の線オブジェクトを生成 |
| `CREATE_RECT` | x (int); y (int); w (int); h (int); color (int) | str (object_id) | 塗りつぶし矩形を生成 |
| `EXTRACT_BBOX` | obj (str); color (int) | str (object_id) | 外接矩形を抽出し指定色で描画 |
| `EXTRACT_RECTS` | obj (str) | List[str] | オブジェクトを矩形成分ごとに抽出 |
| `EXTRACT_HOLLOW_RECTS` | obj (str) | List[str] | 中空矩形のみを抽出 |
| `EXTRACT_LINES` | obj (str) | List[str] | 直線成分のみを抽出 |
| `SPLIT_CONNECTED` | obj (str); connectivity (int {4,8}) | List[str] | 連結成分ごとに分割 |
| `BBOX` | obj (str); color (int, optional) | str (object_id) | 外接矩形を抽出（色指定可） |
| `TILE` | obj (str); count_x (int); count_y (int) | List[str] | オブジェクトをタイル状に複製 |

`CREATE_LINE` の `direction` は `"X"`, `"Y"`, `"-X"`, `"-Y"`, `"XY"`, `"-XY"`, `"X-Y"`, `"-X-Y"` を指定します。`MERGE` は入力順のまま統合した新規オブジェクト ID を返します。

---

## 関係・距離計算（5）

| コマンド | 引数 (型) | 戻り値 (型) | 概要 |
|----------|-----------|-------------|------|
| `INTERSECTION` | obj1 (str); obj2 (str) | str (object_id) | 共通領域のみを抽出 |
| `SUBTRACT` | obj1 (str); obj2 (str) | str (object_id) | obj1 から obj2 のピクセルを除去 |
| `COUNT_HOLES` | obj (str) | int | 穴の数をカウント |
| `COUNT_ADJACENT` | obj1 (str); obj2 (str) | int | 境界で接するピクセル数 |
| `COUNT_OVERLAP` | obj1 (str); obj2 (str) | int | 重複ピクセル数 |

---

## 情報取得（26）

| コマンド | 引数 (型) | 戻り値 (型) | 概要 |
|----------|-----------|-------------|------|
| `GET_ALL_OBJECTS` | connectivity (int {4,8}) | List[str] | 指定した連結性で全オブジェクト ID を取得 |
| `GET_BACKGROUND_COLOR` | - | int | 入力グリッドの背景色 |
| `GET_INPUT_GRID_SIZE` | - | List[int] | `[width, height]` を返す |
| `GET_SIZE` | obj (str) | int | ピクセル総数 |
| `GET_WIDTH` | obj (str) | int | 幅（ピクセル数） |
| `GET_HEIGHT` | obj (str) | int | 高さ（ピクセル数） |
| `GET_X` | obj (str) | int | 左端の X 座標 |
| `GET_Y` | obj (str) | int | 上端の Y 座標 |
| `GET_COLOR` | obj (str) | int | 最頻出色 |
| `GET_COLORS` | obj (str) | List[int] | 出現色一覧 |
| `GET_SYMMETRY_SCORE` | obj (str); axis (str {"X","Y"}) | int | 軸対称度（0〜100） |
| `GET_LINE_TYPE` | obj (str) | str | `"X"`, `"Y"`, `"XY"`, `"-XY"`, `"none"` |
| `GET_RECTANGLE_TYPE` | obj (str) | str | `"solid"` または `"hollow"` |
| `GET_DISTANCE` | obj1 (str); obj2 (str) | int | ユークリッド距離 |
| `GET_X_DISTANCE` | obj1 (str); obj2 (str) | int | X 軸方向距離 |
| `GET_Y_DISTANCE` | obj1 (str); obj2 (str) | int | Y 軸方向距離 |
| `GET_ASPECT_RATIO` | obj (str) | int | アスペクト比（100倍の整数値） |
| `GET_DENSITY` | obj (str) | int | 密度（100倍の整数値） |
| `GET_CENTER_X` | obj (str) | int | 中心X座標 |
| `GET_CENTER_Y` | obj (str) | int | 中心Y座標 |
| `GET_MAX_X` | obj (str) | int | 右端のX座標 |
| `GET_MAX_Y` | obj (str) | int | 下端のY座標 |
| `GET_CENTROID` | obj (str) | str | 重心方向（"X", "Y", "C"など） |
| `GET_DIRECTION` | obj1 (str); obj2 (str) | str | obj1からobj2への方向 |
| `GET_NEAREST` | obj (str); candidates (List[str]) | str (object_id) | 最も近いオブジェクトを取得 |

---

## 配列・集合操作（11）

| コマンド | 引数 (型) | 戻り値 (型) | 概要 |
|----------|-----------|-------------|------|
| `APPEND` | array_name (str); value (Any) | List[Any] | 配列名を指定して値を末尾に追加（非破壊） |
| `LEN` | array (List[Any]) | int | 要素数を取得 |
| `REVERSE` | array (List[Any]) | List[Any] | 配列を逆順にする（新配列を返す） |
| `CONCAT` | array1 (List[Any]); array2 (List[Any]) | List[Any] | 2 配列を結合 |
| `FILTER` | objects (List[str]); condition_expr (式) | List[str] | 条件式でフィルタリング |
| `SORT_BY` | array (List[str]); key_expr (式); order (str {"asc","desc"}) | List[str] | 指定キーで並べ替え（新配列を返す） |
| `EXTEND_PATTERN` | objects (List[str]); side (str {"front","end"}); count (int) | List[str] | パターンを繰り返し延長 |
| `ARRANGE_GRID` | objects (List[str]); columns (int); cell_width (int); cell_height (int) | List[str] | グリッド状に整列した ID を返す |
| `MATCH_PAIRS` | objects1 (List[str]); objects2 (List[str]); condition_expr (式) | List[str] | 条件に合うペアを抽出 |
| `EXCLUDE` | array (List[str]); targets (List[str]) | List[str] | 完全一致のオブジェクトを除外 |
| `CREATE_ARRAY` | - | List[Any] | 空配列を生成 |

`FILTER` と `SORT_BY` の第 2 引数は AST ノードとして渡されるため、条件式内では `$obj` プレースホルダーを用います。`APPEND` は配列名（識別子）を第 1 引数に指定。

---

## 判定関数（4）

| コマンド | 引数 (型) | 戻り値 (型) | 概要 |
|----------|-----------|-------------|------|
| `IS_INSIDE` | obj (str); x (int); y (int); width (int); height (int) | bool | 矩形領域に完全に含まれるか判定 |
| `IS_SAME_SHAPE` | obj1 (str); obj2 (str) | bool | 形状が一致するか判定 |
| `IS_SAME_STRUCT` | obj1 (str); obj2 (str) | bool | 形状と色情報が一致するか判定 |
| `IS_IDENTICAL` | obj1 (str); obj2 (str) | bool | ピクセル単位で完全一致か判定 |

---

## 数値・比較・論理演算

| 種別 | コマンド | 引数 (型) | 戻り値 (型) | 概要 |
|------|----------|-----------|-------------|------|
| 算術 (5) | `ADD`, `SUB`, `MULTIPLY`, `DIVIDE`, `MOD` | a (int); b (int) | int | 整数演算のみ受理 |
| 比較 (4) | `EQUAL`, `NOT_EQUAL`, `GREATER`, `LESS` | a (int); b (int) | bool | 整数同士を比較 |
| 論理 (2) | `AND`, `OR` | a (bool); b (bool) | bool | ブール演算 |

---

## グローバル関数・制御構造

| コマンド / 構造 | 引数 (型) | 戻り値 (型) | 概要 |
|-----------------|-----------|-------------|------|
| `RENDER_GRID` | objects (List[str]); bg (int); width (int); height (int); x (int, optional); y (int, optional) | None | 出力グリッドを確定し `execution_context` に保存 |
| `FOR` | var (識別子); count (int) | None | 指定回数ループ（最大 1000 回まで） |
| `WHILE` | condition (式) | None | 条件が偽になるまで反復 |
| `IF` | condition (式); then_block (ブロック); else_block (ブロック, optional) | None | 条件分岐を実行 |

`RENDER_GRID` 呼び出し後は `program_terminated` フラグが立ち、以降のノードはスキップされます。`FOR` ループ内ではオブジェクト数が過剰になった場合に即座に `SilentException` が投げられます。

---

## プレースホルダーと式評価のルール

- `$obj` は `FILTER`, `SORT_BY`, `MATCH_PAIRS` の条件式で現在処理中のオブジェクト ID を参照するための予約変数です。使用しない場合は `Interpreter` により自動補完されません。
- `APPEND` の第 1 引数は変数名として扱われるため、文字列リテラルではなく識別子を渡してください。
- `SORT_BY` は安定ソート。`order="asc"` で小さい順、`order="desc"` で大きい順となり、戻り値は新しい配列です。
- `EXCLUDE` は完全一致（座標・サイズ・ピクセル内容）で比較を行います。配列や対象は自動的に `None` が除外されます。

---

## GET 系コマンドの注意事項

- 引数なしのコマンド（例: `GET_BACKGROUND_COLOR()`）は必ず丸括弧を付与します。DSL 側で省略するとシンタックスエラーになります。
- `GET_LINE_TYPE` は `"none"` を返す場合があるため、条件分岐の際は必ずフォールバックを用意してください。
- `GET_COLORS` は出現順にソートされていない配列を返します。統計処理が必要な場合は Python 側で整形してください。

---

## 参考: 実装上の制約

- オブジェクト ID を含む配列の長さが `MAX_OBJECTS_IN_VARIABLE` を超えると即座に実行が停止します（`src/core_systems/executor/core.py` 参照）。
- `EXCLUDE` と `FILTER` は大量のオブジェクトを扱う際に警告ログを残す設計です。性能調査時はログレベルに注意してください。
- 数値演算はすべて整数（`int`）のみ受理します。浮動小数を渡すと `TypeError` になります。

---

## 更新履歴

- **v4.1 (2025-12-10)**: インタープリター実装を再確認し、89コマンドに更新。`REVERSE`, `ALIGN`, `PATHFIND`, `BBOX`, `TILE` および追加の情報取得コマンド（`GET_ASPECT_RATIO`, `GET_DENSITY`, `GET_CENTER_X/Y`, `GET_MAX_X/Y`, `GET_CENTROID`, `GET_DIRECTION`, `GET_NEAREST`）を追加。
- **v4.0 (2025-11-10)**: インタープリター実装を再点検し、74 コマンドを再分類。`CREATE_ARRAY` や関係演算の補足を追記。
- **v3.8 (2025-11-10)**: スクリプト整理後の概要更新（旧版）。

必要に応じて `src/core_systems/executor/parsing/interpreter.py` を参照し、追加コマンドが実装された際には本表も更新してください。

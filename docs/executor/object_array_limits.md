# オブジェクト配列の即座終了条件

## 概要
プログラム実行中にオブジェクト数やピクセル数が上限を超えた場合、`SilentException`が発生し、プログラム実行が即座に終了します。

## 上限定数（`src/core_systems/executor/core.py`）

```python
MAX_OBJECTS_DURING_EXECUTION = 10000  # プログラム実行中の総オブジェクト数上限（ARC問題では通常100個以下）
MAX_OBJECTS_IN_VARIABLE = 512         # 1つの変数（配列）に格納できる最大オブジェクト数
MAX_TOTAL_PIXELS_IN_VARIABLE = 4000  # 1つの変数（配列）に蓄積できるオブジェクト総ピクセル数の上限
MAX_OBJECTS_FOR_EXCLUDE = 512        # EXCLUDE操作で処理できる最大オブジェクト数
```

## 即座終了条件の詳細

### 1. 変数（配列）へのオブジェクト数上限
**場所**: `src/core_systems/executor/parsing/interpreter.py` - `_enforce_array_limits()`

**条件**:
- 1つの変数（配列）に格納されたオブジェクト数が **512個** を超えた場合

**エラーメッセージ**:
```
変数 '{variable_name}' に格納されているオブジェクト数が上限を超えました
（{object_count}個 > 512個）。プログラムに問題がある可能性があります。
```

**発生タイミング**:
- 変数への代入時（`_enforce_array_limits()`が呼ばれたとき）

### 2. 変数（配列）へのピクセル数上限
**場所**: `src/core_systems/executor/parsing/interpreter.py` - `_enforce_array_limits()`

**条件**:
- 1つの変数（配列）に蓄積されたオブジェクトの総ピクセル数が **4000ピクセル** を超えた場合

**エラーメッセージ**:
```
変数 '{variable_name}' に格納されているオブジェクトの総ピクセル数が上限を超えました
（{total_pixels}ピクセル > 4000ピクセル）。プログラムに問題がある可能性があります。
```

**発生タイミング**:
- 変数への代入時（`_enforce_array_limits()`が呼ばれたとき）

### 3. FORループ実行中のオブジェクト数上限
**場所**: `src/core_systems/executor/parsing/interpreter.py` - FORループ処理

**条件**:
- FORループ実行中、プログラム全体のオブジェクト数が **10000個** を超えた場合

**エラーメッセージ**:
```
FORループ実行中、オブジェクト数が上限を超えました
（{object_count}個 > 10000個）。プログラムに問題がある可能性があります。
```

**発生タイミング**:
- FORループの各イテレーション後

### 4. WHILEループ実行中のオブジェクト数上限
**場所**: `src/core_systems/executor/parsing/interpreter.py` - WHILEループ処理

**条件**:
- WHILEループ実行中、プログラム全体のオブジェクト数が **10000個** を超えた場合

**エラーメッセージ**:
```
WHILEループ実行中、オブジェクト数が上限を超えました
（{object_count}個 > 10000個）。プログラムに問題がある可能性があります。
```

**発生タイミング**:
- WHILEループの各イテレーション後

### 5. プログラム実行中のオブジェクト数上限
**場所**: `src/core_systems/executor/core.py` - プログラム実行後

**条件**:
- プログラム実行後、オブジェクト数が **10000個** を超えた場合

**エラーメッセージ**:
```
プログラム実行中、オブジェクト数が上限を超えました
（{object_count_after}個 > MAX_OBJECTS_DURING_EXECUTION個、実行前={object_count_before}個）。
プログラムに問題がある可能性があります。
```

**発生タイミング**:
- プログラム実行完了時

### 6. EXCLUDE操作のオブジェクト数上限
**場所**: `src/core_systems/executor/parsing/interpreter.py` - EXCLUDE操作処理

**条件**:
- EXCLUDE操作で処理する配列または対象のオブジェクト数が **512個** を超えた場合

**エラーメッセージ**:
```
EXCLUDE: オブジェクト数が上限を超えています
（array={len(array)}, targets={len(targets)}, 上限=512）。
プログラムに問題がある可能性があります。
```

**発生タイミング**:
- EXCLUDE操作の実行時

## エラー処理の流れ

1. **`SilentException`の発生**
   - 上記のいずれかの条件を満たした場合、`SilentException`が発生

2. **`interpreter.py`または`core.py`でキャッチ**
   - エラーメッセージがログに出力される

3. **`validate_nodes_and_adjust_objects()`で処理**
   - `SilentException`を捕捉
   - 最初のペアで発生: タスク全体を破棄（`return None, None, None, None, None`）
   - 2番目以降のペアで発生: ペアをスキップ（`ValueError`を発生）

4. **`core_executor.py`で処理**
   - ピクセル数上限: `ValueError("プログラム実行エラーによりタスクが破棄されました（ピクセル数上限）")`を発生
   - オブジェクト数上限: `ValueError("プログラム実行エラーによりタスクが破棄されました（オブジェクト数上限）")`を発生

5. **`main.py`で処理**
   - `handle_pair_generation_error()`で、プログラム実行エラーとして即座にスキップ

## まとめ

即座に終了する条件:
- ✅ 1つの変数（配列）に512個以上のオブジェクトが格納された場合
- ✅ 1つの変数（配列）に4000ピクセル以上のオブジェクトが蓄積された場合
- ✅ FOR/WHILEループまたはプログラム実行中に10000個以上のオブジェクトが存在する場合
- ✅ EXCLUDE操作で512個以上のオブジェクトを処理しようとした場合

これらはすべて`SilentException`として処理され、プログラム実行が即座に終了します。

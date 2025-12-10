"""
背景色決定ロジック

Nodeリストから適切な背景色を決定する
"""
from typing import List, Optional, Any
import random


# ARC-AGI2統計に基づく背景色の分布（実データ分析結果に基づく）
# ARC-AGI2実データ: 背景色0が67.1%、色7が8.9%、色8が6.1%など
# 背景色0の確率を67.1%に調整し、他の色（特に7, 8）の確率を上げる
BACKGROUND_COLOR_PROBABILITY_0 = 0.671  # 黒（背景）: 67.1%（87%→67.1%に減少）
BACKGROUND_COLOR_PROBABILITY_7 = 0.089  # オレンジ: 8.9%
BACKGROUND_COLOR_PROBABILITY_8 = 0.061  # 水色: 6.1%
BACKGROUND_COLOR_PROBABILITY_OTHER = (1.0 - BACKGROUND_COLOR_PROBABILITY_0 - BACKGROUND_COLOR_PROBABILITY_7 - BACKGROUND_COLOR_PROBABILITY_8) / 7  # 残りを7色で均等分割

BACKGROUND_COLOR_DISTRIBUTION = {
    0: BACKGROUND_COLOR_PROBABILITY_0,  # 黒（背景）: 67.1%
    1: BACKGROUND_COLOR_PROBABILITY_OTHER,  # 青
    2: BACKGROUND_COLOR_PROBABILITY_OTHER,  # 赤
    3: BACKGROUND_COLOR_PROBABILITY_OTHER,  # 緑
    4: BACKGROUND_COLOR_PROBABILITY_OTHER,  # 黄
    5: BACKGROUND_COLOR_PROBABILITY_OTHER,  # グレー
    6: BACKGROUND_COLOR_PROBABILITY_OTHER,  # マゼンタ
    7: BACKGROUND_COLOR_PROBABILITY_7,  # オレンジ: 8.9%
    8: BACKGROUND_COLOR_PROBABILITY_8,  # 水色: 6.1%
    9: BACKGROUND_COLOR_PROBABILITY_OTHER,  # 茶色
}


def decide_background_color(nodes: Optional[List[Any]] = None, default: int = 0) -> int:
    """Nodeリストから背景色を決定

    Args:
        nodes: プログラムのNodeリスト（Noneまたは空の場合は統計分布に基づいてランダム選択）
        default: デフォルトの背景色（決定できない場合、現在は使用されていない）

    Returns:
        決定された背景色（0-9）
    """
    if not nodes or len(nodes) < 2:
        # ノードがNone、空、または2つ未満の場合は統計分布に基づいてランダム選択
        return _select_random_background_color()

    # 1. プログラムの2行目（インデックス1）のFILTER文を確認
    filter_info = _check_filter_on_line_2(nodes)

    if filter_info is None:
        # ①存在しない場合 → 2.①（ランダム）に移行
        return _select_random_background_color()

    filter_type, color_value = filter_info

    if filter_type == 'get_background_color':
        # ②GET_BACKGROUND_COLOR()が使われている場合 → 2.①（ランダム）に移行
        return _select_random_background_color()

    elif filter_type == 'literal_color' and color_value is not None:
        # ③具体的な色番号（0~9）が使われている場合 → 2.②（その色番号）に移行
        return color_value

    else:
        # その他の場合はランダム
        return _select_random_background_color()


def _check_filter_on_line_2(nodes: List[Any]) -> Optional[tuple]:
    """プログラムの2行目に相当するFILTER文を確認

    注意: InitializationNodeの場合、generate()メソッドが複数行の文字列を返す可能性がある
    "objects = GET_ALL_OBJECTS(4)" と "objects = FILTER(objects, ...)" は同じノードから生成される

    Returns:
        (filter_type, color_value) のタプル
        - filter_type: 'get_background_color' | 'literal_color' | None
        - color_value: 色番号（0-9）またはNone
    """
    if len(nodes) == 0:
        return None

    # 1行目（インデックス0）がInitializationNodeかどうか確認
    first_node = nodes[0]
    first_node_type = type(first_node).__name__

    if first_node_type == 'InitializationNode':
        # InitializationNodeのgenerate()を呼び出して文字列を取得
        # このノードは複数行（"objects = GET_ALL_OBJECTS(...)" と "objects = FILTER(...)"）を返す可能性がある
        generated_code = first_node.generate()

        # 複数行の場合、2行目を抽出（1つ目のノードの2行目として扱う）
        lines = generated_code.split('\n')
        if len(lines) >= 2:
            second_line = lines[1].strip()
            # FILTER文が含まれているか確認
            if 'FILTER' in second_line:
                # 2行目のFILTER文から背景色情報を抽出
                result = _parse_filter_condition_string(second_line)
                if result is not None:
                    return result

        # FILTER文が見つからない、またはパースに失敗した場合
        # 2行目のノードがある場合は、そちらを確認する（後続の処理に続く）
        # ここではNoneを返さず、後続の処理に続ける

    # InitializationNode以外の場合: 2行目（インデックス1）のノードを確認
    if len(nodes) < 2:
        return None

    second_node = nodes[1]
    node_type = type(second_node).__name__

    # FilterNodeまたはAssignmentNodeをチェック
    if node_type == 'FilterNode':
        condition = getattr(second_node, 'condition', None)
        if condition is None:
            return None

        # 文字列として解析を試みる
        if isinstance(condition, str):
            # FilterNodeのconditionは条件部分のみ（例: "NOT_EQUAL(GET_COLOR($obj), 5)"）
            # FILTER文全体ではないので、そのまま解析
            return _parse_filter_condition_string(condition)
        else:
            # Nodeオブジェクトとして解析
            return _parse_filter_condition_node(condition)

    elif node_type == 'AssignmentNode':
        # AssignmentNodeの場合、expressionを確認
        expression = getattr(second_node, 'expression', None)
        if expression is None:
            return None

        # 文字列として解析を試みる
        if isinstance(expression, str):
            # FILTER(...)の文字列から条件を抽出
            return _extract_condition_from_filter_string(expression)
        else:
            # Nodeオブジェクトとして解析（CommandNode FILTER）
            return _parse_filter_command_node(expression)

    return None


def _parse_filter_condition_string(condition_str: str) -> Optional[tuple]:
    """文字列の条件から背景色情報を抽出

    パターン:
    - objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))
    - objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), 5))
    - NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR())  （条件部分のみ）
    - NOT_EQUAL(GET_COLOR($obj), 5)  （条件部分のみ）
    """
    if not isinstance(condition_str, str):
        return None

    import re

    # FILTER文全体の場合と条件部分のみの場合の両方に対応
    # まず、FILTER(...)の中の条件を抽出（FILTER文全体の場合）
    filter_match = re.search(r'FILTER\s*\(\s*\w+\s*,\s*(.+?)\s*\)', condition_str)
    if filter_match:
        condition_part = filter_match.group(1).strip()
    else:
        # FILTERがない場合は、そのまま条件文字列として扱う（条件部分のみの場合）
        condition_part = condition_str.strip()

    # NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()) のパターン
    if 'GET_BACKGROUND_COLOR()' in condition_part:
        return ('get_background_color', None)

    # NOT_EQUAL(GET_COLOR($obj), 0~9) のパターン
    # NOT_EQUAL(GET_COLOR($obj), 数字) のパターンを検索
    pattern = r'NOT_EQUAL\s*\(\s*GET_COLOR\s*\(\s*\$obj\s*\)\s*,\s*(\d+)\s*\)'
    match = re.search(pattern, condition_part)
    if match:
        color_value = int(match.group(1))
        if 0 <= color_value <= 9:
            return ('literal_color', color_value)

    return None


def _extract_condition_from_filter_string(filter_str: str) -> Optional[tuple]:
    """FILTER(...)の文字列から条件を抽出"""
    if not isinstance(filter_str, str):
        return None

    import re
    # FILTER(objects, 条件) のパターンから条件を抽出
    pattern = r'FILTER\s*\(\s*\w+\s*,\s*(.+?)\s*\)'
    match = re.search(pattern, filter_str)
    if match:
        condition_str = match.group(1)
        return _parse_filter_condition_string(condition_str)

    return None


def _parse_filter_condition_node(condition_node: Any) -> Optional[tuple]:
    """Nodeオブジェクトとして条件を解析"""
    if condition_node is None:
        return None

    # NOT_EQUALを確認
    if not _is_not_equal_command(condition_node):
        return None

    # NOT_EQUALの引数を確認
    color_node = _extract_color_from_not_equal(condition_node)

    if color_node is None:
        return None

    # GET_BACKGROUND_COLOR()かどうか確認
    if _is_get_background_color(color_node):
        return ('get_background_color', None)

    # リテラル値（0-9）かどうか確認
    literal_value = _extract_literal_color(color_node)
    if literal_value is not None:
        return ('literal_color', literal_value)

    return None


def _parse_filter_command_node(filter_node: Any) -> Optional[tuple]:
    """FILTERコマンドのNodeから条件を抽出"""
    if filter_node is None:
        return None

    node_type = type(filter_node).__name__

    # CommandNodeでnameがFILTER
    if node_type == 'CommandNode' and hasattr(filter_node, 'name'):
        if filter_node.name == 'FILTER':
            # 引数の2番目が条件
            if hasattr(filter_node, 'arguments') and isinstance(filter_node.arguments, list):
                if len(filter_node.arguments) >= 2:
                    condition = filter_node.arguments[1]
                    return _parse_filter_condition_node(condition)

    return None


def _is_not_equal_command(node: Any) -> bool:
    """NOT_EQUALコマンドかどうかを確認"""
    if node is None:
        return False

    node_type = type(node).__name__

    # CommandNodeでnameがNOT_EQUAL
    if node_type == 'CommandNode' and hasattr(node, 'name'):
        return node.name == 'NOT_EQUAL'

    # 文字列の場合
    if isinstance(node, str):
        return 'NOT_EQUAL' in node

    return False


def _extract_color_from_not_equal(node: Any) -> Optional[Any]:
    """NOT_EQUALコマンドから色に関連するノードを抽出

    NOT_EQUAL(GET_COLOR($obj), X) のXの部分を抽出
    """
    if node is None:
        return None

    # 文字列の場合
    if isinstance(node, str):
        import re
        pattern = r'NOT_EQUAL\s*\(\s*GET_COLOR\s*\(\s*\$obj\s*\)\s*,\s*(.+?)\s*\)'
        match = re.search(pattern, node)
        if match:
            second_arg = match.group(1).strip()
            # GET_BACKGROUND_COLOR()かリテラル値かを判定
            if 'GET_BACKGROUND_COLOR()' in second_arg:
                return 'GET_BACKGROUND_COLOR()'
            # 数字を抽出
            num_match = re.search(r'(\d+)', second_arg)
            if num_match:
                return int(num_match.group(1))
        return None

    # Nodeオブジェクトの場合
    if not _is_not_equal_command(node):
        return None

    if not hasattr(node, 'arguments') or not isinstance(node.arguments, list):
        return None

    if len(node.arguments) < 2:
        return None

    # 第1引数がGET_COLOR($obj)かどうか確認
    first_arg = node.arguments[0]
    if not _is_get_color_with_obj(first_arg):
        return None

    # 第2引数が色指定
    return node.arguments[1]


def _is_get_color_with_obj(node: Any) -> bool:
    """GET_COLOR($obj)かどうかを確認"""
    if node is None:
        return False

    # 文字列の場合
    if isinstance(node, str):
        import re
        pattern = r'GET_COLOR\s*\(\s*\$obj\s*\)'
        return bool(re.search(pattern, node))

    # Nodeオブジェクトの場合
    node_type = type(node).__name__

    if node_type == 'CommandNode' and hasattr(node, 'name'):
        if node.name == 'GET_COLOR':
            # 引数に$objが含まれているか確認
            if hasattr(node, 'arguments') and isinstance(node.arguments, list):
                if len(node.arguments) > 0:
                    arg = node.arguments[0]
                    # VariableNodeまたはPlaceholderNodeで$obj
                    if hasattr(arg, 'variable') and arg.variable == '$obj':
                        return True
                    if type(arg).__name__ == 'PlaceholderNode' and hasattr(arg, 'name') and arg.name == '$obj':
                        return True
                    # 文字列として$objが含まれる場合
                    if isinstance(arg, str) and '$obj' in arg:
                        return True

    return False


def _is_get_background_color(node: Any) -> bool:
    """GET_BACKGROUND_COLOR()かどうかを確認"""
    if node is None:
        return False

    node_type = type(node).__name__

    if node_type == 'CommandNode' and hasattr(node, 'name'):
        return node.name == 'GET_BACKGROUND_COLOR'

    return False


def _extract_literal_color(node: Any) -> Optional[int]:
    """リテラル値（0-9）を抽出"""
    if node is None:
        return None

    node_type = type(node).__name__

    if node_type == 'LiteralNode' and hasattr(node, 'value'):
        value = node.value
        if isinstance(value, int) and 0 <= value <= 9:
            return value

    return None


def _select_random_background_color() -> int:
    """ARC-AGI2統計に基づいてランダムに背景色を選択

    Returns:
        選択された背景色（0-9）
    """
    colors = list(BACKGROUND_COLOR_DISTRIBUTION.keys())
    weights = list(BACKGROUND_COLOR_DISTRIBUTION.values())

    return random.choices(colors, weights=weights, k=1)[0]


def generate_background_grid_pattern(
    background_color: int,
    grid_width: int,
    grid_height: int,
    seed: Optional[int] = None
) -> List[List[int]]:
    """背景グリッドパターンを生成

    Args:
        background_color: 背景色（0-9）
        grid_width: グリッド幅
        grid_height: グリッド高さ
        seed: 乱数シード（オプション）

    Returns:
        背景グリッドパターン（2次元リスト）
        - 97%の確率で [[background_color]]（1x1の単色）
        - 3%の確率で、2～max(2, int(max(grid_width, grid_height) * 0.5))のサイズの幾何学的パターン
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    # 97%の確率で単色パターン
    if rng.random() < 0.97:
        return [[background_color]]

    # 3%の確率で幾何学的パターンを生成
    max_size = max(2, int(max(grid_width, grid_height) * 0.5))
    pattern_size = rng.randint(2, max_size)

    # パターンタイプをランダム選択
    pattern_type = rng.choice([
        'checkerboard',
        'horizontal_stripe',
        'vertical_stripe',
        'diagonal_stripe',
        'dot_pattern',
        'cross_pattern',
        'tiled_pattern',
        'random'
    ])

    # パターンを生成（背景色を必ず使用）
    pattern = _generate_geometric_pattern(
        pattern_type=pattern_type,
        size=pattern_size,
        background_color=background_color,
        rng=rng
    )

    return pattern


def _generate_geometric_pattern(
    pattern_type: str,
    size: int,
    background_color: int,
    rng: random.Random
) -> List[List[int]]:
    """幾何学的パターンを生成

    Args:
        pattern_type: パターンタイプ
        size: パターンサイズ（正方形）
        background_color: 背景色（必ず使用される）
        rng: 乱数生成器

    Returns:
        パターン（2次元リスト）
    """
    # 背景色以外の色を選択（0-9の範囲で、background_color以外）
    other_colors = [c for c in range(10) if c != background_color]
    if not other_colors:
        # background_colorが唯一の色の場合（通常はあり得ない）
        other_colors = [background_color]

    # パターンタイプに応じて色の数を決定
    # background_color + pattern_colors で最大3色まで
    if pattern_type in ['horizontal_stripe', 'vertical_stripe']:
        # ストライプパターンは2色または3色（background_color + 1-2色）
        num_pattern_colors = rng.choice([1, 2])  # 1色または2色
    elif pattern_type in ['checkerboard', 'diagonal_stripe']:
        # チェッカーボードと斜めストライプは2色または3色
        num_pattern_colors = rng.choice([1, 2])
    else:
        # その他のパターンは1色（background_color + 1色）
        num_pattern_colors = 1

    # pattern_colorsを選択（background_color以外から、重複なし）
    if len(other_colors) >= num_pattern_colors:
        sample_size = min(num_pattern_colors, len(other_colors))
        if sample_size > 0 and len(other_colors) > 0:
            pattern_colors = rng.sample(other_colors, sample_size)
        else:
            pattern_colors = []
    else:
        # 色が足りない場合は利用可能な色をすべて使用
        pattern_colors = other_colors.copy()
        # 必要に応じて重複を許可して追加
        while len(pattern_colors) < num_pattern_colors:
            pattern_colors.append(rng.choice(other_colors))

    # パターンを初期化（すべてbackground_colorで埋める）
    pattern = [[background_color for _ in range(size)] for _ in range(size)]

    if pattern_type == 'checkerboard':
        # チェッカーボードパターン
        if len(pattern_colors) == 1:
            # 2色パターン（background_color + pattern_color）
            pattern_color = pattern_colors[0]
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        pattern[i][j] = pattern_color
        else:
            # 3色パターン（background_color + pattern_color1 + pattern_color2）
            pattern_color1 = pattern_colors[0]
            pattern_color2 = pattern_colors[1]
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        # さらに交互に色を変える
                        if (i // 2 + j // 2) % 2 == 0:
                            pattern[i][j] = pattern_color1
                        else:
                            pattern[i][j] = pattern_color2

    elif pattern_type == 'horizontal_stripe':
        # 横ストライプパターン
        stripe_width = max(1, size // 4)
        if len(pattern_colors) == 1:
            # 2色パターン（background_color + pattern_color）
            pattern_color = pattern_colors[0]
            for i in range(size):
                if (i // stripe_width) % 2 == 1:
                    for j in range(size):
                        pattern[i][j] = pattern_color
        else:
            # 3色パターン（background_color + pattern_color1 + pattern_color2）
            pattern_color1 = pattern_colors[0]
            pattern_color2 = pattern_colors[1]
            for i in range(size):
                stripe_index = i // stripe_width
                if stripe_index % 2 == 1:
                    # ストライプの色を交互に変える
                    if (stripe_index // 2) % 2 == 0:
                        pattern_color = pattern_color1
                    else:
                        pattern_color = pattern_color2
                    for j in range(size):
                        pattern[i][j] = pattern_color

    elif pattern_type == 'vertical_stripe':
        # 縦ストライプパターン
        stripe_width = max(1, size // 4)
        if len(pattern_colors) == 1:
            # 2色パターン（background_color + pattern_color）
            pattern_color = pattern_colors[0]
            for j in range(size):
                if (j // stripe_width) % 2 == 1:
                    for i in range(size):
                        pattern[i][j] = pattern_color
        else:
            # 3色パターン（background_color + pattern_color1 + pattern_color2）
            pattern_color1 = pattern_colors[0]
            pattern_color2 = pattern_colors[1]
            for j in range(size):
                stripe_index = j // stripe_width
                if stripe_index % 2 == 1:
                    # ストライプの色を交互に変える
                    if (stripe_index // 2) % 2 == 0:
                        pattern_color = pattern_color1
                    else:
                        pattern_color = pattern_color2
                    for i in range(size):
                        pattern[i][j] = pattern_color

    elif pattern_type == 'diagonal_stripe':
        # 斜めストライプパターン
        if len(pattern_colors) == 1:
            # 2色パターン（background_color + pattern_color）
            pattern_color = pattern_colors[0]
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        pattern[i][j] = pattern_color
        else:
            # 3色パターン（background_color + pattern_color1 + pattern_color2）
            pattern_color1 = pattern_colors[0]
            pattern_color2 = pattern_colors[1]
            for i in range(size):
                for j in range(size):
                    diagonal_index = i + j
                    if diagonal_index % 2 == 0:
                        # さらに交互に色を変える
                        if (diagonal_index // 2) % 2 == 0:
                            pattern[i][j] = pattern_color1
                        else:
                            pattern[i][j] = pattern_color2

    elif pattern_type == 'dot_pattern':
        # ドットパターン（格子状に配置）- 1色のみ
        pattern_color = pattern_colors[0]
        dot_interval = max(2, size // 3)
        for i in range(size):
            for j in range(size):
                if i % dot_interval == 0 and j % dot_interval == 0:
                    pattern[i][j] = pattern_color

    elif pattern_type == 'cross_pattern':
        # 十字パターン（中央に十字）- 1色のみ
        pattern_color = pattern_colors[0]
        center = size // 2
        for i in range(size):
            # 横線
            pattern[center][i] = pattern_color
            # 縦線
            pattern[i][center] = pattern_color

    elif pattern_type == 'random':
        # 完全ランダムパターン（各ピクセルをランダムに色選択）
        # 背景色以外の色リスト（0-9の範囲）
        all_colors = list(range(10))
        # 各ピクセルをランダムに色を選択
        for i in range(size):
            for j in range(size):
                pattern[i][j] = rng.choice(all_colors)
        # background_colorが必ず含まれるように、少なくとも1つのピクセルをbackground_colorに設定
        # ランダムにいくつかのピクセルをbackground_colorに設定（全体の20-40%程度）
        num_background_pixels = rng.randint(
            max(1, size * size // 5),  # 最低20%
            max(1, size * size // 2)   # 最大50%
        )
        all_positions = [(i, j) for i in range(size) for j in range(size)]
        sample_size = min(num_background_pixels, len(all_positions))
        if sample_size > 0 and len(all_positions) > 0:
            background_positions = rng.sample(all_positions, sample_size)
        else:
            background_positions = []
        for i, j in background_positions:
            pattern[i][j] = background_color

    elif pattern_type == 'tiled_pattern':
        # タイリングパターン（グリッドをランダムに分割し、各タイルに同じパターンを繰り返し配置）
        # タイルサイズをランダムに決定（2x2からsize//2まで）
        min_tile_size = 2
        max_tile_size = max(min_tile_size, size // 2)
        tile_width = rng.randint(min_tile_size, max_tile_size)
        tile_height = rng.randint(min_tile_size, max_tile_size)

        # タイル内のパターンに使用する色数を決定（2色以上、最大4色）
        num_tile_colors = rng.randint(2, min(4, len(pattern_colors) + 1))
        if len(pattern_colors) >= num_tile_colors - 1:
            tile_colors = rng.sample(pattern_colors, num_tile_colors - 1)
        else:
            tile_colors = pattern_colors.copy()
            # 不足分を追加
            while len(tile_colors) < num_tile_colors - 1:
                available_colors = [c for c in range(10) if c != background_color and c not in tile_colors]
                if available_colors:
                    tile_colors.append(rng.choice(available_colors))
                else:
                    break

        # タイル内のパターンを生成（ランダムにピクセルを配置）
        tile_pattern = [[background_color for _ in range(tile_width)] for _ in range(tile_height)]

        # タイル内のピクセルをランダムに色付け（背景色以外の色を使用）
        # タイル内の20-60%のピクセルをランダムに色付け
        num_colored_pixels = rng.randint(
            max(1, tile_width * tile_height // 5),  # 最低20%
            max(1, tile_width * tile_height * 3 // 5)  # 最大60%
        )

        # ランダムにピクセルを選択して色付け
        tile_positions = [(i, j) for i in range(tile_height) for j in range(tile_width)]
        colored_positions = rng.sample(tile_positions, min(num_colored_pixels, len(tile_positions)))

        for i, j in colored_positions:
            # ランダムに色を選択（背景色以外）
            tile_pattern[i][j] = rng.choice([background_color] + tile_colors)
            # 背景色が選ばれた場合は、再度選択（背景色以外を優先）
            if tile_pattern[i][j] == background_color and len(tile_colors) > 0:
                tile_pattern[i][j] = rng.choice(tile_colors)

        # タイルパターンを全体のグリッドに繰り返し配置
        for i in range(size):
            for j in range(size):
                tile_i = i % tile_height
                tile_j = j % tile_width
                pattern[i][j] = tile_pattern[tile_i][tile_j]

    # パターンに必ずbackground_colorが含まれることを確認（既にすべてbackground_colorで初期化しているので確実、またはrandomの場合は上記で確実に含めている）

    return pattern

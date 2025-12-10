"""
ASTノード定義（レガシー版 - 修正版）

dataclassエラーを修正したノード定義
"""
from typing import List, Dict, Optional, Any, Union
import random
import uuid


class Node:
    """基底ノードクラス"""
    def __init__(self, context: Dict[str, Any] = None):
        self.context = context or {}
        self.id = str(uuid.uuid4())

    def generate(self) -> str:
        """コード生成（基底実装）"""
        return f"# {type(self).__name__}"


class StatementNode(Node):
    """ステートメントノード"""
    pass


class ExpressionNode(Node):
    """式ノード"""
    pass


class InitializationNode(StatementNode):
    """初期化ノード（GET_ALL_OBJECTS）"""
    def __init__(self, connectivity: int = 4, context: Dict[str, Any] = None, skip_filter: bool = False):
        super().__init__(context)
        self.connectivity = connectivity
        # skip_filterがTrueの場合は、部分プログラムから生成された場合など、
        # 既に背景色フィルタが含まれているため、追加のフィルタを生成しない
        if skip_filter:
            self._has_filter = False
            self._filter_type = None
            self._filter_color = None
        else:
            # FILTERの有無を初期化時に一度だけ決定（generate()を複数回呼んでも同じ結果を返すため）
            import random
            self._has_filter = random.random() < 0.95
            if self._has_filter:
                # 低確率でランダム色との等価判定を混在させる
                if random.random() < 0.05:
                    self._filter_type = 'equal'
                    self._filter_color = random.randint(0, 9)
                else:
                    self._filter_type = 'not_equal_background'
                    self._filter_color = None
            else:
                self._filter_type = None
                self._filter_color = None

    def generate(self) -> str:
        # 基本の初期化
        init_code = f"objects = GET_ALL_OBJECTS({self.connectivity})"

        # 初期化時に決定したフィルタリング設定を使用
        if self._has_filter:
            if self._filter_type == 'equal':
                filter_code = f"objects = FILTER(objects, EQUAL(GET_COLOR($obj), {self._filter_color}))"
            else:  # 'not_equal_background'
                filter_code = "objects = FILTER(objects, NOT_EQUAL(GET_COLOR($obj), GET_BACKGROUND_COLOR()))"
            return f"{init_code}\n{filter_code}"
        else:
            return init_code


class AssignmentNode(StatementNode):
    """代入ノード"""
    def __init__(self, variable: str, expression: Node, context: Dict[str, Any] = None):
        super().__init__(context)
        self.variable = variable
        self.expression = expression

    def generate(self) -> str:
        if self.expression is None:
            raise ValueError(f"AssignmentNodeのexpressionがNoneです: variable={self.variable}")
        return f"{self.variable} = {self.expression.generate()}"


class ArrayAssignmentNode(StatementNode):
    """配列代入ノード"""
    def __init__(self, array: str, index: str, expression: Node, context: Dict[str, Any] = None):
        super().__init__(context)
        self.array = array
        self.index = index
        self.expression = expression

    def generate(self) -> str:
        if self.expression is None:
            raise ValueError(f"ArrayAssignmentNodeのexpressionがNoneです: array={self.array}, index={self.index}")
        return f"{self.array}[{self.index}] = {self.expression.generate()}"


class FilterNode(StatementNode):
    """フィルタノード"""
    def __init__(self, source_array: str, target_array: str, condition: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.source_array = source_array
        self.target_array = target_array
        self.condition = condition

    def generate(self) -> str:
        return f"{self.target_array} = FILTER({self.source_array}, {self.condition})"


class ExcludeNode(StatementNode):
    """EXCLUDEノード"""
    def __init__(self, source_array: str, target_array: str, targets_array: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.source_array = source_array
        self.target_array = target_array
        self.targets_array = targets_array

    def generate(self) -> str:
        return f"{self.target_array} = EXCLUDE({self.source_array}, {self.targets_array})"


class ConcatNode(StatementNode):
    """CONCATノード"""
    def __init__(self, array1: str, array2: str, target_array: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.array1 = array1
        self.array2 = array2
        self.target_array = target_array

    def generate(self) -> str:
        return f"{self.target_array} = CONCAT({self.array1}, {self.array2})"


class AppendNode(StatementNode):
    """APPENDノード"""
    def __init__(self, array: str, obj: str, target_array: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.array = array
        self.obj = obj
        self.target_array = target_array

    def generate(self) -> str:
        return f"{self.target_array} = APPEND({self.array}, {self.obj})"


class MergeNode(StatementNode):
    """MERGEノード"""
    def __init__(self, objects_array: str, target_obj: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.objects_array = objects_array
        self.target_obj = target_obj

    def generate(self) -> str:
        return f"{self.target_obj} = MERGE({self.objects_array})"


class EmptyArrayNode(StatementNode):
    """空のオブジェクト配列定義ノード"""
    def __init__(self, array_name: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.array_name = array_name

    def generate(self) -> str:
        return f"{self.array_name} = []"




class ForStartNode(StatementNode):
    """FORループ開始ノード"""
    def __init__(self, loop_var: str, array: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.loop_var = loop_var
        self.array = array

    def generate(self) -> str:
        return f"FOR {self.loop_var} LEN({self.array}) DO"


class ForStartWithCountNode(StatementNode):
    """COUNT型変数を使用したFORループ開始ノード"""
    def __init__(self, loop_var: str, count_variable: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.loop_var = loop_var
        self.count_variable = count_variable

    def generate(self) -> str:
        return f"FOR {self.loop_var} {self.count_variable} DO"


class ForStartWithConstantNode(StatementNode):
    """定数値を使用したFORループ開始ノード"""
    def __init__(self, loop_var: str, constant_value: int, context: Dict[str, Any] = None):
        super().__init__(context)
        self.loop_var = loop_var
        self.constant_value = constant_value

    def generate(self) -> str:
        return f"FOR {self.loop_var} {self.constant_value} DO"


class ForStartWithMatchPairsNode(StatementNode):
    """MATCH_PAIRS配列用のFORループ開始ノード（DIVIDE(LEN({array}), 2)を使用）"""
    def __init__(self, loop_var: str, match_pairs_array: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.loop_var = loop_var
        self.match_pairs_array = match_pairs_array

    def generate(self) -> str:
        return f"FOR {self.loop_var} DIVIDE(LEN({self.match_pairs_array}), 2) DO"


class EndNode(StatementNode):
    """汎用的な終了ノード（FORループ、IF文などで使用）"""
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__(context)

    def generate(self) -> str:
        return "END"


class IfStartNode(StatementNode):
    """IF分岐開始ノード"""
    def __init__(self, condition: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.condition = condition

    def generate(self) -> str:
        return f"IF {self.condition} THEN"


class IfBranchNode(StatementNode):
    """IF分岐ノード"""
    def __init__(self, condition: str, then_body: List[Node], else_body: List[Node] = None, context: Dict[str, Any] = None):
        super().__init__(context)
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body or []

    def generate(self) -> str:
        lines = [f"IF {self.condition} THEN"]
        for node in self.then_body:
            lines.append(f"    {node.generate()}")
        if self.else_body:
            lines.append("ELSE")
            for node in self.else_body:
                lines.append(f"    {node.generate()}")
        lines.append("END")
        return "\n".join(lines)


class RenderNode(StatementNode):
    """レンダーノード"""
    def __init__(self, array: str, bg_color: Node, width: Node, height: Node, context: Dict[str, Any] = None):
        super().__init__(context)
        self.array = array
        self.bg_color = bg_color
        self.width = width
        self.height = height

    def generate(self) -> str:
        return f"RENDER_GRID({self.array}, {self.bg_color.generate()}, {self.width.generate()}, {self.height.generate()})"


class LiteralNode(ExpressionNode):
    """リテラルノード"""
    def __init__(self, value: Union[int, str, float], context: Dict[str, Any] = None):
        super().__init__(context)
        self.value = value

    def generate(self) -> str:
        if isinstance(self.value, str):
            # プレースホルダーの場合は引用符を付けない
            if self.value.startswith('$'):
                return self.value
            return f'"{self.value}"'
        elif isinstance(self.value, bool):
            # ブール値はFalse/Trueとして出力（FALSE/TRUEではない）
            return str(self.value)
        return str(self.value)


class VariableNode(ExpressionNode):
    """変数ノード"""
    def __init__(self, name: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.name = name

    def generate(self) -> str:
        return self.name


class CommandNode(ExpressionNode):
    """コマンドノード"""
    def __init__(self, command: str, arguments: List[Node], context: Dict[str, Any] = None, return_type_info = None):
        super().__init__(context)
        self.command = command
        self.arguments = arguments
        self.return_type_info = return_type_info  # 算術演算など、動的に戻り値の型が変わる場合に使用

    def generate(self) -> str:
        if not self.arguments:
            return f"{self.command}()"  # 引数なしの場合も()を追加
        # None引数を除外して生成
        valid_args = [arg for arg in self.arguments if arg is not None]
        if not valid_args:
            return f"{self.command}()"  # 有効な引数がない場合
        args = ", ".join(arg.generate() for arg in valid_args)
        return f"{self.command}({args})"


class ArrayElementAccessNode(ExpressionNode):
    """配列要素アクセスノード（例: GET_INPUT_GRID_SIZE()[0]）"""
    def __init__(self, expression: Node, index: int, context: Dict[str, Any] = None):
        super().__init__(context)
        self.expression = expression
        self.index = index  # 0 または 1

    def generate(self) -> str:
        return f"{self.expression.generate()}[{self.index}]"


class BinaryOpNode(ExpressionNode):
    """二項演算ノード"""
    def __init__(self, operator: str, left: Node, right: Node, context: Dict[str, Any] = None):
        super().__init__(context)
        self.operator = operator
        self.left = left
        self.right = right

    def generate(self) -> str:
        return f"{self.left.generate()} {self.operator} {self.right.generate()}"


class PlaceholderNode(ExpressionNode):
    """プレースホルダーノード"""
    def __init__(self, placeholder: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.placeholder = placeholder

    def generate(self) -> str:
        return self.placeholder


class ObjectAccessNode(StatementNode):
    """オブジェクトアクセスノード（obj = objects[0] または obj = objects[SUB(LEN(objects), 1)]）"""
    def __init__(self, obj_var: str, objects_array: str, access_type: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.obj_var = obj_var
        self.objects_array = objects_array
        self.access_type = access_type  # "first" または "last"

    def generate(self) -> str:
        if self.access_type == "first":
            return f"{self.obj_var} = {self.objects_array}[0]"
        elif self.access_type == "last":
            return f"{self.obj_var} = {self.objects_array}[SUB(LEN({self.objects_array}), 1)]"
        else:
            return f"{self.obj_var} = {self.objects_array}[0]"


class SingleObjectArrayNode(StatementNode):
    """単一オブジェクト配列定義ノード（objects = [object]）"""
    def __init__(self, array_name: str, object_name: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.array_name = array_name
        self.object_name = object_name

    def generate(self) -> str:
        return f"{self.array_name} = [{self.object_name}]"


class SplitConnectedNode(StatementNode):
    """SPLIT_CONNECTEDノード"""
    def __init__(self, source_object: str, target_array: str, connectivity: int = 4, context: Dict[str, Any] = None):
        super().__init__(context)
        self.source_object = source_object
        self.target_array = target_array
        self.connectivity = connectivity

    def generate(self) -> str:
        return f"{self.target_array} = SPLIT_CONNECTED({self.source_object}, {self.connectivity})"


class ExtractShapeNode(StatementNode):
    """形状抽出ノード（EXTRACT_RECTS, EXTRACT_HOLLOW_RECTS, EXTRACT_LINES） - 配列を返す"""
    def __init__(self, source_object: str, target_array: str, extract_type: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.source_object = source_object
        self.target_array = target_array
        self.extract_type = extract_type  # "rects", "hollow_rects", "lines"（BBOXは除外）

    def generate(self) -> str:
        if self.extract_type == "rects":
            return f"{self.target_array} = EXTRACT_RECTS({self.source_object})"
        elif self.extract_type == "hollow_rects":
            return f"{self.target_array} = EXTRACT_HOLLOW_RECTS({self.source_object})"
        elif self.extract_type == "lines":
            return f"{self.target_array} = EXTRACT_LINES({self.source_object})"
        else:
            return f"{self.target_array} = EXTRACT_RECTS({self.source_object})"


class ExtendPatternNode(StatementNode):
    """EXTEND_PATTERNノード"""
    def __init__(self, source_array: str, target_array: str, side: str = "end", count: int = 1, context: Dict[str, Any] = None):
        super().__init__(context)
        self.source_array = source_array
        self.target_array = target_array
        self.side = side
        self.count = count

    def generate(self) -> str:
        return f"{self.target_array} = EXTEND_PATTERN({self.source_array}, \"{self.side}\", {self.count})"


class ArrangeGridNode(StatementNode):
    """ARRANGE_GRIDノード"""
    def __init__(self, source_array: str, target_array: str, cols: int = 3, width: int = 10, height: int = 10, context: Dict[str, Any] = None):
        super().__init__(context)
        self.source_array = source_array
        self.target_array = target_array
        self.cols = cols
        self.width = width
        self.height = height

    def generate(self) -> str:
        return f"{self.target_array} = ARRANGE_GRID({self.source_array}, {self.cols}, {self.width}, {self.height})"


class MatchPairsNode(StatementNode):
    """MATCH_PAIRSノード"""
    def __init__(self, array1: str, array2: str, target_array: str, condition: str, context: Dict[str, Any] = None):
        super().__init__(context)
        self.array1 = array1
        self.array2 = array2
        self.target_array = target_array
        self.condition = condition

    def generate(self) -> str:
        return f"{self.target_array} = MATCH_PAIRS({self.array1}, {self.array2}, {self.condition})"
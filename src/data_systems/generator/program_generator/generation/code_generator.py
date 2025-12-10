"""
コード生成ユーティリティ
"""
from typing import List
from .nodes import Node
from .program_context import ProgramContext

# 生成起源コメント付与の有効/無効フラグ（出力の末尾に [iter=..] 等を付けない場合は False）
ENABLE_ORIGIN_ANNOTATION = False


def generate_code(nodes: List[Node], context: ProgramContext) -> str:
    """ノードリストからコードを生成（インデントと生成起源注記付き）"""
    raw_lines: List[str] = []
    raw_line_nodes: List[Node] = []

    for i, node in enumerate(nodes):
        if node is None:
            continue  # Noneノードをスキップ
        node_code = node.generate()
        lines = node_code.split("\n")
        for line in lines:
            raw_lines.append(line)
            raw_line_nodes.append(node)

    # 連続する重複行を除去（対応するノードも同期）
    dedup_lines: List[str] = []
    dedup_nodes: List[Node] = []
    if raw_lines:
        dedup_lines.append(raw_lines[0])
        dedup_nodes.append(raw_line_nodes[0])
        for i in range(1, len(raw_lines)):
            current_line = raw_lines[i].strip()
            previous_line = raw_lines[i - 1].strip()
            if (current_line.startswith('FOR ') or current_line.startswith('IF ') or
                current_line == 'END' or current_line.startswith('WHILE ')):
                dedup_lines.append(raw_lines[i])
                dedup_nodes.append(raw_line_nodes[i])
            elif current_line != previous_line:
                dedup_lines.append(raw_lines[i])
                dedup_nodes.append(raw_line_nodes[i])

    # インデントを適用（行ごとに）
    indented_lines = []
    indent_level = 0
    indent_size = 4
    for line in dedup_lines:
        s = line.strip()
        if not s:
            indented_lines.append("")
            continue
        if any(keyword in s for keyword in ['FOR ', 'IF ', 'WHILE ']):
            indented_lines.append(" " * (indent_level * indent_size) + s)
            indent_level += 1
        elif s == 'END':
            indent_level = max(0, indent_level - 1)
            indented_lines.append(" " * (indent_level * indent_size) + s)
        else:
            indented_lines.append(" " * (indent_level * indent_size) + s)

    # 生成起源注記を末尾に付与（無効化フラグ対応）
    if not ENABLE_ORIGIN_ANNOTATION:
        return "\n".join(indented_lines)

    annotated_lines: List[str] = []
    for line, node in zip(indented_lines, dedup_nodes):
        origin_phase = None
        origin_iter = None
        # 生成起源メタデータ取得（意図されたフォールバック: メタデータが取得できなくても処理を継続）
        try:
            origin_phase = node.context.get("gen_origin_phase")
            origin_iter = node.context.get("gen_origin_iteration")
        except Exception:
            origin_phase = None
            origin_iter = None

        if origin_phase == "iter" and origin_iter is not None:
            annotated_lines.append(f"{line}  # [iter={origin_iter}]")
        elif origin_phase == "remaining":
            annotated_lines.append(f"{line}  # [remaining]")
        elif origin_phase == "final":
            annotated_lines.append(f"{line}  # [final]")
        else:
            annotated_lines.append(line)

    return "\n".join(annotated_lines)


def generate_code_no_indent(nodes: List[Node], context: ProgramContext) -> str:
    """ノードリストからコードを生成（インデント整形なし、連続重複は除去）"""
    raw_lines: List[str] = []
    raw_line_nodes: List[Node] = []

    for node in nodes:
        if node is None:
            continue  # Noneノードをスキップ
        node_code = node.generate()
        lines = node_code.split("\n")
        for line in lines:
            raw_lines.append(line)
            raw_line_nodes.append(node)

    # 連続する重複行を除去（FOR/IF/END/WHILE は除外）
    dedup_lines: List[str] = []
    dedup_nodes: List[Node] = []
    if raw_lines:
        dedup_lines.append(raw_lines[0])
        dedup_nodes.append(raw_line_nodes[0])
        for i in range(1, len(raw_lines)):
            current_line = raw_lines[i].strip()
            previous_line = raw_lines[i - 1].strip()
            if (current_line.startswith('FOR ') or current_line.startswith('IF ') or
                current_line == 'END' or current_line.startswith('WHILE ')):
                dedup_lines.append(raw_lines[i])
                dedup_nodes.append(raw_line_nodes[i])
            elif current_line != previous_line:
                dedup_lines.append(raw_lines[i])
                dedup_nodes.append(raw_line_nodes[i])

    return "\n".join(dedup_lines)


def apply_indentation(code_lines: List[str]) -> List[str]:
    """コード行にインデントを適用"""
    if not code_lines:
        return code_lines

    indented_lines = []
    indent_level = 0
    indent_size = 4  # インデントサイズ（スペース4つ）

    for line in code_lines:
        line = line.strip()
        if not line:
            indented_lines.append("")
            continue

        # ネスト開始のキーワードをチェック
        if any(keyword in line for keyword in ['FOR ', 'IF ', 'WHILE ']):
            # ネスト開始行を追加
            indented_lines.append(" " * (indent_level * indent_size) + line)
            indent_level += 1
        elif line == 'END':
            # ネスト終了
            indent_level = max(0, indent_level - 1)
            indented_lines.append(" " * (indent_level * indent_size) + line)
        else:
            # 通常の行（現在のインデントレベルで追加）
            indented_lines.append(" " * (indent_level * indent_size) + line)

    return indented_lines


def remove_consecutive_duplicates(code_lines: List[str]) -> List[str]:
    """連続する重複行を除去（FOR/IF/END文は除外）"""
    if not code_lines:
        return code_lines

    deduplicated = [code_lines[0]]  # 最初の行は常に保持

    for i in range(1, len(code_lines)):
        current_line = code_lines[i].strip()
        previous_line = code_lines[i-1].strip()

        # FOR文、IF文、END文は連続しても削除しない
        if (current_line.startswith('FOR ') or current_line.startswith('IF ') or
            current_line == 'END' or current_line.startswith('WHILE ')):
            deduplicated.append(code_lines[i])
        # 前の行と同じでない場合のみ追加
        elif current_line != previous_line:
            deduplicated.append(code_lines[i])

    return deduplicated

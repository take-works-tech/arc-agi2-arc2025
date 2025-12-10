"""
データ構造の検証機能

DataPairとTaskの妥当性を検証する機能を提供
"""

from typing import List, Dict, Any
from .data_pair import DataPair
from .task import Task


def validate_data_pair(data_pair: DataPair) -> List[str]:
    """DataPairの妥当性を検証
    
    Args:
        data_pair: 検証するDataPair
    
    Returns:
        問題のリスト（空の場合は問題なし）
    """
    issues = []
    
    # 基本的な存在チェック
    if not data_pair.input or not data_pair.output:
        issues.append("入力または出力が空です")
    
    if not data_pair.program:
        issues.append("プログラムが空です")
    
    # グリッドサイズのチェック
    if data_pair.input and data_pair.output:
        input_size = data_pair.get_grid_size()
        output_size = (len(data_pair.output), len(data_pair.output[0]) if data_pair.output else 0)
        
        if input_size != output_size:
            issues.append(f"入力サイズ {input_size} と出力サイズ {output_size} が一致しません")
    
    # グリッドの妥当性チェック
    if data_pair.input:
        for i, row in enumerate(data_pair.input):
            if not isinstance(row, list):
                issues.append(f"入力の行 {i} がリストではありません")
                continue
            
            for j, color in enumerate(row):
                if not isinstance(color, int) or color < 0 or color > 9:
                    issues.append(f"入力の位置 ({i}, {j}) の色 {color} が無効です")
    
    if data_pair.output:
        for i, row in enumerate(data_pair.output):
            if not isinstance(row, list):
                issues.append(f"出力の行 {i} がリストではありません")
                continue
            
            for j, color in enumerate(row):
                if not isinstance(color, int) or color < 0 or color > 9:
                    issues.append(f"出力の位置 ({i}, {j}) の色 {color} が無効です")
    
    # プログラムの妥当性チェック
    if data_pair.program:
        if len(data_pair.program.strip()) < 5:
            issues.append("プログラムが短すぎます")
        
        # 基本的な構文チェック
        if not any(cmd in data_pair.program for cmd in ['GET_', 'SET_', 'FOR ', 'IF ']):
            issues.append("有効なコマンドが見つかりません")
    
    # メタデータのチェック
    if data_pair.metadata:
        if not isinstance(data_pair.metadata, dict):
            issues.append("メタデータが辞書形式ではありません")
    
    return issues


def validate_task(task: Task) -> List[str]:
    """Taskの妥当性を検証
    
    Args:
        task: 検証するTask
    
    Returns:
        問題のリスト（空の場合は問題なし）
    """
    issues = []
    
    # 基本的な存在チェック
    if not task.train or not task.test:
        issues.append("訓練ペアまたはテストペアが空です")
    
    if not task.program:
        issues.append("プログラムが空です")
    
    # 訓練ペアの検証
    if task.train:
        for i, pair in enumerate(task.train):
            if not isinstance(pair, dict):
                issues.append(f"訓練ペア {i} が辞書形式ではありません")
                continue
            
            if 'input' not in pair or 'output' not in pair:
                issues.append(f"訓練ペア {i} にinputまたはoutputがありません")
                continue
            
            # グリッドの妥当性チェック
            input_grid = pair['input']
            output_grid = pair['output']
            
            if not isinstance(input_grid, list) or not isinstance(output_grid, list):
                issues.append(f"訓練ペア {i} のグリッドがリスト形式ではありません")
                continue
            
            # サイズの一致チェック
            if input_grid and output_grid:
                input_size = (len(input_grid), len(input_grid[0]) if input_grid else 0)
                output_size = (len(output_grid), len(output_grid[0]) if output_grid else 0)
                
                if input_size != output_size:
                    issues.append(f"訓練ペア {i} の入力サイズ {input_size} と出力サイズ {output_size} が一致しません")
    
    # テストペアの検証
    if task.test:
        for i, pair in enumerate(task.test):
            if not isinstance(pair, dict):
                issues.append(f"テストペア {i} が辞書形式ではありません")
                continue
            
            if 'input' not in pair or 'output' not in pair:
                issues.append(f"テストペア {i} にinputまたはoutputがありません")
                continue
            
            # グリッドの妥当性チェック
            input_grid = pair['input']
            output_grid = pair['output']
            
            if not isinstance(input_grid, list) or not isinstance(output_grid, list):
                issues.append(f"テストペア {i} のグリッドがリスト形式ではありません")
                continue
    
    # 一貫性チェック
    if not task.validate_consistency():
        issues.append("タスクの一貫性に問題があります")
    
    # プログラムの妥当性チェック
    if task.program:
        if len(task.program.strip()) < 5:
            issues.append("プログラムが短すぎます")
        
        # 基本的な構文チェック
        if not any(cmd in task.program for cmd in ['GET_', 'SET_', 'FOR ', 'IF ']):
            issues.append("有効なコマンドが見つかりません")
    
    # メタデータのチェック
    if task.metadata:
        if not isinstance(task.metadata, dict):
            issues.append("メタデータが辞書形式ではありません")
    
    return issues


def validate_data_pair_batch(data_pairs: List[DataPair]) -> Dict[str, List[str]]:
    """DataPairのバッチを検証
    
    Args:
        data_pairs: 検証するDataPairのリスト
    
    Returns:
        各DataPairのIDと問題のリストの辞書
    """
    results = {}
    
    for pair in data_pairs:
        issues = validate_data_pair(pair)
        if issues:
            results[pair.pair_id] = issues
    
    return results


def validate_task_batch(tasks: List[Task]) -> Dict[str, List[str]]:
    """Taskのバッチを検証
    
    Args:
        tasks: 検証するTaskのリスト
    
    Returns:
        各TaskのIDと問題のリストの辞書
    """
    results = {}
    
    for task in tasks:
        issues = validate_task(task)
        if issues:
            results[task.task_id] = issues
    
    return results


def get_validation_summary(validation_results: Dict[str, List[str]]) -> Dict[str, Any]:
    """検証結果のサマリーを取得
    
    Args:
        validation_results: 検証結果の辞書
    
    Returns:
        検証サマリー
    """
    total_items = len(validation_results)
    items_with_issues = len([issues for issues in validation_results.values() if issues])
    
    all_issues = []
    for issues in validation_results.values():
        all_issues.extend(issues)
    
    issue_counts = {}
    for issue in all_issues:
        issue_counts[issue] = issue_counts.get(issue, 0) + 1
    
    return {
        'total_items': total_items,
        'items_with_issues': items_with_issues,
        'items_without_issues': total_items - items_with_issues,
        'total_issues': len(all_issues),
        'unique_issues': len(issue_counts),
        'most_common_issues': sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        'validation_rate': (total_items - items_with_issues) / total_items if total_items > 0 else 0.0
    }


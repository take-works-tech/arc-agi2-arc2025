"""
データ検証器

データの妥当性を検証する機能
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from core.data_structures import DataPair, Task


@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    score: float
    details: Dict[str, Any]


class DataValidator:
    """データ検証器"""

    def __init__(self):
        """初期化"""
        self.validation_rules = {
            'min_grid_size': 3,
            'max_grid_size': 30,
            'min_colors': 1,
            'max_colors': 10,
            'min_program_length': 10,
            'max_program_length': 1000,
            'required_commands': ['GET_', 'RENDER_GRID'],
            'forbidden_patterns': ['while True:', 'for i in range(10000):']
        }

    def validate_data_pair(self, data_pair: DataPair) -> ValidationResult:
        """DataPairを検証

        Args:
            data_pair: 検証するDataPair

        Returns:
            検証結果
        """
        issues = []
        warnings = []
        details = {}

        # 基本構造の検証
        if not data_pair.input:
            issues.append("入力グリッドが空です")
        if not data_pair.output:
            issues.append("出力グリッドが空です")
        if not data_pair.program:
            issues.append("プログラムが空です")

        # グリッドの検証
        if data_pair.input:
            grid_validation = self._validate_grid(data_pair.input, "input")
            issues.extend(grid_validation['issues'])
            warnings.extend(grid_validation['warnings'])
            details['input_grid'] = grid_validation['details']

        if data_pair.output:
            grid_validation = self._validate_grid(data_pair.output, "output")
            issues.extend(grid_validation['issues'])
            warnings.extend(grid_validation['warnings'])
            details['output_grid'] = grid_validation['details']

        # プログラムの検証
        if data_pair.program:
            program_validation = self._validate_program(data_pair.program)
            issues.extend(program_validation['issues'])
            warnings.extend(program_validation['warnings'])
            details['program'] = program_validation['details']

        # 一貫性の検証
        if data_pair.input and data_pair.output:
            consistency_validation = self._validate_consistency(
                data_pair.input,
                data_pair.output
            )
            issues.extend(consistency_validation['issues'])
            warnings.extend(consistency_validation['warnings'])
            details['consistency'] = consistency_validation['details']

        # スコア計算
        score = self._calculate_validation_score(issues, warnings, details)

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            score=score,
            details=details
        )

    def validate_task(self, task: Task) -> ValidationResult:
        """Taskを検証

        Args:
            task: 検証するTask

        Returns:
            検証結果
        """
        issues = []
        warnings = []
        details = {}

        # 基本構造の検証
        if not task.train:
            issues.append("訓練ペアが空です")
        if not task.test:
            issues.append("テストペアが空です")
        if not task.program:
            issues.append("プログラムが空です")

        # 各ペアの検証
        train_validation_results = []
        for i, pair_dict in enumerate(task.train):
            if 'input' not in pair_dict or 'output' not in pair_dict:
                issues.append(f"訓練ペア{i}にinputまたはoutputがありません")
                continue

            # DataPairを作成して検証（適切な形式で作成）
            temp_pair = DataPair(
                input=pair_dict['input'],
                output=pair_dict['output'],
                program=task.program if hasattr(task, 'program') else None
            )
            validation_result = self.validate_data_pair(temp_pair)
            train_validation_results.append(validation_result)

            if not validation_result.is_valid:
                issues.extend([f"訓練ペア{i}: {issue}" for issue in validation_result.issues])
            warnings.extend([f"訓練ペア{i}: {warning}" for warning in validation_result.warnings])

        test_validation_results = []
        for i, pair_dict in enumerate(task.test):
            if 'input' not in pair_dict or 'output' not in pair_dict:
                issues.append(f"テストペア{i}にinputまたはoutputがありません")
                continue

            # DataPairを作成して検証（適切な形式で作成）
            temp_pair = DataPair(
                input=pair_dict['input'],
                output=pair_dict['output'],
                program=task.program if hasattr(task, 'program') else None
            )
            validation_result = self.validate_data_pair(temp_pair)
            test_validation_results.append(validation_result)

            if not validation_result.is_valid:
                issues.extend([f"テストペア{i}: {issue}" for issue in validation_result.issues])
            warnings.extend([f"テストペア{i}: {warning}" for warning in validation_result.warnings])

        # タスク全体の一貫性検証
        if task.train and task.test:
            task_consistency = self._validate_task_consistency(task)
            issues.extend(task_consistency['issues'])
            warnings.extend(task_consistency['warnings'])
            details['task_consistency'] = task_consistency['details']

        # 詳細情報
        details['train_validation_results'] = [r.details for r in train_validation_results]
        details['test_validation_results'] = [r.details for r in test_validation_results]

        # スコア計算
        score = self._calculate_validation_score(issues, warnings, details)

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            score=score,
            details=details
        )

    def _validate_grid(self, grid: List[List[int]], grid_type: str) -> Dict[str, Any]:
        """グリッドを検証"""
        issues = []
        warnings = []
        details = {}

        if not isinstance(grid, list):
            issues.append(f"{grid_type}グリッドがリストではありません")
            return {'issues': issues, 'warnings': warnings, 'details': details}

        if not grid:
            issues.append(f"{grid_type}グリッドが空です")
            return {'issues': issues, 'warnings': warnings, 'details': details}

        # 行数の検証
        height = len(grid)
        if height < self.validation_rules['min_grid_size']:
            issues.append(f"{grid_type}グリッドの高さが小さすぎます: {height}")
        elif height > self.validation_rules['max_grid_size']:
            issues.append(f"{grid_type}グリッドの高さが大きすぎます: {height}")

        # 列数の検証
        if not isinstance(grid[0], list):
            issues.append(f"{grid_type}グリッドの行がリストではありません")
            return {'issues': issues, 'warnings': warnings, 'details': details}

        width = len(grid[0])
        if width < self.validation_rules['min_grid_size']:
            issues.append(f"{grid_type}グリッドの幅が小さすぎます: {width}")
        elif width > self.validation_rules['max_grid_size']:
            issues.append(f"{grid_type}グリッドの幅が大きすぎます: {width}")

        # 各行の長さが一致するかチェック
        for i, row in enumerate(grid):
            if not isinstance(row, list):
                issues.append(f"{grid_type}グリッドの行{i}がリストではありません")
                continue

            if len(row) != width:
                issues.append(f"{grid_type}グリッドの行{i}の長さが一致しません: {len(row)} != {width}")

        # 色の検証
        colors = set()
        for row in grid:
            for pixel in row:
                if not isinstance(pixel, int):
                    issues.append(f"{grid_type}グリッドに整数以外の値があります: {pixel}")
                elif pixel < 0 or pixel > 9:
                    issues.append(f"{grid_type}グリッドに範囲外の色があります: {pixel}")
                else:
                    colors.add(pixel)

        color_count = len(colors)
        if color_count < self.validation_rules['min_colors']:
            issues.append(f"{grid_type}グリッドの色数が少なすぎます: {color_count}")
        elif color_count > self.validation_rules['max_colors']:
            warnings.append(f"{grid_type}グリッドの色数が多いです: {color_count}")

        # 詳細情報
        details = {
            'height': height,
            'width': width,
            'color_count': color_count,
            'colors': sorted(list(colors)),
            'total_pixels': height * width
        }

        return {'issues': issues, 'warnings': warnings, 'details': details}

    def _validate_program(self, program: str) -> Dict[str, Any]:
        """プログラムを検証"""
        issues = []
        warnings = []
        details = {}

        if not isinstance(program, str):
            issues.append("プログラムが文字列ではありません")
            return {'issues': issues, 'warnings': warnings, 'details': details}

        # 長さの検証
        program_length = len(program)
        if program_length < self.validation_rules['min_program_length']:
            issues.append(f"プログラムが短すぎます: {program_length}")
        elif program_length > self.validation_rules['max_program_length']:
            issues.append(f"プログラムが長すぎます: {program_length}")

        # 必須コマンドの検証
        for required_cmd in self.validation_rules['required_commands']:
            if required_cmd not in program:
                issues.append(f"必須コマンドが見つかりません: {required_cmd}")

        # 禁止パターンの検証
        for forbidden_pattern in self.validation_rules['forbidden_patterns']:
            if forbidden_pattern in program:
                issues.append(f"禁止パターンが含まれています: {forbidden_pattern}")

        # 構文の基本的な検証
        if program.count('(') != program.count(')'):
            issues.append("括弧の対応が取れていません")

        if program.count('[') != program.count(']'):
            issues.append("角括弧の対応が取れていません")

        # 詳細情報
        details = {
            'length': program_length,
            'line_count': len(program.split('\n')),
            'command_count': len([cmd for cmd in self.validation_rules['required_commands'] if cmd in program]),
            'has_loops': 'FOR ' in program or 'WHILE ' in program,
            'has_conditions': 'IF ' in program or 'FILTER' in program
        }

        return {'issues': issues, 'warnings': warnings, 'details': details}

    def _validate_consistency(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
        """入力と出力の一貫性を検証"""
        issues = []
        warnings = []
        details = {}

        # サイズの一貫性
        input_height, input_width = len(input_grid), len(input_grid[0])
        output_height, output_width = len(output_grid), len(output_grid[0])

        if input_height != output_height or input_width != output_width:
            warnings.append(f"グリッドサイズが異なります: {input_height}x{input_width} -> {output_height}x{output_width}")

        # 色の一貫性
        input_colors = set()
        output_colors = set()

        for row in input_grid:
            input_colors.update(row)
        for row in output_grid:
            output_colors.update(row)

        new_colors = output_colors - input_colors
        if new_colors:
            warnings.append(f"新しい色が追加されています: {sorted(list(new_colors))}")

        # 詳細情報
        details = {
            'input_size': (input_height, input_width),
            'output_size': (output_height, output_width),
            'input_colors': sorted(list(input_colors)),
            'output_colors': sorted(list(output_colors)),
            'new_colors': sorted(list(new_colors)),
            'color_change_ratio': len(new_colors) / len(input_colors) if input_colors else 0
        }

        return {'issues': issues, 'warnings': warnings, 'details': details}

    def _validate_task_consistency(self, task: Task) -> Dict[str, Any]:
        """タスク全体の一貫性を検証"""
        issues = []
        warnings = []
        details = {}

        # 訓練ペアとテストペアの一貫性
        if task.train and task.test:
            # 色の一貫性
            train_colors = set()
            test_colors = set()

            for pair in task.train:
                for row in pair['input']:
                    train_colors.update(row)
                for row in pair['output']:
                    train_colors.update(row)

            for pair in task.test:
                for row in pair['input']:
                    test_colors.update(row)
                for row in pair['output']:
                    test_colors.update(row)

            # テストペアの色が訓練ペアの色の範囲内にあるかチェック
            test_only_colors = test_colors - train_colors
            if test_only_colors:
                warnings.append(f"テストペアに訓練ペアにない色があります: {sorted(list(test_only_colors))}")

            # サイズの一貫性
            train_sizes = set()
            test_sizes = set()

            for pair in task.train:
                train_sizes.add((len(pair['input']), len(pair['input'][0])))
                train_sizes.add((len(pair['output']), len(pair['output'][0])))

            for pair in task.test:
                test_sizes.add((len(pair['input']), len(pair['input'][0])))
                test_sizes.add((len(pair['output']), len(pair['output'][0])))

            # 詳細情報
            details = {
                'train_colors': sorted(list(train_colors)),
                'test_colors': sorted(list(test_colors)),
                'test_only_colors': sorted(list(test_only_colors)),
                'train_sizes': sorted(list(train_sizes)),
                'test_sizes': sorted(list(test_sizes)),
                'color_consistency': len(test_only_colors) == 0,
                'size_consistency': len(test_sizes - train_sizes) == 0
            }

        return {'issues': issues, 'warnings': warnings, 'details': details}

    def _calculate_validation_score(self, issues: List[str], warnings: List[str], details: Dict[str, Any]) -> float:
        """検証スコアを計算"""
        if not issues and not warnings:
            return 1.0

        # 基本スコア
        base_score = 1.0

        # 問題による減点
        issue_penalty = len(issues) * 0.2
        warning_penalty = len(warnings) * 0.05

        # 詳細による調整
        detail_bonus = 0.0
        if 'input_grid' in details and 'output_grid' in details:
            # グリッドサイズの適切性
            input_size = details['input_grid'].get('total_pixels', 0)
            output_size = details['output_grid'].get('total_pixels', 0)

            if 9 <= input_size <= 400:  # 3x3 to 20x20
                detail_bonus += 0.1
            if 9 <= output_size <= 400:
                detail_bonus += 0.1

        # 最終スコア
        final_score = max(0.0, base_score - issue_penalty - warning_penalty + detail_bonus)

        return final_score

    def set_validation_rules(self, rules: Dict[str, Any]):
        """検証ルールを設定"""
        self.validation_rules.update(rules)

    def get_validation_rules(self) -> Dict[str, Any]:
        """検証ルールを取得"""
        return self.validation_rules.copy()

    def validate_batch(self, data_items: List[Union[DataPair, Task]]) -> List[ValidationResult]:
        """バッチ検証"""
        results = []
        for item in data_items:
            if isinstance(item, DataPair):
                result = self.validate_data_pair(item)
            elif isinstance(item, Task):
                result = self.validate_task(item)
            else:
                result = ValidationResult(
                    is_valid=False,
                    issues=["不明なデータタイプです"],
                    warnings=[],
                    score=0.0,
                    details={}
                )
            results.append(result)
        return results

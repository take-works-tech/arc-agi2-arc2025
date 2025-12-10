"""
プログラム合成エンジンのベンチマークスクリプト

ProgramSynthesisEngineとARCEvaluatorを統合して、複数のタスクで評価を実行し、
結果を可視化・保存します。
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import time

import numpy as np

from src.hybrid_system.core.data_structures.task import Task
from src.hybrid_system.inference.program_synthesis.synthesis_engine import (
    ProgramSynthesisEngine,
    SynthesisConfig,
)
try:
    from src.hybrid_system.inference.system_integrator.evaluator import (
        ARCEvaluator,
        ARCEvaluationConfig,
    )
except ImportError:
    # フォールバック: core.evaluation.evaluatorを使用
    from src.hybrid_system.core.evaluation.evaluator import (
        ARCEvaluator,
        ARCEvaluationConfig,
    )


def _json_default(o: Any):
    """numpy のスカラ型などを JSON シリアライズ可能な形に変換"""
    if isinstance(o, (np.bool_, np.bool8)):
        return bool(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return str(o)


def _load_all_arc_training_tasks() -> Dict[str, Task]:
    """ARC-AGI2 training_challenges から全タスクを読み込む"""
    base = Path("data/core_arc_agi2")
    challenges_path = base / "arc-agi_training_challenges.json"
    if not challenges_path.exists():
        raise FileNotFoundError(f"ARC training_challenges が見つかりません: {challenges_path}")

    with challenges_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        raise ValueError("arc-agi_training_challenges.json の形式が想定外です")

    tasks: Dict[str, Task] = {}
    for task_id, task_dict in data.items():
        task = Task.from_dict(
            {
                "train": task_dict["train"],
                "test": task_dict["test"],
                "program": "",
                "metadata": {"source": "arc_agi_training", "task_id": task_id},
            }
        )
        tasks[task_id] = task

    return tasks


def _evaluate_synthesized_program(
    engine: ProgramSynthesisEngine,
    evaluator: ARCEvaluator,
    task: Task
) -> Dict[str, Any]:
    """プログラムを合成して評価"""
    start_time = time.time()

    # プログラムを合成
    synthesis_result = engine.synthesize_program_with_validation(task)
    synthesis_time = time.time() - start_time

    if not synthesis_result.get("success", False):
        return {
            "task_id": task.task_id,
            "synthesis_success": False,
            "synthesis_time": synthesis_time,
            "program": None,
            "evaluation": None,
        }

    synthesized_program = synthesis_result.get("program")
    validation_results = synthesis_result.get("validation_results", {})

    # 合成されたプログラムでテストペアを実行して評価
    # 注意: 現在のARCEvaluatorはタスクのプログラムを評価するが、
    # ここでは合成されたプログラムでテストペアを実行する必要がある
    # 簡易版として、validation_resultsを使用
    evaluation_result = {
        "consistency_score": validation_results.get("consistency_score", 0.0),
        "complexity_score": validation_results.get("complexity_score", 0.0),
        "overall_score": validation_results.get("overall_score", 0.0),
        "passed_validation": validation_results.get("passed_validation", False),
    }

    # ARCEvaluatorで評価（タスクにプログラムを設定）
    task_with_program = Task.from_dict({
        "train": task.train,
        "test": task.test,
        "program": synthesized_program,
        "metadata": task.metadata,
    })
    task_with_program.task_id = task.task_id

    arc_evaluation = evaluator.evaluate_task(task_with_program)

    return {
        "task_id": task.task_id,
        "synthesis_success": True,
        "synthesis_time": synthesis_time,
        "program": synthesized_program,
        "validation_results": validation_results,
        "evaluation": {
            "arc_evaluation": arc_evaluation,
            "consistency_score": evaluation_result["consistency_score"],
            "complexity_score": evaluation_result["complexity_score"],
            "overall_score": evaluation_result["overall_score"],
            "passed_validation": evaluation_result["passed_validation"],
        },
        "synthesis_info": synthesis_result.get("synthesis_info", {}),
    }


def main(num_tasks: int = 10, config_name: str = "default") -> None:
    """メイン処理"""
    tasks = _load_all_arc_training_tasks()
    sorted_ids = sorted(tasks.keys())[:num_tasks]

    print(
        f"[benchmark_program_synthesis] "
        f"{len(sorted_ids)} タスクでベンチマークを実行します。"
    )
    print(f"設定: {config_name}")

    # エンジンと評価器を初期化
    synthesis_config = SynthesisConfig(
        enable_program_scorer=False,  # ベンチマークでは無効化（必要に応じて変更）
        enable_object_matching=True,
        object_matching_confidence_threshold=0.6
    )
    engine = ProgramSynthesisEngine(config=synthesis_config)

    evaluation_config = ARCEvaluationConfig(
        enable_accuracy_evaluation=True,
        enable_consistency_evaluation=True,
        enable_completeness_evaluation=True,
    )
    evaluator = ARCEvaluator(config=evaluation_config)

    results = []

    for idx, task_id in enumerate(sorted_ids, start=1):
        task = tasks[task_id]
        print(f"\n{'='*60}")
        print(f"Task {idx}/{len(sorted_ids)}: id={task_id}")
        print(f"  訓練ペア数: {len(task.train)}, テストペア数: {len(task.test)}")
        print(f"{'='*60}")

        # 統計をリセット
        engine.reset_statistics()

        # 評価を実行
        result = _evaluate_synthesized_program(engine, evaluator, task)
        results.append(result)

        # 結果を表示
        if result["synthesis_success"]:
            print(f"  合成成功: ✓")
            print(f"  合成時間: {result['synthesis_time']:.2f}秒")
            eval_info = result["evaluation"]
            print(f"  一貫性スコア: {eval_info['consistency_score']:.3f}")
            print(f"  複雑度スコア: {eval_info['complexity_score']:.3f}")
            print(f"  最終スコア: {eval_info['overall_score']:.3f}")
            print(f"  検証通過: {'✓' if eval_info['passed_validation'] else '✗'}")
            if eval_info.get("arc_evaluation"):
                arc_eval = eval_info["arc_evaluation"]
                print(f"  ARC評価: {'✓' if arc_eval.get('overall_passed') else '✗'}")
                print(f"  ARC総合スコア: {arc_eval.get('overall_score', 0.0):.3f}")
        else:
            print(f"  合成失敗: ✗")
            print(f"  合成時間: {result['synthesis_time']:.2f}秒")

    # 全体統計を計算
    successful_syntheses = [r for r in results if r["synthesis_success"]]
    failed_syntheses = [r for r in results if not r["synthesis_success"]]

    print(f"\n{'='*60}")
    print("=== ベンチマーク結果サマリ ===")
    print(f"{'='*60}")
    print(f"\n総タスク数: {len(results)}")
    print(f"成功: {len(successful_syntheses)} ({len(successful_syntheses)/len(results)*100:.1f}%)")
    print(f"失敗: {len(failed_syntheses)} ({len(failed_syntheses)/len(results)*100:.1f}%)")

    if successful_syntheses:
        avg_synthesis_time = np.mean([r["synthesis_time"] for r in successful_syntheses])
        avg_consistency = np.mean([r["evaluation"]["consistency_score"] for r in successful_syntheses])
        avg_complexity = np.mean([r["evaluation"]["complexity_score"] for r in successful_syntheses])
        avg_overall = np.mean([r["evaluation"]["overall_score"] for r in successful_syntheses])
        passed_validation_count = sum(1 for r in successful_syntheses if r["evaluation"]["passed_validation"])

        print(f"\n[成功したタスクの統計]")
        print(f"  平均合成時間: {avg_synthesis_time:.2f}秒")
        print(f"  平均一貫性スコア: {avg_consistency:.3f}")
        print(f"  平均複雑度スコア: {avg_complexity:.3f}")
        print(f"  平均最終スコア: {avg_overall:.3f}")
        print(f"  検証通過数: {passed_validation_count}/{len(successful_syntheses)} ({passed_validation_count/len(successful_syntheses)*100:.1f}%)")

        # ARC評価の統計
        arc_evaluations = [r["evaluation"].get("arc_evaluation") for r in successful_syntheses if r["evaluation"].get("arc_evaluation")]
        if arc_evaluations:
            passed_arc_count = sum(1 for e in arc_evaluations if e.get("overall_passed", False))
            avg_arc_score = np.mean([e.get("overall_score", 0.0) for e in arc_evaluations])
            print(f"  ARC評価通過数: {passed_arc_count}/{len(arc_evaluations)} ({passed_arc_count/len(arc_evaluations)*100:.1f}%)")
            print(f"  平均ARC総合スコア: {avg_arc_score:.3f}")

    # 結果をJSONファイルに保存
    output_file = Path(f"outputs/benchmark_program_synthesis_{config_name}_{int(time.time())}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": {
            "config_name": config_name,
            "num_tasks": len(results),
            "synthesis_config": {
                "enable_program_scorer": synthesis_config.enable_program_scorer,
                "enable_object_matching": synthesis_config.enable_object_matching,
                "object_matching_confidence_threshold": synthesis_config.object_matching_confidence_threshold,
            },
            "evaluation_config": {
                "enable_accuracy_evaluation": evaluation_config.enable_accuracy_evaluation,
                "enable_consistency_evaluation": evaluation_config.enable_consistency_evaluation,
                "enable_completeness_evaluation": evaluation_config.enable_completeness_evaluation,
            },
        },
        "statistics": {
            "total_tasks": len(results),
            "successful_syntheses": len(successful_syntheses),
            "failed_syntheses": len(failed_syntheses),
            "success_rate": len(successful_syntheses) / len(results) if results else 0.0,
        },
        "results": results,
    }

    if successful_syntheses:
        summary["statistics"]["average_synthesis_time"] = float(avg_synthesis_time)
        summary["statistics"]["average_consistency_score"] = float(avg_consistency)
        summary["statistics"]["average_complexity_score"] = float(avg_complexity)
        summary["statistics"]["average_overall_score"] = float(avg_overall)
        summary["statistics"]["passed_validation_count"] = passed_validation_count
        summary["statistics"]["passed_validation_rate"] = passed_validation_count / len(successful_syntheses)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"\n結果を保存しました: {output_file}")

    # エンジンの統計も表示
    engine_stats = engine.get_synthesis_statistics()
    print(f"\n[エンジン統計]")
    print(f"  総合成試行数: {engine_stats.get('total_synthesis_attempts', 0)}")
    print(f"  成功数: {engine_stats.get('successful_syntheses', 0)}")
    print(f"  失敗数: {engine_stats.get('failed_syntheses', 0)}")
    if engine_stats.get('object_matching_used', 0) > 0:
        print(f"  オブジェクトマッチング使用: {engine_stats.get('object_matching_used', 0)}回")
    if engine_stats.get('partial_program_used', 0) > 0:
        print(f"  部分プログラム使用: {engine_stats.get('partial_program_used', 0)}回")


if __name__ == "__main__":
    import sys
    num_tasks = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    config_name = sys.argv[2] if len(sys.argv) > 2 else "default"
    main(num_tasks=num_tasks, config_name=config_name)

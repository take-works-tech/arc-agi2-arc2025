"""
Phase 1 訓練スクリプト

DataPairからプログラム合成モデルを訓練
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import glob
import yaml
import torch
import os
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader, random_split

from src.hybrid_system.models.program_synthesis import ProgramSynthesisModel
from src.hybrid_system.models.base.model_config import ProgramSynthesisConfig
from src.hybrid_system.learning.phase1_trainer.full_trainer import FullPhase1Trainer
from src.hybrid_system.learning.phase1_trainer.dataset import DataPairDataset, collate_fn
from src.hybrid_system.data_management.io import DataIO
from src.hybrid_system.utils.logging import Logger, TrainingLogger
from src.hybrid_system.utils.tokenizer import ProgramTokenizer
from src.hybrid_system.ir.parser import RelabelTransformer, RelabelTransformerConfig
from src.hybrid_system.ir.serialization import sequence_to_template_string
from datetime import datetime


class StreamTee:
    """stdout / stderr を複数ストリームへ同時出力するヘルパー."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        for stream in self.streams:
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)

    def fileno(self):
        for stream in self.streams:
            if hasattr(stream, "fileno"):
                return stream.fileno()
        raise AttributeError("fileno")

    def __getattr__(self, name):
        return getattr(self.streams[0], name)
from src.hybrid_system.utils.logging import Logger, TrainingLogger
from src.hybrid_system.utils.tokenizer import ProgramTokenizer


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """ディレクトリ内で最新のチェックポイントを探索"""
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')
    candidates = glob.glob(pattern)

    def extract_epoch(path: str) -> int:
        name = os.path.basename(path)
        try:
            return int(name.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return -1

    candidates = [c for c in candidates if extract_epoch(c) >= 0]
    if candidates:
        candidates.sort(key=extract_epoch, reverse=True)
        return candidates[0]

    # フォールバックとして最新チェックポイントを確認
    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
    if os.path.exists(latest_path):
        return latest_path

    return None


def _run_phase1_training(
    config: Dict[str, Any],
    logger: Logger,
    training_logger: TrainingLogger,
    tokenizer: ProgramTokenizer,
    device: str,
    train_loader: Optional[DataLoader],
    val_loader_own: Optional[DataLoader],
    val_loader_eval: Optional[DataLoader],
    val_data_pairs_own: list,
    val_data_pairs_eval: list,
    eval_data_pairs_eval: list,
) -> None:
    """データ準備完了後の訓練・評価処理を実行"""
    vocab_size = tokenizer.vocab_size()
    config['models']['program_synthesis']['vocab_size'] = vocab_size
    logger.info(f"語彙サイズ: {vocab_size}")

    model_config = ProgramSynthesisConfig(**config['models']['program_synthesis'])
    model_config.device = device

    model = ProgramSynthesisModel(model_config)
    logger.info(f"モデル作成完了: {model.count_parameters():,} パラメータ")

    compile_config = config['training']['phase1'].get('torch_compile', {})
    compile_kwargs: Dict[str, Any] = {}
    compile_mode = None
    compile_backend = None
    if compile_config.get('enable', False):
        compile_mode = compile_config.get('mode')
        compile_backend = compile_config.get('backend')
        if compile_mode:
            compile_kwargs['mode'] = compile_mode
        if compile_backend:
            compile_kwargs['backend'] = compile_backend

    trainer = FullPhase1Trainer(
        model=model,
        config=config['training']['phase1'],
        logger=training_logger
    )

    start_epoch = 1
    phase1_config = config['training']['phase1']
    resume_checkpoint = phase1_config.get('resume_checkpoint', None)
    auto_resume = phase1_config.get('auto_resume', False)

    if auto_resume or (isinstance(resume_checkpoint, str) and resume_checkpoint.lower() == "latest"):
        latest_checkpoint = find_latest_checkpoint(model_config.save_dir)
        if latest_checkpoint:
            resume_checkpoint = latest_checkpoint
            logger.info(f"最新チェックポイントを検出: {resume_checkpoint}")
        else:
            logger.info("自動再開: 有効なチェックポイントが見つかりません。新規学習を開始します")
            resume_checkpoint = None

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        logger.info(f"チェックポイントから再開: {resume_checkpoint}")
        start_epoch = trainer.load_checkpoint(resume_checkpoint) + 1
        logger.info(f"エポック {start_epoch} から再開します")
        logger.info(f"復元された状態: best_val_loss={trainer.best_val_loss:.4f}, patience_counter={trainer.patience_counter}")

        checkpoint_dir = os.path.dirname(resume_checkpoint)
        vocab_path = os.path.join(checkpoint_dir, 'vocab.json')
        if os.path.exists(vocab_path):
            logger.info(f"トークナイザーの語彙を読み込み: {vocab_path}")
            tokenizer.load_vocab(vocab_path)
            vocab_size = tokenizer.vocab_size()
            config['models']['program_synthesis']['vocab_size'] = vocab_size
            logger.info(f"語彙サイズ: {vocab_size}")
        else:
            logger.warning(f"語彙ファイルが見つかりません: {vocab_path}")
            logger.info("データから語彙を再構築します")
    else:
        if resume_checkpoint:
            logger.warning(f"指定されたチェックポイントが見つかりません: {resume_checkpoint}")
            logger.info("新規学習を開始します")
        else:
            logger.info("新規学習を開始します")

    if compile_config.get('enable', False):
        try:
            compiled_model = torch.compile(trainer.model, **compile_kwargs)
            trainer.model = compiled_model
            model = trainer.model
            logger.info(
                f"torch.compileを適用（mode={compile_mode or 'default'}, backend={compile_backend or 'default'}）"
            )
        except Exception as compile_error:
            logger.warning(f"torch.compileの適用に失敗しました: {compile_error}")
            logger.warning("コンパイルをスキップして通常のモデルを使用します")

    logger.info("訓練準備完了")
    logger.info("訓練開始...")

    result = trainer.train(
        train_loader=train_loader,
        val_loader_own=val_loader_own,
        val_loader_eval=val_loader_eval,
        num_epochs=config['training']['phase1']['num_epochs'],
        start_epoch=start_epoch
    )

    logger.info(f"訓練結果: {result}")

    final_checkpoint_path = os.path.join(model_config.save_dir, 'final_model.pt')
    model.save_checkpoint(
        final_checkpoint_path,
        result.get('total_epochs', config['training']['phase1']['num_epochs']),
        trainer.optimizer
    )
    logger.info(f"最終モデルを保存: {final_checkpoint_path}")

    vocab_path = os.path.join(model_config.save_dir, 'vocab.json')
    tokenizer.save_vocab(vocab_path)
    logger.info(f"トークナイザーの語彙を保存: {vocab_path}")

    logger.info("=" * 60)
    logger.info("最終評価開始")
    logger.info("=" * 60)
    logger.info("最終評価では、実行ベース評価を全サンプルで実行します（詳細な評価）")

    try:
        if val_data_pairs_own:
            logger.info("=" * 60)
            logger.info("最終評価（自作データの残り200ペア）")
            logger.info("=" * 60)

            use_ir_templates = config.get('phase1', {}).get('use_ir_templates', True)
            eval_dataset_own = DataPairDataset(
                data_pairs=val_data_pairs_own,
                tokenizer=tokenizer,
                max_program_length=config['models']['program_synthesis']['max_program_length'],
                max_grid_size=config['data']['generation']['max_grid_size'],
                use_ir_templates=use_ir_templates
            )

            eval_loader_own = DataLoader(
                eval_dataset_own,
                batch_size=config['training']['phase1']['batch_size'],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0
            )

            logger.info(f"最終評価データセット（自作データ）: {len(val_data_pairs_own)}ペア")
            logger.info("最終評価実行中（自作データ）...")

            eval_loss_own, eval_metrics_own = trainer._validate_epoch(
                eval_loader_own,
                epoch=result.get('total_epochs', config['training']['phase1']['num_epochs']),
                label="最終評価（自作データ）",
                force_full_eval=True
            )

            logger.info("=" * 60)
            logger.info("最終評価結果（自作データ）")
            logger.info("=" * 60)
            logger.info(f"最終評価損失: {eval_loss_own:.4f}")
            logger.info(f"最終評価メトリクス: {eval_metrics_own}")
            logger.info("=" * 60)

        logger.info("=" * 60)
        logger.info("最終評価（評価データの全530ペア）")
        logger.info("=" * 60)

        use_ir_templates = config.get('phase1', {}).get('use_ir_templates', True)
        eval_dataset_eval = DataPairDataset(
            data_pairs=eval_data_pairs_eval,
            tokenizer=tokenizer,
            max_program_length=config['models']['program_synthesis']['max_program_length'],
            max_grid_size=config['data']['generation']['max_grid_size'],
            use_ir_templates=use_ir_templates
        )

        eval_loader_eval = DataLoader(
            eval_dataset_eval,
            batch_size=config['training']['phase1']['batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        logger.info(f"最終評価データセット（評価データ）: {len(eval_data_pairs_eval)}ペア")
        logger.info("最終評価実行中（評価データ）...")

        eval_loss_eval, eval_metrics_eval = trainer._validate_epoch(
            eval_loader_eval,
            epoch=result.get('total_epochs', config['training']['phase1']['num_epochs']),
            label="最終評価（評価データ）",
            force_full_eval=True
        )

        logger.info("=" * 60)
        logger.info("最終評価結果（評価データ）")
        logger.info("=" * 60)
        logger.info(f"最終評価損失: {eval_loss_eval:.4f}")
        logger.info(f"最終評価メトリクス: {eval_metrics_eval}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"最終評価中にエラーが発生しました: {e}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """メイン処理"""
    training_logger: Optional[TrainingLogger] = None
    logger = None
    run_log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # 設定を読み込み
        with open('configs/default_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 日時単位の標準出力ログファイルを作成
        log_root = Path(config['logging']['log_dir'])
        run_log_dir = log_root / "phase1_runs"
        run_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_log_path = run_log_dir / f"training_output_{timestamp}.txt"
        run_log_file = run_log_path.open("w", encoding="utf-8", buffering=1)

        # stdout / stderr を複製
        sys.stdout = StreamTee(original_stdout, run_log_file)
        sys.stderr = StreamTee(original_stderr, run_log_file)

        # ロガーを初期化
        logger = Logger.get_logger("phase1_training", config['logging']['log_dir'])
        logger.info("Phase 1 訓練開始")
        logger.info(f"標準出力/標準エラーを {run_log_path} に記録します")

        # 訓練ロガーを初期化
        training_logger = TrainingLogger(
            config['logging']['log_dir'],
            config['logging']['experiment_name'] + "_phase1"
        )
        training_logger.log_config(config)

        # デバイス設定
        device_config = config['device']['type']
        # CUDAが利用可能かチェック
        if device_config == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDAが利用できません。CPUに切り替えます。")
            device = "cpu"
        else:
            device = device_config
        logger.info(f"使用デバイス: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # トークナイザーを初期化
        tokenizer = ProgramTokenizer()
        logger.info("トークナイザー初期化完了")

        # データを読み込み
        logger.info("データ読み込み中...")
        data_io = DataIO(config['data']['generated_dir'])

        val_data_pairs_own = []
        val_data_pairs_eval = []
        eval_data_pairs_eval = []
        train_loader = None
        val_loader_own = None
        val_loader_eval = None

        try:
            all_data_pairs = []

            converted_data_path = os.path.join('phase1_pairs', 'data_pairs_from_outputs.jsonl.gz')
            converted_data_full_path = os.path.join(config['data']['generated_dir'], converted_data_path)
            if os.path.exists(converted_data_full_path):
                logger.info(f"outputsから変換したDataPair読み込み中: {converted_data_path}")
                converted_pairs = data_io.load_data_pairs(converted_data_path)
                logger.info(f"outputsから変換したDataPair読み込み完了: {len(converted_pairs)}個")
                all_data_pairs.extend(converted_pairs)

            pipeline_data_path = os.path.join('phase1_pairs', 'data_pairs.jsonl.gz')
            pipeline_data_full_path = os.path.join(config['data']['generated_dir'], pipeline_data_path)
            if os.path.exists(pipeline_data_full_path):
                logger.info(f"Pipelineで生成したDataPair読み込み中: {pipeline_data_path}")
                pipeline_pairs = data_io.load_data_pairs(pipeline_data_path)
                logger.info(f"Pipelineで生成したDataPair読み込み完了: {len(pipeline_pairs)}個")
                all_data_pairs.extend(pipeline_pairs)

            manual_data_path = os.path.join('phase1_pairs', 'data_pairs_manual.jsonl.gz')
            manual_data_full_path = os.path.join(config['data']['generated_dir'], manual_data_path)
            if os.path.exists(manual_data_full_path):
                logger.info(f"手動プログラムから変換したDataPair読み込み中: {manual_data_path}")
                manual_pairs = data_io.load_data_pairs(manual_data_path)
                logger.info(f"手動プログラムから変換したDataPair読み込み完了: {len(manual_pairs)}個")
                all_data_pairs.extend(manual_pairs)

            object_ops_data_path = os.path.join('phase1_pairs', 'data_pairs_object_operations.jsonl.gz')
            object_ops_data_full_path = os.path.join(config['data']['generated_dir'], object_ops_data_path)
            if os.path.exists(object_ops_data_full_path):
                logger.info(f"オブジェクト操作コマンドから変換したDataPair読み込み中: {object_ops_data_path}")
                object_ops_pairs = data_io.load_data_pairs(object_ops_data_path)
                logger.info(f"オブジェクト操作コマンドから変換したDataPair読み込み完了: {len(object_ops_pairs)}個")
                all_data_pairs.extend(object_ops_pairs)

            data_pairs = all_data_pairs
            logger.info(f"合計DataPair読み込み完了: {len(data_pairs)}個")

            logger.info("テンプレートシーケンスへ変換中...")
            transformer = RelabelTransformer(RelabelTransformerConfig())
            converted_count = 0
            for dp in data_pairs:
                if not dp.program:
                    continue
                try:
                    sequence = transformer.transform(
                        dp.program,
                        task_id=dp.pair_id,
                        sequence_metadata={
                            "pair_id": dp.pair_id,
                            "source_metadata": dp.metadata,
                        },
                    )
                    dp.metadata = dp.metadata or {}
                    dp.metadata["original_program"] = dp.program
                    dp.metadata["template_sequence"] = sequence.to_dict()
                    dp.program = sequence_to_template_string(sequence)
                    converted_count += 1
                except Exception as transform_error:
                    logger.warning(f"テンプレート変換に失敗: pair_id={dp.pair_id} error={transform_error}")
            logger.info(f"テンプレート変換完了: {converted_count}/{len(data_pairs)}")

            logger.info("トークナイザーの語彙を構築中...")
            for dp in data_pairs:
                tokenizer.encode(dp.program, add_special_tokens=False)
            logger.info(f"語彙サイズ: {tokenizer.vocab_size()}")

            test_mode = config['training']['phase1'].get('test_mode', False)
            if test_mode:
                target_train_size = config['training']['phase1'].get('test_train_size', 1000)
                test_val_size_own = config['training']['phase1'].get('test_val_size_own', 50)
                test_val_size_eval = config['training']['phase1'].get('test_val_size_eval', 50)
                logger.info("=" * 60)
                logger.info("テストモード: データサイズを制限して動作確認を行います")
                logger.info(f"  訓練データ: {target_train_size}ペア")
                logger.info(f"  検証データ（自作）: {test_val_size_own}ペア")
                logger.info(f"  検証データ（評価）: {test_val_size_eval}ペア")
                logger.info("=" * 60)
            else:
                target_train_size = 199504
                test_val_size_own = 200
                test_val_size_eval = 200

            import random
            random.seed(42)

            if len(data_pairs) > target_train_size:
                shuffled_pairs = data_pairs.copy()
                random.shuffle(shuffled_pairs)
                train_data_pairs = shuffled_pairs[:target_train_size]
                if test_mode:
                    val_data_pairs_own = shuffled_pairs[target_train_size:target_train_size + test_val_size_own]
                else:
                    val_data_pairs_own = shuffled_pairs[target_train_size:]
                logger.info(f"訓練データを{target_train_size}ペアに制限")
                logger.info(f"自作データの検証用: {len(val_data_pairs_own)}ペア")
            else:
                train_data_pairs = data_pairs
                if test_mode and len(data_pairs) > target_train_size + test_val_size_own:
                    shuffled_pairs = data_pairs.copy()
                    random.shuffle(shuffled_pairs)
                    train_data_pairs = shuffled_pairs[:target_train_size]
                    val_data_pairs_own = shuffled_pairs[target_train_size:target_train_size + test_val_size_own]
                else:
                    val_data_pairs_own = []
                logger.info(f"訓練データ: {len(train_data_pairs)}ペア（制限未満）")
                if not test_mode:
                    logger.warning("データが不足しているため、自作データの検証セットは作成されません")

            # evaluation_dataset.py は削除されたため、検証データの作成機能は無効化
            # 必要に応じて、別の方法で検証データを準備してください

            challenges_path = os.path.join('data', 'core_arc_agi2', 'arc-agi_evaluation_challenges.json')
            solutions_path = os.path.join('data', 'core_arc_agi2', 'arc-agi_evaluation_solutions.json')

            val_num_pairs = test_val_size_eval if test_mode else 200
            logger.info(f"評価データから検証データセットを作成中（訓練中検証用: {val_num_pairs}ペア）...")
            # evaluation_dataset.py は削除されたため、検証データの作成機能は無効化
            val_data_pairs_eval = []  # create_validation_data_pairs_with_solutions(
                challenges_path=challenges_path,
                solutions_path=solutions_path,
                num_pairs=val_num_pairs,
                use_train_pairs=True,
                use_test_pairs=True,
                seed=42
            )
            logger.info(f"検証データセット作成完了（評価データから）: {len(val_data_pairs_eval)}ペア")

            if test_mode:
                logger.info(f"評価データから最終評価データセットを作成中（テストモード: {test_val_size_eval}ペア）...")
                eval_data_pairs_eval = create_validation_data_pairs_from_evaluation(
                    challenges_path=challenges_path,
                    solutions_path=solutions_path,
                    num_pairs=test_val_size_eval,
                    use_train_pairs=True,
                    use_test_pairs=True,
                    seed=42
                )
            else:
                logger.info("評価データから最終評価データセットを作成中（全ペア使用）...")
                eval_data_pairs_eval = create_validation_data_pairs_from_evaluation(
                    challenges_path=challenges_path,
                    solutions_path=solutions_path,
                    num_pairs=None,
                    use_train_pairs=True,
                    use_test_pairs=True,
                    seed=42
                )
            logger.info(f"最終評価データセット作成完了（評価データ）: {len(eval_data_pairs_eval)}ペア")

            for dp in val_data_pairs_own + val_data_pairs_eval:
                if dp.program:
                    tokenizer.encode(dp.program, add_special_tokens=False)
            logger.info(f"語彙サイズ更新後: {tokenizer.vocab_size()}")

            # IRテンプレートを使用するかどうか（設定から取得、デフォルトTrue）
            use_ir_templates = config.get('phase1', {}).get('use_ir_templates', True)
            logger.info(f"IRテンプレート使用: {use_ir_templates}")

            train_dataset = DataPairDataset(
                data_pairs=train_data_pairs,
                tokenizer=tokenizer,
                max_program_length=config['models']['program_synthesis']['max_program_length'],
                max_grid_size=config['data']['generation']['max_grid_size'],
                use_ir_templates=use_ir_templates
            )

            val_dataset_own = DataPairDataset(
                data_pairs=val_data_pairs_own,
                tokenizer=tokenizer,
                max_program_length=config['models']['program_synthesis']['max_program_length'],
                max_grid_size=config['data']['generation']['max_grid_size'],
                use_ir_templates=use_ir_templates
            ) if val_data_pairs_own else None

            val_dataset_eval = DataPairDataset(
                data_pairs=val_data_pairs_eval,
                tokenizer=tokenizer,
                max_program_length=config['models']['program_synthesis']['max_program_length'],
                max_grid_size=config['data']['generation']['max_grid_size'],
                use_ir_templates=use_ir_templates
            )

            logger.info(f"データ分割: 訓練={len(train_dataset)}, 検証（自作データ）={len(val_dataset_own) if val_dataset_own else 0}, 検証（評価データ）={len(val_dataset_eval)}")

            dataloader_config = config['training']['phase1'].get('dataloader', {})
            num_workers = dataloader_config.get('num_workers', 0)
            pin_memory = dataloader_config.get('pin_memory', device == 'cuda')
            persistent_workers = dataloader_config.get('persistent_workers', False) and num_workers > 0
            prefetch_factor = dataloader_config.get('prefetch_factor', 2)

            def create_loader(dataset, shuffle: bool):
                if dataset is None:
                    return None

                loader_kwargs = {
                    'batch_size': config['training']['phase1']['batch_size'],
                    'shuffle': shuffle,
                    'collate_fn': collate_fn,
                    'num_workers': num_workers
                }

                if device != 'cpu' and pin_memory:
                    loader_kwargs['pin_memory'] = True
                if num_workers > 0:
                    loader_kwargs['persistent_workers'] = persistent_workers
                    loader_kwargs['prefetch_factor'] = prefetch_factor

                return DataLoader(dataset, **loader_kwargs)

            train_loader = create_loader(train_dataset, shuffle=True)
            val_loader_own = create_loader(val_dataset_own, shuffle=False)
            val_loader_eval = create_loader(val_dataset_eval, shuffle=False)

            logger.info("データローダー作成完了")

        except FileNotFoundError as e:
            logger.error(f"データファイルが見つかりません: {e}")
            logger.info("まず scripts/generate_data.py を実行してデータを生成してください")
            return

        _run_phase1_training(
            config=config,
            logger=logger,
            training_logger=training_logger,
            tokenizer=tokenizer,
            device=device,
            train_loader=train_loader,
            val_loader_own=val_loader_own,
            val_loader_eval=val_loader_eval,
            val_data_pairs_own=val_data_pairs_own,
            val_data_pairs_eval=val_data_pairs_eval,
            eval_data_pairs_eval=eval_data_pairs_eval
        )
    finally:
        if logger is not None:
            logger.info("Phase 1 訓練完了")
        if training_logger is not None:
            training_logger.close()
        if run_log_file is not None:
            run_log_file.flush()
            run_log_file.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr


if __name__ == "__main__":
    main()

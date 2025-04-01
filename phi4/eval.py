import argparse
import os
from pathlib import Path

from accelerate import Accelerator

from .common import (
    EvalDataset,
    evaluate,
    load_model_and_processor,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ntnu-smil/test-slate-dataset",
        help="Dataset name to use for training and evaluation",
    )
    parser.add_argument(
        "--dataset_subset",
        type=str,
        help="Dataset subset to use (e.g., 'zh-TW' for Common Voice 19)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="eval",
        help="Dataset split to use for training",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        help="Maximum number of training samples",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        help="Maximum number of evaluation samples",
    )
    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Use Flash Attention for more efficient training on compatible hardware",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/",
        help="Output directory for saving model checkpoints and logs",
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=128,
        help="Total batch size across all GPUs (global_batch_size = batch_size_per_gpu * num_gpus * gradient_accumulation_steps)",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=4,
        help="Training batch size per GPU (decrease this value if you encounter OOM errors)",
    )
    parser.add_argument(
        "--eval_batch_size_per_gpu",
        type=int,
        default=1,
        help="Evaluation batch size per GPU (can typically be larger than training batch size since no gradients are stored)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of complete passes through the training dataset",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4.0e-5,
        help="Peak learning rate for optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay coefficient for regularization",
    )
    parser.add_argument(
        "--eval_metric",
        type=str,
        default="wer",
        choices=["wer", "cer"],
        help="Evaluation metric: 'wer' for Word Error Rate (word-level) or 'cer' for Character Error Rate (character-level)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the Hugging Face Hub after training",
    )
    parser.add_argument(
        "--no-tqdm", dest="tqdm", action="store_false", help="Disable tqdm"
    )
    parser.add_argument(
        "--skip_initial_eval",
        action="store_true",
        help="Skip evaluation before training",
    )
    args = parser.parse_args()

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        model, processor = load_model_and_processor(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    eval_dataset = EvalDataset(
        processor,
        dataset_name=args.dataset_name,
        split=args.split,
        text_column=args.text_column,
        audio_column=args.audio_column,
        max_samples=args.max_samples,
        rank=rank,
        world_size=world_size,
        dataset_subset=args.dataset_subset,
    )

    # Create output directory if it doesn't exist
    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Generate a descriptive filename for the results
    dataset_name = args.dataset_name.split("/")[-1]
    model_name = args.model_name_or_path.split("/")[-1]
    results_filename = f"{model_name}_{dataset_name}_{args.split}.json"
    save_path = out_path / results_filename

    # Run evaluation
    evaluate(
        model,
        processor,
        eval_dataset,
        save_path=save_path,
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size,
        metric=args.metric,
    )

    if accelerator.is_main_process:
        print(f"Evaluation results saved to {save_path}")


if __name__ == "__main__":
    main()

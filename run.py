"""One entry point: prepare, train, evaluate, register, or the whole pipeline."""
import argparse
from pathlib import Path
import config


def main():
    parser = argparse.ArgumentParser(description="2D learning-based image registration")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="Prepare public OASIS MRI slices")
    preparation.add_argument("--data", type=Path, default=Path("data/oasis"))
    preparation.add_argument("--archive", type=Path)
    preparation.add_argument("--seed", type=int, default=config.SEED)
    for name in ["train", "pipeline"]:
        p = commands.add_parser(name, help="Train" if name == "train" else "Prepare, train, evaluate and save figures")
        p.add_argument("--data", type=Path, default=Path("data/oasis"))
        p.add_argument("--output", type=Path, default=Path("outputs/baseline"))
        p.add_argument("--epochs", type=int, default=config.EPOCHS)
        p.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
        p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
        p.add_argument("--smoothness-weight", type=float, default=config.SMOOTHNESS_WEIGHT)
        p.add_argument("--seed", type=int, default=config.SEED)
        p.add_argument("--integration-steps", type=int, default=config.INTEGRATION_STEPS)
        p.add_argument("--similarity", choices=["mse","ncc"], default="mse")
        p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
        p.add_argument("--resume", type=Path)
        if name == "pipeline":
            p.add_argument("--archive", type=Path)
    e = commands.add_parser("evaluate", help="Measure a saved model on test pairs")
    e.add_argument("--data", type=Path, default=Path("data/oasis"))
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    e.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    e.add_argument("--save-examples", type=int, default=3)
    r = commands.add_parser("register", help="Register one moving/fixed pair")
    r.add_argument("--moving", type=Path, required=True)
    r.add_argument("--fixed", type=Path, required=True)
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--output", type=Path, default=Path("outputs/registration"))
    r.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from prepare_data import prepare
            prepare(args.data, args.archive, args.seed)
        elif args.command in {"train", "pipeline"}:
            if args.integration_steps < 0:
                raise ValueError("Integration steps must be nonnegative.")
            if args.command == "pipeline" and not (args.data/"manifest.json").exists():
                from prepare_data import prepare
                prepare(args.data, args.archive, args.seed)
            from train import train
            checkpoint = train(args.data, args.output, args.epochs, args.learning_rate, args.batch_size,
                               args.smoothness_weight, args.seed, args.integration_steps, args.similarity,
                               args.device, args.resume)
            if args.command == "pipeline":
                from evaluate import evaluate
                evaluate(args.data, checkpoint, args.output/"evaluation", args.device)
        elif args.command == "evaluate":
            if args.save_examples < 0:
                raise ValueError("Save examples must be nonnegative.")
            from evaluate import evaluate
            evaluate(args.data, args.checkpoint, args.output, args.device, args.save_examples)
        else:
            from register import register
            register(args.moving, args.fixed, args.checkpoint, args.output, args.device)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()

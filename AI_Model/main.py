from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from AI_Model.pipeline import evaluate_labeled_dataset, train_pipeline, verify_pipeline
else:
    from .pipeline import evaluate_labeled_dataset, train_pipeline, verify_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="certificate-authenticator")
    sub = parser.add_subparsers(dest="command")

    verify = sub.add_parser("verify")
    verify.add_argument("--input", required=True, help="File or folder containing certificates")
    verify.add_argument("--output", default="tampering_report.csv", help="Output CSV path")
    verify.add_argument("--artifacts", default="artifacts", help="Path to model artifacts")
    verify.add_argument("--non-recursive", action="store_true", help="Disable recursive folder scan")

    train = sub.add_parser("train")
    train.add_argument("--dataset", required=True, help="Dataset root with class folders")
    train.add_argument("--artifacts", default="artifacts", help="Output artifacts folder")
    train.add_argument("--epochs", type=int, default=60, help="Training epochs")
    train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--dataset", required=True, help="Labeled dataset root with authentic/fraudulent folders")
    evaluate.add_argument("--artifacts", default="artifacts", help="Artifacts folder")
    evaluate.add_argument("--output", default="tampering_evaluation.csv", help="Evaluation CSV path")

    parser.add_argument("--input", help="Backward-compatible shortcut for verification")
    parser.add_argument("--output", default="tampering_report.csv")
    parser.add_argument("--artifacts", default="artifacts")
    return parser


def run() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        stats = train_pipeline(args.dataset, args.artifacts, epochs=args.epochs, lr=args.lr)
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "evaluate":
        stats = evaluate_labeled_dataset(args.dataset, args.artifacts, args.output)
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "verify":
        result = verify_pipeline(
            input_path=args.input,
            output_csv=args.output,
            artifacts_dir=args.artifacts,
            recursive=not args.non_recursive,
        )
        print(json.dumps({"rows": len(result.rows), "output_csv": result.output_csv}, indent=2))
        return 0

    if args.input:
        result = verify_pipeline(
            input_path=args.input,
            output_csv=args.output,
            artifacts_dir=args.artifacts,
            recursive=True,
        )
        print(json.dumps({"rows": len(result.rows), "output_csv": result.output_csv}, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(run())
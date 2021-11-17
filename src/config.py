import argparse


def load_config():
    parser = argparse.ArgumentParser()

    # default config
    parser.add_argument("--root-path", type=str, default="src/data")
    parser.add_argument("--ckpt-path", type=str, default="src/checkpoints/")
    parser.add_argument("--result-path", type=str, default="src/results.csv")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-step", type=int, default=200)
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Evaluation will be done at the end of epoch if set to 0.0",
    )
    parser.add_argument(
        "--amp", action="store_true", default=False, help="PyTorch(>=1.6.x) AMP"
    )

    # ddp config
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Whether to use ddp"
    )
    parser.add_argument("--dist-backend", type=str, default="nccl")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:3456", type=str)
    parser.add_argument(
        "--world-size", type=int, default=1, help="Total number of processes to run"
    )
    parser.add_argument(
        "--rank", type=int, default=-1, help="Local GPU rank (-1 if not using ddp)"
    )

    # training config
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="It means batch size per gpu in ddp"
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-step", type=int, default=1)

    args = parser.parse_args()
    return args

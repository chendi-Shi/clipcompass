import argparse
import json
from pathlib import Path

from .data import import_kuairec, make_demo


def main():
    p = argparse.ArgumentParser(description="ClipCompass: content and recommendation lab")
    sub = p.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo-data")
    demo.add_argument("--output", default="data/demo")
    demo.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train")
    train.add_argument("--data", default="data/demo")
    train.add_argument("--output", default="artifacts")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--seed", type=int, default=42)
    rec = sub.add_parser("recommend")
    rec.add_argument("--checkpoint", default="artifacts/model.pt")
    rec.add_argument("--user", type=int, default=0)
    rec.add_argument("--k", type=int, default=5)
    rec.add_argument("--output")
    imp = sub.add_parser("import-kuairec")
    imp.add_argument("--matrix", required=True)
    imp.add_argument("--captions", required=True)
    imp.add_argument("--output", default="data/kuairec")
    imp.add_argument("--max-rows", type=int, default=300000)
    args = p.parse_args()
    if args.command == "demo-data":
        make_demo(args.output, args.seed)
        print(f"Synthetic dataset created: {args.output}")
    elif args.command == "train":
        if args.epochs < 1:
            p.error("epochs must be >=1")
        from .train import train_suite
        report = train_suite(args.data, args.output, args.epochs, args.seed)
        print(json.dumps(report["experiments"], indent=2))
    elif args.command == "recommend":
        from .recommend import recommend
        content = json.dumps(recommend(args.checkpoint, args.user, args.k), ensure_ascii=False, indent=2)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        else:
            print(content)
    elif args.command == "import-kuairec":
        import_kuairec(args.matrix, args.captions, args.output, max_rows=args.max_rows)


if __name__ == "__main__":
    main()

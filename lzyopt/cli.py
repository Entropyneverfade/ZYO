# 旧命令行参数及 JSON 输出接口；新使用说明以 zyo.cli 为准。
# Legacy argument parsing and JSON output; new documentation targets zyo.cli.
import argparse
import json
import platform
from pathlib import Path
import sys
from . import __version__
from .model import Model
from .solver import Options
from .examples import EXAMPLES


def main():
    parser = argparse.ArgumentParser(description="LZYOpt LP/MILP research prototype")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Print local runtime/dependency versions")
    for command in ("solve", "demo"):
        p = sub.add_parser(command)
        if command == "solve":
            p.add_argument("model")
        else:
            p.add_argument("example", choices=EXAMPLES, default="capacity", nargs="?")
        p.add_argument("--engine", choices=("native", "highs"), default="native")
        p.add_argument("--time-limit", type=float, default=60.0)
        p.add_argument("--node-limit", type=int, default=10000)
        p.add_argument("--gap", type=float, default=0.0)
        p.add_argument("--output")
        p.add_argument("--write-model")
    args = parser.parse_args()
    if args.command == "doctor":
        import numpy
        info = {"lzyopt": __version__, "python": sys.version, "platform": platform.platform(),
                "executable": sys.executable, "numpy": numpy.__version__, "scipy": None}
        try:
            import scipy
            info["scipy"] = scipy.__version__
        except ImportError:
            pass
        print(json.dumps(info, indent=2))
        return 0
    try:
        model = Model.read(args.model) if args.command == "solve" else EXAMPLES[args.example]()
        options = Options(time_limit=args.time_limit, node_limit=args.node_limit, mip_rel_gap=args.gap)
        if args.write_model:
            Path(args.write_model).parent.mkdir(parents=True, exist_ok=True)
            model.write(args.write_model)
        result = model.optimize(engine=args.engine, options=options)
        output = {"model": model.name, "lzyopt_version": __version__, **result.to_dict()}
        text = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
        print(text)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(text + "\n", encoding="utf-8")
        # Feasible-but-unproven is distinguishable from optimal in scripts.
        return 0 if result.status == "OPTIMAL" else 2
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        print(f"LZYOpt error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

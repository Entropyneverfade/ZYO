# 统一命令行：参数解析、环境诊断、显式求解入口和结果导出。
"""Command-line interface for ZYO."""

import argparse
from importlib import metadata, util
import json
import platform
from pathlib import Path
import sys

from lzyopt.examples import EXAMPLES

from . import __version__
from .errors import ZYOError
from .model import Model
from .solvers import available_solvers


def _package_version(import_name, distribution_name=None):
    if util.find_spec(import_name) is None:
        return None
    try:
        return metadata.version(distribution_name or import_name)
    except metadata.PackageNotFoundError:
        return "available-version-unknown"


def _doctor_payload():
    import numpy

    solvers = {}
    for name, info in available_solvers().items():
        solvers[name] = {
            "version": info.version,
            "available": info.available,
            "capabilities": sorted(info.capabilities),
            "license_mode": info.license_mode,
            "message": info.message,
        }
    return {
        "zyo": __version__,
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "dependencies": {
            "numpy": numpy.__version__,
            "scipy": _package_version("scipy"),
            "highs": _package_version("highspy"),
            "gurobi": _package_version("gurobipy"),
            "copt": _package_version("coptpy"),
        },
        "engine_policy": {
            "default": "native",
            "implicit_fallback": False,
            "external_engines_are_comparison_backends": True,
        },
        "solvers": solvers,
    }


def _add_solve_arguments(parser, *, demo=False):
    if demo:
        parser.add_argument("example", choices=EXAMPLES, default="capacity", nargs="?")
    else:
        parser.add_argument("model")
    parser.add_argument("--solver", default="native")
    parser.add_argument("--engine", dest="solver", help=argparse.SUPPRESS)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--node-limit", type=int, default=10_000)
    parser.add_argument("--mip-gap", "--gap", dest="mip_gap", type=float, default=0.0)
    parser.add_argument("--output")
    parser.add_argument("--write-model")


def _parser():
    parser = argparse.ArgumentParser(
        prog="zyo",
        description="ZYO general-purpose LP/MILP research optimizer",
    )
    parser.add_argument("--version", action="version", version=f"ZYO {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Report the local runtime and optional packages")
    _add_solve_arguments(subparsers.add_parser("solve"))
    _add_solve_arguments(subparsers.add_parser("demo"), demo=True)
    return parser


def _write_text(path, text):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(_doctor_payload(), ensure_ascii=False, indent=2))
        return 0

    try:
        if args.command == "solve":
            model = Model.read(args.model)
        else:
            model = Model.from_dict(EXAMPLES[args.example]().to_dict())
        if args.write_model:
            target = Path(args.write_model)
            target.parent.mkdir(parents=True, exist_ok=True)
            model.write(target)
        result = model.solve(
            solver=args.solver,
            time_limit=args.time_limit,
            node_limit=args.node_limit,
            mip_gap=args.mip_gap,
        )
        payload = {
            "model": model.name,
            "zyo_version": __version__,
            **result.to_dict(),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        print(text, end="")
        if args.output:
            _write_text(args.output, text)
        return 0 if result.status.value == "OPTIMAL" else 2
    except (ZYOError, ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        print(f"ZYO error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

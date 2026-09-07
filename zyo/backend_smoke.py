# 小规模后端契约检查；外部引擎仅用于比较，冒烟测试不代表性能排名。
"""Generate a small, auditable comparison of every installed ZYO backend."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from . import BINARY, INTEGER, Model, SolveOptions, __version__, available_solvers


def _canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value):
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _git_value(*args):
    try:
        run = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return run.stdout.strip() if run.returncode == 0 else None


def _models():
    lp = Model("backend-smoke-lp")
    x = lp.add_var("x")
    y = lp.add_var("y")
    lp.add_constr(x + y >= 3, "demand")
    lp.add_constr(x <= 2, "x_limit")
    lp.add_constr(y <= 4, "y_limit")
    lp.minimize(2 * x + y)

    milp = Model("backend-smoke-milp")
    units = milp.add_var("units", lb=0, ub=4, vtype=INTEGER)
    switch = milp.add_var("switch", vtype=BINARY)
    milp.add_constr(2 * units + switch >= 5, "coverage")
    milp.minimize(3 * units + switch)
    return {"lp": lp, "milp": milp}


def _model_record(model):
    data = model.to_dict()
    return {
        "sha256": _sha256(data),
        "format": data["format"],
        "dimensions": {
            "variables": len(model.variables),
            "integer_variables": sum(v.kind in {"I", "B"} for v in model.variables),
            "constraints": len(model.constraints),
            "constraint_nonzeros": sum(
                len(row.expression.terms) for row in model.constraints
            ),
            "objective_nonzeros": len(model.objective.terms),
        },
        "model": data,
    }


def build_payload():
    models = _models()
    records = {name: _model_record(model) for name, model in models.items()}
    # This historical protocol has four frozen comparison engines and fixtures.
    # Experimental finite-box kernels have their own independent acceptance set.
    infos = {k:v for k,v in available_solvers().items() if k in ('native','highs','gurobi','copt')}
    options = SolveOptions(threads=1, seed=0, presolve=True)
    runs = []
    for solver, info in infos.items():
        if not info.available:
            continue
        for model_name, model in models.items():
            result = model.solve(solver=solver, options=options)
            runs.append(
                {
                    "solver": solver,
                    "model": model_name,
                    "model_sha256": records[model_name]["sha256"],
                    **result.to_dict(),
                }
            )

    status = _git_value("status", "--porcelain")
    return {
        "schema_version": "zyo-backend-smoke-1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "zyo_version": __version__,
        "code": {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty_before_artifact_write": bool(status) if status is not None else None,
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "policy": {
            "implicit_fallback": False,
            "third_party_results_count_as_native": False,
            "candidate_validation": "independent-original-units",
        },
        "options": {
            name: getattr(options, name)
            for name in options.__dataclass_fields__
        },
        "models": records,
        "solvers": {
            name: {
                "version": info.version,
                "available": info.available,
                "capabilities": sorted(info.capabilities),
                "license_mode": info.license_mode,
                "message": info.message,
            }
            for name, info in infos.items()
        },
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    payload = build_payload()
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    print(target.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate metrics then register model if thresholds pass.

Usage: python scripts/validate_and_register.py --run-id <RUN_ID> --min-pr-auc 0.12
"""
import argparse
import json
import subprocess
from pathlib import Path


def validate_and_register(
    run_id: str,
    run_dir: str = ".",
    min_pr_auc: float = 0.12,
    model_name: str = "avazu_ctr",
    artifact_path: str = "model/ctr_model_hashing.joblib",
):
    mfile = Path(run_dir) / "metrics" / "metrics.json"
    if not mfile.exists():
        raise FileNotFoundError(f"metrics file not found: {mfile}")

    metrics = json.loads(mfile.read_text())
    pr = metrics.get("val_pr_auc", 0.0)
    if pr >= min_pr_auc:
        subprocess.check_call(
            [
                "python",
                "scripts/register_model.py",
                "--run-id",
                run_id,
                "--name",
                model_name,
                "--artifact-path",
                artifact_path,
            ]
        )
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-dir", default=".")
    p.add_argument("--min-pr-auc", type=float, default=0.12)
    p.add_argument("--artifact-path", default="model/ctr_model_hashing.joblib")
    args = p.parse_args()

    ok = validate_and_register(args.run_id, args.run_dir, args.min_pr_auc, artifact_path=args.artifact_path)
    if ok:
        print("Registered")
    else:
        print("Not registered: PR-AUC below threshold")


if __name__ == "__main__":
    main()

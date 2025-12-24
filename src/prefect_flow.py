import subprocess
from pathlib import Path

from prefect import flow, task


@task
def run_training():
    subprocess.check_call(["python", "src/train_streaming.py"])


@task
def load_run_id(path: str = "models/last_run_id.txt") -> str:
    run_id_path = Path(path)
    if not run_id_path.exists():
        raise FileNotFoundError(f"run id not found: {run_id_path}")
    return run_id_path.read_text().strip()


@task
def validate_and_register(run_id: str, min_pr_auc: float = 0.12, run_dir: str = "."):
    subprocess.check_call(
        [
            "python",
            "scripts/validate_and_register.py",
            "--run-id",
            run_id,
            "--run-dir",
            run_dir,
            "--min-pr-auc",
            str(min_pr_auc),
        ]
    )


@flow(name="avazu_ctr_training_pipeline")
def training_pipeline(min_pr_auc: float = 0.12, run_dir: str = "."):
    run_training()
    run_id = load_run_id()
    validate_and_register(run_id, min_pr_auc=min_pr_auc, run_dir=run_dir)


if __name__ == "__main__":
    training_pipeline()

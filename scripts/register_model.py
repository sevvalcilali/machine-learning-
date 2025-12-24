"""Register a model artifact from a run to MLflow Model Registry and transition stage.

Usage: python scripts/register_model.py --run-id <RUN_ID> --name avazu_ctr
"""
import argparse
import time
from mlflow import MlflowClient


def register_model(
    run_id: str,
    name: str,
    artifact_path: str = "model/ctr_model_hashing.joblib",
    stage: str = "Staging",
):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/{artifact_path}"

    # Create model if missing
    existing = [m.name for m in client.list_registered_models()]
    if name not in existing:
        client.create_registered_model(name)

    mv = client.create_model_version(name=name, source=model_uri, run_id=run_id)

    # Wait until ready or timeout
    for _ in range(30):
        v = client.get_model_version(name, mv.version)
        if v.status == "READY":
            break
        time.sleep(1)

    client.transition_model_version_stage(name=name, version=mv.version, stage=stage)
    return mv.version


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--artifact-path", default="model/ctr_model_hashing.joblib")
    p.add_argument("--stage", default="Staging")
    args = p.parse_args()

    ver = register_model(args.run_id, args.name, args.artifact_path, args.stage)
    print(f"Registered {args.name} v{ver} -> {args.stage}")


if __name__ == "__main__":
    main()

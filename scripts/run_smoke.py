"""Small smoke runner for `src/train_streaming.py`.

Usage: python scripts/run_smoke.py
"""
import subprocess


def run():
    print("Running short smoke training (2 train chunks + 1 val chunk)...")
    subprocess.check_call(
        [
            "python",
            "src/train_streaming.py",
            "--chunk-size",
            "200",
            "--max-train-chunks",
            "2",
            "--val-chunks",
            "1",
            "--checkpoint-every",
            "1",
        ]
    )


if __name__ == "__main__":
    run()

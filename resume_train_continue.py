from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    log_path = repo_root / "runs" / "continue_from_69_resume.log"
    model_path = repo_root / "runs" / "detect" / "train_continue" / "weights" / "last.pt"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        print(f"[{datetime.now().isoformat(timespec='seconds')}] Resuming training from {model_path}", file=log_file, flush=True)
        with redirect_stdout(log_file), redirect_stderr(log_file):
            model = YOLO(str(model_path))
            model.train(resume=True)


if __name__ == "__main__":
    main()

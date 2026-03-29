import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd


def get_kaggle_competition() -> str:
    competition = os.getenv("KAGGLE_COMPETITION")
    if not competition:
        raise ValueError("KAGGLE_COMPETITION is not set in the environment.")
    return competition


def create_kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def download_competition_data(data_dir: Path) -> Path:
    competition = get_kaggle_competition()
    api = create_kaggle_api()
    data_dir.mkdir(parents=True, exist_ok=True)

    for path in data_dir.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    api.competition_download_files(
        competition=competition, path=str(data_dir), quiet=True
    )

    archive_path = data_dir / f"{competition}.zip"
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(data_dir)
        archive_path.unlink()

    return data_dir


def identify_files(data_dir: Path) -> dict:
    csv_files = sorted(data_dir.glob("*.csv"))

    def pick_exact_or_contains(exact_name: str, contains_name: str) -> Optional[Path]:
        exact_path = data_dir / exact_name
        if exact_path.exists():
            return exact_path
        for path in csv_files:
            if contains_name in path.name.lower():
                return path
        return None

    return {
        "train_path": pick_exact_or_contains("train.csv", "train"),
        "test_path": pick_exact_or_contains("test.csv", "test"),
        "sample_submission_path": (
            pick_exact_or_contains("sample_submission.csv", "submission")
            or pick_exact_or_contains("sample_submition.csv", "submition")
        ),
    }


def detect_target(train_path: Path, test_path: Path) -> str:
    train_columns = pd.read_csv(train_path, nrows=1).columns.tolist()
    test_columns = pd.read_csv(test_path, nrows=1).columns.tolist()
    return next(column for column in train_columns if column not in test_columns)


def prepare_kaggle_data(data_dir: Path) -> dict:
    download_competition_data(data_dir)
    files = identify_files(data_dir)
    target_column = detect_target(files["train_path"], files["test_path"])

    return {
        "train_path": files["train_path"],
        "test_path": files["test_path"],
        "sample_submission_path": files["sample_submission_path"],
        "target_column": target_column,
    }


def submit_to_kaggle(submission_path: Path, message: str) -> None:
    api = create_kaggle_api()
    api.competition_submit(
        file_name=str(submission_path),
        message=message,
        competition=get_kaggle_competition(),
    )

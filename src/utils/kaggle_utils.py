import os
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def get_kaggle_competition() -> str:
    competition = os.getenv("KAGGLE_COMPETITION")
    if not competition:
        raise ValueError("KAGGLE_COMPETITION is not set in the environment.")
    return competition


def create_kaggle_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def submit_to_kaggle(submission_path: Path, message: str) -> None:
    api = create_kaggle_api()
    api.competition_submit(
        file_name=str(submission_path),
        message=message,
        competition=get_kaggle_competition(),
    )

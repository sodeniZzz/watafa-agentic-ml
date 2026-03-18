import logging
import logging.config
from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json


def setup_logging(save_dir, log_config=None, default_level=logging.INFO):
    """
    Setup logging configuration.

    Args:
        save_dir (Path): path to directory, where all logs and
            checkpoints should be saved.
        log_config (str | None): path to logger config. If none
            'logger_config.json' from the src.logger directory is used.
        default_level (int): default logging level.
    """
    if log_config is None:
        log_config = ROOT_PATH / "src" / "logger" / "logger_config.json"
    log_config = Path(log_config)

    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])
        logging.config.dictConfig(config)
    else:
        print(f"[logger] Config not found at {log_config}, using basicConfig")
        logging.basicConfig(level=default_level)

import logging
import logging.config
from pathlib import Path

from src.utils.io_utils import ROOT_PATH, read_json

LINE_WIDTH = 50


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


def log_header(*lines: str):
    """Print a double-line header block to console."""
    border = "═" * LINE_WIDTH
    print(border)
    for line in lines:
        print(f"  {line}")
    print(border)


def log_stage(name: str):
    """Print a single-line stage separator to console."""
    label = f"── Stage: {name} "
    print(f"\n{label}{'─' * (LINE_WIDTH - len(label))}")


def format_duration(seconds: float) -> str:
    """Format seconds into 'Xm Ys' string."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

import json
from pathlib import Path


class StageReport:
    """Tracks attempts, timing, and token usage for a pipeline stage."""

    def __init__(self, stage_dir: Path):
        self.path = Path(stage_dir) / "stage_report.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "attempts": [],
                "total_duration_sec": 0.0,
                "total_tokens_in": 0,
                "total_tokens_out": 0,
            }
            self._save()

    def log_attempt(self, attempt, duration_sec, tokens_in=0, tokens_out=0,
                    returncode=0, stdout="", stderr="", error=None):
        self.data["attempts"].append({
            "attempt": attempt,
            "duration_sec": round(duration_sec, 1),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
        })
        self.data["total_duration_sec"] = round(
            self.data["total_duration_sec"] + duration_sec, 1
        )
        self.data["total_tokens_in"] += tokens_in
        self.data["total_tokens_out"] += tokens_out
        self._save()

    def log_validation(self, valid, feedback=""):
        if self.data["attempts"]:
            self.data["attempts"][-1]["valid"] = valid
            self.data["attempts"][-1]["feedback"] = feedback
        self.data["final_valid"] = valid
        self._save()

    @property
    def last_attempt(self):
        return self.data["attempts"][-1] if self.data["attempts"] else {}

    def _save(self):
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def build_benchmark_summary(output_path, stage_reports, model_metrics=None, best_model_name=None):
    agents = {}
    for name, report in stage_reports.items():
        agents[name] = {
            "attempts": len(report.data["attempts"]),
            "final_valid": report.data.get("final_valid"),
            "total_duration_sec": report.data["total_duration_sec"],
            "total_tokens_in": report.data["total_tokens_in"],
            "total_tokens_out": report.data["total_tokens_out"],
        }

    summary = {
        "agents": agents,
        "models": model_metrics or {},
        "best_model": best_model_name,
        "total_duration_sec": round(sum(a["total_duration_sec"] for a in agents.values()), 1),
        "total_tokens_in": sum(a["total_tokens_in"] for a in agents.values()),
        "total_tokens_out": sum(a["total_tokens_out"] for a in agents.values()),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

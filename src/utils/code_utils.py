import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

from src.utils.guardrails import validate_code


def extract_python_code(text: str) -> str:
    code = text.strip()
    if code.startswith("```python"):
        code = code.split("```python", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def run_python_code(
    script_path,
    work_dir: Optional[str] = None,
    timeout: int = 60,
    env_vars: Optional[Dict[str, str]] = None,
    capture_files: bool = True,
) -> Dict[str, Any]:
    script_path = Path(script_path)
    if work_dir is None:
        work_dir = script_path.parent
    else:
        work_dir = Path(work_dir)

    code_content = script_path.read_text(encoding="utf-8")
    is_safe, violations = validate_code(code_content)
    if not is_safe:
        return {
            "stdout": "",
            "stderr": "Code validation failed:\n" + "\n".join(f"  - {v}" for v in violations),
            "returncode": -1,
            "files_created": [],
            "error": "Code blocked by security guardrails",
        }

    before_files = set()
    if capture_files:
        before_files = set(work_dir.rglob("*"))

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding="utf-8",
        )

        stdout, stderr = process.communicate(timeout=timeout)
        returncode = process.returncode

        files_created = []
        if capture_files:
            after_files = set(work_dir.rglob("*"))
            new_files = after_files - before_files
            files_created = [str(f.relative_to(work_dir)) for f in new_files]

        result = {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "files_created": files_created,
            "error": None,
        }

    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        result = {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": -1,
            "files_created": [],
            "error": f"Превышен таймаут ({timeout} секунд)",
        }
    except Exception as e:
        result = {
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "files_created": [],
            "error": f"Ошибка при запуске процесса: {e}",
        }

    return result

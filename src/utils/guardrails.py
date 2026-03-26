"""Security guardrails for LLM-generated code and input data validation."""

import ast
import re
import logging
from typing import Tuple, List

import pandas as pd

logger = logging.getLogger(__name__)

BLOCKED_MODULES = frozenset({
    "subprocess", "socket", "requests", "urllib", "http",
    "ftplib", "smtplib", "ctypes", "multiprocessing", "signal", "shutil",
})

BLOCKED_CALLS = frozenset({
    "exec", "eval", "__import__", "compile",
    "globals", "locals", "getattr", "setattr", "delattr",
})

BLOCKED_OS_ATTRS = frozenset({
    "system", "popen", "remove", "rmdir", "unlink",
    "execl", "execle", "execlp", "execlpe",
    "execv", "execve", "execvp", "execvpe",
    "spawnl", "spawnle", "spawnlp", "spawnlpe",
    "spawnv", "spawnve", "spawnvp", "spawnvpe",
})

INJECTION_PATTERNS = [
    re.compile(r"ignore.*instruction", re.IGNORECASE),
    re.compile(r"system.*prompt", re.IGNORECASE),
    re.compile(r"import\s+os\b", re.IGNORECASE),
    re.compile(r"\bsubprocess\b", re.IGNORECASE),
    re.compile(r"__import__", re.IGNORECASE),
    re.compile(r"\beval\s*\(", re.IGNORECASE),
    re.compile(r"\bexec\s*\(", re.IGNORECASE),
]

MAX_CELL_LENGTH = 1000


class _CodeAuditor(ast.NodeVisitor):
    """AST visitor that collects security violations."""

    def __init__(self):
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top in BLOCKED_MODULES:
                self.violations.append(
                    f"Blocked import: '{alias.name}' (line {node.lineno})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            top = node.module.split(".")[0]
            if top in BLOCKED_MODULES:
                self.violations.append(
                    f"Blocked import: 'from {node.module}' (line {node.lineno})"
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func

        if isinstance(func, ast.Name) and func.id in BLOCKED_CALLS:
            self.violations.append(
                f"Blocked call: '{func.id}()' (line {node.lineno})"
            )

        if isinstance(func, ast.Attribute):
            if (
                isinstance(func.value, ast.Name)
                and func.value.id == "os"
                and func.attr in BLOCKED_OS_ATTRS
            ):
                self.violations.append(
                    f"Blocked call: 'os.{func.attr}()' (line {node.lineno})"
                )

        self.generic_visit(node)


def validate_code(code: str) -> Tuple[bool, List[str]]:
    """Validate LLM-generated Python code for dangerous operations.

    Returns (is_safe, violations) where violations is a list of
    human-readable descriptions of blocked operations found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]

    auditor = _CodeAuditor()
    auditor.visit(tree)

    is_safe = len(auditor.violations) == 0
    if is_safe:
        logger.info("Code guardrail passed")
    else:
        logger.warning(
            "Code guardrail BLOCKED — %d violation(s): %s",
            len(auditor.violations),
            "; ".join(auditor.violations),
        )

    return is_safe, auditor.violations


def validate_csv(path: str) -> List[str]:
    """Check CSV file for formula injection and prompt injection patterns.

    Returns a list of warnings (does not block the pipeline).
    """
    warnings: List[str] = []

    try:
        df = pd.read_csv(path, nrows=100)
    except Exception as e:
        warnings.append(f"Failed to read CSV: {e}")
        return warnings

    str_cols = df.select_dtypes(include="object").columns

    for col in str_cols:
        for idx, value in df[col].dropna().items():
            val = str(value).strip()

            # Formula injection: starts with =, +, -, @ (but not negative numbers)
            if val and val[0] in ("=", "@"):
                warnings.append(
                    f"Formula injection suspect: column='{col}', row={idx}, starts with '{val[0]}'"
                )
            if val and val[0] in ("+", "-") and not _is_numeric(val):
                warnings.append(
                    f"Formula injection suspect: column='{col}', row={idx}, starts with '{val[0]}'"
                )

            # Prompt injection patterns
            for pattern in INJECTION_PATTERNS:
                if pattern.search(val):
                    warnings.append(
                        f"Prompt injection suspect: column='{col}', row={idx}, "
                        f"matched pattern '{pattern.pattern}'"
                    )
                    break

            # Anomalous cell length
            if len(val) > MAX_CELL_LENGTH:
                warnings.append(
                    f"Anomalous cell length: column='{col}', row={idx}, length={len(val)}"
                )

    if warnings:
        logger.warning("CSV guardrail found %d warning(s) in %s", len(warnings), path)
    else:
        logger.info("CSV guardrail passed: %s", path)

    return warnings


def _is_numeric(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False

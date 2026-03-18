import os
import sys
import tempfile
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, List, Any

def run_python_code(
    code: str,
    work_dir: Optional[str] = None,
    timeout: int = 60,
    env_vars: Optional[Dict[str, str]] = None,
    capture_files: bool = True
) -> Dict[str, Any]:
    """
    Выполняет переданный Python-код в изолированном подпроцессе и возвращает результаты.

    Аргументы:
        code: строка с Python-кодом для выполнения.
        work_dir: рабочая директория (если None, создаётся временная).
        timeout: максимальное время выполнения в секундах.
        env_vars: дополнительные переменные окружения (передаются в подпроцесс).
        capture_files: если True, возвращает список файлов, созданных/изменённых в work_dir.

    Возвращает словарь с ключами:
        stdout: вывод программы (строка).
        stderr: ошибки (строка).
        returncode: код возврата процесса.
        files_created: список путей к файлам, созданным в work_dir (относительно work_dir).
        error: сообщение об ошибке, если не удалось запустить процесс или превышен таймаут.
    """
    # Создаём рабочую директорию, если не указана
    if work_dir is None:
        work_dir_obj = tempfile.TemporaryDirectory()
        work_dir = work_dir_obj.name
    else:
        work_dir_obj = None
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    # Получаем список файлов до запуска (если нужно)
    before_files = set()
    if capture_files:
        before_files = set(Path(work_dir).rglob('*'))

    # Создаём временный файл с кодом в рабочей директории
    # Используем NamedTemporaryFile, но нужно, чтобы файл существовал после закрытия?
    # Лучше создать файл с фиксированным именем, например temp_code.py, и потом удалить.
    code_file = Path(work_dir) / f"temp_code_{int(time.time())}.py"
    try:
        code_file.write_text(code, encoding='utf-8')
    except Exception as e:
        return {
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "files_created": [],
            "error": f"Не удалось записать файл с кодом: {e}"
        }

    # Подготавливаем окружение
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    try:
        # Запускаем процесс
        process = subprocess.Popen(
            [sys.executable, str(code_file)],
            cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            encoding='utf-8'
        )

        stdout, stderr = process.communicate(timeout=timeout)
        returncode = process.returncode

        # Собираем созданные файлы
        files_created = []
        if capture_files:
            after_files = set(Path(work_dir).rglob('*'))
            new_files = after_files - before_files
            # Исключаем сам временный файл с кодом
            new_files = [f for f in new_files if f != code_file]
            files_created = [str(f.relative_to(work_dir)) for f in new_files]

        result = {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "files_created": files_created,
            "error": None
        }

    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        result = {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": -1,
            "files_created": [],
            "error": f"Превышен таймаут ({timeout} секунд)"
        }
    except Exception as e:
        result = {
            "stdout": "",
            "stderr": "",
            "returncode": -1,
            "files_created": [],
            "error": f"Ошибка при запуске процесса: {e}"
        }
    finally:
        # Удаляем временный файл с кодом
        try:
            code_file.unlink()
        except:
            pass
        # Если мы создавали временную директорию, удаляем её
        if work_dir_obj is not None:
            work_dir_obj.cleanup()

    return result

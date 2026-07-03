from __future__ import annotations
import logging
import os
from datetime import datetime


class StructuredLogger:
    def __init__(self, log_dir: str = "./logs", level: str = "INFO"):
        self._logger = logging.getLogger("ProductStandardMapping")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(module)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        try:
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(
                log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log"
            )
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)
        except Exception:
            pass

    def info(self, module: str, message: str, **kwargs: object) -> None:
        extra = self._format_kwargs(**kwargs)
        self._logger.info(f"[{module}] {message}{extra}")

    def warning(self, module: str, message: str, **kwargs: object) -> None:
        extra = self._format_kwargs(**kwargs)
        self._logger.warning(f"[{module}] {message}{extra}")

    def error(self, module: str, message: str, **kwargs: object) -> None:
        extra = self._format_kwargs(**kwargs)
        self._logger.error(f"[{module}] {message}{extra}")

    @staticmethod
    def _format_kwargs(**kwargs: object) -> str:
        if not kwargs:
            return ""
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        return " | " + ", ".join(parts)

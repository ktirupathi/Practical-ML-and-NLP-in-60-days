"""
Common Utilities: Shared helper functions for the semantic search engine.
"""

import os
import sys
import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__)
        level: Logging level
        log_dir: Optional directory for log files

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_dir:
        ensure_dir(log_dir)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name.replace('.', '_')}.log")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def timer(func):
    """Decorator to measure and log function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger = logging.getLogger(func.__module__)
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result

    return wrapper


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a dictionary to a JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict[str, Any]:
    """Load a dictionary from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_file_size_mb(path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def format_search_result(rank: int, passage: str, score: float, max_len: int = 200) -> str:
    """Format a single search result for display."""
    truncated = passage[:max_len] + "..." if len(passage) > max_len else passage
    return f"[{rank}] (score: {score:.4f}) {truncated}"


def print_search_results(query: str, results: list, latency_ms: float) -> None:
    """Pretty-print search results to console."""
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)} | Latency: {latency_ms:.1f}ms")
    print("-" * 80)

    for result in results:
        if hasattr(result, "rank"):
            print(format_search_result(result.rank, result.passage, result.score))
        elif isinstance(result, dict):
            print(format_search_result(
                result.get("rank", 0),
                result.get("passage", ""),
                result.get("score", 0.0),
            ))
        print()

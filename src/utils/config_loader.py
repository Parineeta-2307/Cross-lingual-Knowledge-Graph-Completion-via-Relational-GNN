"""Centralized configuration loader with caching and dot-notation access.

Why this module exists:
    Every module in the project needs access to config.yaml values. Without
    a centralized loader, each module would independently open and parse the
    YAML file, leading to:
      1. Repeated disk I/O on every import
      2. Inconsistent config handling (different error messages, no caching)
      3. No standard way to access nested keys

    This module reads config.yaml once, caches the result in a module-level
    dict, and provides a dot-notation accessor for clean nested key lookups.

Usage:
    from src.utils.config_loader import load_config, get

    config = load_config()                     # Full dict (cached after first call)
    dim = get("model.embedding_dim")           # Dot-notation access
    lr = get("training.learning_rate", 0.001)  # With default fallback
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger


# Module-level cache: stores the parsed config dict.
# After the first load_config() call, subsequent calls return this instantly.
_CONFIG_CACHE: dict[str, Any] = {}

# Track which path was loaded so we can detect conflicting loads
_LOADED_PATH: Optional[str] = None


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    """Load and parse config.yaml, with module-level caching.

    On the first call, reads the YAML file from disk and stores the result
    in a module-level dictionary. All subsequent calls return the cached
    copy without touching disk.

    Args:
        path: Path to config.yaml. Can be absolute or relative to the
            current working directory. Defaults to "config.yaml" in the
            project root.

    Returns:
        The full configuration dictionary parsed from config.yaml.

    Raises:
        FileNotFoundError: If config.yaml does not exist at the given path.
            The error message includes the resolved absolute path and a
            hint to check the working directory.

    Example:
        >>> config = load_config()
        >>> config["model"]["embedding_dim"]
        128
        >>> config is load_config()  # Same object — cached
        True
    """
    global _CONFIG_CACHE, _LOADED_PATH

    # Return cached config if already loaded
    if _CONFIG_CACHE and _LOADED_PATH is not None:
        return _CONFIG_CACHE

    config_path = Path(path).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"  Expected: config.yaml in the project root directory.\n"
            f"  Current working directory: {Path.cwd()}\n"
            f"  Hint: Run scripts from the project root, or pass "
            f"--config /absolute/path/to/config.yaml"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"config.yaml must be a YAML mapping (dict), got {type(config).__name__}. "
            f"Check the file format at: {config_path}"
        )

    _CONFIG_CACHE = config
    _LOADED_PATH = str(config_path)

    logger.info(f"Config loaded | path={config_path} | top_keys={list(config.keys())}")
    return _CONFIG_CACHE


def get(key_path: str, default: Any = None) -> Any:
    """Dot-notation accessor for nested config values.

    Traverses the config dictionary using dot-separated keys.
    If any key in the path is missing, returns the default value
    instead of raising a KeyError.

    Args:
        key_path: Dot-separated key path into the config dict.
            Examples: "model.embedding_dim", "data.languages",
            "training.learning_rate"
        default: Value to return if the key path doesn't exist.
            Defaults to None.

    Returns:
        The config value at the given path, or default if not found.

    Example:
        >>> get("model.embedding_dim")
        128
        >>> get("model.nonexistent_key", 42)
        42
        >>> get("data.languages")
        ['de', 'ja', 'nl', 'en']
    """
    config = load_config()

    keys = key_path.split(".")
    current: Any = config

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def reload_config(path: str = "config.yaml") -> dict[str, Any]:
    """Force-reload config from disk, bypassing the cache.

    Use this only when you've modified config.yaml at runtime and need
    the updated values. Normal usage should call load_config() which
    returns the cached version.

    Args:
        path: Path to config.yaml.

    Returns:
        Freshly loaded configuration dictionary.
    """
    global _CONFIG_CACHE, _LOADED_PATH

    _CONFIG_CACHE = {}
    _LOADED_PATH = None

    logger.debug(f"Config cache cleared | reloading from {path}")
    return load_config(path)

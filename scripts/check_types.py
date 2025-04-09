"""
Prints a tree with the current status of files with respect to type checking.
"""
# ruff: noqa: T201

import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_files(paths: Iterable[str]) -> list[Path]:
    """Find all Python files in the given paths."""
    files = []
    for p in paths:
        if (path := Path(p)).is_file():
            if path.suffix == ".py":
                files.append(path)
        else:
            files.extend(path.glob("**/*.py"))
    return files


def process_mypy_output(result: subprocess.CompletedProcess[str]) -> Counter[str]:
    """Process mypy JSON output and extract error files."""
    if result.stdout.strip():
        json_dump = "[" + result.stdout.replace("\n", ",\n")[:-2] + "]"
        mypy_data = json.loads(json_dump)
    else:
        mypy_data = []

    # Extract error files
    return Counter(error["file"] for error in mypy_data)


def set_nested_value(d: dict, keys: Sequence, value: Any) -> None:
    """Set value in nested dict."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


def build_file_tree(
    files: list[Path], error_files: Counter[str]
) -> tuple[dict[str, dict | int], int]:
    """Build a tree structure of files for display and count typed files."""
    tree: dict[str, dict | int] = {}
    typed_count = 0

    for path in sorted(files):
        # Check if file has type errors
        error_count = error_files.get(str(path), 0)
        if error_count == 0:
            typed_count += 1
        set_nested_value(tree, path.parts, error_count)
    return tree, typed_count


def print_tree(tree: dict[str, dict | int], prefix: str = "") -> None:
    """Print the tree of files with type hint status."""
    # Intentionally using print for the tree itself to output to stdout
    if not prefix:
        print(".")

    for i, (name, value) in enumerate(sorted(tree.items())):
        is_last = i == len(tree) - 1
        branch = "└" if is_last else "├"
        if isinstance(value, dict):
            print(f"{prefix}{branch}── {name}")
            branch = " " if is_last else "│"
            print_tree(value, f"{prefix}{branch}   ")
        else:
            checkbox = "[ ]" if value else "[x]"
            suffix = f": {value}" if value else ""
            print(f"{prefix}{branch}── {checkbox} {name}{suffix}")


def print_summary(typed: int, total: int) -> None:
    """Print summary statistics."""
    percentage = (typed / total * 100) if total > 0 else 0
    print(
        f"\nSummary: {typed}/{total} files have complete type hints ({percentage:.1f}%)"
    )


def main(paths: Sequence[str]) -> None:
    # Filter to only existing paths
    paths = [p for p in paths if Path(p).exists()]
    if not paths:
        raise ValueError("Error: No valid paths found")

    # Find Python files
    files = get_files(paths)

    # Check that there's at least one file to run the type checking on
    logger.info(f"Found {len(files)} Python files")
    if not files:
        raise ValueError("No Python files found to run type checking on")

    # Run mypy once
    logger.info("Running mypy...")
    result = subprocess.run(
        ["uv", "run", "mypy", "-O", "json", *paths],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )

    # Process mypy output
    error_files = process_mypy_output(result)

    # Build tree structure and count typed files
    tree, typed_count = build_file_tree(files, error_files)

    # Print tree
    logger.info("Type Hints Status:")
    print_tree(tree)

    # Print summary
    total = len(files)
    print_summary(typed_count, total)


if __name__ == "__main__":
    # Use command line arguments or default paths
    paths = sys.argv[1:] or ["prolif", "scripts", "tests"]
    main(paths)

"""
Prints a tree with the current status of files with respect to type checking.
"""
# ruff: noqa: T201

import json
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def get_files(paths):
    """Find all Python files in the given paths."""
    files = []
    for p in paths:
        if (path := Path(p)).is_file():
            if path.suffix == ".py":
                files.append(path)
        else:
            files.extend(path.glob("**/*.py"))
    return files


def process_mypy_output(result):
    """Process mypy JSON output and extract error files."""
    if result.stdout.strip():
        json_dump = "[" + result.stdout.replace("\n", ",\n")[:-2] + "]"
        mypy_data = json.loads(json_dump)
    else:
        mypy_data = []

    # Extract error files
    return {error["file"] for error in mypy_data}


def build_file_tree(files, error_files):
    """Build a tree structure of files for display and count typed files."""
    tree = defaultdict(list)
    typed_count = 0

    for path in sorted(files):
        # Get path as string
        path_str = str(path)

        # Get parts for display
        parts = path.parts

        # Create subdir string (everything except filename)
        subdir = str(Path(*parts[:-1])) if len(parts) > 1 else ""

        # Check if file has type errors
        has_types = path_str not in error_files
        if has_types:
            typed_count += 1

        # Just use the filename for display
        display_name = parts[-1]

        tree[subdir].append((display_name, has_types))
    return tree, typed_count


def print_tree(tree):
    """Print the tree of files with type hint status."""
    # Using logger for the header
    logger.info("\nType Hints Status:")
    # Intentionally using print for the tree itself to output to stdout
    print(".")

    for subdir, files_in_dir in sorted(tree.items()):
        if subdir:
            print(f"├── {subdir}")
            prefix = "│   "
        else:
            prefix = ""

        for j, (filename, has_types) in enumerate(sorted(files_in_dir)):
            is_last = j == len(files_in_dir) - 1
            checkbox = "[x]" if has_types else "[ ]"
            print(f"{prefix}{'└── ' if is_last else '├── '}{checkbox} {filename}")


def print_summary(typed, total):
    """Print summary statistics."""
    percentage = (typed / total * 100) if total > 0 else 0
    print(
        f"\nSummary: {typed}/{total} files have complete type hints ({percentage:.1f}%)"
    )


def main():
    # Setup - use current working directory
    base_dir = Path()

    # Use command line arguments or default paths
    paths = sys.argv[1:] or [
        base_dir / "prolif",
        base_dir / "tests",
    ]

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
    print_tree(tree)

    # Print summary
    total = len(files)
    print_summary(typed_count, total)


if __name__ == "__main__":
    main()

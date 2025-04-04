"""
Prints a tree with the current status of files with respect to type checking.
"""

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


def check_file_for_functions(file_path):
    """Check if a file has any function definitions."""
    with open(file_path) as f:
        content = f.read()
    # Simple check for "def " in file
    return "def " in content


def process_mypy_output(result, base_dir):
    """Process mypy JSON output and normalize error file paths."""
    if result.stdout.strip():
        json_dump = "[" + result.stdout.replace("\n", ",\n")[:-2] + "]"
        mypy_data = json.loads(json_dump)
    else:
        mypy_data = []
    # Extract error files, normalizing paths to be comparable
    error_files = set()
    for error in mypy_data:
        # Convert the absolute path from mypy errors to a path relative to base_dir
        error_path = Path(error["file"])
        try:
            rel_path = error_path.relative_to(base_dir)
            error_files.add(str(rel_path))
        except ValueError:
            # If the file is not relative to base_dir, use the full path
            error_files.add(str(error_path))
    return error_files


def build_file_tree(files, base_dir, error_files):
    """Build a tree structure of files for display."""
    tree = defaultdict(lambda: defaultdict(list))
    # Get files with no functions
    empty_files = []
    for file in files:
        if not check_file_for_functions(file):
            empty_files.append(str(file.relative_to(base_dir)))
    for path in sorted(files):
        rel_path = path.relative_to(base_dir)
        parts = rel_path.parts
        subdir = str(Path(*parts[:-1])) if len(parts) > 1 else ""
        rel_path_str = str(rel_path)
        # Determine status - empty, typed, or untyped
        if rel_path_str in empty_files:
            status = "empty"
        else:
            status = "typed" if rel_path_str not in error_files else "untyped"
        tree[base_dir][subdir].append((parts[-1], status))
    return tree, len(empty_files)


def print_tree(tree):
    """Print the tree of files with type hint status."""
    # here intentionally using print here instead of logger
    # to output the tree to stdout as requested
    print("\nType Hints Status:")  # noqa: T201
    print(".")  # noqa: T201
    for base_dir in sorted(tree):
        print(f"├── {Path(base_dir).name}")  # noqa: T201
        for _, (subdir, files_in_dir) in enumerate(sorted(tree[base_dir].items())):
            if subdir:
                print(f"│   ├── {subdir}")  # noqa: T201
                prefix = "│   │   "
            else:
                prefix = "│   "
            for j, (filename, status) in enumerate(sorted(files_in_dir)):
                is_last = j == len(files_in_dir) - 1
                if status == "empty":
                    checkbox = "[e]"
                elif status == "typed":
                    checkbox = "[x]"
                else:
                    checkbox = "[ ]"
                print(  # noqa: T201
                    f"{prefix}{'└── ' if is_last else '├── '}{checkbox} {filename}"
                )


def print_summary(typed, total, empty):
    """Print summary statistics."""
    files_with_functions = total - empty
    percentage = (typed / files_with_functions * 100) if files_with_functions > 0 else 0
    print(  # noqa: T201
        f"\nSummary: {typed}/{files_with_functions} files with functions have complete type hints ({percentage:.1f}%)"
    )
    if empty > 0:
        print(f"         {empty} files have no function definitions")  # noqa: T201


def main():
    # Setup
    script_path = Path(__file__).resolve().parent
    paths = sys.argv[1:] or [
        script_path.parent / "prolif",
        script_path.parent / "tests",
    ]
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
        ["uv", "run", "mypy", "--disallow-untyped-defs", "-O", "json", *paths],
        stdout=subprocess.PIPE,
        text=True,
        check=False,
    )
    # Get the absolute path of base_dir for comparison
    base_dir = script_path.parent.resolve()
    # Process mypy output
    error_files = process_mypy_output(result, base_dir)
    # Build tree structure
    tree, empty_count = build_file_tree(files, base_dir, error_files)
    # Print tree
    print_tree(tree)
    # Print summary
    typed = sum(
        1
        for path in files
        if str(path.relative_to(base_dir)) not in error_files
        and check_file_for_functions(path)  # Only count files with functions
    )
    total = len(files)
    print_summary(typed, total, empty_count)


if __name__ == "__main__":
    main()

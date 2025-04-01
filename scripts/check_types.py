"""
Prints a tree with the current status of files with respect to type checking.
"""

import json
import logging
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


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
    files = []
    for p in paths:
        if (path := Path(p)).is_file():
            if path.suffix == ".py":
                files.append(path)
        else:
            files.extend(path.glob("**/*.py"))

   # Check that there's at least one file to run the type checking on (and log how many files in total) else raise an error
    logger.info(f"Found {len(files)} Python files")
    if not files:
        raise ValueError("No Python files found to run type checking on")

   # Run mypy once
    logger.info("Running mypy...")
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        #Use JSON output
        subprocess.run(
            ["mypy", "--disallow-untyped-defs", "--json-report", tmp.name, *paths],
            check=True, #Check the return code of the process
            capture_output=True,
        )
    
        with open(tmp.name) as f:
            mypy_data = json.load(f)
    
    error_files = {error["file"] for error in mypy_data.get("errors", [])}
    # Build tree
    tree = defaultdict(lambda: defaultdict(list))
    for path in sorted(files):
        base_dir = paths[0]  
        rel_path = path.relative_to(base_dir) if path.is_relative_to(base_dir) else path
        parts = rel_path.parts
        subdir = str(Path(*parts[:-1])) if len(parts) > 1 else ""
        tree[base_dir][subdir].append((parts[-1], str(path) not in error_files))

    # Print tree
    print("\nType Hints Status:")
    print(".")
    for base_dir in sorted(tree):
        print(f"├── {Path(base_dir).name}")
        for i, (subdir, files_in_dir) in enumerate(sorted(tree[base_dir].items())):
            if subdir:
                print(f"│   ├── {subdir}")
                prefix = "│   │   "
            else:
                prefix = "│   "

            for j, (filename, has_types) in enumerate(sorted(files_in_dir)):
                is_last = j == len(files_in_dir) - 1
                checkbox = "[x]" if has_types else "[ ]"
                print(
                    f"{prefix}{'└── ' if is_last else '├── '}{checkbox} {filename}"
                )

    # Print summary
    typed = sum(1 for path in files if str(path) not in error_files)
    total = len(files)
    percentage = (typed / total * 100) if total else 0
    print(
        f"\nSummary: {typed}/{total} files have complete type hints ({percentage:.1f}%)"
    )


if __name__ == "__main__":
    main()

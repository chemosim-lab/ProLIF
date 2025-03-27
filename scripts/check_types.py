"""
Type hints checker using mypy.
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
    script_dir = Path(__file__).resolve().parent
    dirs = sys.argv[1:] or [
        script_dir.parent / "prolif",
        script_dir.parent / "tests",
    ]
    dirs = [d for d in dirs if Path(d).exists()]
    if not dirs:
        logger.error("Error: No valid directories found")
        return

    # Find Python files
    files = [
        (d, Path(root) / f, str(Path(root) / f))
        for d in dirs
        for root, _, fs in subprocess.os.walk(d)
        for f in fs
        if f.endswith(".py")
    ]

    logger.info(f"Found {len(files)} Python files")

    # Run mypy once
    logger.info("Running mypy...")
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        # Try JSON output
        subprocess.run(
            ["mypy", "--disallow-untyped-defs", "--json-report", tmp.name, *dirs],
            check=False,
            capture_output=True,
        )
        try:
            with open(tmp.name) as f:
                mypy_data = json.load(f)
            error_files = {error["file"] for error in mypy_data.get("errors", [])}
        except (json.JSONDecodeError, FileNotFoundError):
            # Fall back to text parsing
            result = subprocess.run(
                ["mypy", "--disallow-untyped-defs", *dirs],
                capture_output=True,
                text=True,
                check=False,
            )
            error_files = {
                line.split(":", 1)[0]
                for line in result.stdout.splitlines() + result.stderr.splitlines()
                if ": error:" in line and line.split(":", 1)[0].endswith(".py")
            }

    # Build tree
    tree = defaultdict(lambda: defaultdict(list))
    for base_dir, rel_path, abs_path in sorted(files):
        parts = Path(rel_path).parts
        subdir = str(Path(*parts[:-1])) if len(parts) > 1 else ""
        tree[base_dir][subdir].append((parts[-1], abs_path not in error_files))

    # Print tree
    logger.info("\nType Hints Status:")
    logger.info(".")
    for base_dir in sorted(tree):
        logger.info(f"├── {Path(base_dir).name}")
        for _i, (subdir, files_in_dir) in enumerate(sorted(tree[base_dir].items())):
            if subdir:
                logger.info(f"│   ├── {subdir}")
                prefix = "│   │   "
            else:
                prefix = "│   "

            for j, (filename, has_types) in enumerate(sorted(files_in_dir)):
                is_last = j == len(files_in_dir) - 1
                checkbox = "[x]" if has_types else "[ ]"
                logger.info(
                    f"{prefix}{'└── ' if is_last else '├── '}{checkbox} {filename}"
                )

    # Print summary
    typed = sum(1 for _, _, path in files if path not in error_files)
    total = len(files)
    percentage = (typed / total * 100) if total else 0
    logger.info(
        f"\nSummary: {typed}/{total} files have complete type hints ({percentage:.1f}%)"
    )


if __name__ == "__main__":
    main()

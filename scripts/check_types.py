import os
import subprocess

# Directories to scan
CODE_DIRS = ["../prolif", "../tests"]

def find_python_files(base_dirs):
    """Recursively finds all .py files in the given directories."""
    py_files = []
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in sorted(files):  # Ensure sorted order
                if file.endswith(".py"):
                    relative_path = os.path.relpath(os.path.join(root, file), start=base_dir)
                    py_files.append((base_dir, relative_path))
    return py_files

def check_type_hints(files):
    """Runs mypy on the list of Python files and returns fully typed ones."""
    fully_typed = set()
    
    for base_dir, file in files:
        file_path = os.path.join(base_dir, file)
        result = subprocess.run(["mypy", "--disallow-untyped-defs", file_path],
                                capture_output=True, text=True)
        
        # If mypy passes with no errors, mark as fully typed
        if result.returncode == 0:
            fully_typed.add((base_dir, file))
    
    return fully_typed

def generate_todo_structure(files, fully_typed):
    """Generates a tree-like TODO structure with checkboxes."""
    last_dir = None
    lines = []
    
    for base_dir, file in files:
        # Extract parent directory and filename
        parts = file.split(os.sep)
        parent_dirs = parts[:-1]
        filename = parts[-1]
        
        # Print parent directory only if it's new
        current_dir = os.path.join(base_dir, *parent_dirs)
        if current_dir and current_dir != last_dir:
            lines.append(f"    {current_dir}/")
            last_dir = current_dir
        
        # Mark with a checkbox
        status = "[✓]" if (base_dir, file) in fully_typed else "[ ]"
        lines.append(f"{status}     ├── {filename}")
    
    return "\n".join(lines)

# Find Python files
python_files = find_python_files(CODE_DIRS)

# Run type hint check
fully_typed_files = check_type_hints(python_files)

# Generate TODO output
todo_structure = generate_todo_structure(python_files, fully_typed_files)

# Save to a TODO file
with open("TODO_typing.txt", "w") as f:
    f.write(todo_structure)

print("TODO_typing.txt generated!")

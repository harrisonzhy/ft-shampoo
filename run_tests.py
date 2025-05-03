#!/usr/bin/env python3
import os
import sys
import unittest

SKIP = {".git", "__pycache__", "venv", ".venv", "env", "build"}

def add_project_paths():
    ROOT = os.path.abspath(os.path.dirname(__file__))

    # 1) make the project root importable
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    # 2) walk every subdirectory, pruning SKIP’d ones, and append them
    for current_dir, dirnames, _filenames in os.walk(ROOT):
        # in-place prune so we never recurse into e.g. venv or .git
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP and not d.startswith(".")
        ]

        # skip the root itself (we already added it)
        if current_dir == ROOT:
            continue

        # append instead of insert so root remains first
        if current_dir not in sys.path:
            sys.path.append(current_dir)

def main():
    add_project_paths()
    loader = unittest.TestLoader()

    # if user passed exactly one arg, treat it as the test to run
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        # if they gave a .py path
        if arg.endswith(".py"):
            test_file = os.path.abspath(arg)
            start_dir = os.path.dirname(test_file)
            pattern   = os.path.basename(test_file)
        else:
            # otherwise assume a bare test name under tests/
            start_dir = os.path.join(os.path.dirname(__file__), "tests")
            pattern   = arg if arg.endswith(".py") else f"{arg}.py"

        suite = loader.discover(start_dir=start_dir, pattern=pattern)
    else:
        # no args: run everything under tests/
        tests_dir = os.path.join(os.path.dirname(__file__), "tests")
        suite     = loader.discover(start_dir=tests_dir)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()
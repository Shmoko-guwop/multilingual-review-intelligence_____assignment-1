"""
Master test runner for Multilingual Review Intelligence.
Executes all unit and integration test suites.
"""
import unittest
import sys
import os
from pathlib import Path
import importlib.util

ROOT_DIR = Path(__file__).resolve().parent.parent

# Robust PathFinder for 9p/FUSE mounted filesystems
class DirectPathFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname in sys.modules:
            return None
        parts = fullname.split(".")
        base = str(ROOT_DIR)
        candidate_dir = os.path.join(base, *parts)
        init_py = os.path.join(candidate_dir, "__init__.py")
        file_py = candidate_dir + ".py"
        if os.path.isdir(candidate_dir) and os.path.isfile(init_py):
            return importlib.util.spec_from_file_location(fullname, init_py, submodule_search_locations=[candidate_dir])
        elif os.path.isfile(file_py):
            return importlib.util.spec_from_file_location(fullname, file_py)
        return None

sys.meta_path.insert(0, DirectPathFinder)
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def run_all():
    print("=" * 70)
    print("Running Multilingual Review Intelligence Test Suites")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    tests_dir = ROOT_DIR / "tests"
    for test_file in sorted(tests_dir.glob("test_*.py")):
        mod_name = f"tests.{test_file.stem}"
        spec = DirectPathFinder.find_spec(mod_name)
        if spec:
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            suite.addTests(loader.loadTestsFromModule(mod))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 70)
    if result.wasSuccessful():
        print(f"✓ ALL TESTS PASSED ({result.testsRun} tests executed successfully)")
        print("=" * 70)
        return 0
    else:
        print(f"❌ FAILURES OCCURRED: {len(result.failures)} failures, {len(result.errors)} errors")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all())

"""Compile the real host runtime against CPU-only ACL failure-injection stubs."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeGuardTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "CPU runtime guard tests require g++")
    def test_host_guards_and_failed_prepare(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "runtime_guards"
            subprocess.run([
                "g++", "-std=c++17", "-Wall", "-Wextra", "-Itests/stubs", "-Iinclude",
                "tests/test_runtime_guards.cpp", "src/framework/butterfly.cpp",
                "src/framework/reference.cpp", "-o", str(binary),
            ], cwd=ROOT, check=True, capture_output=True, text=True)
            subprocess.run([str(binary), temporary], cwd=ROOT, check=True,
                           capture_output=True, text=True)

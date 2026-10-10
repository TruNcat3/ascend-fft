import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProbeScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "tools").mkdir()
        for name in ("env.sh", "build.sh", "hw_probe.sh", "baseline_o.sh"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        self.env = dict(os.environ, AB_ROOT=str(self.root),
                        AB_BUILD=str(self.root / "build"),
                        AB_WORK=str(self.root / ".tmp"), AB_PY="python3",
                        AB_CANN=str(self.root / "cann"), AB_INC="",
                        AB_MSPROF="missing", AB_ENV_SOURCED="1")
        self.env["PATH"] = str(self.root / "tools") + os.pathsep + os.environ["PATH"]
        (self.root / "cann/include").mkdir(parents=True)
        (self.root / "cann/lib64").mkdir()

    def write_tool(self, name, content):
        path = self.root / "tools" / name
        path.write_text("#!/usr/bin/env bash\n" + content)
        path.chmod(0o755)

    def run_script(self, name, *args):
        return subprocess.run(["bash", "scripts/" + name, *args], cwd=self.root,
                              env=self.env, capture_output=True, text=True)

    def compiler(self):
        self.write_tool("ccec", """if [[ "$*" == *probe_simt* ]]; then
  case "${SIMT_RESULT:-supported}" in
    unsupported) echo 'error: SIMT instruction is not supported on target Ascend910_9382' >&2; exit 1;;
    header) echo "fatal error: 'simt_api/asc_simt.h' file not found" >&2; exit 1;;
    symbol) echo 'error: use of undeclared identifier Simt' >&2; exit 1;;
    mixed) echo 'error: SIMT instruction is not supported on target'; echo "fatal error: header file not found" >&2; exit 1;;
  esac
fi
while [[ $# -gt 0 ]]; do
  if [[ "$1" == -o ]]; then touch "$2"; exit 0; fi
  shift
done
""")
        self.write_tool("g++", """printf '%s\\n' "$*" >> "$AB_WORK/host_commands"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == -o ]]; then printf '#!/usr/bin/env bash\\nexit 0\\n' > "$2"; chmod +x "$2"; exit 0; fi
  shift
done
""")

    def test_clean_probe_build_includes_gather_host(self):
        self.compiler()
        result = self.run_script("build.sh", "probe")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "build/gather_probe").is_file())
        self.assertIn("src/host/gather_probe.cpp", (self.root / ".tmp/host_commands").read_text())

    def test_simt_only_explicit_target_diagnostic_passes(self):
        self.compiler()
        for mode, expected in (("supported", 0), ("unsupported", 0),
                               ("header", 1), ("symbol", 1), ("mixed", 1)):
            with self.subTest(mode=mode):
                self.env["SIMT_RESULT"] = mode
                result = self.run_script("hw_probe.sh", "--only", "simt")
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        self.write_tool("ccec", "echo 'ccec: command not found' >&2; exit 127\n")
        self.assertEqual(self.run_script("hw_probe.sh", "--only", "simt").returncode, 1)

    def test_gather_uses_exit_status_not_label(self):
        (self.root / "build").mkdir()
        path = self.root / "build/gather_probe"
        for code, marker, expected in ((2, True, 1), (0, False, 1), (0, True, 0)):
            message = "echo 'Gather byte-offset/base validation: PASS (0 mismatches)'\n" if marker else ""
            path.write_text(f"#!/usr/bin/env bash\necho 'B(offset*4): 0 0 0'\n{message}exit {code}\n")
            path.chmod(0o755)
            result = self.run_script("hw_probe.sh", "--only", "gather")
            self.assertEqual(result.returncode, expected)

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.root, text=True).strip()

    def test_baseline_archived_environment_and_moving_ref_cache(self):
        self.compiler()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.org")
        self.git("config", "user.name", "Test")
        build = self.root / "scripts/build.sh"
        build.write_text("""#!/usr/bin/env bash
set -eu
source scripts/env.sh
[[ "$AB_ROOT" == "$PWD" && "$AB_BUILD" == "$PWD/build" && "$AB_WORK" == "$PWD/.tmp" ]]
cp include/version "$AB_BUILD/fft_radix2.o"
""")
        build.chmod(0o755)
        (self.root / "include").mkdir()
        for version in ("old", "new"):
            (self.root / "include/version").write_text(version)
            self.git("add", "scripts", "include")
            self.git("commit", "-qm", version)
            sha = self.git("rev-parse", "HEAD")
            result = self.run_script("baseline_o.sh", "HEAD", "fft_radix2", "label")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            outputs = list((self.root / "build").glob(f"baseline_label_{sha}_*_fft_radix2.o"))
            self.assertEqual(len(outputs), 1)
            output = outputs[0]
            self.assertEqual(output.read_text(), version)
            self.assertIn(f"source_commit={sha}", Path(str(output) + ".manifest").read_text())
        self.assertEqual(len(list((self.root / "build").glob("baseline_label_*.o"))), 2)

    def test_baseline_cache_binds_soc_toolchain_and_build_environment(self):
        self.compiler()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.org")
        self.git("config", "user.name", "Test")
        build = self.root / "scripts/build.sh"
        build.write_text("""#!/usr/bin/env bash
set -eu
source scripts/env.sh
printf '%s\\n' built >> "$AB_ROOT/../build_calls"
printf '%s\\n' "$AB_SOC" > "$AB_BUILD/fft_radix2.o"
""")
        build.chmod(0o755)
        self.git("add", "scripts")
        self.git("commit", "-qm", "fixture")

        def run_baseline():
            result = self.run_script("baseline_o.sh", "HEAD")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return Path(result.stdout.splitlines()[-1])

        original = run_baseline()
        self.assertEqual(original, run_baseline())
        self.assertIn("已有", self.run_script("baseline_o.sh", "HEAD").stdout)
        self.env["AB_SOC"] = "AscendOther"
        soc = run_baseline()
        self.assertNotEqual(original, soc)
        self.assertEqual(soc.read_text().strip(), "AscendOther")
        self.write_tool("ccec", "echo 'ccec version 2'\n")
        compiler = run_baseline()
        self.assertNotEqual(soc, compiler)
        (self.root / "cann/version.info").write_text("CANN 10\n")
        toolkit = run_baseline()
        self.assertNotEqual(compiler, toolkit)
        self.env["AB_INC"] = "-I/custom/headers"
        include = run_baseline()
        self.assertNotEqual(toolkit, include)
        Path(str(include) + ".manifest").unlink()
        self.assertEqual(include, run_baseline())
        self.assertEqual(len((self.root / ".tmp/build_calls").read_text().splitlines()), 6)


@unittest.skipUnless(shutil.which("g++"), "g++ required for host probe validation")
class GatherValidationTests(unittest.TestCase):
    def test_real_host_rejects_corrupt_and_nonfinite_gather_results(self):
        # Emulate ACL transport only; exercise the production host comparison.
        header = r'''#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
using aclError=int; using aclrtStream=void*; using aclrtBinHandle=void*; using aclrtFuncHandle=void*;
constexpr int ACL_MEM_MALLOC_NORMAL_ONLY=0, ACL_MEMCPY_HOST_TO_DEVICE=0, ACL_MEMCPY_DEVICE_TO_HOST=1;
inline int aclInit(void*){return 0;} inline int aclFinalize(){return 0;}
inline int aclrtSetDevice(int){return 0;} inline int aclrtResetDevice(int){return 0;}
inline int aclrtCreateStream(void** p){*p=nullptr;return 0;} inline int aclrtDestroyStream(void*){return 0;}
inline int aclrtBinaryLoadFromFile(const char*,void*,void** p){*p=nullptr;return 0;}
inline int aclrtBinaryGetFunction(void*,const char*,void** p){*p=nullptr;return 0;}
inline int aclrtBinaryUnLoad(void*){return 0;}
inline int aclrtMalloc(void** p,size_t n,int){*p=std::malloc(n);return *p?0:1;}
inline int aclrtFree(void* p){std::free(p);return 0;}
inline int aclrtMemcpy(void* d,size_t,const void* s,size_t n,int){std::memcpy(d,s,n);return 0;}
inline int aclrtMemset(void* d,size_t,int v,size_t n){std::memset(d,v,n);return 0;}
inline int aclrtSynchronizeStream(void*){return 0;}
inline int aclrtLaunchKernelWithHostArgs(void*,int,void*,void*,void* args,size_t,void*,int){
  uintptr_t op,sp; uint32_t n;
  std::memcpy(&op,args,8);std::memcpy(&sp,(char*)args+8,8);std::memcpy(&n,(char*)args+16,4);
  if(n!=96)return 1;
  auto out=(float*)op;auto src=(float*)sp;
  for(int i=0;i<32;i++)out[32+i]=src[3*i+1];
  for(int i=0;i<8;i++)out[64+i]=src[3*i+5];
  const char* mode=std::getenv("GATHER_CORRUPT");
  if(mode&&std::strcmp(mode,"zero")==0)out[63]=0;
  if(mode&&std::strcmp(mode,"nan")==0)out[71]=std::numeric_limits<float>::quiet_NaN();
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "acl").mkdir()
            (root / "acl/acl.h").write_text(header)
            binary = root / "gather_probe"
            subprocess.run(["g++", "-std=c++17", "-I" + str(root),
                            str(ROOT / "src/host/gather_probe.cpp"), "-o", str(binary)], check=True)
            for mode, code in (("", 0), ("zero", 2), ("nan", 2)):
                result = subprocess.run([str(binary)], env=dict(os.environ, GATHER_CORRUPT=mode),
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()

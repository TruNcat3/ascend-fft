#!/usr/bin/env python3
"""共享路径解析：把 `scripts/env.sh` 的探测规则在 Python 里复现一遍。

shell 侧（env.sh）与 python 侧（本模块）读同一组环境变量与同一份缓存
（仓库根 `.ab_py`），所以 `AB_PY=... python3 scripts/xxx.py` 与
`AB_PY=... scripts/xxx.sh` 看到的解释器一致。

  from abenv import root, python_bin, work, cann, msprof
"""
import os
import shutil
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_ROOT, ".ab_py")


def root() -> str:
    """仓库根目录（可用 AB_ROOT 覆盖）。"""
    return os.environ.get("AB_ROOT") or _ROOT


def work() -> str:
    """临时工作区（可用 AB_WORK 覆盖），默认 <root>/.tmp。"""
    p = os.environ.get("AB_WORK") or os.path.join(root(), ".tmp")
    os.makedirs(p, exist_ok=True)
    return p


def cann() -> str:
    """CANN 工具包根（可用 AB_CANN / ASCEND_TOOLKIT_HOME 覆盖）。"""
    return (os.environ.get("AB_CANN")
            or os.environ.get("ASCEND_TOOLKIT_HOME")
            or "/usr/local/Ascend/cann-9.0.0")


def _has_npu_torch(p: str) -> bool:
    if not p:
        return False
    exe = p if os.path.isabs(p) else shutil.which(p)
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-c", "import torch, torch_npu"],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def python_bin() -> str:
    """带 torch + torch_npu 的 python 解释器。

    优先级：AB_PY > PYTHON > 仓库根 `.ab_py` 缓存 > sys.executable >
    PATH 里第一个能 import torch_npu 的 python。
    """
    for key in ("AB_PY", "PYTHON"):
        v = os.environ.get(key)
        if v and (os.path.isabs(v) and os.path.exists(v) or shutil.which(v)):
            return v
    try:
        with open(_CACHE, encoding="utf-8") as f:
            v = f.read().strip()
        if v and os.path.exists(v):
            return v
    except OSError:
        pass
    if _has_npu_torch(sys.executable):
        return sys.executable
    for cand in ("python3", "python3.11", "python3.10", "python3.12"):
        if _has_npu_torch(cand):
            p = shutil.which(cand) or cand
            try:
                with open(_CACHE, "w", encoding="utf-8") as f:
                    f.write(p)
            except OSError:
                pass
            return p
    return sys.executable


def msprof() -> str:
    """msprof 路径；找不到返回 ''。"""
    v = os.environ.get("AB_MSPROF")
    if v:
        return v
    for c in (os.path.join(cann(), "bin", "msprof"),
              os.path.join(cann(), "tools", "profiler", "bin", "msprof")):
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which("msprof") or ""


if __name__ == "__main__":
    print(f"root     = {root()}")
    print(f"work     = {work()}")
    print(f"cann     = {cann()}")
    print(f"python   = {python_bin()}")
    print(f"msprof   = {msprof() or '(未安装)'}")

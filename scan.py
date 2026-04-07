#!/usr/bin/env python3
# scan_project.py
# Run from repo root: python scan_project.py
# Scans every file for issues that could cause failures during judging.

import os, sys, ast, importlib, json, subprocess
from pathlib import Path

PASS  = "✅"
WARN  = "⚠️ "
FAIL  = "❌"
issues = []

def ok(msg):   print(f"  {PASS} {msg}")
def warn(msg): print(f"  {WARN} {msg}");  issues.append(("WARN", msg))
def fail(msg): print(f"  {FAIL} {msg}");  issues.append(("FAIL", msg))

def file_exists(path, critical=True):
    if Path(path).exists():
        ok(f"{path} exists")
        return True
    else:
        (fail if critical else warn)(f"{path} MISSING")
        return False

def contains(path, text, label):
    if Path(path).exists() and text in Path(path).read_text(encoding="utf-8"):
        ok(f"{path} contains {label}")
    else:
        fail(f"{path} missing: {label}")

def not_contains(path, text, label):
    if Path(path).exists() and text in Path(path).read_text(encoding="utf-8"):
        warn(f"{path} contains {label} — may be an issue")
    else:
        ok(f"{path} clean: no {label}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 1. FILE STRUCTURE ══")
# ══════════════════════════════════════════════════════════════════════════════
required_files = [
    "app.py", "environment.py", "models.py",
    "inference.py", "openenv.yaml",
    "Dockerfile", "requirements.txt", "README.md",
    "tasks/__init__.py", "tasks/task_easy.py",
    "tasks/task_medium.py", "tasks/task_hard.py",
    "data/__init__.py", "data/generators.py",
]
for f in required_files:
    file_exists(f)


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 2. PYTHON SYNTAX CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
py_files = [
    "app.py", "environment.py", "models.py", "inference.py",
    "tasks/task_easy.py", "tasks/task_medium.py", "tasks/task_hard.py",
    "data/generators.py",
]
for f in py_files:
    if not Path(f).exists():
        continue
    try:
        ast.parse(Path(f).read_text(encoding="utf-8"))
        ok(f"{f} — syntax OK")
    except SyntaxError as e:
        fail(f"{f} — SyntaxError: {e}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 3. IMPORTS CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
required_packages = [
    "fastapi", "uvicorn", "pydantic",
    "sklearn", "numpy", "openai", "requests"
]
for pkg in required_packages:
    try:
        importlib.import_module(pkg)
        ok(f"import {pkg}")
    except ImportError:
        fail(f"import {pkg} — NOT INSTALLED")

# torch separately (may be cpu variant)
try:
    import torch
    ok(f"import torch ({torch.__version__})")
except ImportError:
    fail("import torch — NOT INSTALLED")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 4. GENERATORS CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("data/generators.py").exists():
    gen_src = Path("data/generators.py").read_text(encoding="utf-8")
    for fn in ["generate_easy_task_data", "generate_medium_task_data",
               "generate_distribution_shift_data", "get_data_summary"]:
        if f"def {fn}" in gen_src:
            ok(f"data/generators.py defines {fn}()")
        else:
            fail(f"data/generators.py missing def {fn}()")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 5. MODELS.PY CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("models.py").exists():
    src = Path("models.py").read_text(encoding="utf-8")
    for cls in ["Action", "Observation"]:
        if f"class {cls}" in src:
            ok(f"models.py defines {cls}")
        else:
            fail(f"models.py missing class {cls}")
    for field in ["action_type", "parameter", "reasoning"]:
        if field in src:
            ok(f"Action has field: {field}")
        else:
            fail(f"Action missing field: {field}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 6. APP.PY CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("app.py").exists():
    src = Path("app.py").read_text(encoding="utf-8")
    contains("app.py", "7860",          "port 7860")
    contains("app.py", "/reset",        "POST /reset")
    contains("app.py", "/step",         "POST /step")
    contains("app.py", "/state",        "GET /state")
    contains("app.py", "np_safe",       "numpy serializer")
    contains("app.py", "CORSMiddleware","CORS middleware")
    not_contains("app.py", "on_event",  "deprecated @on_event")
    not_contains("app.py", "workers=4", "multiple workers (breaks state)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 7. ENVIRONMENT.PY CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("environment.py").exists():
    src = Path("environment.py").read_text(encoding="utf-8")
    contains("environment.py", "MAX_STEPS",       "MAX_STEPS constant")
    contains("environment.py", "_terminal_obs",   "_terminal_obs() method")
    contains("environment.py", "_wrap_observation","_wrap_observation()")
    not_contains("environment.py", "self._task.MAX_STEPS",
                                              "task.MAX_STEPS reference (removed)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 8. TASK FILES CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
for fname, task_id in [
    ("tasks/task_easy.py",   "easy"),
    ("tasks/task_medium.py", "medium"),
    ("tasks/task_hard.py",   "hard"),
]:
    if not Path(fname).exists():
        continue
    src = Path(fname).read_text(encoding="utf-8")
    for method in ["def reset", "def step", "def grade"]:
        if method in src:
            ok(f"{fname}: {method}()")
        else:
            fail(f"{fname}: missing {method}()")
    if f'"{task_id}"' in src or f"'{task_id}'" in src:
        ok(f"{fname}: TASK_ID == {task_id}")
    else:
        warn(f"{fname}: TASK_ID not found")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 9. TASK_MEDIUM FIX CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("tasks/task_medium.py").exists():
    src = Path("tasks/task_medium.py").read_text(encoding="utf-8")
    # Check X_test normalization fix
    if "self.X_test = self._scaler.transform(self.X_test)" in src:
        ok("task_medium.py: X_test normalized in fix_normalization ✓")
    else:
        fail("task_medium.py: X_test NOT normalized in fix_normalization — grade won't improve!")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 10. REQUIREMENTS.TXT CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("requirements.txt").exists():
    req = Path("requirements.txt").read_text(encoding="utf-8").lower()
    for pkg in ["fastapi", "uvicorn", "pydantic", "scikit-learn",
                "numpy", "torch", "openai", "requests"]:
        if pkg in req:
            ok(f"requirements.txt: {pkg}")
        else:
            fail(f"requirements.txt: missing {pkg}")
    if "extra-index-url" in req and "cpu" in req:
        ok("requirements.txt: CPU-only torch (fast Docker build)")
    else:
        warn("requirements.txt: no CPU torch pin — Docker build may be slow/large")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 11. DOCKERFILE CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("Dockerfile").exists():
    src = Path("Dockerfile").read_text(encoding="utf-8")
    contains("Dockerfile", "7860",             "EXPOSE 7860")
    contains("Dockerfile", "requirements.txt", "pip install -r requirements.txt")
    contains("Dockerfile", "app.py",           "app.py referenced")
    not_contains("Dockerfile", "COPY . .",     "broad COPY . . (prefer explicit)")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 12. OPENENV.YAML CHECK ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("openenv.yaml").exists():
    src = Path("openenv.yaml").read_text(encoding="utf-8")
    for field in ["name", "description", "tasks", "port"]:
        if field in src:
            ok(f"openenv.yaml: {field} present")
        else:
            fail(f"openenv.yaml: missing {field}")
    if "7860" in src:
        ok("openenv.yaml: port 7860")
    else:
        warn("openenv.yaml: port not 7860")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 13. INFERENCE.PY STDOUT FORMAT ══")
# ══════════════════════════════════════════════════════════════════════════════
if Path("inference.py").exists():
    src = Path("inference.py").read_text(encoding="utf-8")
    for marker in ["[START]", "[STEP]", "[END]"]:
        if marker in src:
            ok(f"inference.py: emits {marker}")
        else:
            fail(f"inference.py: missing {marker} output")
    for env_var in ["HF_TOKEN", "ENV_URL"]:
        if env_var in src:
            ok(f"inference.py: reads {env_var}")
        else:
            warn(f"inference.py: {env_var} not referenced")
    not_contains("inference.py", "sk-", "hardcoded API key")
    not_contains("inference.py", "hf_", "hardcoded HF token")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 14. SECRET SCAN ══")
# ══════════════════════════════════════════════════════════════════════════════
secret_patterns = ["sk-", "hf_", "Bearer ", "password=", "secret="]
scan_exts = [".py", ".yaml", ".yml", ".txt", ".md", ".env"]
found_secrets = False
for ext in scan_exts:
    for fpath in Path(".").rglob(f"*{ext}"):
        if ".git" in str(fpath):
            continue
        content = fpath.read_text(encoding="utf-8", errors="ignore")
        for pat in secret_patterns:
            if pat in content:
                # Check it's not just a variable name reference
                if pat == "hf_" and "HF_TOKEN" in content and content.count("hf_") <= 2:
                    continue
                warn(f"Possible secret in {fpath}: pattern '{pat}' found")
                found_secrets = True
if not found_secrets:
    ok("No hardcoded secrets found")


# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 15. GIT STATUS ══")
# ══════════════════════════════════════════════════════════════════════════════
try:
    result = subprocess.run(["git", "status", "--short"],
                            capture_output=True, text=True, timeout=10)
    untracked = [l for l in result.stdout.strip().split("\n") if l.startswith("??")]
    modified  = [l for l in result.stdout.strip().split("\n") if l.startswith(" M") or l.startswith("M ")]
    if modified:
        warn(f"Uncommitted changes: {[l.split()[-1] for l in modified]}")
    else:
        ok("No uncommitted changes")
    if untracked:
        warn(f"Untracked files: {[l.split()[-1] for l in untracked]}")
    else:
        ok("No untracked files")
except Exception as e:
    warn(f"Could not run git status: {e}")


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
fails = [m for t, m in issues if t == "FAIL"]
warns = [m for t, m in issues if t == "WARN"]
if not fails and not warns:
    print(f"  ✅  ALL CHECKS PASSED — ready to submit!")
elif not fails:
    print(f"  ⚠️   {len(warns)} warnings (no blockers). Review before submitting.")
else:
    print(f"  ❌  {len(fails)} FAILURES  |  {len(warns)} warnings")
    print("\n  Fix these before submitting:")
    for m in fails:
        print(f"    ❌ {m}")
print(f"{'='*60}\n")

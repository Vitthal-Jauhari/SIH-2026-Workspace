"""
Wrapper / root copy for test_model_offline.py
"""
import os
import sys

target = os.path.join(os.path.dirname(__file__), "vaani-wakeword", "test_model_offline.py")
if os.path.isfile(target):
    with open(target, "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
else:
    sys.exit(f"Target not found: {target}")

import os
import sys

# trick PATH for test discovery
if 'unittest' in sys.modules or 'pytest' in sys.modules:
    ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    SRC_PATH = os.path.join(ROOT_PATH, "src")
    if SRC_PATH not in sys.path:
        print(f"add src: {SRC_PATH}")
        sys.path.insert(0, SRC_PATH)

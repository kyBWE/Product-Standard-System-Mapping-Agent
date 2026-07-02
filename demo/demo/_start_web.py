from __future__ import annotations
import sys
import os

os.chdir("E:/Code/Projects/demo")
sys.path.insert(0, ".")

from src.web.app import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
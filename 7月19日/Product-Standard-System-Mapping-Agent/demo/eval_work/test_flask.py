import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from flask import Flask
app = Flask(__name__)

@app.route("/test")
def t():
    return "ok"

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    print("Starting via run_simple...", flush=True)
    run_simple("127.0.0.1", 5001, app, use_reloader=False, use_debugger=False)

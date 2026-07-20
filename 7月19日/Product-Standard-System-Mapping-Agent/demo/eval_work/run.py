import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '.')
from src.web.app import app
app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
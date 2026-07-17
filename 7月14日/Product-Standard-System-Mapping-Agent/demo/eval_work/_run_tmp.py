import sys, os
os.chdir(r'C:/Users/11523/IDEProjects/Product-Standard-System-Mapping-Agent/eval_work')
sys.path.insert(0, '.')
try:
    from src.web.app import app
    print('APP_LOADED_OK')
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
except Exception as e:
    print(f'STARTUP_ERROR: {e}')
    import traceback
    traceback.print_exc()

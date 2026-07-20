import time
from openai import OpenAI
c = OpenAI(base_url="https://apihub.agnes-ai.com/v1", api_key="sk-NeZZ35h3SyD6p9GmnOSOArW7Yz7uhuL3x9ceFFPHQ6TtaQb3", timeout=15)
for i in range(10):
    t0 = time.time()
    try:
        r = c.chat.completions.create(model="agnes-2.0-flash", messages=[{"role":"user","content":f"test {i}"}], temperature=0.1)
        elapsed = time.time() - t0
        print(f"{i}: OK {elapsed:.1f}s")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{i}: FAIL {elapsed:.1f}s {e}")
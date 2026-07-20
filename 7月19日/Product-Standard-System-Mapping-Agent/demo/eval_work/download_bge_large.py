"""下载 bge-large-zh-v1.5 的 ONNX 模型文件。"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

MODEL_DIR = "models/bge-large-zh-v1.5"
os.makedirs(MODEL_DIR, exist_ok=True)

print("正在下载 bge-large-zh-v1.5 ONNX 模型 (hf-mirror)...")
snapshot_download(
    repo_id="BAAI/bge-large-zh-v1.5",
    allow_patterns=["onnx/**", "tokenizer.json", "tokenizer_config.json",
                    "vocab.txt", "special_tokens_map.json", "config.json"],
    local_dir=MODEL_DIR,
)
print(f"下载完成: {MODEL_DIR}")

for root, dirs, files in os.walk(MODEL_DIR):
    for f in files:
        fp = os.path.join(root, f)
        size_mb = os.path.getsize(fp) / 1024 / 1024
        print(f"  {fp}: {size_mb:.1f} MB")
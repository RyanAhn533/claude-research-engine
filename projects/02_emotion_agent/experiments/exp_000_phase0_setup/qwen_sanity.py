"""
Phase 0 Gate 2-3: Qwen2.5-7B 4-bit load + 1 prompt inference sanity.

Constraints:
  - VRAM budget <= 7.7 GB
  - PYTHONNOUSERSITE=1 (run with this env var)
"""
import os, sys, time, json, torch
from pathlib import Path

# Ensure user-site is isolated
assert os.environ.get("PYTHONNOUSERSITE") == "1", "Run with PYTHONNOUSERSITE=1"

CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = CACHE
os.environ["TRANSFORMERS_CACHE"] = CACHE

OUT = Path(__file__).parent / "logs"
OUT.mkdir(exist_ok=True, parents=True)

result = {
    "torch_version": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_free_mb_before": None,
    "gpu_used_mb_after": None,
    "load_time_sec": None,
    "infer_time_sec": None,
    "response_sample": None,
    "status": "start",
}

if not torch.cuda.is_available():
    result["status"] = "FAIL: no CUDA"
    with open(OUT / "qwen_sanity.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    sys.exit(1)

free_before, _ = torch.cuda.mem_get_info(0)
result["gpu_free_mb_before"] = free_before // (1024 * 1024)
print(f"[info] free GPU before: {result['gpu_free_mb_before']} MB")

# === Load Qwen2.5-7B 4-bit ===
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Use direct snapshot path (local, no network)
import glob
snap_dirs = glob.glob(f"{CACHE}/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*")
assert snap_dirs, "Qwen2.5-7B snapshot not found in cache"
MODEL_PATH = snap_dirs[0]
print(f"[info] loading from {MODEL_PATH} in 4-bit...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb,
    device_map={"": 0},
    torch_dtype=torch.float16,
)
load_time = time.time() - t0
result["load_time_sec"] = round(load_time, 2)
used_mb = (free_before - torch.cuda.mem_get_info(0)[0]) // (1024 * 1024)
result["gpu_used_mb_after"] = used_mb
print(f"[info] loaded in {load_time:.1f}s. GPU used: {used_mb} MB")

# === Inference sanity ===
prompt = (
    "You are an emotion recognition assistant. "
    "Classify this utterance into one of {happy, sad, angry, neutral}: "
    "'I can't believe he said that. It really crushed me.'\n"
    "Respond with only the label."
)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

t0 = time.time()
with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=16,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
infer_time = time.time() - t0
response = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
result["infer_time_sec"] = round(infer_time, 2)
result["response_sample"] = response
print(f"[info] infer {infer_time:.2f}s -> '{response}'")

# Check VRAM budget
VRAM_BUDGET_MB = 7700
if used_mb > VRAM_BUDGET_MB:
    result["status"] = f"WARN: used {used_mb}MB > budget {VRAM_BUDGET_MB}MB"
else:
    result["status"] = "PASS"

with open(OUT / "qwen_sanity.json", "w") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"[info] status: {result['status']}")

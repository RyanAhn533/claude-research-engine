"""
Emotion Agent — Qwen2.5-7B 4-bit based, cultural-prior-aware.

Loop:
  1) Perception: transcription (+ optional prosody/visual features if available)
  2) Cultural prior injection: Jack 2012 context + Yonsei consensus hint
  3) LLM reasoning (Qwen2.5-7B 4-bit)
  4) Output: emotion label + reasoning text + confidence

Usage:
  from emotion_agent import EmotionAgent
  agent = EmotionAgent()
  result = agent.predict(text="I can't believe he said that.", prosody={"pitch_mean": 150})
"""
import os, glob, json
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Force local-only cache
HF_CACHE = "/mnt/hdd/ajy/caches/huggingface"
os.environ["HF_HOME"] = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE


LABEL_4 = ["angry", "happy", "neutral", "sad"]


CULTURAL_PRIOR_KR = """Cultural context (Korean subjects, based on Jack et al. 2012 PNAS and Yonsei 298-person consensus study):
- East Asian observers tend to encode emotional intensity in the EYE region more than Western subjects.
- However, large-scale Korean data shows MOUTH region carries the strongest discriminative signal at the AI perception level
  (mouth > nose > cheek > eyes > forehead in linear probe accuracy).
- Koreans often suppress explicit emotional expression; subtle cues matter more than overt ones.
- Annotator disagreement is highest for negative emotions (sad/angry ~50% agreement vs happy ~96%)."""


def build_prompt(text: str, prosody: Optional[Dict[str, float]] = None,
                 use_cultural_prior: bool = True) -> List[Dict[str, str]]:
    """Build chat messages with optional cultural prior injection."""
    sys_prompt = (
        "You are a culturally-aware multimodal emotion recognition agent. "
        "Given user utterance (and optional acoustic features), "
        "classify emotion into one of: angry, happy, neutral, sad. "
        "Briefly explain your reasoning in one sentence."
    )
    if use_cultural_prior:
        sys_prompt += "\n\n" + CULTURAL_PRIOR_KR

    user_parts = [f'Utterance: "{text}"']
    if prosody:
        prosody_str = ", ".join(f"{k}={v:.2f}" for k, v in prosody.items())
        user_parts.append(f"Acoustic features: {prosody_str}")
    user_parts.append("\nOutput format:\nLABEL: <angry|happy|neutral|sad>\nREASON: <one sentence>")
    user_msg = "\n".join(user_parts)

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": user_msg},
    ]


def parse_output(raw: str):
    """Extract label and reason from LLM output."""
    label = None
    reason = ""
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("LABEL:"):
            lab = line.split(":", 1)[1].strip().lower()
            for l in LABEL_4:
                if l in lab:
                    label = l
                    break
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    # fallback keyword scan
    if label is None:
        low = raw.lower()
        for l in LABEL_4:
            if l in low:
                label = l
                break
    return label or "neutral", reason


class EmotionAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct",
                 use_cultural_prior: bool = True):
        snap_dirs = glob.glob(f"{HF_CACHE}/hub/models--{model_name.replace('/', '--')}/snapshots/*")
        assert snap_dirs, f"snapshot not found for {model_name}"
        self.model_path = snap_dirs[0]
        self.use_cultural_prior = use_cultural_prior

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print(f"[agent] loading {model_name} from {self.model_path} (4-bit)")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=bnb,
            device_map={"": 0},
            torch_dtype=torch.float16,
        )
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str,
                prosody: Optional[Dict[str, float]] = None,
                max_new_tokens: int = 64) -> Dict[str, Any]:
        messages = build_prompt(text, prosody, self.use_cultural_prior)
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True
        )
        label, reason = parse_output(raw)
        return {"label": label, "reason": reason, "raw": raw}


if __name__ == "__main__":
    agent = EmotionAgent()
    for text, prosody in [
        ("I can't believe he said that. It really crushed me.", {"pitch_mean": 120, "speaking_rate": 3.2}),
        ("This is the best day of my life!",                    {"pitch_mean": 260, "speaking_rate": 5.0}),
        ("Yeah, I don't care anymore.",                         {"pitch_mean": 90, "speaking_rate": 2.0}),
        ("Please explain that once more.",                      {"pitch_mean": 140, "speaking_rate": 3.5}),
    ]:
        r = agent.predict(text, prosody)
        print(f"\ntext: {text}")
        print(f"  prosody: {prosody}")
        print(f"  → label: {r['label']}")
        print(f"  → reason: {r['reason']}")

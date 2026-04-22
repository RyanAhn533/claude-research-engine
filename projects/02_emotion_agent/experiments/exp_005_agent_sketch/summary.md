# exp_005 — Agent Prototype Sketch

## Architecture
```
User utterance (+ optional prosody) → build_prompt
  → system prompt: cultural prior (Jack 2012 + Yonsei consensus)
  → chat template
  → Qwen2.5-7B 4-bit inference
  → parse_output → (label, reason, raw)
```

## Code
- `src/agent/emotion_agent.py` — `EmotionAgent` class
- `__main__`: 4 sample sanity

## Sanity test results

| Utterance | Prosody | Predicted | Reason |
|-----------|---------|-----------|--------|
| "I can't believe he said that. It really crushed me." | pitch=120 | **sad** | disappointment + emotional pain |
| "This is the best day of my life!" | pitch=260 | **happy** | enthusiastic tone + high pitch ✓ |
| "Yeah, I don't care anymore." | pitch=90 | **sad** | lack of interest |
| "Please explain that once more." | pitch=140 | **neutral** | neutral clarification |

## Key finding
**Prosody feature가 reasoning text에 명시적 인용**됨 (case 2 "high pitch indicate strong positive emotion"). Cultural prior injection + prosody 조합이 LLM에서 actively 사용 중.

## Next
- Week 3: IEMOCAP/MELD **test set 전체 inference** → baseline vs +cultural vs +prosody ablation
- Week 3: VRAM 체크하며 batch 돌리기 (Qwen 4-bit 5.4GB → batch 1-4 적절)

# Kill Switch

자동화 iteration 루프를 즉시 중단:

```bash
touch scripts/kill_switch
```

존재하면 Claude가 다음 iter phase 시작 전 감지해 중단하고 상태 보고.

재개:
```bash
rm scripts/kill_switch
```

"""SR-XMA E1 학습前 교차검증 (PLAN_SRXMA §D 1~5단계, GPU 불필요).

검사:
  1. import/AST          — SRXMAFusion 로드
  2. forward 4모드        — full/fixed/random/learned, 출력 shape + edges/node
  3. grad 유한·nonzero    — 라우터(Wq,Wk,edge_bias) 미분 정상 (NaN 검출)
  4. ★함정 실측          — per-sample k vs 'batched 실제 계산 edge'(union) vs B=1 edge
  5. 이론 FLOPs 비율      — learned/full (edge별 T_i*T_j 합)

실행: python projects/03_ht_dar/tools/srxma_preflight.py
"""
import sys, itertools
sys.path.insert(0, "/home/ajy/HT-DAR/code")
import torch

D, HEADS, LEVELS, B, T = 50, 10, 3, 16, 50   # MOSEI: dst=50, heads=10, batch=16
MODES = ['full', 'fixed', 'random', 'learned']
torch.manual_seed(0)

def banner(s): print(f"\n{'='*60}\n{s}\n{'='*60}")

# ---- 1. import ----
banner("1. import / AST")
from trains.singleTask.model.sr_xma import SRXMAFusion
print("[ok] SRXMAFusion import")

def make(mode, k=2):
    m = SRXMAFusion(d=D, num_heads=HEADS, num_levels=LEVELS, num_modalities=3,
                    route_mode=mode, topk=k)
    return m

def inputs(b=B):
    g = torch.Generator().manual_seed(1)
    return [torch.randn(T, b, D, generator=g) for _ in range(3)]

# ---- 2. forward 4 modes ----
banner("2. forward 4모드 — shape + edges/node")
s_l, s_v, s_a = inputs()
for mode in MODES:
    m = make(mode); m.eval()
    with torch.no_grad():
        hl, hv, ha, aux = m(s_l, s_v, s_a)
    shapes = (tuple(hl.shape), tuple(hv.shape), tuple(ha.shape))
    ok = all(s == (B, D) for s in shapes)
    print(f"[{'ok ' if ok else 'FAIL'}] {mode:8s} out={shapes} edges/node(per-sample mean)={aux['edges_per_node']:.3f}")

# ---- 3. grad finite + nonzero ----
banner("3. grad 유한·nonzero (learned 라우터)")
m = make('learned'); m.train()
hl, hv, ha, aux = m(s_l, s_v, s_a)
loss = (hl.pow(2).mean() + hv.pow(2).mean() + ha.pow(2).mean())
loss.backward()
for name in ['Wq.weight', 'Wk.weight', 'edge_bias']:
    p = dict(m.named_parameters())[name]
    g = p.grad
    finite = bool(torch.isfinite(g).all()) if g is not None else False
    nonzero = bool((g.abs() > 0).any()) if g is not None else False
    print(f"[{'ok ' if (finite and nonzero) else 'FAIL'}] {name:14s} grad finite={finite} nonzero={nonzero}")

# ---- 4. ★ 함정 실측: per-sample k vs batched 실제 계산 edge vs B=1 ----
banner("4. ★FLOPs 함정 — per-sample k vs batched-union(실제 계산) vs B=1")
def computed_edges(model, sl, sv, sa):
    """현재 V1 forward가 실제로 xattn을 호출하는 edge 수 = (g>0).any(batch) 인 (i,j) 개수."""
    model.eval()
    with torch.no_grad():
        Z = []
        mods = [sl, sv, sa]
        for mm in range(model.M):
            for tok in model.hier[mm](mods[mm]):
                Z.append(tok)
        h = torch.stack([z.mean(0) for z in Z], dim=1)
        g, sel = model._route(h)
        union = (g > 0).any(0)                 # (N,N) — 배치 통틀어 한 샘플이라도 고른 edge
        per_sample = sel.float().sum(-1).mean().item()
        return int(union.sum().item()), per_sample, model.N
for mode in ['random', 'learned', 'full']:
    m = make(mode)
    u_b, ps_b, N = computed_edges(m, s_l, s_v, s_a)        # batched B=16
    s1l, s1v, s1a = inputs(b=1)
    u_1, ps_1, _ = computed_edges(m, s1l, s1v, s1a)        # B=1
    maxedges = N * (N - 1)
    print(f"[{mode:8s}] per-sample k≈{ps_b:.2f}  | batched 실제계산 edge={u_b}/{maxedges}  | B=1 실제계산 edge={u_1}/{maxedges}")

# ---- 5. 이론 FLOPs 비율 (edge별 T_i*T_j) ----
banner("5. 이론 FLOPs (edge별 T_i*T_j 합) — learned/full 비율")
def tlen(level): return T // (2 ** level)   # patch-merge stride2 근사
node_T = [tlen(l) for m_ in range(3) for l in range(LEVELS)]   # 9 노드 토큰길이
def edge_cost(union_mat):
    c = 0
    for i in range(len(node_T)):
        for j in range(len(node_T)):
            if union_mat[i, j]: c += node_T[i] * node_T[j]
    return c
m_full = make('full'); m_learn = make('learned')
with torch.no_grad():
    def umat(model, b):
        sl, sv, sa = inputs(b=b); model.eval()
        Z = []; mods=[sl,sv,sa]
        for mm in range(model.M):
            for tok in model.hier[mm](mods[mm]): Z.append(tok)
        h = torch.stack([z.mean(0) for z in Z], dim=1)
        g, _ = model._route(h)
        return (g > 0).any(0)
    full_b1 = edge_cost(umat(m_full, 1))
    learn_b1 = edge_cost(umat(m_learn, 1))
    learn_b16 = edge_cost(umat(m_learn, 16))
print(f"  full(B=1)   theoretical xattn-cost = {full_b1}")
print(f"  learned(B=1) = {learn_b1}  → ratio {learn_b1/full_b1:.2%} of full  (← 진짜 sparsity)")
print(f"  learned(B=16, batched union) = {learn_b16}  → ratio {learn_b16/full_b1:.2%}  (← batched는 절감 사라짐)")
print("\n결론: V1은 per-sample top-k여서 효율은 B=1 inference에서만 실재. batched 학습/평가는 masked-full.")
print("E1 정확도 비교(learned vs random vs full)는 유효. 효율 주장은 B=1 이론-FLOPs로만 방어 가능.")

{
  "summary": "For each HT-DAR component idea, find existing GitHub repos / public implementations to fork instead of writing from scratch",
  "agentCount": 6,
  "logs": [],
  "result": {
    "synth": "# Multimodal Emotion Recognition — Repo Base Plan\n\n## Per-component best repo\n\n### 1. Backbone to fork (MOSI/MOSEI fusion)\n**pwang322/DLF** — https://github.com/pwang322/DLF (AAAI 2025, MIT, PyTorch, maintained).\nNewest SOTA-level baseline with clean shared/specific branches + Language-Focused Attractor = natural insertion points for a new module. **ALMT** (Haoyu-ha/ALMT, EMNLP 2023) is the battle-tested fallback. Use **thuiar/MMSA** (MIT) only as the eval/baseline harness, not the backbone.\n\n### 2. Graph-fusion code to reuse\n**Exploration-Lab/COGMEN** — https://github.com/Exploration-Lab/COGMEN (NAACL 2022).\nCleanest modern GNN-ERC on standard PyG layers (RGCNConv + TransformerConv), covers IEMOCAP + MOSEI, easiest to extend. **Caveat: GPL-3.0 (copyleft).** If you need permissive, fork **zerohd4869/MM-DFN** (MIT, ICASSP-lineage) instead. Keep **MMGCN** as must-cite baseline.\n\n### 3. Dynamic-routing / MoE code\n**MKMaS-GUET/KuDA** — https://github.com/MKMaS-GUET/KuDA (EMNLP 2024 Findings, MIT, PyTorch).\nOnly repo where per-sample dynamic modality routing IS the contribution, runs on MOSI/MOSEI/CH-SIMS via the MMSA feature pipeline. Donate the gate from **DynMM** (efficiency framing) or **QMF** (uncertainty-as-weight, maps to your QualityAwareGate) as needed.\n\n### 4. Missing-modality robustness code/protocol\n**gw-zhong/CIDer** — https://github.com/gw-zhong/CIDer (2025, MIT) for the benchmark/protocol (5 dropout modes RMFM/RMM/TMFM/STMFM/SMM + OOD splits, weights provided). Pair with **AIM3-RUC/MMIN** (MIT, ACL 2021) for the canonical 6-condition {a,v,l,av,al,vl} baseline reviewers expect. GCNet if you want a continuous missing-rate curve.\n\n### 5. K-EmoCon / hierarchical code\nTwo layers. **Data/protocol: Kaist-ICLab/K-EmoCon_SupplementaryCodes** — https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes (MIT) as the protocol spec only (classical ML, ecg/PyTEAP naming differs from your bvp/eda/temp/hr NPZ). **Model/fusion: ispamm/MHyEEG (H2 hierarchical)** — https://github.com/ispamm/MHyEEG, the only maintained hierarchical fusion repo on EEG+peripheral physio for arousal/valence (ICASSP-lineage). License unstated — verify before redistribution.\n\n## Integration verdict (5 lines)\n1. **Fork DLF** as the backbone and insert your routing module into its shared/specific branches; **fork KuDA** as the routing reference and lift its gate, since a usable dynamic graph-routing repo does NOT exist as one package.\n2. **Fork CIDer** for the missing-modality benchmark/protocol and **MMIN** for the canonical baseline — do not write your own robustness protocol, theirs is the field standard.\n3. **Reuse COGMEN's PyG graph layers** (or MM-DFN if GPL blocks you) — do NOT hand-roll HGAR; a generic graph-routing repo does not exist, but COGMEN's RGCNConv/TransformerConv + KuDA's gate together give you ~80% of HGAR, so you only write the graph-construction-over-routing glue yourself.\n4. **Write ourselves:** the bio/physio graph-construction layer, the NPZ→feature dataset adapter (no repo ingests your layout), and the K-EmoCon-specific encoders — none of the forks handle physiological modalities natively.\n5. **Honest blocker:** every fork depends on gated pre-extracted feature .pkl files (CMU-MultimodalSDK / BaiduYun / HF) — secure those downloads and reproduce reported numbers BEFORE adding your module, or your delta is not credible.\n\n**Net:** No single graph-routing repo exists — combine COGMEN (graph) + KuDA (routing) rather than hand-rolling HGAR from scratch.",
    "findings": [
      {
        "idea": "MOSEI/CMU-MOSI multimodal sentiment/emotion fusion SOTA with public code — selecting the best backbone to fork and insert a new module into.",
        "repos": [
          {
            "name": "thuiar/MMSA",
            "url": "https://github.com/thuiar/MMSA",
            "what": "Unified framework/toolkit for Multimodal Sentiment Analysis. Trains, tests, and compares 15+ MSA models (MulT, Self-MM, TFN, LMF, MISA, MAG-BERT, BBFN, etc.) in one codebase with standardized data loading, train loop, and regression+classification metrics (MAE, Corr, Acc-2/Acc-7, F1).",
            "language_framework": "PyTorch (pip-installable: pip install MMSA; also has MMSA-FET for feature extraction)",
            "license": "MIT",
            "maintained": "Yes — widely used, actively maintained, large issue history; de-facto standard benchmark harness.",
            "stars_approx": "~900+",
            "venue_or_paper": "ACL 2022 demo (Integrated Platform for MSA); aggregates Self-MM AAAI'21, MulT ACL'19, etc.",
            "fork_worthiness": "high",
            "why": "Best base if the new module is model-agnostic: standardized MOSI/MOSEI/SIMS pipeline + many baselines means you get fair comparison tables almost for free, exactly what reviewers at ICASSP/ACII expect. Drop your module into one model class and reuse the whole eval harness. Risk: heavier abstraction, slightly dated default models vs 2025 SOTA."
          },
          {
            "name": "pwang322/DLF",
            "url": "https://github.com/pwang322/DLF",
            "what": "Official AAAI 2025 implementation of Disentangled-Language-Focused MSA. Separates modality-shared vs modality-specific features, adds 4 geometric disentanglement losses + a Language-Focused Attractor (cross-attention where language dominates), with hierarchical pre/post-fusion prediction heads.",
            "language_framework": "PyTorch (1.13.0, CUDA 11.7, Python 3.9)",
            "license": "MIT",
            "maintained": "Moderate — recent (2025), 21 commits, pretrained models + train/test scripts provided; small but functional.",
            "stars_approx": "~140",
            "venue_or_paper": "AAAI 2025 (arXiv:2412.12225)",
            "fork_worthiness": "high",
            "why": "Newest SOTA-level backbone with clean disentanglement structure that is a natural insertion point for a new fusion/representation module (you can plug into the shared/specific branches or the attractor). Being 2025 AAAI, basing on it gives a strong, current baseline. Caveat: smaller community, README lacks inline result tables (numbers are in the paper), so verify reproduction before committing."
          },
          {
            "name": "Haoyu-ha/ALMT",
            "url": "https://github.com/Haoyu-ha/ALMT",
            "what": "Official EMNLP 2023 implementation. Adaptive Hyper-modality Learning (AHL) uses multi-scale language features to guide audio/visual into a conflict/redundancy-suppressed hyper-modality, then a language-query cross-modal fusion transformer. SOTA on MOSI/MOSEI/CH-SIMS at publication.",
            "language_framework": "PyTorch 2.5.1, CUDA 12.1, Python 3.11",
            "license": "MIT",
            "maintained": "Yes — updated March 2025 to fix demo issues; active.",
            "stars_approx": "~150",
            "venue_or_paper": "EMNLP 2023 (arXiv:2310.05804)",
            "fork_worthiness": "high",
            "why": "Clean, modern, well-maintained single-model repo with a clear modular structure (AHL + fusion transformer) that is easy to extend. Honest about metric-specific vs same-epoch reporting, which makes reproduction transparent. Slightly older than DLF but more battle-tested. Strong second-choice backbone."
          },
          {
            "name": "thuiar/Self-MM",
            "url": "https://github.com/thuiar/Self-MM",
            "what": "Official AAAI 2021 Self-Supervised Multi-Task Learning for MSA — generates unimodal pseudo-labels to learn modality-specific representations alongside the multimodal task. Long-standing strong baseline on MOSI/MOSEI/SIMS.",
            "language_framework": "PyTorch",
            "license": "MIT",
            "maintained": "Low/stable — superseded by being folded into MMSA, but standalone repo still works.",
            "stars_approx": "~300",
            "venue_or_paper": "AAAI 2021",
            "fork_worthiness": "medium",
            "why": "Excellent, simple, reliable baseline and very commonly cited; good for a clean comparison or a lightweight backbone. Not SOTA in 2025, and you'd get the same model inside MMSA with better tooling, so prefer MMSA over forking this directly."
          },
          {
            "name": "yaohungt/Multimodal-Transformer",
            "url": "https://github.com/yaohungt/Multimodal-Transformer",
            "what": "Original MulT (ACL 2019) — directional pairwise crossmodal transformers fusing unaligned multimodal time series. The seminal cross-modal attention backbone for MOSI/MOSEI/IEMOCAP.",
            "language_framework": "PyTorch (older: Python 3.6/3.7, PyTorch >=1.0, CUDA 10)",
            "license": "Not clearly specified / no explicit LICENSE file",
            "maintained": "No — effectively unmaintained, old dependencies, known reproduction/OOM issues in tracker.",
            "stars_approx": "~800+",
            "venue_or_paper": "ACL 2019",
            "fork_worthiness": "low",
            "why": "Historically important and a required citation/baseline, but stale dependencies, unclear license, and reproduction complaints make it a poor base to build on. Use the MulT implementation inside MMSA instead if you need it."
          }
        ],
        "recommendation": "Fork TWO things, with distinct roles. (1) Primary backbone to insert your module into: pwang322/DLF (AAAI 2025) OR Haoyu-ha/ALMT (EMNLP 2023). Both are MIT, PyTorch, actively maintained, language-guided fusion designs with clean, modular insertion points (DLF's shared/specific branches + Language-Focused Attractor; ALMT's AHL module). Pick DLF if you want the most current SOTA baseline and your contribution targets disentangled/shared-specific representations; pick ALMT if you want a more battle-tested repo and your module targets the language-guided hyper-modality fusion. (2) Evaluation/comparison harness: base your benchmark tables on thuiar/MMSA so you get standardized MOSI/MOSEI splits, identical metrics (Acc-2/Acc-7, F1, MAE, Corr), and free strong baselines (MulT, Self-MM, MISA, MAG-BERT) — this is what ICASSP/ACII reviewers expect for fair comparison. Concrete plan: develop your module on DLF or ALMT, then re-run/cite MMSA numbers for the baseline rows. Caveats: (a) all repos rely on the same pre-extracted MOSI/MOSEI feature .pkl files (BERT text, COVAREP/wav2vec audio, FACET/OpenFace visual) gated via Google Drive / CMU MultimodalSDK — confirm you can download these first, as that is the usual blocker. (b) Reproduce the reported numbers before adding your module so your delta is credible. (c) Avoid forking yaohungt/MulT directly (stale deps, unclear license); use the MMSA copy instead."
      },
      {
        "idea": "Graph neural network (GNN) based multimodal fusion for emotion/sentiment — graph-over-modalities, message passing fusion, modality interaction graphs (MMGCN, Graph-MFN, GraphCAGE, COGMEN and successors), basing on existing public code.",
        "repos": [
          {
            "name": "Exploration-Lab/COGMEN",
            "url": "https://github.com/Exploration-Lab/COGMEN",
            "what": "Official NAACL 2022 implementation. Contextualized GNN (RGCNConv + TransformerConv from PyTorch Geometric) for multimodal (text+audio+video) emotion recognition in conversation. Models intra/inter-speaker dependencies as a graph over utterances. SOTA on IEMOCAP (4-way & 6-way) and CMU-MOSEI.",
            "why": "Cleanest, most readable modern GNN-ERC codebase using standard PyG layers. Pre-extracted features provided; supports both IEMOCAP and MOSEI (covers both emotion and sentiment benchmarks). Easy to swap in new modality encoders or graph construction — ideal base for an ICASSP/ACII contribution. GPL-3.0 is the main caveat (copyleft).",
            "language_framework": "Python / PyTorch + PyTorch Geometric (RGCNConv, TransformerConv)",
            "license": "GPL-3.0",
            "maintained": "Low — ~9 commits, no recent activity, 11 open issues. Code is stable/functional but not actively developed.",
            "stars_approx": "~100",
            "venue_or_paper": "NAACL 2022, arXiv:2205.02455",
            "fork_worthiness": "high"
          },
          {
            "name": "AIM3-RUC/MMGCN",
            "url": "https://github.com/AIM3-RUC/MMGCN",
            "what": "Official ACL 2021 implementation. Multimodal fused graph convolution network for ERC. Builds a single graph connecting all modalities of all utterances, uses speaker embeddings for inter/intra-speaker dependency. Benchmarks on IEMOCAP and MELD.",
            "why": "The canonical/most-cited GNN multimodal-fusion baseline — almost every later paper (MM-DFN, GraphMFT, M3Net) compares against it and reuses its feature pipeline + graph construction. Forking it gives you a recognized baseline and the de-facto feature format the whole subfield uses. Older PyG, less polished than COGMEN.",
            "language_framework": "Python / PyTorch + PyTorch Geometric",
            "license": "No explicit LICENSE file (treat as all-rights-reserved; check before redistribution)",
            "maintained": "Low — research dump, no ongoing maintenance.",
            "stars_approx": "~300",
            "venue_or_paper": "ACL 2021, arXiv:2107.06779",
            "fork_worthiness": "high"
          },
          {
            "name": "zerohd4869/MM-DFN",
            "url": "https://github.com/zerohd4869/MM-DFN",
            "what": "Official ICASSP 2022 code. Multimodal Dynamic Fusion Network: graph-based dynamic fusion module that reduces cross-modal redundancy across stacked GCN layers. Reports IEMOCAP 68.21% acc / 66.54% F1, MELD 62.49% acc / 36.28% F1.",
            "why": "MIT license (clean to reuse/extend), ICASSP venue directly matches the target, and it improves on MMGCN. Same IEMOCAP/MELD feature pipeline as MMGCN so easy to mix-and-match. Good base if you specifically want an ICASSP-lineage starting point with permissive license.",
            "language_framework": "Python / PyTorch + PyTorch Geometric (torch 1.4, pyg 1.4 — old)",
            "license": "MIT",
            "maintained": "Low — ~21 commits, archived feel, but reproducible.",
            "stars_approx": "~95",
            "venue_or_paper": "ICASSP 2022, arXiv:2203.02385",
            "fork_worthiness": "high"
          },
          {
            "name": "feiyuchen7/M3NET",
            "url": "https://github.com/feiyuchen7/M3NET",
            "what": "Official CVPR 2023 code. 'Multivariate, Multi-frequency and Multimodal' — rethinks GNNs for ERC using multi-frequency graph signals (beyond low-pass GCN) plus hypergraph/multivariate modeling. IEMOCAP and MELD.",
            "why": "Most recent strong-venue (CVPR 2023) GNN-ERC method and a likely SOTA baseline you must beat/compare. Good for ideas (multi-frequency message passing). Caveat: no explicit license, results noted to vary by seed/machine, minimal maintenance — riskier as a direct fork than as a reference/baseline.",
            "language_framework": "Python / PyTorch + PyTorch Geometric 1.7.2",
            "license": "Not specified (no LICENSE file)",
            "maintained": "Low — ~15 commits, minimal.",
            "stars_approx": "~60",
            "venue_or_paper": "CVPR 2023",
            "fork_worthiness": "medium"
          },
          {
            "name": "lijfrank-open/GraphMFT",
            "url": "https://github.com/lijfrank-open/GraphMFT",
            "what": "Official code for GraphMFT (Neurocomputing 2023). Multiple improved graph attention networks capturing intra-modal context and inter-modal complementary info for ERC. IEMOCAP and MELD.",
            "why": "Another GAT-based fusion baseline in the same family; useful as a comparison point and for graph-attention design ideas. Lower-tier venue and less adoption than MMGCN/MM-DFN, so more of a reference than a primary base.",
            "language_framework": "Python / PyTorch + PyTorch Geometric",
            "license": "Not clearly specified",
            "maintained": "Low",
            "stars_approx": "~30-40",
            "venue_or_paper": "Neurocomputing 2023, arXiv:2208.00339",
            "fork_worthiness": "low"
          },
          {
            "name": "kenford953/GraphCAGE",
            "url": "https://github.com/kenford953/GraphCAGE",
            "what": "Official ICMI 2021 code. Graph Capsule Aggregation for UNALIGNED multimodal sequences (text/audio/video) — graph + capsule network for CMU-MOSI/MOSEI sentiment without word-level alignment.",
            "why": "Targets unaligned MOSI/MOSEI sentiment (different problem setting than ERC). Useful if you want the unaligned-sequence + capsule angle. Smaller, less-cited, older PyTorch; best treated as an idea source rather than a primary fork.",
            "language_framework": "Python / PyTorch (graph + capsule)",
            "license": "Not clearly specified",
            "maintained": "Low",
            "stars_approx": "~20-30",
            "venue_or_paper": "ICMI 2021, arXiv:2108.07543",
            "fork_worthiness": "low"
          },
          {
            "name": "pliang279/MFN (Graph-MFN lineage)",
            "url": "https://github.com/pliang279/MFN",
            "what": "Memory Fusion Network (AAAI 2018). Graph-MFN (the Dynamic Fusion Graph variant from the CMU-MOSEI paper) is built on this MFN pipeline; it lives inside the CMU-MultimodalSDK rather than as a standalone repo.",
            "why": "Graph-MFN has no clean standalone repo — its DFG code is bundled in CMU-MultimodalSDK (https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK). Only worth touching if you specifically need the interpretable Dynamic Fusion Graph; for a modern GNN-fusion project the ERC repos above dominate.",
            "language_framework": "Python / PyTorch",
            "license": "MIT (MFN repo)",
            "maintained": "Low — historical reference.",
            "stars_approx": "~100 (MFN)",
            "venue_or_paper": "MFN AAAI 2018; Graph-MFN/DFG = ACL 2018 CMU-MOSEI paper",
            "fork_worthiness": "low"
          }
        ],
        "recommendation": "Fork COGMEN (Exploration-Lab/COGMEN) as your primary base, and use MM-DFN (zerohd4869/MM-DFN) as the secondary base/baseline. Rationale: COGMEN is the cleanest modern GNN-ERC codebase, built on standard PyTorch Geometric layers (RGCNConv + TransformerConv), supports BOTH IEMOCAP (emotion) and CMU-MOSEI (sentiment), and is easy to extend — you can swap modality encoders, change graph construction, or add your bio/temporal-alignment ideas without rewriting the pipeline. MM-DFN is MIT-licensed (vs COGMEN's GPL-3.0), is ICASSP-lineage (matches your venue target), and shares the standard MMGCN IEMOCAP/MELD feature pipeline, so it gives you a permissive fallback plus a recognized baseline. Keep MMGCN as the must-cite/must-beat baseline (it defines the feature format the whole subfield reuses) and M3Net (CVPR 2023) as the current strong-SOTA reference to compare against.\n\nCaveats: (1) COGMEN is GPL-3.0 — if you ever need to release derived code under a permissive license or fold it into a non-GPL system, that is a blocker; in that case base on MM-DFN instead. (2) All of these repos are research dumps with low maintenance, old PyTorch Geometric versions (1.4–1.7), and pinned deps — budget setup time for a clean conda env and expect to recompute or re-download pre-extracted features. (3) These are conversation/sentence-level (IEMOCAP/MELD/MOSEI) — they do NOT include physiological/bio modalities; adapting to your K-EmoCon/KEMDy20 bio+audio+video setup means writing a new graph-construction + feature-encoder layer on top of the COGMEN/MM-DFN backbone. (4) Verify license files directly before redistribution for MMGCN/M3Net/GraphMFT/GraphCAGE (no explicit LICENSE found)."
      },
      {
        "idea": "Dynamic modality routing / mixture-of-experts (MoE) fusion / learned modality weighting / anchor or query selection in multimodal emotion recognition — with public code.",
        "repos": [
          {
            "name": "KuDA (Knowledge-Guided Dynamic Modality Attention Fusion)",
            "url": "https://github.com/MKMaS-GUET/KuDA",
            "what": "EMNLP 2024 Findings. Dynamically selects the dominant modality per sample and reweights each modality's contribution using sentiment-knowledge-guided dynamic attention. Two-stage pipeline: knowledge-inject pretraining (pretrain.py) then MSA training (train.py). This is essentially learned per-sample modality weighting/routing.",
            "why": "Best fit for the assigned idea: the dynamic-routing/learned-weighting is the core contribution, not a side feature. Recent (2024), MIT license, PyTorch, runnable training code, and it runs on the four standard benchmarks emotion/sentiment people cite (CMU-MOSI, CMU-MOSEI, plus Chinese CH-SIMS/CH-SIMSv2). Reuses the well-known MMSA feature pipeline so data setup is standard. Reported MOSI Acc-2 84.4/86.4, MOSEI Acc-2 83.3/86.5 — competitive, so you have a credible baseline to extend.",
            "fork_worthiness": "high",
            "language_framework": "PyTorch (1.9.0)",
            "license": "MIT",
            "maintained": "Light: ~12 commits, no releases; EMNLP24 vintage. Code present and runnable but expect to fix env/path issues yourself.",
            "stars_approx": "low (tens)",
            "venue_or_paper": "EMNLP 2024 Findings, arXiv:2410.04491"
          },
          {
            "name": "DynMM (Dynamic Multimodal Fusion)",
            "url": "https://github.com/zihuixue/DynMM",
            "what": "CVPR 2023 Workshop. Gating network that makes sample-wise, data-dependent decisions: modality-level DynMM picks which modalities to use, fusion-level DynMM does fine-grained input-dependent routing with a resource-aware loss for efficiency. Soft/hard gating implemented (--hard-gate, --global-gate).",
            "why": "The canonical reference implementation of dynamic modality routing as MoE-style gating. Cleanest, most-cited code for the 'gating function decides forward path' framing, and it directly implements CMU-MOSEI sentiment. Great if your angle is efficiency/compute-aware routing rather than max accuracy. Built on MultiBench so it composes well with other fusion baselines.",
            "fork_worthiness": "high",
            "language_framework": "PyTorch (adapted from MultiBench / ESANet)",
            "license": "Not specified in repo (no LICENSE file found) — clarify before redistributing",
            "maintained": "Inactive since ~2023; stable but unmaintained.",
            "stars_approx": "~126",
            "venue_or_paper": "CVPR 2023 Workshop (MULA), arXiv:2204.00102"
          },
          {
            "name": "QMF (Provable Dynamic Fusion for Low-Quality Multimodal Data)",
            "url": "https://github.com/QingyangZhang/QMF",
            "what": "ICML 2023. Quality/uncertainty-aware dynamic fusion: an uncertainty estimator scores each modality and dynamically sets fusion strength, with theory on when dynamic fusion provably generalizes better. Maps directly onto the project's existing QualityAwareGate concept.",
            "why": "Strongest theory + a clean uncertainty-as-weight mechanism — exactly the 'quality-aware gate' idea already in S-PACE V6, so the math/loss transfers to emotion easily. Caveat: official repo targets text-image classification and RGB-D scene recognition, NOT emotion benchmarks, so you would port the gate into an emotion backbone rather than fork wholesale.",
            "fork_worthiness": "medium",
            "language_framework": "PyTorch",
            "license": "Check repo (research code)",
            "maintained": "Author also maintains awesome-low-quality-multimodal-learning; moderately active ecosystem.",
            "stars_approx": "low-medium",
            "venue_or_paper": "ICML 2023, arXiv:2306.02050"
          },
          {
            "name": "MiSTER-E (Mixture-of-Experts for MER in Conversations)",
            "url": "https://github.com/iiscleap/MiSTER-E",
            "what": "Modular MoE for emotion recognition in conversations: separate speech-only, text-only, and cross-modal experts combined by a learned gating mechanism. Files: hierarchical_moe.py, train_moe.py, text_qwen.py, train_salmonn.py. Reports WF1 70.9 (IEMOCAP), 69.5 (MELD), 87.9 (MOSI).",
            "why": "Conceptually the purest 'one expert per modality + learned gate' MoE for emotion, and it targets the conversational ERC benchmarks (IEMOCAP/MELD) that ACII/ICASSP reviewers know. Downside: very early-stage repo (2 commits, ~3 stars, no README detail, depends on heavy LLM backbones SALMONN/Qwen), so reproduction risk is high.",
            "fork_worthiness": "medium",
            "language_framework": "PyTorch (LLM-backbone: SALMONN, Qwen)",
            "license": "Not stated",
            "maintained": "Brand new, barely populated; high reproduction risk.",
            "stars_approx": "very low (~3)",
            "venue_or_paper": "arXiv:2602.23300, 'A Mixture-of-Experts Model for Multimodal Emotion Recognition in Conversations'"
          },
          {
            "name": "MMoLRE (Multimodal Mixture of Low-Rank Experts)",
            "url": "https://arxiv.org/abs/2505.14143",
            "what": "Multi-task (MSA+MER) MoE with shared + task-specific low-rank experts to cut params as expert count grows. SOTA on CMU-MOSI, competitive on MER. NOTE: no public code found as of search.",
            "why": "Strong recent idea (low-rank experts + shared/task-specific split = good novelty handle for a thesis) but NO released implementation, so it can only serve as a design reference, not a fork target.",
            "fork_worthiness": "low",
            "language_framework": "Unknown (no code)",
            "license": "N/A (no code)",
            "maintained": "N/A — paper only",
            "stars_approx": "N/A",
            "venue_or_paper": "arXiv:2505.14143 (May 2025)"
          }
        ],
        "recommendation": "Fork KuDA (https://github.com/MKMaS-GUET/KuDA) as your primary base. It is the only repo that (1) makes dynamic per-sample modality weighting/routing its central contribution, (2) has runnable PyTorch training code under a permissive MIT license, and (3) already runs on the standard sentiment/emotion benchmarks (CMU-MOSI/MOSEI + CH-SIMS) using the common MMSA feature pipeline, with credible reported numbers to beat. Your thesis angle (bio-signal representation quality as the routing bottleneck) plugs in naturally: KuDA's \"sentiment-ratio\" weights become quality/uncertainty weights, and you can add a bio modality expert.\n\nAs a second base / mechanism donor, take the gating module from DynMM (https://github.com/zihuixue/DynMM) if you want the efficiency/compute-aware routing framing — it has the cleanest soft/hard gate code and a direct CMU-MOSEI implementation. If your project's QualityAwareGate is the real focus, lift QMF's (https://github.com/QingyangZhang/QMF) uncertainty-as-weight loss into the KuDA backbone (it has the theory but no emotion benchmark of its own).\n\nCaveats: (a) KuDA is lightly maintained (no releases, ~12 commits) — budget time for env/path fixes and pin torch 1.9. (b) DynMM has no LICENSE file; confirm usage rights before redistributing. (c) Avoid MiSTER-E as a base for now (2 commits, heavy SALMONN/Qwen deps, high reproduction risk) and treat MMoLRE as design inspiration only since it has no released code. (d) For ACII/ICASSP 2027, KuDA+MOSI/MOSEI alone is incremental — pair the routing idea with the IEMOCAP/MELD conversational setting or your bio modality to get a defensible novelty delta."
      },
      {
        "idea": "Missing-modality / modality-robust multimodal emotion recognition (modality dropout at inference, robustness benchmarks, evaluation protocols) — with code and how each repo evaluates missing-modality robustness.",
        "repos": [
          {
            "name": "MMIN (Missing Modality Imagination Network)",
            "url": "https://github.com/AIM3-RUC/MMIN",
            "what": "The de-facto reference baseline for uncertain-missing-modality emotion recognition (ACL 2021, Zhao/Li/Jin, RUC). Trains a unified model that imagines/reconstructs the representation of any missing modality (cascade residual autoencoder + cycle consistency) so a single model handles all missing combos.",
            "why": "This is the canonical entry point everyone cites and builds on. Its evaluation protocol is the field standard: enumerate all 6 partial-modality conditions {a, v, l, av, al, vl} plus full {avl}, train under uncertain dropout, report per-condition and average WA/UA/F1. The 'protocol' you want to adopt IS basically MMIN's. Self-contained, MIT, no LLM/heavy deps.",
            "language_framework": "PyTorch",
            "license": "MIT",
            "maintained": "Low activity (~2 commits, last meaningful update years ago) but code is complete and runs; treated as a stable baseline rather than an active project.",
            "stars_approx": "~73",
            "venue_or_paper": "ACL 2021 (aclanthology.org/2021.acl-long.203)",
            "fork_worthiness": "high"
          },
          {
            "name": "CIDer (Robust MER under Missing Modalities + Distribution Shifts)",
            "url": "https://github.com/gw-zhong/CIDer",
            "what": "2025 framework (arXiv:2506.10452) combining Model-Specific Self-Distillation + Model-Agnostic Causal Inference. Built on MulT/SELF-MM. Most importantly ships a full robustness benchmark suite: 5 missing-modality modes (RMFM, RMM, TMFM, STMFM, SMM) AND an OOD/distribution-shift repartition of MOSI/MOSEI.",
            "why": "The single most up-to-date and complete evaluation harness for this exact idea. It defines and implements multiple missing-modality dropout protocols plus distribution-shift testing, with train/eval/inference scripts and released weights. Even if you do not use its method, its benchmark + protocols are directly reusable as your experimental backbone for ICASSP/ACII 2027.",
            "language_framework": "PyTorch (on top of MulT + SELF-MM)",
            "license": "MIT",
            "maintained": "Recent / active (2025 release, weights + HF datasets provided), but young repo so small community (~16 stars).",
            "stars_approx": "~16",
            "venue_or_paper": "arXiv:2506.10452 (2025)",
            "fork_worthiness": "high"
          },
          {
            "name": "GCNet (Graph Completion Network for Incomplete Multimodal Conversation)",
            "url": "https://github.com/zeroQiaoba/GCNet",
            "what": "TPAMI 2023 graph-completion approach for incomplete multimodal learning in conversational emotion recognition. Speaker-GNN + Temporal-GNN with coupled translation-prediction objective; handles random missing rates (e.g. masking ratio 0.0-0.7) on IEMOCAP/CMU-MOSI/MOSEI.",
            "why": "Strongest option if you want a conversational/temporal setting and a continuous missing-rate robustness curve (sweep mask ratio 0->0.7) rather than discrete modality combos. High-quality PyTorch from a well-known author (zeroQiaoba/MER lab). Good second baseline to show robustness across missing rates.",
            "language_framework": "PyTorch (GNN)",
            "license": "Not clearly stated in repo (verify before redistributing)",
            "maintained": "Moderate; stable research repo, author actively publishes in MER.",
            "stars_approx": "~100",
            "venue_or_paper": "IEEE TPAMI 2023 (arXiv:2203.02177)",
            "fork_worthiness": "medium"
          },
          {
            "name": "MPLMM (Multimodal Prompt Learning with Missing Modalities)",
            "url": "https://github.com/zrguo/MPLMM",
            "what": "ACL 2024 Main. Transformer + prompt-learning (generative / missing-signal / missing-type prompts) for MSA and ERC under missing modalities. Supports MOSI, MOSEI, IEMOCAP, CH-SIMS with documented get_dim/get_seq_len/get_missing_mode hooks for custom datasets.",
            "why": "Most modern/parameter-efficient method here and the cleanest codebase for plugging in your own dataset (explicit missing-mode interface). Good if your contribution is method-side (prompts) rather than benchmark-side. Slightly less of a 'standard protocol' anchor than MMIN/CIDer.",
            "language_framework": "PyTorch (Transformer + prompt tuning)",
            "license": "MIT",
            "maintained": "Recently reconstructed (2024), few commits but documented and usable.",
            "stars_approx": "~150",
            "venue_or_paper": "ACL 2024 Main (arXiv:2407.05374)",
            "fork_worthiness": "medium"
          },
          {
            "name": "IF-MMIN (Invariant-Feature MMIN)",
            "url": "https://github.com/ZhuoYulang/IF-MMIN",
            "what": "Extension of the MMIN line (arXiv:2210.15359) adding modality-invariant feature learning via Central Moment Discrepancy + invariant-feature imagination. IEMOCAP AVL features provided. Same MMIN-style two-stage protocol and per-condition eval.",
            "why": "Useful as a drop-in stronger MMIN baseline and shares MMIN's exact data pipeline/eval, so cheap to add once you have MMIN running. Lower priority as a base because it is narrower (IEMOCAP-only) and an incremental improvement.",
            "language_framework": "PyTorch (Python 3.8)",
            "license": "MIT",
            "maintained": "Low (research drop, few commits).",
            "stars_approx": "~38",
            "venue_or_paper": "arXiv:2210.15359",
            "fork_worthiness": "low"
          }
        ],
        "recommendation": "Base your work on TWO repos used together. (1) Fork CIDer (https://github.com/gw-zhong/CIDer) as your experimental/benchmark backbone — it is the only repo that ships a complete, recent, runnable robustness harness covering multiple missing-modality dropout modes (RMFM/RMM/TMFM/STMFM/SMM) PLUS distribution-shift (OOD) splits on MOSI/MOSEI, with weights for reproduction. Adopt its protocol as your evaluation standard. (2) Also stand up MMIN (https://github.com/AIM3-RUC/MMIN) as the canonical baseline + its 6-condition {a,v,l,av,al,vl} protocol that reviewers expect to see; IF-MMIN can be added almost for free since it reuses MMIN's pipeline. If your target setting is conversational/temporal or you want a continuous missing-rate robustness curve, swap in GCNet instead of MMIN. Caveats: (a) these repos use precomputed features (BERT/COVAREP/OpenFace-style) released via BaiduYun/HF — secure feature downloads early, this is the usual blocker; (b) CIDer is a young repo (~16 stars), so budget time to debug and re-verify its numbers before trusting them; (c) confirm GCNet's license before any redistribution — it is not clearly stated. Net: CIDer for the benchmark/protocol, MMIN for the must-have baseline."
      },
      {
        "idea": "K-EmoCon multimodal emotion recognition implementations + hierarchical/temporal multi-scale multimodal fusion repos (with code), for a master's thesis targeting ICASSP/ACII 2027.",
        "repos": [
          {
            "name": "Kaist-ICLab/K-EmoCon_SupplementaryCodes",
            "url": "https://github.com/Kaist-ICLab/K-EmoCon_SupplementaryCodes",
            "what": "Official supplementary code for the K-EmoCon dataset (Scientific Data 2020). Provides the canonical preprocessing pipeline (raw biosignals -> 5s segments with arousal/valence + categorical annotations as JSON), outlier detection, inter-rater reliability, and classical baselines (Gaussian NB, XGBoost) with stratified k-fold and LOSO cross-validation.",
            "why": "This is the authoritative, dataset-author-blessed preprocessing + baseline reference for K-EmoCon. Even though it is classical ML (not deep), forking it guarantees your segmentation/annotation alignment matches the published protocol, and gives you the exact baseline numbers reviewers expect. Critical for not reinventing the data pipeline. Note: it uses 'ecg' not 'hr' naming and PyTEAP features — differs from your NPZ layout (bvp/eda/temp/hr 160x2), so use it as a protocol reference, not a drop-in.",
            "language_framework": "Python / scikit-learn + XGBoost (no deep learning)",
            "license": "MIT",
            "maintained": "Low activity, ~55 commits, stable/mature (last meaningful updates years old) — but it is the official reference so staleness is acceptable",
            "stars_approx": "~28",
            "venue_or_paper": "Park et al., 'K-EmoCon', Scientific Data 7:293 (2020), arXiv:2005.04120",
            "fork_worthiness": "high"
          },
          {
            "name": "ispamm/MHyEEG (PHemoNet / H2 hierarchical hypercomplex)",
            "url": "https://github.com/ispamm/MHyEEG",
            "what": "Official PyTorch repo for a family of multimodal physiological emotion recognition models: HyperFuseNet (ICASSPW 2023), PHemoNet (RTSI 2024), and H2 'Hierarchical Hypercomplex Network' (MLSP 2024). Uses parameterized hypercomplex (PHM/PHC) encoders per modality and a hypercomplex fusion layer instead of naive concatenation. Includes preprocessing, augmentation, training (W&B), and released pretrained weights.",
            "why": "Closest match to BOTH halves of the assigned idea: it is hierarchical multimodal fusion on EEG + peripheral physiological signals (arousal/valence), the exact modality regime of K-EmoCon's bio stream. Actively maintained (pretrained weights released 2025), clean PyTorch, ICASSP-lineage venues (directly aligned with your ICASSP/ACII target). The hierarchical/hypercomplex fusion is a strong, defensible architectural backbone to adapt to K-EmoCon bio channels.",
            "language_framework": "PyTorch (+ Weights & Biases)",
            "license": "Not clearly stated in repo — verify before redistribution; treat as use-with-caution",
            "maintained": "Yes — last updates May 2025, pretrained weights added; small but active",
            "stars_approx": "~93",
            "venue_or_paper": "H2: MLSP 2024; PHemoNet: RTSI 2024; HyperFuseNet: ICASSPW 2023. Reported on MAHNOB-HCI (best H2: Valence Acc 67.9%/F1 0.685, Arousal Acc 56.9%/F1 0.557)",
            "fork_worthiness": "high"
          },
          {
            "name": "praveena2j/RJCMA (Recursive Joint Cross-Modal Attention)",
            "url": "https://github.com/praveena2j/RJCMA",
            "what": "Official PyTorch code (CVPRW 2024, ABAW challenge) for Recursive Joint Cross-Modal Attention — a cross-modal attention fusion that recursively refines audio/visual/text features by attending each modality to the joint representation, optimized for dimensional (CCC) arousal/valence.",
            "why": "Best-in-class, recently published cross-modal temporal fusion module with reported CCC on dimensional emotion (valence 0.585, arousal 0.674 on Affwild2). Your K-EmoCon work is continuous (5s) arousal/valence with CCC as the metric — RJCMA's fusion block is directly transplantable as a fusion stage for video+audio+bio. Good for the 'fusion mechanism' contribution of a paper. Caveat: designed for audio-visual-text, not physiological; you'd adapt the modality encoders.",
            "language_framework": "PyTorch",
            "license": "Check repo (verify before use)",
            "maintained": "Moderate — research-grade, tied to 2024 challenge release",
            "stars_approx": "low-to-moderate (tens)",
            "venue_or_paper": "Praveen & Alam, 'Recursive Joint Cross-Modal Attention...', CVPRW 2024 (ABAW), arXiv:2403.13659",
            "fork_worthiness": "medium"
          },
          {
            "name": "katerynaCh/multimodal-emotion-recognition",
            "url": "https://github.com/katerynaCh/multimodal-emotion-recognition",
            "what": "PyTorch implementation of 'Self-attention fusion for audiovisual emotion recognition with incomplete data'. Offers late / intermediate transformer and intermediate-attention fusion variants plus modality-dropout (softhard/noise) for missing-modality robustness; EfficientFace visual + audio encoders on RAVDESS.",
            "why": "Clean, well-starred, MIT-licensed reference for transformer fusion + missing-modality handling — relevant because K-EmoCon has noisy/incomplete bio and AV streams (your QualityAwareGate addresses the same problem). Good code to borrow the modality-dropout and intermediate-transformer fusion patterns. Caveat: RAVDESS-only, audiovisual, categorical (not dimensional, not physiological) — a fusion-pattern donor, not a base.",
            "language_framework": "PyTorch",
            "license": "MIT",
            "maintained": "Stable/mature, low recent activity",
            "stars_approx": "~163",
            "venue_or_paper": "Chumachenko et al., 'Self-attention fusion for audiovisual emotion recognition with incomplete data' (ICPR/related)",
            "fork_worthiness": "medium"
          },
          {
            "name": "guanghaoyin/RTCAN-1D",
            "url": "https://github.com/guanghaoyin/RTCAN-1D",
            "what": "PyTorch code (TOMM 2022) for a 1D residual temporal-channel attention network fusing music + electrodermal activity (EDA) signals; evaluated on multimodal datasets including DEAP for valence/arousal.",
            "why": "Directly relevant to physiological time-series fusion with EDA — the dominant K-EmoCon bio channel. Useful as a 1D-conv + temporal/channel-attention encoder reference for the bio branch and for DEAP cross-dataset generalization (your Section 4.3). Caveat: narrow modality pair (music+EDA), TOMM 2022 vintage, low maintenance.",
            "language_framework": "PyTorch",
            "license": "Check repo",
            "maintained": "Low / stable, older (2022)",
            "stars_approx": "low (tens)",
            "venue_or_paper": "Yin et al., 'A Multimodal framework for large scale Emotion Recognition by Fusing Music and Electrodermal Activity Signals', ACM TOMM 2022, arXiv:2008.09743",
            "fork_worthiness": "low"
          },
          {
            "name": "feiyuchen7/M3NET",
            "url": "https://github.com/feiyuchen7/M3NET",
            "what": "PyTorch implementation (CVPR 2023) of a multivariate, multi-frequency, multimodal GNN for Emotion Recognition in Conversation (IEMOCAP/MELD).",
            "why": "Strong, well-cited multi-scale (multi-frequency) graph fusion reference if you frame K-EmoCon as conversational/dyadic emotion recognition (it IS paired debates). The multi-frequency idea connects to your temporal multi-scale theme. Caveat: conversation text/audio/visual ERC setting, not physiological — architecturally inspiring but not a direct K-EmoCon base.",
            "language_framework": "PyTorch (DGL/PyG-style GNN)",
            "license": "Check repo",
            "maintained": "Stable, moderate activity",
            "stars_approx": "~100+",
            "venue_or_paper": "Chen et al., 'Multivariate, Multi-frequency and Multimodal...', CVPR 2023",
            "fork_worthiness": "low"
          }
        ],
        "recommendation": "Fork two, for two different layers. (1) Base your data/protocol layer on Kaist-ICLab/K-EmoCon_SupplementaryCodes (high) — it is the official preprocessing + baseline reference, so your 5s segmentation, annotation alignment, and LOSO/k-fold protocol match the published standard and reviewers' expectations. Do NOT use it as your model; it is classical ML and its channel naming (ecg/PyTEAP) differs from your existing NPZ layout (bvp/eda/temp/hr 160x2) — treat it as the ground-truth protocol spec to validate your own pipeline against. (2) Base your model/fusion layer on ispamm/MHyEEG's H2 hierarchical model (high) — it is the only actively maintained PyTorch repo that is simultaneously hierarchical multimodal fusion AND on EEG+peripheral physiological signals for arousal/valence, with pretrained weights and an ICASSP-lineage pedigree that matches your ICASSP/ACII 2027 target. Borrow the RJCMA recursive cross-modal attention block (medium) and katerynaCh's modality-dropout (medium) as drop-in fusion/robustness upgrades for your video+audio+bio setting (your QualityAwareGate addresses the same incomplete-modality problem). Caveats: (a) verify MHyEEG and RJCMA licenses before any redistribution — MHyEEG's license is not clearly stated. (b) Most of these report categorical/AV results; for K-EmoCon's continuous CCC≈0 ceiling none will magically lift it, which actually supports your thesis that bio representation quality is the bottleneck — choose the fusion repo as the architecture, not as a results miracle. (c) None natively ingest your exact NPZ; budget time for a thin dataset adapter rather than expecting plug-and-play."
      }
    ]
  }
}
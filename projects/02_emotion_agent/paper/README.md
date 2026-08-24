# Paper draft — ICL modality-gating (ACL/EMNLP Findings target)

## 컴파일 (Overleaf 권장 — 2분)
1. overleaf.com → New Project → Upload Project, 또는 Blank로 만들고
2. `main.tex` 붙여넣기 + `figures/` 4장(figA/B/C/D) 업로드
3. Recompile → PDF.
   - 로컬엔 pdflatex 없음(texlive 미설치). Overleaf가 제일 쉬움.

## ACL 공식 템플릿으로 바꾸기 (제출 시)
현재 main.tex는 standalone(article)로 바로 컴파일되게 해둠. 제출용으로는:
1. ACL ARR Overleaf 템플릿(acl.sty) 가져와서
2. preamble만 `\documentclass[11pt]{article}\usepackage[review]{acl}`로 교체,
3. 본문/표/figure/bib는 그대로 사용.

## 내용 = PAPER_v1.md backbone + 실측 수치 (밤샘 exp_035~045)
- §4.1 게이팅 6모델 표 (Table 1, figA)
- §4.2 TR 메커니즘 (figC)
- §4.3 fertility 예측자 (figB)
- §4.4 2nd 도메인 leetspeak (figD)
- §4.5 경계조건(diabetes) + §4.6 negative controls/Mistral
- bib: Min'22, Pan'23, IsoBench, TabLLM, Shin'22, Yadlowsky'23, Razeghi'22

## 남은 것 (JY)
- 제목/저자/소속, 298-consensus 데이터 출처 문장 다듬기
- (선택) n=3→7 seed로 CI 보강, related work 인용 추가
- ARR 제출 사이클 날짜 확인 후 OpenReview 업로드

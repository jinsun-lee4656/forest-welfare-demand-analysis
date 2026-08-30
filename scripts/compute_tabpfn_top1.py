# -*- coding: utf-8 -*-
"""
9-1e(신규): TabPFN-2.5의 Top-1/Top-2 성능을 9-1b와 동일한 방식(다수결 베이스라인 대비 개선폭)으로
계산. (사용자 지정: 이 Top-1 분석은 노트북 9-4/9-9절의 TabPFN-3와 달리 TabPFN-2.5 버전으로
계산 -- model_path에 "v2.5"가 들어간 체크포인트 파일명을 넘겨 버전을 명시 고정한다. 게이트된
HuggingFace 레포(Prior-Labs/tabpfn_2_5)라 최초 1회는 라이선스 동의 확인 + 체크포인트 다운로드가
필요하다(사용자가 ux.priorlabs.ai에서 이미 동의 완료).
TabPFN은 예측 비용이 커서 9-9절과 동일하게 test셋 500행 랜덤 서브샘플로 평가한다.
X_train/Y_train(9-1절 원본 분리, X_test/Y_test와 동일 기준)으로 학습해 proba_mat과 완전히
동일한 기준(같은 test 분리)에서 비교 가능하게 한다.

추가 수정(2026-08-31): 전체 X_train(약 5,724행)을 그대로 컨텍스트로 써서 학습(fit)했더니
2시간을 넘겨도 끝나지 않아(9-4/9-9절 TabPFN-3 기록상 최악 케이스도 ~40분이었는데 3배 이상
초과 — v2.5 체크포인트가 v3보다 무겁거나 이번 실행 자체가 유독 느렸던 것으로 추정, 원인
불명) 사용자 승인 하에 중단했다. 학습셋도 test와 동일한 방식(랜덤 서브샘플)으로
TABPFN_TRAIN_SUBSAMPLE_N행으로 줄여 재시도한다 — 결과에는 "[학습 서브샘플]"로 명시해
전체 학습셋 기준이 아님을 투명하게 표기한다.
"""
import os, sys, json, time
import matplotlib
matplotlib.use("Agg")

t_start = time.time()
NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

assert os.environ.get("TABPFN_TOKEN"), "TABPFN_TOKEN 환경변수가 없습니다"

nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

# 주의(버그 수정 2026-08-30): 원래 마커였던 "Top-2 추천 성능" 플롯 타이틀 셀은 prec1/rec1/hit1/
# top1_label_idx/top2_label_idx가 정의되는 9-1b 셀(바로 다음 markdown 다음 셀)보다 앞에 있어,
# 이 마커까지만 재구성하면 아래에서 ns["prec1"] 등이 KeyError로 죽는다(실제로 재현됨). prec1 등이
# 전부 정의되는 지점(9-1b의 topk_metrics(1,...) 호출 셀)을 마커로 바꿔 그 셀까지 포함해서 재구성한다.
idx_target = next(i for i, c in enumerate(cells)
                   if "top2_label_idx = [BROAD_CATS.index" in "".join(c["source"]))
SKIP = set()
for marker in ["boost_models = {", "best_model_name = comparison_df", "gss_val = GroupShuffleSplit", "lgbm_val = Pipeline"]:
    SKIP.add(next(i for i, c in enumerate(cells) if marker in "".join(c["source"])))
print(f"목표 셀: {idx_target}, 건너뛸 셀: {sorted(SKIP)}")

ns = {"display": lambda *a, **k: None}
for i, c in enumerate(cells):
    if i > idx_target:
        break
    if i in SKIP or c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip() or src.lstrip().startswith(("!", "%")):
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print(f"0~{idx_target}번 셀 재구성 완료 -- {time.time()-t_start:.0f}초")

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from tabpfn import TabPFNClassifier

pre2 = ns["pre2"]; X_train = ns["X_train"]; Y_train = ns["Y_train"]
X_test = ns["X_test"]; Y_test = ns["Y_test"]; BROAD_CATS = ns["BROAD_CATS"]
prec1, rec1, hit1 = ns["prec1"], ns["rec1"], ns["hit1"]
prec2, rec2, hit2 = ns["prec2"], ns["rec2"], ns["hit2"]
b1_prec, b1_rec, b1_hit = ns["b1_prec"], ns["b1_rec"], ns["b1_hit"]
b2_prec, b2_rec, b2_hit = ns["b2_prec"], ns["b2_rec"], ns["b2_hit"]
top1_label_idx, top2_label_idx = ns["top1_label_idx"], ns["top2_label_idx"]

TABPFN_SUBSAMPLE_N = 500
TABPFN_TRAIN_SUBSAMPLE_N = 2000  # 전체 X_train(5,724행)은 2시간 넘게도 안 끝나서 학습셋도 서브샘플링
rng_sub = np.random.RandomState(42)
test_sub_pos = rng_sub.choice(len(X_test), size=min(TABPFN_SUBSAMPLE_N, len(X_test)), replace=False)
X_test_sub, Y_test_sub = X_test.iloc[test_sub_pos], Y_test.iloc[test_sub_pos]
print(f"서브샘플 크기: {len(X_test_sub)}행 (전체 test={len(X_test)}행)")

rng_train_sub = np.random.RandomState(42)
train_sub_pos = rng_train_sub.choice(len(X_train), size=min(TABPFN_TRAIN_SUBSAMPLE_N, len(X_train)), replace=False)
X_train_sub, Y_train_sub = X_train.iloc[train_sub_pos], Y_train.iloc[train_sub_pos]
print(f"학습 서브샘플 크기: {len(X_train_sub)}행 (전체 train={len(X_train)}행)")

t0 = time.time()
tabpfn_pipe = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(
    TabPFNClassifier(model_path="tabpfn-v2.5-classifier-v2.5_default.ckpt",
                      ignore_pretraining_limits=True, random_state=42)))])
tabpfn_pipe.fit(X_train_sub, Y_train_sub)
print(f"TabPFN-2.5 fit: {time.time()-t0:.0f}초")

t0 = time.time()
proba_list_t = tabpfn_pipe.predict_proba(X_test_sub)
proba_mat_t = np.column_stack([p[:, 1] for p in proba_list_t])
print(f"TabPFN-2.5 predict_proba({len(X_test_sub)}행): {time.time()-t0:.0f}초")


def topk_metrics(K, proba, Y_true):
    topk_idx = np.argsort(-proba, axis=1)[:, :K]
    Y_arr = Y_true.values
    prec_list, rec_list, hit_list = [], [], []
    for i in range(len(Y_arr)):
        true_idx = set(np.where(Y_arr[i] == 1)[0])
        if len(true_idx) == 0:
            continue
        inter = len(true_idx & set(topk_idx[i]))
        prec_list.append(inter / K)
        rec_list.append(inter / len(true_idx))
        hit_list.append(1 if inter > 0 else 0)
    return float(np.mean(prec_list)), float(np.mean(rec_list)), float(np.mean(hit_list)), topk_idx


def majority_baseline(label_idx_list, Y_true):
    Y_arr = Y_true.values
    K = len(label_idx_list)
    prec_list, rec_list, hit_list = [], [], []
    for i in range(len(Y_arr)):
        true_idx = set(np.where(Y_arr[i] == 1)[0])
        if len(true_idx) == 0:
            continue
        inter = len(true_idx & set(label_idx_list))
        prec_list.append(inter / K)
        rec_list.append(inter / len(true_idx))
        hit_list.append(1 if inter > 0 else 0)
    return float(np.mean(prec_list)), float(np.mean(rec_list)), float(np.mean(hit_list))


t_prec1, t_rec1, t_hit1, t_top1_idx = topk_metrics(1, proba_mat_t, Y_test_sub)
t_prec2, t_rec2, t_hit2, _ = topk_metrics(2, proba_mat_t, Y_test_sub)
# 같은 서브샘플에서의 다수결 베이스라인(라벨은 전체 test 기준 고정 top1/top2 그대로 사용)
sb1_prec, sb1_rec, sb1_hit = majority_baseline([top1_label_idx], Y_test_sub)
sb2_prec, sb2_rec, sb2_hit = majority_baseline(top2_label_idx, Y_test_sub)

out = io_text = []
lines = []
lines.append(f"[TabPFN-2.5, 학습 {len(X_train_sub)}행 서브샘플/test {len(X_test_sub)}행 서브샘플] Top-1: Precision={t_prec1*100:.1f}%  Recall={t_rec1*100:.1f}%  Hit={t_hit1*100:.1f}%")
lines.append(f"[TabPFN-2.5, 학습 {len(X_train_sub)}행 서브샘플/test {len(X_test_sub)}행 서브샘플] Top-2: Precision={t_prec2*100:.1f}%  Recall={t_rec2*100:.1f}%  Hit={t_hit2*100:.1f}%")
lines.append("")
lines.append(f"[다수결 베이스라인, 같은 서브샘플] Top-1: Precision={sb1_prec*100:.1f}%  Hit={sb1_hit*100:.1f}%")
lines.append(f"[다수결 베이스라인, 같은 서브샘플] Top-2: Precision={sb2_prec*100:.1f}%  Recall={sb2_rec*100:.1f}%  Hit={sb2_hit*100:.1f}%")
lines.append("")
lines.append(f"[TabPFN-2.5-베이스라인 개선폭] Top-1: Precision {(t_prec1-sb1_prec)*100:+.1f}%p, Hit {(t_hit1-sb1_hit)*100:+.1f}%p")
lines.append(f"[TabPFN-2.5-베이스라인 개선폭] Top-2: Precision {(t_prec2-sb2_prec)*100:+.1f}%p, Recall {(t_rec2-sb2_rec)*100:+.1f}%p, Hit {(t_hit2-sb2_hit)*100:+.1f}%p")
lines.append("")
lines.append(f"[참고 - 기존 RandomForest, 전체 test {len(X_test)}행] Top-1: Precision={prec1*100:.1f}%  Hit={hit1*100:.1f}%  (베이스라인 대비 +{(prec1-b1_prec)*100:.1f}%p)")
lines.append(f"[참고 - 기존 RandomForest, 전체 test {len(X_test)}행] Top-2: Precision={prec2*100:.1f}%  Recall={rec2*100:.1f}%  Hit={hit2*100:.1f}%  (베이스라인 대비 +{(prec2-b2_prec)*100:.1f}%p/+{(rec2-b2_rec)*100:.1f}%p/+{(hit2-b2_hit)*100:.1f}%p)")

result_text = "\n".join(lines)
print("\n" + result_text)

RESULT_PATH = r"c:\Users\JS\Desktop\MDIS\tabpfn_top1_result.json"
json.dump({"result_text": result_text,
           "t_prec1": t_prec1, "t_rec1": t_rec1, "t_hit1": t_hit1,
           "t_prec2": t_prec2, "t_rec2": t_rec2, "t_hit2": t_hit2,
           "sb1_prec": sb1_prec, "sb1_hit": sb1_hit,
           "sb2_prec": sb2_prec, "sb2_rec": sb2_rec, "sb2_hit": sb2_hit,
           "n_sub": len(X_test_sub), "n_test": len(X_test),
           "n_train_sub": len(X_train_sub), "n_train": len(X_train),
           "prec1": prec1, "hit1": hit1, "prec2": prec2, "rec2": rec2, "hit2": hit2,
           "b1_prec": b1_prec, "b1_hit": b1_hit, "b2_prec": b2_prec, "b2_rec": b2_rec, "b2_hit": b2_hit},
          open(RESULT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n결과 저장: {RESULT_PATH}")
print(f"전체 소요시간: {time.time()-t_start:.0f}초")

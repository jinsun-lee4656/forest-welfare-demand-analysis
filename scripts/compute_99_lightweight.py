# -*- coding: utf-8 -*-
"""
9-9절(LightGBM/CatBoost/TabPFN 임계값 최적화)에 필요한 최소 상태만 가볍게 재구성해서 빠르게 계산.
(668개 원본 문항 SHAP/순열중요도/하이퍼파라미터 탐색 등 무관한 무거운 계산은 전부 건너뛴다)
"""
import os, sys, json, time
import matplotlib
matplotlib.use("Agg")

t_start = time.time()
NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

try:
    from IPython.display import display
except Exception:
    def display(*a, **k):
        for x in a: print(x)

ns = {"display": display}

# 0~96번 셀까지 실행하되, cell 93(boost_models -- 전체 TabPFN 재비교, 매우 비쌈)과 cell 94(그
# 결과를 쓰는 test셋 임계값 탐색)는 9-8/9-9절이 필요로 하지 않으므로 건너뛴다.
idx_96 = next(i for i, c in enumerate(cells) if "gss_val = GroupShuffleSplit" in "".join(c["source"]))
idx_93 = next(i for i, c in enumerate(cells) if "boost_models = {" in "".join(c["source"]))
idx_94 = next(i for i, c in enumerate(cells) if 'best_model_name = comparison_df' in "".join(c["source"]))
print(f"9-8절(train2/val 분리) 셀 인덱스: {idx_96}, 건너뛸 셀: {idx_93}(boost_models), {idx_94}(test 임계값탐색)")
SKIP = {idx_93, idx_94}
# cell 96 마지막 줄이 cell 94에서 정의되는 macro_f1_opt/micro_f1_opt/hamming_opt를 참조하므로
# (내용은 우리 목적과 무관, 그냥 print문이라 값 자체는 안 씀) 더미 값으로 미리 채워 NameError 방지
ns.update(macro_f1_opt=float("nan"), micro_f1_opt=float("nan"), hamming_opt=float("nan"))

for i, c in enumerate(cells):
    if i > idx_96:
        break
    if i in SKIP:
        print(f"[건너뜀] cell {i}")
        continue
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip():
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ns)

print(f"\n0~{idx_96}번 셀(필요 상태 재구성) 완료 -- {time.time()-t_start:.0f}초")

# ---- 여기부터 9-9절 로직을 직접 수행 (cell 98과 동일 로직, TabPFN은 val/test 500행 서브샘플) ----
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, f1_score, hamming_loss
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

g = ns
pre2, X_train2, Y_train2, X_val, Y_val, X_test, Y_test, BROAD_CATS = (
    g["pre2"], g["X_train2"], g["Y_train2"], g["X_val"], g["Y_val"], g["X_test"], g["Y_test"], g["BROAD_CATS"])
macro_f1_v2, micro_f1_v2, hamming_v2 = g["macro_f1_v2"], g["micro_f1_v2"], g["hamming_v2"]
macro_f1_recheck, micro_f1, hamming = g["macro_f1_recheck"], g["micro_f1"], g["hamming"]

t0 = time.time()
lgbm_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(LGBMClassifier(
    n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.03, subsample=0.8,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)))])
lgbm_val.fit(X_train2, Y_train2)
print(f"LightGBM fit: {time.time()-t0:.0f}초")

t0 = time.time()
cat_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(CatBoostClassifier(
    depth=6, learning_rate=0.05, iterations=500, auto_class_weights="Balanced",
    random_state=42, verbose=False)))])
cat_val.fit(X_train2, Y_train2)
print(f"CatBoost fit: {time.time()-t0:.0f}초")

val_models = {"LightGBM": lgbm_val, "CatBoost": cat_val}

assert os.environ.get("TABPFN_TOKEN"), "TABPFN_TOKEN 없음"
from tabpfn import TabPFNClassifier
t0 = time.time()
tabpfn_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(
    TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)))])
tabpfn_val.fit(X_train2, Y_train2)
val_models["TabPFN"] = tabpfn_val
print(f"TabPFN fit: {time.time()-t0:.0f}초")

TABPFN_SUBSAMPLE_N = 500
rng_sub = np.random.RandomState(42)
val_sub_pos = rng_sub.choice(len(X_val), size=min(TABPFN_SUBSAMPLE_N, len(X_val)), replace=False)
test_sub_pos = rng_sub.choice(len(X_test), size=min(TABPFN_SUBSAMPLE_N, len(X_test)), replace=False)

results_99 = {}
for name, model in val_models.items():
    t0 = time.time()
    if name == "TabPFN":
        X_val_eval, Y_val_eval = X_val.iloc[val_sub_pos], Y_val.iloc[val_sub_pos]
        X_test_eval, Y_test_eval = X_test.iloc[test_sub_pos], Y_test.iloc[test_sub_pos]
    else:
        X_val_eval, Y_val_eval = X_val, Y_val
        X_test_eval, Y_test_eval = X_test, Y_test

    proba_val_b = np.column_stack([p[:, 1] for p in model.predict_proba(X_val_eval)])
    proba_test_b = np.column_stack([p[:, 1] for p in model.predict_proba(X_test_eval)])

    thresholds_b = {}
    for i, cat in enumerate(BROAD_CATS):
        prec, rec, thr = precision_recall_curve(Y_val_eval.iloc[:, i], proba_val_b[:, i])
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        thresholds_b[cat] = float(thr[int(np.argmax(f1s[:-1]))]) if len(thr) > 0 else 0.5

    pred_fixed = (proba_test_b >= 0.5).astype(int)
    pred_opt = np.column_stack([
        (proba_test_b[:, i] >= thresholds_b[cat]).astype(int)
        for i, cat in enumerate(BROAD_CATS)
    ])

    suffix = " [500행 서브샘플]" if name == "TabPFN" else ""
    results_99[f"{name}(0.5고정){suffix}"] = dict(
        Macro_F1=f1_score(Y_test_eval, pred_fixed, average="macro"),
        Micro_F1=f1_score(Y_test_eval, pred_fixed, average="micro"),
        Hamming=hamming_loss(Y_test_eval, pred_fixed),
    )
    results_99[f"{name}(임계값최적화){suffix}"] = dict(
        Macro_F1=f1_score(Y_test_eval, pred_opt, average="macro"),
        Micro_F1=f1_score(Y_test_eval, pred_opt, average="micro"),
        Hamming=hamming_loss(Y_test_eval, pred_opt),
    )
    print(f"{name} 완료 ({time.time()-t0:.0f}초)")

results_99["RandomForest(9-8절, 임계값최적화)"] = dict(Macro_F1=macro_f1_v2, Micro_F1=micro_f1_v2, Hamming=hamming_v2)
results_99["RandomForest(기존, 0.5고정)"] = dict(Macro_F1=macro_f1_recheck, Micro_F1=micro_f1, Hamming=hamming)

comparison_99 = pd.DataFrame(results_99).T
print()
print(comparison_99.round(4).to_string())

best_row = comparison_99["Macro_F1"].astype(float).idxmax()
best_val = comparison_99.loc[best_row, "Macro_F1"]
print(f"\nMacro-F1 기준 최종 최고 성능: {best_row} (Macro-F1={best_val:.3f})")
print(f"\n전체 소요시간: {time.time()-t_start:.0f}초")

comparison_99.to_csv(r"C:\Users\JS\AppData\Local\Temp\claude\c--Users-JS-Desktop-MDIS\8a32d8b2-fcd5-4d3c-ab54-6781f3832676\scratchpad\comparison_99_result.csv")

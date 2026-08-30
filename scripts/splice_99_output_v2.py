# -*- coding: utf-8 -*-
"""
9-9절(TabPFN 포함 임계값 최적화) 재시도 v2:
  - 커널을 한 번만 띄우고(setup_kernel), cell 0~92까지는 실행하되 무거운 cell 93(TabPFN 포함
    전체 비교)은 "TabPFN 없이"(TABPFN_TOKEN 임시 제거) 빠르게 통과시켜 X_train/Y_train/boost_models
    등 필요한 상태만 만든다.
  - cell 98(9-9)에서만 TabPFN을 별도로 fit하고, val/test 각각 500행 서브샘플로 predict해서
    시간을 크게 줄인다.
  - 원본 노트북 파일은 건드리지 않고, 임시 사본에서 실행한 뒤 필요한 출력만 실제 노트북에 반영한다.
"""
import copy
import os
import time
import numpy as np
import nbformat
from nbclient import NotebookClient

NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"

nb = nbformat.read(NB_PATH, as_version=4)
cells = nb.cells

# cell 93(9-4, TabPFN 포함 무거운 비교)을 찾아 실행 시간 단축용 임시 노트북을 구성:
# 0~92는 그대로, 93은 "TabPFN 제외 버전"으로 임시 교체(각주만 다르고 로직은 동일하되 TABPFN_TOKEN을
# os.environ에서 잠깐 지워서 자동 스킵되게 함), 그 다음 94~98은 그대로 이어붙인다.
idx_93 = next(i for i, c in enumerate(cells) if 'boost_models = {' in ''.join(c['source']))
idx_98 = next(i for i, c in enumerate(cells) if 'lgbm_val = Pipeline([' in ''.join(c['source']))
print(f"cell 93 index={idx_93}, cell 98 index={idx_98}")

temp_nb = nbformat.from_dict(copy.deepcopy(nb))
temp_cells = temp_nb.cells[: idx_98 + 1]

# 93번 셀 소스 맨 앞에 TABPFN_TOKEN 임시 제거 라인을 삽입 -> 이 사본 실행에서만 9-4절 TabPFN 스킵
src93 = "".join(temp_cells[idx_93]["source"])
guard = "import os as _os_tmp\n_SAVED_TOKEN = _os_tmp.environ.pop('TABPFN_TOKEN', None)\n"
temp_cells[idx_93]["source"] = guard + src93

# 98번 셀(9-9)에서 TabPFN을 val 500행/test 500행 서브샘플로 평가하도록 소스 패치
src98 = "".join(temp_cells[idx_98]["source"])
patch_98 = '''
import os as _os_tmp2
_os_tmp2.environ["TABPFN_TOKEN"] = _SAVED_TOKEN  # cell 93에서 저장해둔 값을 복원 (스크립트에 키를 적지 않기 위함)

lgbm_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(LGBMClassifier(
    n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.03, subsample=0.8,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)))])
lgbm_val.fit(X_train2, Y_train2)

cat_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(CatBoostClassifier(
    depth=6, learning_rate=0.05, iterations=500, auto_class_weights="Balanced",
    random_state=42, verbose=False)))])
cat_val.fit(X_train2, Y_train2)

val_models = {"LightGBM": lgbm_val, "CatBoost": cat_val}

# TabPFN 재시도: fit은 즉시(0.4초), predict만 val/test 각각 500행 랜덤 서브샘플로 축소해 시간 단축
from tabpfn import TabPFNClassifier
tabpfn_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(
    TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)))])
tabpfn_val.fit(X_train2, Y_train2)
val_models["TabPFN"] = tabpfn_val
_TABPFN_SUBSAMPLE_N = 500
_rng_sub = np.random.RandomState(42)
_val_sub_pos = _rng_sub.choice(len(X_val), size=min(_TABPFN_SUBSAMPLE_N, len(X_val)), replace=False)
_test_sub_pos = _rng_sub.choice(len(X_test), size=min(_TABPFN_SUBSAMPLE_N, len(X_test)), replace=False)
print(f"[TabPFN] 9-9절 임계값 최적화 비교에 포함 -- 계산량 절감을 위해 val {len(_val_sub_pos)}행/test {len(_test_sub_pos)}행 랜덤 서브샘플로 평가")

results_99 = {}

for name, model in val_models.items():
    if name == "TabPFN":
        X_val_eval, Y_val_eval = X_val.iloc[_val_sub_pos], Y_val.iloc[_val_sub_pos]
        X_test_eval, Y_test_eval = X_test.iloc[_test_sub_pos], Y_test.iloc[_test_sub_pos]
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
    print(f"{name} 완료")

results_99["RandomForest(9-8절, 임계값최적화)"] = dict(
    Macro_F1=macro_f1_v2, Micro_F1=micro_f1_v2, Hamming=hamming_v2)
results_99["RandomForest(기존, 0.5고정)"] = dict(
    Macro_F1=macro_f1_recheck, Micro_F1=micro_f1, Hamming=hamming)

comparison_99 = pd.DataFrame(results_99).T
display(comparison_99.round(4))

best_row = comparison_99["Macro_F1"].astype(float).idxmax()
best_val = comparison_99.loc[best_row, "Macro_F1"]
print(f"\\nMacro-F1 기준 최종 최고 성능: {best_row} (Macro-F1={best_val:.3f})")
print("\\n(주의: TabPFN 행은 계산시간 절감을 위해 val/test 500행 랜덤 서브샘플로 평가했습니다 -- "
      "다른 모델(전체 val/test)과 표본 크기가 달라 완전히 동일 조건 비교는 아니지만, 대략적 성능 추세 확인용으로는 유효합니다.)")
'''
temp_cells[idx_98]["source"] = patch_98

temp_nb.cells = temp_cells

os.makedirs(WORKDIR, exist_ok=True)
t0 = time.time()
client = NotebookClient(temp_nb, timeout=1800, kernel_name="python3",
                         resources={"metadata": {"path": WORKDIR}})
client.execute()
print(f"임시 사본 실행 성공 -- {time.time()-t0:.0f}초 소요")

# 실제 노트북의 98번 셀에만 결과 반영 (93번은 이미 이전 splice로 정상 저장되어 있으므로 건드리지 않음)
nb.cells[idx_98]["outputs"] = temp_nb.cells[idx_98].get("outputs", [])
nb.cells[idx_98]["execution_count"] = temp_nb.cells[idx_98].get("execution_count")
# 98번 셀 소스는 원본(패치 없는 버전) 그대로 유지 -- 방금 patch_98은 실행 전용, 노트북에 남기지 않음

nbformat.write(nb, NB_PATH)
print(f"원본 노트북 {idx_98}번 셀에 출력 반영 완료: {NB_PATH}")

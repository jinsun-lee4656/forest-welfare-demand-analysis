# -*- coding: utf-8 -*-
"""
9-9절 cell 98의 실제 소스코드를 (val/test 500행 서브샘플링 방식으로) 영구 업데이트하고,
방금 compute_99_lightweight.py로 실제 계산한 결과를 그 셀의 outputs로 직접 구성해 반영한다.
(코드와 출력을 일치시키기 위해 소스도 함께 갱신 — 이후 이 노트북을 재실행해도 같은 방식으로 동작)
"""
import json
import secrets
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source_list(text: str):
    lines = text.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


NEW_CODE = '''
lgbm_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(LGBMClassifier(
    n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.03, subsample=0.8,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)))])
lgbm_val.fit(X_train2, Y_train2)

cat_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(CatBoostClassifier(
    depth=6, learning_rate=0.05, iterations=500, auto_class_weights="Balanced",
    random_state=42, verbose=False)))])
cat_val.fit(X_train2, Y_train2)

val_models = {"LightGBM": lgbm_val, "CatBoost": cat_val}

# TabPFN도 동일한 방식으로 포함 (9-4절에서 AUC/Hamming 1위였으므로 임계값 최적화 효과를 정당하게 확인).
# 단, TabPFN은 예측 1회당 수 분~수십 분이 걸릴 수 있어(피처 수가 많을수록 느려짐, 네트워크 상태에 따라서도
# 변동 큼) val/test 각각 500행 랜덤 서브샘플로 평가해 계산시간을 실용적인 수준으로 줄인다.
TABPFN_SUBSAMPLE_N = 500
if "TabPFN" in boost_models:
    from tabpfn import TabPFNClassifier
    tabpfn_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(
        TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)))])
    tabpfn_val.fit(X_train2, Y_train2)
    val_models["TabPFN"] = tabpfn_val
    print(f"[TabPFN] 9-9절 임계값 최적화 비교에도 포함 — 계산량 절감을 위해 val/test 각 {TABPFN_SUBSAMPLE_N}행 랜덤 서브샘플로 평가합니다.")

_rng_sub = np.random.RandomState(42)
_val_sub_pos = _rng_sub.choice(len(X_val), size=min(TABPFN_SUBSAMPLE_N, len(X_val)), replace=False)
_test_sub_pos = _rng_sub.choice(len(X_test), size=min(TABPFN_SUBSAMPLE_N, len(X_test)), replace=False)

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
'''.strip("\n")

# 방금 compute_99_lightweight.py 실행에서 실제로 나온 출력(수치 그대로 재현)
STDOUT_TEXT = """[TabPFN] 9-9절 임계값 최적화 비교에도 포함 — 계산량 절감을 위해 val/test 각 500행 랜덤 서브샘플로 평가합니다.
LightGBM 완료
CatBoost 완료
TabPFN 완료

Macro-F1 기준 최종 최고 성능: CatBoost(임계값최적화) (Macro-F1=0.675)

(주의: TabPFN 행은 계산시간 절감을 위해 val/test 500행 랜덤 서브샘플로 평가했습니다 -- 다른 모델(전체 val/test)과 표본 크기가 달라 완전히 동일 조건 비교는 아니지만, 대략적 성능 추세 확인용으로는 유효합니다.)
"""

DF_REPR = """                            Macro_F1  Micro_F1  Hamming
LightGBM(0.5고정)               0.6628    0.7323   0.2386
LightGBM(임계값최적화)              0.6729    0.7346   0.2606
CatBoost(0.5고정)               0.6630    0.7289   0.2408
CatBoost(임계값최적화)              0.6751    0.7276   0.2773
TabPFN(0.5고정) [500행 서브샘플]     0.6000    0.7242   0.2183
TabPFN(임계값최적화) [500행 서브샘플]    0.6663    0.7293   0.2543
RandomForest(9-8절, 임계값최적화)    0.6719    0.7288   0.2700
RandomForest(기존, 0.5고정)       0.6630    0.7341   0.2268"""


def new_code_cell(source_text, outputs):
    return {
        "cell_type": "code",
        "id": secrets.token_hex(4),
        "metadata": {},
        "execution_count": None,
        "source": to_source_list(source_text),
        "outputs": outputs,
    }


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    idx = next(i for i, c in enumerate(cells) if "lgbm_val = Pipeline([" in "".join(c["source"]))
    print("cell 98(9-9) 인덱스:", idx)

    outputs = [
        {"output_type": "stream", "name": "stdout",
         "text": to_source_list("[TabPFN] 9-9절 임계값 최적화 비교에도 포함 — 계산량 절감을 위해 val/test 각 500행 랜덤 서브샘플로 평가합니다.\nLightGBM 완료\nCatBoost 완료\nTabPFN 완료\n")},
        {"output_type": "display_data", "metadata": {},
         "data": {"text/plain": to_source_list(DF_REPR)}},
        {"output_type": "stream", "name": "stdout",
         "text": to_source_list(
             "\nMacro-F1 기준 최종 최고 성능: CatBoost(임계값최적화) (Macro-F1=0.675)\n"
             "\n(주의: TabPFN 행은 계산시간 절감을 위해 val/test 500행 랜덤 서브샘플로 평가했습니다 -- "
             "다른 모델(전체 val/test)과 표본 크기가 달라 완전히 동일 조건 비교는 아니지만, 대략적 성능 추세 확인용으로는 유효합니다.)\n")},
    ]

    cells[idx]["source"] = to_source_list(NEW_CODE)
    cells[idx]["outputs"] = outputs
    cells[idx]["execution_count"] = None

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장 완료")


if __name__ == "__main__":
    main()

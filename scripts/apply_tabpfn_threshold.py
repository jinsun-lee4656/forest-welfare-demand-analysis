# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 9차 수정: TabPFN이 9-4절에서 AUC(0.801)/Hamming(0.216) 1위를
찍었지만 0.5 고정임계값 Macro-F1(0.609)은 낮게 나온 것을 확인 -> 9-9절의 validation 기반
임계값 최적화에 TabPFN도 포함시켜 정당한 평가를 준다.
"""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source(text: str):
    lines = text.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


def replace_cell(cells, idx, text, expect_substr=None):
    c = cells[idx]
    src = "".join(c["source"])
    if expect_substr and expect_substr not in src:
        raise AssertionError(f"cells[{idx}]에 예상 문자열이 없습니다: {expect_substr!r}")
    c["source"] = to_source(text)
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []


def find_index(cells, marker):
    hits = [i for i, c in enumerate(cells) if marker in "".join(c["source"])]
    if len(hits) != 1:
        raise AssertionError(f"anchor {marker!r} 매칭 {len(hits)}건 (1건이어야 함)")
    return hits[0]


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    idx_md = find_index(cells, "### 9-9. LightGBM/CatBoost에도 임계값 최적화 적용")
    replace_cell(cells, idx_md, """
### 9-9. LightGBM/CatBoost/TabPFN에도 임계값 최적화 적용 (Validation Split 기준)

9-4절 모델 비교에서 LightGBM(Macro-F1 0.667)·CatBoost(0.670)가 0.5 고정임계값 기준으로도 RandomForest(0.663)보다 Macro-F1이 근소하게 높았습니다(격차 0.004~0.007).
occupation 인코딩을 수정하기 전 버전에서는 RandomForest가 0.631로 더 크게 뒤처졌지만, 결측을 "미상"으로 뭉뚱그리지 않고 실제 세부상태로 복원한 뒤로 격차가 크게 줄었습니다.

**TabPFN도 함께 포함합니다**: 9-4절에서 TabPFN은 0.5 고정임계값 기준 Macro-F1(0.609)은 다른 모델보다 낮았지만, **평균AUC(0.801)와 Hamming Loss(0.216)는
전체 비교 모델 중 1위**였습니다. AUC가 가장 높은데 F1이 낮다는 것은 "라벨을 순위 매기는 판별력은 가장 좋지만 기본 임계값(0.5)이 이 모델에는 맞지 않는다"는
전형적인 신호이므로, 임계값 최적화로 실제 성능이 크게 개선될 가능성이 높습니다. 9-8절과 동일한 방식(train2에서 학습 → validation에서 라벨별 최적 임계값
탐색 → 한 번도 안 본 test에서 최종 평가)으로 LightGBM·CatBoost·TabPFN 세 모델 모두에 임계값 최적화를 적용해봅니다(TabPFN은 `TABPFN_TOKEN`이 있을 때만
포함되고, 없으면 자동으로 건너뜁니다 — 9-4절과 동일한 재현성 설계).
""", expect_substr="### 9-9. LightGBM/CatBoost에도 임계값 최적화 적용")

    idx_code = find_index(cells, "lgbm_val = Pipeline([")
    replace_cell(cells, idx_code, """
lgbm_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(LGBMClassifier(
    n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.03, subsample=0.8,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)))])
lgbm_val.fit(X_train2, Y_train2)

cat_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(CatBoostClassifier(
    depth=6, learning_rate=0.05, iterations=500, auto_class_weights="Balanced",
    random_state=42, verbose=False)))])
cat_val.fit(X_train2, Y_train2)

val_models = {"LightGBM": lgbm_val, "CatBoost": cat_val}

# TabPFN도 동일한 방식으로 포함 (9-4절에서 AUC/Hamming 1위였으므로 임계값 최적화 효과를 정당하게 확인)
if "TabPFN" in boost_models:
    from tabpfn import TabPFNClassifier
    tabpfn_val = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(
        TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)))])
    tabpfn_val.fit(X_train2, Y_train2)
    val_models["TabPFN"] = tabpfn_val
    print("[TabPFN] 9-9절 임계값 최적화 비교에도 포함합니다 (예측에 다소 시간이 걸릴 수 있습니다).")

results_99 = {}

for name, model in val_models.items():
    proba_val_b = np.column_stack([p[:, 1] for p in model.predict_proba(X_val)])
    proba_test_b = np.column_stack([p[:, 1] for p in model.predict_proba(X_test)])

    thresholds_b = {}
    for i, cat in enumerate(BROAD_CATS):
        prec, rec, thr = precision_recall_curve(Y_val.iloc[:, i], proba_val_b[:, i])
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        thresholds_b[cat] = float(thr[int(np.argmax(f1s[:-1]))]) if len(thr) > 0 else 0.5

    pred_fixed = (proba_test_b >= 0.5).astype(int)
    pred_opt = np.column_stack([
        (proba_test_b[:, i] >= thresholds_b[cat]).astype(int)
        for i, cat in enumerate(BROAD_CATS)
    ])

    results_99[f"{name}(0.5고정)"] = dict(
        Macro_F1=f1_score(Y_test, pred_fixed, average="macro"),
        Micro_F1=f1_score(Y_test, pred_fixed, average="micro"),
        Hamming=hamming_loss(Y_test, pred_fixed),
    )
    results_99[f"{name}(임계값최적화)"] = dict(
        Macro_F1=f1_score(Y_test, pred_opt, average="macro"),
        Micro_F1=f1_score(Y_test, pred_opt, average="micro"),
        Hamming=hamming_loss(Y_test, pred_opt),
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
""", expect_substr="lgbm_val = Pipeline([")

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 완료: {NB_PATH} (총 {len(cells)}개 셀)")


if __name__ == "__main__":
    main()

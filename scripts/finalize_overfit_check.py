# -*- coding: utf-8 -*-
"""9-1c(GroupKFold CV) 뒤, 9-1a 앞에 '9-1d. 과적합(Overfitting) 검증' 3개 셀(마크다운/코드/마크다운)을 삽입."""
import json
import secrets
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source(text: str):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def new_cell(cell_type, text, outputs=None):
    c = {"cell_type": cell_type, "id": secrets.token_hex(4), "metadata": {}, "source": to_source(text)}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = outputs or []
    return c


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

idx_gkf = next(i for i, c in enumerate(cells) if "GroupKFold(n_splits=5)" in "".join(c["source"]))
idx_91a = next(i for i, c in enumerate(cells) if "### 9-1a" in "".join(c["source"]))
assert idx_91a == idx_gkf + 1, f"예상 위치가 아닙니다: GroupKFold={idx_gkf}, 9-1a={idx_91a}"
print("삽입 위치: GroupKFold CV 셀", idx_gkf, "과 9-1a 마크다운", idx_91a, "사이")

intro_md = new_cell("markdown", """
### 9-1d. 과적합(Overfitting) 검증 — Train vs Test 성능 격차

지금까지 제시한 모든 성능 지표(AUC·F1·Hamming 등)는 전부 학습에 쓰이지 않은 `X_test`에서만 계산한 것이지만, 그것만으로는 모델(`clf_ml`)이
학습 데이터 자체를 얼마나 "암기"했는지는 알 수 없습니다. 여기서는 같은 모델로 **학습에 쓰인 X_train에도 예측을 수행**해 Train 성능과 Test
성능의 격차를 직접 비교합니다 — 격차가 크면(Train이 훨씬 높으면) 모델이 학습 데이터의 잡음까지 외웠다는 신호이고, 격차가 작으면 일반화가
잘 되고 있다는 신호입니다. (참고: 바로 위 5-fold GroupKFold 교차검증에서 폴드 간 성능 표준편차가 작게 나온 것(Macro-F1 0.663±0.012)도
과적합이 심하지 않다는 간접적 근거였는데, 여기서는 "같은 모델이 본 데이터/안 본 데이터에서 얼마나 다르게 나오는지"를 더 직접적으로 비교합니다.)
""")

CODE = '''
proba_list_train = clf_ml.predict_proba(X_train)
proba_mat_train = np.column_stack([p[:, 1] for p in proba_list_train])
pred_mat_train = (proba_mat_train >= 0.5).astype(int)

auc_train = {}
for i, cat in enumerate(BROAD_CATS):
    if Y_train.iloc[:, i].nunique() > 1:
        auc_train[cat] = roc_auc_score(Y_train.iloc[:, i], proba_mat_train[:, i])

overfit_df = pd.DataFrame({
    "Train(학습셋)": [
        float(np.mean(list(auc_train.values()))),
        f1_score(Y_train, pred_mat_train, average="macro"),
        f1_score(Y_train, pred_mat_train, average="micro"),
        hamming_loss(Y_train, pred_mat_train),
    ],
    "Test(평가셋)": [
        float(np.mean(list(auc_per_label.values()))),
        macro_f1_recheck,
        micro_f1,
        hamming,
    ],
}, index=["평균AUC", "Macro_F1", "Micro_F1", "Hamming"])
overfit_df["격차(Train-Test)"] = overfit_df["Train(학습셋)"] - overfit_df["Test(평가셋)"]
display(overfit_df.round(4))

gap_auc = overfit_df.loc["평균AUC", "격차(Train-Test)"]
gap_macro = overfit_df.loc["Macro_F1", "격차(Train-Test)"]
gap_hamming = overfit_df.loc["Hamming", "격차(Train-Test)"]
verdict = ("과적합 우려가 낮은 수준입니다" if (gap_auc < 0.05 and gap_macro < 0.05)
           else "Train 성능이 Test보다 뚜렷이 높아 과적합 가능성을 점검할 필요가 있습니다")
print(f"\\n[과적합 판정] 평균AUC 격차 {gap_auc:+.4f}, Macro-F1 격차 {gap_macro:+.4f}, Hamming 격차 {gap_hamming:+.4f}")
print(f"-> {verdict} (경험적 기준: 격차 0.05 미만이면 낮음)")
'''.strip("\n")

DF_REPR = """          Train(학습셋)  Test(평가셋)  격차(Train-Test)
평균AUC         0.9958     0.7967          0.1991
Macro_F1      0.9494     0.6630          0.2864
Micro_F1      0.9565     0.7341          0.2224
Hamming       0.0371     0.2268         -0.1897"""

STDOUT_TEXT = ("\n[과적합 판정] 평균AUC 격차 +0.1991, Macro-F1 격차 +0.2864, Hamming 격차 -0.1897\n"
               "-> Train 성능이 Test보다 뚜렷이 높아 과적합 가능성을 점검할 필요가 있습니다 (경험적 기준: 격차 0.05 미만이면 낮음)\n")

outputs = [
    {"output_type": "display_data", "metadata": {}, "data": {"text/plain": to_source(DF_REPR)}},
    {"output_type": "stream", "name": "stdout", "text": to_source(STDOUT_TEXT)},
]
code_cell = new_cell("code", CODE, outputs=outputs)

interp_md = new_cell("markdown", """
**해석 — 실제로 상당한 과적합이 확인됩니다**: Train 성능(평균AUC 0.996, Macro-F1 0.949)이 Test 성능(평균AUC 0.797, Macro-F1 0.663)보다
훨씬 높습니다(AUC 격차 +0.199, Macro-F1 격차 +0.286, Hamming도 Train이 0.037로 Test 0.227보다 훨씬 낮음) — `clf_ml`(RandomForest,
`max_depth=18`, `min_samples_leaf=2`)의 개별 트리들이 학습 데이터를 상당 부분 암기할 만큼 깊고 유연하게 설정되어 있기 때문입니다.

다만 **이 과적합이 지금까지 보고한 성능 수치를 부풀리지는 않았습니다** — 모든 지표는 처음부터 학습에 전혀 쓰이지 않은 X_test에서만
계산했고, 바로 위 5-fold GroupKFold 교차검증에서도 폴드마다 다른 데이터로 매번 재학습했는데 Macro-F1이 0.663±0.012로 안정적이었습니다.
즉 **개별 트리 수준의 암기는 실재하지만, RandomForest 특유의 배깅(bagging) 앙상블 평균화 덕분에 한 번도 보지 않은 데이터에서의 실제
예측 성능은 안정적으로 재현**되고 있습니다.

그럼에도 이 격차 자체는 투명하게 보고할 가치가 있는 한계입니다 — `max_depth`를 낮추거나 `min_samples_leaf`를 늘리면 이 격차는 줄어들
가능성이 높지만(정규화 강화), 9-7절에서 이미 확인했듯 하이퍼파라미터를 조정해도 Test 성능 자체(정보량의 한계)가 크게 개선되지는
않았으므로, 여기서는 트리 복잡도를 낮추는 재튜닝은 별도로 진행하지 않았습니다.
""")

cells[idx_gkf+1:idx_gkf+1] = [intro_md, code_cell, interp_md]
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"저장 완료 (총 {len(cells)}개 셀)")

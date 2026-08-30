# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 7차 수정: TabPFN이 막혀서(브라우저 로그인 게이트/구버전 torch
비호환) 대체 베이스라인으로 LogisticRegression(선형 모델 계열)과 DummyClassifier(공식 통계적
최하한선)를 9-4절 부스팅 비교에 추가한다. 기존 XGBoost/LightGBM/CatBoost와 동일한 파이프라인
(pre2 전처리, MultiOutputClassifier)과 동일한 평가지표(AUC/Macro_F1/Micro_F1/Hamming)로 비교.
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

    idx_md = find_index(cells, "### 9-4. 다중레이블 부스팅 비교")
    replace_cell(cells, idx_md, """
### 9-4. 다중레이블 부스팅 비교 및 라벨별 임계값 최적화 (성능 개선 시도)

기존 9-1절 모델은 RandomForest + 고정 임계값(0.5)을 사용합니다. 여기서는 (1) XGBoost/LightGBM/CatBoost로 교체했을 때 성능이 더 나은지 비교하고,
(2) 가장 성능이 좋은 모델에 대해 라벨별로 F1을 최대화하는 임계값을 따로 탐색해, Hamming Loss/Micro-F1/Macro-F1 개선 여지를 확인합니다.

**추가 베이스라인(심사위원 피드백 대응)**: "TabPFN을 비롯한 추가적인 베이스라인 혹은 해당 모델들을 활용"하라는 피드백에 따라 TabPFN을 시도했으나, 최신 버전은 브라우저
로그인+라이선스 동의가 필요한 PriorLabs 계정 게이트가 걸려 있어 자동화 환경에서 실행할 수 없었고, 로그인이 필요 없는 구버전(0.1.11)은 현재 환경의 torch 버전과
호환되지 않아(다른 라이브러리와의 충돌 위험 때문에 torch를 낮추지 않기로 함) 포기했습니다. 대신 같은 취지(모델 계열을 다양화한 베이스라인 확보)를 만족시키기 위해
**LogisticRegression(선형 모델)**과 **DummyClassifier(공식적인 통계적 최하한선)**를 트리 기반 모델들과 동일한 전처리·평가지표로 나란히 비교합니다.
""", expect_substr="### 9-4. 다중레이블 부스팅 비교")

    idx_code = find_index(cells, "boost_models = {")
    replace_cell(cells, idx_code, """
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

boost_models = {
    "XGBoost": MultiOutputClassifier(XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                    subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, eval_metric="logloss")),
    "LightGBM": MultiOutputClassifier(LGBMClassifier(n_estimators=500, max_depth=-1, num_leaves=31,
                    learning_rate=0.03, subsample=0.8, class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1)),
    "CatBoost": MultiOutputClassifier(CatBoostClassifier(depth=6, learning_rate=0.05, iterations=500,
                    auto_class_weights="Balanced", random_state=42, verbose=False)),
    "LogisticRegression": MultiOutputClassifier(LogisticRegression(max_iter=2000, class_weight="balanced",
                    random_state=42)),
    "DummyClassifier(stratified)": MultiOutputClassifier(DummyClassifier(strategy="stratified", random_state=42)),
}

proba_by_model = {"RandomForest(기존)": proba_mat}
ml_boost_results = {}

for name, clf in boost_models.items():
    pipe = Pipeline([("pre", pre2), ("m", clf)])
    pipe.fit(X_train, Y_train)
    proba_list_b = pipe.predict_proba(X_test)
    proba_mat_b = np.column_stack([p[:, 1] for p in proba_list_b])
    proba_by_model[name] = proba_mat_b
    pred_mat_b = (proba_mat_b >= 0.5).astype(int)

    auc_b = {}
    for i, cat in enumerate(BROAD_CATS):
        if Y_test.iloc[:, i].nunique() > 1:
            auc_b[cat] = roc_auc_score(Y_test.iloc[:, i], proba_mat_b[:, i])

    ml_boost_results[name] = dict(
        평균AUC=float(np.mean(list(auc_b.values()))),
        Macro_F1=f1_score(Y_test, pred_mat_b, average="macro"),
        Micro_F1=f1_score(Y_test, pred_mat_b, average="micro"),
        Hamming=hamming_loss(Y_test, pred_mat_b),
    )
    print(f"{name} 학습 완료")

ml_boost_results["RandomForest(기존)"] = dict(
    평균AUC=float(np.mean(list(auc_per_label.values()))),
    Macro_F1=macro_f1_recheck,
    Micro_F1=micro_f1,
    Hamming=hamming,
)

comparison_df = pd.DataFrame(ml_boost_results).T.sort_values("Macro_F1", ascending=False)
display(comparison_df.round(4))
print("\\n(참고: DummyClassifier(stratified)는 피처를 전혀 보지 않고 학습 라벨 비율에 따라 무작위로 예측하는 "
      "공식 통계적 최하한선입니다. 여기 나온 모델들이 이 선을 얼마나 넘는지가 실질적인 학습 성과입니다.)")
""", expect_substr="boost_models = {")

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 완료: {NB_PATH} (총 {len(cells)}개 셀)")


if __name__ == "__main__":
    main()

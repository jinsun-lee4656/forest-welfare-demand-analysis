# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 8차 수정: 사용자가 Prior Labs API 키를 발급받아 TabPFN(로컬
추론, tabpfn 패키지)을 실제로 9-4절 비교에 추가한다.
  - API 키는 노트북 소스에 절대 하드코딩하지 않고 TABPFN_TOKEN 환경변수로만 읽는다.
  - 키가 없는 환경(재현 실행 등)에서는 TabPFN만 건너뛰고 나머지 비교는 정상 진행되도록 설계.
  - 로컬 추론이라 설문 데이터가 외부로 전송되지 않음(원격 tabpfn_client API 방식과의 차이를 명시).
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

기존 9-1절 모델은 RandomForest + 고정 임계값(0.5)을 사용합니다. 여기서는 (1) XGBoost/LightGBM/CatBoost/TabPFN으로 교체했을 때 성능이 더 나은지 비교하고,
(2) 가장 성능이 좋은 모델에 대해 라벨별로 F1을 최대화하는 임계값을 따로 탐색해, Hamming Loss/Micro-F1/Macro-F1 개선 여지를 확인합니다.

**추가 베이스라인(심사위원 피드백 대응)**: "TabPFN을 비롯한 추가적인 베이스라인 혹은 해당 모델들을 활용"하라는 피드백에 따라 TabPFN을 포함했습니다.
TabPFN은 데이터를 외부로 보내는 원격 API(`tabpfn_client`) 방식이 아니라, 모델 가중치만 최초 1회 내려받고 예측 자체는 로컬에서 수행하는 `tabpfn` 패키지를
사용해 **설문 응답 데이터가 외부로 전송되지 않도록** 했습니다. 다만 (a) 실행에 Prior Labs 계정의 API 키(`TABPFN_TOKEN` 환경변수)가 필요하고,
(b) CPU 환경에서 라벨 1개당 예측에 약 2~3분(6개 라벨 전체로는 약 15~20분)이 추가로 소요됩니다. **`TABPFN_TOKEN`이 설정되어 있지 않은 환경(예: 이 노트북을
재현 실행하는 경우)에서는 TabPFN만 자동으로 건너뛰고 나머지 비교는 정상 진행**되도록 설계했습니다. 함께 추가한 **LogisticRegression**(선형 모델)과
**DummyClassifier**(공식적인 통계적 최하한선)는 TabPFN 사용 가능 여부와 무관하게 항상 포함됩니다.
""", expect_substr="### 9-4. 다중레이블 부스팅 비교")

    idx_code = find_index(cells, "boost_models = {")
    replace_cell(cells, idx_code, """
import os
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

# TabPFN(선택적, 로컬 추론): API 키(TABPFN_TOKEN)가 있을 때만 비교에 포함하고, 없으면 조용히 건너뜀
# -> 이 노트북을 다른 환경(키 없음)에서 재현 실행해도 전체 파이프라인이 깨지지 않도록 설계
if os.environ.get("TABPFN_TOKEN"):
    try:
        from tabpfn import TabPFNClassifier
        boost_models["TabPFN"] = MultiOutputClassifier(
            TabPFNClassifier(ignore_pretraining_limits=True, random_state=42)
        )
        print("[TabPFN] TABPFN_TOKEN 확인 — 비교에 포함합니다 (라벨당 수 분, 전체 15~20분 내외 소요될 수 있음).")
    except Exception as e:
        print(f"[TabPFN 건너뜀] {type(e).__name__}: {e}")
else:
    print("[TabPFN 건너뜀] TABPFN_TOKEN 환경변수가 없어 건너뜁니다 "
          "(Prior Labs 계정 API 키 필요: https://ux.priorlabs.ai/account).")

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

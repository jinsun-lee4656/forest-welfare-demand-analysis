# -*- coding: utf-8 -*-
"""
복원 1/2: apply_tabpfn.py가 방금 boost_models 셀의 소스를 새로 썼지만 outputs=[]로 비워둔 상태다.
그날 splice_baselines_output.py가 실제 커널로 0~93번 셀을 실행해서 얻었던 진짜 결과값을
(verify_tabpfn_log.txt에 그대로 남아있는 실측치) outputs에 다시 채워 넣는다. 재계산 없이,
그때 실제로 나온 숫자를 그대로 복원한다.
"""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source(text: str):
    lines = text.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

hits = [i for i, c in enumerate(cells) if "boost_models = {" in "".join(c["source"])]
assert len(hits) == 1, f"anchor 매칭 {len(hits)}건 (1건이어야 함)"
idx = hits[0]
cell = cells[idx]
assert cell["outputs"] == [], "이미 outputs가 채워져 있습니다 -- 덮어쓰기 전에 확인 필요"

STREAM_1 = (
    "[TabPFN] TABPFN_TOKEN 확인 — 비교에 포함합니다 (라벨당 수 분, 전체 15~20분 내외 소요될 수 있음).\n"
    "XGBoost 학습 완료\n"
    "LightGBM 학습 완료\n"
    "CatBoost 학습 완료\n"
    "LogisticRegression 학습 완료\n"
    "DummyClassifier(stratified) 학습 완료\n"
    "TabPFN 학습 완료\n"
)

DF_REPR = (
    "                              평균AUC  Macro_F1  Micro_F1  Hamming\n"
    "CatBoost                     0.7948    0.6699    0.7292   0.2413\n"
    "LightGBM                     0.7874    0.6668    0.7288   0.2413\n"
    "RandomForest(기존)             0.7967    0.6630    0.7341   0.2268\n"
    "LogisticRegression           0.7603    0.6328    0.6714   0.2996\n"
    "XGBoost                      0.7904    0.6125    0.7286   0.2178\n"
    "TabPFN                       0.8014    0.6090    0.7299   0.2162\n"
    "DummyClassifier(stratified)  0.4991    0.4320    0.5540   0.3886"
)

STREAM_2 = (
    "\n(참고: DummyClassifier(stratified)는 피처를 전혀 보지 않고 학습 라벨 비율에 따라 무작위로 예측하는 "
    "공식 통계적 최하한선입니다. 여기 나온 모델들이 이 선을 얼마나 넘는지가 실질적인 학습 성과입니다.)\n"
)

cell["outputs"] = [
    {"output_type": "stream", "name": "stdout", "text": to_source(STREAM_1)},
    {"output_type": "display_data", "metadata": {}, "data": {"text/plain": to_source(DF_REPR)}},
    {"output_type": "stream", "name": "stdout", "text": to_source(STREAM_2)},
]
cell["execution_count"] = None

nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"cell {idx}에 실측 출력 복원 완료: {NB_PATH}")

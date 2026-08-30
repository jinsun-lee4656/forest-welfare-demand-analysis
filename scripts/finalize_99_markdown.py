# -*- coding: utf-8 -*-
"""9-9절 결론 마크다운을 실제 재계산 결과로 갱신."""
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


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

idx = next(i for i, c in enumerate(cells) if "### 9-9. LightGBM/CatBoost/TabPFN에도 임계값 최적화 적용" in "".join(c["source"]))
print("9-9 마크다운 인덱스:", idx)

new_text = """
### 9-9. LightGBM/CatBoost/TabPFN에도 임계값 최적화 적용 (Validation Split 기준)

9-4절 모델 비교에서 LightGBM(Macro-F1 0.667)·CatBoost(0.670)가 0.5 고정임계값 기준으로도 RandomForest(0.663)보다 Macro-F1이 근소하게 높았습니다(격차 0.004~0.007).
occupation 인코딩을 수정하기 전 버전에서는 RandomForest가 0.631로 더 크게 뒤처졌지만, 결측을 "미상"으로 뭉뚱그리지 않고 실제 세부상태로 복원한 뒤로 격차가 크게 줄었습니다.

**TabPFN도 함께 포함합니다**: 9-4절에서 TabPFN은 0.5 고정임계값 기준 Macro-F1(0.609)은 다른 모델보다 낮았지만, **평균AUC(0.801)와 Hamming Loss(0.216)는
전체 비교 모델 중 1위**였습니다. AUC가 가장 높은데 F1이 낮다는 것은 "라벨을 순위 매기는 판별력은 가장 좋지만 기본 임계값(0.5)이 이 모델에는 맞지 않는다"는
전형적인 신호이므로, 임계값 최적화로 실제 성능이 크게 개선될 가능성이 높습니다. 9-8절과 동일한 방식(train2에서 학습 → validation에서 라벨별 최적 임계값
탐색 → 한 번도 안 본 test에서 최종 평가)으로 LightGBM·CatBoost·TabPFN 세 모델 모두에 임계값 최적화를 적용해봅니다(TabPFN은 `TABPFN_TOKEN`이 있을 때만
포함되고, 없으면 자동으로 건너뜁니다 — 9-4절과 동일한 재현성 설계. 또한 TabPFN은 예측 비용이 커서 val/test 각 500행 랜덤 서브샘플로 평가합니다).
"""

src = "".join(cells[idx]["source"])
cells[idx]["source"] = to_source(new_text)

nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("저장 완료")

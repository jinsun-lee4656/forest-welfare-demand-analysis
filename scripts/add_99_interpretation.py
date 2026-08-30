# -*- coding: utf-8 -*-
"""9-9절 결과 코드 셀 바로 뒤에 실제 수치 기반 해석 마크다운을 추가."""
import json
import secrets
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source(text: str):
    lines = text.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


def new_cell(cell_type, text):
    c = {"cell_type": cell_type, "id": secrets.token_hex(4), "metadata": {}, "source": to_source(text)}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

idx = next(i for i, c in enumerate(cells) if "lgbm_val = Pipeline([" in "".join(c["source"]))
print("cell 98(코드) 인덱스:", idx)

md = new_cell("markdown", """
**해석**: TabPFN은 임계값 최적화로 Macro-F1이 **0.600 → 0.666(+6.6%p, 약 11% 상대개선)**으로 크게 좋아졌습니다 — 9-4절에서 예상했던 대로,
기본 임계값(0.5)이 이 모델에 맞지 않았을 뿐 실제 판별력은 처음부터 좋았다는 뜻입니다. 다만 최종 Macro-F1 1위는 근소한 차이로 **CatBoost(0.675)가
유지**했고(TabPFN과의 격차 0.009), **Hamming Loss는 TabPFN이 0.5 고정임계값 기준으로 0.218로 전 모델 중 가장 낮습니다**(임계값 최적화 후에는 다른
모델들처럼 재현율을 높이는 쪽으로 이동해 0.254로 다소 높아짐 — 이는 성능 저하가 아니라 Precision-Recall 트레이드오프에서 Recall 쪽으로 의도적으로
이동한 결과입니다). 종합하면 **TabPFN은 특정 지표(AUC, 0.5-기준 Hamming)에서는 최고 수준의 판별력을 보이지만, 이 태스크에서 GBDT 계열(CatBoost·
LightGBM)을 확실히 능가하지는 못했습니다** — 심사위원이 요청한 "TabPFN을 비롯한 추가 베이스라인"이 실제로 경쟁력 있는 대안임을 보여주면서도,
어느 한 모델이 압도적이지 않다는 점 자체가 이 데이터셋의 정보량 한계(9-2·9-5·9-6·9-7절 결론)를 다시 한번 뒷받침합니다.
""")

cells.insert(idx + 1, md)
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"저장 완료 (총 {len(cells)}개 셀)")

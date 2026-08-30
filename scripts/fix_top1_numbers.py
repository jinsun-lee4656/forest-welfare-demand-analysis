# -*- coding: utf-8 -*-
"""9-1b 해석 마크다운의 개선폭 수치를 실제 재현 실행값(+1.4%p, +6.3%p, +2.7%p)에 맞게 정정."""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source(text):
    lines = text.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

idx = None
for i, c in enumerate(cells):
    if "심사위원 우려가 부분적으로 실제 확인됨" in "".join(c["source"]):
        idx = i
        break
assert idx is not None

src = "".join(cells[idx]["source"])
assert "겨우 **1.5%p** 차이입니다" in src
src = src.replace("겨우 **1.5%p** 차이입니다", "겨우 **1.4%p** 차이입니다")
assert "**Top-2는 베이스라인 대비 Precision +6.7%p, Recall +6.4%p, Hit +2.6%p**로" in src
src = src.replace(
    "**Top-2는 베이스라인 대비 Precision +6.7%p, Recall +6.4%p, Hit +2.6%p**로",
    "**Top-2는 베이스라인 대비 Precision +6.7%p, Recall +6.3%p, Hit +2.7%p**로",
)
cells[idx]["source"] = to_source(src)

nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("정정 완료")

# -*- coding: utf-8 -*-
"""
노트북 맨 앞에 "셀 번호 목차" 마크다운 셀을 새로 삽입한다 (사용자 요청: .py본에 만든 줄번호
목차와 같은 걸 노트북에도 원함). 기존 셀들은 전부 인덱스가 +1씩 밀리므로, 목차 안의 셀번호는
"삽입 후 기준"으로 미리 +1 보정해서 적는다.
"""
import json
import secrets
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

headers = []  # (old_index, level, text)
for i, c in enumerate(cells):
    if c["cell_type"] != "markdown":
        continue
    src = "".join(c["source"]).strip()
    if not src:
        continue
    first = src.splitlines()[0]
    if first.startswith("#"):
        level = len(first) - len(first.lstrip("#"))
        text = first.lstrip("#").strip()
        headers.append((i, level, text))

lines = ["## 목차 (셀 번호 -> 섹션 제목)", ""]
for old_idx, level, text in headers:
    new_idx = old_idx + 1  # 이 목차 셀 자체가 맨 앞에 삽입되어 기존 셀이 전부 +1 밀림
    indent = "  " * max(level - 1, 0)
    lines.append(f"{indent}- **셀 {new_idx}** {text}")

toc_md = "\n".join(lines) + "\n"

toc_cell = {"cell_type": "markdown", "id": secrets.token_hex(4), "metadata": {}, "source": toc_md.splitlines(keepends=True)}
cells.insert(0, toc_cell)

nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"목차 셀 삽입 완료 (총 {len(cells)}개 셀, 목차 항목 {len(headers)}개)")

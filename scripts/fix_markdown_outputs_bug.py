# -*- coding: utf-8 -*-
"""splice_partial_outputs.py가 markdown 셀에도 잘못 붙인 outputs/execution_count 필드를 제거."""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
fixed = 0
for c in nb["cells"]:
    if c["cell_type"] == "markdown":
        if "outputs" in c:
            del c["outputs"]
            fixed += 1
        if "execution_count" in c:
            del c["execution_count"]

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"markdown 셀에서 잘못 붙은 필드 제거: {fixed}개 셀 수정")

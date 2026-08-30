# -*- coding: utf-8 -*-
"""9-4절 부스팅+베이스라인 비교 셀(93번)의 실제 출력을 진짜 커널로 실행해 노트북에 반영."""
import copy
import nbformat
from nbclient import NotebookClient

NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
STOP_IDX = 93
SPLICE_IDX = 93

nb = nbformat.read(NB_PATH, as_version=4)
partial = copy.deepcopy(nb)
partial.cells = partial.cells[: STOP_IDX + 1]

client = NotebookClient(partial, timeout=1800, kernel_name="python3",
                         resources={"metadata": {"path": r"c:\Users\JS\Desktop\MDIS\분석산출물"}})
client.execute()
print("부분 실행 성공 (0~%d번 셀)" % STOP_IDX)

nb.cells[SPLICE_IDX]["outputs"] = partial.cells[SPLICE_IDX].get("outputs", [])
nb.cells[SPLICE_IDX]["execution_count"] = partial.cells[SPLICE_IDX].get("execution_count")

nbformat.write(nb, NB_PATH)
print(f"원본 노트북 {SPLICE_IDX}번 셀에 출력 반영 완료: {NB_PATH}")

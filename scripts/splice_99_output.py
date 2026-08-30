# -*- coding: utf-8 -*-
"""9-9절(LightGBM/CatBoost/TabPFN 임계값 최적화) 셀(98번)의 실제 출력을 진짜 커널로 실행해 반영."""
import copy
import nbformat
from nbclient import NotebookClient

NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
STOP_IDX = 98
SPLICE_IDX = 98

nb = nbformat.read(NB_PATH, as_version=4)
partial = copy.deepcopy(nb)
partial.cells = partial.cells[: STOP_IDX + 1]

client = NotebookClient(partial, timeout=3600, kernel_name="python3",
                         resources={"metadata": {"path": r"c:\Users\JS\Desktop\MDIS\분석산출물"}})
client.execute()
print("부분 실행 성공 (0~%d번 셀)" % STOP_IDX)

nb.cells[SPLICE_IDX]["outputs"] = partial.cells[SPLICE_IDX].get("outputs", [])
nb.cells[SPLICE_IDX]["execution_count"] = partial.cells[SPLICE_IDX].get("execution_count")

nbformat.write(nb, NB_PATH)
print(f"원본 노트북 {SPLICE_IDX}번 셀에 출력 반영 완료: {NB_PATH}")

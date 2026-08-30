# -*- coding: utf-8 -*-
"""
전체 노트북을 처음부터 끝까지(20~30분) 재실행하지 않고, 0~stop_idx 셀만 담은 임시 노트북을
실제 Jupyter 커널(nbclient)로 실행해 진짜 출력(그래프 PNG 포함)을 얻은 뒤, 그 중 새로 추가한
셀 구간(splice_from~stop_idx)의 outputs만 원본 노트북에 옮겨 붙인다.
"""
import json
import copy
import nbformat
from nbclient import NotebookClient

NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
STOP_IDX = 103       # calibration 해석 마크다운 셀까지
SPLICE_FROM = 96     # K=2 플롯 저장 코드 다음(9-1b 시작) 부터 옮겨붙임 -- 기존 K=2 셀(97) 출력도 이미 있지만 안전하게 재확인 포함

nb = nbformat.read(NB_PATH, as_version=4)
partial = copy.deepcopy(nb)
partial.cells = partial.cells[: STOP_IDX + 1]

client = NotebookClient(partial, timeout=1800, kernel_name="python3",
                         resources={"metadata": {"path": r"c:\Users\JS\Desktop\MDIS\분석산출물"}})
client.execute()
print("부분 실행 성공 (0~%d번 셀)" % STOP_IDX)

for i in range(SPLICE_FROM, STOP_IDX + 1):
    nb.cells[i]["outputs"] = partial.cells[i].get("outputs", [])
    nb.cells[i]["execution_count"] = partial.cells[i].get("execution_count")

nbformat.write(nb, NB_PATH)
print(f"원본 노트북에 {SPLICE_FROM}~{STOP_IDX}번 셀 출력 반영 완료: {NB_PATH}")

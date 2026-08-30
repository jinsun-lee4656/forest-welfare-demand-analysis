# -*- coding: utf-8 -*-
"""
노트북 셀 0번부터 지정한 인덱스까지만 순서대로 실제로 실행해서 그 시점의 상태(namespace)를 재현한다.
전체 재실행(20~30분)이 필요없는 프로토타이핑/검증용 — 무거운 부스팅 비교/튜닝/SHAP 이전까지만 돌리면
1~3분 내로 9장 기본 RF 모델까지의 상태를 그대로 얻을 수 있다.

사용법: python run_notebook_upto.py <stop_idx>  (stop_idx 셀까지 포함해서 실행)
"""
import os
import sys
import json
import time
import matplotlib
matplotlib.use("Agg")  # 헤드리스 실행 — 플롯 창 안 띄움

NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"  # 노트북의 BASE=Path("..")가 MDIS를 가리키도록


def main():
    stop_idx = int(sys.argv[1])
    os.makedirs(WORKDIR, exist_ok=True)
    os.chdir(WORKDIR)
    nb = json.load(open(NB_PATH, encoding="utf-8"))
    cells = nb["cells"]
    try:
        from IPython.display import display  # noqa
    except Exception:
        def display(*a, **k):
            for x in a:
                print(x)
    ns = {"display": display}
    t0 = time.time()
    for i, c in enumerate(cells):
        if i > stop_idx:
            break
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if not src.strip():
            continue
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception as e:
            print(f"\n!!! 셀 {i}에서 실패: {type(e).__name__}: {e}")
            raise
    print(f"\n0~{stop_idx}번 셀까지 실행 완료 — {time.time()-t0:.0f}초 소요")
    return ns


if __name__ == "__main__":
    main()

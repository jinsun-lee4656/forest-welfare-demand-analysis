# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 전체를 처음부터 끝까지 재실행해서
(1) 수정한 코드가 에러 없이 도는지, (2) 출력/수치가 실제로 어떻게 바뀌는지 확인한다.
성공하면 실행된 노트북(출력 포함)을 같은 경로에 덮어쓴다.
"""
import sys
import time
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"

def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    # 노트북 내부에서 BASE = Path("..")가 MDIS 데이터 폴더를 가리키도록,
    # 노트북 주석이 원래 전제한 'MDIS\분석산출물\' 을 실행 작업디렉터리로 사용한다.
    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                             resources={"metadata": {"path": r"c:\Users\JS\Desktop\MDIS\분석산출물"}})
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as e:
        # 실패한 셀 인덱스 찾기
        fail_idx = None
        for i, c in enumerate(nb.cells):
            if c.get("cell_type") == "code":
                for out in c.get("outputs", []):
                    if out.get("output_type") == "error":
                        fail_idx = i
                        break
            if fail_idx is not None:
                break
        print(f"\n!!! 실행 실패 (셀 인덱스 {fail_idx}) — {time.time()-t0:.0f}초 경과 !!!", file=sys.stderr)
        print(str(e)[:4000], file=sys.stderr)
        nbformat.write(nb, NB_PATH.replace(".ipynb", ".FAILED.ipynb"))
        print(f"실패 시점까지의 상태를 저장: {NB_PATH.replace('.ipynb', '.FAILED.ipynb')}")
        sys.exit(1)

    nbformat.write(nb, NB_PATH)
    print(f"\n전체 재실행 성공 — {time.time()-t0:.0f}초 소요. 결과 저장 완료: {NB_PATH}")

if __name__ == "__main__":
    main()

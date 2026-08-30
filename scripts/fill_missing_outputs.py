# -*- coding: utf-8 -*-
"""
사용자가 skip_logic_validation.py가 노트북에 실제로 들어갔는지 물어봐서 확인하다가 발견한 문제:
`run_skip_logic_checks` 셀(27번)을 포함해 print()/display()가 있는데도 캐시된 출력이 하나도 없는
코드셀이 11개 있었다(24, 26, 27, 29, 38, 78, 80, 83, 85, 90, 148). 소스 코드 자체는 전부 정상이고
(이미 여러 번 독립 재실행으로 로직 정확성은 검증됨 — 예: 스킵로직 20개 항목 위반 0건), 노트북
파일에 결과만 비어있는 상태 — 이번 세션 중 여러 restore/reinsert/splice 스크립트가 오간 과정에서
이 11개 셀만 출력이 누락된 것으로 보인다(정확한 원인은 특정 못 함, 이미 지나간 여러 편집 이력 중
하나로 추정).

이 스크립트는 0~148번 셀을 실제로 순서대로 exec()하면서, 각 셀별로 stdout/display()/맨 끝 bare
expression 결과를 IPython과 유사하게 캡처한다. 그런 다음 위 11개 셀에 대해서만 진짜로 캡처된 출력을
nbformat outputs로 채워 넣는다(다른 셀은 이미 정상이므로 절대 건드리지 않음). TABPFN_TOKEN은
일부러 안 줘서 9-4/9-9의 TabPFN-3 비교는 조용히 skip되며(빠름), 나머지 XGBoost/LightGBM/CatBoost
등은 정상 실행된다.
"""
import ast
import contextlib
import io
import json
import os
import time

import matplotlib
matplotlib.use("Agg")

os.environ.pop("TABPFN_TOKEN", None)

t_start = time.time()
NB_PATH = r"c:\Users\JS\Desktop\forest_final.ipynb"
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

TARGET_IDX = {24, 26, 27, 29, 38, 78, 80, 83, 85, 90, 148}
STOP_AFTER = 148

nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

captured = {}  # idx -> {"stdout": str, "displays": [repr,...], "result_repr": str or None}
ns = {}


def make_display(bucket):
    def _display(*args, **kwargs):
        for a in args:
            bucket.append(repr(a))
    return _display


for i, c in enumerate(cells):
    if i > STOP_AFTER:
        break
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip() or src.lstrip().startswith(("!", "%")):
        continue

    displays = []
    ns["display"] = make_display(displays)

    tree = ast.parse(src, mode="exec")
    last_expr_node = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr_node = tree.body.pop()

    buf = io.StringIO()
    result_repr = None
    with contextlib.redirect_stdout(buf):
        exec(compile(tree, f"<cell {i}>", "exec"), ns)
        if last_expr_node is not None:
            val = eval(compile(ast.Expression(last_expr_node.value), f"<cell {i} tail>", "eval"), ns)
            if val is not None:
                result_repr = repr(val)

    if i in TARGET_IDX:
        captured[i] = {"stdout": buf.getvalue(), "displays": displays, "result_repr": result_repr}
        print(f"[캡처] 셀 {i}: stdout {len(buf.getvalue())}자, display {len(displays)}건, "
              f"bare-expr {'있음' if result_repr else '없음'}")

print(f"\n0~{STOP_AFTER}번 셀 재구성 완료 -- {time.time()-t_start:.0f}초")
print(f"캡처된 대상 셀: {sorted(captured.keys())} (목표 {sorted(TARGET_IDX)}와 일치해야 함)")
assert set(captured.keys()) == TARGET_IDX, "일부 대상 셀이 실행되지 않았습니다 — STOP_AFTER 범위 확인 필요"


def to_source(text):
    if not text:
        return []
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


for idx, cap in captured.items():
    outputs = []
    if cap["stdout"]:
        outputs.append({"output_type": "stream", "name": "stdout", "text": to_source(cap["stdout"])})
    for d in cap["displays"]:
        outputs.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": to_source(d)}})
    if cap["result_repr"] is not None:
        outputs.append({"output_type": "execute_result", "metadata": {}, "execution_count": None,
                         "data": {"text/plain": to_source(cap["result_repr"])}})
    assert cells[idx]["cell_type"] == "code"
    cells[idx]["outputs"] = outputs
    print(f"셀 {idx}: outputs {len(outputs)}개 항목으로 채움")

nb["cells"] = cells
NB_PATH_OUT = NB_PATH  # forest_final.ipynb 자체를 갱신
open(NB_PATH_OUT, "w", encoding="utf-8").write(json.dumps(nb, ensure_ascii=False, indent=1))
print(f"\n저장 완료: {NB_PATH_OUT}")
print(f"총 소요시간: {time.time()-t_start:.0f}초")

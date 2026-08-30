# -*- coding: utf-8 -*-
"""
`jupyter nbconvert --to script`로 만든 forest_welfare_analysis (2).py 맨 위에 "섹션 제목 -> 실제
줄번호" 목차 주석 블록을 삽입한다. 노트북(.ipynb) 마크다운 헤더(#, ##, ###)가 .py에서는
`# ## 9-4. ...`처럼 평범한 주석 한 줄이 되어 VS Code의 파이썬 아웃라인이 목차로 인식하지 못하므로,
직접 줄번호를 계산해 파일 맨 앞에 박아 넣어 grep 없이도 훑어볼 수 있게 한다.

사용법: 노트북이 바뀔 때마다
    jupyter nbconvert --to script "forest_welfare_analysis (2).ipynb" --output "forest_welfare_analysis (2)"
로 새로 뽑은 뒤 이 스크립트를 다시 실행하면 목차가 최신 줄번호로 재삽입된다(멱등적이지 않으므로
반드시 nbconvert로 "깨끗한" .py를 새로 만든 뒤에 실행할 것 — 이미 목차가 있는 파일에 또 실행하면
목차가 중복 삽입된다).

방식: 줄번호 폭을 5자리로 고정(L00062 형식)해 두면, 플레이스홀더(0으로 채운 목차)를 먼저 삽입해서
전체 파일을 조립한 뒤 그 안에서 각 헤더 텍스트의 실제 위치를 다시 찾아도 목차 블록 자체의 줄 수가
변하지 않는다 — 그래서 오프셋을 수학적으로 미리 계산할 필요 없이 항상 정확하다(직접 오프셋을
계산하려다 1줄 어긋난 적이 있어 이 방식으로 바꿈).
"""
import re
from pathlib import Path

PY_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).py")

orig_lines = PY_PATH.read_text(encoding="utf-8").splitlines()
assert not any("목차 (섹션 제목" in l for l in orig_lines[:5]), (
    "이미 목차가 삽입된 파일로 보입니다 — nbconvert로 새로 뽑은 깨끗한 .py에 실행하세요"
)

headers = [line for line in orig_lines if re.match(r"^# #{1,3} ", line)]


def build(numbers):
    out = ["#" + "=" * 74, "# 목차 (섹션 제목 -> .py 파일 내 실제 줄번호, 아래 grep -n으로도 재확인 가능)", "#" + "=" * 74]
    for h, n in zip(headers, numbers):
        clean = re.sub(r"^# (#{1,3}) ", "", h)
        depth = h.count("#") - 1
        indent = "  " * max(depth - 1, 0)
        out.append(f"#   L{n:05d} {indent}{clean}")
    out.append("#" + "=" * 74)
    return out


toc_placeholder = build([0] * len(headers))
prefix = ["#!/usr/bin/env python", "# coding: utf-8", ""] + toc_placeholder + ["", ""]
final_placeholder = prefix + orig_lines[2:]

real_numbers = []
search_from = 0
for h in headers:
    idx = final_placeholder.index(h, search_from)
    real_numbers.append(idx + 1)
    search_from = idx + 1

toc_real = build(real_numbers)
assert len(toc_real) == len(toc_placeholder), "목차 블록 줄수가 달라짐 - 자릿수 고정 실패"

prefix2 = ["#!/usr/bin/env python", "# coding: utf-8", ""] + toc_real + ["", ""]
final_lines = prefix2 + orig_lines[2:]

PY_PATH.write_text("\n".join(final_lines) + "\n", encoding="utf-8", newline="\n")
print(f"목차 삽입 완료 — 총 {len(headers)}개 항목")

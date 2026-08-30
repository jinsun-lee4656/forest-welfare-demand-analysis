# -*- coding: utf-8 -*-
"""9-4절 comparison_df 표 바로 뒤에 '평균AUC 기준 최고 성능 모델' 안내 print를 추가.
표 자체는 Macro_F1 기준 정렬을 그대로 유지하되(전체 결론 논리와 일관성 유지),
AUC 1위(TabPFN)가 표만 봐서는 눈에 띄지 않는 문제를 텍스트로 명시한다.
"""
import json
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

idx = next(i for i, c in enumerate(cells)
           if "comparison_df = pd.DataFrame(ml_boost_results)" in "".join(c["source"]))
print("대상 셀 인덱스:", idx)
cell = cells[idx]

src = "".join(cell["source"])
assert src.rstrip().endswith('"공식 통계적 최하한선입니다. 여기 나온 모델들이 이 선을 얼마나 넘는지가 실질적인 학습 성과입니다.)")'), \
    "예상한 마지막 줄이 아닙니다 -- 셀 내용이 바뀌었는지 확인 필요"

NEW_LINE = '\nprint(f"평균AUC 기준 최고 성능 모델: {comparison_df[\'평균AUC\'].astype(float).idxmax()}")'
new_src = src + NEW_LINE
cell["source"] = new_src.splitlines(keepends=True)  # 기존 파일과 동일한 줄 단위 리스트 형식 유지

# 기존 outputs의 마지막 요소가 그 "(참고: ...)" print의 stream 출력이어야 함 -> 텍스트 한 줄 추가
outputs = cell["outputs"]
last = outputs[-1]
assert last["output_type"] == "stream" and last["name"] == "stdout", "마지막 output이 예상과 다릅니다"
last_text = "".join(last["text"]) if isinstance(last["text"], list) else last["text"]
assert last_text.rstrip().endswith("학습 성과입니다.)"), "마지막 stream 텍스트가 예상과 다릅니다"

new_text = last_text + "평균AUC 기준 최고 성능 모델: TabPFN\n"
last["text"] = new_text.splitlines(keepends=True)

nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("저장 완료")

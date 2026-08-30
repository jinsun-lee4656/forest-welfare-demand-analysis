# -*- coding: utf-8 -*-
"""
발견된 버그 수정: forest_welfare_analysis (2).ipynb의 24번 셀(3장 전처리 핵심 코드 —
`df = raw.copy()`부터 가구ID 복원/소득전파/인구통계 라벨링/학력·직업 매핑까지)의
cell_type이 "code"가 아니라 "markdown"으로 잘못 바뀌어 있어, 실제로 처음부터
재실행하면 이 셀이 그냥 텍스트로만 렌더링되고 df가 한 번도 생성되지 않아 이후 모든
셀(26번 위생점검부터 9장 모델들까지)이 NameError로 실패하는 상태였다.
(compute_tabpfn_top1.py 실행 실패, 이전 FAILED.ipynb 실패 모두 이 버그가 원인.)

이 스크립트는:
1. 24번 셀을 code로 되돌리고(outputs=[], execution_count=None — 다음 실제 실행에서
   새로 채워짐), 셀 메타데이터에서 markdown 잔재가 없는지 확인한다.
2. 22번/23번 셀이 완전히 동일한 섹션 제목으로 중복되어 있던 것 중 22번(구버전,
   23번에 흡수된 짧은 버전)을 제거한다.
3. 수정 전 노트북을 안전하게 백업한다.
"""
import json
import shutil
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")
BACKUP_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).backup_before_cell24_fix.ipynb")

shutil.copy2(NB_PATH, BACKUP_PATH)
print(f"수정 전 백업: {BACKUP_PATH}")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

# --- 1) 24번 셀 검증 후 code로 복구 ---
c24 = cells[24]
src24 = "".join(c24["source"])
assert c24["cell_type"] == "markdown", f"예상과 다름: cell 24 cell_type={c24['cell_type']!r} (이미 고쳐졌을 수 있음)"
assert src24.strip().startswith("df = raw.copy()"), "24번 셀 내용이 예상과 다릅니다 — 수동 확인 필요"
assert 'assert df["occupation"].isna().sum() == 0' in src24, "24번 셀에 occupation assert가 없음 — 다른 셀일 가능성"

c24["cell_type"] = "code"
c24["outputs"] = []
c24["execution_count"] = None
print(f"cell 24: markdown -> code 복구 완료 ({len(src24)}자)")

# --- 2) 22번 중복 헤더 셀 제거 (23번이 최종/확장판이므로 22번만 삭제) ---
src22 = "".join(cells[22]["source"])
src23 = "".join(cells[23]["source"])
assert cells[22]["cell_type"] == "markdown" and cells[23]["cell_type"] == "markdown"
assert src22.splitlines()[0] == src23.splitlines()[0] == "## 3. 전처리 (1) — 가구구조 복원 및 인구통계 라벨링"
assert len(src23) > len(src22), "23번이 22번보다 짧습니다 — 예상과 다름, 수동 확인 필요"
del cells[22]
print("cell 22 (중복 섹션 헤더, 구버전) 삭제 완료")

nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"저장 완료: {NB_PATH} (총 {len(cells)}개 셀, 기존 대비 -1)")

# -*- coding: utf-8 -*-
"""
두 번째로 발견된 버그 수정: 5장(방문기록 롱포맷 집계) 셀이 통째로 중복되어 있었다.
현재(cell24 수정 후) 노트북 기준 36번 셀과 38번 셀이 거의 동일한 코드(둘 다
long_day/long_night 생성 + agg_day/agg_night groupby + df.merge)인데, 38번이
36번에 "방문지역코드(sido_code)가 거주지 시도코드(CO11)와 같은 체계인지" 검증하는
블록이 추가된 최신/완전판이다. 36번은 그 검증이 추가되기 전 구버전이 지워지지 않고
남은 잔재로 보인다.

36번이 먼저 실행되면 df에 daytrip_dominant_companion 등 컬럼이 이미 생기고, 이어서
38번이 같은 groupby를 다시 만들어 df.merge()를 한 번 더 하면 컬럼명이 겹쳐
"_x"/"_y"로 접미사가 붙어버려 38번 뒷부분(df[c+"_label"]=...)이
KeyError: 'daytrip_dominant_companion'로 죽는다 — 실제로 compute_tabpfn_top1.py
재실행에서 이 오류가 재현되어 발견했다.

difflib 기반 전수 스캔(모든 코드 셀 쌍, 근접 8셀 이내 + 전체 고임계값)으로 이 외의
동일 패턴은 더 없음을 확인했다(양쪽 결과 모두 이 쌍 하나만 검출).

수정: 36번(구버전, 검증 블록 없음)을 삭제하고 38번(완전판)만 남긴다.
"""
import json
import shutil
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")
BACKUP_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).backup_before_cell36_fix.ipynb")

shutil.copy2(NB_PATH, BACKUP_PATH)
print(f"수정 전 백업: {BACKUP_PATH}")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

src36 = "".join(cells[36]["source"])
src38 = "".join(cells[38]["source"])
assert cells[36]["cell_type"] == "code" and cells[38]["cell_type"] == "code"
assert 'daytrip_dominant_companion=("companion",mode_or_nan)' in src36
assert 'daytrip_dominant_companion=("companion",mode_or_nan)' in src38
assert "VALID_SIDO_CODES" not in src36, "36번에 이미 검증 블록이 있음 — 예상과 다름"
assert "VALID_SIDO_CODES" in src38, "38번에 검증 블록이 없음 — 예상과 다름"
assert len(src38) > len(src36), "38번이 36번보다 짧습니다 — 예상과 다름, 수동 확인 필요"

del cells[36]
print("cell 36 (5장 방문기록 집계, 검증블록 없는 구버전 중복) 삭제 완료")

nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"저장 완료: {NB_PATH} (총 {len(cells)}개 셀)")

# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 4차 수정: 사용자가 지적한 누락 항목 보강.
  A. dtype 점검 (raw 668개 컬럼 dtype 분포, object 컬럼이 전부 ETC 주관식인지, 핵심 컬럼 dtype/범위,
     NO0가 가구별로 1..n 연속인지 - 가구주 1명 확인보다 더 엄격한 조건) — 기존 3-1절 위생점검 셀에 이어붙임
  B. 다중응답 슬롯 내 코드 중복 검증 — 체크리스트형 문항(Q1/Q10/Q17/Q19/Q20)은 중복 0건이어야 하고,
     방문기록형 문항(Q11/Q12)은 같은 활동을 반복 방문한 경우 슬롯마다 같은 코드가 반복되는 게 "정상"임을
     구분해서 검증 (스킵로직 검증 셀 바로 뒤에 추가)
"""
import json
import secrets
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source(text: str):
    lines = text.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


def new_cell(cell_type: str, text: str):
    cell = {"cell_type": cell_type, "id": secrets.token_hex(4), "metadata": {}, "source": to_source(text)}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def find_index(cells, marker, nth=0):
    hits = [i for i, c in enumerate(cells) if marker in "".join(c["source"])]
    if len(hits) <= nth:
        raise AssertionError(f"anchor {marker!r} 을(를) 가진 셀을 {nth+1}번째까지 찾지 못함 (hits={hits})")
    return hits[nth]


def append_to_cell(cells, idx, extra_text):
    c = cells[idx]
    src = "".join(c["source"])
    new_src = src.rstrip("\n") + "\n\n" + extra_text.strip("\n") + "\n"
    c["source"] = to_source(new_src)
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # ------------------------------------------------------------------
    # A. dtype 점검을 기존 위생점검 셀([7] DQ5<->SQ8 바로 뒤)에 이어붙임
    # ------------------------------------------------------------------
    idx_hygiene = find_index(cells, "3장의 ffill/bfill 가구단위 전파가 올바른 응답조건에 기반함을 확인")
    append_to_cell(cells, idx_hygiene, """
# --- dtype 점검 ---
dtype_counts = raw.dtypes.value_counts()
print("[8] 원본 dtype 분포:", dtype_counts.to_dict())

obj_cols = raw.select_dtypes(include=["object", "string"]).columns.tolist()
non_etc_obj = [c for c in obj_cols if "ETC" not in c.upper()]
assert not non_etc_obj, f"ETC(주관식) 아닌데 문자열로 읽힌 컬럼 발견 — 숫자 컬럼 오염 가능성: {non_etc_obj}"
print(f"[9] object(문자열) dtype {len(obj_cols)}개 전부 '_ETC'(주관식 직접작성) 컬럼임을 확인 "
      f"→ 숫자코드 컬럼이 문자열로 잘못 읽힌 경우 없음")

assert raw["NO0"].dtype.kind in "iu" and raw["WT"].dtype.kind == "f"
assert (raw["WT"] > 0).all(), "가중치(WT)에 0 이하 값 존재"
print(f"[10] 핵심 컬럼 dtype 확인 — NO0:{raw['NO0'].dtype}, "
      f"WT:{raw['WT'].dtype}(전부 양수, {raw['WT'].min():.0f}~{raw['WT'].max():.0f}), "
      f"CO11:{raw['CO11'].dtype}, SQ7_1:{raw['SQ7_1'].dtype}")

# NO0가 가구별로 1..n 연속인지 (가구주 1명 확인[3]보다 더 엄격한 조건 — 결번/순서뒤바뀜까지 잡아냄)
bad_seq = sum(1 for _, g in df.groupby("hh_id")["NO0"] if g.tolist() != list(range(1, len(g) + 1)))
assert bad_seq == 0, f"NO0가 1..n 연속이 아닌 가구 {bad_seq}개 존재"
print(f"[11] NO0(가구원 일련번호)가 가구별로 1..n 연속인 것까지 확인 — {df['hh_id'].nunique()}개 가구 전부 통과")
""")
    print(f"A. dtype 점검을 idx {idx_hygiene} 셀에 이어붙임")

    # ------------------------------------------------------------------
    # B. 다중응답 슬롯 내 코드 중복 검증을 스킵로직 검증 셀 뒤에 추가
    # ------------------------------------------------------------------
    idx_skiplogic = find_index(cells, 'assert (skip_logic_result["위반건수"] == 0).all()')

    md_dup = new_cell("markdown", """
### 3-1-1. 다중응답 슬롯 내 코드 중복 점검 (스킵로직 20개 항목에 추가되는 21번째 확인)

위 20개 항목과 별개로, 다중응답 슬롯(Q1/Q10/Q17/Q19_*/Q20_*/Q11/Q12) 안에서 **같은 코드가 두 번 이상 선택된 응답자**가 있는지 확인합니다.
Q1·Q10·Q17·Q19_*·Q20_*는 "항목별 체크리스트"이므로 같은 코드가 중복되면 `collect_selected()`의 개수 집계(`n_daytrip_activity_types` 등)가
부풀려지는 실질적 버그지만, **Q11/Q12(방문기록)는 슬롯 하나가 "방문 1회"를 의미**하므로 같은 활동을 여러 번 방문했다면 슬롯마다 같은 코드가
반복되는 것이 오히려 정상입니다(예: 등산을 3번 갔다면 Q11A1~A3 모두 "등산" 코드). 이 구조적 차이를 반영해 문항 종류별로 다른 기준으로 검증합니다.
""")
    code_dup = new_cell("code", """
# --- 다중응답 슬롯 내 코드 중복 점검 ---
def _check_dup_within_slots(df_chk, prefix, n, none_codes={99.0, 999999999.0}):
    cols = [f"{prefix}A{i}" for i in range(1, n + 1) if f"{prefix}A{i}" in df_chk.columns]
    sub = df_chk[cols]
    def has_dup(row):
        vals = [v for v in row if pd.notna(v) and v not in none_codes]
        return len(vals) != len(set(vals))
    return int(sub.apply(has_dup, axis=1).sum())

# (a) 체크리스트형 문항: 같은 항목을 두 번 선택할 수 없어야 함 -> 위반 0건이 정상
checklist_specs = [("Q1", 4), ("Q10_1", 29), ("Q10_2", 29), ("Q17", 29),
                    ("Q19_1", 5), ("Q19_2", 5), ("Q19_3", 5), ("Q19_4", 5), ("Q19_5", 5),
                    ("Q20_1", 13), ("Q20_2", 13), ("Q20_3", 13), ("Q20_4", 13), ("Q20_5", 13)]
checklist_dup = {prefix: _check_dup_within_slots(raw, prefix, n) for prefix, n in checklist_specs}
print("[체크리스트형 문항] 슬롯 내 코드 중복 응답자 수 (전부 0이어야 정상):")
print(checklist_dup)
assert all(v == 0 for v in checklist_dup.values()), f"체크리스트형 문항에서 중복코드 발견: {checklist_dup}"
print("-> 전부 0건, collect_selected() 개수 집계(n_daytrip_activity_types 등)가 부풀려질 위험 없음\\n")

# (b) 방문기록형 문항(Q11/Q12): 반복방문으로 인한 중복은 정상이므로 참고용으로만 보고(assert 없음)
q11_dup = _check_dup_within_slots(raw, "Q11", 15)
q12_dup = _check_dup_within_slots(raw, "Q12", 15)
print(f"[방문기록형 문항, 참고용] Q11 슬롯 내 코드 중복 응답자: {q11_dup}명 ({q11_dup/len(raw)*100:.1f}%), "
      f"Q12: {q12_dup}명 ({q12_dup/len(raw)*100:.1f}%)")
print("-> 같은 활동유형을 여러 번(최대 15회) 방문한 경우로, daytrip_n_records/overnight_n_records(방문횟수)가"
      " 바로 이 반복방문을 세는 변수이므로 정상적인 현상이며 오류가 아님(3-1절 D2 '활동종류 수 vs 방문횟수' 구분 참고)")
""")
    cells.insert(idx_skiplogic + 1, code_dup)
    cells.insert(idx_skiplogic + 1, md_dup)
    print(f"B. 다중응답 중복코드 검증 셀을 idx {idx_skiplogic} 뒤에 삽입")

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 완료: {NB_PATH} (총 {len(cells)}개 셀)")


if __name__ == "__main__":
    main()

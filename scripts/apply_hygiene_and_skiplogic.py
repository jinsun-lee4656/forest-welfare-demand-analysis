# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 3차 수정: 사용자 체크리스트 반영.
  A. 기초 위생점검 (중복행/결측요약/이상치/가구ID/나이교차검증) 셀 추가
  B. DQ5(가구소득) <-> SQ8(가구주본인) 정합성 assert 추가
  C. skip_logic_validation.py의 20개 논리정합성 검증을 노트북에 인라인 삽입 (Q10<->Q11/Q12,
     Q19-2<->19-3/4 등 데이터 누수 관련 부분집합 검증 포함)
  D. 문1 다중선택 비배타성(44.1%, 57% 아님) 및 "활동종류 수" vs "방문횟수" 용어 구분 문서화
  E. 빈 코드셀 4개 + 디버깅용 powershell 셀 1개 삭제

전부 anchor 문자열 기반으로 셀을 찾아 삽입/삭제하므로, 이전 두 차례 수정으로 인덱스가
이미 밀린 상태와 무관하게 안전하게 동작한다.
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


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # ------------------------------------------------------------------
    # E. 빈 코드셀 + 디버깅용 powershell 셀 삭제 (먼저 처리 — 이후 anchor 탐색에 영향 없음)
    # ------------------------------------------------------------------
    def is_junk(c):
        if c["cell_type"] != "code":
            return False
        src = "".join(c["source"]).strip()
        if src == "":
            return True
        if "powershell -NoProfile" in src and "Get-Command claude" in src:
            return True
        return False

    before = len(cells)
    cells = [c for c in cells if not is_junk(c)]
    print(f"빈 셀/디버깅 셀 삭제: {before}개 -> {len(cells)}개")

    # ------------------------------------------------------------------
    # A + B. 가구구조 복원 직후(= household 라벨링 tail, income_ord/occupation 출력 직후)에
    #        기초 위생점검 + DQ5<->SQ8 정합성 셀 삽입
    # ------------------------------------------------------------------
    anchor_a = find_index(cells, 'df[["age","age_band","gender","hh_type","income_band","income_ord","education","occupation","sido","city_size"]].head(8)')

    md_a = new_cell("markdown", """
### 3-1. 기초 위생점검 및 스킵로직 자동검증

본격적인 분석에 들어가기 전, 원자료 자체의 무결성(중복·결측·이상치)과 가구구조 복원 로직, 그리고 설문
스킵로직(응답조건) 위반 여부를 기계적으로 확인합니다. 이 절의 모든 assert가 통과해야 이후 분석의 전제(가구ID
유일성, 나이·소득 응답조건, 다중응답 문항 간 계층적 종속관계)가 성립합니다.
""")

    code_a = new_cell("code", """
# --- 기초 위생점검 ---
print("[1] 완전 중복행 수:", raw.duplicated().sum(), "(0이어야 함)")
dup_person = df.duplicated(subset=["hh_id", "NO0"]).sum()
print("[2] (hh_id, NO0) 조합 중복 수:", dup_person, "(0이어야 각 가구원이 유일)")
assert raw.duplicated().sum() == 0 and dup_person == 0, "중복행/중복 가구원 발견"

head_count_per_hh = df.groupby("hh_id")["NO0"].apply(lambda s: (s == 1).sum())
bad_heads = int((head_count_per_hh != 1).sum())
assert bad_heads == 0, f"가구주가 정확히 1명이 아닌 가구 {bad_heads}개 존재"
print(f"[3] 가구주(NO0==1) 정확히 1명인 가구: {df['hh_id'].nunique()}개 전부 통과")

# 결측치 요약 (원본 668개 컬럼 기준 — 스킵로직에 따른 구조적 결측이 대부분이라 결측률 자체가 "문제"는 아님)
na_pct = (raw.isna().mean() * 100).sort_values(ascending=False)
print(f"\\n[4] 원본 {raw.shape[1]}개 컬럼 중 결측 0%: {(na_pct == 0).sum()}개 / 결측률 상위 5개(대부분 스킵로직에 의한 구조적 결측):")
print(na_pct.head(5).round(1))

# 나이 계산(SURVEY_YEAR - SQ7_1) vs 공식 라벨(D_SQ7) 경계 교차검증
# SQ7_1이 "출생연도"만 있어 생일 경과여부에 따라 경계에서 최대 ±1세 오차가 남는 것을 확인(정상 — 위 3장 주석 참고)
print("\\n[5] age(연산값) 경계에서 age_band(공식라벨) 분포 — 생일 미상으로 인한 ±1세 경계오차 재확인:")
for boundary in [20, 30, 40, 50, 60, 70]:
    sub = df[df["age"] == boundary]
    print(f"  age=={boundary}세 {len(sub)}명 -> age_band 분포: {sub['age_band'].value_counts().to_dict()}")

# 자기보고식 지출액/숙박일수 이상치 점검 — 음수·비현실적 극단값 여부만 확인, winsorizing은 미적용
spend_day_cols = [c for c in raw.columns if c.startswith("Q11_8A")]
spend_night_cols = [c for c in raw.columns if c.startswith("Q12_8A")]
nights_cols = [c for c in raw.columns if c.startswith("Q12_5_11A")]
day_spend_all = pd.concat([raw[c] for c in spend_day_cols])
night_spend_all = pd.concat([raw[c] for c in spend_night_cols])
nights_all = pd.concat([raw[c] for c in nights_cols])
print("\\n[6] 지출액/숙박일수 이상치 점검(min/25%/50%/75%/max):")
print("  당일형 1인평균소비금액(만원):", day_spend_all.describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
print("  숙박형 1인평균소비금액(만원):", night_spend_all.describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
print("  숙박일수:", nights_all.describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
assert day_spend_all.min() >= 0 and night_spend_all.min() >= 0 and nights_all.min() >= 0, "음수 값 발견"
print("  -> 음수값 없음. 상한(당일 68만원/숙박 80만원/6박)도 자기보고식 여행지출로 불가능한 범위는 아니라고 판단해 winsorizing 미적용")

# --- DQ5(가구소득) ffill/bfill 전파 로직 검증: 원본 응답이 SQ8==1(가구주 본인)과 정확히 일치하는지 ---
mismatch_has_not_head = int(((raw["DQ5"].notna()) & (raw["SQ8"] != 1)).sum())
mismatch_head_no_dq5 = int(((raw["DQ5"].isna()) & (raw["SQ8"] == 1)).sum())
assert mismatch_has_not_head == 0, f"가구주 아닌데 DQ5 응답 있는 행 {mismatch_has_not_head}건"
assert mismatch_head_no_dq5 == 0, f"가구주인데 DQ5 결측인 행 {mismatch_head_no_dq5}건"
print(f"\\n[7] DQ5(가구소득) 원본 응답은 SQ8==1(가구주 본인) {int((raw['SQ8'] == 1).sum())}명과 정확히 1:1 일치 "
      f"(불일치 0건) → 3장의 ffill/bfill 가구단위 전파가 올바른 응답조건에 기반함을 확인")
""")

    cells.insert(anchor_a + 1, code_a)
    cells.insert(anchor_a + 1, md_a)
    print(f"A/B 위생점검 셀 삽입 위치: anchor idx {anchor_a} 뒤")

    # ------------------------------------------------------------------
    # C. skip_logic_validation.py 20개 항목 인라인 삽입 (위 A/B 셀들 바로 뒤)
    # ------------------------------------------------------------------
    anchor_c = find_index(cells, "3-1. 기초 위생점검 및 스킵로직 자동검증")
    # anchor_c는 방금 넣은 md_a 자신이므로, 그 다음(code_a) 뒤에 넣는다
    insert_at = anchor_c + 2

    code_c = new_cell("code", """
# --- 스킵로직/논리정합성 자동검증 (20개 항목, 원본: MDIS\\\\skip_logic_validation.py와 동일 로직) ---
# Q10<->Q11/Q12 부분집합 관계(문10 체크셋과 문11/12 참여활동 코드가 일치하는지 — 데이터 누수 검토
# "역할 분리" 원칙의 근거: Q10은 과거 행태 피처로, Q11/Q12는 season/companion/purpose/spend 등
# 부가 환경변수 피처로만 쓰고 원본 활동코드는 재사용하지 않는다는 9-0-0절 설계가 실제로 타당한지도
# 확인해줌), Q19-2<->19-3/4 및 Q20-2<->20-3/4 계층적 응답조건, Q1<->Part2/3 진입 정합성 등
# 총 20개 항목을 기계적으로 검증합니다.
NONE_CODES_SLV = {9, 99, 999999999}

def _get_code_set(row, prefix, n, exclude_none=True):
    cols = [f"{prefix}A{i}" for i in range(1, n + 1)]
    vals = {v for c in cols if pd.notna(v := row.get(c))}
    return vals - NONE_CODES_SLV if exclude_none else vals

def _any_code(row, cols, code):
    return any(row.get(c) == code for c in cols)

def _notna_any(row, cols):
    return any(pd.notna(row.get(c)) for c in cols)

def run_skip_logic_checks(df_chk: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    q1cols_ = ["Q1A1", "Q1A2", "Q1A3", "Q1A4"]
    has_daily_ = df_chk.apply(lambda r: _any_code(r, q1cols_, 1), axis=1)
    has_day_ = df_chk.apply(lambda r: _any_code(r, q1cols_, 2), axis=1)
    has_stay_ = df_chk.apply(lambda r: _any_code(r, q1cols_, 3), axis=1)
    has_none_ = df_chk.apply(lambda r: _any_code(r, q1cols_, 4), axis=1)
    checks = {}
    checks["Q1_모두없음+다른유형_동시선택"] = int(((has_none_) & (has_daily_ | has_day_ | has_stay_)).sum())
    has_1_ = df_chk.apply(lambda r: (r.get("Q1_1A1") == 1) or (r.get("Q1_1A2") == 1), axis=1)
    has_3_ = df_chk.apply(lambda r: (r.get("Q1_1A1") == 3) or (r.get("Q1_1A2") == 3), axis=1)
    checks["Q1_2_1(흥미)_응답인데_Q1_1에_흥미없음_부재"] = int(((df_chk["Q1_2_1"].notna()) & (~has_1_)).sum())
    checks["Q1_2_2(시간)_응답인데_Q1_1에_시간없음_부재"] = int(((df_chk["Q1_2_2"].notna()) & (~has_3_)).sum())
    part2_cols_ = ["Q2_1_1", "Q3_1", "Q4", "Q5A1", "Q6", "Q7_1", "Q8", "Q9A1"]
    part2_answered_ = df_chk.apply(lambda r: _notna_any(r, part2_cols_), axis=1)
    checks["일상형_미선택인데_Part2_응답존재"] = int(((~has_daily_) & part2_answered_).sum())
    checks["일상형_선택인데_Part2_전부결측"] = int(((has_daily_) & (~part2_answered_)).sum())
    q11cols_ = [f"Q11A{i}" for i in range(1, 16)]
    q11_answered_ = df_chk.apply(lambda r: _notna_any(r, q11cols_), axis=1)
    checks["당일형_미선택인데_문11_응답존재"] = int(((~has_day_) & q11_answered_).sum())
    checks["당일형_선택인데_문11_전부결측"] = int(((has_day_) & (~q11_answered_)).sum())
    q12cols_ = [f"Q12A{i}" for i in range(1, 16)]
    q12_answered_ = df_chk.apply(lambda r: _notna_any(r, q12cols_), axis=1)
    checks["숙박형_미선택인데_문12_응답존재"] = int(((~has_stay_) & q12_answered_).sum())
    checks["숙박형_선택인데_문12_전부결측"] = int(((has_stay_) & (~q12_answered_)).sum())
    checks["숙박형_미선택인데_문13_응답"] = int(((~has_stay_) & df_chk["Q13"].notna()).sum())
    checks["당일+숙박_모두미선택인데_문14_응답"] = int(((~has_day_) & (~has_stay_) & df_chk["Q14"].notna()).sum())
    q17cols_ = [f"Q17A{i}" for i in range(1, 30)]
    q17_answered_ = df_chk.apply(lambda r: _notna_any(r, q17cols_), axis=1)
    checks["당일+숙박_모두미선택인데_문17_응답"] = int(((~has_day_) & (~has_stay_) & q17_answered_).sum())
    checks["일상형_미선택인데_문18_응답"] = int(((~has_daily_) & df_chk["Q18"].notna()).sum())
    checks["모두없음인데_문22_응답"] = int(((has_none_) & (~has_daily_) & (~has_day_) & (~has_stay_) & df_chk["Q22"].notna()).sum())

    v19_3 = v19_4 = v20_3 = v20_4 = 0
    for _, row in df_chk.iterrows():
        used19 = _get_code_set(row, "Q19_2", 5)
        if not _get_code_set(row, "Q19_3", 5).issubset(used19):
            v19_3 += 1
        if not _get_code_set(row, "Q19_4", 5).issubset(used19):
            v19_4 += 1
        used20 = _get_code_set(row, "Q20_2", 13)
        if not _get_code_set(row, "Q20_3", 13).issubset(used20):
            v20_3 += 1
        if not _get_code_set(row, "Q20_4", 13).issubset(used20):
            v20_4 += 1
    checks["Q19_3(예약)_불부합_Q19_2(이용경험)"] = v19_3
    checks["Q19_4(바우처)_불부합_Q19_2(이용경험)"] = v19_4
    checks["Q20_3(예약)_불부합_Q20_2(이용경험)"] = v20_3
    checks["Q20_4(바우처)_불부합_Q20_2(이용경험)"] = v20_4

    v1011 = v1012 = 0
    for _, row in df_chk.iterrows():
        if not _get_code_set(row, "Q11", 15).issubset(_get_code_set(row, "Q10_1", 29)):
            v1011 += 1
        if not _get_code_set(row, "Q12", 15).issubset(_get_code_set(row, "Q10_2", 29)):
            v1012 += 1
    checks["문11_참여활동_불부합_문10_1_체크셋"] = v1011
    checks["문12_참여활동_불부합_문10_2_체크셋"] = v1012

    result_ = pd.DataFrame([{"검증항목": k, "위반건수": v, "비율(%)": round(v / len(df_chk) * 100, 3)} for k, v in checks.items()])
    if verbose:
        n_fail = (result_["위반건수"] > 0).sum()
        print(f"총 {len(result_)}개 검증 항목 중 위반 발견: {n_fail}개")
        if n_fail > 0:
            print(result_[result_["위반건수"] > 0].to_string(index=False))
        else:
            print("모든 스킵로직/논리정합성 검증 통과 (위반 0건)")
    return result_

skip_logic_result = run_skip_logic_checks(raw)
assert (skip_logic_result["위반건수"] == 0).all(), "스킵로직 위반 발견 — 위 표 확인 필요"
""")
    cells.insert(insert_at, code_c)
    print(f"C. 스킵로직 자동검증 셀 삽입 위치: idx {insert_at}")

    # ------------------------------------------------------------------
    # D-1. 문1 다중선택 비배타성 문서화 (Q1 다중응답 처리 직후)
    # ------------------------------------------------------------------
    anchor_d1 = find_index(cells, 'print(df["pref_activity_broad"].value_counts(dropna=False))')

    md_d1 = new_cell("markdown", """
**주의(비상호배타적 그룹)**: `exp_daily`/`exp_daytrip`/`exp_overnight`(문1 활동형태)는 서로 배타적이지 않습니다 — 실제로 전체 응답자의 **44.1%가 2개 이상을 동시에 선택**했고
(일상형 선택자만 놓고 보면 58.4%가 다른 유형도 함께 경험), 3개 모두 선택한 응답자도 8.4%(1,007명) 존재합니다(단순 다수결로 짐작한 "57%"는 실측과 달라 정정합니다 — 참고로 일상형
단독 선택 비율이 56.9%로 이 수치와 가장 비슷합니다).

따라서 이 세 변수를 하나의 "활동유형" 팩터로 묶어 카이제곱검정 등에 사용하면 관측치가 여러 셀에 중복 포함되어 독립성 가정이 깨집니다. 7장의 활동유형 관련 카이제곱검정(성별/1인가구 x 선호활동유형)은
이 세 변수가 아니라 Q17 기반 단일 대표값(`pref_activity_broad`, idxmax)을 쓰고 있어 이 문제 자체는 피하고 있으며, 그 대표값 방식 고유의 한계(광역카테고리별 세부항목 수 차이로 인한 구조적 쏠림)는
7장 결론에서 별도로 명시합니다. 향후 `exp_daily`/`exp_daytrip`/`exp_overnight` 자체를 그룹 요인으로 쓰는 분석을 추가한다면, 상호배타적 세그먼트로 재정의하거나(`lifecycle_segment`처럼)
각 이진 지표별로 개별 검정하는 방식으로 접근해야 합니다.
""")
    cells.insert(anchor_d1 + 1, md_d1)
    print(f"D1. 비배타성 문서화 삽입 위치: anchor idx {anchor_d1} 뒤")

    # ------------------------------------------------------------------
    # D-2. "활동종류 수"(Q10) vs "방문횟수"(Q11/Q12) 용어 구분 문서화
    # ------------------------------------------------------------------
    anchor_d2 = find_index(cells, "당일형 방문기록(long) 행수")

    md_d2 = new_cell("markdown", """
**용어 구분(활동종류 수 vs 방문횟수)**: `n_daytrip_activity_types`/`n_overnight_activity_types`(4장, Q10 기준)는 **몇 "종류"의 활동을 경험했는지**(29개 세부활동 중 몇 개를 체크했는지)이고,
`daytrip_n_records`/`overnight_n_records`(이 절, Q11/Q12 기준)는 **몇 "번" 방문했는지**(방문기록 슬롯 수, 최대 15회)로 서로 다른 개념입니다. 예를 들어 등산만 5번 다녀온 사람은
활동종류 수=1, 방문횟수=5입니다. 두 변수 모두 FEATURES_NUM에 함께 포함되지만(9~10장), 서로 다른 정보를 담고 있으므로 중복 피처가 아닙니다.
""")
    cells.insert(anchor_d2 + 1, md_d2)
    print(f"D2. 용어 구분 문서화 삽입 위치: anchor idx {anchor_d2} 뒤")

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 완료: {NB_PATH} (총 {len(cells)}개 셀)")


if __name__ == "__main__":
    main()

# ============================================================
# 스킵로직 / 논리정합성 검증 유틸리티
# 2024년 산림휴양복지활동 조사 원자료 전용
# 심사위원 피드백 대응: "학습변수-예측대상 간 정보중복/데이터누수
# 여부를 분석흐름도와 변수정의표를 통해 명확히 설명" 근거자료
# ============================================================
import pandas as pd

NONE_CODES = {9, 99, 999999999}  # 설문 전반에 쓰이는 "없음" 특수코드


def get_code_set(row, prefix, n, exclude_none=True):
    """Q10_1A1~A29 같은 슬롯-코드 방식 다중응답을 실제 코드 집합으로 변환.
    Q10, Q17, Q19_*, Q20_* 등 모든 슬롯식 다중응답 문항에 재사용 가능."""
    cols = [f"{prefix}A{i}" for i in range(1, n + 1)]
    vals = {v for c in cols if pd.notna(v := row.get(c))}
    return vals - NONE_CODES if exclude_none else vals


def any_code(row, cols, code):
    return any(row.get(c) == code for c in cols)


def notna_any(row, cols):
    return any(pd.notna(row.get(c)) for c in cols)


def run_skip_logic_checks(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """전체 스킵로직/논리정합성 검증을 실행하고 위반건수 요약 DataFrame을 반환.
    위반이 0이 아니면 해당 응답을 결측처리하거나 별도 플래그로 분리할 것을 권장.
    """
    q1cols = ["Q1A1", "Q1A2", "Q1A3", "Q1A4"]
    has_daily = df.apply(lambda r: any_code(r, q1cols, 1), axis=1)
    has_day = df.apply(lambda r: any_code(r, q1cols, 2), axis=1)
    has_stay = df.apply(lambda r: any_code(r, q1cols, 3), axis=1)
    has_none = df.apply(lambda r: any_code(r, q1cols, 4), axis=1)

    checks = {}

    # 1. 문1 논리적 오류 (조사표 자체 명시: ④와 ①~③ 동시선택 불가)
    checks["Q1_모두없음+다른유형_동시선택"] = int(((has_none) & (has_daily | has_day | has_stay)).sum())

    # 2. 문1-1(미경험이유) → 문1-2(상세이유) 종속관계
    has_1 = df.apply(lambda r: (r.get("Q1_1A1") == 1) or (r.get("Q1_1A2") == 1), axis=1)
    has_3 = df.apply(lambda r: (r.get("Q1_1A1") == 3) or (r.get("Q1_1A2") == 3), axis=1)
    checks["Q1_2_1(흥미)_응답인데_Q1_1에_흥미없음_부재"] = int(((df["Q1_2_1"].notna()) & (~has_1)).sum())
    checks["Q1_2_2(시간)_응답인데_Q1_1에_시간없음_부재"] = int(((df["Q1_2_2"].notna()) & (~has_3)).sum())

    # 3. 문1(활동형태) → Part2/Part3 진입 정합성
    part2_cols = ["Q2_1_1", "Q3_1", "Q4", "Q5A1", "Q6", "Q7_1", "Q8", "Q9A1"]
    part2_answered = df.apply(lambda r: notna_any(r, part2_cols), axis=1)
    checks["일상형_미선택인데_Part2_응답존재"] = int(((~has_daily) & part2_answered).sum())
    checks["일상형_선택인데_Part2_전부결측"] = int(((has_daily) & (~part2_answered)).sum())

    q11cols = [f"Q11A{i}" for i in range(1, 16)]
    q11_answered = df.apply(lambda r: notna_any(r, q11cols), axis=1)
    checks["당일형_미선택인데_문11_응답존재"] = int(((~has_day) & q11_answered).sum())
    checks["당일형_선택인데_문11_전부결측"] = int(((has_day) & (~q11_answered)).sum())

    q12cols = [f"Q12A{i}" for i in range(1, 16)]
    q12_answered = df.apply(lambda r: notna_any(r, q12cols), axis=1)
    checks["숙박형_미선택인데_문12_응답존재"] = int(((~has_stay) & q12_answered).sum())
    checks["숙박형_선택인데_문12_전부결측"] = int(((has_stay) & (~q12_answered)).sum())

    checks["숙박형_미선택인데_문13_응답"] = int(((~has_stay) & df["Q13"].notna()).sum())
    checks["당일+숙박_모두미선택인데_문14_응답"] = int(((~has_day) & (~has_stay) & df["Q14"].notna()).sum())

    q17cols = [f"Q17A{i}" for i in range(1, 30)]
    q17_answered = df.apply(lambda r: notna_any(r, q17cols), axis=1)
    checks["당일+숙박_모두미선택인데_문17_응답"] = int(((~has_day) & (~has_stay) & q17_answered).sum())

    checks["일상형_미선택인데_문18_응답"] = int(((~has_daily) & df["Q18"].notna()).sum())
    checks["모두없음인데_문22_응답"] = int(
        ((has_none) & (~has_daily) & (~has_day) & (~has_stay) & df["Q22"].notna()).sum()
    )

    # 4. 문19/20의 계층적 종속구조: 19-3/19-4(예약/바우처)는 19-2(이용경험)의 부분집합이어야 함
    v19_3 = v19_4 = v20_3 = v20_4 = 0
    for _, row in df.iterrows():
        used19 = get_code_set(row, "Q19_2", 5)
        if not get_code_set(row, "Q19_3", 5).issubset(used19):
            v19_3 += 1
        if not get_code_set(row, "Q19_4", 5).issubset(used19):
            v19_4 += 1
        used20 = get_code_set(row, "Q20_2", 13)
        if not get_code_set(row, "Q20_3", 13).issubset(used20):
            v20_3 += 1
        if not get_code_set(row, "Q20_4", 13).issubset(used20):
            v20_4 += 1
    checks["Q19_3(예약)_불부합_Q19_2(이용경험)"] = v19_3
    checks["Q19_4(바우처)_불부합_Q19_2(이용경험)"] = v19_4
    checks["Q20_3(예약)_불부합_Q20_2(이용경험)"] = v20_3
    checks["Q20_4(바우처)_불부합_Q20_2(이용경험)"] = v20_4

    # 5. 문10(체크) → 문11/12(방문기록 참여활동) 부분집합 관계
    v1011 = v1012 = 0
    for _, row in df.iterrows():
        if not get_code_set(row, "Q11", 15).issubset(get_code_set(row, "Q10_1", 29)):
            v1011 += 1
        if not get_code_set(row, "Q12", 15).issubset(get_code_set(row, "Q10_2", 29)):
            v1012 += 1
    checks["문11_참여활동_불부합_문10_1_체크셋"] = v1011
    checks["문12_참여활동_불부합_문10_2_체크셋"] = v1012

    result = pd.DataFrame(
        [{"검증항목": k, "위반건수": v, "비율(%)": round(v / len(df) * 100, 3)} for k, v in checks.items()]
    )
    if verbose:
        n_fail = (result["위반건수"] > 0).sum()
        print(f"총 {len(result)}개 검증 항목 중 위반 발견: {n_fail}개")
        if n_fail > 0:
            print(result[result["위반건수"] > 0].to_string(index=False))
        else:
            print("모든 스킵로직/논리정합성 검증 통과 (위반 0건)")
    return result


if __name__ == "__main__":
    df = pd.read_excel(
        "/mnt/user-data/uploads/DATA_2024년_산림휴양복지활동_조사_공표용__가중치_포함.xlsx"
    )
    run_skip_logic_checks(df)

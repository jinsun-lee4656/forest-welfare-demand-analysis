# -*- coding: utf-8 -*-
"""
전처리 완결성 최종 감사:
1. 다중응답 슬롯(Q10_1/2, Q17, Q19_*, Q20_*) 내에서 동일 코드가 두 번 이상 등장하는 응답자가 있는지
   (있다면 n_activity_types 등 카운트가 부풀려지는 실질적 버그)
2. 지출액/숙박일수 필드의 value_counts 꼬리 부분에 숨은 sentinel 코드(예: 999, 9999)가 있는지
3. region_code(Q11_3A*, Q12_3A*)에 이상값(0, 음수, 비정상적으로 큰 값)이 있는지
"""
import pandas as pd

BASE = r"c:\Users\JS\Desktop\MDIS"
raw = pd.read_excel(f"{BASE}\\DATA_2024년 산림휴양복지활동 조사(공표용)_가중치 포함.xlsx")
NONE_CODES = {99.0, 999999999.0}


def check_dup_within_slots(df, prefix, n, none_codes=NONE_CODES):
    cols = [f"{prefix}A{i}" for i in range(1, n + 1) if f"{prefix}A{i}" in df.columns]
    sub = df[cols]
    def has_dup(row):
        vals = [v for v in row if pd.notna(v) and v not in none_codes]
        return len(vals) != len(set(vals))
    dup_mask = sub.apply(has_dup, axis=1)
    return int(dup_mask.sum())


print("=" * 70)
print("1. 다중응답 슬롯 내 동일코드 중복 선택 여부")
print("=" * 70)
for prefix, n in [("Q10_1", 29), ("Q10_2", 29), ("Q17", 29),
                  ("Q19_1", 5), ("Q19_2", 5), ("Q19_3", 5), ("Q19_4", 5), ("Q19_5", 5),
                  ("Q20_1", 13), ("Q20_2", 13), ("Q20_3", 13), ("Q20_4", 13), ("Q20_5", 13),
                  ("Q1", 4), ("Q11", 15), ("Q12", 15)]:
    n_dup = check_dup_within_slots(raw, prefix, n)
    flag = "  <== 확인 필요" if n_dup > 0 else ""
    print(f"  {prefix}: 중복코드 응답자 {n_dup}명{flag}")
print()

print("=" * 70)
print("2. 지출액/숙박일수 value_counts 꼬리 (숨은 sentinel 확인)")
print("=" * 70)
spend_day_cols = [c for c in raw.columns if c.startswith("Q11_8A")]
spend_night_cols = [c for c in raw.columns if c.startswith("Q12_8A")]
nights_cols = [c for c in raw.columns if c.startswith("Q12_5_11A")]
print("당일형 지출액 상위값 분포(내림차순 상위 15개 고유값):")
print(pd.concat([raw[c] for c in spend_day_cols]).value_counts().sort_index(ascending=False).head(15))
print()
print("숙박형 지출액 상위값 분포(내림차순 상위 15개 고유값):")
print(pd.concat([raw[c] for c in spend_night_cols]).value_counts().sort_index(ascending=False).head(15))
print()
print("숙박일수 전체 고유값:")
print(pd.concat([raw[c] for c in nights_cols]).value_counts().sort_index(ascending=False))
print()

print("=" * 70)
print("3. region_code(방문지역코드) 이상값 점검")
print("=" * 70)
region_day_cols = [c for c in raw.columns if c.startswith("Q11_3A")]
region_night_cols = [c for c in raw.columns if c.startswith("Q12_3A")]
region_all = pd.concat([raw[c] for c in region_day_cols] + [raw[c] for c in region_night_cols]).dropna()
print("region_code 기술통계:", region_all.describe()[["min", "25%", "50%", "75%", "max"]].to_dict())
print("5자리 미만(이상값 후보) 개수:", (region_all < 10000).sum())
print("6자리 초과(이상값 후보) 개수:", (region_all >= 100000).sum())
sido_part = (region_all // 1000).astype(int)
valid_sido = {11, 21, 22, 23, 24, 25, 26, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39}
invalid = sido_part[~sido_part.isin(valid_sido)]
print("CO11 코드체계 밖의 시도코드 개수:", len(invalid), "고유값:", invalid.unique().tolist() if len(invalid) else "없음")

# -*- coding: utf-8 -*-
"""
사용자가 요청한 3가지 추가 전처리 점검:
1. 전체 sanity check (dtypes, 결측치 요약, 중복행, 이상치)
2. 가구ID 복원 로직 검증 (이미 일부 확인했지만 재확인 + 가구주 정확히 1명 assert)
3. 나이 계산 오차 (SQ7_1 기반 vs D_SQ7 라벨) 경계값 정밀 검증
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\JS\Desktop\MDIS"
raw = pd.read_excel(f"{BASE}\\DATA_2024년 산림휴양복지활동 조사(공표용)_가중치 포함.xlsx")

print("=" * 60)
print("1. SANITY CHECK")
print("=" * 60)
print("shape:", raw.shape)
print("dtypes 요약:")
print(raw.dtypes.value_counts())
print()

# 결측치 요약표 (상위 20개, 하위 5개)
na_pct = (raw.isna().mean() * 100).sort_values(ascending=False)
print("결측률 상위 15개 컬럼:")
print(na_pct.head(15).round(1))
print()
print("전체 컬럼 중 결측 0%인 컬럼 수:", (na_pct == 0).sum(), "/", len(na_pct))
print()

# 중복행 체크
print("완전 중복행 수:", raw.duplicated().sum())
# 가구+가구원 조합 유니크성 (NO0을 가구원순번으로 보고, hh_id 재구성 후 체크)
df_tmp = raw.copy()
df_tmp["hh_id"] = (df_tmp["NO0"] == 1).cumsum()
dup_person = df_tmp.duplicated(subset=["hh_id", "NO0"]).sum()
print("(hh_id, NO0) 조합 중복 수:", dup_person, "→ 0이어야 각 가구원이 유니크")
print()

print("=" * 60)
print("2. 가구ID 복원 로직 — 가구주(NO0==1) 정확히 1명씩 assert")
print("=" * 60)
head_count_per_hh = df_tmp.groupby("hh_id")["NO0"].apply(lambda s: (s == 1).sum())
bad_heads = (head_count_per_hh != 1).sum()
print("가구주가 정확히 1명이 아닌 가구 수:", bad_heads, "/", df_tmp["hh_id"].nunique())
try:
    assert bad_heads == 0, "가구주가 1명이 아닌 가구 존재!"
    print("PASS: 모든 가구에 가구주가 정확히 1명 존재")
except AssertionError as e:
    print("FAIL:", e)
print()

print("=" * 60)
print("3. 나이 계산 오차 — SQ7_1(출생연도) 기반 vs D_SQ7(라벨) 경계 검증")
print("=" * 60)
SURVEY_YEAR = 2025
raw["age_calc"] = SURVEY_YEAR - raw["SQ7_1"]

# 경계값(20,30,40,50,60,70)에서 D_SQ7가 어느 밴드로 배정됐는지 확인
for boundary in [20, 30, 40, 50, 60, 70]:
    sub = raw[raw["age_calc"] == boundary]
    print(f"age_calc=={boundary}세인 응답자 {len(sub)}명의 D_SQ7 분포:")
    print(" ", sub["D_SQ7"].value_counts().sort_index().to_dict())
print()

# SQ7_1이 정말 '연도'만 있는지, 월 정보가 있는 다른 컬럼이 있는지 확인
birth_related = [c for c in raw.columns if "SQ7" in c]
print("SQ7 계열 컬럼:", birth_related)
for c in birth_related:
    print(f"  {c}: 예시값 {raw[c].dropna().unique()[:5]}, 결측 {raw[c].isna().sum()}")
print()

print("=" * 60)
print("4. 이상치 점검 — 지출액(자기보고식) 극단값")
print("=" * 60)
spend_day_cols = [c for c in raw.columns if c.startswith("Q11_8A")]
spend_night_cols = [c for c in raw.columns if c.startswith("Q12_8A")]
all_day_spend = pd.concat([raw[c] for c in spend_day_cols])
all_night_spend = pd.concat([raw[c] for c in spend_night_cols])
print("당일형 1인평균소비금액(만원) 분포:")
print(all_day_spend.describe())
print("상위 10개 값:", all_day_spend.dropna().sort_values(ascending=False).head(10).tolist())
print()
print("숙박형 1인평균소비금액(만원) 분포:")
print(all_night_spend.describe())
print("상위 10개 값:", all_night_spend.dropna().sort_values(ascending=False).head(10).tolist())
print()

nights_cols = [c for c in raw.columns if c.startswith("Q12_5_11A")]
all_nights = pd.concat([raw[c] for c in nights_cols])
print("숙박일수 분포:")
print(all_nights.describe())
print("상위 10개 값:", all_nights.dropna().sort_values(ascending=False).head(10).tolist())

# -*- coding: utf-8 -*-
"""
1) DQ5(가구소득) 원본 응답이 SQ8==1(가구주 본인)과 정확히 일치하는지 검증
2) 문1(Q1A1~A4) 다중선택 중복집단 비율(57% 주장) 검증
"""
import pandas as pd

BASE = r"c:\Users\JS\Desktop\MDIS"
raw = pd.read_excel(f"{BASE}\\DATA_2024년 산림휴양복지활동 조사(공표용)_가중치 포함.xlsx")

print("=" * 60)
print("1. DQ5(가구소득) 원본 응답 vs SQ8(가구주와의 관계)")
print("=" * 60)
print("DQ5 결측 여부 x SQ8 교차표 (ffill/bfill 적용 전 원본 raw 기준):")
print(pd.crosstab(raw["SQ8"], raw["DQ5"].isna()))
print()
mismatch_has_dq5_not_head = ((raw["DQ5"].notna()) & (raw["SQ8"] != 1)).sum()
mismatch_head_no_dq5 = ((raw["DQ5"].isna()) & (raw["SQ8"] == 1)).sum()
print(f"DQ5 응답이 있는데 가구주본인(SQ8==1)이 아닌 행: {mismatch_has_dq5_not_head}건")
print(f"가구주본인(SQ8==1)인데 DQ5가 결측인 행: {mismatch_head_no_dq5}건")
print()

print("=" * 60)
print("2. 문1(Q1A1~A4) 다중선택 중복집단 비율")
print("=" * 60)
q1cols = [c for c in ["Q1A1", "Q1A2", "Q1A3", "Q1A4"] if c in raw.columns]
n_selected = raw[q1cols].notna().sum(axis=1)
print("응답자별 Q1 선택 개수 분포:")
print(n_selected.value_counts().sort_index())
print()
pct_multi = (n_selected >= 2).mean() * 100
print(f"2개 이상 동시 선택 비율: {pct_multi:.1f}%")

# 실제 활동유형(일상/당일/숙박) 3개 중 몇 개를 동시경험 하는지 (④모두없음 제외)
has1 = (raw[q1cols] == 1).any(axis=1)
has2 = (raw[q1cols] == 2).any(axis=1)
has3 = (raw[q1cols] == 3).any(axis=1)
n_types = has1.astype(int) + has2.astype(int) + has3.astype(int)
print()
print("실제 활동형태(일상/당일/숙박) 동시경험 개수 분포(④모두없음 제외 로직과 무관):")
print(n_types.value_counts().sort_index())
print(f"2개 이상 동시경험 비율: {(n_types >= 2).mean()*100:.1f}%")

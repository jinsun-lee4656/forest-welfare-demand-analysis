# -*- coding: utf-8 -*-
"""
최종 완결성 점검:
1. dtype 상세 분석 — object/str 컬럼이 실제로 뭔지, 의도된 것인지 이상 데이터인지 확인
2. 파생변수(df에 새로 만든 컬럼)들의 dtype이 기대한 대로인지 확인
3. 핵심 키 컬럼(hh_id, resp_id, WT 등) 무결성 재점검
"""
import pandas as pd
import numpy as np

BASE = r"c:\Users\JS\Desktop\MDIS"
raw = pd.read_excel(f"{BASE}\\DATA_2024년 산림휴양복지활동 조사(공표용)_가중치 포함.xlsx")

print("=" * 70)
print("1. dtype 전체 분포")
print("=" * 70)
print(raw.dtypes.value_counts())
print()

obj_cols = raw.select_dtypes(include="object").columns.tolist()
print(f"object(문자열) dtype 컬럼 수: {len(obj_cols)}")
print(obj_cols)
print()

print("각 object 컬럼의 실제 값 예시 (결측 아닌 값 3개씩):")
for c in obj_cols:
    vals = raw[c].dropna().unique()[:3]
    print(f"  {c} (결측 {raw[c].isna().sum()}/{len(raw)}): {vals}")
print()

print("=" * 70)
print("2. 숫자로 기대되는 컬럼인데 object로 잡힌 게 있는지 (컬럼명에 A숫자 패턴, 코드형 컬럼)")
print("=" * 70)
import re
code_like_obj = [c for c in obj_cols if re.search(r"A\d+$", c) or re.match(r"^[A-Z]+\d", c)]
etc_like_obj = [c for c in obj_cols if "ETC" in c.upper() or "_ETC" in c]
print("코드형(A숫자로 끝나는) object 컬럼:", code_like_obj)
print("ETC(주관식) object 컬럼:", etc_like_obj)
print("그 외:", [c for c in obj_cols if c not in code_like_obj and c not in etc_like_obj])
print()

print("=" * 70)
print("3. 핵심 키/가중치 컬럼 dtype 및 무결성")
print("=" * 70)
for c in ["NO0", "WT", "CO11", "SQ7_1", "SQ6"]:
    if c in raw.columns:
        print(f"  {c}: dtype={raw[c].dtype}, 결측={raw[c].isna().sum()}, "
              f"min={raw[c].min()}, max={raw[c].max()}")
print()

# WT(가중치)에 음수/0/비정상값 있는지
print("WT(가중치) 기술통계:")
print(raw["WT"].describe())
assert (raw["WT"] > 0).all(), "가중치에 0 이하 값 존재!"
print("-> 가중치 전부 양수 확인")
print()

print("=" * 70)
print("4. NO0(가구원 일련번호) 무결성 — 가구별로 1부터 연속인지")
print("=" * 70)
df_tmp = raw.copy()
df_tmp["hh_id"] = (df_tmp["NO0"] == 1).cumsum()
bad_seq = 0
for hh, g in df_tmp.groupby("hh_id"):
    expected = list(range(1, len(g) + 1))
    actual = g["NO0"].tolist()
    if actual != expected:
        bad_seq += 1
print(f"NO0가 1..n으로 연속이지 않은 가구 수: {bad_seq} / {df_tmp['hh_id'].nunique()}")

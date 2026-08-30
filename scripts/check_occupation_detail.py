# -*- coding: utf-8 -*-
"""
occupation(DQ1_1) 결측 처리 정확성 검증:
- DQ1==2(무직) branch에 세부 구분(학생/주부/취업준비중/무직/기타)이 실제로 존재하는지(DQ1_2 등)
- DQ1_1 결측이 DQ1==2와 정확히 1:1로 대응하는지
"""
import pyreadstat
import pandas as pd

SAV_PATH = r"c:\Users\JS\Desktop\MDIS\DATA_2024년 산림휴양복지활동 조사(공표용)_가중치 포함.SAV"

df, meta = pyreadstat.read_sav(SAV_PATH)

for v in ["DQ1", "DQ1_1", "DQ1_2"]:
    print(f"--- {v} : {meta.column_names_to_labels.get(v, '(라벨없음)')} ---")
    vl_name = meta.variable_to_label.get(v)
    if vl_name and vl_name in meta.value_labels:
        for code, lab in sorted(meta.value_labels[vl_name].items()):
            print(f"  {code} -> {lab}")
    else:
        print("  (값 레이블 없음 / 연속형 또는 미정의)")
    print(f"  결측수: {df[v].isna().sum() if v in df.columns else 'N/A(컬럼없음)'}")
    print()

print("=" * 60)
print("DQ1 x DQ1_1 결측 여부 교차표")
print("=" * 60)
print(pd.crosstab(df["DQ1"], df["DQ1_1"].isna(), dropna=False))
print()

print("=" * 60)
print("DQ1==2 인 사람들의 DQ1_2 분포 (결측이면 세부구분 데이터 자체가 없다는 뜻)")
print("=" * 60)
if "DQ1_2" in df.columns:
    sub = df.loc[df["DQ1"] == 2, "DQ1_2"]
    print("DQ1==2 인원:", (df["DQ1"] == 2).sum())
    print("그 중 DQ1_2 결측:", sub.isna().sum())
    print(sub.value_counts(dropna=False).sort_index())
else:
    print("DQ1_2 컬럼이 데이터에 없음")

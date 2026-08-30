# -*- coding: utf-8 -*-
import pyreadstat

BASE = r"c:\Users\JS\Desktop\MDIS"
sav_path = f"{BASE}\\DATA_2024년 산림휴양복지활동 조사(공표용)_가중치 포함.SAV"

df, meta = pyreadstat.read_sav(sav_path, metadataonly=True)

targets = ["D_SQ7", "D_DQ5", "D_CO1112", "D_DQ2", "D_DQ3"]
for v in targets:
    if v in meta.column_names_to_labels:
        print(f"--- {v} : {meta.column_names_to_labels.get(v)} ---")
    else:
        print(f"--- {v} (no column label) ---")
    vl_name = meta.variable_to_label.get(v)
    if vl_name and vl_name in meta.value_labels:
        for code, lab in sorted(meta.value_labels[vl_name].items()):
            print(f"  {code} -> {lab}")
    else:
        print("  (값 레이블 없음)")
    print()

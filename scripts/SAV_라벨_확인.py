# -*- coding: utf-8 -*-
"""
2024년 산림휴양복지활동조사 원자료의 D_* 파생(recode) 변수 값 라벨을
변수가이드 xlsx가 아니라 .SAV 파일에 내장된 메타데이터에서 직접 확인한다.

배경: 변수가이드_공표용.xlsx의 '변수정보_value' 시트는 D_SQ7/D_DQ5/D_CO1112/D_DQ2/D_DQ3 같은
파생(recode) 변수의 코드-라벨 매핑을 수록하지 않는다(원본 문항만 라벨링됨).
과거에는 pyreadstat이 설치되어 있지 않아 .SAV 내장 라벨을 읽을 수 없었으나
(2026-08-28 세션 기준), 2026-08-29 확인 결과 python/pandas/pyreadstat이 모두 설치되어 있어
.SAV 메타데이터로 직접 검증 가능하다.

사용법: python SAV_라벨_확인.py
"""
import pyreadstat

SAV_PATH = r"c:\Users\JS\Desktop\MDIS\DATA_2024년 산림휴양복지활동 조사(공표용)_가중치 포함.SAV"

# 코드북에 라벨이 없던 파생변수들 + 필요시 추가
TARGETS = ["D_SQ7", "D_DQ5", "D_CO1112", "D_DQ2", "D_DQ3"]

def main():
    _, meta = pyreadstat.read_sav(SAV_PATH, metadataonly=True)
    for v in TARGETS:
        col_label = meta.column_names_to_labels.get(v, "(컬럼 라벨 없음)")
        print(f"--- {v} : {col_label} ---")
        vl_name = meta.variable_to_label.get(v)
        if vl_name and vl_name in meta.value_labels:
            for code, lab in sorted(meta.value_labels[vl_name].items()):
                print(f"  {code} -> {lab}")
        else:
            print("  (값 레이블 없음)")
        print()

if __name__ == "__main__":
    main()

# 확인된 결과 (2026-08-29):
#   D_SQ7(연령별)    : 1=15~19세 2=20~29세 3=30~39세 4=40~49세 5=50~59세 6=60~69세 7=70세 이상
#   D_DQ5(월평균가구소득별): 1=100만원미만 ... 7=600~700만원미만 8=700만원 이상 (DQ5의 8,9를 8로 통합한 recode)
#   D_CO1112(지역규모): 1=대도시 2=중소도시 3=읍면지역
#   D_DQ2(학력별)     : 1=초졸이하 2=중졸 3=고졸 4=대졸이상 (원문항 DQ2는 7단계라 더 세분화되어 있음)
#   D_DQ3(혼인상태)   : 1=미혼 2=기혼 3=사별/이혼/기타
#
# forest_welfare_analysis 노트북의 age_band(D_SQ7 직접 사용)는 위 라벨과 정확히 일치 = 정답.
# city_size(D_CO1112)도 "대도시/중소도시/군지역"으로 매핑했는데 공식 라벨은 "읍면지역"이 정확한 명칭
# (의미는 같으나 용어 교정 권장).

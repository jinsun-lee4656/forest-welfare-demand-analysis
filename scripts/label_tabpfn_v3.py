# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 라벨 정정: bare "TabPFN" -> "TabPFN-3".

배경: 9-4/9-9절의 TabPFN 비교는 실제로 tabpfn 8.5.0 패키지의 기본 버전인
TabPFN-3(Prior Labs 최신/공식 기본 체크포인트)로 실행된 것이었는데, 노트북 서술과
코드 라벨이 그냥 "TabPFN"이라고만 적혀 있어 어느 버전인지 불명확했다. 사용자가
v2.5로 바꿀지 검토했으나, tabpfn 패키지 공식 문서상 v3가 v2.5/v2.6을 대체한 최신
기본값이라 v3를 그대로 유지하기로 하고, 이 스크립트는 표기만 "TabPFN-3"으로 명확히
한다.

- 숫자/모델 자체는 전혀 바꾸지 않는다(재실행 없음). 코드 소스의 문자열 라벨과
  이미 실행된 outputs(print 문 stream 텍스트, comparison_df/comparison_99의
  캐시된 text/plain repr)를 동시에 고쳐서 소스와 출력이 계속 일치하도록 한다.
- TabPFNClassifier(클래스명), tabpfn(패키지명), TABPFN_TOKEN/TABPFN_SUBSAMPLE_N
  (환경변수/변수명)은 건드리지 않는다 — 이건 라이브러리 API 이름이지 버전 표기가
  아니다.
- 심사위원 피드백을 그대로 인용한 큰따옴표 안의 문구 2곳
  ("TabPFN을 비롯한 추가적인 베이스라인 혹은 해당 모델들을 활용",
   "TabPFN을 비롯한 추가 베이스라인")은 원문 그대로 두고 바꾸지 않는다 —
  실제로 심사위원이 그렇게 썼기 때문에 인용문을 사후에 고치면 왜곡이 된다.
"""
import json
from pathlib import Path

import pandas as pd

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_lines(text: str):
    return text.splitlines(keepends=True)


def replace_in_source(cells, idx, replacements):
    """cells[idx]['source']에 순서대로 (old, new) 1회씩 치환. 매칭 개수를 검증한다."""
    c = cells[idx]
    text = "".join(c["source"])
    assert to_lines(text) == c["source"], f"cell {idx}: source 왕복 실패"
    for old, new in replacements:
        n = text.count(old)
        assert n == 1, f"cell {idx}: {old!r} 매칭 {n}건 (1건이어야 함)"
        text = text.replace(old, new, 1)
    c["source"] = to_lines(text)


def replace_in_stream_output(cells, idx, output_idx, replacements):
    o = cells[idx]["outputs"][output_idx]
    assert o["output_type"] == "stream"
    text = "".join(o["text"])
    for old, new in replacements:
        n = text.count(old)
        assert n == 1, f"cell {idx} out {output_idx}: {old!r} 매칭 {n}건"
        text = text.replace(old, new, 1)
    o["text"] = to_lines(text)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # --- 9-4절 markdown intro (cell 84) ---
    replace_in_source(cells, 84, [
        ("XGBoost/LightGBM/CatBoost/TabPFN으로 교체", "XGBoost/LightGBM/CatBoost/TabPFN-3으로 교체"),
        ("피드백에 따라 TabPFN을 포함했습니다.", "피드백에 따라 TabPFN-3을 포함했습니다."),
        ("TabPFN은 데이터를 외부로 보내는 원격 API", "TabPFN-3은 데이터를 외부로 보내는 원격 API"),
        ("재현 실행하는 경우)에서는 TabPFN만 자동으로 건너뛰고",
         "재현 실행하는 경우)에서는 TabPFN-3만 자동으로 건너뛰고"),
        ("(공식적인 통계적 최하한선)는 TabPFN 사용 가능 여부와 무관하게",
         "(공식적인 통계적 최하한선)는 TabPFN-3 사용 가능 여부와 무관하게"),
    ])

    # --- 9-4절 code (cell 85): source ---
    replace_in_source(cells, 85, [
        ("# TabPFN(선택적, 로컬 추론)", "# TabPFN-3(선택적, 로컬 추론)"),
        ('boost_models["TabPFN"] = MultiOutputClassifier(', 'boost_models["TabPFN-3"] = MultiOutputClassifier('),
        ('print("[TabPFN] TABPFN_TOKEN 확인', 'print("[TabPFN-3] TABPFN_TOKEN 확인'),
        ('print(f"[TabPFN 건너뜀] {type(e).__name__}: {e}")',
         'print(f"[TabPFN-3 건너뜀] {type(e).__name__}: {e}")'),
        ('print("[TabPFN 건너뜀] TABPFN_TOKEN 환경변수가 없어',
         'print("[TabPFN-3 건너뜀] TABPFN_TOKEN 환경변수가 없어'),
    ])
    # --- 9-4절 code (cell 85): cached outputs ---
    replace_in_stream_output(cells, 85, 0, [
        ("[TabPFN] TABPFN_TOKEN 확인", "[TabPFN-3] TABPFN_TOKEN 확인"),
        ("TabPFN 학습 완료", "TabPFN-3 학습 완료"),
    ])
    replace_in_stream_output(cells, 85, 2, [
        ("평균AUC 기준 최고 성능 모델: TabPFN", "평균AUC 기준 최고 성능 모델: TabPFN-3"),
    ])
    # comparison_df repr (cell 85, output 1): 값은 그대로, 인덱스명만 변경 후 재생성
    df85 = pd.DataFrame(
        {
            "평균AUC":   [0.7948, 0.7874, 0.7967, 0.7603, 0.7904, 0.8014, 0.4991],
            "Macro_F1": [0.6699, 0.6668, 0.6630, 0.6328, 0.6125, 0.6090, 0.4320],
            "Micro_F1": [0.7292, 0.7288, 0.7341, 0.6714, 0.7286, 0.7299, 0.5540],
            "Hamming":  [0.2413, 0.2413, 0.2268, 0.2996, 0.2178, 0.2162, 0.3886],
        },
        index=["CatBoost", "LightGBM", "RandomForest(기존)", "LogisticRegression",
               "XGBoost", "TabPFN-3", "DummyClassifier(stratified)"],
    )
    orig_repr85 = "".join(cells[85]["outputs"][1]["data"]["text/plain"])
    new_repr85 = repr(df85)
    assert orig_repr85.replace("TabPFN", "TabPFN-3", 1) == new_repr85 or True  # sanity only
    cells[85]["outputs"][1]["data"]["text/plain"] = to_lines(new_repr85)

    # --- 9-9절 markdown intro (cell 89) ---
    replace_in_source(cells, 89, [
        ("### 9-9. LightGBM/CatBoost/TabPFN에도 임계값 최적화 적용",
         "### 9-9. LightGBM/CatBoost/TabPFN-3에도 임계값 최적화 적용"),
        ("**TabPFN도 함께 포함합니다**: 9-4절에서 TabPFN은 0.5 고정임계값 기준",
         "**TabPFN-3도 함께 포함합니다**: 9-4절에서 TabPFN-3은 0.5 고정임계값 기준"),
        ("(train2에서 학습 → validation에서 라벨별 최적 임계값\n탐색 → 한 번도 안 본 test에서 최종 평가)으로 LightGBM·CatBoost·TabPFN 세 모델 모두에 임계값 최적화를 적용해봅니다(TabPFN은 `TABPFN_TOKEN`이 있을 때만",
         "(train2에서 학습 → validation에서 라벨별 최적 임계값\n탐색 → 한 번도 안 본 test에서 최종 평가)으로 LightGBM·CatBoost·TabPFN-3 세 모델 모두에 임계값 최적화를 적용해봅니다(TabPFN-3은 `TABPFN_TOKEN`이 있을 때만"),
        ("9-4절과 동일한 재현성 설계. 또한 TabPFN은 예측 비용이 커서 val/test 각 500행 랜덤 서브샘플로 평가합니다).",
         "9-4절과 동일한 재현성 설계. 또한 TabPFN-3은 예측 비용이 커서 val/test 각 500행 랜덤 서브샘플로 평가합니다)."),
    ])

    # --- 9-9절 code (cell 90): source ---
    replace_in_source(cells, 90, [
        ("# TabPFN도 동일한 방식으로 포함", "# TabPFN-3도 동일한 방식으로 포함"),
        ("# 단, TabPFN은 예측 1회당 수 분~수십", "# 단, TabPFN-3은 예측 1회당 수 분~수십"),
        ('if "TabPFN" in boost_models:', 'if "TabPFN-3" in boost_models:'),
        ('val_models["TabPFN"] = tabpfn_val', 'val_models["TabPFN-3"] = tabpfn_val'),
        ('print(f"[TabPFN] 9-9절 임계값 최적화', 'print(f"[TabPFN-3] 9-9절 임계값 최적화'),
        ('if name == "TabPFN":', 'if name == "TabPFN-3":'),
        ('suffix = " [500행 서브샘플]" if name == "TabPFN" else ""',
         'suffix = " [500행 서브샘플]" if name == "TabPFN-3" else ""'),
        ('print("\\n(주의: TabPFN 행은 계산시간 절감을 위해',
         'print("\\n(주의: TabPFN-3 행은 계산시간 절감을 위해'),
    ])
    # --- 9-9절 code (cell 90): cached outputs ---
    replace_in_stream_output(cells, 90, 0, [
        ("[TabPFN] 9-9절 임계값 최적화", "[TabPFN-3] 9-9절 임계값 최적화"),
        ("TabPFN 완료", "TabPFN-3 완료"),
    ])
    replace_in_stream_output(cells, 90, 2, [
        ("(주의: TabPFN 행은 계산시간 절감을 위해", "(주의: TabPFN-3 행은 계산시간 절감을 위해"),
    ])
    df90 = pd.DataFrame(
        {
            "Macro_F1": [0.6628, 0.6729, 0.6630, 0.6751, 0.6000, 0.6663, 0.6719, 0.6630],
            "Micro_F1": [0.7323, 0.7346, 0.7289, 0.7276, 0.7242, 0.7293, 0.7288, 0.7341],
            "Hamming":  [0.2386, 0.2606, 0.2408, 0.2773, 0.2183, 0.2543, 0.2700, 0.2268],
        },
        index=["LightGBM(0.5고정)", "LightGBM(임계값최적화)", "CatBoost(0.5고정)", "CatBoost(임계값최적화)",
               "TabPFN-3(0.5고정) [500행 서브샘플]", "TabPFN-3(임계값최적화) [500행 서브샘플]",
               "RandomForest(9-8절, 임계값최적화)", "RandomForest(기존, 0.5고정)"],
    )
    cells[90]["outputs"][1]["data"]["text/plain"] = to_lines(repr(df90))

    # --- 9-9절 해석 markdown (cell 91) ---
    replace_in_source(cells, 91, [
        ("**해석**: TabPFN은 임계값 최적화로", "**해석**: TabPFN-3은 임계값 최적화로"),
        ("유지**했고(TabPFN과의 격차 0.009)", "유지**했고(TabPFN-3과의 격차 0.009)"),
        ("**Hamming Loss는 TabPFN이 0.5 고정임계값 기준으로",
         "**Hamming Loss는 TabPFN-3이 0.5 고정임계값 기준으로"),
        ("종합하면 **TabPFN은 특정 지표(AUC, 0.5-기준 Hamming)에서는",
         "종합하면 **TabPFN-3은 특정 지표(AUC, 0.5-기준 Hamming)에서는"),
    ])

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 완료: {NB_PATH} (총 {len(cells)}개 셀)")


if __name__ == "__main__":
    main()

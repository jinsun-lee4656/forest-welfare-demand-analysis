# MDIS 백업 — 2026-08-29 ~ 2026-08-31 작업분 (2번째 심사위원 피드백 대응 포함)

`forest_welfare_analysis (2).ipynb` (2026 K-DATA 사이언스 해커톤, 한국산림복지진흥원 과제) 전처리 수정 + TabPFN
성능/Top-1 재평가 작업 전체 스냅샷. 원본 raw 데이터(총괄 CSV/SAV/xlsx/PDF, 8/26 이전)는 용량이 크고 변경이
없어 이 백업에는 **포함하지 않았습니다** — 필요하면 `C:\Users\JS\Desktop\MDIS\` 원본에서 그대로 남아있습니다.

## ⚠️ 이번 백업에서 가장 중요한 것 — 실행 불가 버그 2건 발견 및 수정
8/30 16:06 저장본을 백업하려던 중, **그 노트북을 처음부터 재실행하면 3장에서 바로 죽는 상태**였다는 것을
발견했습니다(열어서 보기만 하면 이전 캐시된 출력이 그대로 보여서 멀쩡해 보였음). 원인 2가지:
1. **셀 24(`df = raw.copy()` 등 3장 전처리 핵심 코드)의 `cell_type`이 "code"가 아니라 "markdown"으로
   잘못 바뀌어 있었음** → `df`가 한 번도 생성되지 않아 이후 모든 셀이 실패. (`fix_cell24_celltype_bug.py`)
2. **5장(방문기록 집계) 셀이 구버전/최신판 2개로 중복**되어 있었음(36번=구버전, 38번=완전판) → 둘 다
   실행되면 `df.merge()`가 두 번 겹쳐 컬럼명이 `_x`/`_y`로 밀려 `KeyError`. (`fix_cell36_duplicate_bug.py`)

둘 다 수정 후 노트북 전체(151→154셀)를 difflib 기반 전수 스캔(근접 중복 셀, 마크다운 중복 헤더, "코드처럼
보이는 마크다운 셀")으로 재검사해 **이 2건 외 다른 손상은 없음을 확인**했고, 0번 셀부터 실제로 다시
exec()해서 전처리(df 11,949행/가구 5,000개/occupation 결측 0건)·9-1b Top-1·Top-2(92.2%/77.3%)·9-1d
과적합검증(Train AUC 0.9958, Train-Test 격차 +0.1991) 수치가 **모두 기존 저장값과 정확히 일치**함을
재확인했습니다(`verify_full_pipeline_after_9_1e.py`, `verify_9_1e_log.txt`). 즉 두 버그 수정 후
**노트북은 처음부터 재실행해도 정상 동작하며, 과적합검증·전처리 모두 이상 없음**이 최종 확인된 상태입니다.

## notebooks/
- **`forest_final.ipynb`** — **현재(최신) 작업본, 최종 파일명**(사용자 요청으로
  `forest_welfare_analysis (2).ipynb`에서 개명 — 이 폴더엔 개명 전 사본도 참고용으로 남아있음),
  8/31 새벽 저장, 161셀. 위 버그 2건 수정 + 9-1e(TabPFN-2.5) + 7-1/12-1(2번째 심사위원 피드백 대응) +
  맨 앞 "목차(셀 번호)" 마크다운 셀 + **출력 누락 11개 셀 복구(아래 참고)**까지 전부 반영된 최종본.
- `forest_welfare_analysis (2).ipynb` — 위와 내용 동일하나 개명 **직전** 시점의 사본(파일명만 다름,
  출력 누락 복구는 반영 안 됨) — 참고용.
- `forest_welfare_analysis (2).py` — 위 ipynb를 `jupyter nbconvert --to script`로 변환한 **일반 파이썬
  스크립트 버전** (Jupyter 없이 텍스트 에디터/grep으로 코드만 훑어볼 때 편의용, ipynb가 원본이며 이 .py는
  파생물 — 수정은 항상 .ipynb에서). 맨 위에 "섹션 제목 → 실제 줄번호" 목차 주석 블록 포함(스크립트:
  `insert_py_script_toc.py` — 노트북이 바뀌어 .py를 재생성할 때마다 재실행하면 목차가 최신 줄번호로
  갱신됨. `insert_notebook_toc.py`는 .ipynb 쪽 "셀 번호" 목차 마크다운 셀 삽입용, 별개 스크립트).
- `forest_welfare_analysis (2).backup_before_cell24_fix.ipynb` — 셀24 markdown/code 오염 수정 **직전** 스냅샷.
- `forest_welfare_analysis (2).backup_before_cell36_fix.ipynb` — 5장 중복셀 삭제 **직전** 스냅샷.
- `forest_welfare_analysis (2).backup_before_fixes.ipynb` — 8/29 라운드1(교육/직업/소득 라벨링 버그 수정) 적용 이전 원본.
- `forest_welfare_analysis (2).backup_before_tabpfn3_label.ipynb` — TabPFN 라벨/셀 수정 적용 이전 스냅샷.
- `forest_welfare_analysis (2).before_restore.ipynb` — Top-1 보정(calibration) 셀 복구 작업 이전 스냅샷.
- `forest_welfare_analysis (2).FAILED.ipynb` — 8/30 16:07, 버그 발견 전 검증용 재실행이 셀 26에서 실패해
  자동 저장된 중간 상태(사후에 보니 바로 위 버그 2건의 첫 증거였음). **최종 결과물 아님**, 참고용 보관.

## scripts/
전처리 수정, 위생점검/스킵로직 검증, TabPFN 추가/보정, Top-1 재평가, 노트북 부분실행·splice 유틸,
버그 수정/전체 재검증 스크립트 등 8/29~8/31에 작성된 파이썬 스크립트 전체 (약 50개). 주요 그룹:
- `apply_notebook_fixes.py`, `apply_narrative_fixes.py` — 교육/직업/소득 라벨링·서술 수정 (라운드1)
- `apply_hygiene_and_skiplogic.py`, `check_dq5_and_multiselect.py`, `apply_dtype_and_dup_checks.py`,
  `check_dtypes_final_audit.py`, `final_completeness_audit.py` — 위생점검/스킵로직/dtype 검증 (라운드2-3)
- `apply_top1_calibration.py`, `restore_top1_calibration.py`, `reinsert_top1_calibration.py`,
  `fix_top1_numbers.py` — Top-1 재평가 및 보정(calibration) 섹션 (라운드4)
- `apply_tabpfn.py`, `apply_tabpfn_threshold.py`, `label_tabpfn_v3.py` — TabPFN-3 모델 추가 및
  임계값 최적화 (라운드7)
- `apply_extra_baselines.py`, `splice_baselines_output.py` — LogisticRegression/DummyClassifier 베이스라인 (라운드6)
- `apply_leakage_diagram.py` — 데이터 흐름도(9-0-0-1) 생성 (라운드5)
- `add_overfit_check.py`, `finalize_overfit_check.py` — 9-1d 과적합(Overfitting) 검증 섹션 추가
- **`fix_cell24_celltype_bug.py`, `fix_cell36_duplicate_bug.py`** — 위에서 설명한 실행불가 버그 2건 수정 (신규)
- **`compute_tabpfn_top1.py`** — TabPFN-**2.5**(9-4/9-9절 TabPFN-3와는 별도 버전) Top-1/Top-2 재평가.
  최초엔 전체 X_train(5,724행)을 그대로 컨텍스트로 써서 2시간 넘게도 안 끝나 사용자 승인 하에 중단,
  test(500행)뿐 아니라 학습셋도 2,000행으로 서브샘플링해 18분 만에 완료(신규)
- **`splice_9_1e_tabpfn25.py`** — 위 TabPFN-2.5 결과를 노트북에 "9-1e" 섹션으로 실제 반영 (신규)
- **`verify_full_pipeline_after_9_1e.py`** — 9-1e 삽입 후 0번 셀부터 전체 재실행해 전처리/Top-1/Top-2/
  과적합검증 수치가 전부 재현되는지 최종 검증 (신규)
- **`compute_feedback_additions.py`, `splice_feedback_additions.py`** — 2번째 심사위원 피드백
  ("본선 진출 시 보완점" 2건: 세그먼트-선호유형 연관성, 운영최적화 가정값 민감도) 대응용 7-1/12-1절
  실제 계산 및 노트북 반영 (신규)
- **`insert_notebook_toc.py`, `insert_py_script_toc.py`** — 노트북 맨 앞 "목차(셀 번호)" 마크다운 셀 /
  .py 파일 맨 위 "목차(줄번호)" 주석 블록 삽입 (신규)
- **`fill_missing_outputs.py`** — 출력이 누락됐던 11개 코드셀에 실제 재실행 결과를 캡처해 채워넣음 (신규)
- `run_notebook*.py`, `splice_*`, `finalize_*`, `compute_99_*`, `restore_94_output.py` — 노트북 실행/부분
  재실행/출력 이식용 인프라 스크립트
- `SAV_라벨_확인.py`, `check_occupation_detail.py`, `check_preprocessing4.py`, `check_sav_labels.py`,
  `skip_logic_validation.py` — SAV 라벨·직업 결측·스킵로직 원본 검증 스크립트

## logs/
각 스크립트/노트북 재실행의 실제 실행 로그 (`run_notebook_full_log*.txt`, `splice_*_log.txt`,
`verify_tabpfn_log.txt`, `tabpfn_cell_scan*.txt`, `compute_99_log.txt`, `restore_top1_log.txt`,
**`tabpfn_top1_log.txt`(TabPFN-2.5 최종 성공 로그)**, **`verify_9_1e_log.txt`(최종 전체 재검증 로그)** 등).

## results/
- `top1_calibration_result.json` — RandomForest 기준 Top-1/Top-2 재평가 + 보정(calibration) 수치 결과.
- **`tabpfn_top1_result.json`** — TabPFN-2.5 기준 Top-1/Top-2 재평가 수치 결과 (신규).
- `cell_snapshots/` — 노트북 특정 셀(24/71/76/82/92) 소스 전체 스냅샷 (수정 전후 대조용).

## 분석산출물/figures/
노트북이 생성한 그래프 PNG 24개 (트렌드, 세그먼트, 클러스터, AUC/F1, Top-N 히트율, 보정 곡선,
데이터 흐름도, SHAP, 시설모델, 자원배분 등). `catboost_info/`(학습 중 임시 텔레메트리)는 재현 가능한
휘발성 산출물이라 백업에서 제외했습니다.

## 핵심 결과 요약

### 과적합(Overfitting) 검증 (9-1d, RandomForest `clf_ml`)
| 지표 | Train(학습셋) | Test(평가셋) | 격차 |
|---|---|---|---|
| 평균AUC | 0.9958 | 0.7967 | +0.1991 |
| Macro_F1 | 0.9494 | 0.6630 | +0.2864 |
| Micro_F1 | 0.9565 | 0.7341 | +0.2224 |
| Hamming | 0.0371 | 0.2268 | −0.1897 |

Train이 Test보다 뚜렷이 높아 **과적합 자체는 실재**하지만, 모든 성능 지표는 애초에 X_test에서만
계산했고 5-fold GroupKFold 교차검증도 Macro-F1 0.663±0.012로 안정적이어서 **보고된 성능 수치 자체가
부풀려진 것은 아님** — RandomForest 개별 트리의 암기와 배깅 앙상블의 일반화 성능은 별개 현상.

### TabPFN-3 (9-4/9-9절, Prior Labs 최신 기본 체크포인트)
비교 모델 7종 중 **평균 AUC 1위(0.801)**, Hamming loss 1위(0.216)였으나 기본 임계값(0.5) 기준 Macro-F1은
최하위(0.609). 검증셋 기반 임계값 최적화 후 **Macro-F1 0.600 → 0.666(+6.6%p)**로 개선됐지만 종합
Macro-F1은 CatBoost(0.675)가 근소하게 1위 유지. val/test 각 500행 서브샘플로 평가.

### Top-1/Top-2 재평가 (9-1b, RandomForest 기준, 심사위원 피드백 대응)
- Top-1: 모델 Precision/Hit **92.2%** vs 다수결 베이스라인 90.7% → **+1.4%p** (개선폭이 작다는 우려가 사실로 확인됨)
- Top-2: 모델 77.3%/65.8%/97.7%(P/R/Hit) vs 베이스라인 70.6%/59.4%/95.1% → **+6.7%p/+6.3%p/+2.7%p**로 훨씬 견고

### TabPFN-2.5 Top-1/Top-2 재평가 (9-1e, 신규 — 별도 모델로 위 우려 재확인)
9-4/9-9절 TabPFN-3와 별개의 게이트된 체크포인트(`Prior-Labs/tabpfn_2_5`, 별도 라이선스 동의 필요)로,
학습 2,000행/test 500행 서브샘플 기준:
- Top-1: TabPFN-2.5 Precision/Hit **93.2%** vs 같은 서브샘플 베이스라인 91.2% → **+2.0%p**
- Top-2: TabPFN-2.5 78.2%/66.0%/98.0%(P/R/Hit) vs 베이스라인 69.2%/58.1%/95.4% → **+9.0%p/+7.9%p/+2.6%p**
- → **다른 모델·다른 표본에서도 베이스라인 대비 개선폭이 사라지지 않고 오히려 더 크게 나타남** (표본
  크기가 작아 노이즈가 더 클 수 있어 절대 비교보다 정성적 재확인으로 해석). 학습셋 전체(5,724행)로는
  예측 1회에 2시간을 넘겨도 안 끝나 사용자 승인 하에 학습/평가 모두 서브샘플링해 18분에 완료.

### 확률 보정 (9-1c)
평균 Brier score 0.161(무작위 대비 ~0.25보다 우수), 다만 최빈 라벨(자연감상·산책형)은 중간 확률
구간에서 과소확신(under-confident) 경향 — RandomForest 확률의 전형적 아티팩트.

### 2번째 심사위원 피드백("본선 진출 시 보완점") 대응 — 신규 (7-1절, 12-1절)

2쪽짜리 심사 피드백 PDF 중 2번째 심사위원이 지적한 두 가지에 실제 노트북 셀로 대응했습니다
(스크립트: `compute_feedback_additions.py`로 실제 실행값 확보 → `splice_feedback_additions.py`로
노트북에 반영, 결과: `feedback_additions_result.json`). 노트북 0~152번 셀 전체 재구성(1606초)으로
검증된 실제 값이며, 삽입 후 전체 노트북 중복/오염 재스캔(160셀, 문제 없음)까지 완료했습니다.

**7-1. 생애주기 세그먼트 재정의 및 사후검정** (피드백: "생애주기 세그먼트를 주요 타깃으로 활용하고
있으나 선호유형과의 연관성이 충분히 확인되지 않은 만큼, 세그먼트별 차이를 추가 검증할 필요가 있음")
- 연산 나이(65세 경계) 대신 공식 라벨 age_band(D_SQ7, 70세 경계)로 세그먼트를 재정의해도 기존과
  87.8% 일치, Cramer's V 0.104→0.106(차이 +0.002)로 **세그먼트 정의 방식에 결론이 좌우되지 않음**을 확인.
- 기존엔 성별에만 적용했던 표준화잔차 사후검정을 세그먼트에도 추가해, 전체 효과크기(약함)만으로는
  안 보이던 뚜렷한 방향성을 확인: 청소년/청년가구→레포츠·모험형·캠핑·야영형 선호(청소년 레포츠
  잔차 +12.15), 중장년다인가구→등산·트레킹형 선호(+5.05), 고령가구(1인/다인 모두)→자연감상·산책형에만
  수렴(그 외 전부 유의하게 회피), 중장년자녀양육가구→치유·웰니스형 선호(+2.83, 유일 패턴).

**12-1. 운영 최적화 가정값 민감도 분석** (피드백: "운영 최적화는 현재 비용·인력 등을 가정한 개념검증
단계이므로, 주요 가정값의 설정 근거를 명확히 하고 가정값 변화에 따른 결과의 안정성을 검증할 필요가 있음")
- 예산/회당비용/회당인력/인력풀/회당전환인원을 ±30~50% 바꾼 8개 시나리오로 LP를 재풀이.
- **대구·경북·인천·경기는 8개 시나리오 전부에서 Top-5 우선지역으로 유지**(5번째 자리만 서울↔강원
  변동) — "어느 지역을 우선 배정할지"는 가정값 변화에 안정적. 다만 총 배정 규모(시급도가중 충족도)는
  제약이 타이트해지면 기준 대비 22~32% 감소. "예산 +30%"·"회당비용 -30%"가 기준과 완전히 동일한
  결과를 내는 이유(예산·인력 제약이 원래 77회에서 동시에 걸림)까지 명시적으로 설명.

### 출력 누락 11개 셀 복구 — 신규 (forest_final.ipynb)

사용자가 "skip_logic_validation.py가 노트북에 실제로 들어갔는지" 확인을 요청해서 검사하다가, 27번 셀
(스킵로직 검증)을 포함해 **print()/display()로 결과를 출력하는데도 캐시된 출력이 하나도 없는 코드셀이
11개**(3장/3-1/3-1-1/5장/8장/9-0-0/9-0-1/10-1 섹션에 흩어짐) 있는 것을 발견했습니다. 코드 로직 자체는
이미 여러 번 독립 재실행으로 정확성이 검증된 것들이었고(예: 스킵로직 20개 항목 위반 0건), 이번 세션
중 여러 restore/reinsert/splice 스크립트를 거치며 이 11개 셀만 출력이 누락된 것으로 보입니다(정확한
원인은 특정 못함).

`fill_missing_outputs.py`로 0~148번 셀을 실제로 순서대로 재실행(1286초)하면서 각 셀의 stdout/
display()/맨 끝 bare expression 결과를 IPython과 유사하게 캡처해, 위 11개 셀에만 실제 캡처된 출력을
채워 넣었습니다(다른 셀은 이미 정상이므로 전혀 건드리지 않음). 이후 전체 노트북 중복/오염/스키마
재스캔(161셀, 문제 없음)까지 완료 — 이제 `forest_final.ipynb`를 그냥 열기만 해도 모든 코드셀에
결과가 제대로 보입니다.

자세한 배경/경위는 Claude 메모리 `mdis-notebook-preprocessing-gaps.md`(라운드1~7) 참고.

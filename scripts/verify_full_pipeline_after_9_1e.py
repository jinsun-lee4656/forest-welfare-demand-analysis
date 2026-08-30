# -*- coding: utf-8 -*-
"""
9-1e 삽입 후 최종 검증: 0번 셀부터 9-1e(TabPFN-2.5, TABPFN_TOKEN 없이 skip 분기)까지 실제로
exec()해서 (1) 전처리(df 생성/위생점검/스킵로직)가 여전히 정상 통과하는지, (2) 9-1b Top-1/Top-2가
기존 저장값과 일치하는지, (3) 9-1d 과적합 검증이 캐시된 출력과 정확히 같은 수치를 재현하는지,
(4) 새로 넣은 9-1e 코드가 (TABPFN_TOKEN 없는 환경에서) 에러 없이 skip 분기로 깨끗하게 지나가는지
한 번에 확인한다. 이 뒤(9-4절 이상)의 무거운 셀은 목표 지점 밖이라 애초에 실행되지 않는다.
"""
import os, time, json
import matplotlib
matplotlib.use("Agg")

# TABPFN_TOKEN을 일부러 지워서, 이 검증이 "토큰 없는 일반 재현 환경"을 흉내내도록 한다
os.environ.pop("TABPFN_TOKEN", None)

t_start = time.time()
NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

idx_9_1e_code = next(i for i, c in enumerate(cells) if "TabPFN-2.5, 학습" in "".join(c["source"]))
print(f"목표 셀(9-1e 코드): {idx_9_1e_code}")

ns = {"display": lambda *a, **k: None}
for i, c in enumerate(cells):
    if i > idx_9_1e_code:
        break
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip() or src.lstrip().startswith(("!", "%")):
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print(f"\n0~{idx_9_1e_code}번 셀 재구성 완료 -- {time.time()-t_start:.0f}초 (에러 없이 전부 통과)")

# --- 핵심 수치 재검증 ---
checks = []
checks.append(("전처리 df 행 수", ns["df"].shape[0] == 11949))
checks.append(("가구 수", ns["df"]["hh_id"].nunique() == 5000))
checks.append(("occupation 결측 0건", ns["df"]["occupation"].isna().sum() == 0))
checks.append(("9-1b Top-1 Precision==92.2%", round(ns["prec1"] * 100, 1) == 92.2))
checks.append(("9-1b Top-1 Hit==92.2%", round(ns["hit1"] * 100, 1) == 92.2))
checks.append(("9-1b Top-2 Precision==77.3%", round(ns["prec2"] * 100, 1) == 77.3))
checks.append(("9-1d overfit_df 존재", "overfit_df" in ns))

print("\n=== 핵심 수치 재검증 ===")
all_ok = True
for name, ok in checks:
    print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    all_ok = all_ok and ok

if "overfit_df" in ns:
    od = ns["overfit_df"].round(4)
    print("\n9-1d overfit_df (재계산):")
    print(od)
    expected_train_auc = 0.9958
    expected_gap_auc = 0.1991
    ok_overfit = (abs(od.loc["평균AUC", "Train(학습셋)"] - expected_train_auc) < 1e-3 and
                  abs(od.loc["평균AUC", "격차(Train-Test)"] - expected_gap_auc) < 1e-3)
    print(f"  [{'OK' if ok_overfit else 'FAIL'}] 캐시된 노트북 출력(Train AUC 0.9958, 격차 +0.1991)과 일치")
    all_ok = all_ok and ok_overfit

print(f"\n총 소요시간: {time.time()-t_start:.0f}초")
print("\n" + ("=== 전체 검증 통과 ===" if all_ok else "=== 일부 검증 실패, 위 로그 확인 필요 ==="))

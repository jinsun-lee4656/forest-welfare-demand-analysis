# -*- coding: utf-8 -*-
"""
해커톤 본선 피드백(2번째 심사위원) 대응 2건을 프로토타이핑한다 — 실제 노트북 셀 0~152까지
exec()로 재구성해 진짜 수치를 확보한 뒤, 이 결과를 splice_feedback_additions.py가 노트북에
실제 셀로 반영한다.

1) "생애주기 세그먼트를 주요 타깃으로 활용하고 있으나 선호유형과의 연관성이 충분히 확인되지
   않은 만큼, 세그먼트별 차이를 추가 검증할 필요가 있음" (7장에 7-1절 추가)
   - 기존 lifecycle_segment(연산 age, 65세 경계)를 공식 라벨 age_band(D_SQ7, 자연스러운
     구간 경계인 70세 기준)로 재정의해도 Cramer's V가 비슷하게 유지되는지 확인.
   - 기존엔 성별만 했던 표준화잔차 사후검정을 세그먼트에도 적용해 "어느 세그먼트가 어떤
     선호유형에서 유의하게 많은/적은지" 구체적으로 확인.

2) "운영 최적화는 현재 비용·인력 등을 가정한 개념검증 단계이므로, 주요 가정값의 설정 근거를
   명확히 하고 가정값 변화에 따른 결과의 안정성을 검증할 필요가 있음" (12장에 12-1절 추가)
   - 예산/회당비용/회당인력/인력풀을 ±30~50% 바꿔가며 LP를 재풀이해, 총 배정 회차·시급도가중
     충족도·Top5 우선지역이 가정값 변화에도 크게 흔들리지 않는지 확인.
"""
import os, time, json
import matplotlib
matplotlib.use("Agg")
os.environ.pop("TABPFN_TOKEN", None)  # 이 검증엔 TabPFN 불필요, 토큰 있어도 일부러 안 씀(속도)

t_start = time.time()
NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

idx_12_1_target = next(i for i, c in enumerate(cells)
                        if 'savefig(FIGDIR / "16_optimization_allocation.png"' in "".join(c["source"]))
print(f"목표 셀(12장 LP 셀): {idx_12_1_target}")

ns = {"display": lambda *a, **k: None}
for i, c in enumerate(cells):
    if i > idx_12_1_target:
        break
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip() or src.lstrip().startswith(("!", "%")):
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print(f"0~{idx_12_1_target}번 셀 재구성 완료 -- {time.time()-t_start:.0f}초\n")

# ============================================================
# 1) 7-1. 세그먼트 재정의 + 사후검정
# ============================================================
print("=" * 60)
print("7-1. 생애주기 세그먼트 재정의 및 사후검정")
print("=" * 60)

pd = ns["pd"]; np = ns["np"]; df = ns["df"]
cramers_v = ns["cramers_v"]
standardized_residuals = ns["standardized_residuals"]
tests = ns["tests"]


def lifecycle_segment_v2(row):
    ab = row["age_band"]
    if pd.isna(ab):
        return np.nan
    if ab == "15-19세":
        return "청소년"
    if ab in ("20대", "30대"):
        return ("청년1인가구" if row["is_single_hh"]
                 else ("청년자녀양육가구" if row["has_child_under18"] else "청년다인가구"))
    if ab in ("40대", "50대", "60대"):
        return ("중장년1인가구" if row["is_single_hh"]
                 else ("중장년자녀양육가구" if row["has_child_under18"] else "중장년다인가구"))
    return "고령1인가구" if row["is_single_hh"] else "고령다인가구"  # 70세 이상


df["lifecycle_segment_v2"] = df.apply(lifecycle_segment_v2, axis=1)
pref_sub2 = df[df["pref_activity_broad"].notna()]

same_seg = (df["lifecycle_segment"] == df["lifecycle_segment_v2"]).mean() * 100
print(f"기존(연산 age 기준) vs 재정의(공식 age_band 기준) 세그먼트 일치율: {same_seg:.1f}%")
print("\n재정의 세그먼트 분포:")
print(df["lifecycle_segment_v2"].value_counts())

ct_v2 = pd.crosstab(pref_sub2["lifecycle_segment_v2"], pref_sub2["pref_activity_broad"])
chi2_v2, p_v2, dof_v2, v_v2 = cramers_v(ct_v2)
orig_v = tests["세그먼트 x 선호활동유형"][3]
print(f"\n[세그먼트(재정의) x 선호활동유형] chi2={chi2_v2:.1f}, dof={dof_v2}, p={p_v2:.2e}, Cramer's V={v_v2:.3f}")
print(f"(참고: 기존 정의 기준 Cramer's V={orig_v:.3f}, 차이={v_v2-orig_v:+.3f})")

resid_seg = standardized_residuals(ct_v2)
print("\n[세그먼트(재정의) x 선호활동유형] 표준화잔차 (|resid|>=1.96 이 유의):")
print(resid_seg.round(2))
sig_seg = resid_seg[resid_seg.abs() >= 1.96].stack()
print("\n유의한 조합:")
for (seg, act), v in sig_seg.items():
    direction = "많음" if v > 0 else "적음"
    print(f"  {seg} x {act}: {v:+.2f} (기대보다 {direction})")

result_7_1 = {
    "same_seg_pct": float(same_seg),
    "seg_v2_counts": df["lifecycle_segment_v2"].value_counts().to_dict(),
    "chi2_v2": float(chi2_v2), "p_v2": float(p_v2), "dof_v2": int(dof_v2), "cramers_v_v2": float(v_v2),
    "cramers_v_orig": float(orig_v),
    "resid_seg_round2": resid_seg.round(2).to_dict(),
    "sig_seg": [(list(k), float(v)) for k, v in sig_seg.items()],
}

# ============================================================
# 2) 12-1. 가정값 민감도 분석
# ============================================================
print("\n" + "=" * 60)
print("12-1. 운영 최적화 가정값 민감도 분석")
print("=" * 60)

pulp = ns["pulp"]
regions = ns["regions"]; ceiling = ns["ceiling"]; gap_pct = ns["gap_pct"]
MAX_RUNS_PER_REGION = ns["MAX_RUNS_PER_REGION"]
CAPACITY_PER_RUN = ns["CAPACITY_PER_RUN"]; COST_PER_RUN = ns["COST_PER_RUN"]
STAFF_PER_RUN = ns["STAFF_PER_RUN"]; BUDGET = ns["BUDGET"]; STAFF_POOL = ns["STAFF_POOL"]


def solve_allocation(cost_per_run, staff_per_run, budget, staff_pool, capacity_per_run):
    prob_s = pulp.LpProblem("sensitivity", pulp.LpMaximize)
    x_s = {r: pulp.LpVariable(f"runs_{r}_s", lowBound=0, upBound=MAX_RUNS_PER_REGION, cat="Integer") for r in regions}
    served_s = {r: pulp.LpVariable(f"served_{r}_s", lowBound=0) for r in regions}
    for r in regions:
        prob_s += served_s[r] <= ceiling[r]
        prob_s += served_s[r] <= capacity_per_run * x_s[r]
    prob_s += pulp.lpSum(cost_per_run * x_s[r] for r in regions) <= budget
    prob_s += pulp.lpSum(staff_per_run * x_s[r] for r in regions) <= staff_pool
    prob_s += pulp.lpSum(gap_pct[r] * served_s[r] for r in regions)
    prob_s.solve(pulp.PULP_CBC_CMD(msg=0))
    served_vals = pd.Series({r: served_s[r].value() for r in regions})
    runs_vals = pd.Series({r: x_s[r].value() for r in regions})
    weighted = (gap_pct * served_vals).sum()
    top5 = tuple(runs_vals[runs_vals > 0].sort_values(ascending=False).index[:5])
    return float(runs_vals.sum()), float(weighted), set(top5)


scenarios = {
    "기준(원래 가정)": dict(cost_per_run=COST_PER_RUN, staff_per_run=STAFF_PER_RUN, budget=BUDGET,
                        staff_pool=STAFF_POOL, capacity_per_run=CAPACITY_PER_RUN),
    "예산 -30%": dict(cost_per_run=COST_PER_RUN, staff_per_run=STAFF_PER_RUN, budget=BUDGET * 0.7,
                     staff_pool=STAFF_POOL, capacity_per_run=CAPACITY_PER_RUN),
    "예산 +30%": dict(cost_per_run=COST_PER_RUN, staff_per_run=STAFF_PER_RUN, budget=BUDGET * 1.3,
                     staff_pool=STAFF_POOL, capacity_per_run=CAPACITY_PER_RUN),
    "회당비용 +30%(효율저하)": dict(cost_per_run=COST_PER_RUN * 1.3, staff_per_run=STAFF_PER_RUN, budget=BUDGET,
                               staff_pool=STAFF_POOL, capacity_per_run=CAPACITY_PER_RUN),
    "회당비용 -30%(효율개선)": dict(cost_per_run=COST_PER_RUN * 0.7, staff_per_run=STAFF_PER_RUN, budget=BUDGET,
                               staff_pool=STAFF_POOL, capacity_per_run=CAPACITY_PER_RUN),
    "회당인력 +50%": dict(cost_per_run=COST_PER_RUN, staff_per_run=STAFF_PER_RUN * 1.5, budget=BUDGET,
                       staff_pool=STAFF_POOL, capacity_per_run=CAPACITY_PER_RUN),
    "인력풀 -30%": dict(cost_per_run=COST_PER_RUN, staff_per_run=STAFF_PER_RUN, budget=BUDGET,
                      staff_pool=STAFF_POOL * 0.7, capacity_per_run=CAPACITY_PER_RUN),
    "회당전환인원 -30%": dict(cost_per_run=COST_PER_RUN, staff_per_run=STAFF_PER_RUN, budget=BUDGET,
                         staff_pool=STAFF_POOL, capacity_per_run=CAPACITY_PER_RUN * 0.7),
}

sens_rows = []
base_top5 = None
base_weighted = None
for name, params in scenarios.items():
    total_runs_s, weighted_s, top5_s = solve_allocation(**params)
    if base_top5 is None:
        base_top5 = top5_s
        base_weighted = weighted_s
    overlap = len(top5_s & base_top5)
    sens_rows.append({"시나리오": name, "총 배정 회차": total_runs_s,
                       "시급도가중 충족도": weighted_s, "Top5지역 겹침(기준 대비)": f"{overlap}/5",
                       "Top5 우선지역": ", ".join(sorted(top5_s, key=lambda r: -gap_pct.get(r, 0)))})
sens_df = pd.DataFrame(sens_rows).set_index("시나리오")
sens_df["기준 대비 변화율(%)"] = ((sens_df["시급도가중 충족도"] / base_weighted - 1) * 100).round(1)
print(sens_df.round(1).to_string())

result_12_1 = {
    "base_top5": sorted(base_top5),
    "sens_df": sens_df.reset_index().to_dict("records"),
}

OUT_PATH = r"c:\Users\JS\Desktop\MDIS\feedback_additions_result.json"
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump({"result_7_1": result_7_1, "result_12_1": result_12_1}, f, ensure_ascii=False, indent=1, default=str)
print(f"\n결과 저장: {OUT_PATH}")
print(f"총 소요시간: {time.time()-t_start:.0f}초")

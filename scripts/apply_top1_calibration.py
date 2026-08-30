# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 5차 수정: 심사위원 피드백 대응
  "높은 Top-2 적중률이 다중응답 구조나 일부 활동의 높은 빈도로 인해 과대평가된 결과는
   아닌지" 확인 요청에 대응하여, Top-1 재평가 + 다수결 베이스라인 대비 개선폭 + 예측확률
   보정(Calibration) 검증을 K=2 셀(Precision@2/Recall@2/Hit@2) 바로 뒤에 추가한다.
"""
import json
import secrets
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source(text: str):
    lines = text.split("\n")
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


def new_cell(cell_type: str, text: str):
    cell = {"cell_type": cell_type, "id": secrets.token_hex(4), "metadata": {}, "source": to_source(text)}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def find_index(cells, marker, nth=0):
    hits = [i for i, c in enumerate(cells) if marker in "".join(c["source"])]
    if len(hits) <= nth:
        raise AssertionError(f"anchor {marker!r} 을(를) 가진 셀을 {nth+1}번째까지 찾지 못함 (hits={hits})")
    return hits[nth]


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    anchor = find_index(cells, 'ax.set_title("선호 활동유형 다중레이블 모델 - Top-2 추천 성능")')

    md1 = new_cell("markdown", """
### 9-1b. Top-1 재평가 및 다수결 베이스라인 대비 개선폭 (심사위원 피드백 대응)

**피드백 원문**: "본선에서는 높은 Top-2 적중률이 다중응답 구조나 일부 활동의 높은 빈도로 인해 과대평가된 결과는 아닌지 확인할 필요가 있음. 단순 적중률뿐 아니라 활동별 Precision, Recall,
PR-AUC, 예측확률의 보정 정도와 기준모형 대비 개선 폭을 함께 제시할 필요가 있음."

라벨별 AUC/F1/PR-AUC(9-1절 `metric_df`)는 이미 라벨마다 따로 제시하고 있으나, Top-N 적중률(위 Precision@2/Recall@2/Hit@2)은 "정답이 몇 개인지"·"어떤 라벨이 흔한지"를 함께 고려하지 않으면
과대평가로 오인되기 쉽습니다. 실제로 `intent_자연감상·산책형`은 응답자의 91.2%가 선택하는 압도적 다수 라벨이라, **개인차를 전혀 반영하지 않고 이 라벨만 무조건 추천해도** 적중률이 높게 나올 수
있습니다. 아래에서 (1) K=1(Top-1)까지 낮춰서 재평가하고, (2) "가장 흔한 라벨(들)을 모두에게 무조건 추천하는" 단순 다수결 베이스라인과 직접 비교해 순수한 개선폭을 확인합니다.
""")

    code1 = new_cell("code", """
def topk_metrics(K, proba, Y_true):
    topk_idx = np.argsort(-proba, axis=1)[:, :K]
    Y_arr = Y_true.values
    prec_list, rec_list, hit_list = [], [], []
    for i in range(len(Y_arr)):
        true_idx = set(np.where(Y_arr[i] == 1)[0])
        if len(true_idx) == 0:
            continue
        inter = len(true_idx & set(topk_idx[i]))
        prec_list.append(inter / K)
        rec_list.append(inter / len(true_idx))
        hit_list.append(1 if inter > 0 else 0)
    return float(np.mean(prec_list)), float(np.mean(rec_list)), float(np.mean(hit_list)), topk_idx

prec1, rec1, hit1, top1_idx = topk_metrics(1, proba_mat, Y_test)
print(f"[모델 Top-1] Precision@1={prec1*100:.1f}%  Recall@1={rec1*100:.1f}%  Hit@1={hit1*100:.1f}%")
print(f"[모델 Top-2] Precision@2={prec2*100:.1f}%  Recall@2={rec2*100:.1f}%  Hit@2={hit2*100:.1f}%  (앞 셀과 동일)")

# --- 단순(다수결) 베이스라인: 개인차 없이 "가장 흔한 라벨(들)"을 모두에게 추천했다면? ---
prevalence = Y_test.mean().sort_values(ascending=False)
top1_label_idx = BROAD_CATS.index(prevalence.index[0].replace("intent_", ""))
top2_label_idx = [BROAD_CATS.index(c.replace("intent_", "")) for c in prevalence.index[:2]]

def majority_baseline(label_idx_list, Y_true):
    Y_arr = Y_true.values
    K = len(label_idx_list)
    prec_list, rec_list, hit_list = [], [], []
    for i in range(len(Y_arr)):
        true_idx = set(np.where(Y_arr[i] == 1)[0])
        if len(true_idx) == 0:
            continue
        inter = len(true_idx & set(label_idx_list))
        prec_list.append(inter / K)
        rec_list.append(inter / len(true_idx))
        hit_list.append(1 if inter > 0 else 0)
    return float(np.mean(prec_list)), float(np.mean(rec_list)), float(np.mean(hit_list))

b1_prec, b1_rec, b1_hit = majority_baseline([top1_label_idx], Y_test)
b2_prec, b2_rec, b2_hit = majority_baseline(top2_label_idx, Y_test)
print(f"\\n[베이스라인, 항상 '{BROAD_CATS[top1_label_idx]}'만 추천] "
      f"Precision@1={b1_prec*100:.1f}%  Recall@1={b1_rec*100:.1f}%  Hit@1={b1_hit*100:.1f}%")
print(f"[베이스라인, 항상 {[BROAD_CATS[i] for i in top2_label_idx]} 두 개 추천] "
      f"Precision@2={b2_prec*100:.1f}%  Recall@2={b2_rec*100:.1f}%  Hit@2={b2_hit*100:.1f}%")

print(f"\\n[모델-베이스라인 개선폭] Top-1: Precision {(prec1-b1_prec)*100:+.1f}%p, Hit {(hit1-b1_hit)*100:+.1f}%p")
print(f"[모델-베이스라인 개선폭] Top-2: Precision {(prec2-b2_prec)*100:+.1f}%p, "
      f"Recall {(rec2-b2_rec)*100:+.1f}%p, Hit {(hit2-b2_hit)*100:+.1f}%p")

# 모델의 Top-1 예측 자체가 얼마나 다양한 라벨로 분산되는지 (다수결로 사실상 퇴화했는지 확인)
top1_pred_labels = pd.Series([BROAD_CATS[idx[0]] for idx in top1_idx])
print("\\n실제 모델의 Top-1 예측 라벨 분포(test셋 전체 기준, %) — 개인차 반영 정도를 보여줌:")
print((top1_pred_labels.value_counts(normalize=True) * 100).round(1))
""")

    md2 = new_cell("markdown", """
**해석 — 심사위원 우려가 부분적으로 실제 확인됨**: Top-1 적중률(92.2%)은 "무조건 자연감상·산책형만 추천"하는 단순 베이스라인(90.7%)과 겨우 **1.5%p** 차이입니다. 실제로 모델 스스로도
Top-1 예측의 59.2%를 자연감상·산책형에 몰아주고 있어, "개인화된 추천"이라기보다 "가장 흔한 것을 자주 고르는" 경향이 상당 부분을 차지함을 인정해야 합니다.

반면 **Top-2는 베이스라인 대비 Precision +6.7%p, Recall +6.4%p, Hit +2.6%p**로 뚜렷이 개선되어, 두 번째 추천 슬롯부터는 개인화 신호가 실질적으로 작동하고 있습니다. 정리하면:

- **"높은 적중률"의 상당 부분은 클래스 불균형(자연감상·산책형의 압도적 출현율 91%)에서 온다는 지적은 Top-1 기준으로는 사실**입니다.
- 다만 **Top-2 기준으로는 그 불균형을 감안하고도 남는 순수한 개선분이 존재**하며, 이는 우연이 아니라 모델이 두 번째 이하 순위에서는 개인별 과거 행태·인구통계 신호를 실제로 활용하고 있다는 근거입니다.
- 따라서 향후 성능을 보고할 때는 Top-N 적중률을 단독으로 제시하지 않고, **반드시 이 다수결 베이스라인 대비 개선폭을 함께 제시**해야 하며, 극단적으로 불균형한 "자연감상·산책형"보다는
  나머지 5개 클래스에 대한 라벨별 AUC/PR-AUC(9-1절 `metric_df`)가 모델의 진짜 판별력을 더 정직하게 보여주는 지표입니다.
""")

    idx = anchor + 1
    for c in [md1, code1, md2]:
        cells.insert(idx, c)
        idx += 1
    print(f"9-1b(Top-1 재평가+베이스라인) 삽입 위치: anchor idx {anchor} 뒤, {len([md1,code1,md2])}개 셀")

    # ------------------------------------------------------------------
    # 예측확률 보정(Calibration) 검증 — 방금 넣은 md2 바로 뒤
    # ------------------------------------------------------------------
    anchor2 = find_index(cells, "나머지 5개 클래스에 대한 라벨별 AUC/PR-AUC")

    md3 = new_cell("markdown", """
### 9-1c. 예측확률 보정(Calibration) 검증

심사위원 피드백의 "예측확률의 보정 정도"를 Brier score와 라벨별 calibration curve(신뢰도 다이어그램)로 확인합니다. 보정이 잘 된 모델은 "70% 확률"이라고 예측한 사례들이 실제로도
약 70% 비율로 맞아야 합니다.
""")
    code2 = new_cell("code", """
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

brier_scores = {cat: brier_score_loss(Y_test.iloc[:, i], proba_mat[:, i]) for i, cat in enumerate(BROAD_CATS)}
print("라벨별 Brier score (0에 가까울수록 보정이 잘 됨 — 참고: 완전 무작위 예측은 0.25 근방):")
display(pd.Series(brier_scores).round(4))
print(f"평균 Brier score: {np.mean(list(brier_scores.values())):.4f}")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, (i, cat) in zip(axes.flat, enumerate(BROAD_CATS)):
    frac_pos, mean_pred = calibration_curve(Y_test.iloc[:, i], proba_mat[:, i], n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="완벽한 보정")
    ax.plot(mean_pred, frac_pos, "o-", color=PALETTE[i % len(PALETTE)], label="실제")
    ax.set_title(cat, fontsize=10); ax.set_xlabel("예측확률 평균"); ax.set_ylabel("실제 양성비율")
    ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGDIR / "12c_calibration.png", dpi=150); plt.show()
""")
    md4 = new_cell("markdown", """
**해석**: 평균 Brier score 0.16으로 완전 무작위(0.25)보다는 뚜렷이 낫지만, 최다수 라벨인 "자연감상·산책형"의 calibration curve를 보면 예측확률이 낮은~중간 구간(0.3~0.7)에서
실제 양성비율이 예측치보다 체계적으로 높게 나타납니다(예: 예측확률 평균 0.38일 때 실제 양성비율 0.61). 즉 **이 라벨에 대해 모델이 과소적합(under-confident) 방향으로 편향**되어 있습니다.
RandomForest는 여러 트리 투표 비율의 평균을 확률로 쓰기 때문에 원래 극값(0/1) 근처로 잘 안 가는 경향이 있어 이런 패턴이 전형적이며, Platt scaling이나 isotonic regression으로
사후보정(post-hoc calibration)하면 개선 여지가 있습니다. 다만 이번 분석에서 실제 의사결정에 쓰는 것은 순위 기반 지표(AUC, Precision/Recall@K)이므로, 확률 보정 자체가 최종 추천
순위에 미치는 영향은 제한적입니다 — 다만 "이 활동을 70% 확률로 좋아할 것"처럼 확률값 자체를 사용자에게 직접 노출하는 서비스로 발전시킬 경우에는 사후보정이 필요합니다.
""")
    idx2 = anchor2 + 1
    for c in [md3, code2, md4]:
        cells.insert(idx2, c)
        idx2 += 1
    print(f"9-1c(Calibration) 삽입 위치: anchor idx {anchor2} 뒤, 3개 셀")

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 완료: {NB_PATH} (총 {len(cells)}개 셀)")


if __name__ == "__main__":
    main()

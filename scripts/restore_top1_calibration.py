# -*- coding: utf-8 -*-
"""
9-1b(Top-1 재평가+다수결 베이스라인)/9-1c(Calibration) 복원.

이 6개 셀(md1,code1,md2,md3,code2,md4)은 apply_top1_calibration.py로 오늘 오전
(09:26) 실제로 노트북에 삽입되고 splice_partial_outputs.py(09:30)로 실제 실행값까지
반영된 뒤 fix_top1_numbers.py(09:29 -- 로그상 반영은 그 이후)로 서술 수치까지 정정됐던
것을 splice_log.txt로 확인했다. 그런데 이후(Round 5~7의 반복적인 노트북 전체 read-modify
-write 스크립트들 사이 어느 시점) 노트북에서 이 6개 셀 자체가 통째로 사라졌다 --
현재 노트북에서 "9-1b"/"9-1c"/"Brier" 문자열이 전혀 검색되지 않는 것으로 확인.

원본 계산은 전부 고정 random_state를 쓰는 결정론적 파이프라인이므로(9-1절 RandomForest),
9-4~9-9절(부스팅/TabPFN 비교, 이번 목적에 전혀 필요 없음)만 건너뛰고 0~92번 셀을 실제로
재실행해 진짜 값을 다시 얻은 뒤, apply_top1_calibration.py의 원본 셀 6개를 그 실제 출력과
함께 정확한 위치(9-1 섹션의 Precision@2/Recall@2/Hit@2 셀 바로 뒤)에 복원한다.
"""
import base64
import io
import json
import os
import secrets
import time
import contextlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t_start = time.time()
NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

# ------------------------------------------------------------------
# 1) 0~92번 셀(9-1절까지) 재구성, 9-4/9-8/9-9(부스팅+TabPFN, 불필요) 건너뜀
# ------------------------------------------------------------------
idx_target = next(i for i, c in enumerate(cells)
                   if 'ax.set_title("선호 활동유형 다중레이블 모델 - Top-2 추천 성능")' in "".join(c["source"]))
idx_skip_boost = next(i for i, c in enumerate(cells) if "boost_models = {" in "".join(c["source"]))
idx_skip_bestmodel = next(i for i, c in enumerate(cells) if "best_model_name = comparison_df" in "".join(c["source"]))
idx_skip_val = next(i for i, c in enumerate(cells) if "gss_val = GroupShuffleSplit" in "".join(c["source"]))
idx_skip_lgbmval = next(i for i, c in enumerate(cells) if "lgbm_val = Pipeline" in "".join(c["source"]))
SKIP = {idx_skip_boost, idx_skip_bestmodel, idx_skip_val, idx_skip_lgbmval}
print(f"목표 셀(Top-2 히트율 플롯): {idx_target}, 건너뛸 셀: {sorted(SKIP)}")

captured = []  # display()로 넘어온 객체들의 repr을 순서대로 담음


def _display(*args, **kwargs):
    for a in args:
        captured.append(repr(a))


ns = {"display": _display}

for i, c in enumerate(cells):
    if i > idx_target:
        break
    if i in SKIP:
        print(f"[건너뜀] cell {i}")
        continue
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip():
        continue
    if src.lstrip().startswith(("!", "%")):
        print(f"[건너뜀-매직/쉘] cell {i}: {src.strip()[:60]!r}")
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ns)

print(f"0~{idx_target}번 셀 재구성 완료 -- {time.time()-t_start:.0f}초")

# ------------------------------------------------------------------
# 2) apply_top1_calibration.py의 code1(9-1b) 실행, 실제 stdout 캡처
# ------------------------------------------------------------------
CODE1 = r'''
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
print(f"\n[베이스라인, 항상 '{BROAD_CATS[top1_label_idx]}'만 추천] "
      f"Precision@1={b1_prec*100:.1f}%  Recall@1={b1_rec*100:.1f}%  Hit@1={b1_hit*100:.1f}%")
print(f"[베이스라인, 항상 {[BROAD_CATS[i] for i in top2_label_idx]} 두 개 추천] "
      f"Precision@2={b2_prec*100:.1f}%  Recall@2={b2_rec*100:.1f}%  Hit@2={b2_hit*100:.1f}%")

print(f"\n[모델-베이스라인 개선폭] Top-1: Precision {(prec1-b1_prec)*100:+.1f}%p, Hit {(hit1-b1_hit)*100:+.1f}%p")
print(f"[모델-베이스라인 개선폭] Top-2: Precision {(prec2-b2_prec)*100:+.1f}%p, "
      f"Recall {(rec2-b2_rec)*100:+.1f}%p, Hit {(hit2-b2_hit)*100:+.1f}%p")

top1_pred_labels = pd.Series([BROAD_CATS[idx[0]] for idx in top1_idx])
print("\n실제 모델의 Top-1 예측 라벨 분포(test셋 전체 기준, %) — 개인차 반영 정도를 보여줌:")
print((top1_pred_labels.value_counts(normalize=True) * 100).round(1))
'''

buf1 = io.StringIO()
with contextlib.redirect_stdout(buf1):
    exec(compile(CODE1, "<code1>", "exec"), ns)
out1_text = buf1.getvalue()
print("=== code1(9-1b) 실행 완료 ===")
print(out1_text)

# ------------------------------------------------------------------
# 3) apply_top1_calibration.py의 code2(9-1c calibration) 실행
# ------------------------------------------------------------------
CODE2 = r'''
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
'''

captured.clear()
buf2 = io.StringIO()
with contextlib.redirect_stdout(buf2):
    exec(compile(CODE2, "<code2>", "exec"), ns)
out2_stream_parts = buf2.getvalue().split("평균 Brier score:")
assert len(out2_stream_parts) == 2, "Brier 출력 파싱 실패"
out2_stream1 = out2_stream_parts[0]
out2_stream2 = "평균 Brier score:" + out2_stream_parts[1]
print("=== code2(9-1c) 실행 완료 ===")
print(out2_stream1)
print(captured[0][:200], "...")
print(out2_stream2)

png_path = Path(WORKDIR) / "figures" / "12c_calibration.png"
assert png_path.exists(), f"그림 파일이 저장되지 않음: {png_path}"
png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
print(f"PNG 크기: {len(png_b64)} base64 chars, 파일: {png_path}")

# ------------------------------------------------------------------
# 4) 결과를 JSON으로 저장 (다음 스크립트에서 노트북에 삽입할 때 사용)
# ------------------------------------------------------------------
result = {
    "out1_text": out1_text,
    "brier_series_repr": captured[0],
    "out2_stream1": out2_stream1,
    "out2_stream2": out2_stream2,
    "png_b64": png_b64,
}
RESULT_PATH = Path(r"c:\Users\JS\Desktop\MDIS\top1_calibration_result.json")
RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
print(f"\n결과 저장: {RESULT_PATH}")
print(f"전체 소요시간: {time.time()-t_start:.0f}초")

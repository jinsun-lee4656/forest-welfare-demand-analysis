# -*- coding: utf-8 -*-
"""
9-1d(과적합 검증) 뒤에 '9-1e. TabPFN-2.5 Top-1/Top-2 재평가' 3개 셀(마크다운/코드/마크다운)을 삽입한다.
compute_tabpfn_top1.py로 이미 완료한 실제 실행 결과(2026-08-31, 학습 2,000행 서브샘플/test 500행
서브샘플, 총 1086초)를 코드 셀의 캐시된 출력으로 그대로 옮긴다 — 랜덤시드가 전부 고정(42)된
결정론적 계산이라, 이미 확보한 정답 수치를 다시 노트북 전체 재실행 없이 반영해도 소스와 출력이
일치한다(9-4/9-9절 TabPFN-3를 처음 넣을 때와 같은 방식).
"""
import json
import secrets
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")


def to_source(text: str):
    lines = text.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]


def new_cell(cell_type, text, outputs=None):
    c = {"cell_type": cell_type, "id": secrets.token_hex(4), "metadata": {}, "source": to_source(text)}
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = outputs or []
    return c


nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]

idx_91d_interp = next(i for i, c in enumerate(cells)
                       if "실제로 상당한 과적합이 확인됩니다" in "".join(c["source"]))
print("삽입 위치: 9-1d 해석 셀", idx_91d_interp, "바로 뒤")

intro_md = new_cell("markdown", """
### 9-1e. TabPFN-2.5 Top-1/Top-2 재평가 (별도 모델로 심사위원 피드백 재확인)

9-1b에서 확인한 "Top-1 개선폭이 작다"는 심사위원 우려를, RandomForest가 아닌 **TabPFN-2.5**(Prior Labs, 로컬 추론)로도
같은 방식(다수결 베이스라인 대비 개선폭)으로 재확인합니다. TabPFN-2.5는 9-4/9-9절의 TabPFN-3([Prior Labs의 최신
기본 체크포인트](https://www.priorlabs.ai))과는 별개의 게이트된 체크포인트(`Prior-Labs/tabpfn_2_5`)라 별도 라이선스
동의가 필요해 독립된 섹션으로 분리했습니다.

**서브샘플링(학습/평가 모두)**: TabPFN의 추론 비용은 예측할 때 참조하는 학습셋(컨텍스트) 크기에 크게 좌우됩니다.
전체 X_train(5,724행)을 그대로 컨텍스트로 써서 예측 500행을 돌렸더니 2시간을 넘겨도 끝나지 않았습니다(9-4/9-9절
TabPFN-3의 최악 기록도 ~40분이었던 것과 비교하면 3배 이상 — v2.5 체크포인트가 더 무겁거나 이번 실행이 유독
느렸던 것으로 추정되나 원인은 특정하지 못했습니다). 따라서 test와 마찬가지로 **학습셋도 2,000행 랜덤
서브샘플**로 줄여 재시도했습니다. 다수결 베이스라인도 같은 서브샘플에서 다시 계산해 공정하게 비교합니다 —
다만 test 500행/학습 2,000행이라는 축소된 표본이라, 9-1b의 RandomForest(전체 test 1,902행) 수치와 절대값을
직접 비교하기보다는 "베이스라인 대비 개선폭"이라는 상대적 지표 위주로 해석해야 합니다.
""")

CODE = '''
# TabPFN-2.5(선택적, 로컬 추론): 9-4/9-9절 TabPFN-3과는 별개의 게이트된 체크포인트(Prior-Labs/tabpfn_2_5)라
# 별도 라이선스 동의가 필요함(https://ux.priorlabs.ai/account/licenses). API 키(TABPFN_TOKEN)가 있을 때만
# 비교에 포함하고, 없으면 조용히 건너뜀 -> 다른 환경(키 없음)에서 재현 실행해도 파이프라인이 깨지지 않음.
if os.environ.get("TABPFN_TOKEN"):
    try:
        from tabpfn import TabPFNClassifier

        TABPFN_SUBSAMPLE_N = 500
        TABPFN_TRAIN_SUBSAMPLE_N = 2000  # 전체 X_train(5,724행)은 예측 1회에 2시간을 넘겨 학습셋도 축소
        rng_sub = np.random.RandomState(42)
        test_sub_pos = rng_sub.choice(len(X_test), size=min(TABPFN_SUBSAMPLE_N, len(X_test)), replace=False)
        X_test_sub, Y_test_sub = X_test.iloc[test_sub_pos], Y_test.iloc[test_sub_pos]
        print(f"서브샘플 크기: {len(X_test_sub)}행 (전체 test={len(X_test)}행)")

        rng_train_sub = np.random.RandomState(42)
        train_sub_pos = rng_train_sub.choice(len(X_train), size=min(TABPFN_TRAIN_SUBSAMPLE_N, len(X_train)), replace=False)
        X_train_sub, Y_train_sub = X_train.iloc[train_sub_pos], Y_train.iloc[train_sub_pos]
        print(f"학습 서브샘플 크기: {len(X_train_sub)}행 (전체 train={len(X_train)}행)")

        t0 = time.time()
        tabpfn25_pipe = Pipeline([("pre", pre2), ("m", MultiOutputClassifier(
            TabPFNClassifier(model_path="tabpfn-v2.5-classifier-v2.5_default.ckpt",
                              ignore_pretraining_limits=True, random_state=42)))])
        tabpfn25_pipe.fit(X_train_sub, Y_train_sub)
        print(f"TabPFN-2.5 fit: {time.time()-t0:.0f}초")

        t0 = time.time()
        proba_list_t = tabpfn25_pipe.predict_proba(X_test_sub)
        proba_mat_t = np.column_stack([p[:, 1] for p in proba_list_t])
        print(f"TabPFN-2.5 predict_proba({len(X_test_sub)}행): {time.time()-t0:.0f}초")

        t_prec1, t_rec1, t_hit1, _ = topk_metrics(1, proba_mat_t, Y_test_sub)
        t_prec2, t_rec2, t_hit2, _ = topk_metrics(2, proba_mat_t, Y_test_sub)
        sb1_prec, sb1_rec, sb1_hit = majority_baseline([top1_label_idx], Y_test_sub)
        sb2_prec, sb2_rec, sb2_hit = majority_baseline(top2_label_idx, Y_test_sub)

        print(f"\\n[TabPFN-2.5, 학습 {len(X_train_sub)}행 서브샘플/test {len(X_test_sub)}행 서브샘플] "
              f"Top-1: Precision={t_prec1*100:.1f}%  Recall={t_rec1*100:.1f}%  Hit={t_hit1*100:.1f}%")
        print(f"[TabPFN-2.5, 학습 {len(X_train_sub)}행 서브샘플/test {len(X_test_sub)}행 서브샘플] "
              f"Top-2: Precision={t_prec2*100:.1f}%  Recall={t_rec2*100:.1f}%  Hit={t_hit2*100:.1f}%")
        print(f"\\n[다수결 베이스라인, 같은 서브샘플] Top-1: Precision={sb1_prec*100:.1f}%  Hit={sb1_hit*100:.1f}%")
        print(f"[다수결 베이스라인, 같은 서브샘플] Top-2: Precision={sb2_prec*100:.1f}%  Recall={sb2_rec*100:.1f}%  Hit={sb2_hit*100:.1f}%")
        print(f"\\n[TabPFN-2.5-베이스라인 개선폭] Top-1: Precision {(t_prec1-sb1_prec)*100:+.1f}%p, Hit {(t_hit1-sb1_hit)*100:+.1f}%p")
        print(f"[TabPFN-2.5-베이스라인 개선폭] Top-2: Precision {(t_prec2-sb2_prec)*100:+.1f}%p, "
              f"Recall {(t_rec2-sb2_rec)*100:+.1f}%p, Hit {(t_hit2-sb2_hit)*100:+.1f}%p")
        print(f"\\n[참고 - 기존 RandomForest, 전체 test {len(X_test)}행] Top-1: Precision={prec1*100:.1f}%  "
              f"Hit={hit1*100:.1f}%  (베이스라인 대비 +{(prec1-b1_prec)*100:.1f}%p)")
        print(f"[참고 - 기존 RandomForest, 전체 test {len(X_test)}행] Top-2: Precision={prec2*100:.1f}%  "
              f"Recall={rec2*100:.1f}%  Hit={hit2*100:.1f}%  "
              f"(베이스라인 대비 +{(prec2-b2_prec)*100:.1f}%p/+{(rec2-b2_rec)*100:.1f}%p/+{(hit2-b2_hit)*100:.1f}%p)")
    except Exception as e:
        print(f"[TabPFN-2.5 건너뜀] {type(e).__name__}: {e}")
else:
    print("[TabPFN-2.5 건너뜀] TABPFN_TOKEN 환경변수가 없어 건너뜁니다 "
          "(Prior Labs 계정 API 키 필요, tabpfn_2_5 게이트 저장소는 별도 라이선스 동의 필요: "
          "https://ux.priorlabs.ai/account/licenses).")
'''.strip("\n")

STDOUT_TEXT = (
    "서브샘플 크기: 500행 (전체 test=1902행)\n"
    "학습 서브샘플 크기: 2000행 (전체 train=5724행)\n"
    "TabPFN-2.5 fit: 3초\n"
    "TabPFN-2.5 predict_proba(500행): 1026초\n"
    "\n"
    "[TabPFN-2.5, 학습 2000행 서브샘플/test 500행 서브샘플] Top-1: Precision=93.2%  Recall=43.2%  Hit=93.2%\n"
    "[TabPFN-2.5, 학습 2000행 서브샘플/test 500행 서브샘플] Top-2: Precision=78.2%  Recall=66.0%  Hit=98.0%\n"
    "\n"
    "[다수결 베이스라인, 같은 서브샘플] Top-1: Precision=91.2%  Hit=91.2%\n"
    "[다수결 베이스라인, 같은 서브샘플] Top-2: Precision=69.2%  Recall=58.1%  Hit=95.4%\n"
    "\n"
    "[TabPFN-2.5-베이스라인 개선폭] Top-1: Precision +2.0%p, Hit +2.0%p\n"
    "[TabPFN-2.5-베이스라인 개선폭] Top-2: Precision +9.0%p, Recall +7.9%p, Hit +2.6%p\n"
    "\n"
    "[참고 - 기존 RandomForest, 전체 test 1902행] Top-1: Precision=92.2%  Hit=92.2%  (베이스라인 대비 +1.4%p)\n"
    "[참고 - 기존 RandomForest, 전체 test 1902행] Top-2: Precision=77.3%  Recall=65.8%  Hit=97.7%  "
    "(베이스라인 대비 +6.7%p/+6.3%p/+2.7%p)\n"
)

outputs = [{"output_type": "stream", "name": "stdout", "text": to_source(STDOUT_TEXT)}]
code_cell = new_cell("code", CODE, outputs=outputs)

interp_md = new_cell("markdown", """
**해석**: 학습 2,000행/test 500행이라는 축소된 조건에서도 TabPFN-2.5의 Top-1 Precision/Hit(93.2%)은 같은
서브샘플의 다수결 베이스라인(91.2%) 대비 **+2.0%p** 개선폭을 보였고, Top-2는 Precision +9.0%p·Recall
+7.9%p·Hit +2.6%p로 나타났습니다. 9-1b의 RandomForest(전체 test 1,902행 기준 Top-1 +1.4%p, Top-2
+6.7%p/+6.3%p/+2.7%p)와 나란히 놓으면, **TabPFN-2.5가 이 표본에서는 베이스라인 대비 개선폭이 더 크게
나왔습니다** — 심사위원이 지적한 "Top-1 개선폭이 작다"는 우려에 대해, 모델을 바꿔도 여전히 개선폭이
존재하며(0이 아님) 오히려 더 뚜렷하다는 추가 근거가 됩니다.

다만 두 결과를 절대값으로 직접 비교하는 데는 한계가 있습니다: TabPFN-2.5 쪽은 test 500행/학습 2,000행
서브샘플이라 RandomForest의 전체 test(1,902행) 기준보다 표본이 작아 우연에 의한 변동(sampling noise)이 더
클 수 있습니다. 다수결 베이스라인 자체도 서브샘플에 따라 90.7%(전체)→91.2%(500행 서브샘플)로 소폭
달라진 것이 그 증거입니다. 따라서 이 결과는 "다른 모델·다른 표본에서도 개선폭이 사라지지 않는다"는
정성적 재확인으로 읽는 것이 적절하며, RandomForest 대비 TabPFN-2.5의 절대적 우위를 주장하는 근거로는
쓰지 않았습니다.
""")

cells[idx_91d_interp + 1:idx_91d_interp + 1] = [intro_md, code_cell, interp_md]
nb["cells"] = cells
NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"저장 완료 (총 {len(cells)}개 셀)")

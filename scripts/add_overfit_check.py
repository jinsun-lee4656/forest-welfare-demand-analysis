# -*- coding: utf-8 -*-
"""
9-1c(calibration + 5-fold CV) 뒤에 "9-1d. 과적합(Overfitting) 검증" 섹션을 추가한다.
clf_ml(9-1절 메인 RandomForest 파이프라인)로 X_train 자체를 예측해 Train 성능과 Test 성능(X_test) 격차를 직접 비교.
필요한 상태(clf_ml, X_train/Y_train/X_test/Y_test, auc_per_label, macro_f1_recheck, micro_f1, hamming, BROAD_CATS)는
cell 0~91까지만 실제로 실행하면 얻어지므로(9-4절 TabPFN 등 무거운 셀은 이 뒤에 있어 건드릴 필요 없음) 가볍게 재구성한다.
"""
import os, sys, json, time, secrets, io, contextlib
import matplotlib
matplotlib.use("Agg")

t_start = time.time()
NB_PATH = r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb"
WORKDIR = r"c:\Users\JS\Desktop\MDIS\분석산출물"
os.makedirs(WORKDIR, exist_ok=True)
os.chdir(WORKDIR)

nb = json.load(open(NB_PATH, encoding="utf-8"))
cells = nb["cells"]

# --- 1) 필요한 상태만 가볍게 재구성 (cell 0~91) ---
captured_reprs = []


def display(*args, **kwargs):
    for x in args:
        captured_reprs.append(repr(x))


ns = {"display": display}
for i, c in enumerate(cells):
    if i > 91:
        break
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if not src.strip():
        continue
    exec(compile(src, f"<cell {i}>", "exec"), ns)
print(f"0~91번 셀(9-1절까지) 재구성 완료 -- {time.time()-t_start:.0f}초")

# --- 2) 새 셀 코드를 실제로 실행해서 진짜 출력 확보 ---
NEW_CODE = '''
proba_list_train = clf_ml.predict_proba(X_train)
proba_mat_train = np.column_stack([p[:, 1] for p in proba_list_train])
pred_mat_train = (proba_mat_train >= 0.5).astype(int)

auc_train = {}
for i, cat in enumerate(BROAD_CATS):
    if Y_train.iloc[:, i].nunique() > 1:
        auc_train[cat] = roc_auc_score(Y_train.iloc[:, i], proba_mat_train[:, i])

overfit_df = pd.DataFrame({
    "Train(학습셋)": [
        float(np.mean(list(auc_train.values()))),
        f1_score(Y_train, pred_mat_train, average="macro"),
        f1_score(Y_train, pred_mat_train, average="micro"),
        hamming_loss(Y_train, pred_mat_train),
    ],
    "Test(평가셋)": [
        float(np.mean(list(auc_per_label.values()))),
        macro_f1_recheck,
        micro_f1,
        hamming,
    ],
}, index=["평균AUC", "Macro_F1", "Micro_F1", "Hamming"])
overfit_df["격차(Train-Test)"] = overfit_df["Train(학습셋)"] - overfit_df["Test(평가셋)"]
display(overfit_df.round(4))

gap_auc = overfit_df.loc["평균AUC", "격차(Train-Test)"]
gap_macro = overfit_df.loc["Macro_F1", "격차(Train-Test)"]
gap_hamming = overfit_df.loc["Hamming", "격차(Train-Test)"]
verdict = ("과적합 우려가 낮은 수준입니다" if (gap_auc < 0.05 and gap_macro < 0.05)
           else "Train 성능이 Test보다 뚜렷이 높아 과적합 가능성을 점검할 필요가 있습니다")
print(f"\\n[과적합 판정] 평균AUC 격차 {gap_auc:+.4f}, Macro-F1 격차 {gap_macro:+.4f}, Hamming 격차 {gap_hamming:+.4f}")
print(f"-> {verdict} (경험적 기준: 격차 0.05 미만이면 낮음)")
'''.strip("\n")

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(NEW_CODE, "<new_cell>", "exec"), ns)
stdout_text = buf.getvalue()
print("=== 새 셀 stdout ===")
print(stdout_text)
print("=== 새 셀 display() 캡처 ===")
for r in captured_reprs:
    print(r)

# 결과를 스크래치패드에도 저장해 다음 단계(splice)에서 정확히 재사용
out_path = r"C:\Users\JS\AppData\Local\Temp\claude\c--Users-JS-Desktop-MDIS\8a32d8b2-fcd5-4d3c-ab54-6781f3832676\scratchpad\overfit_check_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({"stdout": stdout_text, "df_repr": captured_reprs[-1] if captured_reprs else None,
               "df_round4": ns["overfit_df"].round(4).to_dict()}, f, ensure_ascii=False, indent=1)
print("\n결과 저장:", out_path)
print(f"총 소요시간: {time.time()-t_start:.0f}초")

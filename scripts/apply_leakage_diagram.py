# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 6차 수정: 심사위원 피드백 대응
  "학습변수와 예측대상 간 정보 중복 또는 데이터 누수가 발생하지 않았는지도 분석 흐름도와
   변수 정의표를 통해 명확히 설명이 필요함"에 대응하여, 9-0-0절 검증 셀 바로 뒤에
   Feature/Target 분리를 보여주는 시각적 데이터 흐름도(matplotlib)를 추가한다.
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


DIAGRAM_CODE = '''
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def _leak_box(ax, cx, y, w, h, text, color, fontsize=9.5, fontweight="normal", lw=1.6, alpha=0.20):
    x = cx - w / 2
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=lw, edgecolor=color, facecolor=color, alpha=alpha)
    ax.add_patch(b)
    ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color="black")
    return dict(cx=cx, y=y, w=w, h=h, top=y + h, bottom=y, left=x, right=x + w)

def _leak_vline(ax, b1, b2, color, style="-", lw=2.0, rad=0.0):
    p1 = (b1["cx"], b1["bottom"]); p2 = (b2["cx"], b2["top"])
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                         color=color, linestyle=style, connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)

_COL_RAW, _COL_KEEP, _COL_DROP = "#8C8C8C", "#4C72B0", "#C44E52"
_COL_FEAT, _COL_TARGET, _COL_GATE = "#55A868", "#937860", "#DD8452"

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 15.2); ax.set_ylim(0, 10); ax.axis("off")

XA, XB, XC, XD, XE = 1.6, 4.6, 7.6, 10.6, 13.6

q1 = _leak_box(ax, 8.6, 8.5, 3.0, 0.9, "문1 (활동형태)\\n일상형 / 당일형 / 숙박형", _COL_GATE, fontweight="bold")
ax.text(8.6, 8.15, "(②당일형 또는 ③숙박형을 선택한 사람만 아래 문항에 응답)",
        ha="center", fontsize=8.5, color=_COL_GATE, style="italic")

q17 = _leak_box(ax, XA, 6.6, 2.6, 1.0, "Q17\\n(향후 의향 체크리스트)", _COL_RAW)
q10 = _leak_box(ax, XB, 6.6, 2.6, 1.0, "Q10_1/Q10_2\\n(과거 활동 체크리스트)", _COL_RAW)
q11 = _leak_box(ax, XC, 6.6, 2.6, 1.0, "Q11/Q12\\n(방문기록 15슬롯)", _COL_RAW)
q192 = _leak_box(ax, XD, 6.6, 2.6, 1.0, "Q19_2/Q20_2\\n(프로그램·시설 이용경험)", _COL_RAW)
q1934 = _leak_box(ax, XE, 6.6, 2.6, 1.0, "Q19_3/4, Q20_3/4\\n(사전예약·바우처)", _COL_RAW)
for b in [q17, q10, q11, q192, q1934]:
    _leak_vline(ax, q1, b, _COL_GATE, style="--", lw=1.3)

skip = FancyArrowPatch((q17["cx"], 6.35), (q10["cx"], 6.35), arrowstyle="<|-|>", mutation_scale=13,
                        linewidth=1.6, color=_COL_GATE, linestyle=(0, (4, 3)), zorder=1)
ax.add_patch(skip)
ax.text((q17["cx"] + q10["cx"]) / 2, 6.48, "Q10 무경험 = Q17 무응답 (일치율 100%, 9-0-1절 assert)",
        ha="center", fontsize=7.8, color=_COL_GATE)

intent = _leak_box(ax, XA, 4.7, 2.6, 0.9, "intent_*\\n(광역카테고리별 향후의향)", _COL_KEEP)
past = _leak_box(ax, XB, 4.7, 2.6, 0.9, "past_*, pastcnt_*\\n(광역카테고리별 과거경험)", _COL_KEEP)
env = _leak_box(ax, XC, 4.7, 2.6, 0.9, "season/companion/\\npurpose/spend (환경변수)", _COL_KEEP)
aware = _leak_box(ax, XD, 4.7, 2.6, 0.9, "aware_n / used_n\\n(인지도·이용경험 카운트)", _COL_KEEP)
drop_res = _leak_box(ax, XE, 4.7, 2.6, 1.3, "[미사용]\\nFEATURES 미포함\\n(이용경험의 부분집합\\n-> 동어반복 위험)", _COL_DROP, fontsize=8.7)
_leak_vline(ax, q17, intent, _COL_KEEP, lw=2.2)
_leak_vline(ax, q10, past, _COL_KEEP, lw=2.2)
_leak_vline(ax, q11, env, _COL_KEEP, lw=2.2)
_leak_vline(ax, q192, aware, _COL_KEEP, lw=2.2)
_leak_vline(ax, q1934, drop_res, _COL_DROP, style=":", lw=2.2)

drop_act = _leak_box(ax, XC, 2.7, 3.0, 0.9, "[미사용] activity 필드 제외\\n(Q10과 동일정보 중복)", _COL_DROP, fontsize=8.7)
_leak_vline(ax, env, drop_act, _COL_DROP, style=":", lw=1.8)
ax.annotate("", xy=(XC - 0.15, drop_act["top"]), xytext=(q11["cx"] + 0.9, q11["bottom"]),
            arrowprops=dict(arrowstyle="-|>", color=_COL_DROP, linestyle=":", lw=1.8,
                             connectionstyle="arc3,rad=0.35"))

y_cols = _leak_box(ax, XA, 0.5, 2.6, 1.2, "Y_COLS\\n(intent_* 6개,\\n다중레이블 타깃)", _COL_TARGET,
                    fontsize=10.5, fontweight="bold", alpha=0.30)
feat_cx = (XB + XD) / 2 + 0.15
feat = _leak_box(ax, feat_cx, 0.5, (XD - XB) + 2.7, 1.2, "FEATURES_NUM + FEATURES_CAT\\n(41개: 인구통계 + 환경변수 + 과거행태)",
                  _COL_FEAT, fontsize=11, fontweight="bold", alpha=0.30)
_leak_vline(ax, intent, y_cols, _COL_TARGET, lw=2.6)
_leak_vline(ax, past, feat, _COL_KEEP, lw=2.2)
_leak_vline(ax, env, feat, _COL_KEEP, lw=2.2)
_leak_vline(ax, aware, feat, _COL_KEEP, lw=2.2)

ax.text(7.6, 9.55, "9-0-0. 데이터 누수 방지 흐름도 - 어떤 문항이 피처(FEATURES)로, 어떤 문항이 타깃(Y_COLS)으로 가는가",
        ha="center", fontsize=13.5, fontweight="bold")
legend_elems = [
    mpatches.Patch(color=_COL_KEEP, alpha=0.3, label="피처(FEATURES)로 사용"),
    mpatches.Patch(color=_COL_TARGET, alpha=0.3, label="타깃(Y_COLS)"),
    mpatches.Patch(color=_COL_DROP, alpha=0.3, label="누수 위험 -> 의도적으로 미사용"),
    mpatches.Patch(color=_COL_GATE, alpha=0.3, label="설문 스킵로직(응답조건)"),
]
ax.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=9, framealpha=0.95)
plt.tight_layout(); plt.savefig(FIGDIR / "12d_leakage_flow.png", dpi=150); plt.show()
'''.strip("\n")


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    anchor = find_index(cells, "### 9-0-0. 데이터 누수(Data Leakage) 방지 검증")
    # anchor 바로 다음 셀(assert 코드) 뒤에 삽입
    insert_at = anchor + 2

    md = new_cell("markdown", """
### 9-0-0-1. 데이터 흐름도 — Feature/Target 분리 시각화 (심사위원 피드백 대응)

**피드백 원문**: "학습변수와 예측대상 간 정보 중복 또는 데이터 누수가 발생하지 않았는지도 분석 흐름도와 변수 정의표를 통해 명확히 설명이 필요함."

위 어설션들이 통과한다는 것을 텍스트로만 확인하는 대신, 어떤 원본 문항이 피처(FEATURES)로 가고 어떤 문항이 타깃(Y_COLS)으로 가는지, 그리고 어떤 문항을 왜 의도적으로 제외했는지를
한 장의 흐름도로 요약합니다. 별도의 변수정의표 문서(`변수정의표_데이터누수검증.md`)와 함께 참고하면 됩니다.
""")
    code = new_cell("code", DIAGRAM_CODE)

    cells.insert(insert_at, code)
    cells.insert(insert_at, md)
    print(f"데이터 흐름도 삽입 위치: anchor idx {anchor} 뒤 (insert_at={insert_at})")

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 완료: {NB_PATH} (총 {len(cells)}개 셀)")


if __name__ == "__main__":
    main()

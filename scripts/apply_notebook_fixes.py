# -*- coding: utf-8 -*-
"""
forest_welfare_analysis (2).ipynb 전처리 정확성 개선 일괄 적용 스크립트.
(Read 도구가 이 노트북을 열기엔 너무 커서 NotebookEdit 대신 JSON을 직접 조작한다.)

적용 항목:
  1) education/occupation/income_ord를 3장(전처리)으로 이전 + 정의 오류 수정
     - education: DQ2 4~7 코드가 "재학/졸업" 상태가 아니라 순수 학교급임을 반영
     - occupation: DQ1==2(무직) 분기의 DQ1_2 세부코드(학생/주부/취업준비중/무직/기타)를 복원, 결측 0건화
     - city_size: "군지역" -> SAV 공식 라벨 "읍면지역"
  2) 8장 K-means CLUSTER_CAT에 education/occupation/marital 추가 + 클러스터별 학력/직업/혼인 비중 출력 추가
  3) 9장 education/occupation 중복 정의 제거(3장으로 이전됐으므로)
  4) 데이터 누수 방지 가드 추가: (a) 5장에 방문지역코드(sido_code)-거주지역코드(CO11) 코드체계 일치 assert,
     (b) 9장 FEATURES에 Q19-3/19-4(예약·바우처) 파생변수 혼입 여부 assert, (c) 9-0-1절 스킵로직 실증에 assert 추가
  5) "군지역" 잔여 표기 전부 "읍면지역"으로 정정 (75번, 122번 셀)
"""
import json
import secrets
from pathlib import Path

NB_PATH = Path(r"c:\Users\JS\Desktop\forest_welfare_analysis (2).ipynb")
BACKUP_PATH = NB_PATH.with_name(NB_PATH.stem + ".backup_before_fixes.ipynb")


def to_source(text: str):
    """파이썬 문자열 -> nbformat source 리스트(각 줄 끝에 \n, 마지막 줄은 없음)"""
    lines = text.split("\n")
    # 맨 앞/뒤 빈 줄 제거(트리플쿼트 문자열 관례상 생기는 것들)
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


def new_cell(cell_type: str, text: str):
    cell = {
        "cell_type": cell_type,
        "id": secrets.token_hex(4),
        "metadata": {},
        "source": to_source(text),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def replace_cell(cells, idx, text):
    c = cells[idx]
    c["source"] = to_source(text)
    if c["cell_type"] == "code":
        c["execution_count"] = None
        c["outputs"] = []


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    if not BACKUP_PATH.exists():
        BACKUP_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("백업 저장:", BACKUP_PATH)
    else:
        print("백업 이미 존재(재실행이므로 새로 만들지 않음):", BACKUP_PATH)

    cells = nb["cells"]

    # ------------------------------------------------------------------
    # 1) 셀 24 (markdown) — 섹션 소개 갱신
    # ------------------------------------------------------------------
    replace_cell(cells, 24, """
## 3. 전처리 (1) — 가구구조 복원 및 인구통계 라벨링

`NO0==1`(가구주)을 기준으로 가구ID를 만들고, 가구주에게만 응답된 소득(DQ5)을 같은 가구원 전체에 전파합니다.
이어서 연령, 성별, 가구유형, 지역, **학력·직업·소득순위(income_ord)** 등 주요 인구통계 변수에 라벨을 매핑합니다.

**이번 버전에서 정정한 항목**(원자료 SAV 내장 메타데이터로 직접 재검증):
1. **학력(education)**: DQ2의 4~7번 코드는 "재학/졸업" 여부가 아니라 순수 학교급(4=전문대(4년제 미만), 5=대학교(4년제 이상), 6=대학원 석사과정, 7=대학원 박사과정) 구분입니다. 재학/중퇴/휴학/수료/졸업 여부는 별도 문항(DQ2_1)의 몫인데, 이전 버전은 이를 혼동해 4~7을 "대학재학/대졸/대학원재학/대학원졸"로 잘못 매핑했습니다.
2. **직업(occupation)**: DQ1_1(직업분류)은 DQ1==1(직업 있음)인 사람만 응답해, DQ1==2(없음)인 3,922명(32.8%)은 구조적으로 결측입니다. 이전 버전은 이 구조적 결측을 나중에 "미상"으로 뭉뚱그렸는데, 실제로는 DQ1==2 분기에서도 세부상태(DQ1_2: 학생/주부/취업준비중/무직/기타)를 온전히 수집하고 있어(두 분기 결측이 정확히 상호배타적 100% 커버 — `check_occupation_detail.py`로 검증) 이를 복원하면 occupation 결측을 0건으로 없앨 수 있습니다.
3. **지역규모(city_size)**: 3번 코드의 공식 라벨은 "읍면지역"입니다(`SAV_라벨_확인.py`로 확인). 이전 버전의 "군지역"은 의미는 같지만 공식 용어가 아니므로 정정합니다.
4. **정의 시점**: education/occupation/income_ord를 이전 버전처럼 8~9장에서야 처음 만들지 않고 이 시점(3장)에 미리 만들어, 이후 6장(EDA)·7장(가설검정)·8장(K-means 페르소나)에서도 바로 활용할 수 있게 했습니다(이전에는 이 세 변수가 존재하지 않아 세 장 모두 학력·직업·소득을 전혀 반영하지 못했습니다).
""")

    # ------------------------------------------------------------------
    # 2) 셀 25 (code) — 인구통계 라벨링 본체
    # ------------------------------------------------------------------
    replace_cell(cells, 25, """
df = raw.copy()

# --- 가구ID 복원: NO0==1(가구주) 등장할 때마다 새 가구 시작 ---
df["hh_id"] = (df["NO0"] == 1).cumsum()
print("복원된 가구 수:", df["hh_id"].nunique(), " / 개인 응답자 수:", len(df))

# --- 가구단위 변수 전파 (소득) ---
for v in ["DQ5", "D_DQ5"]:
    df[v] = df.groupby("hh_id")[v].transform(lambda s: s.ffill().bfill())
print("가구단위 보정 후 DQ5 결측 수:", df["DQ5"].isna().sum())

# --- 인구통계 라벨링 ---
SURVEY_YEAR = 2025  # 조사표 파일명 기준 '2024년 조사(2025년 실시)'
# 주의: SQ7_1은 출생'연도'만 있고 월/일 정보가 없어(조사지침서 확인), 그해 생일이 지났는지에 따라
# 실제 만 나이와 최대 ±1세 오차가 날 수 있는 근사치입니다. age_band는 이 오차와 무관하게 조사기관이
# 원자료(D_SQ7)에서 직접 확정한 공식 라벨을 그대로 쓰므로, 연속형이 필요 없다면 age_band가 더 신뢰할 수 있습니다.
df["age"] = SURVEY_YEAR - df["SQ7_1"]
df["age_band"] = df["D_SQ7"].map({1:"15-19세",2:"20대",3:"30대",4:"40대",5:"50대",6:"60대",7:"70세 이상"})
df["gender"] = df["SQ6"].map({1:"남",2:"여"})
df["hh_type"] = df["SQ3"].map({1:"1인가구",2:"2인가구",3:"3인가구",4:"4인이상가구"})
df["marital"] = df["DQ3"].map({1:"미혼",2:"배우자 있음",3:"사별",4:"이혼",5:"기타"})
df["income_band"] = df["DQ5"].map({1:"100만원미만",2:"100-200만원",3:"200-300만원",4:"300-400만원",
    5:"400-500만원",6:"500-600만원",7:"600-700만원",8:"700-800만원",9:"800만원이상"})
income_ord_map = {"100만원미만":1,"100-200만원":2,"200-300만원":3,"300-400만원":4,"400-500만원":5,
                   "500-600만원":6,"600-700만원":7,"700-800만원":8,"800만원이상":9}
df["income_ord"] = df["income_band"].map(income_ord_map)
df["sido"] = df["CO11"].map({11:"서울",21:"부산",22:"대구",23:"인천",24:"광주",25:"대전",26:"울산",29:"세종",
    31:"경기",32:"강원",33:"충북",34:"충남",35:"전북",36:"전남",37:"경북",38:"경남",39:"제주"})
# D_CO1112 공식 라벨은 "대도시/중소도시/읍면지역"입니다(SAV 내장 메타데이터, SAV_라벨_확인.py로 확인).
df["city_size"] = df["D_CO1112"].map({1:"대도시(특광역시)",2:"중소도시",3:"읍면지역"})

# 최종학력(DQ2): 4~7은 재학/졸업이 아니라 순수 학교급 구분(재학/중퇴/졸업 등은 별도 문항 DQ2_1의 몫).
df["education"] = df["DQ2"].map({1:"초졸이하",2:"중졸",3:"고졸",4:"전문대(4년제미만)",
    5:"대학교(4년제이상)",6:"대학원(석사과정)",7:"대학원(박사과정)"})

# 직업(occupation): DQ1==1(있음) 분기는 DQ1_1, DQ1==2(없음) 분기는 DQ1_2로 세부상태를 복원.
# (check_occupation_detail.py로 SAV 원본 검증: 두 분기 결측이 정확히 상호배타적 & 합쳐서 전체 커버 → 결측 0건 보장)
OCC_MAP_EMPLOYED = {1:"관리자",2:"전문가",3:"사무",4:"서비스",5:"판매",6:"농림어업",
                     7:"기능원",8:"장치조작",9:"단순노무",10:"군인"}
OCC_MAP_NOT_EMPLOYED = {11:"학생",12:"주부",13:"취업준비중",14:"무직",15:"기타(비경제활동)"}
df["occupation"] = df["DQ1_1"].map(OCC_MAP_EMPLOYED)
_no_job_mask = df["DQ1"] == 2
df.loc[_no_job_mask, "occupation"] = df.loc[_no_job_mask, "DQ1_2"].map(OCC_MAP_NOT_EMPLOYED)
assert df["occupation"].isna().sum() == 0, "occupation에 결측 발생 — DQ1/DQ1_1/DQ1_2 매핑 재확인 필요"
print("직업(occupation) 분포(무직 세분화 완료, 결측 0건):")
print(df["occupation"].value_counts())

df["is_single_hh"] = (df["SQ3"] == 1)
df["is_elderly"] = (df["age"] >= 65)
df["has_child_under18"] = ((df["DQ4_2"].fillna(0) + df["DQ4_3"].fillna(0)) > 0)

df[["age","age_band","gender","hh_type","income_band","income_ord","education","occupation","sido","city_size"]].head(8)
""")

    # ------------------------------------------------------------------
    # 3) 셀 33 (code) — 방문지역 코드(sido_code)와 거주지역코드(CO11) 체계 일치 검증 삽입
    # ------------------------------------------------------------------
    cell33_text = """
df = df.reset_index(drop=True)
df["resp_id"] = df.index
MONTH_TO_SEASON = {12:"겨울",1:"겨울",2:"겨울",3:"봄",4:"봄",5:"봄",6:"여름",7:"여름",8:"여름",9:"가을",10:"가을",11:"가을"}

def melt_visit_block(df, prefix, n_slots, field_map):
    frames = []
    for i in range(1, n_slots + 1):
        cols = {out: tmpl.format(prefix=prefix, i=i) for out, tmpl in field_map.items()}
        exist = {out: c for out, c in cols.items() if c in df.columns}
        sub = df[["resp_id"] + list(exist.values())].copy()
        sub = sub.rename(columns={v: k for k, v in exist.items()})
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    return long.dropna(subset=["activity"])

daytrip_field_map = {"activity":"{prefix}A{i}","month":"{prefix}_2_1A{i}","weekday_type":"{prefix}_2_2A{i}",
    "region_code":"{prefix}_3A{i}","companion":"{prefix}_6A{i}","purpose":"{prefix}_7A{i}","spend":"{prefix}_8A{i}"}
long_day = melt_visit_block(df, "Q11", 15, daytrip_field_map)
long_day["season"] = long_day["month"].map(MONTH_TO_SEASON)
long_day["sido_code"] = (long_day["region_code"] // 1000).astype("Int64")

overnight_field_map = {"activity":"{prefix}A{i}","month":"{prefix}_2_1A{i}","weekday_type":"{prefix}_2_2A{i}",
    "region_code":"{prefix}_3A{i}","nights":"{prefix}_5_11A{i}","companion":"{prefix}_6A{i}",
    "purpose":"{prefix}_7A{i}","spend":"{prefix}_8A{i}"}
long_night = melt_visit_block(df, "Q12", 15, overnight_field_map)
long_night["season"] = long_night["month"].map(MONTH_TO_SEASON)
long_night["sido_code"] = (long_night["region_code"] // 1000).astype("Int64")

# --- 데이터 누수/조인키 검증: 방문지역 코드(sido_code)가 거주지역 코드(CO11)와 같은 코드체계인지 확인 ---
# (요구사항: "실제 시설위치 데이터와 결합하려면 지역 join key를 표준화해야 한다"는 지적에 대한 근거 —
#  방문지는 거주지와 다를 수 있으므로 값이 "같아야" 하는 게 아니라, 코드값 "체계"가 같은지가 핵심)
VALID_SIDO_CODES = {11,21,22,23,24,25,26,29,31,32,33,34,35,36,37,38,39}
observed_sido = (set(long_day["sido_code"].dropna().unique().tolist())
                  | set(long_night["sido_code"].dropna().unique().tolist()))
observed_sido = {int(c) for c in observed_sido}
assert observed_sido <= VALID_SIDO_CODES, f"CO11 코드체계 밖의 방문지역 코드 발견: {observed_sido - VALID_SIDO_CODES}"
print(f"[OK] Q11/Q12 방문지역코드(sido_code)가 거주지 시도코드(CO11)와 동일한 체계임을 확인 "
      f"(관측된 코드 {len(observed_sido)}개, 전체 17개 시도 코드의 부분집합)")

print("당일형 방문기록(long) 행수:", len(long_day), " / 숙박형:", len(long_night))

def mode_or_nan(s):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else np.nan

agg_day = long_day.groupby("resp_id").agg(daytrip_n_records=("activity","count"),
    daytrip_dominant_season=("season",mode_or_nan), daytrip_dominant_weekday=("weekday_type",mode_or_nan),
    daytrip_dominant_companion=("companion",mode_or_nan), daytrip_dominant_purpose=("purpose",mode_or_nan),
    daytrip_avg_spend=("spend","mean"), daytrip_dominant_sido=("sido_code",mode_or_nan))

agg_night = long_night.groupby("resp_id").agg(overnight_n_records=("activity","count"),
    overnight_dominant_season=("season",mode_or_nan), overnight_dominant_weekday=("weekday_type",mode_or_nan),
    overnight_dominant_companion=("companion",mode_or_nan), overnight_dominant_purpose=("purpose",mode_or_nan),
    overnight_avg_spend=("spend","mean"), overnight_avg_nights=("nights","mean"), overnight_dominant_sido=("sido_code",mode_or_nan))

df = df.merge(agg_day, on="resp_id", how="left").merge(agg_night, on="resp_id", how="left")

companion_map = {1:"혼자",2:"가족",3:"친척",4:"친구/연인",5:"직장동료",6:"친목단체/동호회",7:"학교단체",8:"기타"}
purpose_map = {1:"휴양/휴식",2:"건강증진",3:"질병치유",4:"교육/학습",5:"취미/레포츠",6:"친목활동",7:"기타"}
weekday_map = {1:"주중",2:"주말",3:"공휴일"}
for c in ["daytrip_dominant_companion","overnight_dominant_companion"]:
    df[c+"_label"] = df[c].map(companion_map)
for c in ["daytrip_dominant_purpose","overnight_dominant_purpose"]:
    df[c+"_label"] = df[c].map(purpose_map)
for c in ["daytrip_dominant_weekday","overnight_dominant_weekday"]:
    df[c+"_label"] = df[c].map(weekday_map)

df[["daytrip_n_records","daytrip_dominant_season","daytrip_avg_spend","overnight_n_records","overnight_avg_nights"]].describe()
"""
    replace_cell(cells, 33, cell33_text)

    # ------------------------------------------------------------------
    # 4) 셀 71 (code) — K-means: income_ord 중복정의 제거 + CLUSTER_CAT 확장
    # ------------------------------------------------------------------
    replace_cell(cells, 71, """
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# income_ord/education/occupation/marital은 3장(전처리)에서 이미 정의됩니다.
# (이전 버전은 income_ord를 이 셀에서 처음 만들고 education/occupation은 9장에서야 만들어,
#  6~7장 EDA·가설검정은 물론 여기(8장 K-means)에서도 학력·직업·소득을 전혀 쓸 수 없었습니다 — 순서 수정)

CLUSTER_NUM = ["age","income_ord","n_activity_types_experienced","n_daytrip_activity_types",
               "n_overnight_activity_types","facility_gap","program_gap","Q18","Q22",
               "daytrip_avg_spend","overnight_avg_spend"]
# 이전 버전은 gender/hh_type/city_size 3개뿐이었으나(정의 순서 문제로 학력·직업이 존재하지 않았음),
# 이제 사용 가능해진 학력·직업·혼인상태를 추가해 페르소나 해석에 실질적으로 도움이 되게 확장합니다.
CLUSTER_CAT = ["gender","hh_type","city_size","education","occupation","marital"]

work = df[CLUSTER_NUM + CLUSTER_CAT + ["resp_id","WT"]].copy()
for c in ["daytrip_avg_spend","overnight_avg_spend"]:
    work[c] = work[c].fillna(0)
for c in ["Q18","Q22","income_ord"]:
    work[c] = work[c].fillna(work[c].median())
work = work.dropna(subset=CLUSTER_CAT)
print("군집분석 대상 표본 수(범주형 결측 제외 후):", len(work), "/", len(df))

pre = ColumnTransformer([("num", StandardScaler(), CLUSTER_NUM), ("cat", OneHotEncoder(drop="first"), CLUSTER_CAT)])
X = pre.fit_transform(work[CLUSTER_NUM + CLUSTER_CAT])

rng = np.random.RandomState(42)
sample_idx = rng.choice(X.shape[0], size=3000, replace=False)
sil_scores = {}
for k in range(3, 9):
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    sil_scores[k] = silhouette_score(X[sample_idx], km_k.labels_[sample_idx])
    print(f"k={k}: silhouette={sil_scores[k]:.4f}")

best_k = max(sil_scores, key=sil_scores.get)
print("선택된 k:", best_k)
""")

    # ------------------------------------------------------------------
    # 5) 셀 73 (code) — 클러스터 프로파일에 학력/직업/혼인 비중 추가
    # ------------------------------------------------------------------
    replace_cell(cells, 73, """
def wavg2(s, w):
    m = s.notna()
    return np.average(s[m], weights=w[m]) if m.sum() else np.nan

profile_rows = []
for c in sorted(work["cluster"].unique()):
    g = work[work["cluster"]==c]
    row = {"cluster": c, "n": len(g), "인구비중(%)": g["WT"].sum()/work["WT"].sum()*100}
    for col in CLUSTER_NUM:
        row[col] = wavg2(g[col], g["WT"])
    profile_rows.append(row)
profile = pd.DataFrame(profile_rows).set_index("cluster")
display(profile.round(2))

df = df.merge(work[["resp_id","cluster"]], on="resp_id", how="left")
print("성별 비중(%):"); display((pd.crosstab(df["cluster"], df["gender"], normalize="index")*100).round(1))
print("생애주기 비중(%):"); display((pd.crosstab(df["cluster"], df["lifecycle_segment"], normalize="index")*100).round(1))
print("혼인상태 비중(%):"); display((pd.crosstab(df["cluster"], df["marital"], normalize="index")*100).round(1))
print("학력 비중(%):"); display((pd.crosstab(df["cluster"], df["education"], normalize="index")*100).round(1))
print("직업 비중(%, 군집별 상위 5개):")
for c in sorted(df["cluster"].dropna().unique()):
    top5 = df.loc[df["cluster"]==c, "occupation"].value_counts(normalize=True).head(5) * 100
    print(f"  군집{int(c)}:", {k: round(v, 1) for k, v in top5.items()})
""")

    # ------------------------------------------------------------------
    # 6) 셀 75 (markdown) — "군지역" -> "읍면지역"
    # ------------------------------------------------------------------
    replace_cell(cells, 75, """
## 9. 선호 활동유형 다중레이블 예측모델

**목표(요약서 원문)**: "개인 특성(연령, 가구유형 등)과 환경요인(계절, 접근성)을 입력했을 때, 가장 선호할 것으로 예측되는 산림휴양활동유형을 판별. 수요 예측 정확도 85% 이상 달성"

**설계 변경 근거**: 이전 버전에서는 Q17(향후 참여의향, 원래 다중응답 문항)을 "가장 많이 선택한 광역카테고리 1개"로 단순화한 단일분류로 접근했습니다.
이 방식은 (1) 광역카테고리별 세부활동 개수가 2~9개로 불균등해 idxmax()가 항목이 많은 카테고리로 구조적으로 쏠릴 수 있고, (2) "가장 많이 고른 것"이 실제로 "가장 선호"를 의미하지 않는다는
두 가지 근본적 한계가 있었습니다. 이번 버전에서는 **10장의 시설모델과 동일하게, 6개 광역카테고리 각각에 대해 독립적인 이진분류기를 학습하는 다중레이블(Binary Relevance) 방식**으로 재설계했습니다.

**입력 피처**: 나이, 소득, 성별, 가구유형, 지역규모, 혼인상태, 학력, 직업, 거주 시도, 당일·숙박형 활동의 주 계절/동반유형/목적(환경요인), 과거 활동 경험(광역카테고리별 인코딩) + 참여강도·만족도 신호

**지역규모(city_size)에 대한 주의**: "대도시/중소도시/읍면지역" 구분을 산림 "접근성"의 대리변수로 사용하고 있으나, 엄밀히는 도시 규모와 산림 접근성은 다른 개념입니다
(읍면지역은 인구는 적어도 산림에 물리적으로 더 가까울 수 있습니다). 실제 접근성(이동시간·거리) 데이터가 없어 차선으로 사용하는 것이며, 향후 실제 접근성 데이터가 확보되면 교체가 필요합니다.

**타깃**: 6개 광역카테고리(등산·트레킹형/자연감상·산책형/캠핑·야영형/체험·학습형/치유·웰니스형/레포츠·모험형) 각각에 대한 "향후 의향 있음/없음" 이진 플래그 (표본이 극소수인 문화향유형·기타는 체험·학습형에 통합)
""")

    # ------------------------------------------------------------------
    # 7) 셀 76 (code) — education/occupation 중복정의 제거 (3장으로 이전됨)
    # ------------------------------------------------------------------
    replace_cell(cells, 76, """
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score,
                              label_ranking_average_precision_score, average_precision_score)

BROAD_CATS = ["등산·트레킹형","자연감상·산책형","캠핑·야영형","체험·학습형","치유·웰니스형","레포츠·모험형"]
CAT_MERGE = {"문화향유형": "체험·학습형", "기타": "체험·학습형"}  # 표본 1~2명뿐인 희소 카테고리 통합

def broad_merged(code_):
    b = ACTIVITY_TO_BROAD.get(code_, "기타")
    return CAT_MERGE.get(b, b)

for cat in BROAD_CATS:
    codes_in_cat = [c for c in ACTIVITY_TO_BROAD if broad_merged(c) == cat]
    # 과거 행태 피처 (Q10, 타깃과 별개 문항)
    df[f"past_{cat}"] = df.apply(lambda r, cc=codes_in_cat: int(
        any(c in (r["daytrip_activity_codes"] or []) for c in cc) or
        any(c in (r["overnight_activity_codes"] or []) for c in cc)), axis=1)
    df[f"pastcnt_{cat}"] = df.apply(lambda r, cc=codes_in_cat: (
        sum(1 for c in (r["daytrip_activity_codes"] or []) if c in cc) +
        sum(1 for c in (r["overnight_activity_codes"] or []) if c in cc)), axis=1)
    # 다중레이블 타깃 (Q17, 향후의향)
    df[f"intent_{cat}"] = df["intent_activity_codes"].apply(lambda cs, cc=codes_in_cat: int(any(c in cs for c in cc)))

# education/occupation은 3장(전처리)에서 이미 정의됨(중복 정의 제거).

LIKERT_COLS = ["Q18", "Q22", "Q8"]  # 0~10점 척도 — 0("부정적/불만족")도 유효 응답이라 fillna(0)으로 채우면 "무응답"과 "실제 최저점"이 섞임

FEATURES_NUM = (["age","income_ord","n_activity_types_experienced","n_daytrip_activity_types",
                  "n_overnight_activity_types","daytrip_avg_spend","overnight_avg_spend",
                  "daytrip_n_records","overnight_n_records",
                  "facility_aware_n","facility_used_n","program_aware_n","program_used_n"]
                 + LIKERT_COLS + [f"{c}_missing" for c in LIKERT_COLS]
                 + [f"past_{c}" for c in BROAD_CATS] + [f"pastcnt_{c}" for c in BROAD_CATS])
FEATURES_CAT = ["gender","hh_type","city_size","marital","education","occupation","sido",
                "daytrip_dominant_season","daytrip_dominant_companion_label","daytrip_dominant_purpose_label",
                "overnight_dominant_season","overnight_dominant_companion_label","overnight_dominant_purpose_label"]
Y_COLS = [f"intent_{c}" for c in BROAD_CATS]

# 구조적 결측(무응답) 처리: 0~10점 척도는 0도 유효 응답이므로, 척도 밖 값(-1)+별도 결측 플래그로 "무응답"을 명시적으로 구분
# df 자체에 적용해 10장(시설모델)·9-5절(이용목적 모델)에서도 동일하게 재사용
for c in LIKERT_COLS:
    df[f"{c}_missing"] = df[c].isna().astype(int)
    df[c] = df[c].fillna(-1)

model_df = df[df["pref_activity_broad"].notna()].copy()
for c in FEATURES_NUM:
    if c not in LIKERT_COLS and not c.endswith("_missing"):
        model_df[c] = model_df[c].fillna(0)
for c in FEATURES_CAT:
    model_df[c] = model_df[c].fillna("미상")

print("라벨(광역카테고리)별 향후의향 비율(%):")
print((model_df[Y_COLS].mean()*100).round(1))
print("\\n1인당 평균 선택 카테고리 수:", model_df[Y_COLS].sum(axis=1).mean().round(2), "(참고: 1개보다 크다는 것 자체가 단일분류가 부적합했다는 근거)")

pre2 = ColumnTransformer([("num","passthrough",FEATURES_NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT)])
""")

    # ------------------------------------------------------------------
    # 8) 셀 76 뒤에 데이터 누수 방지 검증(마크다운+코드) 2개 셀 삽입
    # ------------------------------------------------------------------
    md_leak = new_cell("markdown", """
### 9-0-0. 데이터 누수(Data Leakage) 방지 검증 — 피처/타깃 분리 확인

리뷰에서 지적된 위험 지점 중, "계층적 응답조건이 있는 문항을 서로의 피처로 잘못 사용"하는 유형의 누수를 코드로 직접 확인합니다.
- **Q19-3(사전예약제 이용경험)/Q19-4(바우처 사용경험)**는 Q19-2(이용경험 있음)가 "예"인 사람에게만 실질적으로 의미가 있는 계층적 응답입니다. 이 파생변수를 다른 문항의 예측 피처로 쓰면
  "이용경험이 있어야만 답할 수 있는 정보"로 "이용경험 여부"를 맞히는 순환논리가 될 수 있어, 애초에 FEATURES에 포함하지 않았습니다 — 아래에서 실제로 그런지 확인합니다.
- **Q11/Q12(방문기록) 롱포맷**에서 활동코드(activity) 자체는 Q10과 사실상 동일 정보이므로 피처로 재사용하지 않고, 계절·동반유형·목적·지출액 등 부가정보만 사용했는지 확인합니다.
""")
    code_leak = new_cell("code", """
# --- 데이터 누수(feature/target 중복·순환 인코딩) 방지 검증 ---
leak_terms = ["reserved", "voucher"]
leaked_features = [f for f in FEATURES_NUM + FEATURES_CAT if any(t in f for t in leak_terms)]
assert not leaked_features, f"Q19-3/19-4(예약·바우처) 파생변수가 피처에 포함되어 있습니다: {leaked_features}"
print("[OK] program/facility_reserved·voucher(Q19-3/19-4, Q20-3/20-4) 파생변수는 모델 피처에 포함되지 않았습니다.")

activity_like = [f for f in FEATURES_NUM + FEATURES_CAT
                 if "activity" in f.lower() and not f.startswith(("n_", "past", "pastcnt"))]
assert not activity_like, f"Q11/Q12의 원본 활동코드가 그대로 피처에 포함되어 있습니다: {activity_like}"
print("[OK] Q11/Q12 방문기록의 활동코드 원본은 피처로 재사용되지 않았습니다(계절·동반·목적·지출만 사용).")

print("\\n데이터 누수 방지 검증 통과 — FEATURES_NUM/FEATURES_CAT 목록에 계층응답·중복인코딩 변수 없음.")
""")
    cells.insert(77, code_leak)
    cells.insert(77, md_leak)
    # (77 위치에 md, 그 다음 78 위치에 code가 오도록 순서대로 두 번 insert)

    # ------------------------------------------------------------------
    # 9) 원래의 9-0-1절 코드 셀(Q17 스킵로직 검증)에 assert 보강
    #    -- 삽입으로 인덱스가 2칸 밀렸으므로 원래 79 -> 81
    # ------------------------------------------------------------------
    idx_901 = 79 + 2
    assert "explicit_none_count" in "".join(cells[idx_901]["source"]), \
        f"인덱스 밀림 계산 오류: cells[{idx_901}]가 9-0-1절 코드셀이 아닙니다"
    replace_cell(cells, idx_901, """
explicit_none_count = (raw[[f"Q17A{i}" for i in range(1,30) if f"Q17A{i}" in raw.columns]] == 999999999.0).any(axis=1).sum()
print(f"Q17에서 명시적 '없음(999999999)' 코드를 실제로 사용한 응답자 수: {explicit_none_count}명")

never_detailed_activity = (df["n_daytrip_activity_types"] == 0) & (df["n_overnight_activity_types"] == 0)
q17_no_answer = df["pref_activity_broad"].isna()
overlap_table = pd.crosstab(never_detailed_activity, q17_no_answer)
print("\\n[Q10 상세활동 무경험 여부] x [Q17 무응답 여부] 교차표:")
print(overlap_table)
match_rate = (never_detailed_activity == q17_no_answer).mean()
print(f"\\n두 조건이 정확히 일치하는 비율: {match_rate*100:.1f}%")

# --- 데이터 누수 방지 검증: 이 스킵 로직을 그대로 타깃으로 쓰는 Stage1/2 순환논리 설계를 다시 만들지 않기 위한 회귀 assert ---
assert explicit_none_count == 0, "Q17 명시적 '없음' 코드 사용자 수 가정이 깨졌습니다 — 9-0-1절 로직 재검토 필요"
assert match_rate > 0.99, f"Q10 무경험/Q17 무응답 일치율이 {match_rate*100:.1f}%로 낮아졌습니다 — 스킵 로직 가정 재검토 필요"
print(f"\\n[OK] Q1/Q10→Q17 스킵 로직 재확인 — Stage1(참여의향 있음/없음) 순환논리 모델을 다시 설계하면 AUC=1.000 함정에 빠질 위험이 여전히 있습니다.")
""")

    # ------------------------------------------------------------------
    # 10) "군지역" 잔여 표기 정정 (데모 프로필 셀)
    #    -- 하드코딩된 인덱스("122+2")는 이후 세션에서 다른 셀들이 여럿 추가/삽입되며 더 이상
    #       유효하지 않아 anchor 문자열 기반 탐색으로 교체함 (2026-08-30 재실행 시 발견/수정)
    # ------------------------------------------------------------------
    idx_demo = next(i for i, c in enumerate(cells) if "demo_profiles = pd.DataFrame" in "".join(c["source"]))
    demo_src = "".join(cells[idx_demo]["source"])
    assert "군지역" in demo_src, f"anchor 탐색 오류: cells[{idx_demo}]에 '군지역' 문자열이 없습니다"
    demo_src_fixed = demo_src.replace("군지역", "읍면지역")
    cells[idx_demo]["source"] = to_source(demo_src_fixed)
    cells[idx_demo]["execution_count"] = None
    cells[idx_demo]["outputs"] = []

    nb["cells"] = cells
    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장 완료: {NB_PATH} (총 {len(cells)}개 셀, 이전 {len(cells)-2}개에서 +2)")


if __name__ == "__main__":
    main()

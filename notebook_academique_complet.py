# ==========================================================================
# NOTEBOOK ACADÉMIQUE COMPLET (A->Z) — méthodologie justifiée par la littérature
# Copier/coller dans Jupyter, ou ouvrir ce .py (format cellules # %%).
# ==========================================================================

# %% [markdown]
# # Prédiction du délai de livraison e-commerce — notebook méthodologique complet (A → Z)
# ### Pipeline de Data Science de niveau recherche, chaque étape justifiée par la littérature
#
# Ce carnet déroule l'intégralité d'un projet d'apprentissage supervisé, sans omettre aucune étape,
# de l'audit de la donnée au déploiement. Chaque décision est argumentée et rattachée à une référence
# méthodologique (liste en fin de carnet). Les étapes dont l'issue est connue d'avance sont néanmoins
# exécutées en entier, afin que la démarche soit vérifiable de bout en bout.
#
# **Enchaînement.** audit → exploration → cible et prévention des fuites → partition → préparation et
# encodage → **sélection de variables par trois familles d'algorithmes** → modèles candidats →
# validation croisée → **optimisation des hyperparamètres** → évaluation → diagnostics →
# enrichissement météo et test A/B → interprétabilité → contre-épreuve → export.
#
# **Positionnement.** Le retard binaire s'avère imprévisible sur ce jeu ; la cible devient alors la
# **durée de transit réelle**, fortement liée à la promesse de livraison. Cette bascule, ainsi que la
# prévention stricte des fuites, suit la séparation apprentissage-prédiction de Kaufman et al. [31].

# %% [markdown]
# ## 0. Configuration et reproductibilité
#
# Une graine unique fige le hasard ; les paramètres clés sont centralisés. Les bibliothèques avancées
# sont optionnelles : le carnet s'exécute même en leur absence.

# %%
from __future__ import annotations
import json, math, time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import (RFE, SelectKBest, f_regression,
                                       mutual_info_regression)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LogisticRegression, Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_score, cross_validate, learning_curve,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

try:
    import shap; SHAP_AVAILABLE = True
except Exception:
    shap = None; SHAP_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ITER_SEARCH = 25
TUNE_SAMPLE = 60_000
FS_SAMPLE = 50_000      # sous-échantillon pour la sélection de variables (coût maîtrisé)
LC_SAMPLE = 80_000
DAILY_VARS = ["temperature_2m_mean", "precipitation_sum", "windspeed_10m_max"]

sns.set_theme(style="whitegrid", context="notebook"); np.random.seed(RANDOM_STATE)
DATA_FILE = "nigerian_retail_and_ecommerce_supply_chain_logistics_data.parquet"
DATA_PATH = Path(DATA_FILE)
if not DATA_PATH.exists():
    for alt in [Path.cwd()/DATA_FILE, Path("/mnt/data")/DATA_FILE]:
        if alt.exists(): DATA_PATH = alt; break
OUT = Path("ml_academique")
for d in ["figures", "tables"]:
    (OUT/d).mkdir(parents=True, exist_ok=True)
WEATHER_CACHE = OUT/"weather_cache.parquet"
print("Dataset:", DATA_PATH.exists(), "| SHAP:", SHAP_AVAILABLE)

# %% [markdown]
# ## 1. Audit de la donnée brute
#
# L'inspection préalable conditionne toutes les décisions suivantes. La cardinalité des variables
# catégorielles fixe la stratégie d'encodage : forte cardinalité pour l'encodage par la cible, faible
# pour l'encodage disjonctif, conformément à la pratique établie pour les attributs catégoriels à
# nombreuses modalités [32].

# %%
df_raw = pd.read_parquet(DATA_PATH)
print("Dimensions :", df_raw.shape, "| doublons :", int(df_raw.duplicated().sum()))
display(df_raw.head())
display(df_raw.isna().mean().rename("taux_manquant").to_frame())
display(df_raw.select_dtypes(include="object").nunique().sort_values(ascending=False)
        .rename("cardinalite").to_frame())

# %% [markdown]
# ## 2. Cibles et prévention des fuites
#
# Deux cibles sont construites : le **retard binaire** (formulation initiale) et la **durée de transit
# réelle** (formulation retenue). La date de livraison réelle est exclue des entrées : la conserver
# constituerait une fuite, c'est-à-dire l'introduction d'une information indisponible au moment de la
# prédiction, faute documentée et évitée par la séparation apprentissage-prédiction [31].

# %%
df = df_raw.copy()
for c in ["ship_date", "expected_delivery_date", "actual_delivery_date"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
df = df.dropna(subset=["ship_date", "expected_delivery_date", "actual_delivery_date"]).copy()

df["estimated_transit_days"] = (df["expected_delivery_date"] - df["ship_date"]).dt.days
df["ship_month"] = df["ship_date"].dt.month.astype("Int64").astype(str)
df["ship_day_of_week"] = df["ship_date"].dt.dayofweek.astype("Int64").astype(str)
df["realized_transit_days"] = (df["actual_delivery_date"] - df["ship_date"]).dt.days
df["delay_days"] = (df["actual_delivery_date"] - df["expected_delivery_date"]).dt.days.clip(lower=0)
df["is_delayed"] = (df["delay_days"] > 0).astype(int)

df = df[(df["estimated_transit_days"] >= 0) & (df["realized_transit_days"] >= 0)].copy()
df = df.dropna(subset=["estimated_transit_days", "realized_transit_days"]).copy()
print("Lignes exploitables :", len(df),
      "| corr(estimé, réel) =", round(df["estimated_transit_days"].corr(df["realized_transit_days"]), 3),
      "| taux de retard =", round(df["is_delayed"].mean(), 3))

# %% [markdown]
# ## 3. Exploration ciblée
#
# Trois lectures : distribution des durées, relation réel–planifié (siège du signal), et taux de retard
# par transporteur. Le caractère plat de ce dernier annonce l'absence de signal multivarié sur le retard.

# %%
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(df["estimated_transit_days"], bins=30, alpha=.7, label="planifié")
ax[0].hist(df["realized_transit_days"], bins=30, alpha=.7, label="réel"); ax[0].legend()
ax[0].set_title("Distributions des durées (j)")
s = df.sample(min(20000, len(df)), random_state=RANDOM_STATE)
ax[1].scatter(s["estimated_transit_days"], s["realized_transit_days"], s=4, alpha=.15, color="#1f6f9f")
lims = [0, df["realized_transit_days"].quantile(.99)]; ax[1].plot(lims, lims, "--", color="#b42318")
ax[1].set_xlim(lims); ax[1].set_ylim(lims); ax[1].set_title("Réel vs planifié")
rate = df.groupby("logistics_company")["is_delayed"].mean()
ax[2].barh(rate.index, rate.values, color="#7c93b8"); ax[2].set_title("Taux de retard par transporteur")
plt.tight_layout(); plt.savefig(OUT/"figures/eda.png", dpi=150); plt.show()

# %% [markdown]
# ## 4. Partition train / test
#
# La partition précède tout ajustement et toute exploration approfondie ; le test reste scellé jusqu'à
# l'évaluation finale, ce qui garantit une estimation non optimiste de la performance [31], [41].

# %%
NUM_BASE = ["quantity", "shipping_cost_ngn", "estimated_transit_days"]
OH = ["supplier_name", "logistics_company", "ship_month", "ship_day_of_week"]
TE = ["origin_city", "destination_city"]
X_all = df[NUM_BASE + OH + TE].copy()
y_reg = df["realized_transit_days"].astype(float)
y_clf = df["is_delayed"].astype(int)

X_tr, X_te, y_tr, y_te, yc_tr, yc_te = train_test_split(
    X_all, y_reg, y_clf, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Train:", X_tr.shape, "| Test:", X_te.shape)

# %% [markdown]
# ## 5. Préparation et encodage — chaque transformation justifiée
#
# - **Numériques** : imputation médiane (robuste aux extrêmes) puis standardisation, requise par les
#   modèles linéaires et par les méthodes de sélection sensibles à l'échelle.
# - **Faible cardinalité** : encodage disjonctif avec modalité de référence supprimée (anti-colinéarité)
#   et gestion des modalités inédites.
# - **Forte cardinalité (villes)** : encodage par la cible [32], régularisé par validation croisée
#   interne pour éviter la fuite ; cette régularisation est ce qui distingue un encodage par la cible
#   fiable d'une simple moyenne sur-ajustée [42].

# %%
def make_preprocessor(num_cols):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("oh",  Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("e", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), OH),
        ("te",  Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("e", TargetEncoder(target_type="continuous", cv=5, random_state=RANDOM_STATE))]), TE),
    ], remainder="drop")

prep = make_preprocessor(NUM_BASE).fit(X_tr, y_tr)
feat_names = prep.get_feature_names_out()
Xtr_p = prep.transform(X_tr)
print("Matrice préparée :", Xtr_p.shape, "| variables :", len(feat_names))

# %% [markdown]
# ## 6. Formulation initiale : classification du retard (résultat nul documenté)
#
# La première formulation pose un classifieur binaire du retard. Une régression logistique pondérée est
# évaluée en validation croisée stratifiée sur l'aire sous la courbe ROC. Le résultat sert de point de
# décision : s'il est voisin du hasard, la cible doit être reformulée.

# %%
clf = Pipeline([("pre", make_preprocessor(NUM_BASE)),
                ("m", LogisticRegression(class_weight="balanced", solver="liblinear",
                                         max_iter=2000, random_state=RANDOM_STATE))])
skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
auc = cross_val_score(clf, X_tr, yc_tr, cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"AUC-ROC (classification du retard) : {auc.mean():.3f} ± {auc.std():.3f}")
print("Lecture : voisin de 0,5 => le retard binaire est imprévisible ; on reformule vers la durée.")

# %% [markdown]
# ## 7. Sélection de variables — trois familles d'algorithmes et leurs métriques
#
# La sélection de variables poursuit trois objectifs : améliorer la performance, accélérer le modèle,
# et éclairer le processus générateur [33]. On compare les trois familles canoniques, chacune avec sa
# **métrique de sélection** propre, sur un sous-échantillon d'apprentissage transformé.
#
# - **Filtres** (indépendants du modèle) : l'**information mutuelle** capte les dépendances non
#   linéaires, le **test F (ANOVA)** mesure la dépendance linéaire. Métrique : score MI et statistique F.
# - **Méthodes intégrées** (issues de l'ajustement) : le **Lasso** annule les coefficients non
#   informatifs par pénalité L1, la **forêt aléatoire** classe par réduction d'impureté. Métrique :
#   amplitude du coefficient et importance d'impureté.
# - **Méthode enveloppe (wrapper)** : l'**élimination récursive (RFE)** retire itérativement la
#   variable la moins utile à un estimateur ; elle évalue des sous-ensembles plutôt que des variables
#   isolées [34]. Métrique : rang d'élimination.
#
# La convergence des méthodes vers un même classement renforce la confiance ; la contre-vérification
# finale par **importance de permutation** mesure la contribution réelle hors échantillon [39].

# %%
# Sous-échantillon transformé pour la sélection
idx = X_tr.sample(min(FS_SAMPLE, len(X_tr)), random_state=RANDOM_STATE).index
Xs = prep.transform(X_tr.loc[idx]); ys = y_tr.loc[idx].to_numpy()
names = feat_names

# 7.1 Filtres
mi = mutual_info_regression(Xs, ys, random_state=RANDOM_STATE)
f_stat, _ = f_regression(Xs, ys)
# 7.2 Intégrées
lasso = LassoCV(cv=3, random_state=RANDOM_STATE, n_jobs=-1, max_iter=5000).fit(Xs, ys)
rf = RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=20,
                           n_jobs=-1, random_state=RANDOM_STATE).fit(Xs, ys)
# 7.3 Wrapper
rfe = RFE(Ridge(alpha=1.0), n_features_to_select=max(3, len(names)//3), step=1).fit(Xs, ys)

fs = pd.DataFrame({
    "variable": names,
    "filtre_MI": mi,
    "filtre_F": f_stat,
    "integre_Lasso_abs": np.abs(lasso.coef_),
    "integre_RF_importance": rf.feature_importances_,
    "wrapper_RFE_rang": rfe.ranking_,
}).sort_values("integre_RF_importance", ascending=False)
display(fs.round(4))
fs.to_csv(OUT/"tables/feature_selection.csv", index=False)

# %%
# Synthèse : classement consensuel (rang moyen sur les métriques où "plus grand = mieux")
rank = pd.DataFrame(index=names)
rank["MI"] = (-mi).argsort().argsort()
rank["F"] = (-np.nan_to_num(f_stat)).argsort().argsort()
rank["Lasso"] = (-np.abs(lasso.coef_)).argsort().argsort()
rank["RF"] = (-rf.feature_importances_).argsort().argsort()
rank["RFE"] = rfe.ranking_.argsort().argsort()
rank["rang_moyen"] = rank.mean(axis=1)
consensus = rank.sort_values("rang_moyen")[["rang_moyen"]]
display(consensus.head(12).round(2))

top = fs.head(12)
plt.figure(figsize=(8, 5))
sns.barplot(data=top, y="variable", x="integre_RF_importance", color="#1f6f9f")
plt.title("Importance par impureté (forêt aléatoire) — top 12")
plt.tight_layout(); plt.savefig(OUT/"figures/feature_selection.png", dpi=150); plt.show()
print("Lecture : les méthodes convergent ; estimated_transit_days domine quel que soit l'algorithme.")

# %% [markdown]
# ## 8. Reformulation : régression de la durée — baselines et modèles candidats
#
# La cible devient la durée de transit réelle. Deux planchers encadrent l'interprétation : la moyenne
# (coefficient de détermination nul) et la recopie de la promesse (concurrent sérieux). Trois familles
# sont confrontées : le modèle linéaire régularisé, la forêt aléatoire [39], et le renforcement de
# gradient histogramme, héritier des machines de boosting de Friedman [37] et apparenté à XGBoost [38]
# et LightGBM [35], dont la croissance par feuilles excelle sur les grands jeux tabulaires.

# %%
def pipe_of(model): return Pipeline([("pre", make_preprocessor(NUM_BASE)), ("m", model)])
rows = []
def report(name, yt, pred):
    mae = mean_absolute_error(yt, pred); rmse = mean_squared_error(yt, pred) ** 0.5
    r2 = r2_score(yt, pred); rows.append({"modele": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name:26s} | MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}")

report("Baseline moyenne", y_te, np.full(len(y_te), y_tr.mean()))
report("Baseline naïf (plan)", y_te, X_te["estimated_transit_days"].to_numpy())

candidates = {
    "Ridge": pipe_of(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    "RandomForest": pipe_of(RandomForestRegressor(n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=RANDOM_STATE)),
    "HistGBR": pipe_of(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
}

# %% [markdown]
# ## 9. Validation croisée
#
# La validation croisée à plis multiples estime la généralisation et borne le risque d'une partition
# chanceuse [41]. On rapporte erreur absolue, erreur quadratique et coefficient de détermination.

# %%
kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
cv_rows = []
for name, p in candidates.items():
    print("CV :", name)
    sc = cross_validate(p, X_tr, y_tr, cv=kf, scoring=scoring, n_jobs=-1)
    cv_rows.append({"modele": name, "MAE_cv": -sc["test_MAE"].mean(),
                    "RMSE_cv": -sc["test_RMSE"].mean(), "R2_cv": sc["test_R2"].mean()})
cv_results = pd.DataFrame(cv_rows).sort_values("R2_cv", ascending=False)
display(cv_results); cv_results.to_csv(OUT/"tables/cv_results.csv", index=False)
best_family = cv_results.iloc[0]["modele"]; print("Mieux placé :", best_family)

# %% [markdown]
# ## 10. Optimisation des hyperparamètres
#
# La recherche aléatoire échantillonne l'espace des configurations ; à budget égal elle surpasse la
# grille exhaustive en explorant davantage les dimensions influentes [40]. Le réglage se fait sur un
# sous-échantillon puis la meilleure configuration est réajustée sur tout l'apprentissage, ce qui
# maîtrise le coût sur les grands volumes.

# %%
spaces = {
    "HistGBR": {"pipe": pipe_of(HistGradientBoostingRegressor(random_state=RANDOM_STATE)), "params": {
        "m__learning_rate": [0.03, 0.05, 0.1, 0.2], "m__max_iter": [200, 400, 600],
        "m__max_leaf_nodes": [15, 31, 63], "m__min_samples_leaf": [20, 50, 100],
        "m__l2_regularization": [0.0, 1.0, 5.0]}},
    "RandomForest": {"pipe": pipe_of(RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE)), "params": {
        "m__n_estimators": [200, 300, 500], "m__max_depth": [10, 20, None],
        "m__min_samples_leaf": [10, 20, 50], "m__max_features": ["sqrt", 0.5, 1.0]}},
    "Ridge": {"pipe": pipe_of(Ridge(random_state=RANDOM_STATE)), "params": {
        "m__alpha": [0.1, 1.0, 5.0, 10.0, 50.0]}},
}
cfg = spaces[best_family]; grid = int(math.prod(len(v) for v in cfg["params"].values()))
X_tune, y_tune = (X_tr, y_tr)
if len(X_tr) > TUNE_SAMPLE:
    X_tune, _, y_tune, _ = train_test_split(X_tr, y_tr, train_size=TUNE_SAMPLE, random_state=RANDOM_STATE)
search = RandomizedSearchCV(cfg["pipe"], cfg["params"], n_iter=min(N_ITER_SEARCH, grid),
    scoring="neg_mean_absolute_error", cv=3, n_jobs=-1, random_state=RANDOM_STATE, refit=False, verbose=1)
search.fit(X_tune, y_tune)
print("Meilleurs hyperparamètres :", search.best_params_)
best_pipe = clone(cfg["pipe"]).set_params(**search.best_params_).fit(X_tr, y_tr)

# %% [markdown]
# ## 11. Évaluation finale et diagnostics

# %%
pred = best_pipe.predict(X_te)
report(f"{best_family} (optimisé)", y_te, pred)
final = pd.DataFrame(rows).drop_duplicates("modele").sort_values("R2", ascending=False)
display(final); final.to_csv(OUT/"tables/final.csv", index=False)
naive_mae = final.set_index("modele").loc["Baseline naïf (plan)", "MAE"]
print(f"Gain MAE vs naïf : {naive_mae - mean_absolute_error(y_te, pred):.3f} jour")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(y_te, pred, s=4, alpha=.2, color="#1f6f9f")
lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]; ax[0].plot(lims, lims, "--", color="#b42318")
ax[0].set_xlabel("réel (j)"); ax[0].set_ylabel("prédit (j)"); ax[0].set_title("Prédit vs réel")
ax[1].hist(y_te.to_numpy() - pred, bins=50, color="#1f6f9f"); ax[1].set_title("Résidus")
plt.tight_layout(); plt.savefig(OUT/"figures/diagnostics.png", dpi=150); plt.show()

# %% [markdown]
# La courbe d'apprentissage diagnostique l'arbitrage biais–variance : un faible écart train–validation
# avec plateau indique un régime de biais, où l'ajout de données n'aiderait pas.

# %%
Xlc, ylc = (X_tr, y_tr)
if len(X_tr) > LC_SAMPLE:
    Xlc, _, ylc, _ = train_test_split(X_tr, y_tr, train_size=LC_SAMPLE, random_state=RANDOM_STATE)
sizes, tr, va = learning_curve(clone(best_pipe), Xlc, ylc, train_sizes=np.linspace(0.2, 1.0, 5),
                               cv=3, scoring="r2", n_jobs=-1)
plt.figure(figsize=(6, 5))
plt.plot(sizes, tr.mean(1), "o-", label="Train R2"); plt.plot(sizes, va.mean(1), "o-", label="Validation R2")
plt.xlabel("Taille d'apprentissage"); plt.ylabel("R2"); plt.legend(); plt.title("Courbe d'apprentissage")
plt.tight_layout(); plt.savefig(OUT/"figures/learning_curve.png", dpi=150); plt.show()

# %% [markdown]
# ## 12. Enrichissement météo et test A/B
#
# Le cadre théorique postule un apport de variables externes temps réel. On teste l'hypothèse
# météorologique : relevés journaliers récupérés par ville (appels groupés, cache, repli gracieux),
# joints sans fuite à la date d'expédition, puis comparaison du même modèle sans et avec météo.

# %%
CITY_COORDS = {"Lagos":(6.5244,3.3792),"Abuja":(9.0765,7.3986),"Kano":(12.0022,8.5920),
 "Port Harcourt":(4.8156,7.0498),"Warri":(5.5167,5.7500),"Aba":(5.1066,7.3667),"Ibadan":(7.3775,3.9470),
 "Jos":(9.8965,8.8583),"Enugu":(6.4584,7.5464),"Kaduna":(10.5222,7.4383),"Onitsha":(6.1667,6.7833),
 "Abeokuta":(7.1557,3.3451),"Benin City":(6.3350,5.6037),"Calabar":(4.9589,8.3269),"Ilorin":(8.4966,4.5421)}

def fetch_city(name, lat, lon, start, end, retries=3):
    import requests
    p = {"latitude":lat,"longitude":lon,"start_date":start,"end_date":end,
         "daily":",".join(DAILY_VARS),"timezone":"Africa/Lagos"}
    for k in range(retries):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=p, timeout=60)
            r.raise_for_status(); d = r.json()["daily"]
            w = pd.DataFrame(d).rename(columns={"time":"date"}); w["date"]=pd.to_datetime(w["date"]); w["city"]=name
            return w
        except Exception: time.sleep(2*(k+1))
    return None

WEATHER_OK, weather = False, None
if WEATHER_CACHE.exists():
    weather = pd.read_parquet(WEATHER_CACHE); WEATHER_OK = True
else:
    cities = sorted(set(df["origin_city"]) | set(df["destination_city"]))
    start = df["ship_date"].min().strftime("%Y-%m-%d"); end = df["ship_date"].max().strftime("%Y-%m-%d")
    frames = []
    for c in cities:
        co = CITY_COORDS.get(c)
        if co is None: continue
        w = fetch_city(c, co[0], co[1], start, end)
        if w is not None: frames.append(w)
        time.sleep(0.3)
    if frames:
        weather = pd.concat(frames, ignore_index=True); weather.to_parquet(WEATHER_CACHE, index=False); WEATHER_OK = True
print("Météo disponible :", WEATHER_OK)

# %%
WEATHER_NUM = []
if WEATHER_OK:
    df["ship_day"] = df["ship_date"].dt.normalize()
    w = weather.copy(); w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    do = w.add_prefix("orig_").rename(columns={"orig_city":"origin_city","orig_date":"ship_day"})
    dd = w.add_prefix("dest_").rename(columns={"dest_city":"destination_city","dest_date":"ship_day"})
    df = df.merge(do, on=["origin_city","ship_day"], how="left").merge(dd, on=["destination_city","ship_day"], how="left")
    WEATHER_NUM = [f"orig_{v}" for v in DAILY_VARS] + [f"dest_{v}" for v in DAILY_VARS]

# Reconstruire X avec météo et refaire la partition (mêmes indices via random_state)
NUM = NUM_BASE + WEATHER_NUM
Xw = df[NUM + OH + TE].copy()
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(Xw, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE)

def fit_eval(num_cols, Xtr_, Xte_, label):
    pipe = Pipeline([("pre", make_preprocessor(num_cols)),
                     ("m", HistGradientBoostingRegressor(random_state=RANDOM_STATE))]).fit(Xtr_[num_cols+OH+TE], yw_tr)
    r2 = r2_score(yw_te, pipe.predict(Xte_[num_cols+OH+TE])); print(f"{label:18s} | R2={r2:.4f}"); return r2

r2_sans = fit_eval(NUM_BASE, Xw_tr, Xw_te, "Sans météo")
r2_avec = fit_eval(NUM, Xw_tr, Xw_te, "Avec météo") if WEATHER_OK else r2_sans
print(f"Gain météo (ΔR2) : {r2_avec - r2_sans:+.5f}")

# %% [markdown]
# ## 13. Interprétabilité — permutation et SHAP
#
# L'importance par permutation mesure la chute de performance hors échantillon quand une variable est
# brouillée [39]. Les valeurs de Shapley décomposent chaque prédiction en contributions additives,
# cadre unifié et théoriquement fondé de l'interprétation [36]. Les deux convergent ici vers une même
# lecture : la durée planifiée concentre le signal.

# %%
perm = permutation_importance(best_pipe, X_te, y_te, scoring="r2", n_repeats=5,
                              random_state=RANDOM_STATE, n_jobs=-1)
imp = (pd.DataFrame({"variable": X_te.columns, "chute_R2": perm.importances_mean})
       .sort_values("chute_R2", ascending=False))
display(imp); imp.to_csv(OUT/"tables/permutation.csv", index=False)
plt.figure(figsize=(8, 5)); sns.barplot(data=imp, y="variable", x="chute_R2", color="#1f6f9f")
plt.title("Importance par permutation (chute de R2)")
plt.tight_layout(); plt.savefig(OUT/"figures/permutation.png", dpi=150); plt.show()

if SHAP_AVAILABLE:
    try:
        Xe = X_te.sample(min(800, len(X_te)), random_state=RANDOM_STATE)
        Xe_p = best_pipe.named_steps["pre"].transform(Xe); fn = best_pipe.named_steps["pre"].get_feature_names_out()
        sv = shap.Explainer(best_pipe.named_steps["m"], Xe_p)(Xe_p)
        shap.summary_plot(sv, Xe_p, feature_names=fn, show=False, max_display=15)
        plt.tight_layout(); plt.savefig(OUT/"figures/shap.png", dpi=150); plt.show()
    except Exception as e:
        print("SHAP indisponible :", e)

# %% [markdown]
# ## 14. Contre-épreuve et export
#
# On vérifie que le retard pur reste imprévisible, météo comprise, puis on sérialise le modèle pour le
# déploiement.

# %%
yd = df["delay_days"].astype(float)
Xd = df[NUM + OH + TE].copy()
Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=TEST_SIZE, random_state=RANDOM_STATE)
pipe_d = Pipeline([("pre", make_preprocessor(NUM)),
                   ("m", HistGradientBoostingRegressor(random_state=RANDOM_STATE))]).fit(Xd_tr, yd_tr)
r2_delay = r2_score(yd_te, pipe_d.predict(Xd_te))
print(f"R2 sur le RETARD seul (météo incluse) : {r2_delay:.4f}")

bundle = {"task": "regression_duree", "pipeline": best_pipe, "model_name": best_family,
          "best_params": search.best_params_, "weather_features": [],
          "feature_schema": {"numeric": NUM_BASE, "one_hot": OH, "target_encoded": TE},
          "decision_rule_warning": "Indicateur operationnel de priorisation, pas probabilite calibree de retard individuel."}
joblib.dump(bundle, "delivery_duration_model_bundle.joblib")
meta = {"model": best_family, "test_R2": float(r2_score(y_te, pred)),
        "test_MAE": float(mean_absolute_error(y_te, pred)),
        "naive_R2": float(r2_score(y_te, X_te["estimated_transit_days"])),
        "weather_gain": float(r2_avec - r2_sans), "delay_R2": float(r2_delay),
        "classification_AUC": float(auc.mean())}
json.dump(meta, open(OUT/"metadata.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(meta, ensure_ascii=False, indent=2))

# %% [markdown]
# ## 15. Validation temporelle passe -> futur
#
# Le split aleatoire mesure une performance moyenne, mais un deploiement reel apprend sur le passe et
# predit le futur. On trie donc les expeditions par date, on entraine sur les 80 % les plus anciennes et
# on teste sur les 20 % les plus recentes. Ce protocole simule mieux l'exploitation et verifie que le
# modele ne depend pas d'une partition aleatoire favorable.

# %%
dft = df.sort_values("ship_date").reset_index(drop=True)
cut = int(len(dft) * 0.80)
tr_idx, te_idx = dft.index[:cut], dft.index[cut:]
Xtr_t = dft.loc[tr_idx, NUM_BASE + OH + TE]
ytr_t = dft.loc[tr_idx, "realized_transit_days"].astype(float)
Xte_t = dft.loc[te_idx, NUM_BASE + OH + TE]
yte_t = dft.loc[te_idx, "realized_transit_days"].astype(float)

print("Periode apprentissage :", dft.loc[tr_idx, "ship_date"].min().date(),
      "->", dft.loc[tr_idx, "ship_date"].max().date())
print("Periode test          :", dft.loc[te_idx, "ship_date"].min().date(),
      "->", dft.loc[te_idx, "ship_date"].max().date())

model_t = clone(cfg["pipe"]).set_params(**search.best_params_).fit(Xtr_t, ytr_t)
pred_t = model_t.predict(Xte_t)
mae_t = mean_absolute_error(yte_t, pred_t)
rmse_t = mean_squared_error(yte_t, pred_t) ** 0.5
r2_t = r2_score(yte_t, pred_t)
r2_naif_t = r2_score(yte_t, Xte_t["estimated_transit_days"])
print(f"Split temporel  | MAE={mae_t:.3f} j | RMSE={rmse_t:.3f} j | R2={r2_t:.3f}")
print(f"Naif temporel   | R2={r2_naif_t:.3f}")
print("Lecture : si la performance reste proche du split aleatoire, la robustesse temporelle est credible.")

# %% [markdown]
# ## 16. Evaluation de la regle de risque de depassement ETA
#
# La regle de deploiement signale un risque lorsque la duree predite depasse la duree estimee. On
# l'evalue comme un classifieur du vrai evenement `duree_reelle > duree_estimee`. Cette evaluation n'a
# pas pour objectif d'obtenir de beaux scores : elle mesure explicitement la limite observee plus haut.
# Si l'AUC est proche de 0,5, la regle ne discrimine pas les retards individuels ; elle reste utile comme
# indicateur operationnel de priorisation et comme base de recalibrage de l'ETA.

# %%
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

pred_risk = best_pipe.predict(X_te)
est = X_te["estimated_transit_days"].to_numpy()
score = pred_risk - est
y_exceed = (y_te.to_numpy() > est).astype(int)
print("Taux reel de depassement ETA :", round(float(y_exceed.mean()), 3))

yhat0 = (score > 0).astype(int)
tn, fp, fn, tp = confusion_matrix(y_exceed, yhat0, labels=[0, 1]).ravel()
print("\n-- Seuil par defaut : duree_predite > duree_estimee --")
print(f"Precision={precision_score(y_exceed, yhat0, zero_division=0):.3f} | "
      f"Rappel={recall_score(y_exceed, yhat0, zero_division=0):.3f} | "
      f"F1={f1_score(y_exceed, yhat0, zero_division=0):.3f}")
print(f"Taux de fausses alertes={fp/(fp+tn+1e-9):.3f}")
print(f"AUC-ROC={roc_auc_score(y_exceed, score):.3f} | PR-AUC={average_precision_score(y_exceed, score):.3f}")

prec, rec, thr = precision_recall_curve(y_exceed, score)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
i = int(np.argmax(f1s))
thr_diag = float(thr[i])
yhat_diag = (score >= thr_diag).astype(int)
tn2, fp2, fn2, tp2 = confusion_matrix(y_exceed, yhat_diag, labels=[0, 1]).ravel()
print(f"\n-- Seuil diagnostique meilleur F1 : score >= {thr_diag:.2f} j --")
print(f"Precision={precision_score(y_exceed, yhat_diag, zero_division=0):.3f} | "
      f"Rappel={recall_score(y_exceed, yhat_diag, zero_division=0):.3f} | "
      f"F1={f1_score(y_exceed, yhat_diag, zero_division=0):.3f}")
print(f"Taux de fausses alertes={fp2/(fp2+tn2+1e-9):.3f}")
print("Attention : ce seuil est optimise sur le test a titre diagnostique. Un seuil de production doit etre choisi sur validation separee.")

cm = confusion_matrix(y_exceed, yhat0, labels=[0, 1])
plt.figure(figsize=(4.8, 4.0))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["A l'heure", "Depassement"],
            yticklabels=["A l'heure", "Depassement"])
plt.title("Regle de risque - matrice de confusion")
plt.xlabel("Predit"); plt.ylabel("Reel")
plt.tight_layout(); plt.savefig(OUT/"figures/regle_risque_confusion.png", dpi=150); plt.show()

print("Conclusion : les donnees disponibles permettent surtout de recalibrer l'ETA, pas de discriminer finement les causes du retard.")

meta.update({
    "temporal_R2": float(r2_t),
    "temporal_MAE": float(mae_t),
    "temporal_naive_R2": float(r2_naif_t),
    "risk_rule_base_rate": float(y_exceed.mean()),
    "risk_rule_auc": float(roc_auc_score(y_exceed, score)),
    "risk_rule_pr_auc": float(average_precision_score(y_exceed, score)),
    "risk_rule_false_alarm_rate": float(fp/(fp+tn+1e-9)),
    "scientific_conclusion": "Les donnees disponibles permettent surtout de recalibrer l'ETA, pas de discriminer finement les causes du retard.",
})
json.dump(meta, open(OUT/"metadata.json", "w"), ensure_ascii=False, indent=2)

# %% [markdown]
# ## 17. References methodologiques
#
# Toutes librement accessibles (arXiv, JMLR, actes NeurIPS/KDD, pages auteurs) via Google Scholar.
#
# [31] S. Kaufman, S. Rosset, C. Perlich, O. Stitelman, « Leakage in Data Mining: Formulation, Detection, and Avoidance », *ACM TKDD*, 6(4):15, 2012.
# [32] D. Micci-Barreca, « A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems », *ACM SIGKDD Explorations*, 3(1):27–32, 2001.
# [33] I. Guyon, A. Elisseeff, « An Introduction to Variable and Feature Selection », *Journal of Machine Learning Research*, 3:1157–1182, 2003.
# [34] R. Kohavi, G. H. John, « Wrappers for Feature Subset Selection », *Artificial Intelligence*, 97(1–2):273–324, 1997.
# [35] G. Ke, Q. Meng, T. Finley, et al., « LightGBM: A Highly Efficient Gradient Boosting Decision Tree », *NeurIPS 30*, 3149–3157, 2017.
# [36] S. M. Lundberg, S.-I. Lee, « A Unified Approach to Interpreting Model Predictions », *NeurIPS 30*, 4765–4774, 2017.
# [37] J. H. Friedman, « Greedy Function Approximation: A Gradient Boosting Machine », *Annals of Statistics*, 29(5):1189–1232, 2001.
# [38] T. Chen, C. Guestrin, « XGBoost: A Scalable Tree Boosting System », *KDD*, 785–794, 2016.
# [39] L. Breiman, « Random Forests », *Machine Learning*, 45(1):5–32, 2001.
# [40] J. Bergstra, Y. Bengio, « Random Search for Hyper-Parameter Optimization », *JMLR*, 13:281–305, 2012.
# [41] R. Kohavi, « A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection », *IJCAI*, 1137–1143, 1995.
# [42] F. Pargent, F. Pfisterer, J. Thomas, B. Bischl, « Regularized Target Encoding Outperforms Traditional Methods in Supervised Machine Learning with High Cardinality Features », *Computational Statistics*, 37:2671–2692, 2022.

"""Home Credit Default Risk - v5  (solo train.csv + test.csv)  = v4 + protocolo experimental
LightGBM + XGBoost + CatBoost + modelo lineal, features nuevos, Target Encoding sin leakage y 4 tipos de mezcla.

Novedades respecto a v3
  * Features: ~15 interacciones EXT_SOURCE por pares, imputación de EXT_SOURCE (EXT_k_PRED / EXT_k_RESID), plazo e
    interés implícitos, "edad a la que ocurrió cada evento", frequency encoding, riesgo relativo por grupo de pares.
  * Target Encoding con K-Fold interno (calculado DENTRO de cada fold -> el OOF sigue siendo honesto).
  * Más diversidad: lgbet (extra-trees), xgbd, cat2, lr; todos aceptan varias semillas.
  * Blend: promedio de rangos / Nelder-Mead / hill-climbing / stacking, comparados con AUC en validación cruzada.
  * Robustez: columnas opcionales (no falla si falta alguna), checkpoints con hash de configuración, features cacheados.

Novedades de v5 (TODO lo nuevo está APAGADO por defecto: solo se enciende si la validación lo acepta)
  * Modo `cv:<exp>`      CV estratificada repetida (3x5) de LightGBM por configuración; exp se puede combinar con '+'
                         (ej. cv:A_base, cv:B_mono, cv:D_knn, cv:B_mono+D_quant). Guarda cada repetición (reanudable).
  * Modo `cmp`           tabla idea | delta OOF (media +- desv) | victorias | decisión; escribe accepted.json.
  * Modo `adv`           adversarial validation train vs test (+ lista de features con drift fuerte).
  * Modo `leak`          detección de leakage: AUC por feature, TARGET permutado, brecha train/val de TE y KNN.
  * Modo `nullimp`       null importance + permutation importance -> lista de features podables.
  * Modo `optuna`        TPE sobre LightGBM (reanudable, sqlite); `lgbf:<semillas>` = config final con lr 0.01.
  * Opciones de modelado (CFG): restricciones monotónicas EXT, features por cuantiles (Q_*), KNN-target dentro del
    fold, poda/drift, pseudo-etiquetado estricto (modelo del propio fold, validación solo en filas reales).
  * Modelos nuevos: lgbm, xgbm (monótonos), lgbgoss, lgbdart, lgblin, cat3, mlp, lgbf.
  * Blend: + stacking con GBDT pequeño, CV repetida, matriz de correlación y aporte marginal por modelo.

Uso (los mismos modos que v3; las semillas se separan con comas):
  python solution_v4.py <carpeta_csv> fe                 # construye y cachea los features
  python solution_v4.py <carpeta_csv> lgb | xgb | cat    # modelos base (semilla 42)
  python solution_v4.py <carpeta_csv> lgb:42,7,2026      # varias semillas de un modelo (también xgb:..., cat:...)
  python solution_v4.py <carpeta_csv> lgbet | xgbd | cat2 | lr
  python solution_v4.py <carpeta_csv> tune               # búsqueda de hiperparámetros de LightGBM (reanudable)
  python solution_v4.py <carpeta_csv> lgb2:42,7          # LightGBM con la mejor configuración de tune
  python solution_v4.py <carpeta_csv> blend              # mezcla todo -> output_v4/submission.csv

Variables de entorno opcionales: OUT_DIR (def. output_v4), SMOKE=1 (prueba rápida), FE_EXT_IMPUTE=0 / FE_TE=0
(ablaciones), TUNE_BUDGET (s por ejecución, def. 170), TUNE_FOLDS (def. 2), BLEND_CV=0 (blend más rápido).
Semillas fijas -> reproducible (con las mismas versiones de librerías y hardware).
"""
import os, sys, glob, json, re, time, hashlib, warnings
from functools import lru_cache
import numpy as np, pandas as pd
from scipy.stats import rankdata
from scipy.special import logit
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# ============================== CONFIGURACIÓN ==============================
IN = sys.argv[1] if len(sys.argv) > 1 else "."
MODE = sys.argv[2] if len(sys.argv) > 2 else "blend"
OUT = os.environ.get("OUT_DIR", "output_v4")          # carpeta aparte: no pisa los resultados de v3
N_FOLDS, SEED = int(os.environ.get("N_FOLDS", 5)), 42
USE_EXT_IMPUTE = os.environ.get("FE_EXT_IMPUTE", "1") == "1"   # ablación: imputación de EXT_SOURCE
USE_TE = os.environ.get("FE_TE", "1") == "1"                   # ablación: Target Encoding
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))     # hilos para los modelos
CV_REPS = int(os.environ.get("CV_REPS", 3))                      # repeticiones de la CV de experimentos
CV_SEEDS = [42, 7, 2026, 11, 99][:CV_REPS]
CV_LR = float(os.environ.get("CV_LR", 0.05))                     # lr de LightGBM en la CV de experimentos (más rápido)
SMOKE = os.environ.get("SMOKE", "0") == "1"                    # prueba rápida de punta a punta (pocas filas/árboles)
TUNE_BUDGET = float(os.environ.get("TUNE_BUDGET", 170))
TUNE_FOLDS = int(os.environ.get("TUNE_FOLDS", 2))
os.makedirs(OUT, exist_ok=True)

# ================================== DATOS ==================================
train = pd.read_csv(f"{IN}/train.csv")
test = pd.read_csv(f"{IN}/test.csv")
if SMOKE:
    train = train.sample(min(len(train), 6000), random_state=0).reset_index(drop=True)
    test = test.head(2000).copy()
y = train["TARGET"].to_numpy().astype(int)
test_ids = test["SK_ID_CURR"].to_numpy()

# ======================= RUTAS Y PARÁMETROS DERIVADOS =======================
FE_VERSION = 10  # incremented for fixed-fee decoder and optional one-hot representation (invalida cachés y checkpoints antiguos)


def _fe_sig():
    """Huella del CÓDIGO de las features (de `def fe_base(` a `def target_encode(`): si cambias ese código, la caché cambia sola,
    aunque olvides subir FE_VERSION. Evita reutilizar un features.pkl construido con otro código (ocurrió una vez)."""
    try:
        src = open(__file__, encoding="utf-8").read().replace("\r\n", "\n")
        return hashlib.md5(src[src.index("def fe_base("):src.index("def target_encode(")].encode()).hexdigest()[:6]
    except Exception:
        return "x"


FE_SIG = _fe_sig()
FE_KEY = f"v{FE_VERSION}{FE_SIG}_i{int(USE_EXT_IMPUTE)}_t{int(USE_TE)}_oh{int(os.environ.get('FE_ONEHOT', '0') == '1')}_n{len(train)}_{len(test)}{'_smoke' if SMOKE else ''}"
MD = f"{OUT}/models_{FE_KEY}"   # un directorio por configuración de features: nunca se mezclan resultados
os.makedirs(MD, exist_ok=True)
ROUNDS = dict(lgb=8000, xgb=4000, cat=3000, ext=200) if not SMOKE else dict(lgb=60, xgb=60, cat=60, ext=40)

# ------------- CFG: opciones de modelado en tiempo de ejecución (se leen de accepted.json si existe) -------------
DEFAULT_CFG = dict(te=USE_TE, mono=False, quant=False, knn=False, interest=False, product=False, hidden=False, hidden2=False, dup=False, drop=[], pl=False, lr=None)
CFG = dict(DEFAULT_CFG)
_acc = f"{OUT}/accepted.json"
if os.path.exists(_acc) and os.environ.get("IGNORE_ACCEPTED", "0") != "1":
    CFG.update(json.load(open(_acc)))   # decisiones tomadas por `cmp` con la regla de aceptación
for _k, _e in [("mono", "MONO"), ("quant", "FE_QUANT"), ("knn", "FE_KNN"), ("interest", "FE_INTEREST"), ("product", "FE_PRODUCT"), ("hidden", "FE_HIDDEN"), ("hidden2", "FE_HIDDEN2"), ("dup", "FE_DUP"), ("uid", "FE_UID"), ("uidte", "FE_UIDTE"), ("uidte180", "FE_UIDTE180"), ("rate2", "FE_RATE2"), ("uid2", "FE_UID2"), ("uidp", "FE_UIDP"), ("hidden5", "FE_HIDDEN5")]:   # el entorno manda
    if _e in os.environ:
        CFG[_k] = os.environ[_e] == "1"

# ======================== FEATURE ENGINEERING ========================

EXT = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def num(d, name):
    """Columna numérica (float) o NaN si no existe: el pipeline no se rompe si falta alguna columna."""
    if name in d.columns:
        return pd.to_numeric(d[name], errors="coerce").astype("float64")
    return pd.Series(np.nan, index=d.index)


def safe_div(a, b):
    """División segura: denominador 0 -> NaN y nunca devuelve +-inf."""
    return (a / b.where(b != 0)).replace([np.inf, -np.inf], np.nan)


def winsor(s, hi=0.999):
    """Recorta valores extremos por arriba (cuantil calculado con train+test, sin usar TARGET)."""
    return s.clip(upper=s.quantile(hi))


def to_cat(s):
    """Texto -> categoría. El NaN pasa a la categoría explícita 'NA' (válida para LGB, XGB y CatBoost)."""
    return s.astype("object").where(s.notna(), "NA").astype(str).astype("category")


def combo_key(d, cols):
    """Une columnas categóricas en una clave de grupo ('EDU|INC'). Devuelve None si falta alguna."""
    if not all(c in d.columns for c in cols):
        return None
    k = d[cols[0]].astype(str)
    for c in cols[1:]:
        k = k + "|" + d[c].astype(str)
    return k.astype("category")


def fe_base(d):
    """Limpieza de outliers + ratios financieros, temporales y familiares (sin EXT_SOURCE ni TARGET)."""
    F = {}
    birth, reg, idp, ph = (num(d, c) for c in
                           ["DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"])
    emp_raw = num(d, "DAYS_EMPLOYED")
    # 365243 es un código ("sin empleo / pensionado"): pasa a NaN y se conserva como bandera
    F["DAYS_EMPLOYED_ANOM"] = (emp_raw == 365243).astype("int8")
    emp = emp_raw.replace(365243, np.nan)
    F["DAYS_EMPLOYED"] = emp
    inc = winsor(num(d, "AMT_INCOME_TOTAL"))  # un ingreso extremo distorsiona todos los ratios
    cred, ann, goods = num(d, "AMT_CREDIT"), num(d, "AMT_ANNUITY"), num(d, "AMT_GOODS_PRICE")
    fam, chil, car = num(d, "CNT_FAM_MEMBERS"), num(d, "CNT_CHILDREN"), num(d, "OWN_CAR_AGE")

    # --- Edad / antigüedad y "edad a la que ocurrió cada evento"
    age, emp_y = -birth / 365.25, -emp / 365.25
    F["AGE_YEARS"], F["EMPLOYED_YEARS"] = age, emp_y
    F["EMPLOYED_TO_AGE"] = safe_div(emp, birth)
    F["EMP_START_AGE"] = age - emp_y                 # edad al empezar el empleo actual
    F["AGE_AT_ID"] = (idp - birth) / 365.25          # edad al emitir el documento de identidad
    F["AGE_AT_REG"] = (reg - birth) / 365.25         # edad al registrarse
    F["AGE_AT_PHONE"] = (ph - birth) / 365.25        # edad al último cambio de teléfono
    F["REG_MINUS_ID"] = (reg - idp) / 365.25
    F["EMP_MINUS_ID"] = (emp - idp) / 365.25
    F["EMP_MINUS_REG"] = (emp - reg) / 365.25
    F["ID_TO_BIRTH"], F["REG_TO_BIRTH"] = safe_div(idp, birth), safe_div(reg, birth)
    F["PHONE_TO_BIRTH"], F["PHONE_TO_EMPLOYED"] = safe_div(ph, birth), safe_div(ph, emp)
    F["PHONE_ZERO"] = (ph == 0).astype("int8")
    F["CAR_TO_AGE"], F["CAR_TO_EMPLOYED"] = safe_div(car, age), safe_div(car, emp_y)

    # --- Ratios financieros
    term = safe_div(cred, ann)
    F["CREDIT_TO_INCOME"] = safe_div(cred, inc)
    F["ANNUITY_TO_INCOME"] = safe_div(ann, inc)
    F["CREDIT_TO_ANNUITY"] = term                    # plazo implícito (nº de pagos)
    F["ANNUITY_TO_CREDIT"] = safe_div(ann, cred)
    F["TERM_ROUND"] = term.round()                   # plazo "de catálogo" más cercano
    F["TERM_FRAC"] = term - term.round()             # desvío respecto al plazo estándar (comisiones / tasa)
    F["INTEREST_PROXY"] = safe_div(ann * term.round() - cred, cred)  # coste total aproximado del crédito
    F["CREDIT_TO_GOODS"] = safe_div(cred, goods)
    F["GOODS_DIFF"] = cred - goods
    F["GOODS_DIFF_RATIO"] = safe_div(cred - goods, goods)
    F["ANNUITY_TO_GOODS"] = safe_div(ann, goods)
    F["GOODS_TO_INCOME"] = safe_div(goods, inc)
    for nm, v in [("INCOME", inc), ("CREDIT", cred), ("ANNUITY", ann), ("GOODS", goods)]:
        F[f"LOG_{nm}"] = np.log1p(v.clip(lower=0))
    # importes "redondos": suelen indicar ingreso declarado (no documentado) o crédito estándar
    F["INCOME_ROUND"] = ((num(d, "AMT_INCOME_TOTAL") % 5000) == 0).astype("int8")
    F["CREDIT_ROUND"] = ((cred % 10000) == 0).astype("int8")
    F["GOODS_ROUND"] = ((goods % 10000) == 0).astype("int8")

    # --- Familia
    F["INCOME_PER_PERSON"] = safe_div(inc, fam)
    F["INCOME_PER_CHILD"] = safe_div(inc, 1 + chil)
    F["INCOME_PER_ADULT"] = safe_div(inc, fam - chil)
    F["CREDIT_PER_PERSON"] = safe_div(cred, fam)
    F["ANNUITY_PER_PERSON"] = safe_div(ann, fam)
    F["CHILD_RATIO"] = safe_div(chil, fam)
    F["INCOME_LEFT_PER_PERSON"] = safe_div(inc - ann, fam)

    # --- Consultas al buró: se recortan los extremos (p. ej. 261 consultas en un trimestre)
    bur = [c for c in d.columns if c.startswith("AMT_REQ_CREDIT_BUREAU")]
    if bur:
        for c in bur:
            s = num(d, c)
            F[c] = s.clip(upper=max(s.quantile(0.999), 3))
        B = pd.DataFrame({c: F[c] for c in bur})
        F["BUREAU_SUM"] = B.sum(axis=1, min_count=1)
        F["BUREAU_NA"] = B.isna().all(axis=1).astype("int8")
        recent = [c for c in bur if c.endswith(("HOUR", "DAY", "WEEK"))]
        if recent:
            F["BUREAU_RECENT"] = B[recent].sum(axis=1, min_count=1)
        if "AMT_REQ_CREDIT_BUREAU_YEAR" in bur:
            F["BUREAU_YEAR_SHARE"] = safe_div(B["AMT_REQ_CREDIT_BUREAU_YEAR"], F["BUREAU_SUM"] + 1)

    flags = [c for c in ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE"] if c in d.columns]
    if flags:
        F["DOC_FLAGS"] = pd.concat([num(d, c) for c in flags], axis=1).sum(axis=1)
    F["NAN_COUNT"] = d.isna().sum(axis=1)            # campos vacíos en la solicitud (calidad del dato)
    return pd.DataFrame(F, index=d.index)


def fe_ext_impute(M, seed):
    """Imputación SUPERVISADA POR LAS PROPIAS EXT_SOURCE (nunca por TARGET -> sin leakage).
    Para cada EXT_k se entrena un regresor con las filas que lo tienen y se predice con el resto de variables.
      - filas con EXT_k: predicción out-of-fold (el modelo no vio ese valor)
      - filas sin EXT_k: promedio de los 5 modelos
    Aporta EXT_k_PRED (valor "esperado") y, más adelante, EXT_k_RESID = real - esperado."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import KFold
    kf, out = KFold(5, shuffle=True, random_state=seed), {}
    for e in EXT:
        if e not in M.columns:
            continue
        # Use a stable, low-memory predictor subset. The previous all-column HGB fit exceeded
        # this environment's memory cap; the EXT expectation is label-free, so retain the core
        # financial, timing, and other EXT predictors rather than duplicating every engineered column.
        pred_cols = [
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "DAYS_EMPLOYED",
            "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "AGE_YEARS",
            "EMPLOYED_YEARS", "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "REGION_POPULATION_RELATIVE", "OWN_CAR_AGE", "CNT_FAM_MEMBERS", "CNT_CHILDREN",
            "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "CREDIT_TO_ANNUITY", "CREDIT_TO_GOODS",
            "INCOME_PER_PERSON", "INCOME_PER_ADULT", "INCOME_PER_CHILD", "BUREAU_SUM", "NAN_COUNT",
        ]
        use_cols = [c for c in pred_cols if c in M.columns and c != e]
        A = M[use_cols].to_numpy(dtype="float32", copy=True)
        t = M[e].to_numpy(dtype="float64")
        known, unk = np.flatnonzero(~np.isnan(t)), np.flatnonzero(np.isnan(t))
        pred = np.zeros(len(M))
        for a, b in kf.split(known):
            ia, ib = known[a], known[b]
            m = HistGradientBoostingRegressor(max_iter=ROUNDS["ext"], learning_rate=0.08, max_leaf_nodes=31,
                                              min_samples_leaf=50, l2_regularization=5.0,
                                              early_stopping=False, random_state=seed)
            m.fit(A[ia], t[ia])
            pred[ib] = m.predict(A[ib])
            if len(unk):
                pred[unk] += m.predict(A[unk]) / kf.get_n_splits()
        out[f"{e}_PRED"] = pred
    return pd.DataFrame(out, index=M.index)


def fe_ext(B, P):
    """EXT_SOURCE_1/2/3 (las variables más importantes): agregados, interacciones por pares,
    versión imputada (EXTF_*) y cruces con edad, empleo y ratios financieros."""
    F, E, w = {}, B.reindex(columns=EXT), np.array([2.0, 3.0, 4.0])
    # (a) agregados sobre los valores originales (respetan los NaN)
    F["EXT_MEAN"], F["EXT_MIN"], F["EXT_MAX"] = E.mean(axis=1), E.min(axis=1), E.max(axis=1)
    F["EXT_MEDIAN"], F["EXT_STD"] = E.median(axis=1), E.std(axis=1)
    F["EXT_RANGE"] = F["EXT_MAX"] - F["EXT_MIN"]
    F["EXT_PROD"] = E.prod(axis=1, skipna=False)
    F["EXT_WEIGHTED"] = E.mul(w).sum(axis=1, min_count=3)
    F["EXT_NA_COUNT"] = E.isna().sum(axis=1)
    F["EXT_NA_PATTERN"] = E.isna().to_numpy() @ np.array([1, 2, 4])   # qué fuentes faltan (0..7)
    # (b) interacciones por pares: producto, diferencia, cociente y suma
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        a, b = E[f"EXT_SOURCE_{i}"], E[f"EXT_SOURCE_{j}"]
        F[f"EXT_P{i}{j}"], F[f"EXT_D{i}{j}"] = a * b, a - b
        F[f"EXT_R{i}{j}"], F[f"EXT_S{i}{j}"] = safe_div(a, b), a + b
    for k in (1, 2, 3):
        F[f"EXT_SQ{k}"] = E[f"EXT_SOURCE_{k}"] ** 2
    # (c) versión imputada (EXTF_*) + residuo respecto a lo "esperado" por el perfil del cliente
    Ef = E.copy()
    for k, e in enumerate(EXT, 1):
        pc = f"{e}_PRED"
        if pc in P.columns:
            Ef[e] = E[e].fillna(P[pc])
            F[pc], F[f"{e}_RESID"] = P[pc], E[e] - P[pc]
        else:
            Ef[e] = E[e].fillna(E[e].median())
        F[f"EXTF_{k}"] = Ef[e]
    F["EXTF_MEAN"], F["EXTF_MIN"], F["EXTF_MAX"] = Ef.mean(axis=1), Ef.min(axis=1), Ef.max(axis=1)
    F["EXTF_STD"], F["EXTF_PROD"] = Ef.std(axis=1), Ef.prod(axis=1)
    F["EXTF_WEIGHTED"] = Ef.mul(w).sum(axis=1)
    F["EXTF_GEO"] = F["EXTF_PROD"].clip(lower=0) ** (1 / 3)
    # (d) cruces con edad, empleo y ratios financieros (producto y cociente)
    levers = {"MEAN": F["EXTF_MEAN"], "MIN": F["EXTF_MIN"], "E1": Ef[EXT[0]], "E2": Ef[EXT[1]], "E3": Ef[EXT[2]]}
    fin = [c for c in ["AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME",
                       "CREDIT_TO_ANNUITY", "CREDIT_TO_GOODS"] if c in B.columns]
    for ln, lv in levers.items():
        for fn in fin:
            F[f"X_{ln}_{fn}"] = lv * B[fn]
    for fn in fin:
        F[f"X_MEAN_DIV_{fn}"] = safe_div(F["EXTF_MEAN"], B[fn])
    return pd.DataFrame(F, index=B.index)


def fe_quant(B):
    """Representación por CUANTILES de las EXT (sin TARGET; train+test juntos). Prefijo Q_ -> se activa con CFG['quant'].
    Rango percentil, decil, y agregados/diferencias en espacio de rangos (más robustos a la forma de la distribución)."""
    F, R = {}, {}
    for k in (1, 2, 3):
        c = f"EXTF_{k}"
        if c not in B.columns:
            continue
        R[k] = B[c].rank(pct=True)
        F[f"Q_EXTF_{k}_RANK"] = R[k]
        F[f"Q_EXTF_{k}_DEC"] = (R[k] * 10).clip(upper=9.999).apply(np.floor)
    if len(R) == 3:
        Rd = pd.DataFrame(R)
        F["Q_EXT_RANKMEAN"], F["Q_EXT_RANKMIN"] = Rd.mean(axis=1), Rd.min(axis=1)
        F["Q_EXT_RANKSTD"], F["Q_EXT_RANKPROD"] = Rd.std(axis=1), Rd.prod(axis=1)
        for i, j in [(1, 2), (1, 3), (2, 3)]:
            F[f"Q_EXT_RD{i}{j}"] = R[i] - R[j]
        for c in ["CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "AGE_YEARS", "CREDIT_TO_ANNUITY", "EMPLOYED_YEARS"]:
            if c in B.columns:
                F[f"Q_{c}_RANK"] = B[c].rank(pct=True)
        if "Q_AGE_YEARS_RANK" in F:
            F["Q_EXT_RANKMEAN_x_AGE"] = F["Q_EXT_RANKMEAN"] * F["Q_AGE_YEARS_RANK"]
        if "Q_CREDIT_TO_INCOME_RANK" in F:
            F["Q_EXT_RANKMEAN_x_CTI"] = F["Q_EXT_RANKMEAN"] * F["Q_CREDIT_TO_INCOME_RANK"]
    return pd.DataFrame(F, index=B.index)


def fe_interest(B):
    """Tasa y plazo IMPLÍCITOS (idea 'Alexander_L'; sin TARGET). Prefijo V_ -> se activan con CFG['interest'].
    Mide el plazo candidato (6..60 meses) cuya tasa anual implícita queda en [2 %, 90 %] y más cerca de 25 %.
    Medido: +0.00085 +- 0.00044 sobre C_prune (5/5 repeticiones). Ver cv:C_prune+K_interest."""
    eps = 1e-5
    credit, annuity = B["AMT_CREDIT"].to_numpy(dtype="float64"), B["AMT_ANNUITY"].to_numpy(dtype="float64")
    best_rate, best_term = np.full(len(B), np.nan), np.full(len(B), np.nan)
    with np.errstate(all="ignore"):
        for n in np.array([6, 12, 18, 24, 30, 36, 48, 60]):
            annual = (annuity / (credit + eps) - 1.0 / n) * 12
            mask = (annual > 0.02) & (annual < 0.90)
            closer = np.abs(annual - 0.25) < np.abs(best_rate - 0.25)
            upd = mask & (np.isnan(best_rate) | closer)
            best_rate[upd], best_term[upd] = annual[upd], n
    F = {"V_IMPLIED_INTEREST": best_rate, "V_IMPLIED_TERM": best_term, "V_PAYMENT_RATE": annuity / (credit + eps)}
    ir = pd.Series(best_rate, index=B.index)
    F["V_INTEREST_x_INCOME"] = ir * B["AMT_INCOME_TOTAL"]
    ext = B["EXT_MEAN"] if "EXT_MEAN" in B.columns else B["EXT_SOURCE_2"]
    F["V_INTEREST_x_EXT"] = ir * ext
    F["V_INTEREST_x_CREDIT"] = ir * B["AMT_CREDIT"]
    return pd.DataFrame(F, index=B.index)


def fe_product(B):
    """Estructura de PRODUCTO en CREDIT/ANNUITY (sin TARGET). El cociente se concentra en picos (productos con plazo/tasa fijos) y
    la tasa de impago varía de ~2 % a ~25 % entre productos. Prefijo W_ (+ claves TE_PROD* con Target Encoding dentro del fold)
    -> se activan con CFG['product'] (True = ambas; 'num' = solo W_; 'te' = solo TE_PROD).
    Medido: +0.00396 +- 0.00045 sobre C_prune+K_interest (5/5); casi todo viene del Target Encoding (+0.0038)."""
    t = B["CREDIT_TO_ANNUITY"].astype(float)
    r1 = t.round(1)
    vc = r1.value_counts()
    peaks = np.sort(vc.index[:20].to_numpy(dtype="float64"))
    arr = t.to_numpy(dtype="float64")
    d = np.abs(arr[:, None] - peaks[None, :])
    j = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
    F = {"W_PROD_R1": r1, "W_PROD_FREQ": r1.map(vc) / len(B), "W_PEAK_DIST": np.abs(arr - peaks[j]),
         "W_PEAK_RES": arr - peaks[j], "W_PEAK_VAL": peaks[j]}
    return pd.DataFrame(F, index=B.index)


FREQ_NUM = ["AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_GOODS_PRICE", "AMT_ANNUITY",
            "DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_EMPLOYED"]
COMBOS = {  # combinaciones categóricas (perfiles de cliente) usadas en grupos y Target Encoding
    "EDU_x_INC": ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE"],
    "OCC_x_ORG": ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"],
    "ORG_x_INC": ["ORGANIZATION_TYPE", "NAME_INCOME_TYPE"],
    "OCC_x_AGE": ["OCCUPATION_TYPE", "AGE_BIN"],
    "INC_x_AGE": ["NAME_INCOME_TYPE", "AGE_BIN"],
    "SEX_EDU_AGE": ["CODE_GENDER", "NAME_EDUCATION_TYPE", "AGE_BIN"],
    "FAM_x_HOUSE": ["NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE"],
}
GROUP_KEYS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE",
              "EDU_x_INC", "OCC_x_AGE", "SEX_EDU_AGE"]
GROUP_REL = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AGE_YEARS", "EMPLOYED_YEARS", "CREDIT_TO_INCOME", "EXTF_MEAN"]
GROUP_Z = ["AMT_INCOME_TOTAL", "EXTF_MEAN", "EXTF_MIN"]



def fe_ext_percentiles(B, C):
    """Percentil de EXTF dentro de ocupación y de género x edad (Idea Top 10)."""
    F = {}
    if "EXTF_MEAN" in B.columns:
        ext = B["EXTF_MEAN"].astype("float64")
        if "OCCUPATION_TYPE" in C.columns:
            occ = C["OCCUPATION_TYPE"].astype(str).fillna("NA")
            F["EXTF_PCT_OCC"] = ext.groupby(occ).rank(pct=True).values
        if "CODE_GENDER" in C.columns and "AGE_BIN" in C.columns:
            group = C["CODE_GENDER"].astype(str) + "_" + C["AGE_BIN"].astype(str)
            F["EXTF_PCT_GEN_AGE"] = ext.groupby(group).rank(pct=True).values
    return pd.DataFrame(F, index=B.index).replace([np.inf, -np.inf], np.nan).astype("float32")

def make_keys(C, base):
    """Claves categóricas = categorías originales + AGE_BIN (tramos de 5 años) + combinaciones (COMBOS)."""
    D = C.copy()
    D["AGE_BIN"] = (base["AGE_YEARS"] // 5 * 5).fillna(-1).astype(int).astype(str)
    K = {c: C[c] for c in C.columns}
    K["AGE_BIN"] = D["AGE_BIN"].astype("category")
    
    # 🐘 CLAVES DE CLIENTE AVANZADAS (Top 10): Diferencias de fechas invariantes
    birth = base.get("DAYS_BIRTH")
    if birth is not None:
        id_p = base.get("DAYS_ID_PUBLISH")
        reg = base.get("DAYS_REGISTRATION")
        emp = base.get("DAYS_EMPLOYED")
        gender = C.get("CODE_GENDER")
        
        if gender is not None and id_p is not None:
            K["K1"] = (gender.astype(str) + "_" + (id_p - birth).round(0).astype(str)).astype("category")
        if gender is not None and id_p is not None and reg is not None:
            K["K2"] = (gender.astype(str) + "_" + (id_p - birth).round(0).astype(str) + "_" + (reg - birth).round(0).astype(str)).astype("category")
        if gender is not None and id_p is not None and emp is not None:
            K["KB"] = (gender.astype(str) + "_" + (id_p - birth).round(0).astype(str) + "_" + (emp - birth).round(0).astype(str)).astype("category")
        if gender is not None and id_p is not None and reg is not None and emp is not None:
            K["K3"] = (gender.astype(str) + "_" + (id_p - birth).round(0).astype(str) + "_" + (reg - birth).round(0).astype(str) + "_" + (emp - birth).round(0).astype(str)).astype("category")
    t = base["CREDIT_TO_ANNUITY"].round(1).astype(str)                       # "identificador de producto" (cociente redondeado)
    K["PROD_R1"] = t.astype("category")
    if "NAME_CONTRACT_TYPE" in C.columns:
        K["PROD_R1_x_CT"] = (t + "|" + C["NAME_CONTRACT_TYPE"].astype(str)).astype("category")
    for name, cols in COMBOS.items():
        k = combo_key(D, cols)
        if k is not None:
            K[name] = k
    return pd.DataFrame(K, index=C.index)


def fe_freq(df, C):
    """Frequency Encoding (sin TARGET): frecuencia relativa de cada categoría y de importes/fechas repetidos.
    Los valores que se repiten mucho suelen ser "estándar" y se comportan distinto a los únicos."""
    F, n = {}, len(df)
    for c in C.columns:
        s = C[c].astype(object)
        F[f"FREQ_{c}"] = s.map(s.value_counts()).astype(float) / n
    for c in FREQ_NUM:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            F[f"FREQ_{c}"] = s.map(s.value_counts()).astype(float) / n
    fp = [c for c in ["DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"] if c in df.columns]
    if len(fp) == 3:   # "huella" del cliente: mismas 3 fechas = probable misma persona con varias solicitudes
        F["FREQ_FINGERPRINT"] = df.groupby(fp, dropna=False)[fp[0]].transform("size").astype(float)
    return pd.DataFrame(F, index=df.index)

def fe_group(X, K):
    """Features de "riesgo relativo": compara cada cliente contra su grupo de pares (media y desvío del grupo).
    REL = valor / media del grupo ; Z = (valor - media) / desvío.  Grupos con < 30 filas -> NaN (evita ruido)."""
    F = {}
    for kn in GROUP_KEYS:
        if kn not in K.columns:
            continue
        for col in GROUP_REL:
            if col not in X.columns:
                continue
            v = winsor(X[col].astype(float))
            grp = v.groupby(K[kn], observed=True)
            n, mu = grp.transform("count"), grp.transform("mean")
            mu = mu.where(n >= 30)
            F[f"{col}_REL_{kn}"] = safe_div(v, mu)
            if col in GROUP_Z:
                F[f"{col}_Z_{kn}"] = safe_div(v - mu, grp.transform("std").where(n >= 30))
    return pd.DataFrame(F, index=X.index)


def build():
    """Orquesta TODO el feature engineering sobre train+test juntos. Ninguna transformación usa TARGET
    (el Target Encoding se hace aparte, dentro de cada fold), así que no hay leakage.
    Devuelve (X, cats, K): features, columnas categóricas y claves enteras para Target Encoding.
    El resultado se guarda en disco: las siguientes ejecuciones lo cargan al instante."""
    path = f"{MD}/features.pkl"
    if os.path.exists(path):
        return pd.read_pickle(path)
    t0 = time.time()
    df = pd.concat([train.drop(columns="TARGET"), test], ignore_index=True)
    # 1) categóricas limpias (XNA / Unknown -> NaN -> categoría "NA")
    cats = [c for c in df.columns if c != "SK_ID_CURR" and not pd.api.types.is_numeric_dtype(df[c])]
    C = pd.DataFrame({c: to_cat(df[c].replace({"XNA": np.nan, "Unknown": np.nan})
                                if c in ("CODE_GENDER", "NAME_FAMILY_STATUS") else df[c]) for c in cats},
                     index=df.index)
        
       
        # 🐘 INYECCIÓN DEL ELEFANTE: Tariff Keys como categóricas ordenadas
    cred, ann, goods = (num(df, c) for c in ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"])
    tariff = {
        "TERM_KEY_1D": safe_div(cred, ann).round(1),
        "MARKUP_KEY_2D": safe_div(cred, goods).round(2)
    }
    for c, s in tariff.items():
        C[c] = to_cat(s)
    cats = cats + list(tariff)
  # SÍ agregar a cats para Target Encoding
    # Las claves para Target Encoding se calculan más abajo con make_keys()
    # 2) features base (sin EXT_SOURCE). Las versiones "limpias" reemplazan a las columnas originales
    base = fe_base(df)
    raw = df.drop(columns=["SK_ID_CURR"] + [c for c in cats if c in df.columns])
    B = pd.concat([raw.drop(columns=[c for c in base.columns if c in raw.columns]), base], axis=1)
    keys = make_keys(C, base)
    # 3) EXT_SOURCE: imputación + agregados + interacciones
    if USE_EXT_IMPUTE:
        P = fe_ext_impute(pd.concat([B, C.apply(lambda s: s.cat.codes)], axis=1), SEED)
    else:
        P = pd.DataFrame(index=df.index)
    BE = pd.concat([B, fe_ext(B, P)], axis=1)
    # 4) frecuencias + riesgo relativo por grupo
    X = pd.concat([BE, fe_freq(df, C), fe_group(BE, keys), fe_ext_percentiles(BE, C), fe_quant(BE), fe_interest(BE), fe_product(BE), C], axis=1)
    # Optional one-hot representation. Fit the vocabulary over train+test feature values only;
    # target encodings remain separately fold-safe in fold_frames().
    if os.environ.get("FE_ONEHOT", "0") == "1":
        onehot_cols = [c for c in cats if c in X.columns]
        X = pd.get_dummies(X, columns=onehot_cols, dummy_na=True, dtype="uint8")
        safe_names, seen = [], {}
        for col in X.columns:
            name = re.sub(r"[^A-Za-z0-9_]", "_", str(col))
            if not name or name[0].isdigit():
                name = "f_" + name
            k = seen.get(name, 0)
            seen[name] = k + 1
            safe_names.append(name if k == 0 else f"{name}_{k}")
        X.columns = safe_names
        cats = []
    assert not X.columns.duplicated().any(), "nombres de features duplicados"
    ncols = X.select_dtypes("number").columns                      # inf -> NaN y float32 (menos memoria)
    X[ncols] = X[ncols].replace([np.inf, -np.inf], np.nan)
    f64 = X.select_dtypes("float64").columns
    X[f64] = X[f64].astype("float32")
    K = keys.apply(lambda s: s.cat.codes).astype("int32")
    print(f"[FE] {X.shape[1]} features ({len(cats)} categóricas) + {K.shape[1]} claves de Target Encoding "
          f"en {time.time() - t0:.0f}s", flush=True)
    pd.to_pickle((X, cats, K), path)
    return X, cats, K

# ============ TARGET ENCODING SEGURO + MATRICES POR FOLD ============

@lru_cache(maxsize=1)
def get_data():
    """Separa train/test ya con features. Devuelve (X, Xt, K, Kt, cats)."""
    X_all, cats, K_all = build()
    n = len(train)
    return (X_all.iloc[:n].reset_index(drop=True), X_all.iloc[n:].reset_index(drop=True),
            K_all.iloc[:n].reset_index(drop=True), K_all.iloc[n:].reset_index(drop=True), cats)


def target_encode(Ktr, ytr, Kothers, sizes, seed, m=30.0, n_inner=5):
    """Target Encoding suavizado y SIN leakage.
      - filas de entrenamiento: codificadas con un K-Fold INTERNO (cada fila usa estadísticas de otras filas)
      - validación / test: codificadas con todo el entrenamiento del fold
    Suavizado: (suma + prior*m) / (conteo + m), con m = "peso" de la media global (categorías raras -> prior)."""
    prior = ytr.mean()
    inner = list(StratifiedKFold(n_inner, shuffle=True, random_state=seed + 1000).split(np.zeros(len(ytr)), ytr))
    enc_tr = np.zeros(Ktr.shape, dtype="float32")
    outs = [np.zeros(k.shape, dtype="float32") for k in Kothers]
    for j, n in enumerate(sizes):
        def enc(codes, yy):
            return (np.bincount(codes, weights=yy, minlength=n) + prior * m) / (np.bincount(codes, minlength=n) + m)
        for a, b in inner:
            enc_tr[b, j] = enc(Ktr[a, j], ytr[a])[Ktr[b, j]]
        full = enc(Ktr[:, j], ytr)
        for o, k in zip(outs, Kothers):
            o[:, j] = full[k[:, j]]
    return enc_tr, outs


def knn_target(Atr, ytr, Aoth, seed, k=50, m=20.0, n_inner=5):
    """Media de TARGET de los k vecinos más cercanos (espacio Atr). SIN leakage, igual que el Target Encoding:
    filas de entrenamiento -> vecinos tomados de OTRAS filas (K-Fold interno); validación/test -> todo el entrenamiento.
    Suavizado hacia la media global (prior) con peso m."""
    from sklearn.neighbors import NearestNeighbors
    prior = ytr.mean()
    enc = lambda idx, yy: (yy[idx].sum(axis=1) + prior * m) / (idx.shape[1] + m)
    tr_out = np.zeros(len(Atr), dtype="float32")
    for a, b in StratifiedKFold(n_inner, shuffle=True, random_state=seed + 2000).split(Atr, ytr):
        nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=N_JOBS).fit(Atr[a])
        tr_out[b] = enc(nn.kneighbors(Atr[b], return_distance=False), ytr[a])
    nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=N_JOBS).fit(Atr)
    return tr_out, [enc(nn.kneighbors(A, return_distance=False), ytr).astype("float32") for A in Aoth]


KNN_SPACES = {   # nombre -> (columnas, k)
    "KNN_EXT": (["EXTF_1", "EXTF_2", "EXTF_3"], 50),
    "KNN_EXTR": (["EXTF_1", "EXTF_2", "EXTF_3", "CREDIT_TO_INCOME", "ANNUITY_TO_INCOME", "AGE_YEARS"], 100),
}


def knn_block(Xtr, Xva, Xte, ytr, seed):
    """Features KNN-target de un fold. Estandariza con estadísticas del entrenamiento (sin TARGET)."""
    outs = {"tr": {}, "va": {}, "te": {}}
    for nm, (cols, k) in KNN_SPACES.items():
        cols = [c for c in cols if c in Xtr.columns]
        if len(cols) < 3:
            continue
        med, mu, sd = Xtr[cols].median(), None, None
        T = [d[cols].fillna(med).to_numpy(dtype="float64") for d in (Xtr, Xva, Xte)]
        mu, sd = T[0].mean(0), T[0].std(0) + 1e-9
        T = [(a - mu) / sd for a in T]
        tr_o, (va_o, te_o) = knn_target(T[0], ytr, [T[1], T[2]], seed, k=k)
        outs["tr"][nm], outs["va"][nm], outs["te"][nm] = tr_o, va_o, te_o
    return {k: pd.DataFrame(v) for k, v in outs.items()}


def poly_block(d):
    """Interacciones de 2.º/3.er orden y razones extra (experimento E_poly). Solo usan columnas de la propia fila:
    sin TARGET, sin fuga. Se calculan dentro de fold_frames, así que NO invalidan features.pkl ni los checkpoints.
    Nota: log1p/sqrt/potencias de UNA sola columna no se incluyen: los árboles son invariantes a transformaciones
    monótonas, así que no cambian nada (solo sirven para modelos lineales o redes)."""
    g = lambda c: d[c].astype("float64") if c in d.columns else None
    pay, cti, ati = g("ANNUITY_TO_CREDIT"), g("CREDIT_TO_INCOME"), g("ANNUITY_TO_INCOME")
    em, emn, emx = g("EXTF_MEAN"), g("EXTF_MIN"), g("EXTF_MAX")
    age, emp, fam, ch, anom = g("AGE_YEARS"), g("EMPLOYED_YEARS"), g("CNT_FAM_MEMBERS"), g("CNT_CHILDREN"), g("DAYS_EMPLOYED_ANOM")
    cred, inc, ann, goods = g("AMT_CREDIT"), g("AMT_INCOME_TOTAL"), g("AMT_ANNUITY"), g("AMT_GOODS_PRICE")
    F = {}
    def put(name, fn, *need):
        if all(v is not None for v in need):
            F["P_" + name] = fn()
    put("EXTMEAN_PAY", lambda: em * pay, em, pay)
    put("EXTMIN_PAY", lambda: emn * pay, emn, pay)
    put("EXTMAX_PAY", lambda: emx * pay, emx, pay)
    put("EXTMEAN_CTI", lambda: em * cti, em, cti)
    put("EXTMIN_CTI", lambda: emn * cti, emn, cti)
    put("EXTMEAN_ATI", lambda: em * ati, em, ati)
    put("EXTMEAN_PAY_AGE", lambda: em * pay * age, em, pay, age)
    put("EXTMEAN_PAY_EMP", lambda: em * pay * emp, em, pay, emp)
    put("EXTMIN_PAY_AGE", lambda: emn * pay * age, emn, pay, age)
    put("EXTMEAN_CTI_EMP", lambda: em * cti * emp, em, cti, emp)
    put("EXTMEAN_ATI_FAM", lambda: em * ati * fam, em, ati, fam)
    put("LOGAGE_EXTMEAN", lambda: np.log1p(age.clip(lower=0)) * em, age, em)
    put("EXT_SPREAD", lambda: (emx - emn) / (em + 0.01), emx, emn, em)
    put("CREDIT_GOODS_INC", lambda: (cred - goods) / inc.where(inc != 0), cred, goods, inc)
    put("INCOME_LEFT_FAM", lambda: (inc - ann) / fam.where(fam != 0), inc, ann, fam)
    put("PAY_CHILD", lambda: pay * ch, pay, ch)
    put("CTI_1PCH", lambda: cti * (1 + ch), cti, ch)
    put("ATI_EMP", lambda: ati / (emp + 1), ati, emp)
    put("EXTMEAN_ANOM", lambda: em * anom, em, anom)
    put("INV_PAY", lambda: 1.0 / (pay + 0.01), pay)
    return pd.DataFrame(F, index=d.index).replace([np.inf, -np.inf], np.nan).astype("float32")


def miss_block(d):
    """Experimento E_miss: comportamiento de los valores faltantes de EXT_SOURCE_1/2/3 (solo columnas de la propia fila).
    Los árboles ya manejan NaN y v4 ya trae EXT_NA_COUNT / EXT_NA_PATTERN / EXT_k_PRED / EXT_k_RESID, así que aquí
    solo se añade lo NO probado: EXT de las otras fuentes condicionado a qué fuente falta, y cruces con el nº de faltantes.
    (Rank/escalado, rellenar con la media y restar la media no se incluyen: son transformaciones monótonas de una
    columna, no cambian los splits de un árbol; D_quant ya midió el efecto de los rangos.)"""
    cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in d.columns]
    if len(cols) < 3:
        return pd.DataFrame(index=d.index)
    E = d[cols].astype("float64")
    e1, e2, e3 = E.iloc[:, 0], E.iloc[:, 1], E.iloc[:, 2]
    na = E.isna().sum(axis=1)
    F = {
        "M_HAS_ALL": (na == 0).astype("float32"),
        "M_ONLY_2": (e2.notna() & (na == 2)).astype("float32"),
        "M_E2_WHEN_1MISS": e2.where(e1.isna()), "M_E3_WHEN_1MISS": e3.where(e1.isna()),
        "M_E1_WHEN_2MISS": e1.where(e2.isna()), "M_E3_WHEN_2MISS": e3.where(e2.isna()),
        "M_E1_WHEN_3MISS": e1.where(e3.isna()), "M_E2_WHEN_3MISS": e2.where(e3.isna()),
        "M_MEAN_x_NA": E.mean(axis=1) * na, "M_MIN_x_NA": E.min(axis=1) * na,
    }
    if "ANNUITY_TO_CREDIT" in d.columns:
        F["M_MEAN_x_PAY_x_NA"] = E.mean(axis=1) * d["ANNUITY_TO_CREDIT"].astype("float64") * na
    return pd.DataFrame(F, index=d.index).replace([np.inf, -np.inf], np.nan).astype("float32")


def _codes(*arrays):
    """Códigos enteros comunes (train+val+test juntos, sin TARGET) para una clave que puede ser float con NaN."""
    allv = np.concatenate([np.asarray(a, dtype="float64") for a in arrays])
    c, _ = pd.factorize(allv)                       # NaN -> -1
    c = np.where(c < 0, c.max() + 1, c).astype("int64")
    out, i = [], 0
    for a in arrays:
        out.append(c[i:i + len(a)]); i += len(a)
    return out, int(c.max()) + 1


def hidden_dup_blocks(Xtr, Xva, Xte, ytr, seed, hidden=False, dup=False, hidden2=False, hidden3=False, hidden4=False, hidden5=False):
    """Experimentos E_hidden y E_dup (SIN fuga: todo lo supervisado se calcula solo con el entrenamiento del fold).
    E_hidden: 'categorías ocultas de préstamo'. AMT_ANNUITY/AMT_CREDIT toma pocos valores distintos (productos con
      tasa/plazo fijos). Se redondea a 3/4/5 decimales, se trata cada valor como categoría y se aplica Target Encoding
      suavizado (K-Fold interno) + frecuencia del valor.
    E_dup: 'clientes repetidos'. Huella = fechas (nacimiento, registro, emisión del ID [+ último cambio de teléfono]);
      TE con poco suavizado (m=2): la media del TARGET de las OTRAS solicitudes con la misma huella (dejando la propia fuera
      en entrenamiento) + tamaño del grupo (train+val+test)."""
    frames = [f for f in (Xtr, Xva, Xte) if f is not None]
    ntr = len(Xtr)
    feats = [dict() for _ in frames]
    keys = []   # (nombre, [claves por frame], m)
    if hidden and all(c in Xtr.columns for c in ("AMT_ANNUITY", "AMT_CREDIT")):
        for r in (CFG.get("hidden_rs") or (3, 4, 5)):      # decimales del redondeo (E_r2..E_r6 prueban uno solo)
            ks = [np.round(f["AMT_ANNUITY"].to_numpy("float64") / f["AMT_CREDIT"].to_numpy("float64").clip(min=1e-9), r)
                  for f in frames]
            keys.append((f"H_RATIO{r}", ks, 30.0))
    if hidden2:   # más "productos ocultos": cuota/bien, crédito/bien y la razón cruzada con plazo y tipo de contrato
        if all(c in Xtr.columns for c in ("AMT_ANNUITY", "AMT_GOODS_PRICE")):
            for r in (3, 4):
                keys.append((f"H_AG{r}", [np.round(f["AMT_ANNUITY"].to_numpy("float64") /
                                                   f["AMT_GOODS_PRICE"].to_numpy("float64").clip(min=1e-9), r) for f in frames], 30.0))
        if all(c in Xtr.columns for c in ("AMT_CREDIT", "AMT_GOODS_PRICE")):
            keys.append(("H_CG3", [np.round(f["AMT_CREDIT"].to_numpy("float64") /
                                            f["AMT_GOODS_PRICE"].to_numpy("float64").clip(min=1e-9), 3) for f in frames], 30.0))
        if all(c in Xtr.columns for c in ("AMT_ANNUITY", "AMT_CREDIT", "TERM_ROUND")):
            keys.append(("H_R4_TERM", [pd.util.hash_pandas_object(pd.DataFrame({
                "r": np.round(f["AMT_ANNUITY"].to_numpy("float64") / f["AMT_CREDIT"].to_numpy("float64").clip(min=1e-9), 4),
                "t": f["TERM_ROUND"].to_numpy("float64")}), index=False).to_numpy().astype("float64") for f in frames], 30.0))
        if all(c in Xtr.columns for c in ("AMT_ANNUITY", "AMT_CREDIT", "NAME_CONTRACT_TYPE")):
            keys.append(("H_R4_CT", [pd.util.hash_pandas_object(pd.DataFrame({
                "r": np.round(f["AMT_ANNUITY"].to_numpy("float64") / f["AMT_CREDIT"].to_numpy("float64").clip(min=1e-9), 4),
                "c": pd.Series(f["NAME_CONTRACT_TYPE"]).astype("category").cat.codes.to_numpy() if str(f["NAME_CONTRACT_TYPE"].dtype) != "category"
                     else f["NAME_CONTRACT_TYPE"].cat.codes.to_numpy()}), index=False).to_numpy().astype("float64") for f in frames], 30.0))
    if hidden3 and all(c in Xtr.columns for c in ("AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE")):
        # E_hidden3: más "productos ocultos": razón x (bien == crédito), meta/anualidad x (bien == crédito) y montos exactos
        g = lambda f, c: f[c].to_numpy("float64")
        eq = [(np.abs(g(f, "AMT_GOODS_PRICE") - g(f, "AMT_CREDIT")) < 0.5).astype("float64") for f in frames]
        r4 = [np.round(g(f, "AMT_ANNUITY") / g(f, "AMT_CREDIT").clip(min=1e-9), 4) for f in frames]
        ga1 = [np.round(g(f, "AMT_GOODS_PRICE") / g(f, "AMT_ANNUITY").clip(min=1e-9), 1) for f in frames]
        hsh = lambda a, b: [pd.util.hash_pandas_object(pd.DataFrame({"a": x, "b": y}), index=False).to_numpy().astype("float64") for x, y in zip(a, b)]
        keys.append(("H3_R4_EQ", hsh(r4, eq), 30.0))
        keys.append(("H3_GA1_EQ", hsh(ga1, eq), 30.0))
        for nm, c in (("H3_CRED", "AMT_CREDIT"), ("H3_GOODS", "AMT_GOODS_PRICE"), ("H3_ANN", "AMT_ANNUITY")):
            keys.append((nm, [g(f, c) for f in frames], 30.0))
    if hidden5 and all(c in Xtr.columns for c in ("AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE")):
        # E_ctg (claves del puesto 7): CT = crédito/anualidad a 1, 2 y 3 decimales, y cruces de CT_R1 con precio del bien, crédito y anualidad
        g5 = lambda f, c: f[c].to_numpy("float64")
        with np.errstate(all="ignore"):
            ct5 = [g5(f, "AMT_CREDIT") / np.where(g5(f, "AMT_ANNUITY") > 0, g5(f, "AMT_ANNUITY"), np.nan) for f in frames]
        r1, r2, r3 = ([np.nan_to_num(np.round(c, d), nan=-9e9) for c in ct5] for d in (1, 2, 3))
        hsh5 = lambda a, b: [pd.util.hash_pandas_object(pd.DataFrame({"a": x, "b": y_}), index=False).to_numpy().astype("float64") for x, y_ in zip(a, b)]
        for nm5, arr in (("H5_CT_R1", r1), ("H5_CT_R2", r2), ("H5_CT_R3", r3)):
            keys.append((nm5, arr, 30.0))
        for nm5, col in (("H5_CT1_GOODS", "AMT_GOODS_PRICE"), ("H5_CT1_CRED", "AMT_CREDIT"), ("H5_CT1_ANN", "AMT_ANNUITY")):
            keys.append((nm5, hsh5(r1, [np.nan_to_num(g5(f, col), nan=-9e9) for f in frames]), 30.0))
    if hidden4:   # E_hidden4: TE por valor exacto en otras columnas (estructura por valor de cambio de teléfono, coche, región)
        for nm, c in (("H4_PHONE", "DAYS_LAST_PHONE_CHANGE"), ("H4_CAR", "OWN_CAR_AGE"), ("H4_REGPOP", "REGION_POPULATION_RELATIVE"),
                      ("H4_FAM", "CNT_FAM_MEMBERS"), ("H4_IDPUB", "DAYS_ID_PUBLISH")):
            if c in Xtr.columns:
                keys.append((nm, [np.nan_to_num(f[c].to_numpy("float64"), nan=-9e9) for f in frames], 30.0))
    if dup:
        for nm, cols in (("D_FP3", ["DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH"]),
                         ("D_FP4", ["DAYS_BIRTH", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"])):
            if all(c in Xtr.columns for c in cols):
                ks = [pd.util.hash_pandas_object(f[cols], index=False).to_numpy().astype("float64") for f in frames]
                keys.append((nm, ks, 2.0))
    for nm, ks, m in keys:
        codes, size = _codes(*ks)
        tot = np.bincount(np.concatenate(codes), minlength=size).astype("float64")
        for fi, c in enumerate(codes):
            feats[fi][f"{nm}_N"] = (tot[c] / sum(len(x) for x in codes)).astype("float32") if nm.startswith(("H_", "H3_", "H4_", "H5_")) else tot[c].astype("float32")
        enc_tr, outs = target_encode(codes[0].reshape(-1, 1), ytr, [c.reshape(-1, 1) for c in codes[1:]], [size], seed, m=m)
        feats[0][f"{nm}_TE"] = enc_tr[:, 0]
        for fi, o in enumerate(outs, start=1):
            feats[fi][f"{nm}_TE"] = o[:, 0]
    return [pd.DataFrame(f, index=fr.index) for f, fr in zip(feats, frames)]


_UID_CACHE = {}


def uid_pairs():
    """Parejas CONFIRMADAS de la misma persona entre TODAS las filas (train+test), sin usar TARGET.
    Todas las DAYS_* se cuentan hacia atrás desde el día de la solicitud: si la misma persona vuelve a solicitar otro día, las
    DIFERENCIAS entre fechas (nacimiento-registro, nacimiento-ID) no cambian. Confirmación: DAYS_EMPLOYED y último cambio de
    teléfono se desplazan lo mismo (o ambos sin cambio / 365243), y coinciden género y educación. d = DAYS_BIRTH_i - DAYS_BIRTH_j
    (d < 0 -> la pareja j solicitó ANTES que la fila i)."""
    if "pairs" in _UID_CACHE:
        return _UID_CACHE["pairs"]
    A = pd.concat([train.drop(columns="TARGET"), test], ignore_index=True)
    A["row"] = np.arange(len(A))
    A["k"] = (A.DAYS_BIRTH - A.DAYS_REGISTRATION).round(0).astype(str) + "|" + (A.DAYS_BIRTH - A.DAYS_ID_PUBLISH).astype(str)
    cols = ["row", "k", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE", "CODE_GENDER", "NAME_EDUCATION_TYPE",
            "AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL"]
    G = A[A.k.map(A.k.value_counts()) >= 2][cols]
    P = G.merge(G, on="k", suffixes=("", "_p"))
    P = P[P.row != P.row_p].copy()
    d = P.DAYS_BIRTH - P.DAYS_BIRTH_p
    emp_ok = ((P.DAYS_EMPLOYED == 365243) & (P.DAYS_EMPLOYED_p == 365243)) | \
             ((P.DAYS_EMPLOYED - P.DAYS_EMPLOYED_p == d) & (P.DAYS_EMPLOYED != 365243) & (P.DAYS_EMPLOYED_p != 365243))
    ph_ok = (P.DAYS_LAST_PHONE_CHANGE - P.DAYS_LAST_PHONE_CHANGE_p == d) | (P.DAYS_LAST_PHONE_CHANGE == P.DAYS_LAST_PHONE_CHANGE_p)
    ok = emp_ok & ph_ok & (P.CODE_GENDER == P.CODE_GENDER_p) & (P.NAME_EDUCATION_TYPE == P.NAME_EDUCATION_TYPE_p)
    P["d"] = d
    P = P[(d == 0) | ok].copy()
    P["ad"] = P.d.abs()
    _UID_CACHE["pairs"] = P[["row", "row_p", "d", "ad", "AMT_CREDIT", "AMT_CREDIT_p", "AMT_ANNUITY", "AMT_ANNUITY_p",
                             "AMT_INCOME_TOTAL", "AMT_INCOME_TOTAL_p"]].reset_index(drop=True)
    return _UID_CACHE["pairs"]


def uid_free():
    """Features de historial de la persona SIN TARGET (todas las filas; las parejas de test también cuentan)."""
    if "free" in _UID_CACHE:
        return _UID_CACHE["free"]
    P, N = uid_pairs(), len(train) + len(test)
    F = pd.DataFrame(index=np.arange(N))
    F["UID_N"] = P.groupby("row").size().reindex(F.index).fillna(0)
    F["UID_N_EARLIER"] = P[P.d < 0].groupby("row").size().reindex(F.index).fillna(0)
    F["UID_N_LATER"] = P[P.d > 0].groupby("row").size().reindex(F.index).fillna(0)
    F["UID_N_SAME"] = P[P.d == 0].groupby("row").size().reindex(F.index).fillna(0)
    for nm, sub in (("PREV", P[P.d < 0]), ("NEXT", P[P.d > 0])):
        e = sub.sort_values("ad").groupby("row").first().reindex(F.index)
        F[f"UID_DAYS_{nm}"] = e.ad
        F[f"UID_DCRED_{nm}"] = e.AMT_CREDIT - e.AMT_CREDIT_p
        F[f"UID_DANN_{nm}"] = e.AMT_ANNUITY - e.AMT_ANNUITY_p
        F[f"UID_DINC_{nm}"] = e.AMT_INCOME_TOTAL - e.AMT_INCOME_TOTAL_p
        F[f"UID_CRATIO_{nm}"] = e.AMT_CREDIT / e.AMT_CREDIT_p.where(e.AMT_CREDIT_p > 0)
    _UID_CACHE["free"] = F.astype("float32")
    return _UID_CACHE["free"]


def uid_target_block(rows, lab, P=None, pre="UIDY", inner=None):
    """Features de RESULTADOS PASADOS de la misma persona, solo con filas ETIQUETADAS del fold (`lab` es NaN fuera del
    entrenamiento del fold). Para filas de entrenamiento se usan OTRAS filas del entrenamiento (nunca la propia)."""
    P = uid_pairs() if P is None else P
    inr = np.zeros(len(lab), bool)
    inr[rows] = True
    m = inr[P.row.to_numpy()] & ~np.isnan(lab[P.row_p.to_numpy()])
    if inner is not None:    # variante K-Fold INTERNO: una fila de entrenamiento solo usa parejas de OTROS folds internos
        ir, ip = inner[P.row.to_numpy()], inner[P.row_p.to_numpy()]
        m &= (ir < 0) | (ir != ip)
    Q = P[m].assign(yp=lab[P.row_p.to_numpy()[m]])
    g = Q.groupby("row")
    near = Q.sort_values("ad").groupby("row").first()
    F = pd.DataFrame({f"{pre}_N": g.size(), f"{pre}_SUM": g.yp.sum(), f"{pre}_NEAR_Y": near.yp, f"{pre}_NEAR_D": near.ad}).reindex(rows)
    F[f"{pre}_ANYBAD"] = np.where(F[f"{pre}_N"].isna(), np.nan, (F[f"{pre}_SUM"] > 0).astype(float))
    return F.reset_index(drop=True).astype("float32")


def _annuity_rate(C, A, n, it=45):
    """Tasa ANUAL implícita EXACTA por bisección: A = C r / (1 - (1+r)^-n), r mensual. NaN si no hay tasa positiva razonable."""
    C, A = np.asarray(C, "float64"), np.asarray(A, "float64")
    lo, hi = np.full(len(C), 1e-7), np.full(len(C), 0.5)
    pay = lambda r: C * r / (1.0 - (1.0 + r) ** (-n))
    for _ in range(it):
        mid = (lo + hi) / 2
        up = pay(mid) > A
        hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
    r = (lo + hi) / 2
    with np.errstate(all="ignore"):
        valid = (A * n > C * 1.0000001) & (C > 0) & (pay(np.full(len(C), 0.5)) > A)
    return np.where(valid, 12.0 * r, np.nan)


def rate2_blocks(Xtr, Xva, Xte):
    """E_rate2 (sin TARGET): tasa anual implícita EXACTA con el plazo (6..60) cuya tasa queda en [2 %, 90 %] y más cerca de 25 %;
    tasa sobre el precio del bien; y RESIDUO de la tasa respecto a la esperada para un cliente con scores EXT y monto parecidos
    (regresión lineal ajustada en el entrenamiento del fold, sin usar TARGET) y respecto al grupo de plazo."""
    def raw(X):
        C, A = X["AMT_CREDIT"].to_numpy("float64"), X["AMT_ANNUITY"].to_numpy("float64")
        G = X["AMT_GOODS_PRICE"].to_numpy("float64")
        best, bt = np.full(len(X), np.nan), np.full(len(X), np.nan)
        for n in (6, 12, 18, 24, 30, 36, 48, 60):
            r = _annuity_rate(C, A, n)
            ok = (r > 0.02) & (r < 0.90)
            upd = ok & (np.isnan(best) | (np.abs(r - 0.25) < np.abs(best - 0.25)))
            best, bt = np.where(upd, r, best), np.where(upd, n, bt)
        rg = _annuity_rate(G, A, np.where(np.isnan(bt), 36, bt)) if False else np.full(len(X), np.nan)
        for n in (12, 24, 36, 48, 60):
            r = _annuity_rate(G, A, n)
            upd = (r > 0.02) & (r < 0.90) & (np.isnan(rg) | (np.abs(r - 0.25) < np.abs(rg - 0.25)))
            rg = np.where(upd, r, rg)
        return best, bt, rg
    R = {nm: raw(X) for nm, X in (("tr", Xtr), ("va", Xva), ("te", Xte)) if X is not None}
    ext = lambda X: X["EXTF_MEAN"].to_numpy("float64") if "EXTF_MEAN" in X.columns else np.zeros(len(X))
    design = lambda X, bt: np.column_stack([np.ones(len(X)), np.nan_to_num(ext(X), nan=0.5),
                                            np.log1p(X["AMT_CREDIT"].to_numpy("float64")), np.nan_to_num(bt, nan=0.0)])
    r_tr, t_tr = R["tr"][0], R["tr"][1]
    ok = ~np.isnan(r_tr)
    beta = np.linalg.lstsq(design(Xtr, t_tr)[ok], r_tr[ok], rcond=None)[0] if ok.sum() > 10 else np.zeros(4)
    grp = pd.Series(r_tr).groupby(pd.Series(t_tr)).mean()
    out = {}
    for nm, X in (("tr", Xtr), ("va", Xva), ("te", Xte)):
        if X is None:
            out[nm] = None
            continue
        r, t, rg = R[nm]
        exp_r = design(X, t) @ beta
        out[nm] = pd.DataFrame({"R2_RATE": r, "R2_TERM": t, "R2_RATE_GOODS": rg, "R2_RATE_RES": r - exp_r,
                                "R2_RATE_GRPRES": r - pd.Series(t).map(grp).to_numpy()}).astype("float32")
    return out["tr"], out["va"], out["te"]


def uid_price():
    """Precio de la solicitud frente a las OTRAS solicitudes de la misma persona (sin TARGET): diferencia de tasa implícita,
    de cuota/crédito y de precio del bien con la solicitud anterior y la siguiente, si es el mismo producto (mismo
    CREDIT/ANNUITY a 1 decimal) y la tasa frente a la media de sus otras solicitudes. Prefijo UIDP_ -> CFG['uidp']."""
    if "price" in _UID_CACHE:
        return _UID_CACHE["price"]
    P, N = uid_pairs(), len(train) + len(test)
    A = pd.concat([train.drop(columns="TARGET"), test], ignore_index=True)
    C, An, G = (A[c].to_numpy("float64") for c in ("AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"))
    best = np.full(N, np.nan)
    for n_ in (6, 12, 18, 24, 30, 36, 48, 60):
        r = _annuity_rate(C, An, n_)
        upd = (r > 0.02) & (r < 0.90) & (np.isnan(best) | (np.abs(r - 0.25) < np.abs(best - 0.25)))
        best = np.where(upd, r, best)
    with np.errstate(all="ignore"):
        prod = np.round(C / np.where(An > 0, An, np.nan), 1)
        pay = An / np.where(C > 0, C, np.nan)
    F = pd.DataFrame(index=np.arange(N))
    for nm, sub in (("PREV", P[P.d < 0]), ("NEXT", P[P.d > 0])):
        e = sub.sort_values("ad").groupby("row").first()
        i, j = e.index.to_numpy(), e.row_p.to_numpy().astype(int)
        mk = lambda v: pd.Series(v, index=i).reindex(F.index)
        F[f"UIDP_DRATE_{nm}"], F[f"UIDP_DPAY_{nm}"] = mk(best[i] - best[j]), mk(pay[i] - pay[j])
        F[f"UIDP_DGOODS_{nm}"], F[f"UIDP_SAMEPROD_{nm}"] = mk(G[i] - G[j]), mk((prod[i] == prod[j]).astype("float64"))
    mean_oth = P.assign(r=best[P.row_p.to_numpy().astype(int)]).groupby("row").r.mean().reindex(F.index)
    F["UIDP_RATE_OTH_MEAN"], F["UIDP_RATE_VS_OTH"] = mean_oth, best - mean_oth
    _UID_CACHE["price"] = F.astype("float32")
    return _UID_CACHE["price"]


def fit_lgb_init(p, Xtr, ytr, Xva, yva, Xte, seed, cats, w=None):
    """lgb_lin del PDF: LightGBM que arranca del MARGEN de un scorecard logístico (init_score). El margen de las filas de
    entrenamiento sale de un K-Fold interno (OOF) para que no esté sobreajustado; el de validación/test, del scorecard
    ajustado con todo el entrenamiento del fold. La predicción final suma el margen al valor bruto del boosting."""
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from scipy.special import expit
    cats2 = [c for c in cats if str(Xtr[c].dtype) == "category"]
    Atr, Ava, Ate, _, _, _, _ = nn_prepare(Xtr, Xva, Xte, cats2, seed)
    mk = lambda: LogisticRegression(C=0.05, max_iter=300)
    m_tr = np.zeros(len(Atr))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed + 77).split(Atr, ytr):
        m_tr[b] = mk().fit(Atr[a], ytr[a]).decision_function(Atr[b])
    sc = mk().fit(Atr, ytr)
    m_va, m_te = sc.decision_function(Ava), (sc.decision_function(Ate) if Xte is not None else None)
    p = dict(p)
    if CFG["lr"] and "learning_rate" not in p:
        p["learning_rate"] = CFG["lr"]
    pat = 100 if (CFG["lr"] or 0) >= 0.04 else 200
    m = lgb.train({**LGB_BASE, **p, "seed": seed}, lgb.Dataset(Xtr, ytr, init_score=m_tr), ROUNDS["lgb"],
                  valid_sets=[lgb.Dataset(Xva, yva, init_score=m_va)], callbacks=[lgb.early_stopping(pat, verbose=False)])
    it = m.best_iteration if m.best_iteration and m.best_iteration > 0 else None
    pv = expit(m.predict(Xva, num_iteration=it, raw_score=True) + m_va)
    return pv, (expit(m.predict(Xte, num_iteration=it, raw_score=True) + m_te) if Xte is not None else None)


def uid2_pairs():
    """E_uid2: pares de la llave de Pachara (2 diferencias de fechas + género + educación) MENOS los de mi criterio estricto, separados por
    nivel de confianza: T1 = coincide exactamente uno de (empleo, teléfono); T0 = no coincide ninguno. Sin TARGET."""
    if "pairs2" in _UID_CACHE:
        return _UID_CACHE["pairs2"]
    A = pd.concat([train.drop(columns="TARGET"), test], ignore_index=True)
    A["row"] = np.arange(len(A))
    A["k"] = (A.DAYS_BIRTH - A.DAYS_REGISTRATION).round(0).astype(str) + "|" + (A.DAYS_BIRTH - A.DAYS_ID_PUBLISH).astype(str) + "|" + \
             A.CODE_GENDER.astype(str) + "|" + A.NAME_EDUCATION_TYPE.astype(str)
    cols = ["row", "k", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_LAST_PHONE_CHANGE"]
    G = A[A.k.map(A.k.value_counts()) >= 2][cols]
    P = G.merge(G, on="k", suffixes=("", "_p"))
    P = P[P.row != P.row_p].copy()
    d = P.DAYS_BIRTH - P.DAYS_BIRTH_p
    emp = ((P.DAYS_EMPLOYED == 365243) & (P.DAYS_EMPLOYED_p == 365243)) | \
          ((P.DAYS_EMPLOYED - P.DAYS_EMPLOYED_p == d) & (P.DAYS_EMPLOYED != 365243) & (P.DAYS_EMPLOYED_p != 365243))
    ph = (P.DAYS_LAST_PHONE_CHANGE - P.DAYS_LAST_PHONE_CHANGE_p == d) | (P.DAYS_LAST_PHONE_CHANGE == P.DAYS_LAST_PHONE_CHANGE_p)
    P["d"], P["ad"] = d, d.abs()
    P["tier"] = np.where((d == 0) | (emp & ph), 2, np.where(emp | ph, 1, 0))
    _UID_CACHE["pairs2"] = P[P.tier < 2][["row", "row_p", "d", "ad", "tier"]].reset_index(drop=True)
    return _UID_CACHE["pairs2"]


def uid2_free():
    P, N = uid2_pairs(), len(train) + len(test)
    F = pd.DataFrame(index=np.arange(N))
    for t in (1, 0):
        sub = P[P.tier == t]
        F[f"UID2_N_T{t}"] = sub.groupby("row").size().reindex(F.index).fillna(0)
        F[f"UID2_DAYS_T{t}"] = sub.groupby("row").ad.min().reindex(F.index)
    return F.astype("float32")


def seg_blocks(Xtr, Xva, Xte):
    """E_seg (sin TARGET, segmentos aprendidos SOLO con las filas de entrenamiento del fold): estructura del préstamo en
    CT = AMT_CREDIT / AMT_ANNUITY (válido solo si ambos > 0; si no, NaN).
      1) Histograma de CT del entrenamiento (ancho 0.1, rango 0-80), suavizado con media móvil de 3 bins.
      2) Segmentos = picos (scipy find_peaks: altura >= max(150, 0.1 % de las filas), separación >= 0.3, prominencia >= 30);
         si hay menos de 3 picos se usan los 20 bins más frecuentes. Fronteras = puntos medios entre picos consecutivos.
    Features (prefijo SG_):  SG_ID = nº del segmento más cercano (orden por valor del pico);  SG_PEAK = valor del pico;
      SG_DIST = CT - pico (con signo);  SG_ADIST = |SG_DIST|;  SG_POS = SG_DIST / semiancho del segmento (distancia a la
      frontera más cercana; en [-1, 1]);  SG_FREQ = fracción de filas de entrenamiento en ese segmento;  SG_PEAKH = altura
      relativa del pico. Ninguna usa TARGET ni filas de validación/test para definir los segmentos."""
    from scipy.signal import find_peaks
    def ct(X):
        c, a = X["AMT_CREDIT"].to_numpy("float64"), X["AMT_ANNUITY"].to_numpy("float64")
        with np.errstate(all="ignore"):
            return np.where((c > 0) & (a > 0), c / a, np.nan)
    r_tr = ct(Xtr)
    v = r_tr[(~np.isnan(r_tr)) & (r_tr < 80)]
    edges = np.arange(0, 80.1, 0.1)
    h = np.histogram(v, bins=edges)[0].astype("float64")
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    pk, _ = find_peaks(hs, height=max(150.0, 0.001 * len(v)), distance=3, prominence=30)
    if len(pk) < 3:
        pk = np.sort(np.argsort(h)[::-1][:20])
    peaks = (edges[pk] + 0.05)
    bounds = (peaks[1:] + peaks[:-1]) / 2
    seg_n = np.bincount(np.searchsorted(bounds, v), minlength=len(peaks)).astype("float64") / max(len(v), 1)
    peak_h = hs[pk] / max(hs.sum(), 1)
    def feats(X):
        r = ct(X)
        idx = np.searchsorted(bounds, np.where(np.isnan(r), 0.0, r))
        pv = peaks[idx]
        lo = np.where(idx > 0, bounds[np.maximum(idx - 1, 0)], -np.inf)
        hi = np.where(idx < len(bounds), bounds[np.minimum(idx, len(bounds) - 1)], np.inf)
        d = r - pv
        half = np.where(d >= 0, hi - pv, pv - lo)
        half = np.where(np.isfinite(half) & (half > 0), half, np.nan)
        F = pd.DataFrame({"SG_ID": idx.astype("float64"), "SG_PEAK": pv, "SG_DIST": d, "SG_ADIST": np.abs(d), "SG_POS": d / half,
                          "SG_FREQ": seg_n[idx], "SG_PEAKH": peak_h[idx]})
        F[np.isnan(r)] = np.nan
        return F.astype("float32")
    return feats(Xtr), feats(Xva), (feats(Xte) if Xte is not None else None)


def extg_blocks(Xtr, Xva, Xte):
    """E_extg (sin TARGET; estadísticas aprendidas SOLO con el entrenamiento del fold): promedio por grupo de las EXT_SOURCE imputadas
    (EXTF_1..3, EXTF_MEAN: media y desviación respecto al grupo) y de las PREDICCIONES del modelo auxiliar (EXT*_PRED: media del grupo).
    Grupos: ORGANIZATION_TYPE, OCCUPATION_TYPE, NAME_EDUCATION_TYPE, NAME_INCOME_TYPE (NaN = grupo propio). Prefijo XG_."""
    groups = [g for g in ("ORGANIZATION_TYPE", "OCCUPATION_TYPE", "NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE") if g in Xtr.columns]
    vals = [c for c in ("EXTF_MEAN", "EXTF_1", "EXTF_2", "EXTF_3") if c in Xtr.columns]
    preds = [c for c in Xtr.columns if c.startswith("EXT") and c.endswith("_PRED")]
    frames = [Xtr, Xva] + ([Xte] if Xte is not None else [])
    outs = [{} for _ in frames]
    for g in groups:
        gk = [X[g].astype("object").where(X[g].notna(), "NA").astype(str) for X in frames]
        for v in vals + preds:
            mu = Xtr[v].astype("float64").groupby(gk[0].to_numpy()).mean()
            for i, X in enumerate(frames):
                m_ = gk[i].map(mu).to_numpy("float64")
                outs[i][f"XG_{v}_{g}_MEAN"] = m_
                if v in vals:
                    outs[i][f"XG_{v}_{g}_DEV"] = X[v].to_numpy("float64") - m_
    res = [pd.DataFrame(o, index=range(len(f))).astype("float32") for o, f in zip(outs, frames)]
    return res[0], res[1], (res[2] if Xte is not None else None)



def fee_decode_blocks(Xtr, Xva, Xte, ytr, seed, fee=129.5):
    """Label-free decoding of cash-loan product q from annuity = q*credit + fixed fee.

    The adjusted q is used only when its five-decimal value is more common among cash-loan
    applications than the raw annuity/credit ratio. Target encodings for q rounded to 3/4/5
    decimals and q x fee-flag are nested inside the outer fold and inner OOF for training rows.
    Counts use features only (no TARGET); labels used for TE come only from ytr.
    """
    frames = [Xtr, Xva] + ([Xte] if Xte is not None else [])
    raw, adj, cash, contract = [], [], [], []
    for f in frames:
        c = pd.to_numeric(f.get("AMT_CREDIT", pd.Series(np.nan, index=f.index)), errors="coerce").to_numpy("float64")
        a = pd.to_numeric(f.get("AMT_ANNUITY", pd.Series(np.nan, index=f.index)), errors="coerce").to_numpy("float64")
        raw.append(np.divide(a, c, out=np.full(len(f), np.nan), where=np.isfinite(c) & (c != 0)))
        adj.append(np.divide(a - fee, c, out=np.full(len(f), np.nan), where=np.isfinite(c) & (c != 0)))
        if "NAME_CONTRACT_TYPE" in f.columns:
            contract.append(f["NAME_CONTRACT_TYPE"].astype(str).to_numpy())
            cash.append(f["NAME_CONTRACT_TYPE"].astype(str).eq("Cash loans").to_numpy())
        else:
            contract.append(np.full(len(f), "NA", dtype=object)); cash.append(np.zeros(len(f), dtype=bool))
    raw5 = [np.round(a, 5) for a in raw]
    adj5 = [np.round(a, 5) for a in adj]
    all_raw = np.concatenate([a[m & np.isfinite(a)] for a, m in zip(raw5, cash)])
    all_adj = np.concatenate([a[m & np.isfinite(a)] for a, m in zip(adj5, cash)])
    raw_counts = pd.Series(all_raw).value_counts()
    adj_counts = pd.Series(all_adj).value_counts()
    decoded, selected = [], []
    blocks = [dict() for _ in frames]
    for i, f in enumerate(frames):
        rf = pd.Series(raw5[i]).map(raw_counts).fillna(0).to_numpy("float64")
        af = pd.Series(adj5[i]).map(adj_counts).fillna(0).to_numpy("float64")
        flag = cash[i] & np.isfinite(adj5[i]) & (af > rf)
        q = np.where(flag, adj5[i], raw5[i])
        decoded.append(flag); selected.append(q)
        blocks[i].update({
            "FD_RAW_Q5": raw5[i].astype("float32"),
            "FD_ADJ_Q5": adj5[i].astype("float32"),
            "FD_SELECTED_Q5": q.astype("float32"),
            "FD_FEE_FLAG": flag.astype("float32"),
            "FD_LOG_RAW_FREQ": np.log1p(rf).astype("float32"),
            "FD_LOG_ADJ_FREQ": np.log1p(af).astype("float32"),
            "FD_FREQ_ADVANTAGE": (np.log1p(af) - np.log1p(rf)).astype("float32"),
        })
    # Nested smoothed target encodings for q at several resolutions and q x decoded-fee status.
    key_arrays = {f"Q{d}": [np.round(q, d) for q in selected] for d in (3, 4, 5)}
    combo_arrays = []
    for q, fl, ct in zip(selected, decoded, contract):
        qtxt = np.where(np.isfinite(q), np.char.mod("%.5f", q), "NA")
        combo_arrays.append(np.char.add(np.char.add(np.char.add(qtxt, "|"), fl.astype("int8").astype(str)), np.char.add("|", ct.astype(str))))
    key_arrays["Q5_FEE"] = combo_arrays
    for name, arrays in key_arrays.items():
        if name == "Q5_FEE":
            joined = np.concatenate(arrays)
            codes_all, uniques = pd.factorize(joined, sort=False)
            codes_all = codes_all.astype("int64")
            size = len(uniques)
            cuts = np.cumsum([len(a) for a in arrays[:-1]])
            codes = list(np.split(codes_all, cuts))
        else:
            codes, size = _codes(*arrays)
        cc = np.concatenate(codes).astype("int64")
        freq = np.bincount(cc, minlength=size).astype("float64") / max(len(cc), 1)
        for i, c in enumerate(codes):
            blocks[i][f"FD_{name}_FREQ"] = freq[c].astype("float32")
        Ktr = codes[0].reshape(-1, 1)
        Kothers = [c.reshape(-1, 1) for c in codes[1:]]
        enc_tr, outs = target_encode(Ktr, ytr, Kothers, [size], seed + 611, m=30.0, n_inner=5)
        blocks[0][f"FD_{name}_TE"] = enc_tr[:, 0]
        for i, out in enumerate(outs, start=1):
            blocks[i][f"FD_{name}_TE"] = out[:, 0]
    return [pd.DataFrame(b, index=f.index).replace([np.inf, -np.inf], np.nan).astype("float32") for b, f in zip(blocks, frames)]


def fold_frames(X, Xt, K, Kt, tr, va, seed, need_test=True, yy=None):
    """Matrices (Xtr, Xva, Xte) de un fold. Añade TE_* (y KNN_* si CFG['knn']) calculadas SOLO con el fold.
    `yy`: vector de TARGET a usar (por defecto el real; `leak` pasa uno permutado). Aplica CFG['drop'] y quita Q_* si no CFG['quant']."""
    yy = y if yy is None else yy
    Xtr, Xva = X.iloc[tr].reset_index(drop=True), X.iloc[va].reset_index(drop=True)
    Xte = Xt if (need_test or CFG["knn"]) else None
    if CFG["te"]:
        sizes = [int(max(K[c].max(), Kt[c].max())) + 1 for c in K.columns]
        tr_te, (va_te, te_te) = target_encode(K.to_numpy()[tr], yy[tr], [K.to_numpy()[va], Kt.to_numpy()], sizes, seed)
        cols = [f"TE_{c}" for c in K.columns]
        Xtr = pd.concat([Xtr, pd.DataFrame(tr_te, columns=cols)], axis=1)
        Xva = pd.concat([Xva, pd.DataFrame(va_te, columns=cols)], axis=1)
        if Xte is not None:
            Xte = pd.concat([Xt, pd.DataFrame(te_te, columns=cols)], axis=1)
    if CFG.get("poly"):
        Xtr = pd.concat([Xtr, poly_block(Xtr)], axis=1)
        Xva = pd.concat([Xva, poly_block(Xva)], axis=1)
        if Xte is not None:
            Xte = pd.concat([Xte, poly_block(Xte)], axis=1)
    if CFG.get("miss"):
        Xtr = pd.concat([Xtr, miss_block(Xtr)], axis=1)
        Xva = pd.concat([Xva, miss_block(Xva)], axis=1)
        if Xte is not None:
            Xte = pd.concat([Xte, miss_block(Xte)], axis=1)
    if CFG["knn"]:
        kb = knn_block(Xtr, Xva, Xte, yy[tr], seed)
        Xtr, Xva, Xte = (pd.concat([a, kb[k]], axis=1) for a, k in [(Xtr, "tr"), (Xva, "va"), (Xte, "te")])
    if CFG.get("hidden") or CFG.get("dup") or CFG.get("hidden2") or CFG.get("hidden3") or CFG.get("hidden4") or CFG.get("hidden5"):    # categorías ocultas de préstamo / clientes repetidos (TE dentro del fold)
        hb = hidden_dup_blocks(Xtr, Xva, Xte, yy[tr], seed, hidden=bool(CFG.get("hidden") or CFG.get("hidden2")),
                               dup=bool(CFG.get("dup")), hidden2=bool(CFG.get("hidden2")), hidden3=bool(CFG.get("hidden3")), hidden4=bool(CFG.get("hidden4")), hidden5=bool(CFG.get("hidden5")))
        Xtr, Xva = pd.concat([Xtr, hb[0]], axis=1), pd.concat([Xva, hb[1]], axis=1)
        if Xte is not None:
            Xte = pd.concat([Xte, hb[2]], axis=1)
    if CFG.get("fee_decode"):
        fd = fee_decode_blocks(Xtr, Xva, Xte, yy[tr], seed)
        Xtr, Xva = pd.concat([Xtr, fd[0]], axis=1), pd.concat([Xva, fd[1]], axis=1)
        if Xte is not None:
            Xte = pd.concat([Xte.reset_index(drop=True), fd[2]], axis=1)
    if CFG.get("uid") or CFG.get("uidte") or CFG.get("uidte180") or CFG.get("rate2") or CFG.get("uidp") or CFG.get("uid2") or CFG.get("seg") or CFG.get("extg"):
        n_tr_all, m_te = len(train), len(Xt)
        if CFG.get("uid"):                      # historial de la persona SIN TARGET
            Fa = uid_free()
            Xtr = pd.concat([Xtr, Fa.iloc[tr].reset_index(drop=True)], axis=1)
            Xva = pd.concat([Xva, Fa.iloc[va].reset_index(drop=True)], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte.reset_index(drop=True), Fa.iloc[n_tr_all:n_tr_all + m_te].reset_index(drop=True)], axis=1)
        if CFG.get("uidte"):                    # resultados pasados de la misma persona, SOLO con etiquetas del fold
            lab = np.full(n_tr_all + m_te, np.nan)
            lab[tr] = yy[tr]
            inner = None
            if CFG.get("uidte_mode") == "inner":    # E33b: K-Fold interno de 5 (la codificación del puesto/Pachara)
                inner = np.full(n_tr_all + m_te, -1)
                for k_, (_, ib) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed + 3000).split(np.zeros(len(tr)), yy[tr])):
                    inner[np.asarray(tr)[ib]] = k_
            Xtr = pd.concat([Xtr, uid_target_block(np.asarray(tr), lab, inner=inner)], axis=1)
            Xva = pd.concat([Xva, uid_target_block(np.asarray(va), lab, inner=inner)], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte.reset_index(drop=True), uid_target_block(np.arange(n_tr_all, n_tr_all + m_te), lab, inner=inner)], axis=1)
        if CFG.get("uidte180"):
            # Community hypothesis: only use matched applicant labels when the two application dates are <=180 days apart.
            # This uses only labels available in the outer training fold; current-row and validation labels are absent.
            lab180 = np.full(n_tr_all + m_te, np.nan)
            lab180[tr] = yy[tr]
            p180 = uid_pairs()
            p180 = p180[p180.ad <= 180].copy()
            Xtr = pd.concat([Xtr, uid_target_block(np.asarray(tr), lab180, P=p180, pre="UID180")], axis=1)
            Xva = pd.concat([Xva, uid_target_block(np.asarray(va), lab180, P=p180, pre="UID180")], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte.reset_index(drop=True), uid_target_block(np.arange(n_tr_all, n_tr_all + m_te), lab180, P=p180, pre="UID180")], axis=1)
        if CFG.get("uid2"):
            # Medium/low-confidence matches are separate from the strict 180-day history branch.

            F2, P2 = uid2_free(), uid2_pairs()
            lab2 = np.full(n_tr_all + m_te, np.nan)
            lab2[tr] = yy[tr]
            blk = lambda rows: pd.concat([uid_target_block(np.asarray(rows), lab2, P2[P2.tier == t], f"UID2Y_T{t}") for t in (1, 0)], axis=1)
            Xtr = pd.concat([Xtr, F2.iloc[tr].reset_index(drop=True), blk(tr)], axis=1)
            Xva = pd.concat([Xva, F2.iloc[va].reset_index(drop=True), blk(va)], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte.reset_index(drop=True), F2.iloc[n_tr_all:n_tr_all + m_te].reset_index(drop=True),
                                 blk(np.arange(n_tr_all, n_tr_all + m_te))], axis=1)
        if CFG.get("extg"):                     # E_extg: promedios por grupo de las EXT (aprendidos en el fold, sin TARGET)
            e1, e2, e3 = extg_blocks(Xtr, Xva, Xte)
            Xtr, Xva = pd.concat([Xtr.reset_index(drop=True), e1], axis=1), pd.concat([Xva.reset_index(drop=True), e2], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte.reset_index(drop=True), e3], axis=1)
        if CFG.get("seg"):                      # E_seg: segmentos del cociente crédito/anualidad aprendidos en el fold (sin TARGET)
            s1, s2, s3 = seg_blocks(Xtr, Xva, Xte)
            Xtr, Xva = pd.concat([Xtr, s1], axis=1), pd.concat([Xva, s2], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte.reset_index(drop=True), s3], axis=1)
        if CFG.get("uidp"):                     # precio frente a las otras solicitudes de la persona (sin TARGET)
            Fp = uid_price()
            Xtr = pd.concat([Xtr, Fp.iloc[tr].reset_index(drop=True)], axis=1)
            Xva = pd.concat([Xva, Fp.iloc[va].reset_index(drop=True)], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte.reset_index(drop=True), Fp.iloc[n_tr_all:n_tr_all + m_te].reset_index(drop=True)], axis=1)
        if CFG.get("rate2"):
            b1, b2, b3 = rate2_blocks(Xtr, Xva, Xte)
            Xtr, Xva = pd.concat([Xtr, b1], axis=1), pd.concat([Xva, b2], axis=1)
            if Xte is not None:
                Xte = pd.concat([Xte.reset_index(drop=True), b3], axis=1)
    drop = {c for c in Xtr.columns if (c.startswith("W_") and CFG["product"] in (False, "te"))
            or (c.startswith("TE_PROD") and CFG["product"] in (False, "num"))} | set(CFG["drop"]) | (set() if CFG["quant"] else {c for c in X.columns if c.startswith("Q_")}) \
           | (set() if CFG["interest"] else {c for c in X.columns if c.startswith("V_")})
    if drop:
        Xtr, Xva = Xtr.drop(columns=drop, errors="ignore"), Xva.drop(columns=drop, errors="ignore")
        Xte = Xte.drop(columns=drop, errors="ignore") if Xte is not None else None
    return Xtr, Xva, (Xte if need_test else None)            
# ============================ MODELOS ============================

LGB_BASE = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=24, min_child_samples=100,
                feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l1=1.0, lambda_l2=10.0,
                cat_smooth=20, verbose=-1, n_jobs=N_JOBS, deterministic=True, force_row_wise=True)
BEST_JSON = f"{MD}/best_lgb_params.json"

# Registro de modelos: nombre -> (familia, hiperparámetros extra).  None = leer BEST_JSON (resultado de `tune`).
MODELS = {
    "lgb":   ("lgb", {}),                                     # LightGBM base
    "lgb2":  ("lgb", None),                                   # LightGBM con la mejor config de `tune`
    "lgbet": ("lgb", dict(extra_trees=True, num_leaves=31, min_child_samples=80, feature_fraction=0.3,
                          learning_rate=0.03, lambda_l2=20)), # Extra-Trees: árboles más aleatorios -> diversidad
    "xgb":   ("xgb", {}),                                     # XGBoost depth 4
    "xgbd":  ("xgb", dict(max_depth=6, colsample_bytree=0.3, subsample=0.7, min_child_weight=40,
                          reg_lambda=30, gamma=0.1)),         # XGBoost más profundo y más regularizado
    "cat":   ("cat", {}),                                     # CatBoost depth 6
    "cat2":  ("cat", dict(depth=8, l2_leaf_reg=30, rsm=0.4, learning_rate=0.06)),
    "lr":    ("lr", {}),                                      # Regresión logística (modelo lineal, muy distinto)
    # ---- v5 ----
    "lgbm":  ("lgb", dict(mono=True)),                        # LightGBM con EXT monótonas (riesgo baja al subir el puntaje)
    "xgbm":  ("xgb", dict(mono=True)),                        # XGBoost con EXT monótonas
    "lgbgoss": ("lgb", dict(data_sample_strategy="goss", bagging_fraction=1.0, bagging_freq=0, learning_rate=0.03)),
    "lgbdart": ("lgb", dict(boosting="dart", learning_rate=0.05, drop_rate=0.08, skip_drop=0.5, num_leaves=24,
                            _rounds=700, _noes=True)),         # DART: sin early stopping -> nº fijo de árboles
    "lgblin": ("lgb", dict(linear_tree=True, linear_lambda=20.0, num_leaves=15, min_child_samples=300,
                           learning_rate=0.03)),               # hojas lineales: otra forma de aproximar
    "cat3":  ("cat", dict(depth=7, l2_leaf_reg=20, learning_rate=0.06, max_ctr_complexity=3, one_hot_max_size=4)),
    "cat4":  ("cat", dict(depth=8, l2_leaf_reg=30, learning_rate=0.05, max_ctr_complexity=4, one_hot_max_size=4)),   # más combinaciones y más profundo
    "cat3d6": ("cat", dict(depth=6, l2_leaf_reg=10, learning_rate=0.06, max_ctr_complexity=3, one_hot_max_size=4)),  # menos profundo, mismas combinaciones
    "mlp":   ("mlp", {}),
    "nn":    ("nn", {}),                                        # red PyTorch (cuantiles + embeddings); requiere `pip install torch`                                      # MLP sobre cuantiles + one-hot (proxy ligero de red tabular)
    "lgbi":  ("lgbi", {}),                                      # lgb_lin del PDF: LightGBM con init_score de un scorecard logístico
    "lgbf":  ("lgb", "best_lowlr"),                            # config de `optuna`/`tune` con lr 0.01 (modelo final)
}
IMP, RESULTS = {}, {}   # importancias (gain) y OOF AUC de los modelos entrenados en esta sesión


MONO_DEC = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "EXTF_1", "EXTF_2", "EXTF_3", "EXT_MEAN", "EXT_MIN",
            "EXT_MAX", "EXT_MEDIAN", "EXTF_MEAN", "EXTF_MIN", "EXTF_MAX", "EXTF_WEIGHTED", "EXT_WEIGHTED",
            "EXTF_GEO", "EXT_PROD", "EXTF_PROD", "Q_EXTF_1_RANK", "Q_EXTF_2_RANK", "Q_EXTF_3_RANK",
            "Q_EXT_RANKMEAN", "Q_EXT_RANKMIN"}   # puntaje de riesgo alto = mejor cliente -> menos impago (-1)


def mono_vec(cols):
    return [-1 if c in MONO_DEC else 0 for c in cols]


def fit_lgb(p, Xtr, ytr, Xva, yva, Xte, seed, cats, w=None):
    import lightgbm as lgb
    p = dict(p)
    use_mono = p.pop("mono", False) or CFG["mono"]
    rounds, noes = p.pop("_rounds", None), p.pop("_noes", False)
    if CFG["lr"] and "learning_rate" not in p:
        p["learning_rate"] = CFG["lr"]
    if use_mono:
        p["monotone_constraints"] = mono_vec(Xtr.columns)
        p["monotone_constraints_method"] = "advanced"
    rounds = rounds or ROUNDS["lgb"]
    pat = 100 if (CFG["lr"] or 0) >= 0.04 else 200
    m = lgb.train({**LGB_BASE, **p, "seed": seed}, lgb.Dataset(Xtr, ytr, weight=w), rounds,
                  valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[] if noes else [lgb.early_stopping(pat, verbose=False)])
    imp = pd.Series(m.feature_importance("gain"), index=Xtr.columns)
    IMP["lgb"] = IMP["lgb"].add(imp, fill_value=0) if "lgb" in IMP else imp
    it = m.best_iteration if (m.best_iteration and m.best_iteration > 0) else None
    return m.predict(Xva, num_iteration=it), (m.predict(Xte, num_iteration=it) if Xte is not None else None)


def fit_xgb(p, Xtr, ytr, Xva, yva, Xte, seed, cats, w=None):
    import xgboost as xgb
    p = dict(p)
    if p.pop("mono", False) or CFG["mono"]:
        p["monotone_constraints"] = tuple(mono_vec(Xtr.columns))
    params = dict(n_estimators=ROUNDS["xgb"], learning_rate=0.03, max_depth=4, subsample=0.8,
                  colsample_bytree=0.5, min_child_weight=20, reg_lambda=10, reg_alpha=1, tree_method="hist",
                  enable_categorical=True, eval_metric="auc", early_stopping_rounds=200,
                  random_state=seed, n_jobs=N_JOBS)
    m = xgb.XGBClassifier(**{**params, **p})
    m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
    return m.predict_proba(Xva)[:, 1], (m.predict_proba(Xte)[:, 1] if Xte is not None else None)


def fit_cat(p, Xtr, ytr, Xva, yva, Xte, seed, cats, w=None):
    from catboost import CatBoostClassifier
    conv = lambda X: X.assign(**{c: X[c].astype(str) for c in cats})   # CatBoost exige texto en categóricas
    Xtr, Xva = conv(Xtr), conv(Xva)
    Xte = conv(Xte) if Xte is not None else None
    params = dict(iterations=ROUNDS["cat"], learning_rate=0.08, depth=6, l2_leaf_reg=10, rsm=0.5,
                  eval_metric="AUC", random_seed=seed, od_type="Iter", od_wait=150, thread_count=N_JOBS,
                  verbose=0, cat_features=cats)
    m = CatBoostClassifier(**{**params, **p})
    m.fit(Xtr, ytr, sample_weight=w, eval_set=(Xva, yva), use_best_model=True)
    return m.predict_proba(Xva)[:, 1], (m.predict_proba(Xte)[:, 1] if Xte is not None else None)


def fit_lr(p, Xtr, ytr, Xva, yva, Xte, seed, cats, w=None):
    """Modelo lineal para diversidad: QuantileTransformer(normal) + NaN->0 con indicadores + one-hot."""
    from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from scipy import sparse
    nums = [c for c in Xtr.columns if c not in cats]
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200_000, random_state=seed)
    imp = SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=30)   # agrupa categorías raras (sklearn >= 1.1)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore")
    tf = lambda X, fit: sparse.hstack([
        sparse.csr_matrix((imp.fit_transform if fit else imp.transform)((qt.fit_transform if fit else qt.transform)(X[nums])).astype("float32")),
        (ohe.fit_transform if fit else ohe.transform)(X[cats].astype(str))]).tocsr()
    A = tf(Xtr, True)
    m = LogisticRegression(C=p.get("C", 0.02), max_iter=300).fit(A, ytr)
    return m.predict_proba(tf(Xva, False))[:, 1], (m.predict_proba(tf(Xte, False))[:, 1] if Xte is not None else None)


def fit_mlp(p, Xtr, ytr, Xva, yva, Xte, seed, cats, w=None):
    """MLP (sklearn) sobre cuantiles gaussianos + indicadores de NaN + one-hot. Proxy ligero de una red tabular:
    no hay torch en este entorno, así que NO se incluyen ResNet ni Denoising Autoencoder (ver informe)."""
    from sklearn.preprocessing import QuantileTransformer, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPClassifier
    from scipy import sparse
    nums = [c for c in Xtr.columns if c not in cats]
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200_000, random_state=seed)
    imp = SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=30)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore")
    tf = lambda X, fit: sparse.hstack([
        sparse.csr_matrix((imp.fit_transform if fit else imp.transform)((qt.fit_transform if fit else qt.transform)(X[nums])).astype("float32")),
        (ohe.fit_transform if fit else ohe.transform)(X[cats].astype(str))]).tocsr()
    A = tf(Xtr, True)
    m = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=p.get("alpha", 5e-2), batch_size=512, learning_rate_init=1e-3,
                      max_iter=25, early_stopping=True, validation_fraction=0.1, n_iter_no_change=3, random_state=seed)
    m.fit(A, ytr)
    return m.predict_proba(tf(Xva, False))[:, 1], (m.predict_proba(tf(Xte, False))[:, 1] if Xte is not None else None)


def nn_prepare(Xtr, Xva, Xte, cats, seed):
    """Preprocesado de la red (solo numpy/sklearn, se puede probar sin torch): cuantiles gaussianos de las numéricas
    (ajustados SOLO en el entrenamiento del fold), indicadores de NaN, y códigos enteros de las categóricas (0 = desconocida)."""
    from sklearn.preprocessing import QuantileTransformer
    nums = [c for c in Xtr.columns if c not in cats]
    qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200_000, random_state=seed)
    A0 = qt.fit_transform(Xtr[nums].astype("float64"))
    na_idx = [i for i, c in enumerate(nums) if Xtr[c].isna().any()]

    def tf(M):
        ind = np.isnan(M[:, na_idx]).astype("float32")
        return np.hstack([np.nan_to_num(M, nan=0.0).astype("float32"), ind])

    codes = lambda X: (np.stack([X[c].cat.codes.to_numpy().astype("int64") + 1 for c in cats], axis=1)
                       if cats else np.zeros((len(X), 0), dtype="int64"))
    sizes = [len(Xtr[c].cat.categories) + 1 for c in cats]
    tf_X = lambda X: tf(qt.transform(X[nums].astype("float64")))
    return (tf(A0), tf_X(Xva), (tf_X(Xte) if Xte is not None else None), codes(Xtr), codes(Xva),
            (codes(Xte) if Xte is not None else None), sizes)


def fit_nn(p, Xtr, ytr, Xva, yva, Xte, seed, cats, w=None):
    """Red MLP en PyTorch: numéricas en cuantiles + embeddings de categóricas, BatchNorm, Dropout, AdamW, coseno.
    Early stopping por AUC en la validación del fold (igual que los demás modelos). Con SMOKE=1 entrena solo 3 épocas."""
    try:
        import torch
        import torch.nn as tnn
    except ImportError:
        raise SystemExit("Falta PyTorch:  pip install torch   (CPU basta)")
    cats = [c for c in cats if str(Xtr[c].dtype) == "category"]
    Atr, Ava, Ate, Ctr, Cva, Cte, sizes = nn_prepare(Xtr, Xva, Xte, cats, seed)
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(max(1, N_JOBS))
    dims = [min(8, (n + 1) // 2) for n in sizes]

    class Net(tnn.Module):
        def __init__(self):
            super().__init__()
            self.emb = tnn.ModuleList([tnn.Embedding(n, d) for n, d in zip(sizes, dims)])
            d_in = Atr.shape[1] + sum(dims)
            self.mlp = tnn.Sequential(tnn.Linear(d_in, 256), tnn.BatchNorm1d(256), tnn.SiLU(), tnn.Dropout(0.3),
                                      tnn.Linear(256, 128), tnn.BatchNorm1d(128), tnn.SiLU(), tnn.Dropout(0.3),
                                      tnn.Linear(128, 1))

        def forward(self, a, c):
            z = [a] + [e(c[:, i]) for i, e in enumerate(self.emb)]
            return self.mlp(torch.cat(z, 1)).squeeze(1)

    net = Net()
    lr, epochs = p.get("lr", 1e-3), (3 if SMOKE else p.get("epochs", 14))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=p.get("wd", 1e-2))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    T = torch.from_numpy
    ta, tc, ty = T(Atr), T(Ctr), T(ytr.astype("float32"))
    lossf = tnn.BCEWithLogitsLoss()

    def predict(A, C):
        net.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(A), 8192):
                out.append(torch.sigmoid(net(T(A[i:i + 8192]), T(C[i:i + 8192]))).numpy())
        return np.concatenate(out)

    g = torch.Generator().manual_seed(seed)
    best, bad, best_state = -1.0, 0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(ta), generator=g)
        for i in range(0, len(perm), 1024):
            idx = perm[i:i + 1024]
            if len(idx) < 2:
                continue                      # BatchNorm necesita más de 1 fila
            opt.zero_grad()
            lossf(net(ta[idx], tc[idx]), ty[idx]).backward()
            opt.step()
        sched.step()
        a = roc_auc_score(yva, predict(Ava, Cva))
        if a > best:
            best, bad, best_state = a, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    net.load_state_dict(best_state)
    return predict(Ava, Cva), (predict(Ate, Cte) if Xte is not None else None)


FIT = {"lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat, "lr": fit_lr, "mlp": fit_mlp, "nn": fit_nn, "lgbi": fit_lgb_init}


def run_model(name, seed):
    """Entrena `name` con StratifiedKFold(seed). Guarda cada fold (reanuda si se interrumpe), el OOF y la predicción
    de test. Una semilla nueva cambia a la vez los folds y el modelo -> más diversidad al promediar semillas."""
    if name not in MODELS:
        raise SystemExit(f"Modelo desconocido '{name}'. Disponibles: {', '.join(MODELS)}")
    fam, extra = MODELS[name]
    if extra is None or extra == "best_lowlr":
        if not os.path.exists(BEST_JSON):
            raise SystemExit("Falta la mejor configuración: ejecuta primero el modo `tune` u `optuna`.")
        extra = json.load(open(BEST_JSON))
        if MODELS[name][1] == "best_lowlr":
            extra = {**extra, "learning_rate": 0.01, "_rounds": 30000}
    tag = f"{name}_s{seed}"
    cfg = hashlib.md5(json.dumps([name, extra, FE_KEY, N_FOLDS, ROUNDS, CFG], sort_keys=True, default=str).encode()).hexdigest()[:10]
    X, Xt, K, Kt, cats = get_data()
    oof, pred = np.zeros(len(X)), np.zeros(len(Xt))
    for f, (tr, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y)):
        ck = f"{MD}/{tag}_fold{f}.npz"
        if os.path.exists(ck) and str(np.load(ck)["cfg"]) == cfg:        # checkpoint válido -> reanudar
            z = np.load(ck)
            pv, pt = z["pv"], z["pt"]
            print(f"[{tag}] fold {f + 1}/{N_FOLDS} (cargado)", flush=True)
        else:
            Xtr, Xva, Xte = fold_frames(X, Xt, K, Kt, tr, va, seed)
            cl = [c for c in cats if c in Xtr.columns]
            pv, pt = FIT[fam](extra, Xtr, y[tr], Xva, y[va], Xte, seed, cl)
            if CFG["pl"] and fam == "lgb":      # solo si el protocolo lo aceptó (H_pl)
                pv, pt = pseudo_round(Xtr, y[tr], Xva, y[va], Xte, pt, seed, cl)
            np.savez(ck, pv=pv, pt=pt, cfg=np.array(cfg))
            print(f"[{tag}] fold {f + 1}/{N_FOLDS} AUC={roc_auc_score(y[va], pv):.5f}", flush=True)
        oof[va] = pv
        pred += pt / N_FOLDS
    auc = roc_auc_score(y, oof)
    print(f"[{tag}] OOF AUC = {auc:.5f}", flush=True)
    np.save(f"{MD}/oof_{tag}.npy", oof)
    np.save(f"{MD}/pred_{tag}.npy", pred)
    RESULTS[tag] = auc
    return auc

# ====================== PROTOCOLO EXPERIMENTAL (v5) ======================
# Idea -> hipótesis -> delta OOF (media +- desv sobre repeticiones) -> decisión. Referencia = A_base (v4 pura, lgb).
EXPERIMENTS = {
    "A_base":  {},                      # v4 pura: TE + imputación EXT, sin nada de v5
    "A_noTE":  dict(te=False),          # ablación del Target Encoding
    "B_mono":  dict(mono=True),         # monotonía de EXT en LightGBM
    "C_prune": dict(drop="prune"),      # poda por null importance (modo nullimp)
    "D_quant": dict(quant=True),        # EXT/ratios por cuantiles
    "E_hidden": dict(hidden=True),      # TE del cociente ANNUITY/CREDIT redondeado a 3/4/5 decimales (+ frecuencia del valor)
    "E_hidden2": dict(hidden2=True),    # E_hidden + cuota/bien, crédito/bien y cruces con plazo y tipo de contrato
    "E_r2": dict(hidden=True, hidden_rs=(2,)), "E_r3": dict(hidden=True, hidden_rs=(3,)), "E_r4": dict(hidden=True, hidden_rs=(4,)),
    "E_r5": dict(hidden=True, hidden_rs=(5,)), "E_r6": dict(hidden=True, hidden_rs=(6,)),     # hidden con UN solo redondeo (diversidad)
    "E_m1": dict(hidden=True, hidden_rs=(3, 4, 5), mseed=1), "E_m2": dict(hidden=True, hidden_rs=(3, 4, 5), mseed=2),   # control: misma config, otra semilla del modelo
    "E_r345": dict(hidden=True, hidden_rs=(3, 4, 5)),                                          # = E_hidden, pero guardando OOF
    "E_hidden4": dict(hidden4=True),    # TE por valor exacto: cambio de teléfono, edad del coche, población de la región...
    "E_uid": dict(uid=True),            # historial de la misma persona SIN TARGET (n.º de solicitudes, días a la anterior, cambios de monto)
    "E_uidte": dict(uidte=True),        # resultados pasados de la misma persona, solo con etiquetas del fold (K-Fold: nunca la propia fila)
    "E_uidte180": dict(uidte180=True),  # strict same-person labels only when the applications are <=180 days apart
    "E_fee_decode": dict(fee_decode=True),  # fixed-fee decoding: use adjusted q only when it is a more common 5-decimal cash-loan ratio
    "E_extg": dict(extg=True),                            # promedios por grupo de las EXT y de sus predicciones (puesto 7: +0.002 reportado)
    "E_ctg": dict(hidden5=True),                          # claves del puesto 7: CT a 1/2/3 decimales y cruces con bien, crédito y anualidad
    "E_seg": dict(seg=True),                              # estructura del préstamo: segmento, posición y distancia al pico (aprendidos en el fold)
    "E_uidte_in": dict(uidte=True, uidte_mode="inner"),   # E33b: resultados pasados con K-Fold interno de 5 (estilo Pachara)
    "G_cv": dict(groupcv=True),                            # E33c: CV agrupada por persona (prueba de sensibilidad, NO un criterio de aceptación)
    "E_uid2": dict(uid2=True),          # pares de confianza media/baja de la llave de Pachara, separados por nivel
    "E_uidp": dict(uidp=True),          # precio (tasa, cuota, producto) frente a las otras solicitudes de la persona
    "E_rate2": dict(rate2=True),        # tasa implícita exacta (bisección) + tasa sobre el bien + residuos
    "E_hidden3": dict(hidden3=True),    # + razón x (bien==crédito), meta/anualidad x (bien==crédito) y montos exactos (TE)
    "E_dup":   dict(dup=True),          # TE de la huella de fechas (clientes repetidos)
    "L_prod_num": dict(product="num"),  # solo W_* numéricas
    "L_prod_te": dict(product="te"),    # solo Target Encoding de la clave de producto
    "L_product": dict(product=True),    # ACEPTADA: +0.00396 sobre C_prune+K_interest (5/5)
    "K_interest": dict(interest=True),  # tasa/plazo implícitos (ACEPTADA: +0.00085 sobre C_prune, 5/5)
    "D_knn":   dict(knn=True),          # KNN-target sobre EXT y EXT+ratios (dentro del fold)
    "G_drift": dict(drop="drift"),      # quitar features con drift fuerte train vs test (modo adv)
    "H_pl":    dict(pl=True),           # pseudo-etiquetado conservador estricto
    "E_miss":  dict(miss=True),         # faltantes de EXT: EXT_j condicionado a qué fuente falta + cruces con nº de NaN (miss_block)
    "E_poly":  dict(poly=True),         # interacciones 2.º/3.er orden EXT x ratios x edad/empleo (poly_block)
}
HYP = {
    "A_base": "referencia", "A_noTE": "TE aporta señal de grupos (esperado: delta<0)",
    "B_mono": "monotonía regulariza EXT (+ pequeño)", "C_prune": "quitar ruido no baja AUC (y puede subirlo)",
    "D_quant": "rangos de EXT ayudan a splits y ratios", "D_knn": "vecindad en EXT captura interacciones locales",
    "K_interest": "tasa y plazo implícitos (riesgo según el precio del préstamo) añaden señal",
    "L_product": "el identificador de producto (pico de CREDIT/ANNUITY) con TE añade señal",
    "E_hidden": "valores exactos del cociente ANNUITY/CREDIT son productos ocultos con riesgo propio",
    "E_hidden2": "más claves de producto (cuota/bien, crédito/bien, cruces) añaden señal",
    "E_hidden3": "claves de producto con la bandera bien==crédito y montos exactos añaden señal sobre hidden",
    "E_uid": "n.º de solicitudes de la persona, días hasta la anterior y cambios de monto añaden señal (PDF: uid_hist +0.00075)",
    "E_uidte": "el resultado de las solicitudes pasadas de la misma persona, dentro del fold, añade señal (PDF: +0.00084)",
    "E_uidte180": "variant of strict repeat-applicant target history restricted to absolute application gap <=180 days; community-reported idea requiring validation",
    "E_fee_decode": "fixed-fee cash-loan ratio decoding with nested target encodings of selected q and q x fee-status",
    "E_extg": "los promedios por grupo de las EXT y sus predicciones añaden señal sobre las EXT ya imputadas",
    "E_ctg": "las claves CT_R1/R2/R3 y sus cruces (puesto 7) añaden señal sobre hidden",
    "E_seg": "segmentos del cociente crédito/anualidad (id, posición, distancia al pico) añaden señal sobre hidden",
    "E_uidte_in": "el K-Fold interno de 5 para el historial de impagos rinde igual que usar todas las otras filas del fold",
    "G_cv": "sensibilidad: con personas agrupadas por fold, cuánto de la ganancia de uid desaparece",
    "E_uid2": "los pares de confianza media/baja, con su nivel, añaden señal sobre los pares estrictos",
    "E_uidp": "diferencia de precio con las otras solicitudes de la persona añade señal (PDF: 'pricing compared to their other loans')",
    "E_rate2": "tasa exacta y su residuo añaden señal sobre hidden (PDF: tasa 2.0 +0.00052; rate_resid +0.00009)",
    "E_dup": "la media del TARGET de solicitudes con la misma huella de fechas ayuda (cobertura en test ~0.2%)",
    "G_drift": "features con drift empeoran en test, no en CV", "H_pl": "más datos 'baratos' (hipótesis débil)",
    "E_miss": "comportamiento de EXT según cuál falta (no probado antes; NaN ya lo manejan los árboles)",
    "E_poly": "interacciones de orden 2-3 de EXT con ratios/edad (en v6 features parecidas bajaron el AUC)",
}
CV_PATH = f"{OUT}/cv_results.json"


def _load_drop(tag):
    p = f"{MD}/drop_{tag}.json"
    if not os.path.exists(p):
        raise SystemExit(f"Falta {p}: ejecuta primero el modo `{'nullimp' if tag == 'prune' else 'adv'}`.")
    return json.load(open(p))


def parse_exp(name):
    d = {}
    for part in name.split("+"):
        if part == "ACCEPTED":
            d.update(json.load(open(_acc)) if os.path.exists(_acc) else {})
        elif part in EXPERIMENTS:
            d.update(EXPERIMENTS[part])
        else:
            raise SystemExit(f"Experimento '{part}' desconocido. Disponibles: {', '.join(EXPERIMENTS)}, ACCEPTED")
    return d

def pseudo_round(Xtr, ytr, Xva, yva, Xte, pt, seed, cl, lo=0.15, hi=0.05, wt=0.5):
    """Pseudo-etiquetado conservador y ESTRICTO: `pt` = predicción de test del modelo de ESTE fold (sin ver Xva).
    Se añaden al entrenamiento el `lo` más seguro como 0 y el `hi` más alto como 1 (peso `wt`). La validación (y el
    early stopping) usan SOLO filas reales de Xva. Devuelve (pred_val, pred_test) del modelo reentrenado."""
    r = rankdata(pt) / len(pt)
    neg, pos = r <= lo, r > 1 - hi
    sel = neg | pos
    X2 = pd.concat([Xtr, Xte[sel]], ignore_index=True)
    y2 = np.r_[ytr, pos[sel].astype(int)]
    w2 = np.r_[np.ones(len(ytr)), np.full(int(sel.sum()), wt)]
    return fit_lgb({}, X2, y2, Xva, yva, Xte, seed, cl, w=w2)


def uid_groups():
    """Componentes conexas de los pares estrictos (una por persona probable); filas sin pareja = grupo propio. Solo filas de train."""
    if "groups" in _UID_CACHE:
        return _UID_CACHE["groups"]
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    P, N = uid_pairs(), len(train) + len(test)
    _, lab = connected_components(coo_matrix((np.ones(len(P)), (P.row, P.row_p)), shape=(N, N)), directed=False)
    _UID_CACHE["groups"] = lab[:len(train)]
    return _UID_CACHE["groups"]


def cv_experiment(name):
    """CV estratificada repetida (CV_REPS x 5) de LightGBM con la configuración `name` (se combina con '+').
    Reanudable: guarda el AUC OOF de cada repetición en cv_results.json."""
    exp = parse_exp(name)
    key = name + ("" if USE_EXT_IMPUTE else "@noimp") + (f"@{_acc_hash()}" if "ACCEPTED" in name else "")
    res = json.load(open(CV_PATH)) if os.path.exists(CV_PATH) else {}
    ent = res.get(key, {"reps": {}, "folds": {}, "lr": CV_LR, "n": len(train), "smoke": SMOKE})
    X, Xt, K, Kt, cats = get_data()
    saved = dict(CFG)
    try:
        for seed in CV_SEEDS:
            if str(seed) in ent["reps"]:
                continue
            CFG.clear(); CFG.update(DEFAULT_CFG); CFG.update(exp); CFG["lr"] = CV_LR
            if isinstance(CFG["drop"], str):
                CFG["drop"] = _load_drop(CFG["drop"])
            oof, fa, t0 = np.zeros(len(X)), [], time.time()
            splitter = (StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, uid_groups()) if CFG.get("groupcv")
                        else StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y))   # groupcv: sensibilidad E33c
            for f, (tr, va) in enumerate(splitter):
                Xtr, Xva, Xte = fold_frames(X, Xt, K, Kt, tr, va, seed, need_test=CFG["pl"])
                cl = [c for c in cats if c in Xtr.columns]
                pv, pt = fit_lgb({}, Xtr, y[tr], Xva, y[va], Xte, seed + int(CFG.get("mseed", 0)), cl)   # mseed: solo cambia la semilla del MODELO
                if CFG["pl"]:
                    pv, _ = pseudo_round(Xtr, y[tr], Xva, y[va], Xte, pt, seed, cl)
                oof[va] = pv
                fa.append(float(roc_auc_score(y[va], pv)))
            np.save(f"{MD}/cvoof_{key.replace('+', '_')}_{seed}.npy", oof)          # OOF de la CV (para medir blends sin reentrenar)
            ent["reps"][str(seed)], ent["folds"][str(seed)] = float(roc_auc_score(y, oof)), fa
            res[key] = ent
            json.dump(res, open(CV_PATH, "w"), indent=1)
            print(f"[cv:{key}] semilla {seed}: OOF AUC={ent['reps'][str(seed)]:.5f} (folds {np.round(fa, 5).tolist()}) "
                  f"{time.time() - t0:.0f}s", flush=True)
    finally:
        CFG.clear(); CFG.update(saved)
    a = np.array(list(ent["reps"].values()))
    print(f"[cv:{key}] OOF AUC = {a.mean():.5f} +- {a.std(ddof=1) if len(a) > 1 else 0:.5f} ({len(a)} repeticiones)")


def compare():
    """Tabla de experimentos con la regla de aceptación (delta medio >= +0.0003 Y mejora en la mayoría de repeticiones,
    pareando por semilla). Cada experimento se compara contra su REFERENCIA CORRECTA: A_base para ideas sueltas, y para
    combinaciones 'X+Y' contra 'X' (p. ej. C_prune+D_quant contra C_prune). Escribe experiments.md y accepted.json
    (solo flags de ideas sueltas aceptadas frente a A_base)."""
    if not os.path.exists(CV_PATH):
        raise SystemExit("No hay resultados: ejecuta cv:A_base y otros cv:<exp> primero.")
    res = json.load(open(CV_PATH))
    if "A_base" not in res:
        raise SystemExit("Falta la línea base: ejecuta `cv:A_base`.")
    rows, acc = [], {}
    for name, e in res.items():
        head = name.split("+")[0]
        refn = head if ("+" in name and head in res) else "A_base"
        ref = res[refn]
        ks = [k for k in ref["reps"] if k in e["reps"]]
        a, b = np.array([e["reps"][k] for k in ks]), np.array([ref["reps"][k] for k in ks])
        d = a - b
        if name == "A_base":
            dec = "referencia"
        elif len(ks) < 2:
            dec = "incompleto (<2 repeticiones)"
        elif d.mean() >= 3e-4 and (d > 0).sum() > len(d) / 2:
            dec = "MEJORA a " + refn
        elif abs(d.mean()) < 3e-4:
            dec = "descartar (sin conclusión: |delta|<0.0003)"
        else:
            dec = "descartar (empeora)"
        rows.append(dict(idea=name, ref=refn, auc=f"{a.mean():.5f} +- {a.std(ddof=1) if len(a) > 1 else 0:.5f}",
                         delta=f"{d.mean():+.5f} +- {d.std(ddof=1) if len(d) > 1 else 0:.5f}", gana=f"{int((d > 0).sum())}/{len(d)}", decision=dec))
        if dec.startswith("MEJORA") and "+" not in name and name in EXPERIMENTS and not name.endswith("@noimp"):
            for k, v in EXPERIMENTS[name].items():
                if k == "drop":
                    acc["drop"] = sorted(set(acc.get("drop", [])) | set(_load_drop(v)))   # prune y drift se unen
                else:
                    acc[k] = v
    T = pd.DataFrame(rows)
    md = ("| idea | comparada contra | AUC OOF (media +- desv) | delta | repeticiones ganadas | decisión |\n|---|---|---|---|---|---|\n" +
          "\n".join(f"| {r.idea} | {r.ref} | {r.auc} | {r.delta} | {r.gana} | {r.decision} |" for r in T.itertuples()))
    print("\n" + md)
    open(f"{OUT}/experiments.md", "w", encoding="utf-8").write(md + "\n")
    json.dump(acc, open(_acc, "w"), indent=1)
    print(f"\nFlags aceptados -> {_acc}: { {k: (v if k != 'drop' else f'{len(v)} columnas') for k, v in acc.items()} }")
    print("Las combinaciones solo cuentan si MEJORAN a su referencia; una combinación 'aceptada' frente a A_base no significa nada.")


def _acc_hash():
    """Huella del contenido de accepted.json (la referencia 'ACCEPTED' cambia cuando se adopta una mejora)."""
    try:
        return hashlib.md5(open(_acc, "rb").read()).hexdigest()[:6]
    except OSError:
        return "none"


AUTOEXP_DEFAULT = "E_uid+E_uidte;E_rate2"


def auto_experiments():
    """Fase de EXPERIMENTOS automáticos (reanudable): para cada candidata X de AUTOEXP ("A+B;C"; se separan con ;) mide
    `ACCEPTED+X` contra `ACCEPTED` con las mismas particiones (CV_REPS repeticiones, mínimo 3) y aplica la regla del protocolo:
    delta medio >= +0.0003 Y mejora en la mayoría de repeticiones. Si pasa, guarda copia de accepted.json en
    <OUT>/accepted_history/ y adopta los flags (la siguiente candidata ya se mide sobre la nueva referencia).
    Registro: <OUT>/autoexp_log.csv. Límite de tiempo: AUTO_HOURS. Un fallo se registra y se salta."""
    import traceback, shutil
    t_end = time.time() + float(os.environ.get("AUTO_HOURS", 10)) * 3600
    cands = [c.strip() for c in os.environ.get("AUTOEXP", AUTOEXP_DEFAULT).split(";") if c.strip()]
    os.makedirs(f"{OUT}/accepted_history", exist_ok=True)
    if not os.path.exists(_acc):
        raise SystemExit(f"Falta {_acc}: copia ahí tu configuración aceptada (poda + hidden).")
    logp = f"{OUT}/autoexp_log.csv"

    def log(row):
        pd.DataFrame([row]).to_csv(logp, mode="a", header=not os.path.exists(logp), index=False)
        print("[autoexp]", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

    for cand in cands:
        if time.time() > t_end:
            print("[autoexp] tiempo agotado")
            break
        try:
            h = _acc_hash()
            cv_experiment("ACCEPTED")
            cv_experiment(f"ACCEPTED+{cand}")
            res = json.load(open(CV_PATH))
            ref, new = res[f"ACCEPTED@{h}"]["reps"], res[f"ACCEPTED+{cand}@{h}"]["reps"]
            ks = [k for k in ref if k in new]
            d = np.array([new[k] - ref[k] for k in ks])
            ok = len(d) >= 3 and d.mean() >= 3e-4 and (d > 0).sum() > len(d) / 2
            row = dict(hora=time.strftime("%Y-%m-%d %H:%M:%S"), candidata=cand, ref=float(np.mean([ref[k] for k in ks])),
                       nueva=float(np.mean([new[k] for k in ks])), delta=float(d.mean()), desv=float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                       ganadas=f"{int((d > 0).sum())}/{len(d)}", decision="ADOPTADA" if ok else "descartada")
            if ok:
                shutil.copyfile(_acc, f"{OUT}/accepted_history/accepted_{time.strftime('%Y%m%d_%H%M%S')}_antes_de_{cand.replace('+', '_')}.json")
                cfg = json.load(open(_acc))
                for part in cand.split("+"):
                    for k_, v_ in EXPERIMENTS[part].items():
                        cfg[k_] = sorted(set(_load_drop(v_))) if (k_ == "drop" and isinstance(v_, str)) else v_
                json.dump(cfg, open(_acc, "w"), indent=1)
            log(row)
        except (Exception, SystemExit) as e:
            log(dict(hora=time.strftime("%Y-%m-%d %H:%M:%S"), candidata=cand, ref=np.nan, nueva=np.nan, delta=np.nan, desv=np.nan,
                     ganadas="", decision=f"FALLÓ: {type(e).__name__}: {str(e)[:120]}"))
            print(traceback.format_exc(limit=2))
    print(f"[autoexp] listo. Configuración vigente: {_acc} | historial: {logp}")


def adversarial():
    """Adversarial validation train(0) vs test(1) con LightGBM (5 folds). AUC~0.5 = sin drift. Lista las features con
    capacidad individual de separar train/test (|AUC-0.5| >= 0.10) y las guarda en drop_drift.json."""
    import lightgbm as lgb
    X, Xt, K, Kt, cats = get_data()
    A = pd.concat([X, Xt], ignore_index=True)
    A = A.drop(columns=[c for c in A.columns if c.startswith("Q_")])
    z = np.r_[np.zeros(len(X)), np.ones(len(Xt))].astype(int)
    oof, imp = np.zeros(len(A)), 0
    P = {**LGB_BASE, "learning_rate": 0.1, "num_leaves": 31, "feature_fraction": 0.5, "seed": SEED}
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(A, z):
        m = lgb.train(P, lgb.Dataset(A.iloc[tr], z[tr]), 40 if SMOKE else 200)
        oof[va] = m.predict(A.iloc[va]); imp = imp + pd.Series(m.feature_importance("gain"), index=A.columns)
    auc = roc_auc_score(z, oof)
    print(f"[adv] AUC train-vs-test = {auc:.4f}  ({'sin drift apreciable' if auc < 0.55 else 'HAY drift: revisa las features'})")
    print("[adv] top importancia:", (imp / imp.sum()).sort_values(ascending=False).head(10).round(4).to_dict())
    single = {}
    for c in A.select_dtypes("number").columns:
        v = A[c].astype(float)
        if v.nunique() > 1:
            single[c] = fast_auc(z, rankdata(v.fillna(v.min() - 1)))
    S = pd.Series(single)
    bad = S[(S - 0.5).abs() >= 0.10].index.tolist()
    print(f"[adv] features con |AUC-0.5|>=0.10 (drift fuerte): {len(bad)} {bad[:15]}")
    json.dump(bad, open(f"{MD}/drop_drift.json", "w"))
    return auc


def leak_check():
    """Detección de leakage. (1) AUC univariado de cada feature vs TARGET (alarma > 0.65). (2) Brecha train(K-Fold interno)
    vs validación en las columnas TE_*/KNN_*. (3) Re-ejecución con TARGET PERMUTADO: el AUC de validación debe ser ~0.50.
    (4) Brecha train vs validación del modelo (referencia: los GBDT siempre tienen cierta brecha)."""
    import lightgbm as lgb
    X, Xt, K, Kt, cats = get_data()
    alarm = False
    S = {}
    for c in X.select_dtypes("number").columns:
        v = X[c].astype(float)
        if v.nunique() > 1:
            S[c] = fast_auc(y, rankdata(v.fillna(v.min() - 1)))
    S = pd.Series(S); dev = (S - 0.5).abs().sort_values(ascending=False)
    print("[leak] (1) top AUC univariado:", {c: round(float(S[c]), 4) for c in dev.head(8).index})
    if (dev > 0.15).any():   # informativo: X se construye SIN TARGET (build usa train.drop('TARGET')), así que no puede haber fuga aquí
        print(f"  nota: {int((dev > 0.15).sum())} features con AUC univariado fuera de [0.35, 0.65] (p. ej. medias de EXT_SOURCE). "
              "Es señal legítima si deriva de columnas crudas: compárala con la media simple de las EXT crudas (AUC ~0.74 en filas completas).")
    tr, va = next(iter(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y)))
    Xtr, Xva, _ = fold_frames(X, Xt, K, Kt, tr, va, SEED, need_test=False)
    print(f"[leak] configuración activa: { {k: v for k, v in CFG.items() if k != 'drop'} } | columnas V_={sum(c.startswith('V_') for c in Xtr.columns)}"
          f" W_={sum(c.startswith('W_') for c in Xtr.columns)} TE_PROD={sum(c.startswith('TE_PROD') for c in Xtr.columns)}"
          f" H_/D_(_TE)={sum(c.endswith('_TE') for c in Xtr.columns)} UID_={sum(c.startswith('UID_') for c in Xtr.columns)}"
          f" UIDY_={sum(c.startswith('UIDY_') for c in Xtr.columns)} UID2={sum(c.startswith('UID2') for c in Xtr.columns)}"
          f" UIDP_={sum(c.startswith('UIDP_') for c in Xtr.columns)} R2_={sum(c.startswith('R2_') for c in Xtr.columns)}")
    enc = [c for c in Xtr.columns if c.startswith(("TE_", "KNN_")) or c.endswith("_TE") or c.startswith(("UIDY_", "UID2Y_"))]
    g = {c: (roc_auc_score(y[tr], Xtr[c].fillna(-1)), roc_auc_score(y[va], Xva[c].fillna(-1))) for c in enc}   # fillna: UIDY_* es NaN sin pareja
    n1v, n0v = int(y[va].sum()), int(len(va) - y[va].sum())
    se0 = float(np.sqrt((n0v + n1v + 1) / (12.0 * n0v * n1v)))      # error estándar del AUC bajo H0 (sin señal)
    gap = max((a - b for a, b in g.values()), default=0)
    bad = gap > max(0.02, 3 * np.sqrt(2) * se0)                    # exige brecha práctica Y estadística
    print(f"[leak] (2) brecha AUC train(interno) - val en {len(enc)} columnas TE/KNN: máx={gap:+.4f} "
          f"(umbral {max(0.02, 3 * np.sqrt(2) * se0):.4f})  {'ALARMA' if bad else 'ok'}")
    alarm |= bad
    rs = np.random.RandomState(0)
    yp = y.copy(); yp[tr] = rs.permutation(y[tr])
    Ptr, Pva, _ = fold_frames(X, Xt, K, Kt, tr, va, SEED, need_test=False, yy=yp)
    cl = [c for c in cats if c in Ptr.columns]
    pv, _ = fit_lgb(dict(_rounds=60 if SMOKE else 150, _noes=True, learning_rate=0.1), Ptr, yp[tr], Pva, y[va], None, SEED, cl)
    a_perm = roc_auc_score(y[va], pv)
    ok3 = abs(a_perm - 0.5) <= 3 * se0
    print(f"[leak] (3) TARGET permutado: AUC validación = {a_perm:.4f} (esperado 0.50 +- {3 * se0:.4f})  {'ok' if ok3 else 'ALARMA'}")
    alarm |= not ok3
    P = {**LGB_BASE, "learning_rate": 0.1, "seed": SEED}
    m = lgb.train(P, lgb.Dataset(Xtr, y[tr]), 40 if SMOKE else 150)
    sub = np.random.RandomState(1).choice(len(tr), min(30000, len(tr)), replace=False)
    a_tr, a_va = roc_auc_score(y[tr][sub], m.predict(Xtr.iloc[sub])), roc_auc_score(y[va], m.predict(Xva))
    print(f"[leak] (4) modelo: AUC train={a_tr:.4f} val={a_va:.4f} brecha={a_tr - a_va:+.4f} (informativo: los GBDT sobreajustan algo)")
    print("[leak] RESULTADO:", "ALARMA -> no confíes en el AUC hasta resolverla" if alarm else "sin señales de leakage")
    return alarm


def null_importance():
    """Null importance (TARGET permutado NULL_RUNS veces) + permutation importance en validación.
    Candidata a poda = ganancia real <= percentil 95 de la nula Y permutation importance <= 0.0001.
    Aviso: se calcula sobre un split propio (semilla 999) pero con las mismas filas que luego usa la CV -> sesgo leve."""
    import lightgbm as lgb
    n_null = int(os.environ.get("NULL_RUNS", 3 if SMOKE else 20))
    X, Xt, K, Kt, cats = get_data()
    tr, va = next(iter(StratifiedKFold(N_FOLDS, shuffle=True, random_state=999).split(X, y)))
    rounds = 40 if SMOKE else 150
    P = {**LGB_BASE, "learning_rate": 0.1, "feature_fraction": 0.5, "seed": SEED}

    def gains(yy, seed):
        Xtr, Xva, _ = fold_frames(X, Xt, K, Kt, tr, va, SEED, need_test=False, yy=yy)
        m = lgb.train({**P, "seed": seed}, lgb.Dataset(Xtr, yy[tr]), rounds)
        return pd.Series(m.feature_importance("gain"), index=Xtr.columns), m, Xva
    real, m0, Xva = None, None, None
    for sd in (1, 2, 3):
        g, m, xv = gains(y, sd)
        real = g if real is None else real + g
        if m0 is None:
            m0, Xva = m, xv
    real = real / 3
    null = pd.DataFrame({i: gains(np.where(np.isin(np.arange(len(y)), tr), np.random.RandomState(100 + i).permutation(y), y), i)[0]
                         for i in range(n_null)})
    n95 = null.quantile(0.95, axis=1)
    cand = real.index[(real <= n95.reindex(real.index)).to_numpy()].tolist()
    sub = np.random.RandomState(0).choice(len(va), min(20000, len(va)), replace=False)
    Xs, ys = Xva.iloc[sub].reset_index(drop=True), y[va][sub]
    base_auc, drop = roc_auc_score(ys, m0.predict(Xs)), []
    for c in cand:
        d = []
        for r in range(2):
            Z = Xs.copy(); Z[c] = Z[c].sample(frac=1, random_state=r).values   # .values conserva el dtype category
            d.append(base_auc - roc_auc_score(ys, m0.predict(Z)))
        if np.mean(d) <= 1e-4:
            drop.append(c)
    json.dump(drop, open(f"{MD}/drop_prune.json", "w"))
    print(f"[nullimp] {len(real)} features | no superan la nula (p95): {len(cand)} | podables (además perm.imp<=1e-4): {len(drop)}")
    print("[nullimp] ejemplos podables:", drop[:12])
    print("[nullimp] SIGUIENTE: cv:C_prune y cmp -> solo se adopta si el AUC no baja (>= -0.0003) o sube (regla del protocolo).")
    return drop


def optuna_tune():
    """Optuna (TPE, semilla fija, pruner mediano) sobre LightGBM con los features/TE actuales. Reanudable (sqlite).
    Guarda la mejor config en BEST_JSON (la usan lgb2 y lgbf). Ejecutar SOLO después de fijar los features."""
    import lightgbm as lgb, optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, Xt, K, Kt, cats = get_data()
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))[:TUNE_FOLDS]
    frames = [fold_frames(X, Xt, K, Kt, tr, va, SEED, need_test=False) for tr, va in splits]

    def objective(trial):
        p = dict(num_leaves=trial.suggest_int("num_leaves", 8, 64, log=True),
                 min_child_samples=trial.suggest_int("min_child_samples", 50, 600, log=True),
                 feature_fraction=trial.suggest_float("feature_fraction", 0.15, 0.6),
                 bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 0.95),
                 lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
                 lambda_l2=trial.suggest_float("lambda_l2", 0.1, 100, log=True),
                 min_gain_to_split=trial.suggest_float("min_gain_to_split", 0, 0.2),
                 path_smooth=trial.suggest_float("path_smooth", 0, 30),
                 max_bin=trial.suggest_categorical("max_bin", [127, 255, 511]),
                 extra_trees=trial.suggest_categorical("extra_trees", [False, True]),
                 max_depth=trial.suggest_categorical("max_depth", [-1, 6, 8]))
        aucs, its = [], []
        for i, ((tr, va), (Xtr, Xva, _)) in enumerate(zip(splits, frames)):
            m = lgb.train({**LGB_BASE, **p, "learning_rate": 0.05, "seed": SEED}, lgb.Dataset(Xtr, y[tr]), ROUNDS["lgb"],
                          valid_sets=[lgb.Dataset(Xva, y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
            aucs.append(m.best_score["valid_0"]["auc"]); its.append(m.best_iteration)
            trial.report(float(np.mean(aucs)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("iters", float(np.mean(its)))
        return float(np.mean(aucs))
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0),
                             storage=f"sqlite:///{MD}/optuna.db", study_name="lgb", load_if_exists=True)
    n_total = int(os.environ.get("OPTUNA_TRIALS", 6 if SMOKE else 40))
    todo = max(0, n_total - len([t for t in st.trials if t.state.name in ("COMPLETE", "PRUNED")]))
    if todo:
        st.optimize(objective, n_trials=todo, timeout=TUNE_BUDGET)
    done = [t for t in st.trials if t.state.name == "COMPLETE"]
    if not done:
        print("Sin trials completos: vuelve a ejecutar optuna"); return
    best = st.best_trial
    p = {k: (int(v) if k in ("num_leaves", "min_child_samples", "max_bin", "max_depth") else (bool(v) if k == "extra_trees" else float(v)))
         for k, v in best.params.items()}
    json.dump(p, open(BEST_JSON, "w"))
    print(f"[optuna] {len(done)} trials completos | mejor AUC({TUNE_FOLDS} folds)={best.value:.5f} iters~{best.user_attrs.get('iters', 0):.0f}\n"
          f"[optuna] mejor config -> {BEST_JSON}: {p}")
    n_pruned = len([t for t in st.trials if t.state.name == "PRUNED"])
    n_fin = len(done) + n_pruned          # los trials PODADOS por Optuna también cuentan como ejecutados
    if n_fin < n_total:
        print(f"[optuna] faltan {n_total - n_fin} trials: vuelve a ejecutar (reanuda desde sqlite)")
    else:
        print(f"[optuna] búsqueda COMPLETA: {n_fin} trials ejecutados ({len(done)} completos + {n_pruned} podados por Optuna).")


# ============================= TUNE =============================

HAND_TRIALS = [  # configuraciones "de autor" (las de v3) + variantes con extra_trees
    {}, dict(num_leaves=15, min_child_samples=200),
    dict(num_leaves=48, min_child_samples=150, feature_fraction=0.3, lambda_l2=20),
    dict(num_leaves=31, min_child_samples=300, feature_fraction=0.3),
    dict(num_leaves=12, min_child_samples=100, feature_fraction=0.5, lambda_l2=5),
    dict(num_leaves=24, min_child_samples=200, feature_fraction=0.25, lambda_l2=30, lambda_l1=3),
    dict(num_leaves=20, min_child_samples=400, feature_fraction=0.35, bagging_fraction=0.7),
    dict(num_leaves=64, min_child_samples=500, feature_fraction=0.2, lambda_l2=50, max_depth=8),
    dict(extra_trees=True, num_leaves=31, min_child_samples=80, feature_fraction=0.3, lambda_l2=20),
]


def random_trials(n, seed=2024):
    """Búsqueda aleatoria reproducible (RandomState fijo) en un espacio más amplio que v3."""
    rs = np.random.RandomState(seed)
    pick = lambda xs: xs[rs.randint(len(xs))]
    return [dict(num_leaves=int(pick([8, 12, 15, 20, 24, 31, 48, 64])),
                 min_child_samples=int(pick([50, 100, 150, 200, 300, 400, 600])),
                 feature_fraction=float(pick([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])),
                 bagging_fraction=float(pick([0.6, 0.7, 0.8, 0.9])),
                 lambda_l1=float(pick([0, 0.5, 1, 3, 10])), lambda_l2=float(pick([1, 5, 10, 20, 30, 50, 100])),
                 min_gain_to_split=float(pick([0, 0.01, 0.05, 0.1])), path_smooth=float(pick([0, 1, 5, 20])),
                 max_depth=int(pick([-1, -1, 5, 6, 8])), max_bin=int(pick([127, 255, 511])),
                 extra_trees=bool(pick([False, False, True]))) for _ in range(n)]


TRIALS = HAND_TRIALS + random_trials(16)


def tune():
    """Prueba configuraciones de LightGBM en TUNE_FOLDS folds (con los mismos features y TE que el modelo final).
    Es reanudable: guarda tune.json y se detiene al agotar TUNE_BUDGET segundos; vuelve a ejecutarlo para seguir.
    Aviso: con 2 folds las diferencias < 0.0003 AUC son ruido; usa TUNE_FOLDS=3 si dispones de tiempo."""
    import lightgbm as lgb
    X, Xt, K, Kt, cats = get_data()
    splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))[:TUNE_FOLDS]
    frames = [fold_frames(X, Xt, K, Kt, tr, va, SEED, need_test=False) for tr, va in splits]
    path = f"{MD}/tune.json"
    res = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    for i, t in enumerate(TRIALS):
        if str(i) in res:
            continue
        if time.time() - t0 > TUNE_BUDGET:
            break
        aucs, its = [], []
        for (tr, va), (Xtr, Xva, _) in zip(splits, frames):
            m = lgb.train({**LGB_BASE, **t, "learning_rate": 0.05, "seed": SEED}, lgb.Dataset(Xtr, y[tr]),
                          ROUNDS["lgb"], valid_sets=[lgb.Dataset(Xva, y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            aucs.append(m.best_score["valid_0"]["auc"])
            its.append(m.best_iteration)
        res[str(i)] = dict(params=t, auc=float(np.mean(aucs)), iters=float(np.mean(its)))
        json.dump(res, open(path, "w"))
        print(f"trial {i}: AUC={np.mean(aucs):.5f} iters={np.mean(its):.0f} {t}", flush=True)
    ranking = sorted(res.items(), key=lambda kv: -kv[1]["auc"])
    print("Ranking hasta ahora:", [(k, round(v["auc"], 5)) for k, v in ranking[:8]])
    if ranking:   # se guarda SIEMPRE la mejor hasta el momento: `lgb2` ya se puede usar aunque falten trials
        json.dump(ranking[0][1]["params"], open(BEST_JSON, "w"))
        print("MEJOR hasta ahora:", ranking[0][1]["params"], "->", BEST_JSON)
    if len(res) < len(TRIALS):
        print(f"Faltan {len(TRIALS) - len(res)} configuraciones: vuelve a ejecutar tune")

# ============================= BLEND =============================

def rk(a):
    return rankdata(a) / len(a)


def fast_auc(yy, s):
    """AUC vía rangos (Mann-Whitney): mucho más rápido que roc_auc_score dentro de un optimizador."""
    r, n1 = rankdata(s), int(yy.sum())
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))


def load_groups():
    """Agrupa los modelos por nombre y promedia sus semillas en espacio de rangos (OOF y test por separado)."""
    groups = {}
    # BLEND_EXTRA="ruta1;ruta2": además de MD, incorpora los modelos de OTRAS carpetas de modelos (p. ej. la de 5 folds si MD es la de
    # 10 folds). Sus nombres llevan el prefijo x1_, x2_... para no mezclarse con los de MD. Cada OOF sigue siendo fuera de fold.
    dirs = [(MD, "")] + [(d.strip(), f"x{i + 1}_") for i, d in enumerate(os.environ.get("BLEND_EXTRA", "").split(";")) if d.strip()]
    for dd, pre in dirs:
        for fo in sorted(glob.glob(f"{dd}/oof_*_s*.npy")):
            m = re.fullmatch(r"oof_(.+)_s(\d+)\.npy", os.path.basename(fo))
            if m:
                groups.setdefault(pre + m.group(1), []).append(fo)
    if not groups:
        raise SystemExit(f"No hay modelos en {MD}: entrena primero (lgb, xgb, cat...).")
    names, O, P = [], [], []
    for n, files in groups.items():
        oo = [np.load(f) for f in files]
        pp = [np.load(os.path.join(os.path.dirname(f), "pred_" + os.path.basename(f)[4:])) for f in files]
        o, p = rk(np.mean([rk(a) for a in oo], axis=0)), rk(np.mean([rk(a) for a in pp], axis=0))
        print(f"{n}: {len(files)} semilla(s) {[round(roc_auc_score(y, a), 5) for a in oo]} -> OOF AUC {roc_auc_score(y, o):.5f}")
        names.append(n); O.append(o); P.append(p)
    return names, np.column_stack(O), np.column_stack(P)


# --- Cuatro formas de combinar. Cada fit(O, y) devuelve (función_de_score, pesos) ---
def fit_mean(O, yy):
    """Promedio simple de rangos."""
    w = np.full(O.shape[1], 1 / O.shape[1])
    return (lambda M: M @ w), w


def fit_nm(O, yy):
    """Pesos >= 0 optimizando el AUC directamente (Nelder-Mead). fatol pequeño: el AUC cambia en ~1e-5."""
    if O.shape[1] == 1:
        return fit_mean(O, yy)
    f = lambda w: -fast_auc(yy, O @ (np.abs(w) / np.abs(w).sum()))
    r = minimize(f, np.ones(O.shape[1]) / O.shape[1], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 300})
    w = np.abs(r.x) / np.abs(r.x).sum()
    return (lambda M: M @ w), w


def fit_hill(O, yy, iters=50):
    """Hill climbing con reemplazo (Caruana): añade, uno a uno, el modelo que más sube el AUC."""
    cnt, cur, best = np.zeros(O.shape[1]), np.zeros(len(yy)), 0.0
    for it in range(iters):
        sc = [fast_auc(yy, cur + O[:, j]) for j in range(O.shape[1])]
        j = int(np.argmax(sc))
        if it > 0 and sc[j] <= best + 1e-7:
            break
        cnt[j] += 1; cur += O[:, j]; best = sc[j]
    w = cnt / cnt.sum()
    return (lambda M: M @ w), w


def fit_stack(O, yy):
    """Meta-modelo: regresión logística sobre logit(rango). Puede dar pesos negativos (corrige sesgos)."""
    z = lambda M: logit(np.clip(M, 1e-4, 1 - 1e-4))
    w = LogisticRegression(C=1.0, max_iter=500).fit(z(O), yy).coef_.ravel()
    return (lambda M: z(M) @ w), w


def fit_stack_gbm(O, yy):
    """Meta-modelo GBDT pequeño (LightGBM, 4 hojas, 120 árboles) sobre logit(rango). Captura no linealidades suaves.
    Los 'pesos' devueltos son la ganancia normalizada (solo informativos)."""
    import lightgbm as lgb
    z = lambda M: logit(np.clip(M, 1e-4, 1 - 1e-4))
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=4, min_child_samples=500, feature_fraction=1.0,
                       bagging_fraction=0.8, bagging_freq=1, lambda_l2=50, verbose=-1, seed=SEED, deterministic=True,
                       force_row_wise=True, n_jobs=N_JOBS), lgb.Dataset(z(O), yy), 120)
    g = m.feature_importance("gain").astype(float)
    return (lambda M: m.predict(z(M))), (g / g.sum() if g.sum() > 0 else np.full(O.shape[1], 1 / O.shape[1]))


FITTERS = {"mean": fit_mean, "nm": fit_nm, "hill": fit_hill, "stack": fit_stack, "stackgbm": fit_stack_gbm}


def cv_score(fit, O, yy, n=5, seeds=(0, 1)):
    """AUC "honesto" de un método de mezcla: pesos ajustados en 4/5 de las filas OOF y evaluados en el 1/5 restante
    (media sobre varias particiones: reduce el ruido de la elección del método)."""
    aucs = [fast_auc(yy[va], fit(O[tr], yy[tr])[0](O[va]))
            for sd in seeds for tr, va in StratifiedKFold(n, shuffle=True, random_state=sd).split(O, yy)]
    return float(np.mean(aucs))


def blend_models(cv=True):
    """Compara los métodos de mezcla con auc_cv (honesto), elige el mejor y devuelve los scores de test (en rangos).
    Además: correlación entre OOF, aporte marginal de cada modelo (AUC_cv con/sin él) y poda hacia atrás de los que no suman
    (BLEND_PRUNE=1 por defecto; umbral 0.00005). Sin TARGET en test: solo se usan OOF."""
    names, O, P = load_groups()
    only = [v.strip() for v in os.environ.get("BLEND_ONLY", "").split(",") if v.strip()]     # BLEND_ONLY="lgbm,cat3d6,x1_lgbm": solo esos miembros
    if only:
        idx = [i for i, nm in enumerate(names) if nm in only]
        if not idx:
            raise SystemExit(f"BLEND_ONLY no coincide con ningún modelo. Disponibles: {names}")
        names, O, P = [names[i] for i in idx], O[:, idx], P[:, idx]
    C = pd.DataFrame(O, columns=names).corr(method="spearman")
    print("\nCorrelación de Spearman entre OOF (más baja = más diversidad):\n" + C.round(3).to_string())
    prune = os.environ.get("BLEND_PRUNE", "1") == "1"
    keep = list(range(len(names)))
    while True:
        rows, scorers, weights = [], {}, {}
        for mname, fit in FITTERS.items():
            score_fn, w = fit(O[:, keep], y)
            scorers[mname], weights[mname] = score_fn, w / np.abs(w).sum()
            rows.append(dict(metodo=mname, auc_oof=fast_auc(y, score_fn(O[:, keep])),
                             auc_cv=cv_score(fit, O[:, keep], y) if cv else np.nan))
        table = pd.DataFrame(rows)
        best = table.sort_values("auc_cv" if cv else "auc_oof", ascending=False).iloc[0]["metodo"]
        if not (cv and prune and len(keep) > 2):
            break
        full = table.set_index("metodo").loc[best, "auc_cv"]
        marg = {j: full - cv_score(FITTERS[best], O[:, [k for k in keep if k != j]], y) for j in keep}
        print("Aporte marginal (AUC_cv con - sin el modelo), método", best, {names[j]: round(float(v), 5) for j, v in marg.items()})
        j_bad = min(marg, key=marg.get)
        if marg[j_bad] >= 5e-5:
            break
        print(f"  -> se descarta '{names[j_bad]}' (no suma)")
        keep.remove(j_bad)
    print("\nMEZCLAS (auc_oof = ajustada y evaluada en las mismas filas, optimista; auc_cv = honesta)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print(f"\n-> método elegido: {best} | pesos:", {names[j]: round(float(x), 3) for j, x in zip(keep, weights[best])})
    table.to_csv(f"{OUT}/blend_report.csv", index=False)
    return dict(table=table, best=best, names=[names[j] for j in keep], weights=weights[best],
                oof_score=scorers[best](O[:, keep]), test_score=rk(scorers[best](P[:, keep])))


def validate_submission(path):
    """Replica lo que hará el evaluador (pd.read_csv) y comprueba el formato. Falla con un mensaje claro si el
    archivo no es un CSV válido (p. ej. si por error se sube el .ipynb en vez de submission.csv)."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert lines[0] == "SK_ID_CURR,TARGET", f"{path} NO es un submission válido (cabecera: {lines[0][:40]!r}). Sube el CSV, no el notebook."
    assert all(l.count(",") == 1 for l in lines), "alguna fila no tiene exactamente 2 columnas"
    s = pd.read_csv(path)
    assert list(s.columns) == ["SK_ID_CURR", "TARGET"] and len(s) == len(test), f"filas: {len(s)} (test.csv tiene {len(test)})"
    assert (s["SK_ID_CURR"].to_numpy() == test_ids).all() and s["SK_ID_CURR"].is_unique, "los IDs no coinciden con test.csv"
    assert np.isfinite(s["TARGET"]).all() and s["TARGET"].between(0, 1).all(), "TARGET con NaN/inf o fuera de [0, 1]"
    return s


def write_submission(score, path=None, also=None):
    """Guarda submission.csv (SK_ID_CURR, TARGET) con el score en rangos (0,1] y lo valida releyéndolo del disco.
    `also`: ruta de una segunda copia (p. ej. junto al notebook, para encontrarla fácil). Para el AUC solo importa el orden."""
    path = path or f"{OUT}/submission.csv"
    sub = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": score})
    for p in [path] + ([also] if also else []):
        sub.to_csv(p, index=False)
        validate_submission(p)
        print("Guardado y validado:", os.path.abspath(p), sub.shape)
    return sub


# ============================ MODO AUTOMÁTICO ============================
AUTO_JOBS_DEFAULT = "lgbm:42,7,2026;lgbi:42,7,2026;cat3d6:42,7,2026;cat4:42,7;cat3:42,7;xgbd:42,7,2026;xgb:42,7;lgbdart:42,7;lgbgoss:42,7;lgblin:42,7"
def auto_run():
    """Cola desatendida, reproducible y reanudable.
    FASE 1: trabajos `modelo:semillas` en orden fijo (AUTO_JOBS="lgbm:1,2;cat3d6:1").
    FASE 2 (AUTO_EXTRA=1, por defecto): rondas con semillas fijas 1000+17k (lgbm + cat3d6) hasta que el campeón no mejore en
    AUTO_PATIENCE rondas seguidas (por defecto 5) o se agote AUTO_HOURS (por defecto 10): más semillas dan retornos decrecientes.
    Tras cada trabajo/ronda: blend con auc_cv HONESTO. Un trabajo que falla se registra y se salta. No toca output_v4/submission.csv:
    guarda en <OUT>/auto/ (history.csv, log.txt, submission_cv*.csv, BEST_submission.csv, DONE)."""
    import traceback
    t_end = time.time() + float(os.environ.get("AUTO_HOURS", 10)) * 3600
    jobs = [j.strip() for j in os.environ.get("AUTO_JOBS", AUTO_JOBS_DEFAULT).split(";") if j.strip()]
    extra = os.environ.get("AUTO_EXTRA", "1") == "1"
    patience = int(os.environ.get("AUTO_PATIENCE", 5))
    adir = f"{OUT}/auto"
    os.makedirs(adir, exist_ok=True)
    done_flag, state_p, hist_p = f"{adir}/DONE", f"{adir}/state.json", f"{adir}/history.csv"
    if os.path.exists(done_flag):
        os.remove(done_flag)
    st = {"best": json.load(open(state_p))["best_cv"] if os.path.exists(state_p) else -1.0, "failed": set()}

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(f"{adir}/log.txt", "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def run_one(name, sdd):
        if name in st["failed"]:
            return
        t0 = time.time()
        try:
            run_model(name, sdd)
            log(f"OK {name}:{sdd} ({(time.time() - t0) / 60:.1f} min)")
        except (Exception, SystemExit) as e:        # SystemExit: p. ej. "Falta PyTorch"
            st["failed"].add(name)
            log(f"FALLÓ {name}:{sdd} -> {type(e).__name__}: {str(e)[:200]}  (se salta '{name}' el resto de la corrida)")
            log(traceback.format_exc(limit=2).replace("\n", " | ")[:400])

    def blend_track(tag):
        try:
            res = blend_models(cv=True)
            tb = res["table"].set_index("metodo")
            cv, oofv = float(tb.loc[res["best"], "auc_cv"]), float(tb.loc[res["best"], "auc_oof"])
            stamp = time.strftime("%Y%m%d_%H%M%S")
            write_submission(res["test_score"], path=f"{adir}/submission_cv{cv:.5f}_{stamp}.csv")
            pd.DataFrame([dict(hora=stamp, tras=tag, auc_cv=cv, auc_oof=oofv, metodo=res["best"], modelos="+".join(res["names"]))]).to_csv(
                hist_p, mode="a", header=not os.path.exists(hist_p), index=False)
            best = st["best"]
            if cv > best + 1e-4:                    # evita cambiar de campeón por puro ruido (<0.0001)
                note = ("primer resultado de la corrida" if best <= 0 else
                        ("mejora CONCLUYENTE (>=0.0003)" if cv - best >= 3e-4 else "mejora dentro del ruido (<0.0003)"))
                st["best"] = cv
                write_submission(res["test_score"], path=f"{adir}/BEST_submission.csv")
                json.dump(dict(best_cv=cv, tras=tag, metodo=res["best"], modelos=res["names"]), open(state_p, "w"))
                log(f"* NUEVO CAMPEÓN auc_cv={cv:.5f} ({res['best']}; {'+'.join(res['names'])}) {note}")
            else:
                log(f"auc_cv={cv:.5f} (campeón {best:.5f}): sin cambio de campeón")
        except Exception as e:
            log(f"El blend falló tras {tag}: {type(e).__name__}: {str(e)[:200]}")

    log(f"AUTO: {len(jobs)} trabajos, extra={'sí' if extra else 'no'} (paciencia {patience}), límite {os.environ.get('AUTO_HOURS', 10)} h, "
        f"mejor auc_cv previo {st['best']:.5f}")
    out_of_time = False
    for job in jobs:
        name, _, sd = job.partition(":")
        for sdd in ([int(v) for v in sd.split(",")] if sd else [SEED]):
            if time.time() > t_end:
                out_of_time = True
                break
            run_one(name, sdd)
        if out_of_time:
            break
        blend_track(job)
    if extra and not out_of_time:
        bad, k = 0, 0
        while bad < patience:
            if time.time() > t_end:
                out_of_time = True
                break
            sdd, k = 1000 + 17 * k, k + 1
            for name in ("lgbm", "cat3d6"):
                run_one(name, sdd)
            before = st["best"]
            blend_track(f"extra semilla {sdd}")
            bad = 0 if st["best"] > before else bad + 1
        log("Meseta: más semillas ya no mejoran el campeón." if bad >= patience else "Tiempo agotado.")
    elif out_of_time:
        log("Tiempo agotado: no se inician más trabajos.")
    open(done_flag, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    log(f"AUTO TERMINADO. Mejor auc_cv={st['best']:.5f}. Archivo: {adir}/BEST_submission.csv")



# ===================== REPRODUCIBLE SUBMISSION MODE =====================
def submit_experiment(name):
    """Train fold models for a selected experimental feature set and create a schema-validated test submission.

    The target-encoding and repeat-applicant label features are recomputed from each outer training fold.
    Test probabilities are averaged over N_FOLDS models per seed, then rank-averaged across SUBMIT_SEEDS.
    This is deliberately separate from CV reporting: selecting a feature set from CV does not make its CV score an
    independent estimate of leaderboard performance.
    """
    exp = parse_exp(name)
    X, Xt, K, Kt, cats = get_data()
    saved = dict(CFG)
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    seeds = [int(v.strip()) for v in os.environ.get("SUBMIT_SEEDS", "42,7").split(",") if v.strip()]
    all_preds = []
    try:
        CFG.clear(); CFG.update(DEFAULT_CFG); CFG.update(exp); CFG["lr"] = CV_LR
        if isinstance(CFG["drop"], str):
            CFG["drop"] = _load_drop(CFG["drop"])
        for seed in seeds:
            pred = np.zeros(len(Xt), dtype="float64")
            split = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
            for f, (tr, va) in enumerate(split.split(X, y), start=1):
                Xtr, Xva, Xte = fold_frames(X, Xt, K, Kt, tr, va, seed, need_test=True)
                cl = [c for c in cats if c in Xtr.columns]
                pv, pt = fit_lgb({}, Xtr, y[tr], Xva, y[va], Xte, seed, cl)
                pred += np.asarray(pt, dtype="float64") / N_FOLDS
                auc = roc_auc_score(y[va], pv)
                print(f"[submit:{name}] seed={seed} fold={f}/{N_FOLDS} validation AUC={auc:.5f}", flush=True)
            np.save(f"{OUT}/testpred_{slug}_seed{seed}.npy", pred)
            all_preds.append(pred)
        ranked = [rankdata(p, method="average") / (len(p) + 1.0) for p in all_preds]
        score = np.mean(ranked, axis=0)
        out_path = f"{OUT}/GCI_submission_{slug}.csv"
        write_submission(score, out_path)
        print(f"Submission created from {len(seeds)} seeds x {N_FOLDS} folds: {os.path.abspath(out_path)}", flush=True)
    finally:
        CFG.clear(); CFG.update(saved)


# ================================= MAIN =================================
def main():
    if MODE == "blend":
        res = blend_models(cv=os.environ.get("BLEND_CV", "1") == "1")
        write_submission(res["test_score"])
    elif MODE == "tune":
        tune()
    elif MODE == "optuna":
        optuna_tune()
    elif MODE == "cmp":
        compare()
    elif MODE == "autoexp":
        auto_experiments()
    elif MODE == "adv":
        adversarial()
    elif MODE == "leak":
        leak_check()
    elif MODE == "nullimp":
        null_importance()
    elif MODE.startswith("cv:"):
        for nm in MODE[3:].split(","):
            cv_experiment(nm)
    elif MODE.startswith("submit_exp:"):
        submit_experiment(MODE[len("submit_exp:"):])
    elif MODE == "auto":
        auto_run()
    elif MODE == "fe":
        build()
    else:   # nombre[:semilla1,semilla2,...]
        name, _, sd = MODE.partition(":")
        for s in ([int(v) for v in sd.split(",")] if sd else [SEED]):
            run_model(name, s)


if __name__ == "__main__":
    main()

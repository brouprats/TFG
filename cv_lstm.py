# ==============================================================================
# cv_lstm.py — Cross-validation dels hiperparàmetres LSTM per S&P 500
# ==============================================================================
# Índex: S&P 500 (^GSPC), 1994-01 → 2024-01, log-preus mensuals
# Models: LSTM Univariant · LSTM Multivariant
# Cerca: look_back × units × n_layers × dropout (graella exhaustiva)
# CV: TimeSeriesSplit (5 folds, finestra expandible sobre el 80% de train)
# Mètrica: RMSE log-preu, mitja aritmètica entre folds (predicció determinista)
# Sortida: taula Top-10 per consola + cv_lstm_uni.csv / cv_lstm_multi.csv
#
# ==============================================================================

import os
import math
import random
import warnings
import itertools

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential

try:
    import pandas_datareader.data as web
    HAS_PDR = True
except ImportError:
    HAS_PDR = False
    print("AVÍS: pandas_datareader no instal·lat — exog. cauen a zeros.")

from DadesPresidents import DFPresidents

# ==============================================================================
# LLAVORS I CONFIGURACIÓ (idèntiques a main_3.py)
# ==============================================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

warnings.filterwarnings("ignore")

START        = "1994-01-01"
END          = "2024-01-01"
TRAIN_RATIO  = 0.80
FRED_API_KEY = "7ffb3429bbbf66c1ce43a74f18f751d0"
FRED_CACHE_DIR = os.path.join(os.path.dirname(__file__), "fred_cache")

# ==============================================================================
# GRAELLA D'HIPERPARÀMETRES
# ==============================================================================
CV_GRID = {
    'look_back': [3, 6, 12],
    'units':     [8, 16, 32],
    'n_layers':  [1],
    'dropout':   [0.1, 0.2, 0.3, 0.4],
}

N_SPLITS   = 3   # TimeSeriesSplit folds
CV_EPOCHS  = 30  # Idèntic a main_3.py per comparabilitat
BATCH_SIZE = 32

# ==============================================================================
# SÈRIES FRED (S&P 500 → regió US)
# ==============================================================================
FRED_SERIES_US = {
    'inflation':    'CPIAUCSL',
    'unemployment': 'UNRATE',
    'interest':     'FEDFUNDS',
}

# ==============================================================================
# LOOKUPS POLÍTICS US (còpia de main_3.py)
# ==============================================================================
_df_pres = DFPresidents()
_df_pres['Start_Date'] = pd.to_datetime(_df_pres['Start_Date'])
_df_pres['End_Date']   = pd.to_datetime(_df_pres['End_Date'])


def _us_code(date):
    mask = (_df_pres['Start_Date'] <= date) & (_df_pres['End_Date'] >= date)
    row  = _df_pres.loc[mask]
    if row.empty:
        return 0
    return 1 if 'Republican' in row.iloc[0]['Party'] else 0


def get_political_code_us(date_index):
    codes = [_us_code(d) for d in date_index]
    return pd.Series(codes, index=date_index, name='party_code')


# ==============================================================================
# DESCÀRREGA DE DADES (idèntica a main_3.py)
# ==============================================================================

def download_index(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw[('Close', ticker)]
    else:
        close = raw['Close']
    monthly     = close.resample('ME').mean()
    log_monthly = np.log(monthly).dropna()
    log_monthly.name = 'log_price'
    return log_monthly


def _fred_cache_path(series_id):
    return os.path.join(FRED_CACHE_DIR, f"{series_id}.csv")


def _load_fred_from_cache(series_id):
    path = _fred_cache_path(series_id)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['date'])
    s  = pd.Series(df[series_id].values, index=df['date'], name=series_id)
    s.index = pd.PeriodIndex(s.index, freq='M')
    return s


def _save_fred_to_cache(series_id, s):
    os.makedirs(FRED_CACHE_DIR, exist_ok=True)
    out = pd.DataFrame({'date': s.index.to_timestamp(), series_id: s.values})
    out.to_csv(_fred_cache_path(series_id), index=False)


def download_fred(series_id, start, end):
    cached = _load_fred_from_cache(series_id)
    if cached is not None:
        return cached
    if not HAS_PDR:
        return None
    try:
        os.environ['FRED_API_KEY'] = FRED_API_KEY
        fetch_start = pd.to_datetime(start) - pd.DateOffset(months=15)
        df = web.DataReader(series_id, 'fred', fetch_start, end)
        s  = df.iloc[:, 0]
        s  = s.resample('ME').last().ffill()
        s.index = pd.PeriodIndex(s.index, freq='M')
        s.name  = series_id
        _save_fred_to_cache(series_id, s)
        return s
    except Exception as e:
        print(f"    AVÍS: No s'ha pogut descarregar '{series_id}': {e}")
        return None


def _transform_exog(var_name, s):
    if var_name == 'inflation':
        return 100.0 * (s / s.shift(12) - 1.0)
    if var_name in ('unemployment', 'interest'):
        return s.diff(1)
    return s


def build_exog_us(date_index):
    """
    Construeix el DataFrame de variables exògenes per a la regió US.
    Idèntic a build_exog('US', ...) de main_3.py però sense dependència
    del diccionari INDEX_CONFIG.
    """
    target_period = pd.PeriodIndex(date_index, freq='M')
    data = {}
    for var, sid in FRED_SERIES_US.items():
        s_raw = download_fred(sid, START, END)
        if s_raw is not None:
            s_tx      = _transform_exog(var, s_raw).dropna()
            s_aligned = s_tx.reindex(target_period, method='ffill').bfill()
            data[var] = s_aligned.values.astype(float)
        else:
            data[var] = np.zeros(len(date_index))
    data['party_code'] = get_political_code_us(date_index).values.astype(float)
    return pd.DataFrame(data, index=date_index)


# ==============================================================================
# CONSTRUCCIÓ DE DATASETS LSTM (idèntica a main_3.py)
# ==============================================================================

def crear_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def crear_dataset_multi(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# ==============================================================================
# CONSTRUCCIÓ DEL MODEL LSTM (idèntica a main_3.py)
# ==============================================================================

def build_lstm_model(look_back, n_features, n_layers=1, dropout=0.2, units=32):
    layers_list = []
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        kwargs = {'return_sequences': return_seq}
        if i == 0:
            kwargs['input_shape'] = (look_back, n_features)
        layers_list.append(LSTM(units, **kwargs))
        layers_list.append(Dropout(dropout))
    layers_list.append(Dense(1))
    return Sequential(layers_list)


# ==============================================================================
# AVALUACIÓ D'UN SOL FOLD — UNIVARIANT
# ==============================================================================

def cv_fold_univariate(price_array, train_idx, val_idx,
                       look_back, units, n_layers, dropout):
    """
    Entrena i avalua el LSTM Univariant en un sol fold de TimeSeriesSplit.

    Paràmetres
    ----------
    price_array : np.ndarray, forma (N,)
        Log-preus mensuals de la partició de train total.
    train_idx, val_idx : np.ndarray d'enters
        Índexs del fold generat per TimeSeriesSplit.

    Retorna
    -------
    float : RMSE en escala log-preu, o np.nan si el fold és massa petit.
    """
    price_tr  = price_array[train_idx]
    price_val = price_array[val_idx]

    # Escalat: fit sobre train del fold, transform sobre val del fold
    scaler = MinMaxScaler()
    price_tr_sc  = scaler.fit_transform(price_tr.reshape(-1, 1))
    price_val_sc = scaler.transform(price_val.reshape(-1, 1))

    # Concatenació: els últims look_back punts de train serveixen de context
    # per construir les primeres seqüències de validació
    all_sc = np.vstack([price_tr_sc, price_val_sc])

    X_all, y_all = crear_dataset(all_sc, look_back)

    n_train_seq = len(price_tr_sc) - look_back
    if n_train_seq <= 0 or len(X_all) <= n_train_seq:
        return np.nan   # Fold massa petit per al look_back demanat

    X_train = X_all[:n_train_seq].reshape(-1, look_back, 1)
    y_train = y_all[:n_train_seq]
    X_val   = X_all[n_train_seq:].reshape(-1, look_back, 1)
    y_val   = y_all[n_train_seq:]

    if len(X_val) == 0:
        return np.nan

    # Reiniciar seed TF per a reproducibilitat entre folds
    tf.random.set_seed(SEED)

    model = build_lstm_model(look_back, 1, n_layers, dropout, units)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train,
              epochs=CV_EPOCHS, batch_size=BATCH_SIZE,
              verbose=0, validation_split=0.0)

    # Predicció determinista (training=False → dropout desactivat)
    pred_sc  = model(tf.constant(X_val, dtype=tf.float32),
                     training=False).numpy().flatten()

    # Invertir escalat per obtenir RMSE en escala log
    pred_log = scaler.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
    real_log = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    return math.sqrt(mean_squared_error(real_log, pred_log))


# ==============================================================================
# AVALUACIÓ D'UN SOL FOLD — MULTIVARIANT
# ==============================================================================

def cv_fold_multivariate(price_array, exog_array, train_idx, val_idx,
                          look_back, units, n_layers, dropout):
    """
    Entrena i avalua el LSTM Multivariant en un sol fold de TimeSeriesSplit.

    Paràmetres addicionals
    ----------------------
    exog_array : np.ndarray, forma (N, n_features)
        Variables exògenes (ja transformades, pre-escalat) per a la CV.

    Retorna
    -------
    float : RMSE en escala log-preu, o np.nan si el fold és massa petit.
    """
    price_tr  = price_array[train_idx]
    price_val = price_array[val_idx]
    exog_tr   = exog_array[train_idx]
    exog_val  = exog_array[val_idx]

    # Escalat independent per a preu i exògenes
    scaler_y = MinMaxScaler()
    scaler_x = MinMaxScaler()

    price_tr_sc  = scaler_y.fit_transform(price_tr.reshape(-1, 1))
    price_val_sc = scaler_y.transform(price_val.reshape(-1, 1))
    exog_tr_sc   = scaler_x.fit_transform(exog_tr)
    exog_val_sc  = scaler_x.transform(exog_val)

    train_data = np.hstack([price_tr_sc, exog_tr_sc])
    val_data   = np.hstack([price_val_sc, exog_val_sc])
    all_data   = np.vstack([train_data, val_data])

    X_all, y_all = crear_dataset_multi(all_data, look_back)

    n_train_seq = len(price_tr_sc) - look_back
    if n_train_seq <= 0 or len(X_all) <= n_train_seq:
        return np.nan

    X_train = X_all[:n_train_seq]
    y_train = y_all[:n_train_seq]
    X_val   = X_all[n_train_seq:]
    y_val   = y_all[n_train_seq:]

    if len(X_val) == 0:
        return np.nan

    tf.random.set_seed(SEED)

    n_features = X_all.shape[2]
    model = build_lstm_model(look_back, n_features, n_layers, dropout, units)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train,
              epochs=CV_EPOCHS, batch_size=BATCH_SIZE,
              verbose=0, validation_split=0.0)

    pred_sc  = model(tf.constant(X_val, dtype=tf.float32),
                     training=False).numpy().flatten()

    pred_log = scaler_y.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
    real_log = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

    return math.sqrt(mean_squared_error(real_log, pred_log))


# ==============================================================================
# CERCA EXHAUSTIVA SOBRE LA GRAELLA
# ==============================================================================

def grid_search_cv(price_array, exog_array=None,
                   model_type='uni', grid=CV_GRID, n_splits=N_SPLITS):
    """
    Executa la cerca exhaustiva d'hiperparàmetres per TimeSeriesSplit CV.

    Paràmetres
    ----------
    price_array : np.ndarray, forma (N,)     — log-preus de la partició train
    exog_array  : np.ndarray, forma (N, k)   — exog. (None per a 'uni')
    model_type  : 'uni' | 'multi'
    grid        : dict amb les llistes de valors per a cada hiperparàmetre
    n_splits    : nombre de folds TimeSeriesSplit

    Retorna
    -------
    pd.DataFrame : files = combinació, columnes = hiperparàmetres + RMSE_mean + RMSE_std
                   ordenat per RMSE_mean ascendent.
    """
    tscv         = TimeSeriesSplit(n_splits=n_splits)
    keys         = list(grid.keys())
    combinations = list(itertools.product(*[grid[k] for k in keys]))
    total        = len(combinations)

    print(f"\n  Graella: {total} combinacions × {n_splits} folds = "
          f"{total * n_splits} entrenaments")

    results = []

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        lb, un, nl, dr = (params['look_back'], params['units'],
                          params['n_layers'],  params['dropout'])

        print(f"  [{i:3d}/{total}]  look_back={lb:2d}  units={un:2d}  "
              f"n_layers={nl}  dropout={dr:.1f}", end="  ", flush=True)

        fold_rmses = []
        for fold_n, (train_idx, val_idx) in enumerate(tscv.split(price_array), 1):
            if model_type == 'uni':
                rmse = cv_fold_univariate(
                    price_array, train_idx, val_idx, lb, un, nl, dr)
            else:
                rmse = cv_fold_multivariate(
                    price_array, exog_array, train_idx, val_idx, lb, un, nl, dr)

            status = f"{rmse:.5f}" if not np.isnan(rmse) else "skip"
            print(f"f{fold_n}={status}", end=" ", flush=True)

            if not np.isnan(rmse):
                fold_rmses.append(rmse)

        if fold_rmses:
            mean_rmse = float(np.mean(fold_rmses))
            std_rmse  = float(np.std(fold_rmses))
            n_valid   = len(fold_rmses)
        else:
            mean_rmse = np.nan
            std_rmse  = np.nan
            n_valid   = 0

        print(f"→ RMSE_mean={mean_rmse:.5f} ±{std_rmse:.5f} "
              f"({'%d/%d folds' % (n_valid, n_splits)})")

        results.append({
            **params,
            'RMSE_mean':  mean_rmse,
            'RMSE_std':   std_rmse,
            'folds_valid': n_valid,
        })

    df = (pd.DataFrame(results)
          .sort_values('RMSE_mean', na_position='last')
          .reset_index(drop=True))
    return df


# ==============================================================================
# IMPRESSIÓ DE RESULTATS
# ==============================================================================

def print_results_table(label, df, top_n=10):
    W = 11
    sep = "  " + "─" * (W * 7)
    total_w = W * 7 + 2

    print(f"\n{'═' * total_w}")
    print(f"  TOP {top_n} — {label}")
    print(f"{'═' * total_w}")
    print(f"  {'look_back':>{W}}{'units':>{W}}{'n_layers':>{W}}"
          f"{'dropout':>{W}}{'RMSE_mean':>{W}}{'RMSE_std':>{W}}{'folds_ok':>{W}}")
    print(sep)

    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        print(f"  {int(row.look_back):>{W}}{int(row.units):>{W}}"
              f"{int(row.n_layers):>{W}}{row.dropout:>{W}.1f}"
              f"{row.RMSE_mean:>{W}.5f}{row.RMSE_std:>{W}.5f}"
              f"{int(row.folds_valid):>{W}}")
    print(sep)

    # Millor combinació en negreta
    best = df.iloc[0]
    print(f"\n  ★  Millor: look_back={int(best.look_back)}  "
          f"units={int(best.units)}  n_layers={int(best.n_layers)}  "
          f"dropout={best.dropout:.1f}  →  RMSE={best.RMSE_mean:.5f}")


def print_best_summary(df_uni, df_multi):
    total_w = 72
    print(f"\n{'═' * total_w}")
    print(f"  RESUM — MILLORS HIPERPARÀMETRES SELECCIONATS PER S&P 500")
    print(f"{'═' * total_w}")

    for label, df in [("LSTM Univariant  ", df_uni), ("LSTM Multivariant", df_multi)]:
        r = df.iloc[0]
        print(f"  {label}:  look_back={int(r.look_back):2d}  units={int(r.units):2d}  "
              f"n_layers={int(r.n_layers)}  dropout={r.dropout:.1f}  "
              f"RMSE={r.RMSE_mean:.5f} ±{r.RMSE_std:.5f}")

    print(f"{'═' * total_w}")
    print()
    print("  Nota: transfereix aquests hiperparàmetres a main_3.py per a tots")
    print("  els índexs (S&P 500, Nasdaq, Euro Stoxx 50, Nikkei 225), o repeteix")
    print("  la CV per a cada índex si espereu comportaments molt heterogenis.")
    print(f"{'═' * total_w}")


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

if __name__ == '__main__':
    OUT_DIR = os.path.dirname(__file__)

    print("\n" + "=" * 72)
    print("  CV LSTM — S&P 500 — Cerca d'hiperparàmetres per TimeSeriesSplit")
    print("=" * 72)

    # ── Descàrrega de dades ──────────────────────────────────────────────────
    print("\n  ▸ Descarregant S&P 500 (^GSPC)...")
    log_price = download_index('^GSPC', START, END)
    n     = len(log_price)
    split = int(n * TRAIN_RATIO)
    print(f"    {n} mesos totals  ({log_price.index[0]} – {log_price.index[-1]})")
    print(f"    Train (CV): {split} mesos  |  Test (reservat): {n - split} mesos")

    # Extrem la partició de train (la CV NO toca el test)
    price_train = log_price.values[:split]

    # ── Variables exògenes (US) ──────────────────────────────────────────────
    print("\n  ▸ Construint variables exògenes (US)...")
    exog_df    = build_exog_us(log_price.index)
    exog_train = exog_df.values[:split].astype(float)
    print(f"    Columnes: {list(exog_df.columns)}")

    # ── Informació de la graella ─────────────────────────────────────────────
    total_combis = 1
    for v in CV_GRID.values():
        total_combis *= len(v)
    print(f"\n  Graella: {CV_GRID}")
    print(f"  Total combinacions: {total_combis} per model  "
          f"({total_combis * 2} en total)")
    print(f"  Total entrenaments: {total_combis * 2 * N_SPLITS} "
          f"({total_combis} comb × 2 models × {N_SPLITS} folds)")
    print(f"  Epochs per entrenament: {CV_EPOCHS}")

    # ── TimeSeriesSplit: estructura dels folds ───────────────────────────────
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"\n  Estructura dels {N_SPLITS} folds (sobre {split} mesos de train):")
    for k, (tr_idx, val_idx) in enumerate(tscv.split(price_train), 1):
        print(f"    Fold {k}:  train=[{tr_idx[0]}:{tr_idx[-1]+1}] "
              f"({len(tr_idx)} m.)  val=[{val_idx[0]}:{val_idx[-1]+1}] "
              f"({len(val_idx)} m.)")

    # ── LSTM Univariant ──────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  LSTM UNIVARIANT")
    print(f"{'─' * 72}")
    df_uni = grid_search_cv(price_train, exog_array=None,
                            model_type='uni', grid=CV_GRID, n_splits=N_SPLITS)
    print_results_table("LSTM Univariant — Top 10 per RMSE log-preu", df_uni)

    csv_uni = os.path.join(OUT_DIR, "cv_lstm_uni.csv")
    df_uni.to_csv(csv_uni, index=False)
    print(f"  Resultats complets guardats a: {csv_uni}")

    # ── LSTM Multivariant ────────────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  LSTM MULTIVARIANT")
    print(f"{'─' * 72}")
    df_multi = grid_search_cv(price_train, exog_array=exog_train,
                              model_type='multi', grid=CV_GRID, n_splits=N_SPLITS)
    print_results_table("LSTM Multivariant — Top 10 per RMSE log-preu", df_multi)

    csv_multi = os.path.join(OUT_DIR, "cv_lstm_multi.csv")
    df_multi.to_csv(csv_multi, index=False)
    print(f"  Resultats complets guardats a: {csv_multi}")

    # ── Resum final ──────────────────────────────────────────────────────────
    print_best_summary(df_uni, df_multi)

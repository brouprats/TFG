# ==============================================================================
# main_3.py — Modelització de Sèries Temporals Multi-Índex (SENSE GRÀFICS)
# Índexs: S&P 500, Nasdaq Composite, Euro Stoxx 50, Nikkei 225
# Models: ARIMA(1,1,1) · ARIMAX(1,1,1) · LSTM Univariant · LSTM Multivariant
# Variables exògenes: Inflació · Atur · Tipus d'interès · Partit en el govern
# Intervals de Confiança: 95% — analítics (ARIMA/ARIMAX), MC Dropout (LSTM)
# Comparació LSTM: lookbacks {1, 3, 6
# Sortida: taula d'error per índex — files=models, columnes=mètriques×escala
# ==============================================================================

import os
import math
import random
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    import pandas_datareader.data as web
    HAS_PDR = True
except ImportError:
    HAS_PDR = False
    print("AVÍS: pandas_datareader no instal·lat. Dades FRED no disponibles.")
    print("      Instal·la amb: pip install pandas_datareader")

from DadesPresidents import DFPresidents

# ==============================================================================
# LLAVORS I CONFIGURACIÓ GLOBAL
# ==============================================================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

warnings.filterwarnings("ignore")

START        = "1994-01-01"
END          = "2024-01-01"
LOOK_BACK    = 12
TRAIN_RATIO  = 0.80
MC_PASSES    = 100
ALPHA        = 0.05
FRED_API_KEY = ""

FRED_CACHE_DIR = os.path.join(os.path.dirname(__file__), "fred_cache")

LOOK_BACKS = [1, 3, 6]
EPOCHS     = 30
LSTM_UNITS = 32

GRAFICS_DIR = os.path.join(os.path.dirname(__file__), "grafics_tfg")
os.makedirs(GRAFICS_DIR, exist_ok=True)

_COLORS = {
    'ARIMA':      'purple',
    'ARIMAX':     'darkorange',
    'LSTM Uni':   'steelblue',
    'LSTM Multi': 'crimson',
}

# ==============================================================================
# CONFIGURACIÓ D'ÍNDEXS I SÈRIES FRED
# ==============================================================================
INDEX_CONFIG = {
    'SP500':     {'ticker': '^GSPC',     'region': 'US', 'label': 'S&P 500',          'currency': 'USD'},
    'Nasdaq':    {'ticker': '^IXIC',     'region': 'US', 'label': 'Nasdaq Composite', 'currency': 'USD'},
    'EuroStoxx': {'ticker': '^STOXX50E', 'region': 'EU', 'label': 'Euro Stoxx 50',    'currency': 'EUR'},
    'Nikkei':    {'ticker': '^N225',     'region': 'JP', 'label': 'Nikkei 225',       'currency': 'JPY'},
}

FX_TO_USD = {
    'EUR': 'EURUSD=X',   # USD per EUR
    'JPY': 'JPYUSD=X',   # USD per JPY
}

FRED_SERIES = {
    'US': {
        'inflation':    'CPIAUCSL',
        'unemployment': 'UNRATE',
        'interest':     'FEDFUNDS',
    },
    'EU': {
        'inflation':    'CP0000EZ19M086NEST',
        'unemployment': 'LRHUTTTTEZM156S',
        'interest':     'ECBDFR',
    },
    'JP': {
        'inflation':    'JPNCPIALLMINMEI',
        'unemployment': 'LRHUTTTTJPM156S',
        'interest':     'IRSTCI01JPM156N',
    },
}

# ==============================================================================
# LOOKUPS POLÍTICS PER REGIÓ
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

_eu_data = [
    ("1985-01-07", "1995-01-22", 0),   # Delors (fins 22 gen 1995)
    ("1995-01-23", "1999-09-15", 1),   # Santer (inici 23 gen 1995; en funcions fins 15 set 1999)
    ("1999-09-16", "2004-11-21", 0),   # Prodi
    ("2004-11-22", "2014-10-31", 1),   # Barroso
    ("2014-11-01", "2019-11-30", 1),   # Juncker
    ("2019-12-01", "2030-01-01", 1),   # von der Leyen
]
_eu_df = pd.DataFrame(_eu_data, columns=['Start_Date', 'End_Date', 'Code'])
_eu_df['Start_Date'] = pd.to_datetime(_eu_df['Start_Date'])
_eu_df['End_Date']   = pd.to_datetime(_eu_df['End_Date'])

def _eu_code(date):
    mask = (_eu_df['Start_Date'] <= date) & (_eu_df['End_Date'] >= date)
    row  = _eu_df.loc[mask]
    return int(row.iloc[0]['Code']) if not row.empty else 1

_jp_data = [
    ("1993-01-01", "1993-08-08", 1),   # Miyazawa (LDP)
    ("1993-08-09", "1994-04-27", 0),   # Hosokawa (coalició no-LDP)
    ("1994-04-28", "1994-06-29", 0),   # Hata (no-LDP)
    ("1994-06-30", "1996-01-10", 0),   # Murayama (JSP; fi 10 gen 1996 per evitar solapament)
    ("1996-01-11", "2009-09-15", 1),   # Hashimoto → Aso (LDP continu)
    ("2009-09-16", "2012-12-25", 0),   # Hatoyama → Noda (DPJ)
    ("2012-12-26", "2030-01-01", 1),   # Abe 2n → Ishiba (LDP continu)
]
_jp_df = pd.DataFrame(_jp_data, columns=['Start_Date', 'End_Date', 'Code'])
_jp_df['Start_Date'] = pd.to_datetime(_jp_df['Start_Date'])
_jp_df['End_Date']   = pd.to_datetime(_jp_df['End_Date'])

def _jp_code(date):
    mask = (_jp_df['Start_Date'] <= date) & (_jp_df['End_Date'] >= date)
    row  = _jp_df.loc[mask]
    return int(row.iloc[0]['Code']) if not row.empty else 1

_PARTY_FN = {'US': _us_code, 'EU': _eu_code, 'JP': _jp_code}

def get_political_code(region, date_index):
    fn    = _PARTY_FN[region]
    codes = [fn(d) for d in date_index]
    return pd.Series(codes, index=date_index, name='party_code')


# ==============================================================================
# DESCÀRREGA DE DADES
# ==============================================================================

def download_fx(fx_ticker, start, end):
    raw = yf.download(fx_ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw[('Close', fx_ticker)]
    else:
        close = raw['Close']
    return close.resample('ME').mean()


def download_index(ticker, start, end, currency='USD'):
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw[('Close', ticker)]
    else:
        close = raw['Close']
    monthly = close.resample('ME').mean()
    if currency != 'USD':
        fx_monthly = download_fx(FX_TO_USD[currency], start, end)
        fx_aligned = fx_monthly.reindex(monthly.index).ffill().bfill()
        monthly    = monthly * fx_aligned
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


def build_exog(region, date_index):
    target_period = pd.PeriodIndex(date_index, freq='M')
    series_ids    = FRED_SERIES[region]
    data          = {}
    for var, sid in series_ids.items():
        s_raw = download_fred(sid, START, END)
        if s_raw is not None:
            s_tx      = _transform_exog(var, s_raw).dropna()
            s_aligned = s_tx.reindex(target_period, method='ffill').bfill()
            data[var] = s_aligned.values.astype(float)
        else:
            data[var] = np.zeros(len(date_index))
    data['party_code'] = get_political_code(region, date_index).values.astype(float)
    return pd.DataFrame(data, index=date_index)


# ==============================================================================
# FUNCIONS AUXILIARS
# ==============================================================================

def mape(y_real, y_pred):
    y_real = y_real.flatten()
    y_pred = y_pred.flatten()
    return float(np.mean(np.abs((y_real - y_pred) / y_real)) * 100)


def calcular_metriques(titol, y_real, y_pred):
    y_real = y_real.flatten()
    y_pred = y_pred.flatten()
    rmse_v = math.sqrt(mean_squared_error(y_real, y_pred))
    mae_v  = mean_absolute_error(y_real, y_pred)
    mape_v = mape(y_real, y_pred)
    print(f"    {titol:44s}  RMSE={rmse_v:.5f}  MAE={mae_v:.5f}  MAPE={mape_v:.2f}%")
    return rmse_v, mae_v, mape_v


def ci_coverage(y_real, lower, upper):
    y_real = y_real.flatten()
    inside = np.sum((y_real >= lower.flatten()) & (y_real <= upper.flatten()))
    return 100.0 * inside / len(y_real)


def crear_dataset(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def crear_dataset_multi(dataset, look_back=12):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), :])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def split_train_test(series, ratio=TRAIN_RATIO):
    n     = len(series)
    split = int(n * ratio)
    return series.iloc[:split], series.iloc[split:], split


# ==============================================================================
# DIAGNÒSTIC: ADF · LJUNG-BOX  (només consola, sense gràfics)
# ==============================================================================

def _adf(series):
    s = pd.Series(series).dropna()
    if len(s) < 10:
        return np.nan, np.nan, "n/d"
    if s.nunique() < 2:
        return np.nan, np.nan, "constant"
    stat, pval, *_ = adfuller(s, autolag='AIC')
    concl = "ESTACIONÀRIA" if pval < 0.05 else "no estacionària"
    return stat, pval, concl


def adf_report(label, log_price, exog_df):
    print(f"\n  ADF — {label}")
    print(f"  {'─'*72}")
    print(f"    {'Sèrie':40s}  {'Stat':>10s}  {'p-valor':>10s}  Conclusió")
    print(f"  {'─'*72}")
    rows = [
        ("log(P_t)",    log_price),
        ("Δ log(P_t)",  log_price.diff()),
        ("Δ² log(P_t)", log_price.diff().diff()),
    ]
    for name, s in rows:
        stat, p, c = _adf(s)
        print(f"    {name:40s}  {stat:>10.3f}  {p:>10.4f}  {c}")
    print(f"  {'─'*72}")
    print(f"    Variables exògenes (ja transformades a forma estacionària):")
    for col in exog_df.columns:
        stat, p, c = _adf(exog_df[col])
        print(f"    {col:40s}  {stat:>10.3f}  {p:>10.4f}  {c}")


def ljungbox_report(model_name, residuals, lags=(3,4,6,12), n_params=2):
    s = pd.Series(residuals).dropna()
    print(f"    {model_name}:")
    print(f"      {'lags':>6s}  {'Q-stat':>10s}  {'p-valor':>10s}  Conclusió")
    for lag in lags:
        if lag <= n_params or lag >= len(s):
            continue
        df = acorr_ljungbox(s, lags=[lag], model_df=n_params, return_df=True)
        Q  = float(df['lb_stat'].iloc[0])
        p  = float(df['lb_pvalue'].iloc[0])
        c  = "OK (residus ~ soroll blanc)" if p > 0.05 else "REBUTJAR (autocorr.)"
        print(f"      {lag:>6d}  {Q:>10.3f}  {p:>10.4f}  {c}")


def run_diagnostics(idx_name, p):
    label     = p['label']
    log_price = p['log_price']
    exog_df   = p['exog_df']
    train_ser = p['train_ser']
    exog_tr   = p['exog_train_sc']

    print(f"\n{'─'*72}")
    print(f"  DIAGNÒSTIC — {label}")
    print(f"{'─'*72}")

    adf_report(label, log_price, exog_df)

    print(f"\n  Ljung-Box sobre residus (model_df = p+q = 2):")
    arima_fit  = SARIMAX(train_ser.values, order=(1, 1, 1)).fit(disp=False)
    arimax_fit = SARIMAX(train_ser.values, exog=exog_tr, order=(1, 1, 1)).fit(disp=False)
    ljungbox_report("ARIMA(1,1,1)",  arima_fit.resid)
    ljungbox_report("ARIMAX(1,1,1)", arimax_fit.resid)


# ==============================================================================
# MODEL ARIMA — IC ANALÍTICA 95%
# ==============================================================================

def run_arima(train_series, test_series):
    history         = list(train_series.values)
    preds, los, his = [], [], []
    for t in range(len(test_series)):
        model  = SARIMAX(history, order=(1, 1, 1))
        result = model.fit(disp=False)
        fc     = result.get_forecast(steps=1, alpha=ALPHA)
        pm     = np.asarray(fc.predicted_mean)
        ci     = np.asarray(fc.conf_int(alpha=ALPHA))
        preds.append(float(pm[0]))
        los.append(float(ci[0, 0]))
        his.append(float(ci[0, 1]))
        history.append(float(test_series.iloc[t]))
    return np.array(preds), np.array(los), np.array(his)


# ==============================================================================
# MODEL ARIMAX — IC ANALÍTICA 95%
# ==============================================================================

def run_arimax(train_series, test_series, exog_train_sc, exog_test_sc):
    history_vals    = list(train_series.values)
    history_exog    = list(exog_train_sc)
    preds, los, his = [], [], []
    for t in range(len(test_series)):
        model  = SARIMAX(history_vals, exog=history_exog, order=(1, 1, 1))
        result = model.fit(disp=False)
        next_x = np.array(exog_test_sc[t]).reshape(1, -1)
        fc     = result.get_forecast(steps=1, exog=next_x, alpha=ALPHA)
        pm     = np.asarray(fc.predicted_mean)
        ci     = np.asarray(fc.conf_int(alpha=ALPHA))
        preds.append(float(pm[0]))
        los.append(float(ci[0, 0]))
        his.append(float(ci[0, 1]))
        history_vals.append(float(test_series.iloc[t]))
        history_exog.append(exog_test_sc[t])
    return np.array(preds), np.array(los), np.array(his)


# ==============================================================================
# MODELS LSTM — ARQUITECTURA PARAMETRITZABLE + IC 95% (MC DROPOUT)
# ==============================================================================

def build_lstm_model(look_back, n_features, n_layers=1, dropout=0.2, units=LSTM_UNITS):
    layers = []
    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        kwargs = {'return_sequences': return_seq}
        if i == 0:
            kwargs['input_shape'] = (look_back, n_features)
        layers.append(LSTM(units, **kwargs))
        layers.append(Dropout(dropout))
    layers.append(Dense(1))
    return Sequential(layers)


def _predict_with_uncertainty(model, X_test, mc_passes):
    X_test_tf = tf.constant(X_test, dtype=tf.float32)
    if mc_passes <= 1:
        mean_sc = model(X_test_tf, training=False).numpy().flatten()
        return mean_sc, np.zeros_like(mean_sc)
    mc = np.array([
        model(X_test_tf, training=True).numpy().flatten()
        for _ in range(mc_passes)
    ])
    return mc.mean(axis=0), mc.std(axis=0)


def run_lstm_univariate(price_full, split,
                        look_back=LOOK_BACK, n_layers=1, dropout=0.1,
                        mc_passes=MC_PASSES, epochs=EPOCHS):
    scaler = MinMaxScaler()
    scaler.fit(price_full[:split].reshape(-1, 1))
    all_sc = scaler.transform(price_full.reshape(-1, 1))

    X, y       = crear_dataset(all_sc, look_back)
    test_start = split - look_back
    X_train    = X[:test_start].reshape(-1, look_back, 1)
    y_train    = y[:test_start]
    X_test     = X[test_start:].reshape(-1, look_back, 1)
    y_test     = y[test_start:]

    model = build_lstm_model(look_back, 1, n_layers, dropout)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0,
              validation_split=0.1)

    mean_sc, std_sc = _predict_with_uncertainty(model, X_test, mc_passes)

    def inv(arr):
        return scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

    return inv(mean_sc), inv(mean_sc - 1.96 * std_sc), inv(mean_sc + 1.96 * std_sc), inv(y_test)


def run_lstm_multivariate(price_full, split, exog_train_sc, exog_test_sc,
                          look_back=LOOK_BACK, n_layers=1, dropout=0.1,
                          mc_passes=MC_PASSES, epochs=EPOCHS):
    scaler_y       = MinMaxScaler()
    price_train_sc = scaler_y.fit_transform(price_full[:split].reshape(-1, 1))
    price_test_sc  = scaler_y.transform(price_full[split:].reshape(-1, 1))

    train_data = np.hstack([price_train_sc, exog_train_sc])
    test_data  = np.hstack([price_test_sc,  exog_test_sc])
    all_data   = np.vstack([train_data, test_data])

    X, y            = crear_dataset_multi(all_data, look_back)
    test_start      = split - look_back
    X_train, X_test = X[:test_start], X[test_start:]
    y_train, y_test = y[:test_start], y[test_start:]

    n_features = X.shape[2]
    model = build_lstm_model(look_back, n_features, n_layers, dropout)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0,
              validation_split=0.1)

    mean_sc, std_sc = _predict_with_uncertainty(model, X_test, mc_passes)

    def inv(arr):
        return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()

    return inv(mean_sc), inv(mean_sc - 1.96 * std_sc), inv(mean_sc + 1.96 * std_sc), inv(y_test)


# ==============================================================================
# TAULA D'ERROR — impressió per índex
# Files  : ARIMA · ARIMAX · LSTM Uni (lb=1/3/6) · LSTM Multi (lb=1/3/6)
# Columnes: RMSE(log) · MAE(log) · MAPE(log)%  │  RMSE($) · MAE($) · MAPE($)%
# ==============================================================================

def print_metrics_table(label, results):
    W_MOD = 22
    W_NUM = 10

    row_order = (
        ['ARIMA', 'ARIMAX'] +
        [f'LSTM Uni (lb={lb})'   for lb in LOOK_BACKS] +
        [f'LSTM Multi (lb={lb})' for lb in LOOK_BACKS]
    )

    sep   = "  " + "─" * (W_MOD + 6 * W_NUM)
    total = W_MOD + 6 * W_NUM + 2

    print(f"\n{'═' * total}")
    print(f"  TAULA D'ERROR — {label}")
    print(f"{'═' * total}")
    # Subencapçalaments d'escala
    log_w = 3 * W_NUM
    usd_w = 3 * W_NUM
    print(f"  {'':{W_MOD}}{'─── Log-Preu ───':^{log_w}}{'─── Preu Real (USD) ───':^{usd_w}}")
    print(f"  {'Model':<{W_MOD}}"
          f"{'RMSE(log)':>{W_NUM}}"
          f"{'MAE(log)':>{W_NUM}}"
          f"{'MAPE(log)':>{W_NUM}}"
          f"{'RMSE($)':>{W_NUM}}"
          f"{'MAE($)':>{W_NUM}}"
          f"{'MAPE($)':>{W_NUM}}")
    print(sep)

    for model in row_order:
        if model not in results:
            continue
        rmse_l, mae_l, mape_l = results[model]['log']
        rmse_u, mae_u, mape_u = results[model]['usd']
        print(f"  {model:<{W_MOD}}"
              f"{rmse_l:>{W_NUM}.5f}"
              f"{mae_l:>{W_NUM}.5f}"
              f"{f'{mape_l:.2f}%':>{W_NUM}}"
              f"{rmse_u:>{W_NUM}.2f}"
              f"{mae_u:>{W_NUM}.2f}"
              f"{f'{mape_u:.2f}%':>{W_NUM}}")
    print(sep)


# ==============================================================================
# VISUALITZACIÓ
# ==============================================================================

def _save_fig(fname):
    path = os.path.join(GRAFICS_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Desat: {path}")


_CURRENCY_SYMBOL = {'USD': '$', 'EUR': '€', 'JPY': '¥'}


def plot_model_comparison(idx_name, label, lb, dates_al, y_log,
                          results_log, results_usd, currency='USD'):
    """
    2 subplots (log + moneda local). ARIMA/ARIMAX amb IC sombreada, LSTM sense.
    results_*: dict[nom] = (preds, lower_or_None, upper_or_None, mape_val)
    """
    sym = _CURRENCY_SYMBOL.get(currency, currency)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    y_usd_al = np.exp(y_log)

    for ax, (results, y_real, ylabel) in zip(
        axes,
        [(results_log, y_log,    'Log Preu'),
         (results_usd, y_usd_al, f'Preu ({sym})')]
    ):
        ax.plot(dates_al, y_real, label='Real', color='gray',
                linewidth=2, alpha=0.6)
        for mname, (preds, lower, upper, mape_v) in results.items():
            col = _COLORS.get(mname, 'black')
            ax.plot(dates_al, preds,
                    label=f'{mname} (MAPE={mape_v:.2f}%)',
                    color=col, linewidth=1.5)
            if lower is not None and upper is not None:
                ax.fill_between(dates_al, lower, upper,
                                color=col, alpha=0.12, linewidth=0)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    axes[0].set_title(
        f"{label} — Lookback={lb} — Comparació Models (Log) — IC 95% ARIMA/ARIMAX")
    axes[1].set_title(
        f"{label} — Lookback={lb} — Comparació Models ({sym}) — IC 95% ARIMA/ARIMAX")
    axes[1].set_xlabel("Data")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    _save_fig(f"comp_{idx_name.lower()}_lb{lb}.pdf")


def plot_mape_vs_lookback(idx_name, label, mape_per_lb, currency='USD'):
    """
    MAPE (moneda local) vs lookback per LSTM Uni i Multi + línies de referència ARIMA/ARIMAX.
    mape_per_lb: dict[lb] = {'ARIMA': float, 'ARIMAX': float,
                              'LSTM Uni': float, 'LSTM Multi': float}
    """
    sym = _CURRENCY_SYMBOL.get(currency, currency)
    lbs        = sorted(mape_per_lb.keys())
    mape_arima = mape_per_lb[lbs[0]]['ARIMA']
    mape_arx   = mape_per_lb[lbs[0]]['ARIMAX']
    mape_uni   = [mape_per_lb[lb]['LSTM Uni']   for lb in lbs]
    mape_multi = [mape_per_lb[lb]['LSTM Multi'] for lb in lbs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lbs, mape_uni,   marker='o', linewidth=2,
            label='LSTM Uni',   color=_COLORS['LSTM Uni'])
    ax.plot(lbs, mape_multi, marker='s', linewidth=2,
            label='LSTM Multi', color=_COLORS['LSTM Multi'])
    ax.axhline(mape_arima, linestyle='--', linewidth=1.5,
               color=_COLORS['ARIMA'],
               label=f'ARIMA (MAPE={mape_arima:.2f}%)')
    ax.axhline(mape_arx, linestyle='--', linewidth=1.5,
               color=_COLORS['ARIMAX'],
               label=f'ARIMAX (MAPE={mape_arx:.2f}%)')
    ax.set_xlabel('Lookback (mesos)')
    ax.set_ylabel(f'MAPE (%) — escala {sym}')
    ax.set_title(f'{label} — MAPE per lookback')
    ax.set_xticks(lbs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(f"mape_lb_{idx_name.lower()}.pdf")


def plot_lstm_all_lookbacks(idx_name, label, model_type, dates_al, y_usd, lb_results,
                            ref_preds, ref_lower, ref_upper, ref_name, ref_mape,
                            currency='USD'):
    """
    Escala moneda local: sèrie real + prediccions LSTM (Uni o Multi) per a tots els lookbacks
    + model de referència (ARIMA per Uni, ARIMAX per Multi) amb IC 95% analítica.
    lb_results: dict[lb] = (preds_usd, lower_usd, upper_usd, mape_v)
    """
    sym = _CURRENCY_SYMBOL.get(currency, currency)
    lbs = sorted(lb_results.keys())
    pal = ['steelblue', 'seagreen', 'tomato', 'goldenrod', 'orchid']

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates_al, y_usd, label='Real', color='gray', linewidth=2, alpha=0.6)

    ref_col = _COLORS.get(ref_name, 'black')
    ax.plot(dates_al, ref_preds,
            label=f'{ref_name} (MAPE={ref_mape:.2f}%)',
            color=ref_col, linewidth=1.5, linestyle='--')
    ax.fill_between(dates_al, ref_lower, ref_upper,
                    color=ref_col, alpha=0.12, linewidth=0)

    for i, lb in enumerate(lbs):
        preds_usd, _lower, _upper, mape_v = lb_results[lb]
        col = pal[i % len(pal)]
        ax.plot(dates_al, preds_usd,
                label=f'lb={lb} (MAPE={mape_v:.2f}%)',
                color=col, linewidth=1.5)

    ax.set_ylabel(f'Preu ({sym})')
    ax.set_xlabel('Data')
    ax.set_title(
        f'{label} — LSTM {model_type} — Comparació Lookbacks + {ref_name} IC 95% — Escala {sym}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    _save_fig(f"lstm_{model_type.lower()}_{idx_name.lower()}_alllb.pdf")


# ==============================================================================
# STAGE 1 — Descàrrega de dades per a tots els índexs
# ==============================================================================
print("\n" + "=" * 72)
print("  STAGE 1 — Descàrrega i preparació de dades")
print("=" * 72)

prepared = {}
for idx_name, cfg in INDEX_CONFIG.items():
    ticker   = cfg['ticker']
    region   = cfg['region']
    label    = cfg['label']
    currency = cfg['currency']
    print(f"\n  ▸ {label}  ({ticker})  —  Regió: {region}")

    log_price = download_index(ticker, START, END, currency=currency)
    n         = len(log_price)
    fx_note   = f" → convertit a USD via {FX_TO_USD[currency]}" if currency != 'USD' else ""
    print(f"      Preu:   {n} mesos "
          f"({log_price.index[0].date()} – {log_price.index[-1].date()}){fx_note}")

    exog_df = build_exog(region, log_price.index)
    print(f"      Exog:   {list(exog_df.columns)}")

    train_ser, test_ser, split = split_train_test(log_price, TRAIN_RATIO)
    print(f"      Split:  train={len(train_ser)} / test={len(test_ser)} "
          f"(ratio={TRAIN_RATIO:.0%})")

    exog_all      = exog_df.values.astype(float)
    scaler_exog   = MinMaxScaler()
    exog_train_sc = scaler_exog.fit_transform(exog_all[:split])
    exog_test_sc  = scaler_exog.transform(exog_all[split:])

    prepared[idx_name] = {
        'label':         label,
        'currency':      currency,
        'log_price':     log_price,
        'exog_df':       exog_df,
        'train_ser':     train_ser,
        'test_ser':      test_ser,
        'split':         split,
        'test_dates':    log_price.index[split:],
        'price_full':    log_price.values,
        'exog_train_sc': exog_train_sc,
        'exog_test_sc':  exog_test_sc,
    }


# ==============================================================================
# STAGE 1.5 — Diagnòstic per índex (ADF · Ljung-Box) — sense gràfics
# ==============================================================================
print("\n" + "=" * 72)
print("  STAGE 1.5 — Diagnòstic (ADF · Ljung-Box)")
print("=" * 72)
for idx_name, p in prepared.items():
    run_diagnostics(idx_name, p)


# ==============================================================================
# STAGE 2 — Comparació de models per cada índex
#           ARIMA i ARIMAX: una execució per índex (no depenen del lookback)
#           LSTM Uni i LSTM Multi: una execució per lookback ∈ {1, 3, 6}
# ==============================================================================
print("\n" + "=" * 72)
print("  STAGE 2 — Comparació de models per lookback")
print(f"           Lookbacks LSTM: {LOOK_BACKS}")
print("=" * 72)

all_results = {}

for idx_name, p in prepared.items():
    label    = p['label']
    currency = p['currency']
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")

    # ── ARIMA i ARIMAX: una sola execució per índex ──
    print("  ▸ ARIMA(1,1,1) rolling forecast...")
    sar_p, sar_lo, sar_hi = run_arima(p['train_ser'], p['test_ser'])

    print("  ▸ ARIMAX(1,1,1) rolling forecast...")
    arx_p, arx_lo, arx_hi = run_arimax(p['train_ser'], p['test_ser'],
                                       p['exog_train_sc'], p['exog_test_sc'])

    y_log = p['test_ser'].values
    y_usd = np.exp(y_log)

    print(f"\n  Mètriques ARIMA / ARIMAX:")
    sar_rl = calcular_metriques("ARIMA  (log)", y_log, sar_p)
    arx_rl = calcular_metriques("ARIMAX (log)", y_log, arx_p)
    sar_ru = calcular_metriques("ARIMA  (USD)", y_usd, np.exp(sar_p))
    arx_ru = calcular_metriques("ARIMAX (USD)", y_usd, np.exp(arx_p))

    print(f"  {'─' * 65}")
    print(f"  {'Cobertura IC 95% (escala log)':^63}")
    print(f"    {'ARIMA':38s}  Cobertura={ci_coverage(y_log, sar_lo, sar_hi):.1f}%")
    print(f"    {'ARIMAX':38s}  Cobertura={ci_coverage(y_log, arx_lo, arx_hi):.1f}%")

    all_results[idx_name] = {
        'ARIMA':  {'log': sar_rl, 'usd': sar_ru},
        'ARIMAX': {'log': arx_rl, 'usd': arx_ru},
    }

    # ── LSTM Uni i LSTM Multi: una execució per lookback ──
    mape_per_lb      = {}
    lb_results_uni   = {}
    lb_results_multi = {}
    for lb in LOOK_BACKS:
        print(f"\n  ── Lookback = {lb} ──")
        print(f"  ▸ LSTM Univariant   (MC Dropout, {MC_PASSES} passes, look_back={lb})...")
        lu_p, lu_lo, lu_hi, lu_real = run_lstm_univariate(
            p['price_full'], p['split'], look_back=lb, mc_passes=MC_PASSES)

        print(f"  ▸ LSTM Multivariant (MC Dropout, {MC_PASSES} passes, look_back={lb})...")
        lm_p, lm_lo, lm_hi, lm_real = run_lstm_multivariate(
            p['price_full'], p['split'],
            p['exog_train_sc'], p['exog_test_sc'],
            look_back=lb, mc_passes=MC_PASSES)

        n_min   = min(len(y_log), len(sar_p), len(arx_p), len(lu_p), len(lm_p))
        lu_r    = lu_real[:n_min]
        lm_r    = lm_real[:n_min]
        lu_p_a  = lu_p[:n_min]
        lm_p_a  = lm_p[:n_min]
        lu_lo_a = lu_lo[:n_min]; lu_hi_a = lu_hi[:n_min]
        lm_lo_a = lm_lo[:n_min]; lm_hi_a = lm_hi[:n_min]

        print(f"\n  Mètriques (lookback={lb}):")
        lu_rl = calcular_metriques(f"LSTM Uni   (log, lb={lb})", lu_r,         lu_p_a)
        lm_rl = calcular_metriques(f"LSTM Multi (log, lb={lb})", lm_r,         lm_p_a)
        lu_ru = calcular_metriques(f"LSTM Uni   (USD, lb={lb})", np.exp(lu_r), np.exp(lu_p_a))
        lm_ru = calcular_metriques(f"LSTM Multi (USD, lb={lb})", np.exp(lm_r), np.exp(lm_p_a))

        print(f"    {'LSTM Univariant':38s}  Cobertura={ci_coverage(lu_r, lu_lo_a, lu_hi_a):.1f}%")
        print(f"    {'LSTM Multivariant':38s}  Cobertura={ci_coverage(lm_r, lm_lo_a, lm_hi_a):.1f}%")

        all_results[idx_name][f'LSTM Uni (lb={lb})']   = {'log': lu_rl, 'usd': lu_ru}
        all_results[idx_name][f'LSTM Multi (lb={lb})'] = {'log': lm_rl, 'usd': lm_ru}

        lb_results_uni[lb]   = (np.exp(lu_p_a), np.exp(lu_lo_a), np.exp(lu_hi_a), lu_ru[2])
        lb_results_multi[lb] = (np.exp(lm_p_a), np.exp(lm_lo_a), np.exp(lm_hi_a), lm_ru[2])

        mape_per_lb[lb] = {
            'ARIMA':      sar_ru[2],
            'ARIMAX':     arx_ru[2],
            'LSTM Uni':   lu_ru[2],
            'LSTM Multi': lm_ru[2],
        }
        d_al = p['test_dates'][:n_min]
        results_log = {
            'ARIMA':      (sar_p[:n_min],         sar_lo[:n_min],         sar_hi[:n_min],         sar_rl[2]),
            'ARIMAX':     (arx_p[:n_min],         arx_lo[:n_min],         arx_hi[:n_min],         arx_rl[2]),
            'LSTM Uni':   (lu_p_a,                None,                   None,                   lu_rl[2]),
            'LSTM Multi': (lm_p_a,                None,                   None,                   lm_rl[2]),
        }
        results_usd = {
            'ARIMA':      (np.exp(sar_p[:n_min]),  np.exp(sar_lo[:n_min]), np.exp(sar_hi[:n_min]), sar_ru[2]),
            'ARIMAX':     (np.exp(arx_p[:n_min]),  np.exp(arx_lo[:n_min]), np.exp(arx_hi[:n_min]), arx_ru[2]),
            'LSTM Uni':   (np.exp(lu_p_a),         None,                   None,                   lu_ru[2]),
            'LSTM Multi': (np.exp(lm_p_a),         None,                   None,                   lm_ru[2]),
        }
        plot_model_comparison(idx_name, label, lb, d_al, y_log[:n_min],
                              results_log, results_usd, currency=currency)

    plot_lstm_all_lookbacks(idx_name, label, 'Uni',
                            d_al, y_usd[:n_min], lb_results_uni,
                            ref_preds=np.exp(sar_p[:n_min]),
                            ref_lower=np.exp(sar_lo[:n_min]),
                            ref_upper=np.exp(sar_hi[:n_min]),
                            ref_name='ARIMA', ref_mape=sar_ru[2],
                            currency=currency)
    plot_lstm_all_lookbacks(idx_name, label, 'Multi',
                            d_al, y_usd[:n_min], lb_results_multi,
                            ref_preds=np.exp(arx_p[:n_min]),
                            ref_lower=np.exp(arx_lo[:n_min]),
                            ref_upper=np.exp(arx_hi[:n_min]),
                            ref_name='ARIMAX', ref_mape=arx_ru[2],
                            currency=currency)
    plot_mape_vs_lookback(idx_name, label, mape_per_lb, currency=currency)
    # Taula d'error per a aquest índex (imprimida en acabar tots els lookbacks)
    print_metrics_table(label, all_results[idx_name])


# ==============================================================================
# TAULA GLOBAL FINAL — MAPE (%) per tots els índexs
# Dues sub-taules: escala log-preu i escala preu real ($)
# ==============================================================================

row_order = (
    ['ARIMA', 'ARIMAX'] +
    [f'LSTM Uni (lb={lb})'   for lb in LOOK_BACKS] +
    [f'LSTM Multi (lb={lb})' for lb in LOOK_BACKS]
)

idx_keys   = list(all_results.keys())
idx_labels = [INDEX_CONFIG[k]['label'] for k in idx_keys]
W_MOD      = 22
W_IDX      = 16


def _global_mape_table(scale_key, scale_label):
    total_w = W_MOD + len(idx_keys) * W_IDX + 2
    print(f"\n{'═' * total_w}")
    print(f"  MAPE (%) — {scale_label}")
    print(f"{'═' * total_w}")
    print(f"  {'Model':<{W_MOD}}" +
          "".join(f"{lbl:>{W_IDX}}" for lbl in idx_labels))
    print("  " + "─" * (W_MOD + len(idx_keys) * W_IDX))
    for model in row_order:
        row = f"  {model:<{W_MOD}}"
        for k in idx_keys:
            if model in all_results[k]:
                mape_v = all_results[k][model][scale_key][2]
                row += f"{f'{mape_v:.2f}%':>{W_IDX}}"
            else:
                row += f"{'n/d':>{W_IDX}}"
        print(row)
    print("  " + "─" * (W_MOD + len(idx_keys) * W_IDX))


print("\n\n" + "=" * 72)
print("  RESUM GLOBAL — MAPE (%) per model i índex")
print("=" * 72)
_global_mape_table('log', 'Escala Log-Preu')
_global_mape_table('usd', 'Escala Preu Real ($)')

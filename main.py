import os
import glob
import time
import threading
import joblib
import hashlib
import json
import requests
import random
import logging
import datetime
import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
import tensorflow as tf

from xgboost import XGBRegressor
from threading import Lock
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Optional, Dict, List, Tuple
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator, CCIIndicator
from ta.momentum import ROCIndicator, RSIIndicator, WilliamsRIndicator
from ta.volume import VolumeWeightedAveragePrice
from ta import momentum, trend, volatility, volume
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from sklearn.model_selection import ParameterSampler

# === Konfigurasi Bot ===
TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN")
CHAT_ID          = os.environ.get("CHAT_ID")
ATR_MULTIPLIER   = 2.5
RETRAIN_INTERVAL = 7
AKURASI_THRESHOLD = 0.80
MAE_THRESHOLD = 50.0  # Atur sesuai karakteristik saham (misal 50 rupiah)
BACKUP_CSV_PATH  = "stock_data_backup.csv"
HASH_PATH = "features_hash.json"

# === Daftar Saham ===
STOCK_LIST = [
    "AALI.JK", "ABBA.JK", "ABMM.JK", "ACES.JK", "ACST.JK", "ADHI.JK", "ADMF.JK", "ADMG.JK", "ADRO.JK", "AGII.JK",
    "AGRO.JK", "AKRA.JK", "AKSI.JK", "ALDO.JK", "ALKA.JK", "ALMI.JK", "AMAG.JK", "AMRT.JK", "ANDI.JK", "ANJT.JK",
    "ANTM.JK", "APIC.JK", "APLN.JK", "ARNA.JK", "ARTA.JK", "ASII.JK", "ASJT.JK", "ASRI.JK", "ASSA.JK", "ATIC.JK",
    "AUTO.JK", "BABP.JK", "BACA.JK", "BAEK.JK", "BALI.JK", "BAPA.JK", "BAPI.JK", "BATA.JK", "BBCA.JK", "BBHI.JK",
    "BBKP.JK", "BBNI.JK", "BBRI.JK", "BBTN.JK", "BINA.JK", "BIPP.JK", "BISI.JK", "BJBR.JK", "BJTM.JK", "BKDP.JK",
    "BKSL.JK", "BKSW.JK", "BLTA.JK", "BLTZ.JK", "BLUE.JK", "BMAS.JK", "BMRI.JK", "BMSR.JK", "BMTR.JK", "BNBA.JK",
    "BNGA.JK", "BNII.JK", "BNLI.JK", "BOBA.JK", "BOGA.JK", "BOLT.JK", "BOSS.JK", "BPFI.JK", "BPII.JK", "BPTR.JK",
    "BRAM.JK", "BRIS.JK", "BRMS.JK", "BRPT.JK", "BSDE.JK", "BSSR.JK", "BTEK.JK", "BTEL.JK", "BTON.JK", "BTPN.JK",
    "BTPS.JK", "BUDI.JK", "BUVA.JK", "BVSN.JK", "BYAN.JK", "CAKK.JK", "CAMP.JK", "CANI.JK", "CARS.JK", "CASA.JK",
    "CASH.JK", "CBMF.JK", "CEKA.JK", "CENT.JK", "CFIN.JK", "CINT.JK", "CITA.JK", "CITY.JK", "CLAY.JK", "CLEO.JK",
    "CLPI.JK", "CMNP.JK", "CMRY.JK", "CMPP.JK", "CNKO.JK", "CNTX.JK", "COCO.JK", "COWL.JK", "CPIN.JK", "CPRO.JK",
    "CSAP.JK", "CSIS.JK", "CTRA.JK", "CTTH.JK", "DEAL.JK", "DEFI.JK", "DEPO.JK", "DGIK.JK", "DIGI.JK", "DILD.JK",
    "DIVA.JK", "DKFT.JK", "DLTA.JK", "DMAS.JK", "DNAR.JK", "DOID.JK", "DSSA.JK", "DUCK.JK", "DUTI.JK", "DVLA.JK",
    "DYAN.JK", "EAST.JK", "ECII.JK", "EDGE.JK", "EKAD.JK", "ELSA.JK", "EMDE.JK", "EMTK.JK", "ENRG.JK", "ENZO.JK",
    "EPAC.JK", "ERA.JK", "ERAA.JK", "ESSA.JK", "ESTA.JK", "FAST.JK", "FASW.JK", "FILM.JK", "FISH.JK", "FITT.JK",
    "FLMC.JK", "FMII.JK", "FOOD.JK", "FORU.JK", "FPNI.JK", "GAMA.JK", "GEMS.JK", "GGRM.JK", "GJTL.JK", "GLVA.JK",
    "GOOD.JK", "GPRA.JK", "GSMF.JK", "GZCO.JK", "HDTX.JK", "HERO.JK", "HEXA.JK", "HITS.JK", "HKMU.JK", "HMSP.JK",
    "HOKI.JK", "HRUM.JK", "ICBP.JK", "IDPR.JK", "IFII.JK", "INAF.JK", "INAI.JK", "INCF.JK", "INCI.JK", "INCO.JK",
    "INDF.JK", "INDY.JK", "INKP.JK", "INPP.JK", "INTA.JK", "INTD.JK", "INTP.JK", "IPCC.JK", "IPCM.JK", "IPOL.JK",
    "IPTV.JK", "IRRA.JK", "ISAT.JK", "ITMG.JK", "JAST.JK", "JAWA.JK", "JGLE.JK", "JKON.JK", "JPFA.JK", "JSMR.JK",
    "KAEF.JK", "KARW.JK", "KBLI.JK", "KBLM.JK", "KDSI.JK", "KIAS.JK", "KIJA.JK", "KINO.JK", "KLBF.JK", "KMTR.JK",
    "LEAD.JK", "LIFE.JK", "LINK.JK", "LPKR.JK", "LPPF.JK", "LUCK.JK", "MAIN.JK", "MAPB.JK", "MAPA.JK", "MASA.JK",
    "MCAS.JK", "MDKA.JK", "MEDC.JK", "MFIN.JK", "MIDI.JK", "MIRA.JK", "MITI.JK", "MKNT.JK", "MLPL.JK", "MLPT.JK",
    "MNCN.JK", "MPPA.JK", "MPRO.JK", "MTDL.JK", "MYOR.JK", "NATO.JK", "NELY.JK", "NFCX.JK", "NISP.JK", "NRCA.JK",
    "OKAS.JK", "OMRE.JK", "PANI.JK", "PBID.JK", "PCAR.JK", "PDES.JK", "PEHA.JK", "PGAS.JK", "PJAA.JK", "PMJS.JK",
    "PNBN.JK", "PNLF.JK", "POLA.JK", "POOL.JK", "PPGL.JK", "PPRO.JK", "PSSI.JK", "PTBA.JK", "PTIS.JK", "PWON.JK",
    "RAJA.JK", "RDTX.JK", "REAL.JK", "RICY.JK", "RIGS.JK", "ROTI.JK", "SAME.JK", "SAPX.JK", "SCCO.JK", "SCMA.JK",
    "SIDO.JK", "SILO.JK", "SIMP.JK", "SIPD.JK", "SMBR.JK", "SMCB.JK", "SMDR.JK", "SMGR.JK", "SMKL.JK", "SMRA.JK",
    "SMSM.JK", "SOCI.JK", "SQMI.JK", "SRAJ.JK", "SRTG.JK", "STAA.JK", "STTP.JK", "TALF.JK", "TARA.JK", "TBIG.JK",
    "TCID.JK", "TIFA.JK", "TINS.JK", "TKIM.JK", "TLKM.JK", "TOTO.JK", "TPIA.JK", "TRIM.JK", "TURI.JK", "ULTJ.JK",
    "UNIC.JK", "UNTR.JK", "UNVR.JK", "WIKA.JK", "WSBP.JK", "WSKT.JK", "YPAS.JK", "ZINC.JK"
]

# === Logging Setup ===
# === Logging Setup ===
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler   = RotatingFileHandler("trading.log", maxBytes=5*1024*1024, backupCount=3)
log_handler.setFormatter(log_formatter)
logging.getLogger().addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

# Cek dan update header file CSV
def check_and_update_csv_header():
    log_path = "prediksi_log.csv"
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path, header=None)
        # Periksa jika header tidak sesuai
        if df_log.shape[1] != 6:  # 6 kolom termasuk 'model'
            logging.warning("Header CSV tidak sesuai, memperbarui header.")
            df_log.columns = ["ticker", "tanggal", "predicted_price", "upper_bound", "lower_bound", "model"]
            df_log.to_csv(log_path, index=False, header=True)
        else:
            logging.info("Header CSV sudah sesuai.")
    else:
        logging.warning("File prediksi_log.csv tidak ditemukan.")

check_and_update_csv_header()

def log_prediction(ticker: str, tanggal: str, pred_high: float, pred_low: float, harga_awal: float, model_name: str = "LightGBM-v1"):
    # Tulis ke CSV
    with open("prediksi_log.csv", "a") as f:
        f.write(f"{ticker},{tanggal},{harga_awal},{pred_high},{pred_low},{model_name}\n")
    
    # Tulis ke log file
    logging.info(
        f"[{model_name}] Prediksi {ticker} pada {tanggal} | Harga Awal: {harga_awal:.2f} | "
        f"High: {pred_high:.2f} | Low: {pred_low:.2f}"
    )

# === Lock untuk Thread-Safe Model Saving ===
model_save_lock = threading.Lock()

# === Fungsi Kirim Telegram ===
def send_telegram_message(message: str):
    url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        logging.error(f"Telegram error: {e}")

# === Ambil & Validasi Data Saham ===
def get_stock_data(ticker: str) -> pd.DataFrame:
    try:
        # Gunakan 60 hari jika pakai interval 1 jam
        stock = yf.Ticker(ticker)
        df = stock.history(period="180d", interval="1h")

        required_cols = ["High", "Low", "Close", "Volume"]
        if df is not None and not df.empty and all(col in df.columns for col in required_cols) and len(df) >= 200:
            df["ticker"] = ticker
            return df

        logging.warning(f"{ticker}: Data kosong/kurang atau kolom tidak lengkap.")
        logging.debug(f"{ticker}: Kolom tersedia: {df.columns.tolist()}")
    except Exception as e:
        logging.error(f"Error mengambil data {ticker}: {e}")
    return None

def validate_raw_data(df: pd.DataFrame, required_columns=None, min_rows=100, ticker: str = "UNKNOWN") -> bool:
    if required_columns is None:
        required_columns = ["Open", "High", "Low", "Close", "Volume"]

    # Cek apakah data kosong
    if df.empty:
        logging.warning(f"{ticker}: Data kosong.")
        return False

    # Cek apakah kolom penting ada semua
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logging.warning(f"{ticker}: Kolom hilang: {missing_cols}")
        return False

    # Cek apakah data cukup untuk indikator berbasis window
    if len(df) < min_rows:
        logging.warning(f"{ticker}: Data kurang dari {min_rows} baris (saat ini: {len(df)}).")
        return False

    # Cek apakah kolom penting isinya valid
    if df[required_columns].isnull().all().any():
        logging.warning(f"{ticker}: Salah satu kolom utama hanya berisi NaN.")
        return False

    return True

# === Hitung Indikator ===
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    from ta.trend import (
        EMAIndicator, SMAIndicator, ADXIndicator, MACD,
        CCIIndicator
    )
    from ta.momentum import (
        RSIIndicator, ROCIndicator, WilliamsRIndicator
    )
    from ta.volatility import (
        AverageTrueRange, BollingerBands
    )
    from ta.volume import (
        VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
    )
    from ta.momentum import StochasticOscillator
    from sklearn.linear_model import LinearRegression

    HOURS_PER_DAY = 5

    # Pastikan timezone benar
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("Asia/Jakarta")
    else:
        df.index = df.index.tz_convert("Asia/Jakarta")

    if df.empty or len(df) < 2 * HOURS_PER_DAY:
        return df

    # === Fitur waktu & harian ===
    df["hour"] = df.index.hour
    df["is_opening_hour"] = (df["hour"] == 9).astype(int)
    df["is_closing_hour"] = (df["hour"] == 15).astype(int)
    df["daily_avg"] = df["Close"].rolling(HOURS_PER_DAY).mean()
    df["daily_std"] = df["Close"].rolling(HOURS_PER_DAY).std()
    df["daily_range"] = df["High"].rolling(HOURS_PER_DAY).max() - df["Low"].rolling(HOURS_PER_DAY).min()
    df["return_prev_day"] = df["Close"].pct_change(periods=HOURS_PER_DAY)
    df["gap_close"] = df["Open"] - df["Close"].shift(HOURS_PER_DAY)
    df["zscore"] = (df["Close"] - df["daily_avg"]) / df["daily_std"]

    # === Volatilitas dan range ===
    df["ATR"] = volatility.AverageTrueRange(df["High"], df["Low"], df["Close"], window=5).average_true_range()

    # === Volume: OBV & VWAP ===
    df["OBV"] = OnBalanceVolumeIndicator(df["Close"], df["Volume"]).on_balance_volume()
    df["OBV_MA_5"] = df["OBV"].rolling(window=5).mean()
    df["OBV_MA_10"] = df["OBV"].rolling(window=10).mean()
    df["OBV_Diff"] = df["OBV"].diff()
    df["OBV_vs_MA"] = df["OBV"] - df["OBV_MA_10"]
    df["VWAP"] = VolumeWeightedAveragePrice(df["High"], df["Low"], df["Close"], df["Volume"]).volume_weighted_average_price()

    # === Trend: EMA, SMA, Slope, MACD ===
    for w in [5, 10, 15, 20, 25, 50]:
        df[f"EMA_{w}"] = EMAIndicator(df["Close"], window=w).ema_indicator()
        df[f"SMA_{w}"] = SMAIndicator(df["Close"], window=w).sma_indicator()

    macd = MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Hist"] = macd.macd_diff()

    def calc_slope(series):
        y = series.values.reshape(-1, 1)
        x = np.arange(len(series)).reshape(-1, 1)
        if len(series.dropna()) < len(series):
            return np.nan
        model = LinearRegression().fit(x, y)
        return model.coef_[0][0]

    df["slope_5"] = df["Close"].rolling(window=5).apply(calc_slope)
    df["Body"] = abs(df["Close"] - df["Open"])
    df["HA_Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    # === Momentum: RSI, ROC, Williams, Stoch, PROC ===
    df["RSI"] = RSIIndicator(df["Close"], window=5).rsi()
    df["ROC"] = ROCIndicator(df["Close"], window=5).roc()
    df["Momentum"] = df["ROC"]  # alias, opsional
    df["PROC_3"] = df["Close"].pct_change(periods=3)
    df["WilliamsR"] = WilliamsRIndicator(df["High"], df["Low"], df["Close"], lbp=5).williams_r()
    stoch = StochasticOscillator(df["High"], df["Low"], df["Close"], window=5, smooth_window=3)
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # === Support, Resistance, CCI, ADX ===
    df["Support"] = df["Low"].rolling(window=5).min()
    df["Resistance"] = df["High"].rolling(window=5).max()
    df["Support_25"] = df["Low"].rolling(window=25).min()
    df["Resistance_25"] = df["High"].rolling(window=25).max()
    df["CCI"] = CCIIndicator(df["High"], df["Low"], df["Close"], window=5).cci()
    df["ADX"] = ADXIndicator(df["High"], df["Low"], df["Close"], window=5).adx()

    # === Bollinger Bands ===
    bb = BollingerBands(df["Close"], window=5)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()

    # === Fibonacci Pivot & Retracement ===
    pivot = (df["High"] + df["Low"] + df["Close"]) / 3
    range_hl = df["High"] - df["Low"]
    df["Pivot"] = pivot
    df["Fib_R1"] = pivot + range_hl * 0.382
    df["Fib_R2"] = pivot + range_hl * 0.618
    df["Fib_R3"] = pivot + range_hl * 1.000
    df["Fib_S1"] = pivot - range_hl * 0.382
    df["Fib_S2"] = pivot - range_hl * 0.618
    df["Fib_S3"] = pivot - range_hl * 1.000

    # === Target (label prediksi) ===
    df["future_high"] = df["High"].shift(-HOURS_PER_DAY).rolling(HOURS_PER_DAY).max()
    df["future_low"] = df["Low"].shift(-HOURS_PER_DAY).rolling(HOURS_PER_DAY).min()

    return df.dropna()

def evaluate_model(model, X, y_true):
    try:
        # Handle input reshaping jika model adalah LSTM dari Keras
        is_keras = "tensorflow" in str(type(model)).lower() or "keras" in str(type(model)).lower()

        if is_keras:
            if isinstance(X, pd.DataFrame):
                X = X.values
            X = X.reshape((X.shape[0], X.shape[1], 1))

        # Prediksi
        y_pred = model.predict(X)
        if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        # Hitung metrik evaluasi
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # Logging fallback jika belum diset
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(level=logging.INFO)

        logging.info(f"Model Evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        return rmse, mae

    except Exception as e:
        logging.error(f"Gagal evaluasi model: {e}")
        return None, None

# === Training LightGBM ===
def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, n_estimators: int = 500, learning_rate: float = 0.05, early_stopping_rounds: Optional[int] = 50, random_state: int = 42) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)
    else:
        model.fit(X_train, y_train)

    # Evaluate Model
    evaluate_model(model, X_train, y_train)
    
    return model

# === Training XGBoost ===
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None, n_estimators: int = 500, learning_rate: float = 0.05, early_stopping_rounds: Optional[int] = 50, random_state: int = 42) -> XGBRegressor:
    model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    
    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=early_stopping_rounds, verbose=False)
    else:
        model.fit(X_train, y_train)

    # Evaluate Model
    evaluate_model(model, X_train, y_train)
    
    return model

# === Training LSTM ===
def train_lstm(
    X: pd.DataFrame,
    y: pd.Series,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    epochs: int = 77,
    batch_size: int = 32,
    verbose: int = 1
) -> Sequential:
    X_arr = X.values.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_arr, y.values, epochs=epochs, batch_size=batch_size, verbose=verbose)

    evaluate_model(model, X_arr, y.values)  # Pastikan fungsi bisa handle numpy array

    return model

def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    logging.info(f"Cross-validation scores: {scores}")
    
# === Hitung Probabilitas Arah Prediksi ===
def calculate_probability(model, X: pd.DataFrame, y_true: pd.Series) -> float:
    if "Close" not in X.columns:
        raise ValueError("'Close' column is required in input features (X).")
    if len(X) != len(y_true):
        raise ValueError("Length of X and y_true must match.")

    y_pred = model.predict(X)
    y_pred_series = pd.Series(y_pred, index=X.index)
    close_price = X["Close"]

    correct_dir = ((y_pred_series > close_price) & (y_true > close_price)) | \
                  ((y_pred_series < close_price) & (y_true < close_price))
    correct_dir = correct_dir.dropna()

    if len(correct_dir) == 0:
        return 0.0

    return correct_dir.sum() / len(correct_dir)

# Fungsi load_or_train_model
# === Fungsi Utama Load or Train Model ===
def load_or_train_model(path, train_func, X, y, model_type="joblib"):
    if os.path.exists(path):
        model = joblib.load(path) if model_type == "joblib" else tf.keras.models.load_model(path)
        logging.info(f"Loaded model from {path}")
    else:
        model = train_func(X, y)
        with model_save_lock:
            if model_type == "joblib":
                joblib.dump(model, path)
            else:
                model.save(path)
        logging.info(f"Trained & saved model to {path}")
    return model

def train_and_select_best_model(ticker: str, X: pd.DataFrame, y: pd.Series) -> str:
    logging.info(f"Training semua model untuk {ticker}...")

    models = {}
    scores = {}

    # LightGBM
    model_lgb = train_lightgbm(X, y)
    rmse_lgb, mae_lgb = evaluate_model(model_lgb, X, y)
    models["lightgbm"] = model_lgb
    scores["lightgbm"] = mae_lgb

    # XGBoost
    model_xgb = train_xgboost(X, y)
    rmse_xgb, mae_xgb = evaluate_model(model_xgb, X, y)
    models["xgboost"] = model_xgb
    scores["xgboost"] = mae_xgb

    # LSTM
    model_lstm = train_lstm(X, y, verbose=0)
    rmse_lstm, mae_lstm = evaluate_model(model_lstm, X, y)
    models["lstm"] = model_lstm
    scores["lstm"] = mae_lstm

    # Pilih model terbaik (berdasarkan MAE)
    best_model_type = min(scores, key=scores.get)
    best_model = models[best_model_type]
    logging.info(f"Model terbaik untuk {ticker}: {best_model_type.upper()} (MAE: {scores[best_model_type]:.4f})")

    # Simpan model
    if best_model_type == "lstm":
        best_model.save(f"model_lstm_best_{ticker}.keras")
    else:
        joblib.dump(best_model, f"model_{best_model_type}_best_{ticker}.pkl")

    # Simpan info model terbaik
    with open(f"model_best_type_{ticker}.txt", "w") as f:
        f.write(best_model_type)

    return best_model_type

# === Hyperparameter Tuning untuk XGBoost ===
def tune_xgboost_hyperparameters_optuna(X_train, y_train, n_trials=50):
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': 42
        }

        model = xgb.XGBRegressor(**params)
        score = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
        return -score.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_model = xgb.XGBRegressor(**study.best_params)
    best_model.fit(X_train, y_train)

    return best_model, study.best_params

#=== Pelatihan Final XGBoost ===
def train_final_xgb_with_best_params(X_train, y_train, best_params):
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    return model
    
#=== Pelatihan Otomatis XGBoost ===
def train_xgb_with_auto_tuning(X, y, ticker, force_retrain=False, n_trials=50):
    import os
    import json

    param_path = f"xgb_best_params_{ticker}.json"
    if os.path.exists(param_path) and not force_retrain:
        with open(param_path, "r") as f:
            best_params = json.load(f)
        logging.info(f"Parameter XGBoost {ticker} dimuat dari file.")
    else:
        logging.info(f"Melakukan tuning hyperparameter XGBoost untuk {ticker}...")
        model, best_params = tune_xgboost_hyperparameters_optuna(X, y, n_trials=n_trials)
        with open(param_path, "w") as f:
            json.dump(best_params, f, indent=2)
        logging.info(f"Parameter terbaik disimpan ke {param_path}")

    model = train_final_xgb_with_best_params(X, y, best_params)
    joblib.dump(model, f"model_xgb_{ticker}.joblib")
    logging.info(f"Model XGBoost {ticker} selesai dilatih dan disimpan.")
    
#=== Build Model XGBoost (Basic Wrapper) ===
def build_xgb_model(X_train, y_train, params=None):
    if params is None:
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model
    
# === Hyperparameter Tuning untuk LightGBM ===
def tune_lightgbm_hyperparameters_optuna(X_train, y_train, n_trials=50):
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': trial.suggest_int('num_leaves', 7, 255),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'verbosity': -1
        }
        # GOSS tidak cocok dengan subsample
        if params['boosting_type'] == 'goss':
            params['subsample'] = 1.0

        model = lgb.LGBMRegressor(**params, random_state=42)
        score = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
        return -score.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_model = lgb.LGBMRegressor(**study.best_params, random_state=42)
    best_model.fit(X_train, y_train)

    return best_model, study.best_params

#=== Train Final LightGBM dengan Best Params ===
def train_final_lgbm_with_best_params(X_train, y_train, best_params):
    model = lgb.LGBMRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    return model
    
#=== Auto Tuning + Simpan/Load Parameter LightGBM ===
def train_lgbm_with_auto_tuning(X, y, ticker, force_retrain=False, n_trials=50):
    import os
    import json

    param_path = f"lgbm_best_params_{ticker}.json"

    if os.path.exists(param_path) and not force_retrain:
        with open(param_path, "r") as f:
            best_params = json.load(f)
        logging.info(f"Parameter LightGBM {ticker} dimuat dari file.")
    else:
        logging.info(f"Melakukan tuning hyperparameter LightGBM untuk {ticker}...")
        model, best_params = tune_lightgbm_hyperparameters_optuna(X, y, n_trials=n_trials)
        with open(param_path, "w") as f:
            json.dump(best_params, f, indent=2)
        logging.info(f"Parameter terbaik LightGBM disimpan ke {param_path}")

    model = train_final_lgbm_with_best_params(X, y, best_params)
    joblib.dump(model, f"model_lgbm_{ticker}.joblib")
    logging.info(f"Model LightGBM {ticker} selesai dilatih dan disimpan.")
    
#=== Build LightGBM Model (Versi Ringan) ===
def build_lgbm_model(X_train, y_train, params=None):
    if params is None:
        params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6}
    model = lgb.LGBMRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    return model

# === Hyperparameter Tuning untuk LSTM ===
def tune_lstm_hyperparameters_optuna(X, y, n_trials=30):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
    from tensorflow.keras.callbacks import EarlyStopping

    def objective(trial):
        # Suggest hyperparameters
        best_params = {
            "lstm_units": trial.suggest_int("lstm_units", 32, 128, step=32),
            "dense_units": trial.suggest_int("dense_units", 16, 64, step=16),
            "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
            "recurrent_dropout": trial.suggest_float("recurrent_dropout", 0.0, 0.5),
            "activation": trial.suggest_categorical("activation", ["tanh", "relu"]),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "rmsprop", "nadam"]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64])
        }

        # Split & scale
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        X_train_r = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
        X_val_r = X_val_scaled.reshape((X_val_scaled.shape[0], X_val_scaled.shape[1], 1))

        # Build model
        model = Sequential()
        model.add(LSTM(
            best_params["lstm_units"],
            activation=best_params["activation"],
            recurrent_dropout=best_params["recurrent_dropout"],
            input_shape=(X_train_r.shape[1], 1)
        ))
        model.add(Dropout(best_params["dropout_rate"]))
        model.add(Dense(best_params["dense_units"], activation=best_params["activation"]))
        model.add(Dense(1))

        # Compile
        optimizer = {
            "adam": Adam,
            "rmsprop": RMSprop,
            "nadam": Nadam
        }[best_params["optimizer"]](learning_rate=best_params["learning_rate"])
        model.compile(loss="mean_absolute_error", optimizer=optimizer)

        # Fit
        early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)
        model.fit(X_train_r, y_train, validation_data=(X_val_r, y_val),
                  epochs=50, batch_size=best_params["batch_size"], verbose=0, callbacks=[early_stop])

        val_loss = model.evaluate(X_val_r, y_val, verbose=0)
        return val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
    
def train_final_lstm_with_best_params(X, y, best_params, epochs=50):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam, RMSprop, Nadam

    X_reshaped = np.reshape(X.values, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(
        best_params["lstm_units"],
        activation=best_params["activation"],
        recurrent_dropout=best_params["recurrent_dropout"],
        input_shape=(X_reshaped.shape[1], X_reshaped.shape[2]),
        return_sequences=False
    ))
    model.add(Dropout(best_params["dropout_rate"]))
    model.add(Dense(best_params["dense_units"], activation=best_params["activation"]))
    model.add(Dense(1))

    # Pilih optimizer sesuai hasil
    optimizers = {
        "adam": Adam,
        "rmsprop": RMSprop,
        "nadam": Nadam
    }
    optimizer_class = optimizers[best_params["optimizer"]]
    optimizer = optimizer_class(learning_rate=best_params["learning_rate"])

    model.compile(loss="mean_absolute_error", optimizer=optimizer)

    model.fit(X_reshaped, y, epochs=epochs, batch_size=best_params["batch_size"], verbose=1)

    return model
    
def train_lstm_with_auto_tuning(X, y, ticker, epochs=50, force_retrain=False):
    import os
    import json

    param_path = f"lstm_best_params_{ticker}.json"

    # Gunakan file param jika sudah ada, kecuali force retrain
    if os.path.exists(param_path) and not force_retrain:
        with open(param_path, "r") as f:
            best_params = json.load(f)
        logging.info(f"Parameter terbaik LSTM {ticker} dimuat dari file.")
    else:
        logging.info(f"Melakukan tuning hyperparameter LSTM untuk {ticker}...")
        best_params = tune_lstm_hyperparameters_optuna(X, y, n_trials=30)

        with open(param_path, "w") as f:
            json.dump(best_params, f, indent=2)
        logging.info(f"Parameter terbaik LSTM {ticker} disimpan ke {param_path}")

    model = train_final_lstm_with_best_params(X, y, best_params, epochs=epochs)
    model.save(f"model_lstm_{ticker}.keras")
    logging.info(f"Model LSTM {ticker} selesai dilatih dan disimpan.")

# Fungsi untuk membangun model LSTM
def build_lstm_model(X_train, lstm_units, dropout_rate, dense_units, optimizer):
    model = Sequential()
    model.add(LSTM(lstm_units, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dense(1))  # Output layer untuk prediksi harga

    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model
    
def save_best_params(ticker: str, model_type: str, params: dict):
    os.makedirs("params", exist_ok=True)
    with open(f"params/{ticker}_{model_type}.json", "w") as f:
        json.dump(params, f, indent=2)

def load_best_params(ticker: str, model_type: str) -> Optional[dict]:
    path = f"params/{ticker}_{model_type}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None
    
# === Fungsi Kirim Alert ===
def send_alert(message):
    logging.error(f"ALERT: {message}")

# === Fungsi Simpan dan Load Preprocessed Data ===
def save_preprocessed_data(df, path):
    df.to_csv(path, index=False)
    logging.info(f"Data diproses dan disimpan ke {path}")

def load_preprocessed_data(path):
    return pd.read_csv(path)

# === Fungsi Save Model dengan Versioning ===
def save_model_with_versioning(model, path):
    versioned_path = f"{path}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pkl"
    joblib.dump(model, versioned_path)
    logging.info(f"Model disimpan dengan versi: {versioned_path}")

def get_feature_hash(features: list[str]) -> str:
    features_str = ",".join(sorted(features))
    return hashlib.md5(features_str.encode()).hexdigest()

def check_and_reset_model_if_needed(ticker: str, current_features: list[str]):
    current_hash = get_feature_hash(current_features)

    try:
        with open(HASH_PATH, "r") as f:
            saved_hashes = json.load(f)
    except FileNotFoundError:
        saved_hashes = {}

    if saved_hashes.get(ticker) != current_hash:
        logging.info(f"{ticker}: Struktur fitur berubah — melakukan reset model")

        # Hapus model XGBoost jika ada
        for suffix in ["high", "low"]:
            model_path_xgb = f"model_xgb_{suffix}_{ticker}.pkl"
            if os.path.exists(model_path_xgb):
                os.remove(model_path_xgb)
                logging.info(f"{ticker}: Model XGBoost '{suffix}' dihapus")

        # Hapus model LightGBM jika ada
        for suffix in ["high", "low"]:
            model_path_lgb = f"model_lgb_{suffix}_{ticker}.pkl"
            if os.path.exists(model_path_lgb):
                os.remove(model_path_lgb)
                logging.info(f"{ticker}: Model LightGBM '{suffix}' dihapus")

        # Hapus model LSTM jika ada
        lstm_path = f"model_lstm_{ticker}.keras"
        if os.path.exists(lstm_path):
            os.remove(lstm_path)
            logging.info(f"{ticker}: Model LSTM dihapus")

        # Simpan hash baru
        saved_hashes[ticker] = current_hash
        with open(HASH_PATH, "w") as f:
            json.dump(saved_hashes, f, indent=2)

        logging.info(f"{ticker}: Model akan dilatih ulang dengan struktur fitur terbaru")
    else:
        logging.debug(f"{ticker}: Struktur fitur sama — model tidak di-reset")
        
# Konstanta threshold (letakkan di atas fungsi analyze_stock)
MIN_PRICE = 50
MAX_PRICE = 100000
MIN_VOLUME = 100000
MIN_VOLATILITY = 0.005
MIN_PROB = 0.8

def is_stock_eligible(price, avg_volume, atr, ticker):
    if price < MIN_PRICE:
        logging.info(f"{ticker} dilewati: harga terlalu rendah ({price:.2f})")
        return False
    if price > MAX_PRICE:
        logging.info(f"{ticker} dilewati: harga terlalu tinggi ({price:.2f})")
        return False
    if avg_volume < MIN_VOLUME:
        logging.info(f"{ticker} dilewati: volume terlalu rendah ({avg_volume:.0f})")
        return False
    if (atr / price) < MIN_VOLATILITY:
        logging.info(f"{ticker} dilewati: volatilitas terlalu rendah (ATR={atr:.4f})")
        return False
    return True

def prepare_features_and_labels(df, features):
    df = df.dropna(subset=features + ["future_high", "future_low"])
    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]
    return train_test_split(X, y_high, y_low, test_size=0.2, random_state=42)

def load_or_train_model(path, train_fn, X_train, y_train, model_type="joblib"):
    if os.path.exists(path):
        model = joblib.load(path) if model_type == "joblib" else load_model(path)
        logging.info(f"Loaded model from {path}")
    else:
        model = train_fn(X_train, y_train)
        with model_save_lock:
            if model_type == "joblib":
                joblib.dump(model, path)
            else:
                model.save(path)
        logging.info(f"Trained & saved model to {path}")
    return model
    
def get_latest_close(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1d", interval="1h")

        if df is None or df.empty:
            logging.warning(f"{ticker}: Data daily kosong saat ambil harga terbaru.")
            return None

        close_price = df["Close"].dropna()
        if close_price.empty:
            logging.warning(f"{ticker}: Kolom Close kosong di data daily.")
            return None

        return close_price.iloc[-1]

    except Exception as e:
        logging.error(f"{ticker}: Gagal ambil harga terbaru - {e}")
        return None

def analyze_stock(ticker: str):
    df = get_stock_data(ticker)
    if df is None or df.empty:
        logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
        return None

    df = calculate_indicators(df)

    # Pastikan kolom dan nilainya ada sebelum lanjut
    if "ATR" not in df.columns or df["ATR"].replace(0, np.nan).dropna().empty:
        logging.warning(f"{ticker}: ATR tidak valid (semua nol atau kosong) setelah kalkulasi.")
        return None

    atr = df["ATR"].dropna().iloc[-1]
    required_columns = ["High", "Low", "Close", "Volume", "ATR"]
    if not all(col in df.columns for col in required_columns):
        logging.error(f"{ticker}: Kolom yang diperlukan tidak lengkap.")
        logging.debug(f"{ticker}: Kolom tersedia: {df.columns.tolist()}")
        return None

    # Gunakan harga terbaru dari daily
    price = get_latest_close(ticker)

    # Fallback hanya jika df tidak kosong
    if price is None:
        if df is not None and not df.empty and "Close" in df.columns:
            price = df["Close"].dropna().iloc[-1] if not df["Close"].dropna().empty else None
        else:
            logging.warning(f"{ticker}: Data fallback juga kosong.")
            return None

    # Hentikan jika harga tetap tidak bisa didapatkan
    if price is None:
        logging.warning(f"{ticker}: Tidak bisa mendapatkan harga terbaru.")
        return None

    avg_volume = df["Volume"].tail(20).mean()

    if not is_stock_eligible(price, avg_volume, atr, ticker):
        logging.debug(f"{ticker}: Tidak memenuhi kriteria awal.")
        return None

    features = [
        "Close",
        # === Fitur Waktu ===
        "is_opening_hour", "is_closing_hour", "return_prev_day", "gap_close",
        "daily_avg", "daily_std", "daily_range", "zscore",

        # === Volatilitas ===
        "ATR",

        # === Volume ===
        "OBV", "OBV_MA_5", "OBV_MA_10", "OBV_Diff", "OBV_vs_MA", "VWAP",

        # === Trend ===
        "EMA_5", "EMA_10", "EMA_15", "EMA_20", "EMA_25", "EMA_50",
        "SMA_5", "SMA_10", "SMA_15", "SMA_20", "SMA_25", "SMA_50",
        "MACD", "MACD_Hist", "slope_5", "Body", "HA_Close",

        # === Momentum ===
        "RSI", "ROC", "Momentum", "PROC_3", "WilliamsR", "Stoch_K", "Stoch_D",

        # === Support & Resistance ===
        "Support", "Resistance", "Support_25", "Resistance_25", "CCI", "ADX",

        # === Bollinger Bands ===
        "BB_Upper", "BB_Lower", "BB_Middle",

        # === Fibonacci Pivot Points ===
        "Pivot", "Fib_R1", "Fib_R2", "Fib_R3", "Fib_S1", "Fib_S2", "Fib_S3",

        # === Target (Harga Tertinggi & Terendah Besok) ===
        "future_high", "future_low"
    ]
    check_and_reset_model_if_needed(ticker, features)

    try:
        X_tr, X_te, yh_tr, yh_te, yl_tr, yl_te = prepare_features_and_labels(df, features)
    except Exception as e:
        logging.error(f"{ticker}: Error saat mempersiapkan data - {e}")
        return None

    # Latih dan muat model LightGBM dan XGBoost
    model_high_lgb = load_or_train_model(f"model_high_lgb_{ticker}.pkl", train_lightgbm, X_tr, yh_tr)
    model_low_lgb  = load_or_train_model(f"model_low_lgb_{ticker}.pkl", train_lightgbm, X_tr, yl_tr)
    model_high_xgb = load_or_train_model(f"model_high_xgb_{ticker}.pkl", train_xgboost, X_tr, yh_tr)
    model_low_xgb  = load_or_train_model(f"model_low_xgb_{ticker}.pkl", train_xgboost, X_tr, yl_tr)

    # Latih dan muat model LSTM
    model_lstm = load_or_train_model(f"model_lstm_{ticker}.keras", train_lstm, X_tr, yh_tr, model_type="keras")

    try:
        # Hitung probabilitas dari model LightGBM dan XGBoost
        prob_high_lgb = calculate_probability(model_high_lgb, X_te, yh_te)
        prob_low_lgb  = calculate_probability(model_low_lgb,  X_te, yl_te)
        prob_high_xgb = calculate_probability(model_high_xgb, X_te, yh_te)
        prob_low_xgb  = calculate_probability(model_low_xgb,  X_te, yl_te)
    except Exception as e:
        logging.error(f"{ticker}: Error saat menghitung probabilitas - {e}")
        return None

    # Gabungkan probabilitas dari kedua model (LightGBM dan XGBoost)
    prob_high = np.median([prob_high_lgb, prob_high_xgb])
    prob_low  = np.median([prob_low_lgb, prob_low_xgb])

    if prob_high < MIN_PROB or prob_low < MIN_PROB:
        logging.info(f"{ticker} dilewati: Prob rendah (H={prob_high:.2f}, L={prob_low:.2f})")
        return None

    # Ambil data terakhir untuk prediksi harga
    X_last = df[features].iloc[[-1]]
    
    # Prediksi harga dari LightGBM dan XGBoost
    ph_lgb = model_high_lgb.predict(X_last)[0]
    pl_lgb = model_low_lgb.predict(X_last)[0]
    ph_xgb = model_high_xgb.predict(X_last)[0]
    pl_xgb = model_low_xgb.predict(X_last)[0]

    # Prediksi harga dari LSTM (untuk HIGH saja)
    pred_lstm = model_lstm.predict(np.reshape(X_last.values, (X_last.shape[0], X_last.shape[1], 1)))
    ph_lstm = pred_lstm[0][0]

    # pl_lstm tidak digunakan

    # Median Ensemble dari hasil prediksi
    ph = np.median([ph_lgb, ph_xgb, ph_lstm])
    pl = np.median([pl_lgb, pl_xgb])

    action = "beli" if (ph - price) / price > 0.02 else "jual"
    profit_potential_pct = (ph - price) / price * 100 if action == "beli" else (price - pl) / price * 100
    prob_succ = (prob_high + prob_low) / 2
    
    # Validasi sederhana agar TP dan SL masuk akal
    if action == "beli":
        if ph <= price or pl >= price:
            return None
    else:  # aksi == "jual"
        if ph >= price or pl <= price:
            return None
            
    if profit_potential_pct < 2:
        logging.info(f"{ticker} dilewati: potensi profit rendah ({profit_potential_pct:.2f}%)")
        return None

    tanggal = pd.Timestamp.now(tz="Asia/Jakarta")
    log_prediction(ticker, tanggal, ph, pl, price)

    return {
        "ticker": ticker,
        "harga": round(price, 2),
        "take_profit": round(ph, 2),
        "stop_loss": round(pl, 2),
        "aksi": action,
        "prob_high": round(prob_high, 2),
        "prob_low": round(prob_low, 2),
        "prob_success": round(prob_succ, 2),
        "profit_potential_pct": round(profit_potential_pct, 2),
    }

def main():
    # Jalankan analisis secara paralel untuk semua saham
    results = list(filter(None, executor.map(analyze_stock, STOCK_LIST)))

    if not results:
        logging.warning("Tidak ada sinyal yang valid ditemukan.")
        return

    # Urutkan hasil berdasarkan probabilitas sukses tertinggi
    sorted_results = sorted(results, key=lambda x: x["prob_success"], reverse=True)

    # Ambil Top N sinyal terbaik
    top_n = min(10, len(sorted_results))
    top_signals = sorted_results[:top_n]

    logging.info(f"{len(results)} sinyal dianalisis. Menampilkan top {top_n} sinyal terbaik.")

    for signal in top_signals:
        print_signal(signal)
        
def retrain_if_needed(ticker: str, mae_threshold_pct: float = 0.02, incremental=False):
    evaluasi_map = evaluate_prediction_accuracy()
    metrik = evaluasi_map.get(ticker, {})
    
    akurasi = metrik.get("akurasi", 1.0)
    mae_high = metrik.get("mae_high", 0)
    mae_low = metrik.get("mae_low", 0)

    # Ambil harga saat ini sebagai basis MAE threshold
    df_now = get_stock_data(ticker)
    if df_now is None or df_now.empty:
        logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
        return

    harga_now = df_now["Close"].iloc[-1]
    mae_threshold = harga_now * mae_threshold_pct

    if akurasi < 0.80 or mae_high > mae_threshold or mae_low > mae_threshold:
        logging.info(f"Retraining diperlukan untuk {ticker} - Akurasi: {akurasi:.2%}, MAE High: {mae_high:.2f}, MAE Low: {mae_low:.2f}")

        df = calculate_indicators(df_now)
        df = df.dropna(subset=["future_high", "future_low"])
        
        # Tentukan fitur yang akan digunakan
        features = [
            "Close",
            # === Fitur Waktu ===
            "is_opening_hour", "is_closing_hour", "return_prev_day", "gap_close",
            "daily_avg", "daily_std", "daily_range", "zscore",

            # === Volatilitas ===
            "ATR",

            # === Volume ===
            "OBV", "OBV_MA_5", "OBV_MA_10", "OBV_Diff", "OBV_vs_MA", "VWAP",

            # === Trend ===
            "EMA_5", "EMA_10", "EMA_15", "EMA_20", "EMA_25", "EMA_50",
            "SMA_5", "SMA_10", "SMA_15", "SMA_20", "SMA_25", "SMA_50",
            "MACD", "MACD_Hist", "slope_5", "Body", "HA_Close",

            # === Momentum ===
            "RSI", "ROC", "Momentum", "PROC_3", "WilliamsR", "Stoch_K", "Stoch_D",

            # === Support & Resistance ===
            "Support", "Resistance", "Support_25", "Resistance_25", "CCI", "ADX",

            # === Bollinger Bands ===
            "BB_Upper", "BB_Lower", "BB_Middle",

            # === Fibonacci Pivot Points ===
            "Pivot", "Fib_R1", "Fib_R2", "Fib_R3", "Fib_S1", "Fib_S2", "Fib_S3",

            # === Target (Harga Tertinggi & Terendah Besok) ===
            "future_high", "future_low"
        ]
        
        X = df[features]
        y_high = df["future_high"]
        y_low = df["future_low"]
        
        # Incremental Learning untuk XGBoost
        if incremental:
            model_high_xgb = xgb.Booster()
            model_low_xgb = xgb.Booster()
            try:
                model_high_xgb.load_model(f"model_high_xgb_{ticker}.json")
                model_low_xgb.load_model(f"model_low_xgb_{ticker}.json")
                logging.info(f"Melanjutkan pelatihan model XGBoost untuk {ticker}")
            except Exception as e:
                logging.info(f"{ticker}: Tidak dapat memuat model sebelumnya. Pelatihan dari awal.")
                model_high_xgb = train_xgboost(X, y_high)
                model_low_xgb = train_xgboost(X, y_low)
            
            # Lanjutkan pelatihan
            dtrain_high = xgb.DMatrix(X, label=y_high)
            model_high_xgb = xgb.train(params, dtrain_high, xgb_model=model_high_xgb)
            model_high_xgb.save_model(f"model_high_xgb_{ticker}.json")
            
            dtrain_low = xgb.DMatrix(X, label=y_low)
            model_low_xgb = xgb.train(params, dtrain_low, xgb_model=model_low_xgb)
            model_low_xgb.save_model(f"model_low_xgb_{ticker}.json")
        
        else:
            # Latih model baru untuk XGBoost jika tidak incremental
            model_high_xgb = train_xgboost(X, y_high)
            joblib.dump(model_high_xgb, f"model_high_xgb_{ticker}.pkl")
            model_low_xgb = train_xgboost(X, y_low)
            joblib.dump(model_low_xgb, f"model_low_xgb_{ticker}.pkl")
        
        # Incremental Learning untuk LightGBM
        if incremental:
            model_high_lgb = lgb.Booster(model_file=f"model_high_lgb_{ticker}.txt")
            model_low_lgb = lgb.Booster(model_file=f"model_low_lgb_{ticker}.txt")
            try:
                logging.info(f"Melanjutkan pelatihan model LightGBM untuk {ticker}")
            except Exception as e:
                logging.info(f"{ticker}: Tidak dapat memuat model sebelumnya. Pelatihan dari awal.")
                model_high_lgb = train_lightgbm(X, y_high)
                model_low_lgb = train_lightgbm(X, y_low)

            train_data_high = lgb.Dataset(X, label=y_high)
            model_high_lgb = lgb.train(params, train_data_high, init_model=model_high_lgb)
            model_high_lgb.save_model(f"model_high_lgb_{ticker}.txt")

            train_data_low = lgb.Dataset(X, label=y_low)
            model_low_lgb = lgb.train(params, train_data_low, init_model=model_low_lgb)
            model_low_lgb.save_model(f"model_low_lgb_{ticker}.txt")
        
        else:
            # Latih model baru untuk LightGBM jika tidak incremental
            model_high_lgb = train_lightgbm(X, y_high)
            joblib.dump(model_high_lgb, f"model_high_lgb_{ticker}.pkl")
            model_low_lgb = train_lightgbm(X, y_low)
            joblib.dump(model_low_lgb, f"model_low_lgb_{ticker}.pkl")
        
        # LSTM tidak mendukung incremental training secara langsung, pertimbangkan untuk menggunakan "fine-tuning"
        best_params = tune_lstm_hyperparameters_optuna(X, y_high)
        model_lstm = train_final_lstm_with_best_params(X, y_high, best_params)
        model_lstm.save(f"model_lstm_{ticker}.keras")
        
        logging.info(f"Model untuk {ticker} telah dilatih ulang dan disimpan.")
    else:
        logging.info(f"Akurasi model {ticker} sudah cukup baik ({akurasi:.2%}), tidak perlu retraining.")
        
def get_realized_price_data() -> pd.DataFrame:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        logging.warning("File prediksi_log.csv tidak ditemukan.")
        return pd.DataFrame()

    df_log = pd.read_csv(log_path, names=["ticker", "tanggal", "predicted_price", "upper_bound", "lower_bound"])
    df_log.columns = df_log.columns.str.strip().str.lower()
    df_log["tanggal"] = pd.to_datetime(df_log["tanggal"], format='mixed', utc=True).dt.tz_convert("Asia/Jakarta")
    results = []

    for ticker in df_log["ticker"].unique():
        df_ticker = df_log[df_log["ticker"] == ticker].copy()
        start_date = df_ticker["tanggal"].min()
        end_date = df_ticker["tanggal"].max() + pd.Timedelta(days=2)

        try:
            df_price = yf.download(
                ticker,
                start=start_date.tz_convert("UTC").strftime("%Y-%m-%d"),
                end=end_date.tz_convert("UTC").strftime("%Y-%m-%d"),
                interval="1h",
                progress=False,
                threads=False
            )
        except Exception as e:
            logging.error(f"[{ticker}] Gagal download data: {e}")
            continue

        if df_price.empty or "High" not in df_price.columns or "Low" not in df_price.columns:
            logging.warning(f"[{ticker}] Data kosong atau kolom High/Low hilang.")
            continue

        df_price.index = pd.to_datetime(df_price.index)
        if df_price.index.tz is None:
            df_price.index = df_price.index.tz_localize("UTC").tz_convert("Asia/Jakarta")
        else:
            df_price.index = df_price.index.tz_convert("Asia/Jakarta")
        df_price = df_price.sort_index()

        for _, row in df_ticker.iterrows():
            tanggal_prediksi = row["tanggal"]
            start_window = tanggal_prediksi + pd.Timedelta(days=1)
            end_window = tanggal_prediksi + pd.Timedelta(days=2)

            df_window = df_price.loc[(df_price.index >= start_window) & (df_price.index <= end_window)]

            if df_window.empty or "High" not in df_window.columns or "Low" not in df_window.columns:
                logging.warning(f"[{ticker}] Data kosong atau kolom High/Low hilang untuk {tanggal_prediksi.date()}")
                continue

            if df_window.shape[0] < 3 or df_window[["High", "Low"]].isna().all().all():
                logging.warning(f"[{ticker}] Data prediksi {tanggal_prediksi.date()} tidak cukup untuk evaluasi.")
                continue

            try:
                actual_high = pd.to_numeric(df_window["High"], errors="coerce").max()
                actual_low = pd.to_numeric(df_window["Low"], errors="coerce").min()
            except Exception as e:
                logging.warning(f"[{ticker}] Gagal konversi harga jadi numerik: {e}")
                continue

            if pd.isna(actual_high) or pd.isna(actual_low):
                logging.warning(f"[{ticker}] Nilai actual_high/low NaN pada {tanggal_prediksi.date()}")
                continue

            results.append({
                "ticker": ticker,
                "tanggal": tanggal_prediksi,
                "actual_high": actual_high,
                "actual_low": actual_low
            })

    return pd.DataFrame(results)
    
def evaluate_prediction_accuracy() -> Dict[str, Dict[str, float]]:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        logging.warning("File prediksi_log.csv tidak ditemukan.")
        return {}

    try:
        df_log = pd.read_csv(log_path, names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
        df_log["tanggal"] = pd.to_datetime(df_log["tanggal"], format='mixed', utc=True).dt.tz_convert("Asia/Jakarta")
        df_log.drop_duplicates(subset=["ticker", "tanggal"], keep="last", inplace=True)
    except Exception as e:
        logging.error(f"Gagal membaca file log prediksi: {e}")
        return {}

    df_data = get_realized_price_data()
    if df_data.empty:
        logging.warning("Data realisasi harga kosong.")
        return {}

    df_data["tanggal"] = pd.to_datetime(df_data["tanggal"]).dt.tz_convert("Asia/Jakarta")
    df_data.drop_duplicates(subset=["ticker", "tanggal"], keep="last", inplace=True)

    df_merged = df_log.merge(df_data, on=["ticker", "tanggal"], how="inner")

    if df_merged.empty:
        logging.info("Tidak ada prediksi yang cocok dengan data realisasi.")
        return {}

    # Hapus baris dengan nilai NaN penting
    df_merged = df_merged.dropna(subset=["actual_high", "actual_low", "pred_high", "pred_low"])

    if df_merged.empty:
        logging.warning("Setelah pembersihan, tidak ada data valid untuk evaluasi.")
        return {}

    # Hitung metrik akurasi dan error
    df_merged["benar"] = (
        (df_merged["actual_high"] >= df_merged["pred_high"]) &
        (df_merged["actual_low"]  <= df_merged["pred_low"])
    )
    df_merged["error_high"] = (df_merged["actual_high"] - df_merged["pred_high"]).abs()
    df_merged["error_low"]  = (df_merged["actual_low"]  - df_merged["pred_low"]).abs()

    evaluasi = df_merged.groupby("ticker").agg({
        "benar": "mean",
        "error_high": "mean",
        "error_low": "mean"
    }).rename(columns={
        "benar": "akurasi",
        "error_high": "mae_high",
        "error_low": "mae_low"
    }).to_dict(orient="index")

    for ticker, metrik in evaluasi.items():
        logging.debug(f"{ticker}: Akurasi={metrik['akurasi']:.2%}, MAE High={metrik['mae_high']:.2f}, MAE Low={metrik['mae_low']:.2f}")

    logging.info(f"Evaluasi lengkap dihitung untuk {len(evaluasi)} ticker.")
    return evaluasi
    
def evaluate_prediction_mae() -> Dict[str, float]:
    log_path = "prediksi_log.csv"
    if not os.path.exists(log_path):
        return {}

    try:
        df_log = pd.read_csv(log_path, names=["ticker", "tanggal", "harga_awal", "pred_high", "pred_low"])
        df_log["tanggal"] = pd.to_datetime(df_log["tanggal"], format='mixed', utc=True).dt.tz_convert("Asia/Jakarta")
        df_log.drop_duplicates(subset=["ticker", "tanggal"], keep="last", inplace=True)
    except Exception as e:
        logging.error(f"Gagal membaca log prediksi: {e}")
        return {}

    df_data = get_realized_price_data()
    if df_data.empty:
        return {}

    df_data["tanggal"] = pd.to_datetime(df_data["tanggal"])
    df_data.drop_duplicates(subset=["ticker", "tanggal"], keep="last", inplace=True)

    df_merged = df_log.merge(df_data, on=["ticker", "tanggal"], how="inner")
    if df_merged.empty:
        return {}

    df_merged["mae_high"] = (df_merged["pred_high"] - df_merged["actual_high"]).abs()
    df_merged["mae_low"] = (df_merged["pred_low"] - df_merged["actual_low"]).abs()
    df_merged["mae_avg"] = (df_merged["mae_high"] + df_merged["mae_low"]) / 2

    return df_merged.groupby("ticker")["mae_avg"].mean().apply(lambda x: {"mae": x}).to_dict()
    
def get_model_predictions(ticker: str, X: pd.DataFrame):
    # Muat model
    model_high_lgb = joblib.load(f"model_high_lgb_{ticker}.pkl")
    model_low_lgb = joblib.load(f"model_low_lgb_{ticker}.pkl")
    model_high_xgb = joblib.load(f"model_high_xgb_{ticker}.pkl")
    model_low_xgb = joblib.load(f"model_low_xgb_{ticker}.pkl")
    model_lstm = keras.models.load_model(f"model_lstm_{ticker}.keras")
    
    # Prediksi dari model LightGBM, XGBoost, dan LSTM
    pred_high_lgb = model_high_lgb.predict(X)
    pred_low_lgb = model_low_lgb.predict(X)
    
    pred_high_xgb = model_high_xgb.predict(X)
    pred_low_xgb = model_low_xgb.predict(X)
    
    pred_lstm = model_lstm.predict(X)
    
    return {
        'high_lgb': pred_high_lgb,
        'low_lgb': pred_low_lgb,
        'high_xgb': pred_high_xgb,
        'low_xgb': pred_low_xgb,
        'lstm': pred_lstm
    }
    
def meta_learner(ticker: str, X: pd.DataFrame, y_high: pd.DataFrame, y_low: pd.DataFrame):
    # Dapatkan prediksi dari masing-masing model
    model_predictions = get_model_predictions(ticker, X)
    
    # Gabungkan prediksi untuk model high dan low
    X_meta = pd.DataFrame({
        'high_lgb': model_predictions['high_lgb'],
        'low_lgb': model_predictions['low_lgb'],
        'high_xgb': model_predictions['high_xgb'],
        'low_xgb': model_predictions['low_xgb'],
        'lstm': model_predictions['lstm']
    })
    
    # Target untuk model meta-learner adalah nilai sebenarnya dari harga tinggi dan rendah
    y_meta_high = y_high
    y_meta_low = y_low
    
    # Latih model meta-learner untuk memprediksi harga tertinggi dan terendah
    meta_model_high = LogisticRegression()  # Bisa diganti dengan model lain, misalnya Random Forest, XGBoost, dll.
    meta_model_high.fit(X_meta, y_meta_high)
    
    meta_model_low = LogisticRegression()
    meta_model_low.fit(X_meta, y_meta_low)
    
    return meta_model_high, meta_model_low
    
def select_best_model(ticker: str, X: pd.DataFrame, y_high: pd.DataFrame, y_low: pd.DataFrame):
    # Latih meta-learner
    meta_model_high, meta_model_low = meta_learner(ticker, X, y_high, y_low)
    
    # Prediksi menggunakan meta-learner
    model_predictions = get_model_predictions(ticker, X)
    X_meta = pd.DataFrame({
        'high_lgb': model_predictions['high_lgb'],
        'low_lgb': model_predictions['low_lgb'],
        'high_xgb': model_predictions['high_xgb'],
        'low_xgb': model_predictions['low_xgb'],
        'lstm': model_predictions['lstm']
    })
    
    meta_pred_high = meta_model_high.predict(X_meta)
    meta_pred_low = meta_model_low.predict(X_meta)
    
    # Evaluasi prediksi dari meta-learner
    mae_meta_high = mean_absolute_error(y_high, meta_pred_high)
    mae_meta_low = mean_absolute_error(y_low, meta_pred_low)
    
    # Pilih model berdasarkan kinerja (misalnya, jika meta-learner lebih baik dari masing-masing model individu)
    if mae_meta_high < min(mean_absolute_error(y_high, model_predictions['high_lgb']),
                           mean_absolute_error(y_high, model_predictions['high_xgb']),
                           mean_absolute_error(y_high, model_predictions['lstm'])):
        logging.info("Meta-learner memilih prediksi harga tertinggi.")
        return meta_pred_high
    else:
        # Jika meta-learner tidak lebih baik, pilih model terbaik
        logging.info("Memilih model terbaik untuk harga tertinggi.")
        return model_predictions['high_lgb']  # Ganti dengan model terbaik sesuai evaluasi
        
def retrain_if_needed_with_meta(ticker: str, mae_threshold_pct: float = 0.02):
    evaluasi_map = evaluate_prediction_accuracy()
    metrik = evaluasi_map.get(ticker, {})
    
    akurasi = metrik.get("akurasi", 1.0)
    mae_high = metrik.get("mae_high", 0)
    mae_low = metrik.get("mae_low", 0)

    # Ambil harga saat ini sebagai basis MAE threshold
    df_now = get_stock_data(ticker)
    if df_now is None or df_now.empty:
        logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
        return

    harga_now = df_now["Close"].iloc[-1]
    mae_threshold = harga_now * mae_threshold_pct

    if akurasi < 0.80 or mae_high > mae_threshold or mae_low > mae_threshold:
        logging.info(f"Retraining diperlukan untuk {ticker} - Akurasi: {akurasi:.2%}, MAE High: {mae_high:.2f}, MAE Low: {mae_low:.2f}")

        df = calculate_indicators(df_now)
        df = df.dropna(subset=["future_high", "future_low"])
        
        # Tentukan fitur yang akan digunakan
        features = [
            "Close",
            # === Fitur Waktu ===
            "is_opening_hour", "is_closing_hour", "return_prev_day", "gap_close",
            "daily_avg", "daily_std", "daily_range", "zscore",

            # === Volatilitas ===
            "ATR",

            # === Volume ===
            "OBV", "OBV_MA_5", "OBV_MA_10", "OBV_Diff", "OBV_vs_MA", "VWAP",

            # === Trend ===
            "EMA_5", "EMA_10", "EMA_15", "EMA_20", "EMA_25", "EMA_50",
            "SMA_5", "SMA_10", "SMA_15", "SMA_20", "SMA_25", "SMA_50",
            "MACD", "MACD_Hist", "slope_5", "Body", "HA_Close",

            # === Momentum ===
            "RSI", "ROC", "Momentum", "PROC_3", "WilliamsR", "Stoch_K", "Stoch_D",

            # === Support & Resistance ===
            "Support", "Resistance", "Support_25", "Resistance_25", "CCI", "ADX",

            # === Bollinger Bands ===
            "BB_Upper", "BB_Lower", "BB_Middle",

            # === Fibonacci Pivot Points ===
            "Pivot", "Fib_R1", "Fib_R2", "Fib_R3", "Fib_S1", "Fib_S2", "Fib_S3",

            # === Target (Harga Tertinggi & Terendah Besok) ===
            "future_high", "future_low"
        ]
        
        X = df[features]
        y_high = df["future_high"]
        y_low = df["future_low"]

        # Retrain model dan meta-learner
        retrain_if_needed(ticker, mae_threshold_pct)  # Retraining model individu
        select_best_model(ticker, X, y_high, y_low)  # Memilih model terbaik menggunakan meta-learner
        
        logging.info(f"Model untuk {ticker} telah dilatih ulang dan dipilih dengan meta-learner.")
    else:
        logging.info(f"Akurasi model {ticker} sudah cukup baik ({akurasi:.2%}), tidak perlu retraining.")
        
def update_models_with_incremental_learning_and_meta(ticker: str):
    # Ambil data baru untuk incremental learning
    df_now = get_stock_data(ticker)
    if df_now is None or df_now.empty:
        logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
        return

    df = calculate_indicators(df_now)
    df = df.dropna(subset=["future_high", "future_low"])
    
    # Tentukan fitur yang akan digunakan
    features = [
        "Close",
        # === Fitur Waktu ===
        "is_opening_hour", "is_closing_hour", "return_prev_day", "gap_close",
        "daily_avg", "daily_std", "daily_range", "zscore",

        # === Volatilitas ===
        "ATR",

        # === Volume ===
        "OBV", "OBV_MA_5", "OBV_MA_10", "OBV_Diff", "OBV_vs_MA", "VWAP",

        # === Trend ===
        "EMA_5", "EMA_10", "EMA_15", "EMA_20", "EMA_25", "EMA_50",
        "SMA_5", "SMA_10", "SMA_15", "SMA_20", "SMA_25", "SMA_50",
        "MACD", "MACD_Hist", "slope_5", "Body", "HA_Close",

        # === Momentum ===
        "RSI", "ROC", "Momentum", "PROC_3", "WilliamsR", "Stoch_K", "Stoch_D",

        # === Support & Resistance ===
        "Support", "Resistance", "Support_25", "Resistance_25", "CCI", "ADX",

        # === Bollinger Bands ===
        "BB_Upper", "BB_Lower", "BB_Middle",

        # === Fibonacci Pivot Points ===
        "Pivot", "Fib_R1", "Fib_R2", "Fib_R3", "Fib_S1", "Fib_S2", "Fib_S3",

        # === Target (Harga Tertinggi & Terendah Besok) ===
        "future_high", "future_low"
    ]
    X = df[features]
    y_high = df["future_high"]
    y_low = df["future_low"]

    # Perbarui model dengan incremental learning (contoh: LightGBM)
    model_high_lgb = joblib.load(f"model_high_lgb_{ticker}.pkl")
    model_low_lgb = joblib.load(f"model_low_lgb_{ticker}.pkl")
    model_high_xgb = joblib.load(f"model_high_xgb_{ticker}.pkl")
    model_low_xgb = joblib.load(f"model_low_xgb_{ticker}.pkl")
    model_lstm = keras.models.load_model(f"model_lstm_{ticker}.keras")

    # Update model dengan data baru (incremental learning)
    model_high_lgb = incremental_learning(model_high_lgb, X, y_high)
    model_low_lgb = incremental_learning(model_low_lgb, X, y_low)
    model_high_xgb = incremental_learning(model_high_xgb, X, y_high)
    model_low_xgb = incremental_learning(model_low_xgb, X, y_low)
    model_lstm = incremental_learning(model_lstm, X, y_high)

    # Simpan model yang telah diperbarui
    joblib.dump(model_high_lgb, f"model_high_lgb_{ticker}.pkl")
    joblib.dump(model_low_lgb, f"model_low_lgb_{ticker}.pkl")
    joblib.dump(model_high_xgb, f"model_high_xgb_{ticker}.pkl")
    joblib.dump(model_low_xgb, f"model_low_xgb_{ticker}.pkl")
    model_lstm.save(f"model_lstm_{ticker}.keras")

    # Retrain dengan meta-learner setelah pembaruan model
    retrain_with_meta(ticker)
    
def retrain_and_select_best_model(ticker: str, mae_threshold_pct: float = 0.02):
    evaluasi_map = evaluate_prediction_accuracy()
    metrik = evaluasi_map.get(ticker, {})
    
    akurasi = metrik.get("akurasi", 1.0)
    mae_high = metrik.get("mae_high", 0)
    mae_low = metrik.get("mae_low", 0)

    # Ambil harga saat ini sebagai basis MAE threshold
    df_now = get_stock_data(ticker)
    if df_now is None or df_now.empty:
        logging.error(f"{ticker}: Data saham tidak ditemukan atau kosong.")
        return

    harga_now = df_now["Close"].iloc[-1]
    mae_threshold = harga_now * mae_threshold_pct

    if akurasi < 0.80 or mae_high > mae_threshold or mae_low > mae_threshold:
        logging.info(f"Retraining diperlukan untuk {ticker} - Akurasi: {akurasi:.2%}, MAE High: {mae_high:.2f}, MAE Low: {mae_low:.2f}")

        df = calculate_indicators(df_now)
        df = df.dropna(subset=["future_high", "future_low"])

        # Tentukan fitur yang akan digunakan
        features = [
            "Close",
            # === Fitur Waktu ===
            "is_opening_hour", "is_closing_hour", "return_prev_day", "gap_close",
            "daily_avg", "daily_std", "daily_range", "zscore",

            # === Volatilitas ===
            "ATR",

            # === Volume ===
            "OBV", "OBV_MA_5", "OBV_MA_10", "OBV_Diff", "OBV_vs_MA", "VWAP",

            # === Trend ===
            "EMA_5", "EMA_10", "EMA_15", "EMA_20", "EMA_25", "EMA_50",
            "SMA_5", "SMA_10", "SMA_15", "SMA_20", "SMA_25", "SMA_50",
            "MACD", "MACD_Hist", "slope_5", "Body", "HA_Close",

            # === Momentum ===
            "RSI", "ROC", "Momentum", "PROC_3", "WilliamsR", "Stoch_K", "Stoch_D",

            # === Support & Resistance ===
            "Support", "Resistance", "Support_25", "Resistance_25", "CCI", "ADX",

            # === Bollinger Bands ===
            "BB_Upper", "BB_Lower", "BB_Middle",

            # === Fibonacci Pivot Points ===
            "Pivot", "Fib_R1", "Fib_R2", "Fib_R3", "Fib_S1", "Fib_S2", "Fib_S3",

            # === Target (Harga Tertinggi & Terendah Besok) ===
            "future_high", "future_low"
        ]
        
        X = df[features]
        y_high = df["future_high"]
        y_low = df["future_low"]

        # Latih atau update model individu
        model_high_lgb = train_lightgbm(X, y_high)
        joblib.dump(model_high_lgb, f"model_high_lgb_{ticker}.pkl")
        
        model_low_lgb = train_lightgbm(X, y_low)
        joblib.dump(model_low_lgb, f"model_low_lgb_{ticker}.pkl")

        model_high_xgb = train_xgboost(X, y_high)
        joblib.dump(model_high_xgb, f"model_high_xgb_{ticker}.pkl")
        
        model_low_xgb = train_xgboost(X, y_low)
        joblib.dump(model_low_xgb, f"model_low_xgb_{ticker}.pkl")

        # Model LSTM juga dilatih dengan data baru
        best_params = tune_lstm_hyperparameters_optuna(X, y_high)
        model_lstm = train_final_lstm_with_best_params(X, y_high, best_params)
        model_lstm.save(f"model_lstm_{ticker}.keras")

        # Setelah retraining, gunakan meta-learner untuk memilih model terbaik
        selected_model = select_best_model(ticker, X, y_high, y_low)  # Memilih model terbaik
        
        logging.info(f"Model terbaik untuk {ticker} telah dipilih dan disimpan: {selected_model}")
    else:
        logging.info(f"Akurasi model {ticker} sudah cukup baik ({akurasi:.2%}), tidak perlu retraining.")
        
def check_and_reset_model_if_needed(ticker, features):
    hash_path = f"model_feature_hashes.json"
    current_hash = hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()

    saved_hashes = {}
    if os.path.exists(hash_path):
        try:
            with open(hash_path, "r") as f:
                content = f.read().strip()
                if content:
                    saved_hashes = json.loads(content)
        except json.JSONDecodeError:
            logging.warning("Hash file corrupted, resetting...")
            saved_hashes = {}

    # Cek performa model (akurasi dan MAE)
    akurasi_map = evaluate_prediction_accuracy()
    mae_map = evaluate_prediction_mae()
    akurasi = akurasi_map.get(ticker, {}).get("akurasi", 1.0)
    mae = mae_map.get(ticker, {}).get("mae", 0.0)

    perlu_reset = (
        saved_hashes.get(ticker) != current_hash or
        akurasi < AKURASI_THRESHOLD or
        mae > MAE_THRESHOLD
    )

    if perlu_reset:
        logging.info(f"Reset model {ticker} | Fitur berubah: {saved_hashes.get(ticker) != current_hash} | Akurasi: {akurasi:.2%} | MAE: {mae:.2f}")

        model_files = [
            f"model_high_lgb_{ticker}.pkl",
            f"model_low_lgb_{ticker}.pkl",
            f"model_high_xgb_{ticker}.pkl",
            f"model_low_xgb_{ticker}.pkl",
            f"model_lstm_{ticker}.keras"
        ]

        for fname in model_files:
            if os.path.exists(fname):
                os.remove(fname)
                logging.info(f"Model {fname} dihapus.")

        saved_hashes[ticker] = current_hash
        with open(hash_path, "w") as f:
            json.dump(saved_hashes, f, indent=2)
    else:
        logging.info(f"Model {ticker} tidak perlu di-reset. Akurasi dan MAE masih wajar.")

def reset_models():
    # Pola file model
    patterns = [
        "model_high_lgb_*.pkl",  # LightGBM
        "model_low_lgb_*.pkl",   # LightGBM
        "model_high_xgb_*.pkl",  # XGBoost
        "model_low_xgb_*.pkl",   # XGBoost
        "model_lstm_*.keras"     # LSTM
    ]

    total_deleted = 0
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                logging.info(f"Dihapus: {filepath}")
                total_deleted += 1
            except Exception as e:
                logging.error(f"Gagal menghapus {filepath}: {e}")
    
    if total_deleted == 0:
        logging.info("Tidak ada model yang ditemukan untuk dihapus.")
    else:
        logging.info(f"Total {total_deleted} model dihapus.")
        
# === Daftar Kutipan Motivasi ===
MOTIVATION_QUOTES = [
    "Setiap peluang adalah langkah kecil menuju kebebasan finansial.",
    "Cuan bukan tentang keberuntungan, tapi tentang konsistensi dan strategi.",
    "Disiplin hari ini, hasil luar biasa nanti.",
    "Trader sukses bukan yang selalu benar, tapi yang selalu siap.",
    "Naik turun harga itu biasa, yang penting arah portofolio naik.",
    "Fokus pada proses, profit akan menyusul.",
    "Jangan hanya lihat harga, lihat potensi di baliknya.",
    "Ketika orang ragu, itulah peluang sesungguhnya muncul.",
    "Investasi terbaik adalah pada pengetahuan dan ketenangan diri.",
    "Satu langkah hari ini lebih baik dari seribu penyesalan besok."
    "Moal rugi jalma nu talek jeung tekadna kuat.",
    "Rejeki mah moal ka tukang, asal usaha jeung sabar.",
    "Lamun hayang hasil nu beda, ulah make cara nu sarua terus.",
    "Ulah sieun gagal, sieun lamun teu nyobaan.",
    "Cuan nu leres asal ti élmu jeung kasabaran.",
    "Sabada hujan pasti aya panonpoé, sabada rugi bisa aya untung.",
    "Ngabagéakeun resiko teh bagian tina kamajuan.",
    "Jalma nu kuat téh lain nu teu pernah rugi, tapi nu sanggup bangkit deui.",
    "Ngora kudu wani nyoba, heubeul kudu wani investasi.",
    "Reureujeungan ayeuna, kabagjaan engké."
    "Niat alus, usaha terus, hasil bakal nuturkeun.",
    "Ulah ngadagoan waktu nu pas, tapi cobian ayeuna.",
    "Hirup teh kawas saham, kadang naek kadang turun, tapi ulah leungit arah.",
    "Sakumaha gede ruginya, élmu nu diala leuwih mahal hargana.",
    "Ulah beuki loba mikir, beuki saeutik tindakan.",
    "Kabagjaan datang ti tangtungan jeung harepan nu dilaksanakeun.",
    "Panghasilan teu datang ti ngalamun, tapi ti aksi jeung analisa.",
    "Sasat nu bener, bakal mawa kana untung nu lila.",
    "Tong ukur ningali batur nu untung, tapi diajar kumaha cara maranéhna usaha.",
    "Jalma sukses mah sok narima gagal minangka bagian ti perjalanan."
    "Saham bisa turun, tapi semangat kudu tetap ngora. Jalan terus, rejeki moal salah alamat.",
    "Kadang market galak, tapi inget, nu sabar jeung konsisten nu bakal panén hasilna.",
    "Cuan moal datang ti harepan hungkul, kudu dibarengan ku strategi jeung tekad.",
    "Teu aya jalan pintas ka sukses, ngan aya jalan nu jelas jeung disiplin nu kuat.",
    "Di balik koreksi aya akumulasi, di balik gagal aya élmu anyar. Ulah pundung!",
    "Sakumaha seredna pasar, nu kuat haténa bakal salamet.",
    "Rejeki teu datang ti candaan, tapi ti candak kaputusan jeung tindakan.",
    "Sugan ayeuna can untung, tapi tong hilap, tiap analisa téh tabungan pangalaman.",
    "Tenang lain berarti nyerah, tapi ngatur posisi jeung nunggu waktu nu pas.",
    "Sagalana dimimitian ku niat, dilaksanakeun ku disiplin, jeung dipanen ku waktu."
    "“Suatu saat akan datang hari di mana semua akan menjadi kenangan.” – Erza Scarlet (Fairy Tail)",
    "“Lebih baik menerima kejujuran yang pahit, daripada kebohongan yang manis.” – Soichiro Yagami (Death Note)",
    "“Jangan menyerah. Hal memalukan bukanlah ketika kau jatuh, tetapi ketika kau tidak mau bangkit lagi.” – Midorima Shintarou (Kuroko no Basuke)",
    "“Jangan khawatirkan apa yang dipikirkan orang lain. Tegakkan kepalamu dan melangkahlah ke depan.” – Izuku Midoriya (Boku no Hero Academia)",
    "“Tuhan tak akan menempatkan kita di sini melalui derita demi derita bila Ia tak yakin kita bisa melaluinya.” – Kano Yuki (Sword Art Online)",
    "“Mula-mula, kau harus mengubah dirimu sendiri atau tidak akan ada yang berubah untukmu.” – Sakata Gintoki (Gintama)",
    "“Banyak orang gagal karena mereka tidak memahami usaha yang diperlukan untuk menjadi sukses.” – Yukino Yukinoshita (Oregairu)",
    "“Kekuatan sejati dari umat manusia adalah bahwa kita memiliki kuasa penuh untuk mengubah diri kita sendiri.” – Saitama (One Punch Man)",
    "“Hidup bukanlah permainan keberuntungan. Jika kau ingin menang, kau harus bekerja keras.” – Sora (No Game No Life)",
    "“Kita harus mensyukuri apa yang kita punya saat ini karena mungkin orang lain belum tentu mempunyainya.” – Kayaba Akihiko (Sword Art Online)",
    "“Kalau kau ingin menangis karena gagal, berlatihlah lebih keras lagi sehingga kau pantas menangis ketika kau gagal.” – Megumi Takani (Samurai X)",
    "“Ketika kau bekerja keras dan gagal, penyesalan itu akan cepat berlalu. Berbeda dengan penyesalan ketika tidak berani mencoba.” – Akihiko Usami (Junjou Romantica)",
    "“Ketakutan bukanlah kejahatan. Itu memberitahukan apa kelemahanmu. Dan begitu tahu kelemahanmu, kamu bisa menjadi lebih kuat.” – Gildarts (Fairy Tail)",
    "“Untuk mendapatkan kesuksesan, keberanianmu harus lebih besar daripada ketakutanmu.” – Han Juno (Eureka Seven)",
    "“Kegagalan seorang pria yang paling sulit yaitu ketika dia gagal untuk menghentikan air mata seorang wanita.” – Kasuka Heiwajima (Durarara!)",
    "“Air mata palsu bisa menyakiti orang lain. Tapi, senyuman palsu hanya akan menyakiti dirimu sendiri.” – C.C (Code Geass)",
    "“Kita harus menjalani hidup kita sepenuhnya. Kamu tidak pernah tahu, kita mungkin sudah mati besok.” – Kaori Miyazono (Shigatsu wa Kimi no Uso)",
    "“Bagaimana kamu bisa bergerak maju kalau kamu terus menyesali masa lalu?” – Edward Elric (Fullmetal Alchemist: Brotherhood)",
    "“Jika kau seorang pria, buatlah wanita yang kau cintai jatuh cinta denganmu apa pun yang terjadi!” – Akhio (Clannad)",
    "“Semua laki-laki mudah cemburu dan bego, tapi perempuan malah menyukainya. Orang jadi bodoh saat jatuh cinta.” – Horo (Spice and Wolf)",
    "“Wanita itu sangat indah, satu senyuman mereka saja sudah menjadi sebuah keajaiban.” – Onigiri (Air Gear)",
    "“Saat kamu harus memilih satu cinta aja, pasti ada orang lain yang menangis.” – Tsubame (Ai Kora)",
    "“Aku tidak suka hubungan yang tidak jelas.” – Senjougahara (Bakemonogatari)",
    "“Cewek itu seharusnya lembut dan baik, dan bisa menyembuhkan luka di hati.” – Yoshii (Baka to Test)",
    "“Keluargamu adalah pahlawanmu.” – Sinchan (C. Sinchan)"
    "Hidup itu sederhana, kita yang membuatnya sulit. – Confucius.",
    "Hal yang paling penting adalah menikmati hidupmu, menjadi bahagia, apa pun yang terjadi. - Audrey Hepburn.",
    "Hidup itu bukan soal menemukan diri Anda sendiri, hidup itu membuat diri Anda sendiri. - George Bernard Shaw.",
    "Hidup adalah mimpi bagi mereka yang bijaksana, permainan bagi mereka yang bodoh, komedi bagi mereka yang kaya, dan tragedi bagi mereka yang miskin. - Sholom Aleichem.",
    "Kenyataannya, Anda tidak tahu apa yang akan terjadi besok. Hidup adalah pengendaraan yang gila dan tidak ada yang menjaminnya. – Eminem.",
    "Tujuan hidup kita adalah menjadi bahagia. - Dalai Lama.",
    "Hidup yang baik adalah hidup yang diinspirasi oleh cinta dan dipandu oleh ilmu pengetahuan. - Bertrand Russell.",
    "Seribu orang tua bisa bermimpi, satu orang pemuda bisa mengubah dunia. – Soekarno.",
    "Pendidikan adalah senjata paling ampuh untuk mengubah dunia. - Nelson Mandela.",
    "Usaha dan keberanian tidak cukup tanpa tujuan dan arah perencanaan. - John F. Kennedy.",
    "Dunia ini cukup untuk memenuhi kebutuhan manusia, bukan untuk memenuhi keserakahan manusia. - Mahatma Gandhi.",
    "Jika kamu berpikir terlalu kecil untuk membuat sebuah perubahan, cobalah tidur di ruangan dengan seekor nyamuk. - Dalai Lama.",
    "Anda mungkin bisa menunda, tapi waktu tidak akan menunggu. - Benjamin Franklin.",
    "Kamu tidak perlu menjadi luar biasa untuk memulai, tapi kamu harus memulai untuk menjadi luar biasa. - Zig Ziglar.",
    "Jangan habiskan waktumu memukuli dinding dan berharap bisa mengubahnya menjadi pintu. - Coco Chanel.",
    "Tidak ada yang akan berhasil kecuali kau melakukannya. - Maya Angelou.",
    "Kamu tidak bisa kembali dan mengubah awal saat kamu memulainya, tapi kamu bisa memulainya lagi dari mana kamu berada sekarang dan ubah akhirnya. - C.S Lewis.",
    "Beberapa orang memimpikan kesuksesan, sementara yang lain bangun setiap pagi untuk mewujudkannya. - Wayne Huizenga.",
    "Pekerjaan-pekerjaan kecil yang selesai dilakukan lebih baik daripada rencana-rencana besar yang hanya didiskusikan. - Peter Marshall.",
    "Kita harus berarti untuk diri kita sendiri dulu sebelum kita menjadi orang yang berharga bagi orang lain. - Ralph Waldo Emerson.",
    "Hal yang paling menyakitkan adalah kehilangan jati dirimu saat engkau terlalu mencintai seseorang. Serta lupa bahwa sebenarnya engkau juga spesial. - Ernest Hemingway.",
    "Beberapa orang akan pergi dari hidupmu, tapi itu bukan akhir dari ceritamu. Itu cuma akhir dari bagian mereka di ceritamu. - Faraaz Kazi.",
    "Cinta terjadi begitu singkat, namun melupakan memakan waktu begitu lama. - Pablo Neruda.",
    "Seseorang tak akan pernah tahu betapa dalam kadar cintanya sampai terjadi sebuah perpisahan. - Kahlil Gibran.",
    "Hubungan asmara itu seperti kaca. Terkadang lebih baik meninggalkannya dalam keadaan pecah daripada menyakiti dirimu dengan cara menyatukan mereka kembali. - D.Love.",
    "Cinta itu seperti angin. Kau tak dapat melihatnya, tapi kau dapat merasakannya. - Nicholas Sparks.",
    "Cinta adalah ketika kebahagiaan orang lain lebih penting dari kebahagiaanmu. - H. Jackson Brown.",
    "Asmara bukan hanya sekadar saling memandang satu sama lain. Tapi, juga sama-sama melihat ke satu arah yang sama. - Antoine de Saint-Exupéry.",
    "Bagaimana kau mengeja ‘cinta’? tanya Piglet. Kau tak usah mengejanya, rasakan saja, jawab Pooh. - A.A Milne.",
    "Kehidupan adalah 10 persen apa yang terjadi terhadap Anda dan 90 persen adalah bagaimana Anda meresponnya. - Lou Holtz.",
    "Satu-satunya keterbatasan dalam hidup adalah perilaku yang buruk. - Scott Hamilton.",
    "Seseorang yang berani membuang satu jam waktunya tidak mengetahui nilai dari kehidupan. - Charles Darwin.",
    "Apa yang kita pikirkan menentukan apa yang akan terjadi pada kita. Jadi jika kita ingin mengubah hidup, kita perlu sedikit mengubah pikiran kita. - Wayne Dyer.",
    "Ia yang mengerjakan lebih dari apa yang dibayar pada suatu saat nanti akan dibayar lebih dari apa yang ia kerjakan. - Napoleon Hill.",
    "Saya selalu mencoba untuk mengubah kemalangan menjadi kesempatan. - John D. Rockefeller.",
    "Seseorang yang pernah melakukan kesalahan dan tidak pernah memperbaikinya berarti ia telah melakukan satu kesalahan lagi. - Konfusius.",
    "Anda tidak akan pernah belajar sabar dan berani jika di dunia ini hanya ada kebahagiaan. - Helen Keller.",
    "Tidak apa-apa untuk merayakan kesuksesan, tapi lebih penting untuk memperhatikan pelajaran tentang kegagalan. – Bill Gates."
]

def get_random_motivation() -> str:
    return random.choice(MOTIVATION_QUOTES)

# === Eksekusi & Kirim Sinyal ===
if __name__ == "__main__":
    reset_models()
    logging.info("🚀 Memulai analisis saham...")
    max_workers = min(8, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(analyze_stock, STOCK_LIST))

    results = [r for r in results if r]

    pd.DataFrame(results).to_csv(BACKUP_CSV_PATH, index=False)
    logging.info("✅ Backup CSV disimpan")

    top_10 = sorted(results, key=lambda x: x["profit_potential_pct"], reverse=True)[:10]
    if top_10:
        motivation = get_random_motivation()
        message = (
            f"<b>🔮Hai K.N.T.L. Clan Member🔮</b>\n"
            f"<b>Apapun Yang Sedang Kalian Hadapi Saat Ini, Ingatlah...</b>\n"
            f"<b><i>{motivation}</i></b>\n\n"
            f"<b>Berikut Rekomendasi Saham Pilihan Berdasarkan Analisa K.N.T.L. (Kernel Neural Trading Logic) A.I 🤖:</b>\n"
        )
        for r in top_10:
            message += (
                f"\n🔹 {r['ticker']}\n"
                f"   💰 Harga: {r['harga']:.2f}\n"
                f"   🎯 TP: {r['take_profit']:.2f}\n"
                f"   🛑 SL: {r['stop_loss']:.2f}\n"
                f"   📈 Potensi Profit: {r['profit_potential_pct']:.2f}%\n"
                f"   ✅ Probabilitas: {r['prob_success']*100:.1f}%\n"
                f"   📌 Aksi: <b>{r['aksi'].upper()}</b>\n"
            )
        send_telegram_message(message)

    logging.info("✅ Selesai.")

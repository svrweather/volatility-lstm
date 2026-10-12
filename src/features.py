"""Признаки, таргет и разбиение по времени"""
import numpy as np
from sklearn.preprocessing import StandardScaler

HORIZON = 5

FEATURES = [
    "RET",        # лог-доходность акции
    "RANGE",      # (HIGH - LOW) / LOW — внутридневной размах
    "VOL_CHG",    # изменение лог-объёма
    "VOLAT",      # std доходности за 20 дней
    "MA_RATIO",   # отклонение цены от скользящей средней за 20 дней
    "IMOEX_RET",  # доходность индекса Мосбиржи
    "CNY_RET",    # доходность курса юаня
    "REL_RET",    # доходность акции относительно рынка
]


def prepare(raw, horizon=HORIZON):
    """Считает признаки и таргет, выкидывает строки с NaN/inf."""
    df = raw.copy()
    df["RET"] = np.log(df["CLOSE"]).diff()
    df["RANGE"] = (df["HIGH"] - df["LOW"]) / df["LOW"]
    df["VOL_CHG"] = np.log(df["VOLUME"] + 1).diff()
    df["VOLAT"] = df["RET"].rolling(20).std()
    df["MA_RATIO"] = df["CLOSE"] / df["CLOSE"].rolling(20).mean() - 1
    df["IMOEX_RET"] = np.log(df["IMOEX"]).diff()
    df["CNY_RET"] = np.log(df["CNY"]).diff()
    df["REL_RET"] = df["RET"] - df["IMOEX_RET"]

    abs_ret = df["RET"].abs() * 100
    df["PAST_VOL"] = abs_ret.rolling(horizon).mean()
    df["TARGET"] = df["PAST_VOL"].shift(-horizon)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def split(df, val_size=0.15, test_size=0.15, horizon=HORIZON):
    n = len(df)
    i_val = int(n * (1 - val_size - test_size))
    i_test = int(n * (1 - test_size))
    return (df.iloc[:i_val - horizon],
            df.iloc[i_val:i_test - horizon],
            df.iloc[i_test:])


def scale(train, *others, features=FEATURES):
    scaler = StandardScaler().fit(train[features])
    return [scaler.transform(d[features]) for d in (train, *others)]

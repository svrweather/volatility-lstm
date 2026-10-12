"""Загрузка дневных котировок с MOEX ISS и кэширование в datа"""
from pathlib import Path

import apimoex
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
START = "2016-01-01"

CONTEXT = {
    "IMOEX": dict(board="SNDX", market="index", engine="stock"),
    "CNYRUB_TOM": dict(board="CETS", market="selt", engine="currency"),
}


def _to_df(rows):
    df = pd.DataFrame(rows)
    df["TRADEDATE"] = pd.to_datetime(df["TRADEDATE"])
    return df.set_index("TRADEDATE")


def download(ticker="SBER", start=START):
    DATA_DIR.mkdir(exist_ok=True)
    with requests.Session() as session:
        stock = apimoex.get_board_history(
            session, ticker, start=start,
            columns=("TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"))
        _to_df(stock).to_csv(DATA_DIR / f"{ticker}.csv")

        for name, kw in CONTEXT.items():
            rows = apimoex.get_board_history(
                session, name, start=start, columns=("TRADEDATE", "CLOSE"), **kw)
            _to_df(rows).to_csv(DATA_DIR / f"{name}.csv")


def load(ticker="SBER", start=START, refresh=False):
    """OHLCV акции + столбцы IMOEX и CNY"""
    files = [DATA_DIR / f"{n}.csv" for n in (ticker, *CONTEXT)]
    if refresh or not all(f.exists() for f in files):
        download(ticker, start)

    read = lambda f: pd.read_csv(f, index_col="TRADEDATE", parse_dates=True)
    df = read(files[0])
    df = df.join(read(files[1])["CLOSE"].rename("IMOEX"), how="inner")
    df = df.join(read(files[2])["CLOSE"].rename("CNY"), how="inner")
    prices = ["OPEN", "HIGH", "LOW", "CLOSE", "IMOEX", "CNY"]
    return df[(df[prices] > 0).all(axis=1)].dropna()

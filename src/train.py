"""Обучение LSTM, сравнение с бейзлайнами и графики в results/"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from src.data import load
from src.dataset import SEQ_LEN, WindowDataset, align, make_windows
from src.features import FEATURES, prepare, scale, split
from src.model import PredictionModel

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

C_ACTUAL = "#9a9890"
C_MODEL = "#2a78d6"
C_BASE = "#eb6834"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, dl, loss_fn, device, optimizer=None):
    """optimizer - обучение, без - оценка."""
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            loss = loss_fn(pred, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(x)
    return total / len(dl.dataset)


def fit(model, train_dl, val_dl, device, epochs=50, lr=1e-3, patience=7):
    """Обучение с early stopping по val"""
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train": [], "val": []}
    best_val, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(epochs):
        train_loss = run_epoch(model, train_dl, loss_fn, device, optimizer)
        val_loss = run_epoch(model, val_dl, loss_fn, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"epoch {epoch + 1:3d}  train {train_loss:.4f}  val {val_loss:.4f}")

        if val_loss < best_val:
            best_val, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"early stopping, лучшая эпоха {epoch + 1 - patience}")
                break

    model.load_state_dict(best_state)
    return history


@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    return torch.cat([model(x.to(device)).squeeze(-1).cpu() for x, _ in dl]).numpy()


def metrics(y, pred, y_mean_train):
    mse = float(np.mean((y - pred) ** 2))
    mse_const = float(np.mean((y - y_mean_train) ** 2))
    return {
        "MSE": mse,
        "MAE": float(np.mean(np.abs(y - pred))),
        "Корреляция": float(np.corrcoef(y, pred)[0, 1]) if np.ptp(pred) > 0 else float("nan"),
        "Улучшение MSE над константой": 1 - mse / mse_const,
    }


def fit_ridge(X_tr, y_tr, X_val, y_val, X_te):
    Xw_tr, yw_tr = make_windows(X_tr, y_tr)
    Xw_val, yw_val = make_windows(X_val, y_val)
    Xw_te, _ = make_windows(X_te, np.zeros(len(X_te)))
    best = min(
        (Ridge(alpha=a).fit(Xw_tr, yw_tr) for a in (0.1, 1, 10, 100, 1000)),
        key=lambda r: np.mean((r.predict(Xw_val) - yw_val) ** 2),
    )
    return best.predict(Xw_te), best.alpha


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], color=C_ACTUAL, lw=2, label="train")
    ax.plot(epochs, history["val"], color=C_MODEL, lw=2, label="val")
    best = int(np.argmin(history["val"])) + 1
    ax.axvline(best, color="#c3c2b7", lw=1, ls="--")
    ax.text(best, ax.get_ylim()[1], " лучшая эпоха", va="top", fontsize=9, color="#52514e")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE")
    ax.set_title("Кривые обучения", loc="left")
    _style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_forecast(res, path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(res.index, res["actual"], color=C_ACTUAL, lw=1.2, label="Факт")
    ax.plot(res.index, res["persistence"], color=C_BASE, lw=1.2, alpha=0.8,
            label="Бейзлайн: последние 5 дней")
    ax.plot(res.index, res["lstm"], color=C_MODEL, lw=2, label="LSTM")
    ax.set_ylabel("Средняя |доходность|, %")
    ax.set_title("Волатильность на 5 дней вперёд, тестовая выборка", loc="left")
    _style(ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(res, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(res["lstm"], res["actual"], s=10, alpha=0.5, color=C_MODEL, linewidths=0)
    lim = [0, max(res["lstm"].max(), res["actual"].max()) * 1.05]
    ax.plot(lim, lim, color="#c3c2b7", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Прогноз LSTM, %")
    ax.set_ylabel("Факт, %")
    ax.set_title("Прогноз vs факт (тест)", loc="left")
    _style(ax, legend=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _style(ax, legend=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.2)
    if legend:
        ax.legend(frameon=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="SBER")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refresh", action="store_true", help="перекачать данные с MOEX")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = prepare(load(args.ticker, refresh=args.refresh))
    train_df, val_df, test_df = split(df)
    X_tr, X_val, X_te = scale(train_df, val_df, test_df)
    y_tr, y_val, y_te = (d["TARGET"].values for d in (train_df, val_df, test_df))
    print(f"{args.ticker}: {df.index[0].date()} — {df.index[-1].date()}, "
          f"train {len(train_df)} / val {len(val_df)} / test {len(test_df)} дней")

    train_dl = DataLoader(WindowDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    val_dl = DataLoader(WindowDataset(X_val, y_val), batch_size=64)
    test_dl = DataLoader(WindowDataset(X_te, y_te), batch_size=64)

    model = PredictionModel(len(FEATURES), args.hidden, args.layers, custom=True).to(device)
    history = fit(model, train_dl, val_dl, device, args.epochs, args.lr, args.patience)

    y = align(y_te)
    y_mean = y_tr.mean()
    ridge_pred, alpha = fit_ridge(X_tr, y_tr, X_val, y_val, X_te)
    preds = {
        "Константа (среднее train)": np.full_like(y, y_mean),
        "Последние 5 дней": align(test_df["PAST_VOL"].values),
        f"Ridge по окну (alpha={alpha:g})": ridge_pred,
        "LSTM": predict(model, test_dl, device),
    }
    table = pd.DataFrame({name: metrics(y, pr, y_mean) for name, pr in preds.items()}).T
    print("\nТест:\n" + table.to_string(float_format=lambda v: f"{v:.4f}"))

    RESULTS_DIR.mkdir(exist_ok=True)
    res = pd.DataFrame({"actual": y, "lstm": preds["LSTM"],
                        "persistence": preds["Последние 5 дней"]},
                       index=align(test_df.index))
    plot_loss(history, RESULTS_DIR / "loss.png")
    plot_forecast(res, RESULTS_DIR / "forecast.png")
    plot_scatter(res, RESULTS_DIR / "scatter.png")
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(table.round(4).astype(object).where(table.notna(), None).to_dict(orient="index"),
                   ensure_ascii=False, indent=2))
    print(f"\nГрафики и метрики сохранены в {RESULTS_DIR}")


if __name__ == "__main__":
    main()

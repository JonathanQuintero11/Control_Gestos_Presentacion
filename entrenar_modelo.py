#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 entrenar_modelo.py
 Entrena una red ligera (CNN 1D o LSTM) con el dataset de recolector_datos.py
==========================================================================

REQUISITOS:
    python -m pip install torch numpy

USO:
    python entrenar_modelo.py                 # CNN 1D (recomendado: rapida y ligera)
    python entrenar_modelo.py --modelo lstm   # LSTM

ENTRADA : ./dataset/X.npy, X_muneca.npy, y.npy, meta.json   (de recolector_datos.py)
SALIDA  : ./modelo/modelo_gestos.pt     modelo TorchScript (incluye la estandarizacion)
          ./modelo/modelo_meta.json     configuracion para el script de inferencia

Entrada de la red: (lote, 20, 67) = forma de la mano (63) + desplazamiento (2)
+ velocidad (2) de la muneca. Ver features_gestos.construir_features().
==========================================================================
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from features_gestos import construir_features

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
OUT_DIR = BASE_DIR / "modelo"


# --------------------------------------------------------------------------
# Modelos (la estandarizacion va DENTRO del modelo como buffers)
# --------------------------------------------------------------------------
class CNN1D(nn.Module):
    def __init__(self, n_feat, n_cls, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.head = nn.Linear(128, n_cls)

    def forward(self, x):                       # x: (B, T, F)
        x = (x - self.mean) / self.std
        h = self.conv(x.transpose(1, 2))        # (B, 64, T)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(h)                     # logits (B, n_cls)


class LSTMNet(nn.Module):
    def __init__(self, n_feat, n_cls, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.lstm = nn.LSTM(n_feat, 64, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(64, n_cls)

    def forward(self, x):                       # x: (B, T, F)
        x = (x - self.mean) / self.std
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1]))


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def cargar_datos():
    for nombre in ("X.npy", "X_muneca.npy", "y.npy", "meta.json"):
        if not (DATA_DIR / nombre).exists():
            print(f"[ERROR] Falta '{DATA_DIR / nombre}'. Ejecuta primero recolector_datos.py.")
            sys.exit(1)
    X = np.load(DATA_DIR / "X.npy")
    Xm = np.load(DATA_DIR / "X_muneca.npy")
    y = np.load(DATA_DIR / "y.npy").astype(np.int64)
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    return X, Xm, y, meta


def dividir_estratificado(y, frac_val, rng):
    idx_tr, idx_va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * frac_val)))
        idx_va.extend(idx[:n_val])
        idx_tr.extend(idx[n_val:])
    return np.array(idx_tr), np.array(idx_va)


def matriz_confusion(y_true, y_pred, n_cls):
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=["cnn", "lstm"], default="cnn")
    ap.add_argument("--epocas", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.2, help="fraccion de validacion")
    ap.add_argument("--ruido", type=float, default=0.05, help="ruido gaussiano (en desviaciones tipicas)")
    ap.add_argument("--paciencia", type=int, default=25, help="epocas sin mejora antes de parar")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    X, Xm, y, meta = cargar_datos()
    clases = {int(k): v for k, v in meta["clases"].items()}
    n_cls = len(clases)
    nombres = [clases[i] for i in range(n_cls)]

    conteos = np.bincount(y, minlength=n_cls)
    print("Secuencias por clase:", {nombres[i]: int(conteos[i]) for i in range(n_cls)})
    if (conteos < 5).any():
        print("[ERROR] Cada clase necesita al menos 5 secuencias (idealmente 50+). Graba mas datos.")
        sys.exit(1)

    feats = construir_features(X, Xm)                    # (N, T, 67)
    N, T, F = feats.shape
    idx_tr, idx_va = dividir_estratificado(y, args.val, rng)
    Xtr, ytr, Xva, yva = feats[idx_tr], y[idx_tr], feats[idx_va], y[idx_va]
    print(f"Entrenamiento: {len(ytr)} | Validacion: {len(yva)} | Entrada: ({T}, {F})")

    mean = Xtr.reshape(-1, F).mean(axis=0)
    std = np.maximum(Xtr.reshape(-1, F).std(axis=0), 1e-3)

    Net = CNN1D if args.modelo == "cnn" else LSTMNet
    modelo = Net(F, n_cls, mean, std)
    n_par = sum(p.numel() for p in modelo.parameters())
    print(f"Arquitectura: {args.modelo.upper()} ({n_par:,} parametros)")

    Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva.astype(np.float32))
    yva_t = torch.from_numpy(yva)
    std_t = torch.from_numpy(std.astype(np.float32))

    # Pesos por clase para compensar desbalance
    pesos = len(ytr) / (n_cls * np.maximum(np.bincount(ytr, minlength=n_cls), 1))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32))
    opt = torch.optim.Adam(modelo.parameters(), lr=args.lr, weight_decay=1e-4)

    mejor_loss, mejor_estado, sin_mejora = float("inf"), None, 0
    for ep in range(1, args.epocas + 1):
        modelo.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            xb, yb = Xtr_t[b], ytr_t[b]
            xb = xb + torch.randn_like(xb) * std_t * args.ruido     # aumento de datos
            opt.zero_grad()
            loss = crit(modelo(xb), yb)
            loss.backward()
            opt.step()

        modelo.eval()
        with torch.no_grad():
            out = modelo(Xva_t)
            v_loss = crit(out, yva_t).item()
            v_acc = (out.argmax(dim=1) == yva_t).float().mean().item()
        if ep == 1 or ep % 10 == 0:
            print(f"  epoca {ep:3d} | val_loss {v_loss:.4f} | val_acc {v_acc:.3f}")

        if v_loss < mejor_loss - 1e-4:
            mejor_loss, sin_mejora = v_loss, 0
            mejor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print(f"  parada temprana en la epoca {ep}")
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    with torch.no_grad():
        pred = modelo(Xva_t).argmax(dim=1).numpy()
    acc = float((pred == yva).mean())
    cm = matriz_confusion(yva, pred, n_cls)

    print(f"\nPrecision de validacion: {acc:.3f}")
    print("Matriz de confusion (filas = real, columnas = predicho):")
    print("            " + " ".join(f"{n[:9]:>9}" for n in nombres))
    for i, fila in enumerate(cm):
        print(f"{nombres[i][:11]:>11} " + " ".join(f"{v:9d}" for v in fila))
    print("(Las secuencias grabadas seguidas se parecen mucho: esta cifra es optimista. "
          "La prueba real es usarlo en vivo.)")

    # --- Exportar a TorchScript ---
    OUT_DIR.mkdir(exist_ok=True)
    ejemplo = torch.zeros(1, T, F)
    ts = torch.jit.trace(modelo, ejemplo)
    with torch.no_grad():
        assert torch.allclose(ts(Xva_t[:8]), modelo(Xva_t[:8]), atol=1e-4), "TorchScript difiere del modelo"
    ts.save(str(OUT_DIR / "modelo_gestos.pt"))

    meta_out = {
        "seq_len": int(T),
        "n_features_entrada": int(F),
        "features": "rel_muneca(63) + desplazamiento_muneca(2) + velocidad_muneca(2)",
        "clases": meta["clases"],
        "normalizar_escala": meta["normalizar_escala"],
        "espejo": meta.get("espejo", True),
        "mano": meta.get("mano", "Right"),
        "arquitectura": args.modelo,
        "val_accuracy": acc,
        "muestras_por_clase": {nombres[i]: int(conteos[i]) for i in range(n_cls)},
    }
    (OUT_DIR / "modelo_meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    print(f"\nModelo guardado en '{OUT_DIR / 'modelo_gestos.pt'}'.")
    print("Siguiente paso: ejecuta control_gestos_presentacion.py (detecta el modelo automaticamente).")


if __name__ == "__main__":
    main()

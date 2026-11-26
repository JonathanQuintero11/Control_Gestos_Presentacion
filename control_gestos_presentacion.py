#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 CONTROL DE PRESENTACIONES POR GESTOS (PowerPoint / Google Slides / Canva)
 Version 5 - gestos dinamicos + deteccion de dedos mas robusta
==========================================================================

INSTALACION (con el venv activo):

    python -m pip install opencv-python mediapipe pyautogui numpy torch

ARCHIVOS NECESARIOS EN LA MISMA CARPETA:
    features_gestos.py             (preprocesamiento compartido)
    hand_landmarker.task           (se descarga solo si falta)
    modelo/modelo_gestos.pt        (lo genera entrenar_modelo.py)
    modelo/modelo_meta.json        (lo genera entrenar_modelo.py)

MODOS DE NAVEGACION (se elige solo al arrancar):
  * DINAMICO (si existe el modelo y torch esta instalado):
        deslizar la mano DERECHA de derecha a izquierda -> SIGUIENTE (right)
        deslizar la mano DERECHA de izquierda a derecha -> ANTERIOR  (left)
        (la mano y las direcciones dependen de como grabaste el dataset)
  * ESTATICO (respaldo si no hay modelo):
        mano DERECHA abierta -> SIGUIENTE | mano IZQUIERDA abierta -> ANTERIOR

EN AMBOS MODOS:
    Solo indice levantado -> PUNTERO (mueve el cursor, sin cooldown)
    Puno cerrado / sin mano -> REPOSO

    En modo dinamico el reconocimiento de deslizamientos se pausa mientras
    apuntas con el indice, para que mover el puntero no dispare diapositivas.

DIAGNOSTICO EN PANTALLA:
    Bajo cada mano se muestra, p. ej., "POINT 1000": el tipo detectado y los dedos
    extendidos (indice, medio, anular, menique; 1 = extendido). Si un gesto no se
    reconoce, ahi se ve por que. El pulgar NO se usa para clasificar.
    "IZQ"/"DER" es la lateralidad (depende de SWAP_HANDEDNESS).

AJUSTES RAPIDOS (arriba, en "Parametros ajustables"):
    MODO_NAV         "auto" | "estatico" | "dinamico"
    FINGER_EXT_RATIO subelo si detecta dedos extendidos que estan doblados;
                     bajalo si no reconoce dedos que si estan extendidos
    DETECT_CONF      confianza de MediaPipe (0.5 por defecto)

SALIR: Q, ESC o cerrar la ventana (X).
NOTA: cv2.putText no soporta acentos ni la n con tilde (HUD en ASCII).
==========================================================================
"""

import json
import math
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from features_gestos import construir_features, normalizar_mano

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# --------------------------------------------------------------------------
# Parametros ajustables
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODELO_GESTOS = BASE_DIR / "modelo" / "modelo_gestos.pt"
MODELO_META = BASE_DIR / "modelo" / "modelo_meta.json"

CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 960, 540, 30
WINDOW_NAME = "Control de diapositivas por gestos"

SWAP_HANDEDNESS = True   # True si ves "IZQ" sobre tu mano derecha (mismo valor que al grabar)

# "auto": usa el modelo si existe, si no el modo estatico.
# "estatico": fuerza DER abierta = SIGUIENTE / IZQ abierta = ANTERIOR (ignora el modelo).
# "dinamico": exige el modelo (si falta, avisa y cae al estatico).
MODO_NAV = "auto"

# Confianza de MediaPipe: bajarla ayuda a que "no pierda" la mano (sobre todo la izquierda)
DETECT_CONF = 0.5

COOLDOWN_NAV = 1.2        # segundos entre Siguiente / Anterior

# --- Modelo dinamico ---
UMBRAL_DINAMICO = 0.85    # probabilidad minima para aceptar un deslizamiento
CONFIRMACIONES = 2        # predicciones consecutivas que deben coincidir
PAUSAR_DINAMICO_EN_PUNTERO = True

# --- Puntero ---
ZONE_MARGIN_X = 0.15
ZONE_MARGIN_Y = 0.18
POINTER_MIN_ALPHA = 0.30
POINTER_MAX_ALPHA = 0.95
POINTER_FAST_PX = 150.0
POINTER_DEADBAND_PX = 1.0

# --- Umbrales geometricos de los dedos ---
# Un dedo se considera EXTENDIDO si su punta esta mas lejos de la muneca que su base
# (nudillo) por este factor, medido en 3D. El pulgar se IGNORA a proposito: al
# apuntar o abrir la mano casi nadie lo coloca igual, y era la causa de que el
# puntero y la mano abierta fallaran.
FINGER_EXT_RATIO = 1.30

# --------------------------------------------------------------------------
# Landmarks
# --------------------------------------------------------------------------
WRIST = 0
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP, MIDDLE_TIP = 9, 12
RING_MCP, RING_TIP = 13, 16
PINKY_MCP, PINKY_TIP = 17, 20

# nombre: (base/nudillo, punta)
LONG_FINGERS = {
    "index": (INDEX_MCP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_TIP),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

G_REST, G_NEXT, G_PREV, G_POINTER, G_NONE = (
    "REPOSO", "SIGUIENTE", "ANTERIOR", "PUNTERO LASER", "SIN GESTO")

GESTURE_COLORS = {  # BGR
    G_REST: (180, 180, 180), G_NEXT: (0, 220, 0), G_PREV: (255, 160, 0),
    G_POINTER: (0, 0, 255), G_NONE: (120, 120, 120),
}
NAV_DINAMICA_COLOR = (200, 200, 0)

STABLE_FRAMES = {G_POINTER: 2, G_NEXT: 6, G_PREV: 6, G_REST: 3, G_NONE: 2}


# --------------------------------------------------------------------------
# Detector de gestos dinamicos (modelo TorchScript)
# --------------------------------------------------------------------------
class DetectorDinamico:
    """Mantiene una ventana deslizante de SEQ_LEN fotogramas y clasifica el movimiento."""

    def __init__(self, modelo, torch_mod, meta):
        self.modelo = modelo
        self.torch = torch_mod
        self.seq_len = int(meta["seq_len"])
        self.escala = bool(meta["normalizar_escala"])
        self.mano = meta.get("mano", "Right")
        self.clases = {int(k): v for k, v in meta["clases"].items()}
        self.id_sig = next(k for k, v in self.clases.items() if v == "siguiente")
        self.id_ant = next(k for k, v in self.clases.items() if v == "anterior")
        self.buf_rel = deque(maxlen=self.seq_len)
        self.buf_mun = deque(maxlen=self.seq_len)

    @property
    def llenado(self):
        return len(self.buf_rel)

    def reset(self):
        self.buf_rel.clear()
        self.buf_mun.clear()

    def actualizar(self, landmarks, w, h):
        """Anade un fotograma; devuelve probabilidades por clase cuando la ventana esta llena."""
        rel, mun = normalizar_mano(landmarks, w, h, self.escala)
        self.buf_rel.append(rel)
        self.buf_mun.append(mun)
        if len(self.buf_rel) < self.seq_len:
            return None
        feats = construir_features(np.stack(self.buf_rel), np.stack(self.buf_mun))
        with self.torch.no_grad():
            logits = self.modelo(self.torch.from_numpy(feats[None]).float())
            return self.torch.softmax(logits, dim=1)[0].numpy()


def cargar_detector_dinamico():
    """Devuelve un DetectorDinamico o None (y explica por que) para usar el modo estatico."""
    if not (MODELO_GESTOS.is_file() and MODELO_META.is_file()):
        print("[AVISO] No se encontro el modelo entrenado en 'modelo/'. Usando navegacion ESTATICA.")
        return None
    try:
        import torch
    except ImportError:
        print("[AVISO] PyTorch no esta instalado (python -m pip install torch). Usando navegacion ESTATICA.")
        return None
    try:
        meta = json.loads(MODELO_META.read_text(encoding="utf-8"))
        modelo = torch.jit.load(str(MODELO_GESTOS), map_location="cpu").eval()
        torch.set_num_threads(1)
        det = DetectorDinamico(modelo, torch, meta)
    except Exception as exc:  # noqa: BLE001
        print(f"[AVISO] No se pudo cargar el modelo ({exc}). Usando navegacion ESTATICA.")
        return None
    print(f"[INFO] Modelo dinamico cargado ({meta.get('arquitectura', '?')}, "
          f"val_acc={meta.get('val_accuracy', 0):.2f}, mano={det.mano}). Navegacion DINAMICA.")
    return det


# --------------------------------------------------------------------------
# Gestos estaticos (puntero y respaldo)
# --------------------------------------------------------------------------
def finger_flags(p3):
    """
    p3: array (21, 3) en pixeles (x*w, y*h, z*w). Devuelve {dedo: extendido}.
    Un dedo esta extendido si su punta esta mas lejos de la muneca que su nudillo
    por FINGER_EXT_RATIO. Se mide en 3D para no fallar cuando la mano apunta
    hacia la camara (en 2D el dedo se "acorta" y parecia doblado).
    """
    muneca = p3[WRIST]
    flags = {}
    for nombre, (base, punta) in LONG_FINGERS.items():
        d_base = max(float(np.linalg.norm(p3[base] - muneca)), 1e-6)
        d_punta = float(np.linalg.norm(p3[punta] - muneca))
        flags[nombre] = d_punta > d_base * FINGER_EXT_RATIO
    return flags


def classify_hand(p3):
    """Devuelve (tipo, flags). Tipos: FIST, OPEN (4 dedos), POINT (solo indice), OTHER."""
    flags = finger_flags(p3)
    n = sum(flags.values())
    if n == 0:
        kind = "FIST"
    elif n == 4:
        kind = "OPEN"
    elif n == 1 and flags["index"]:
        kind = "POINT"
    else:
        kind = "OTHER"
    return kind, flags


def classify_frame(kinds, labels, nav_estatica):
    """Devuelve (gesto, indice_de_mano). OPEN solo navega en modo estatico."""
    active = []
    for i, (kind, label) in enumerate(zip(kinds, labels)):
        if kind == "OPEN" and nav_estatica:
            active.append((G_NEXT if label == "Right" else G_PREV, i))
        elif kind == "POINT":
            active.append((G_POINTER, i))
    if len(active) == 1:
        return active[0]
    if len(active) > 1:
        return G_NONE, None
    if all(k == "FIST" for k in kinds):
        return G_REST, None
    return G_NONE, None


def hand_label(result, i):
    try:
        name = result.handedness[i][0].category_name
    except (IndexError, AttributeError):
        name = "Right"
    if SWAP_HANDEDNESS:
        name = "Left" if name == "Right" else "Right"
    return name


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------
def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hand(img, pts, color, label, info=""):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 4, color, 1, cv2.LINE_AA)
    put_text(img, "IZQ" if label == "Left" else "DER",
             (int(pts[WRIST][0]) - 20, int(pts[WRIST][1]) + 30), 0.6, color, 2)
    if info:   # diagnostico: tipo detectado y dedos extendidos (indice, medio, anular, menique)
        put_text(img, info, (int(pts[WRIST][0]) - 40, int(pts[WRIST][1]) + 54), 0.5, (255, 255, 0), 1)


def draw_dyn_panel(img, det, probs):
    """Barras de probabilidad por clase + progreso de la ventana de fotogramas."""
    h, w = img.shape[:2]
    x0, y0, bw = w - 280, 110, 200
    put_text(img, f"Ventana {det.llenado}/{det.seq_len}", (x0, y0 - 10), 0.5, (220, 220, 220), 1)
    for i, (cid, nombre) in enumerate(sorted(det.clases.items())):
        p = float(probs[cid]) if probs is not None else 0.0
        y = y0 + 12 + i * 28
        cv2.rectangle(img, (x0, y), (x0 + bw, y + 16), (60, 60, 60), -1)
        color = (0, 220, 0) if (cid in (det.id_sig, det.id_ant) and p >= UMBRAL_DINAMICO) else (200, 200, 200)
        cv2.rectangle(img, (x0, y), (x0 + int(bw * p), y + 16), color, -1)
        cv2.line(img, (x0 + int(bw * UMBRAL_DINAMICO), y - 2), (x0 + int(bw * UMBRAL_DINAMICO), y + 18),
                 (0, 140, 255), 1)
        put_text(img, f"{nombre} {p:.2f}", (x0 + bw + 8, y + 14), 0.5, (220, 220, 220), 1)


def draw_hud(img, gesture, etiqueta, zone, cd_left, last_action, fps, n_hands, modo_dinamico,
             flash_text):
    h, w = img.shape[:2]
    zx1, zy1, zx2, zy2 = zone
    color = GESTURE_COLORS.get(gesture, (255, 255, 255))
    if modo_dinamico and etiqueta != gesture:
        color = NAV_DINAMICA_COLOR

    zone_color = (0, 0, 255) if gesture == G_POINTER else (90, 90, 90)
    cv2.rectangle(img, (zx1, zy1), (zx2, zy2), zone_color, 2 if gesture == G_POINTER else 1)
    put_text(img, "ZONA PUNTERO", (zx1 + 6, zy1 + 22), 0.5, zone_color, 1)

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 64), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    put_text(img, f"GESTO: {etiqueta}", (14, 40), 1.0, color, 2)
    put_text(img, f"FPS {fps:3.0f} | Manos {n_hands}", (w - 230, 28), 0.55, (220, 220, 220), 1)
    put_text(img, "NAV: DINAMICO (modelo)" if modo_dinamico else "NAV: ESTATICO (sin modelo)",
             (w - 230, 54), 0.5, (0, 220, 220) if modo_dinamico else (0, 140, 255), 1)

    bar_x, bar_y, bar_w, bar_h = 14, h - 44, 320, 18
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    if cd_left > 0:
        frac = min(cd_left / COOLDOWN_NAV, 1.0)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * frac), bar_y + bar_h), (0, 140, 255), -1)
        put_text(img, f"COOLDOWN {cd_left:.1f}s", (bar_x + bar_w + 12, bar_y + 15), 0.6, (0, 140, 255), 2)
    else:
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (0, 180, 0), -1)
        put_text(img, "LISTO", (bar_x + bar_w + 12, bar_y + 15), 0.6, (0, 220, 0), 2)
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)

    if last_action:
        put_text(img, f"Ultima accion: {last_action}", (14, h - 56), 0.55, (200, 200, 200), 1)
    leyenda = ("Desliza mano DER: <- SIGUIENTE | -> ANTERIOR | Indice = PUNTERO" if modo_dinamico
               else "IZQ abierta=ANTERIOR | DER abierta=SIGUIENTE | Indice=PUNTERO")
    put_text(img, leyenda, (14, 90), 0.5, (220, 220, 220), 1)
    put_text(img, "Q / ESC / X = salir", (w - 210, h - 16), 0.5, (200, 200, 200), 1)

    if flash_text:
        (tw, _), _ = cv2.getTextSize(flash_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
        put_text(img, flash_text, ((w - tw) // 2, h // 2), 2.0, (0, 255, 0), 5)


# --------------------------------------------------------------------------
# Modelo de MediaPipe y camara
# --------------------------------------------------------------------------
def ensure_model():
    if MODEL_PATH.is_file():
        return str(MODEL_PATH)
    print(f"[INFO] No se encontro '{MODEL_PATH}'. Descargando modelo...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return str(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo descargar el modelo: {exc}\n        Descargalo desde {MODEL_URL}")
        sys.exit(1)


def open_camera():
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAM_INDEX, backend)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la camara web.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    model_path = ensure_model()
    detector = None if MODO_NAV == "estatico" else cargar_detector_dinamico()
    if MODO_NAV == "dinamico" and detector is None:
        print("[AVISO] MODO_NAV='dinamico' pero no hay modelo utilizable: se usa el modo ESTATICO.")
    modo_dinamico = detector is not None
    print(f"[INFO] Navegacion: {'DINAMICA (modelo)' if modo_dinamico else 'ESTATICA (DER abierta = siguiente, IZQ abierta = anterior)'}")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=DETECT_CONF,
        min_hand_presence_confidence=DETECT_CONF,
        min_tracking_confidence=DETECT_CONF,
    )

    cap = open_camera()
    screen_w, screen_h = pyautogui.size()
    cv2.namedWindow(WINDOW_NAME)

    candidate, cand_count, stable = G_REST, 0, G_REST
    last_nav_time = -1e9
    last_action_label = ""
    pointer_pos = None
    confirm_cls, confirm_n = None, 0
    flash_text, flash_until = "", 0.0
    probs = None
    last_ts_ms, prev_t, fps = 0, time.monotonic(), 0.0

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[ERROR] No se pudo leer un fotograma de la camara.")
                    break

                frame = cv2.flip(frame, 1)       # espejo ANTES de detectar (igual que al grabar)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = max(int(time.monotonic() * 1000), last_ts_ms + 1)
                last_ts_ms = ts_ms
                result = landmarker.detect_for_video(mp_image, ts_ms)

                hands3 = [np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in hand], dtype=np.float32)
                          for hand in result.hand_landmarks]
                hands = [[(float(x), float(y)) for x, y, _ in p3] for p3 in hands3]
                labels = [hand_label(result, i) for i in range(len(hands))]
                now = time.monotonic()

                clasif = [classify_hand(p3) for p3 in hands3]
                kinds = [k for k, _ in clasif]
                raw, raw_idx = classify_frame(kinds, labels, nav_estatica=not modo_dinamico)

                if raw == candidate:
                    cand_count += 1
                else:
                    candidate, cand_count = raw, 1
                if cand_count >= STABLE_FRAMES[candidate]:
                    stable = candidate
                act = stable if stable == raw else G_NONE

                nav_left = max(0.0, last_nav_time + COOLDOWN_NAV - now)
                zone = (int(w * ZONE_MARGIN_X), int(h * ZONE_MARGIN_Y),
                        int(w * (1 - ZONE_MARGIN_X)), int(h * (1 - ZONE_MARGIN_Y)))

                # ---------------- Navegacion DINAMICA (modelo) ----------------
                hay_mano_dyn = False
                if modo_dinamico:
                    idx_dyn = None
                    for i, lab in enumerate(labels):
                        if detector.mano == "Any" or lab == detector.mano:
                            idx_dyn = i
                            break
                    apuntando = any(k == "POINT" for k in kinds)
                    if idx_dyn is None or (PAUSAR_DINAMICO_EN_PUNTERO and apuntando):
                        detector.reset()
                        probs, confirm_cls, confirm_n = None, None, 0
                    else:
                        hay_mano_dyn = True
                        probs = detector.actualizar(result.hand_landmarks[idx_dyn], w, h)
                        if probs is not None:
                            c = int(np.argmax(probs))
                            p = float(probs[c])
                            if c in (detector.id_sig, detector.id_ant) and p >= UMBRAL_DINAMICO:
                                confirm_n = confirm_n + 1 if c == confirm_cls else 1
                                confirm_cls = c
                            else:
                                confirm_cls, confirm_n = None, 0

                            if confirm_n >= CONFIRMACIONES and nav_left == 0:
                                if c == detector.id_sig:
                                    pyautogui.press("right")
                                    last_action_label, flash_text = "SIGUIENTE (swipe)", "SIGUIENTE >>"
                                else:
                                    pyautogui.press("left")
                                    last_action_label, flash_text = "ANTERIOR (swipe)", "<< ANTERIOR"
                                last_nav_time, flash_until = now, now + 0.8
                                detector.reset()          # evita re-disparos con el mismo movimiento
                                probs, confirm_cls, confirm_n = None, None, 0

                # ---------------- Puntero y navegacion ESTATICA ----------------
                if act != G_POINTER:
                    pointer_pos = None

                if act == G_POINTER and raw_idx is not None:
                    tip = hands[raw_idx][INDEX_TIP]
                    tx = float(np.interp(tip[0], (zone[0], zone[2]), (0, screen_w - 1)))
                    ty = float(np.interp(tip[1], (zone[1], zone[3]), (0, screen_h - 1)))
                    if pointer_pos is None:
                        cx, cy = pyautogui.position()
                        pointer_pos = [float(cx), float(cy)]
                    d = math.hypot(tx - pointer_pos[0], ty - pointer_pos[1])
                    if d > POINTER_DEADBAND_PX:
                        speed = min(d / POINTER_FAST_PX, 1.0)
                        alpha = POINTER_MIN_ALPHA + (POINTER_MAX_ALPHA - POINTER_MIN_ALPHA) * speed
                        pointer_pos[0] += (tx - pointer_pos[0]) * alpha
                        pointer_pos[1] += (ty - pointer_pos[1]) * alpha
                        pyautogui.moveTo(int(pointer_pos[0]), int(pointer_pos[1]), _pause=False)

                elif act == G_NEXT and nav_left == 0:
                    pyautogui.press("right")
                    last_nav_time, last_action_label = now, "SIGUIENTE (right)"

                elif act == G_PREV and nav_left == 0:
                    pyautogui.press("left")
                    last_nav_time, last_action_label = now, "ANTERIOR (left)"

                cd_left = max(0.0, last_nav_time + COOLDOWN_NAV - time.monotonic())

                # ---------------- HUD ----------------
                etiqueta = stable
                if modo_dinamico and hay_mano_dyn and stable in (G_REST, G_NONE):
                    etiqueta = "NAVEGACION (modelo)"
                gcolor = GESTURE_COLORS.get(stable, (255, 255, 255))
                for p_, lab, (kind_, fl_) in zip(hands, labels, clasif):
                    dedos = "".join("1" if fl_[n] else "0" for n in ("index", "middle", "ring", "pinky"))
                    draw_hand(frame, p_, gcolor, lab, f"{kind_} {dedos}")
                if stable == G_POINTER and raw_idx is not None:
                    tip = hands[raw_idx][INDEX_TIP]
                    cv2.circle(frame, (int(tip[0]), int(tip[1])), 12, (0, 0, 255), 2, cv2.LINE_AA)

                fps = 0.9 * fps + 0.1 * (1.0 / max(time.monotonic() - prev_t, 1e-6))
                prev_t = time.monotonic()
                draw_hud(frame, stable, etiqueta, zone, cd_left, last_action_label, fps, len(hands),
                         modo_dinamico, flash_text if now < flash_until else "")
                if modo_dinamico:
                    draw_dyn_panel(frame, detector, probs)

                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Programa finalizado.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n[INFO] Interrumpido por el usuario.")
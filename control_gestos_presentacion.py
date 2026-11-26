#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 CONTROL DE PRESENTACIONES POR GESTOS (PowerPoint / Google Slides / Canva)
 Version 3.1
==========================================================================

INSTALACION (una sola vez, con el venv activo):

    python -m pip install opencv-python mediapipe pyautogui numpy

MODELO:
    Necesita `hand_landmarker.task` junto a este script. Si no existe, el
    programa intenta descargarlo automaticamente.

GESTOS:
- Sin mano / puno cerrado ............ REPOSO (no hace nada)
- MANO DERECHA abierta (5 dedos) ..... SIGUIENTE -> tecla `right`
- MANO IZQUIERDA abierta (5 dedos) ... ANTERIOR  -> tecla `left`
- Solo indice levantado (cualquier mano) PUNTERO -> mueve el cursor

    Si las dos manos estan abiertas a la vez, o una abierta y otra con
    indice, se considera "SIN GESTO" para evitar acciones accidentales.

TECLAS DE LA VENTANA:
    Q / ESC / X ... salir

USO:
    Haz clic en tu presentacion para que tenga el foco (PyAutoGUI envia las
    teclas a la ventana activa) y despues muestra los gestos a la camara.
    En pantalla aparece "IZQ" / "DER" sobre cada mano: asi verificas que
    MediaPipe distingue bien cual es cual. Si las ves al reves, pon
    SWAP_HANDEDNESS = True.

NOTA: cv2.putText no soporta acentos ni la n con tilde, por eso el HUD
      esta escrito en ASCII.
==========================================================================
"""

import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# PyAutoGUI sin retrasos
# --------------------------------------------------------------------------
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# --------------------------------------------------------------------------
# Parametros ajustables
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 960, 540, 30
WINDOW_NAME = "Control de diapositivas por gestos"

# Si ves "IZQ" sobre tu mano derecha (y viceversa), cambia esto a True
SWAP_HANDEDNESS = True

COOLDOWN_NAV = 1.2        # segundos entre Siguiente / Anterior

# --- Puntero ---
ZONE_MARGIN_X = 0.10      # zona activa: margenes horizontales
ZONE_MARGIN_TOP = 0.19    # margen superior (deja libre la barra de texto)
ZONE_MARGIN_BOTTOM = 0.14 # margen inferior (deja libre la barra de cooldown)
POINTER_MIN_ALPHA = 0.30  # suavizado con la mano casi quieta (anti-temblor)
POINTER_MAX_ALPHA = 0.95  # suavizado con movimiento rapido (casi sin retardo)
POINTER_FAST_PX = 150.0   # distancia (px de pantalla) considerada "movimiento rapido"
POINTER_DEADBAND_PX = 1.0

# --- Umbrales geometricos de los dedos ---
FINGER_EXT_RATIO = 1.15   # punta debe estar >15 % mas lejos de la muneca que el PIP
THUMB_SPREAD = 0.55       # separacion minima pulgar-indice (en "palmas")

# --------------------------------------------------------------------------
# Landmarks (MediaPipe Hands, 21 puntos)
# --------------------------------------------------------------------------
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20

LONG_FINGERS = {
    "index": (INDEX_TIP, INDEX_PIP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP),
    "ring": (RING_TIP, RING_PIP),
    "pinky": (PINKY_TIP, PINKY_PIP),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# --------------------------------------------------------------------------
# Gestos del sistema
# --------------------------------------------------------------------------
G_REST, G_NEXT, G_PREV, G_POINTER, G_NONE = (
    "REPOSO", "SIGUIENTE", "ANTERIOR", "PUNTERO LASER", "SIN GESTO")

GESTURE_COLORS = {  # BGR
    G_REST: (180, 180, 180),
    G_NEXT: (0, 220, 0),
    G_PREV: (255, 160, 0),
    G_POINTER: (0, 0, 255),
    G_NONE: (120, 120, 120),
}

# Fotogramas consecutivos que un gesto debe mantenerse para validarse.
STABLE_FRAMES = {G_POINTER: 2, G_NEXT: 6, G_PREV: 6, G_REST: 3, G_NONE: 2}


# --------------------------------------------------------------------------
# Geometria
# --------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(p):
    """Devuelve {dedo: extendido} para los 5 dedos de una mano (landmarks en px)."""
    palm = max(dist(p[WRIST], p[MIDDLE_MCP]), 1e-6)
    states = {}
    for name, (tip, pip) in LONG_FINGERS.items():
        states[name] = dist(p[tip], p[WRIST]) > dist(p[pip], p[WRIST]) * FINGER_EXT_RATIO
    states["thumb"] = (
        dist(p[THUMB_TIP], p[PINKY_MCP]) > dist(p[THUMB_IP], p[PINKY_MCP]) * 1.05
        and dist(p[THUMB_TIP], p[INDEX_MCP]) > THUMB_SPREAD * palm
    )
    return states


def classify_hand(p):
    """Clasifica UNA mano: FIST, OPEN (5 dedos), POINT (solo indice) u OTHER."""
    st = finger_states(p)
    count = sum(st.values())
    if count == 0:
        return "FIST"
    if count == 5:
        return "OPEN"
    if st["index"] and count == 1:
        return "POINT"
    return "OTHER"


def classify_frame(kinds, labels):
    """
    Combina todas las manos visibles en UN gesto global.
    Devuelve (gesto, indice_de_la_mano_que_lo_origina).
    """
    active = []
    for i, (kind, label) in enumerate(zip(kinds, labels)):
        if kind == "OPEN":
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


# --------------------------------------------------------------------------
# Dibujo / HUD
# --------------------------------------------------------------------------
def put_text(img, text, org, scale=0.7, color=(255, 255, 255), thick=2):
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hand(img, pts, color, label):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), color, 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), 4, color, 1, cv2.LINE_AA)
    put_text(img, "IZQ" if label == "Left" else "DER",
             (int(pts[WRIST][0]) - 20, int(pts[WRIST][1]) + 30), 0.6, color, 2)


def draw_hud(img, gesture, zone, cd_left, last_action, fps, n_hands):
    h, w = img.shape[:2]
    zx1, zy1, zx2, zy2 = zone
    color = GESTURE_COLORS.get(gesture, (255, 255, 255))

    # 1. Panel superior completo
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    put_text(img, f"GESTO: {gesture}", (14, 38), 0.95, color, 2)
    put_text(img, f"FPS {fps:3.0f} | Manos {n_hands}", (w - 230, 38), 0.55, (220, 220, 220), 1)
    put_text(img, "IZQ abierta=ANTERIOR | DER abierta=SIGUIENTE | Indice=PUNTERO",
             (14, 74), 0.52, (220, 220, 220), 1)

    # 2. Zona activa del puntero
    zone_color = (0, 0, 255) if gesture == G_POINTER else (90, 90, 90)
    cv2.rectangle(img, (zx1, zy1), (zx2, zy2), zone_color, 2 if gesture == G_POINTER else 1)
    put_text(img, "ZONA PUNTERO", (zx1 + 8, zy1 + 22), 0.5, zone_color, 1)

    # 3. Indicador de cooldown inferior
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
    put_text(img, "Q / ESC / X = salir", (w - 210, h - 16), 0.5, (200, 200, 200), 1)


# --------------------------------------------------------------------------
# Modelo y camara
# --------------------------------------------------------------------------
def ensure_model():
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    print(f"[INFO] No se encontro '{MODEL_PATH}'. Descargando modelo...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Modelo descargado correctamente.")
        return MODEL_PATH
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo descargar el modelo: {exc}")
        print(f"        Descargalo manualmente desde:\n        {MODEL_URL}\n"
              f"        y guardalo como: {MODEL_PATH}")
        sys.exit(1)


def open_camera():
    """Abre la camara con ajustes orientados a baja latencia."""
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


def hand_label(result, i):
    try:
        name = result.handedness[i][0].category_name
    except (IndexError, AttributeError):
        name = "Right"
    if SWAP_HANDEDNESS:
        name = "Left" if name == "Right" else "Right"
    return name


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    model_path = ensure_model()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = open_camera()
    screen_w, screen_h = pyautogui.size()
    cv2.namedWindow(WINDOW_NAME)

    # --- Estado ---
    candidate, cand_count, stable = G_REST, 0, G_REST
    last_nav_time = -1e9
    last_action_label = ""
    pointer_pos = None
    last_ts_ms = 0
    prev_t = time.monotonic()
    fps = 0.0

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[ERROR] No se pudo leer un fotograma de la camara.")
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                # --- Deteccion ---
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = max(int(time.monotonic() * 1000), last_ts_ms + 1)
                last_ts_ms = ts_ms
                result = landmarker.detect_for_video(mp_image, ts_ms)

                hands = [[(lm.x * w, lm.y * h) for lm in hand] for hand in result.hand_landmarks]
                labels = [hand_label(result, i) for i in range(len(hands))]

                now = time.monotonic()

                # --- Clasificacion + antirrebote ---
                kinds = [classify_hand(p) for p in hands]
                raw, raw_idx = classify_frame(kinds, labels)

                if raw == candidate:
                    cand_count += 1
                else:
                    candidate, cand_count = raw, 1
                if cand_count >= STABLE_FRAMES[candidate]:
                    stable = candidate

                act = stable if stable == raw else G_NONE

                nav_left = max(0.0, last_nav_time + COOLDOWN_NAV - now)
                zone = (int(w * ZONE_MARGIN_X), int(h * ZONE_MARGIN_TOP),
                        int(w * (1 - ZONE_MARGIN_X)), int(h * (1 - ZONE_MARGIN_BOTTOM)))

                # --- Acciones ---
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

                # --- HUD ---
                gcolor = GESTURE_COLORS.get(stable, (255, 255, 255))
                for p, lab in zip(hands, labels):
                    draw_hand(frame, p, gcolor, lab)
                if stable == G_POINTER and raw_idx is not None:
                    tip = hands[raw_idx][INDEX_TIP]
                    cv2.circle(frame, (int(tip[0]), int(tip[1])), 12, (0, 0, 255), 2, cv2.LINE_AA)

                fps = 0.9 * fps + 0.1 * (1.0 / max(time.monotonic() - prev_t, 1e-6))
                prev_t = time.monotonic()
                draw_hud(frame, stable, zone, cd_left, last_action_label, fps, len(hands))

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
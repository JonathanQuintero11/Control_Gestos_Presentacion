#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
 recolector_datos.py
 Recolector de secuencias de gestos dinamicos (ventanas de 20 fotogramas)
 para entrenar un LSTM / CNN 1D.
==========================================================================

REQUISITOS:
    python -m pip install opencv-python mediapipe numpy
    (y `hand_landmarker.task` junto a este script; si falta se descarga solo)

CONTROLES:
    0 ........ graba una secuencia de la clase NEUTRO
    1 ........ graba una secuencia de la clase SIGUIENTE (deslizar der -> izq)
    2 ........ graba una secuencia de la clase ANTERIOR  (deslizar izq -> der)
    U ........ deshace (borra) la ultima secuencia guardada
    C ........ cancela la cuenta atras / grabacion en curso
    Q / ESC / X ... salir

FLUJO DE CADA GRABACION:
    pulsas una tecla -> cuenta atras (COUNTDOWN_S) para que te coloques ->
    se capturan SEQ_LEN fotogramas consecutivos -> se guarda automaticamente.
    Si la mano se pierde (o no es la mano configurada) durante la captura,
    la secuencia se DESCARTA, para no entrenar con datos incompletos.

SALIDA (carpeta ./dataset/, se acumula entre sesiones):
    X.npy          (N, 20, 63)  float32  landmarks relativos a la muneca
                                         (21 puntos x [x, y, z] por fotograma)
    X_muneca.npy   (N, 20, 2)   float32  trayectoria ABSOLUTA de la muneca
                                         (x, y normalizados 0-1 en la imagen)
    y.npy          (N,)         int64    0 = neutro, 1 = siguiente, 2 = anterior
    meta.json                            configuracion usada (para reproducirla
                                         identica en el script de inferencia)

IMPORTANTE - POR QUE EXISTE X_muneca.npy:
    Al restar la muneca a los 21 puntos, X conserva la FORMA de la mano pero
    elimina su DESPLAZAMIENTO por la imagen... y un deslizamiento es justo
    un desplazamiento. Por eso se guarda aparte la trayectoria de la muneca.
    Opciones al entrenar:
      a) Concatenar: np.concatenate([X, X_muneca], axis=-1)  -> (N, 20, 65)
      b) Usar velocidades: np.diff(X_muneca, axis=1)
    Recomendado: (a) o (a)+(b). Con solo X el modelo apenas podria distinguir
    "siguiente" de "anterior".

NOTA: la imagen se voltea (espejo) ANTES de detectar, igual que en
      control_gestos_presentacion.py. En el script de inferencia hay que
      aplicar exactamente el mismo preprocesamiento.
==========================================================================
"""

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------
SEQ_LEN = 20                      # fotogramas por secuencia
N_LANDMARKS = 21
N_FEATURES = N_LANDMARKS * 3      # 63 valores por fotograma

CLASES = {0: "neutro", 1: "siguiente", 2: "anterior"}
TECLAS_CLASE = {ord("0"): 0, ord("1"): 1, ord("2"): 2}
CLASE_COLORES = {0: (180, 180, 180), 1: (0, 220, 0), 2: (255, 160, 0)}  # BGR

COUNTDOWN_S = 1.5                 # cuenta atras antes de grabar
ACCEPT_HAND = "Any"           # "Right" | "Left" | "Any": mano valida para grabar
SWAP_HANDEDNESS = True            # pon True si el HUD muestra IZQ sobre tu mano derecha

# True: ademas de restar la muneca, divide por el tamano de la mano (muneca ->
# base del dedo medio). Hace los datos invariantes a la distancia a la camara.
# False: solo resta la muneca y divide por el ancho del fotograma.
NORMALIZAR_ESCALA = True

CAM_INDEX = 0
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 960, 540, 30
WINDOW_NAME = "Recolector de datos - gestos dinamicos"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# --------------------------------------------------------------------------
# Normalizacion (debe replicarse IDENTICA en el script de inferencia)
# --------------------------------------------------------------------------
def normalizar_mano(landmarks, w, h):
    """
    landmarks: 21 NormalizedLandmark de MediaPipe.
    Devuelve:
      rel    (63,)  puntos relativos a la muneca, aplanados [x0,y0,z0,x1,...]
      muneca (2,)   posicion absoluta de la muneca (x, y en 0-1)
    Se trabaja en espacio "pixel" (x*w, y*h, z*w) para no deformar la mano por
    la relacion de aspecto; MediaPipe indica que z usa aproximadamente la
    misma escala que x.
    """
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
    rel = pts - pts[0]                                   # punto 0 = muneca -> origen
    if NORMALIZAR_ESCALA:
        escala = float(np.linalg.norm(rel[9, :2]))       # muneca -> base del dedo medio
        rel /= max(escala, 1e-6)
    else:
        rel /= float(w)
    muneca = np.array([landmarks[0].x, landmarks[0].y], dtype=np.float32)
    return rel.reshape(-1), muneca


def etiqueta_mano(result, i):
    """'Left'/'Right' de la mano i (la imagen ya esta en espejo, modo selfie)."""
    try:
        nombre = result.handedness[i][0].category_name
    except (IndexError, AttributeError):
        nombre = "Right"
    if SWAP_HANDEDNESS:
        nombre = "Left" if nombre == "Right" else "Right"
    return nombre


# --------------------------------------------------------------------------
# Persistencia
# --------------------------------------------------------------------------
def meta_actual():
    return {
        "seq_len": SEQ_LEN,
        "n_features": N_FEATURES,
        "clases": {str(k): v for k, v in CLASES.items()},
        "normalizar_escala": NORMALIZAR_ESCALA,
        "espejo": True,
        "mano": ACCEPT_HAND,
    }


def cargar_dataset():
    """Carga lo acumulado en sesiones anteriores (o crea listas vacias)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = DATA_DIR / "meta.json"

    if meta_path.exists():
        previo = json.loads(meta_path.read_text(encoding="utf-8"))
        for clave in ("seq_len", "n_features", "normalizar_escala", "espejo"):
            if previo.get(clave) != meta_actual()[clave]:
                print(f"[ERROR] El dataset existente se creo con {clave}={previo.get(clave)!r} y ahora "
                      f"tienes {meta_actual()[clave]!r}. Mezclarlos corromperia los datos.\n"
                      f"        Restaura el valor original o mueve/borra la carpeta '{DATA_DIR}'.")
                sys.exit(1)

    archivos = [DATA_DIR / n for n in ("X.npy", "X_muneca.npy", "y.npy")]
    if all(a.exists() for a in archivos):
        X, Xm, y = (np.load(a) for a in archivos)
        if X.shape[1:] != (SEQ_LEN, N_FEATURES) or not (len(X) == len(Xm) == len(y)):
            print("[ERROR] Los archivos de dataset existentes tienen formas inconsistentes.")
            sys.exit(1)
        print(f"[INFO] Dataset cargado: {len(y)} secuencias previas.")
        return list(X), list(Xm), list(y)
    return [], [], []


def guardar_dataset(X_list, Xm_list, y_list):
    """Guarda con escritura atomica (archivo temporal + reemplazo) para no corromper nada."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if X_list:
        X = np.stack(X_list).astype(np.float32)
        Xm = np.stack(Xm_list).astype(np.float32)
        y = np.asarray(y_list, dtype=np.int64)
    else:
        X = np.zeros((0, SEQ_LEN, N_FEATURES), dtype=np.float32)
        Xm = np.zeros((0, SEQ_LEN, 2), dtype=np.float32)
        y = np.zeros((0,), dtype=np.int64)

    for nombre, arr in (("X", X), ("X_muneca", Xm), ("y", y)):
        tmp = DATA_DIR / f"{nombre}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, DATA_DIR / f"{nombre}.npy")
    (DATA_DIR / "meta.json").write_text(json.dumps(meta_actual(), indent=2, ensure_ascii=False),
                                        encoding="utf-8")


# --------------------------------------------------------------------------
# Modelo y camara
# --------------------------------------------------------------------------
def asegurar_modelo():
    if MODEL_PATH.is_file():
        return str(MODEL_PATH)
    print(f"[INFO] No se encontro '{MODEL_PATH}'. Descargando modelo...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        return str(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] No se pudo descargar el modelo: {exc}\n        Descargalo desde {MODEL_URL}")
        sys.exit(1)


def abrir_camara():
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
# HUD
# --------------------------------------------------------------------------
def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def dibujar_mano(img, pts_px, color):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, tuple(pts_px[a]), tuple(pts_px[b]), color, 2, cv2.LINE_AA)
    for p in pts_px:
        cv2.circle(img, tuple(p), 3, (255, 255, 255), -1, cv2.LINE_AA)


def dibujar_hud(img, estado, clase, restante, n_grabados, conteos, mensaje, fps, mano_det):
    h, w = img.shape[:2]

    # Panel superior
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    if estado == "IDLE":
        put_text(img, "LISTO - pulsa 0 / 1 / 2 para grabar", (14, 42), 0.9, (0, 220, 0), 2)
    elif estado == "CUENTA":
        c = CLASE_COLORES[clase]
        put_text(img, f"PREPARATE: {CLASES[clase].upper()}  {restante:.1f}s", (14, 42), 0.9, c, 2)
        put_text(img, str(int(restante) + 1), (w // 2 - 30, h // 2 + 40), 4.0, c, 8)
    elif estado == "GRABANDO":
        put_text(img, f"GRABANDO: {CLASES[clase].upper()}", (14, 42), 0.9, (0, 0, 255), 2)
        bw = 260
        x0 = w - bw - 20
        cv2.rectangle(img, (x0, 28), (x0 + bw, 50), (60, 60, 60), -1)
        cv2.rectangle(img, (x0, 28), (x0 + int(bw * n_grabados / SEQ_LEN), 50), (0, 0, 255), -1)
        put_text(img, f"{n_grabados}/{SEQ_LEN}", (x0 + bw // 2 - 22, 46), 0.55)

    # Mano detectada
    if mano_det is None:
        put_text(img, "Mano: NO DETECTADA", (14, 94), 0.55, (0, 140, 255), 1)
    else:
        txt = "IZQ" if mano_det == "Left" else "DER"
        ok = ACCEPT_HAND == "Any" or mano_det == ACCEPT_HAND
        put_text(img, f"Mano: {txt}" + ("" if ok else "  (no es la configurada)"), (14, 94), 0.55,
                 (0, 220, 0) if ok else (0, 140, 255), 1)
    put_text(img, f"FPS {fps:3.0f}", (w - 100, 94), 0.55, (220, 220, 220), 1)

    # Contadores por clase
    y0 = 130
    for k, nombre in CLASES.items():
        put_text(img, f"[{k}] {nombre}: {conteos[k]}", (14, y0), 0.6, CLASE_COLORES[k], 1)
        y0 += 26
    put_text(img, f"TOTAL: {sum(conteos)}", (14, y0), 0.6, (255, 255, 255), 1)

    if mensaje:
        put_text(img, mensaje, (14, h - 50), 0.6, (0, 255, 255), 1)
    put_text(img, "0/1/2 grabar | U deshacer | C cancelar | Q/ESC/X salir", (14, h - 16), 0.5, (200, 200, 200), 1)


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def main():
    X_list, Xm_list, y_list = cargar_dataset()
    model_path = asegurar_modelo()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = abrir_camara()
    cv2.namedWindow(WINDOW_NAME)

    estado = "IDLE"                   # IDLE -> CUENTA -> GRABANDO -> IDLE
    clase = 0
    t_inicio_grab = 0.0               # instante en que termina la cuenta atras
    t_primer_frame = 0.0
    buf_rel, buf_mun = [], []
    mensaje, mensaje_hasta = "", 0.0
    last_ts_ms, prev_t, fps = 0, time.monotonic(), 0.0

    def avisar(texto, seg=3.0):
        nonlocal mensaje, mensaje_hasta
        mensaje, mensaje_hasta = texto, time.monotonic() + seg
        print(f"[INFO] {texto}")

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("[ERROR] No se pudo leer un fotograma de la camara.")
                    break

                frame = cv2.flip(frame, 1)                    # espejo (igual que en inferencia)
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = max(int(time.monotonic() * 1000), last_ts_ms + 1)
                last_ts_ms = ts_ms
                result = landmarker.detect_for_video(mp_image, ts_ms)
                now = time.monotonic()

                # Elegir la mano valida (la que coincide con ACCEPT_HAND)
                idx_valida, mano_det = None, None
                for i in range(len(result.hand_landmarks)):
                    lab = etiqueta_mano(result, i)
                    if mano_det is None:
                        mano_det = lab
                    if ACCEPT_HAND == "Any" or lab == ACCEPT_HAND:
                        idx_valida, mano_det = i, lab
                        break

                # --- Maquina de estados ---
                if estado == "CUENTA" and now >= t_inicio_grab:
                    estado = "GRABANDO"
                    buf_rel.clear()
                    buf_mun.clear()
                    t_primer_frame = now

                if estado == "GRABANDO":
                    if idx_valida is None:
                        estado = "IDLE"
                        avisar("Secuencia DESCARTADA: la mano se perdio o no es la configurada.")
                    else:
                        rel, mun = normalizar_mano(result.hand_landmarks[idx_valida], w, h)
                        buf_rel.append(rel)
                        buf_mun.append(mun)
                        if len(buf_rel) == SEQ_LEN:
                            dur = max(now - t_primer_frame, 1e-6)
                            X_list.append(np.stack(buf_rel))
                            Xm_list.append(np.stack(buf_mun))
                            y_list.append(clase)
                            guardar_dataset(X_list, Xm_list, y_list)
                            n_clase = int(np.sum(np.asarray(y_list) == clase))
                            avisar(f"Guardada '{CLASES[clase]}' #{n_clase} "
                                   f"({dur:.2f}s, ~{(SEQ_LEN - 1) / dur:.0f} FPS efectivos)")
                            estado = "IDLE"

                # --- Dibujo ---
                if mensaje and now > mensaje_hasta:
                    mensaje = ""
                color_mano = CLASE_COLORES[clase] if estado != "IDLE" else (0, 200, 0)
                for i, lms in enumerate(result.hand_landmarks):
                    pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                    dibujar_mano(frame, pts_px, color_mano if i == idx_valida else (90, 90, 90))
                if estado == "GRABANDO" and len(buf_mun) > 1:   # estela de la muneca
                    trail = np.array([(int(m[0] * w), int(m[1] * h)) for m in buf_mun], dtype=np.int32)
                    cv2.polylines(frame, [trail], False, (0, 0, 255), 3, cv2.LINE_AA)

                fps = 0.9 * fps + 0.1 * (1.0 / max(time.monotonic() - prev_t, 1e-6))
                prev_t = time.monotonic()
                conteos = np.bincount(np.asarray(y_list, dtype=np.int64), minlength=len(CLASES)) \
                    if y_list else np.zeros(len(CLASES), dtype=int)
                dibujar_hud(frame, estado, clase, max(0.0, t_inicio_grab - now),
                            len(buf_rel), conteos, mensaje, fps, mano_det)
                cv2.imshow(WINDOW_NAME, frame)

                # --- Teclado ---
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in TECLAS_CLASE and estado == "IDLE":
                    clase = TECLAS_CLASE[key]
                    estado = "CUENTA"
                    t_inicio_grab = time.monotonic() + COUNTDOWN_S
                elif key in (ord("c"), ord("C")) and estado != "IDLE":
                    estado = "IDLE"
                    avisar("Grabacion cancelada.")
                elif key in (ord("u"), ord("U")) and estado == "IDLE":
                    if y_list:
                        ultima = y_list.pop()
                        X_list.pop()
                        Xm_list.pop()
                        guardar_dataset(X_list, Xm_list, y_list)
                        avisar(f"Eliminada la ultima secuencia ('{CLASES[int(ultima)]}').")
                    else:
                        avisar("No hay secuencias que deshacer.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Finalizado. Dataset en '{DATA_DIR}': {len(y_list)} secuencias.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        print("\n[INFO] Interrumpido por el usuario.")

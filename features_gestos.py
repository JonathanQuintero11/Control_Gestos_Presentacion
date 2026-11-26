# -*- coding: utf-8 -*-
"""
features_gestos.py
Preprocesamiento COMPARTIDO por entrenar_modelo.py y control_gestos_presentacion.py.
Solo depende de NumPy. Mantenerlo en un unico lugar garantiza que el modelo
vea en vivo exactamente el mismo tipo de datos con el que se entreno.

(recolector_datos.py tiene una copia equivalente de `normalizar_mano`.)
"""

import numpy as np

N_FEATURES_ENTRADA = 67   # 63 (mano relativa) + 2 (desplazamiento muneca) + 2 (velocidad muneca)


def normalizar_mano(landmarks, w, h, normalizar_escala=True):
    """
    landmarks: 21 NormalizedLandmark de MediaPipe (imagen YA volteada en espejo).
    Devuelve:
      rel    (63,)  puntos relativos a la muneca, aplanados [x0,y0,z0,x1,...]
      muneca (2,)   posicion absoluta de la muneca (x, y en 0-1)
    """
    pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=np.float32)
    rel = pts - pts[0]
    if normalizar_escala:
        escala = float(np.linalg.norm(rel[9, :2]))      # muneca -> base del dedo medio
        rel /= max(escala, 1e-6)
    else:
        rel /= float(w)
    muneca = np.array([landmarks[0].x, landmarks[0].y], dtype=np.float32)
    return rel.reshape(-1), muneca


def construir_features(rel, mun):
    """
    Combina la forma de la mano con el MOVIMIENTO de la muneca.
      rel (..., T, 63), mun (..., T, 2)  ->  (..., T, 67)
    Columnas: [ rel(63) | desplazamiento respecto al 1er fotograma (2) | velocidad por fotograma (2) ]
    Se usa el desplazamiento (no la posicion absoluta) para que el modelo no dependa
    de en que parte de la imagen empieza el gesto.
    Funciona con una sola secuencia (T, .) o con un lote (N, T, .).
    """
    rel = np.asarray(rel, dtype=np.float32)
    mun = np.asarray(mun, dtype=np.float32)
    desp = mun - mun[..., :1, :]
    vel = np.zeros_like(mun)
    vel[..., 1:, :] = mun[..., 1:, :] - mun[..., :-1, :]
    return np.concatenate([rel, desp, vel], axis=-1).astype(np.float32)

import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Configurar el modelo para detectar hasta 2 manos
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
detector = vision.HandLandmarker.create_from_options(options)

CONEXIONES_MANO = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anular
    (13, 17), (17, 18), (18, 19), (19, 20),# Meñique
    (0, 17)                                # Base de la palma
]

def distancia(p1, p2):
    return math.hypot(p2[1] - p1[1], p2[2] - p1[2])

cap = cv2.VideoCapture(0)
puntas_dedos = [8, 12, 16, 20]

while True:
    exito, img = cap.read()
    if not exito:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    resultados = detector.detect(mp_image)
    total_dedos = 0

    if resultados.hand_landmarks:
        for hand_landmarks in resultados.hand_landmarks:
            puntos = []
            alto, ancho, _ = img.shape
            
            for id, lm in enumerate(hand_landmarks):
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                puntos.append((id, cx, cy))
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            for inicio, fin in CONEXIONES_MANO:
                cv2.line(img, (puntos[inicio][1], puntos[inicio][2]), 
                         (puntos[fin][1], puntos[fin][2]), (0, 255, 0), 2)

            if puntos:
                # 1. Pulgar: comparar distancia al nudillo del meñique (punto 17)
                # Si la punta (4) está más lejos que la articulación anterior (3), está abierto
                if distancia(puntos[4], puntos[17]) > distancia(puntos[3], puntos[17]):
                    total_dedos += 1

                # 2. Índice, medio, anular y meñique: comparar punta contra articulación anterior (punta - 1)
                for punta in puntas_dedos:
                    if puntos[punta][2] < puntos[punta - 1][2]:
                        total_dedos += 1

    cv2.putText(img, f"Total dedos: {total_dedos}", (30, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 3)

    cv2.imshow("Contador de Dedos", img)

    tecla = cv2.waitKey(1) & 0xFF
    if (tecla == ord('q') or tecla == ord('Q') or tecla == 27 or 
        cv2.getWindowProperty("Contador de Dedos", cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()
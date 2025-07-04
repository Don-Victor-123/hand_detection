import mediapipe as mp
import cv2
import os
import keyboard
import time

# Inicializar MediaPipe Hands y Drawing
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Captura de video desde cámara
cap = cv2.VideoCapture(0)

# Banderas globales
saludo_realizado = False
mano_agarrando = False
ventana_movilizada = False

# Control de frecuencia para evitar movimientos repetidos por error
last_move_time = 0
MOVE_COOLDOWN = 1.0  # segundos

# === FUNCIONES PARA GESTOS ===

def mano_cerrada(hand_landmarks):
    return (
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x and
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    )

def mano_apuntando_derecha(hand_landmarks):
    return (
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    )

def mano_apuntando_izquierda(hand_landmarks):
    return (
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y and
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    )

def ok_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distancia = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    dedos_juntos = distancia < 0.05

    middle_extended = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_extended = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_extended = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

    return dedos_juntos and middle_extended and ring_extended and pinky_extended

def shutdown_gesture(index_tip, index_pip, middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip):
    return (middle_tip < middle_pip and index_tip > index_pip and ring_tip > ring_pip and pinky_tip > pinky_pip)

def minimice_gesture(index_tip, index_pip, middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip):
    return (middle_tip > middle_pip and index_tip > index_pip and ring_tip > ring_pip and pinky_tip > pinky_pip)

def mostrar_escritorio():
    keyboard.press('windows')
    keyboard.press_and_release('d')
    keyboard.release('windows')

# Utilidades para mover la ventana activa entre monitores.
def mover_ventana_derecha():
    """Mueve la ventana actual a la pantalla de la derecha y la maximiza."""
    keyboard.press_and_release('windows+shift+right')
    time.sleep(0.2)
    keyboard.press_and_release('windows+up')
    time.sleep(0.2)


def mover_ventana_izquierda():
    """Mueve la ventana actual a la pantalla de la izquierda y la maximiza."""
    keyboard.press_and_release('windows+shift+left')
    time.sleep(0.2)
    keyboard.press_and_release('windows+up')
    time.sleep(0.2)

# === BUCLE PRINCIPAL ===
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

                # === DETECCIÓN DE GESTO DE "AGARRAR Y MOVER VENTANA" ===
                if mano_cerrada(hand_landmarks) and not mano_agarrando:
                    mano_agarrando = True
                    ventana_movilizada = False
                    cv2.putText(frame, "Mano cerrada (agarrando)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 255), 2)

                elif mano_agarrando and not ventana_movilizada:
                    now = time.time()
                    if mano_apuntando_derecha(hand_landmarks) and now - last_move_time > MOVE_COOLDOWN:
                        # Mover la ventana activa a la siguiente pantalla
                        mover_ventana_derecha()
                        ventana_movilizada = True
                        mano_agarrando = False
                        last_move_time = now
                        cv2.putText(frame, "Mover derecha", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    elif mano_apuntando_izquierda(hand_landmarks) and now - last_move_time > MOVE_COOLDOWN:
                        # Mover la ventana activa a la pantalla previa
                        mover_ventana_izquierda()
                        ventana_movilizada = True
                        mano_agarrando = False
                        last_move_time = now
                        cv2.putText(frame, "Mover izquierda", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # === OTROS GESTOS ===
                if ok_gesture(hand_landmarks):
                    cv2.putText(frame, "Saludo OK (Circulo)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                elif shutdown_gesture(index_tip, index_pip, middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip):
                    cv2.putText(frame, "Saludo Shutdown", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # os.system("shutdown /s /t 1")  # Precaución

                elif minimice_gesture(index_tip, index_pip, middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip):
                    cv2.putText(frame, "Saludo Minimizar", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    if not saludo_realizado:
                        #mostrar_escritorio()
                        saludo_realizado = True
                else:
                    saludo_realizado = False
                    cv2.putText(frame, "No Saludo", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

        cv2.imshow("Boton con MediaPipe", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

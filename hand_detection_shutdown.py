import mediapipe as mp
import cv2
import keyboard
import time
import os

# Ejecutar este script con privilegios de Administrador.

# Inicializar MediaPipe Hands y Drawing
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Captura de vídeo
cap = cv2.VideoCapture(0)

# Tiempo de espera entre acciones para evitar repeticiones accidentales
last_move_time = 0
MOVE_COOLDOWN = 1.0  # segundos
shutdown_scheduled = False


# --- GESTOS ---

def mano_apuntando_derecha(hand_landmarks):
    """Indice extendido, resto de dedos doblados, pulgar hacia la izquierda."""
    return (
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        and all(
            hand_landmarks.landmark[t].y > hand_landmarks.landmark[t - 2].y
            for t in (
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            )
        )
        and hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x <
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    )


def mano_apuntando_izquierda(hand_landmarks):
    """Indice extendido, resto de dedos doblados, pulgar hacia la derecha."""
    return (
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y <
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
        and all(
            hand_landmarks.landmark[t].y > hand_landmarks.landmark[t - 2].y
            for t in (
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            )
        )
        and hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x >
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    )

def shutdown_gesture(index_tip, index_pip, middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip):
    """Devuelve True cuando solo el dedo medio está extendido."""
    return (
        middle_tip < middle_pip
        and index_tip > index_pip
        and ring_tip > ring_pip
        and pinky_tip > pinky_pip
    )


def ok_gesture(hand_landmarks):
    """Detecta el gesto OK haciendo un círculo con pulgar e índice."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    distancia = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    dedos_juntos = distancia < 0.05

    middle_extended = (
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    )
    ring_extended = (
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
        < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    )
    pinky_extended = (
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
        < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    )

    return dedos_juntos and middle_extended and ring_extended and pinky_extended


# --- MOVIMIENTO DE VENTANA ---

def mover_ventana_derecha():
    """Envía la ventana activa al monitor derecho."""
    keyboard.press_and_release('win+shift+right')
    time.sleep(0.2)


def mover_ventana_izquierda():
    """Envía la ventana activa al monitor izquierdo."""
    keyboard.press_and_release('win+shift+left')
    time.sleep(0.2)


# --- BUCLE PRINCIPAL ---
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                now = time.time()

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

                if mano_apuntando_derecha(hand_landmarks) and now - last_move_time > MOVE_COOLDOWN:
                    mover_ventana_derecha()
                    last_move_time = now
                    cv2.putText(frame, "Movido → derecha", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

                elif mano_apuntando_izquierda(hand_landmarks) and now - last_move_time > MOVE_COOLDOWN:
                    mover_ventana_izquierda()
                    last_move_time = now
                    cv2.putText(frame, "Movido ← izquierda", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

                if shutdown_gesture(index_tip, index_pip, middle_tip, middle_pip, ring_tip, ring_pip, pinky_tip, pinky_pip) and not shutdown_scheduled:
                    os.system("shutdown /r /t 2")
                    shutdown_scheduled = True
                    cv2.putText(frame, "Reinicio en 2s", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                elif ok_gesture(hand_landmarks) and shutdown_scheduled:
                    os.system("shutdown /a")
                    shutdown_scheduled = False
                    cv2.putText(frame, "Reinicio cancelado", (30, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Control de ventanas con la mano", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

import mediapipe as mp
import cv2
import keyboard
import time

# Ejecutar este script con privilegios de Administrador.

# Inicializar MediaPipe Hands y Drawing
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Tiempo de espera entre acciones para evitar repeticiones accidentales
last_move_time = 0
MOVE_COOLDOWN = 1.0  # segundos


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


# --- MOVIMIENTO DE VENTANA ---

def mover_ventana_derecha():
    """Envía la ventana activa al monitor derecho."""
    keyboard.press_and_release('win+shift+right')
    time.sleep(0.2)


def mover_ventana_izquierda():
    """Envía la ventana activa al monitor izquierdo."""
    keyboard.press_and_release('win+shift+left')
    time.sleep(0.2)

def procesar_gesto(hand_landmarks, now):
    """Detecta el gesto y mueve la ventana si corresponde."""
    global last_move_time

    if mano_apuntando_derecha(hand_landmarks) and now - last_move_time > MOVE_COOLDOWN:
        mover_ventana_derecha()
        last_move_time = now
        return "Movido → derecha"

    if mano_apuntando_izquierda(hand_landmarks) and now - last_move_time > MOVE_COOLDOWN:
        mover_ventana_izquierda()
        last_move_time = now
        return "Movido ← izquierda"

    return None


def procesar_manos(results, frame):
    """Dibuja manos y aplica la lógica de gestos."""
    now = time.time()
    for hand_landmarks in results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        mensaje = procesar_gesto(hand_landmarks, now)
        if mensaje:
            cv2.putText(frame, mensaje, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)


def main():
    """Captura vídeo y gestiona el bucle principal."""
    cap = cv2.VideoCapture(0)

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
                procesar_manos(results, frame)

            cv2.imshow("Control de ventanas con la mano", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

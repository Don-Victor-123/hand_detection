import argparse
import sys
import time

import keyboard
import cv2
import mediapipe as mp


class HandGestureDetector:
    def __init__(self,
                 max_num_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame):
        """Procesa el frame BGR y devuelve la lista de landmarks para cada mano."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results.multi_hand_landmarks or []

    @staticmethod
    def is_wave_gesture(landmarks) -> bool:
        """Detecta el gesto ‘saludo’: solo dedo medio extendido hacia arriba."""
        lm = landmarks.landmark
        h = mp.solutions.hands

        return (
            lm[h.HandLandmark.MIDDLE_FINGER_TIP].y < lm[h.HandLandmark.MIDDLE_FINGER_PIP].y and
            lm[h.HandLandmark.INDEX_FINGER_TIP].y > lm[h.HandLandmark.INDEX_FINGER_PIP].y and
            lm[h.HandLandmark.RING_FINGER_TIP].y > lm[h.HandLandmark.RING_FINGER_PIP].y and
            lm[h.HandLandmark.PINKY_TIP].y > lm[h.HandLandmark.PINKY_PIP].y
        )

    @staticmethod
    def is_pointing_right(landmarks, horiz_threshold: float = 0.1) -> bool:
        """
        Detecta el gesto ‘apuntar a la derecha’ con el dedo índice.
        - Las x de INDEX_MCP→PIP→DIP→TIP crecen (hacia la derecha).
        - La diferencia vertical (y) entre MCP y TIP es pequeña.
        - Los otros dedos están doblados.
        """
        lm = landmarks.landmark
        h = mp.solutions.hands

        mcp = lm[h.HandLandmark.INDEX_FINGER_MCP]
        pip = lm[h.HandLandmark.INDEX_FINGER_PIP]
        dip = lm[h.HandLandmark.INDEX_FINGER_DIP]
        tip = lm[h.HandLandmark.INDEX_FINGER_TIP]

        if not (mcp.x < pip.x < dip.x < tip.x):
            return False
        if abs(tip.y - mcp.y) > horiz_threshold:
            return False

        for finger in ('MIDDLE_FINGER', 'RING_FINGER', 'PINKY'):
            if lm[getattr(h.HandLandmark, f'{finger}_TIP')].y < lm[getattr(h.HandLandmark, f'{finger}_PIP')].y:
                return False

        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detector de gestos con MediaPipe y OpenCV"
    )
    parser.add_argument(
        "-c", "--camera", type=int, default=0,
        help="Índice de la cámara (por defecto 0)"
    )
    parser.add_argument(
        "--max-hands", type=int, default=1,
        help="Número máximo de manos a detectar"
    )
    parser.add_argument(
        "--min-detect-confidence", type=float, default=0.5,
        help="Confianza mínima para la detección"
    )
    parser.add_argument(
        "--min-track-confidence", type=float, default=0.5,
        help="Confianza mínima para el tracking"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {args.camera}", file=sys.stderr)
        sys.exit(1)

    detector = HandGestureDetector(
        max_num_hands=args.max_hands,
        min_detection_confidence=args.min_detect_confidence,
        min_tracking_confidence=args.min_track_confidence
    )

    pointing_active = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            hands = detector.detect(frame)

            for hl in hands:
                detector.mp_draw.draw_landmarks(
                    frame, hl, detector.mp_hands.HAND_CONNECTIONS
                )

                if detector.is_wave_gesture(hl):
                    label, color = "Saludo 👋", (0,255,0)
                elif detector.is_pointing_right(hl):
                    label, color = "Senialando derecha", (255,0,0)
                    if not pointing_active:

                        keyboard.press('win')
                        time.sleep(0.01)  # Espera para evitar múltiples activaciones
                        keyboard.press_and_release('right')
                        time.sleep(0.02)  # Espera para evitar múltiples activaciones
                        keyboard.press_and_release('right')
                        time.sleep(0.03)  # Espera para evitar múltiples activaciones
                        keyboard.press_and_release('up')    
                        time.sleep(0.04)  # Espera para evitar múltiples activaciones
                        keyboard.press_and_release('up')
                        time.sleep(0.05)  # Espera para evitar múltiples activaciones
                        keyboard.release('win')
                        time.sleep(5)  # Espera para evitar múltiples activaciones
                        pointing_active = True
                else:
                    label, color = "Sin gesto", (0,0,255)
                    pointing_active = False

                cv2.putText(
                    frame, label, (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
                )

            cv2.imshow("Detector de gestos", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

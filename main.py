import cv2
import time
from logging import getLogger, StreamHandler, INFO

import hand_tracking_module as htm


# Loggingの設定
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


def main():
    logger.info('Start capture!')
    p_time = 0
    hand_no = 0

    cap = cv2.VideoCapture(0)
    detector = htm.HandDetector()

    while True:
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_position(img, hand_no=hand_no)

        if len(lm_list) != 0:
            logger.info(lm_list[hand_no])

        # FPSの計算
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time

        cv2.putText(img, str(int(fps)) + 'fps', (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

        cv2.imshow("Image", img)

        # qキーを押すことで停止する
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info('Finish!')
            break


if __name__ == "__main__":
    main()

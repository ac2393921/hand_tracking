import cv2
import mediapipe as mp


class HandDetector:
    """
    Hand trackingの各属性値やヘルパー関数を保持する。
    """
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands,
                                         self.detection_con, self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def find_hands(self, img, draw=True):
        """
        イメージから手を認識して描画する

        :param img: キャンパスイメージ
        :param draw: 描画するかどうかの判定
        :return img: ハンドトラッキング描画後のイメージ(draw=Falseの場合元のイメージのまま)
        """
        # mediapipeはRGBしか対応していないため変換
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_lms,
                                                self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, img, hand_no=0, draw=True):
        """
        手の部位ごとの位置をリストにて抽出

        :param img: キャンパスイメージ
        :param hand_no: 手の部位の指定(0~20)
        :param draw: 描画するかどうかの判定
        :return: lm_list: 手の部位ごとの位置リスト
        """
        if hand_no < 0 or hand_no > 20:
            raise ValueError("hand_noは0~20の間です。")

        lm_list = []

        if self.results.multi_hand_landmarks:
            hand_pos = self.results.multi_hand_landmarks[hand_no]
            for idx, lm in enumerate(hand_pos.landmark):
                # lm: x, y, z
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([idx, cx, cy])

                if draw:
                    if idx == hand_no:
                        cv2.circle(img, (cx, cy), 25, (0, 0, 255), cv2.FILLED)

        return lm_list

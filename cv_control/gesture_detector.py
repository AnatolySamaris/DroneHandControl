import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
from geometry_msgs.msg import Twist
from cv_control.utils import load_keras_model

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-

class HandGestureDetector(Node):
    def __init__(self):
        super().__init__("hand_gesture_detector")

        self.sign_names = {
            1: 'one', 2: 'two', 3: 'three', 4: 'four', 
            5: 'five', 6: 'ok', 7: 'rock', 8: 'thumbs_up'
        }
        
        # Загрузка модели и скалера
        # self.model, self.scaler = load_keras_model()
        self.model = load_keras_model()
        self.get_logger().info("MODEL INITIALIZED: " + str(type(self.model)))
        
        # Подписки и публикации
        self.sub_color = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.color_callback, 10
        )
        self.sub_depth = self.create_subscription(
            Image, "/camera/camera/depth/image/rect_raw", self.depth_callback, 10
        )
        self.get_logger().info("SUBSCRIPTIONS SUCCESSFULLY CREATED!")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1)
        self.mp_draw = mp.solutions.drawing_utils
        self.cv_bridge = CvBridge()
        self.current_depth = None
        self.get_logger().info("GESTURE DETECTOR SUCCESSFULLY INITIALIZED!")
    
    def color_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.hands.process(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    cv_image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                # Извлечение ключевых точек
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                landmarks = np.array(landmarks).reshape(1, -1)
                
                # Нормализация и предсказание
                # landmarks_norm = self.scaler.transform(landmarks)
                landmarks_norm = landmarks
                gesture_id = np.argmax(self.model.predict(landmarks_norm)) + 1
                
                # Визуализация
                cv2.putText(
                    cv_image, f"Gesture: {gesture_id} ({self.sign_names[gesture_id]})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
                
                # Управление квадрокоптером
                if self.current_depth is not None:
                    throttle, roll, pitch, yaw = self.calculate_angles(hand_landmarks)
                    self.publish_command(gesture_id, throttle, roll, pitch, yaw)
        
        cv2.imshow("Hand Tracking", cv_image)
        cv2.waitKey(1)
    
    def depth_callback(self, msg):
        self.current_depth = self.cv_bridge.imgmsg_to_cv2(msg)
    
    def calculate_angles(self, hand_landmarks):
        # Ключевые точки для расчетов
        wrist = hand_landmarks.landmark[0]
        index_finger_base = hand_landmarks.landmark[5]
        pinkie_finger_base = hand_landmarks.landmark[17]

        # Перевод в нампай массивы
        wrist = np.array([wrist.x, wrist.y, wrist.z])
        index_finger_base = np.array([index_finger_base.x, index_finger_base.y, index_finger_base.z])
        pinkie_finger_base = np.array([pinkie_finger_base.x, pinkie_finger_base.y, pinkie_finger_base.z])
        
        # Векторы от запястья и средняя точка
        wrist_index_vector = index_finger_base - wrist
        wrist_pinkie_vector = pinkie_finger_base - wrist
        middle_point = (wrist + index_finger_base + pinkie_finger_base) / 3

        # Направляющие векторы
        X_vector = middle_point - wrist
        Z_vector = np.cross(wrist_pinkie_vector, wrist_index_vector)
        Y_vector = np.cross(X_vector, Z_vector)

        # Расчет кватернионов
        q_w = 0.5 * np.sqrt(X_vector[0] + Y_vector[1] + Z_vector[2])
        q_x = (Z_vector[1] + Y_vector[2]) / (4 * q_w)
        q_y = (X_vector[2] + Z_vector[0]) / (4 * q_w)
        q_z = (Y_vector[0] + X_vector[1]) / (4 * q_w)

        # Расчет углов (roll - крен, pitch - тангаж, yaw - рыскание)
        roll = np.arctan2(
            2 * (q_w * q_x + q_y * q_z),
            1 - 2 * (q_x**2 + q_y**2)
        )
        pitch = - 0.5 * np.pi + 2 * np.arctan2(
            np.sqrt(1 + 2 * (q_w * q_y - q_x * q_z)),
            np.sqrt(1 - 2 * (q_w * q_y - q_x * q_z))
        )
        yaw = np.arctan2(
            2 * (q_w * q_x + q_x * q_y),
            1 - 2 * (q_y**2 + q_z**2)
        )

        # Расчет газа (throttle)
        throttle = middle_point.z   # Лучше проверять по глубине
        
        return throttle, roll, pitch, yaw
    
    def publish_command(self, gesture_id, throttle, roll, pitch, yaw):
        cmd = Twist()

        # Управление скоростью
        speed = 1
        if gesture_id in [1, 2, 3, 4, 5]:
            speed = 0.2 * gesture_id
        
        # TODO : передача управления на коптер
        # Пример: управление квадрокоптером
        cmd.linear.x = speed * np.cos(yaw)
        cmd.linear.y = speed * np.sin(yaw)
        cmd.angular.z = roll * 0.5  # Пример управления рысканием
        
        self.cmd_pub.publish(cmd)

def main():
    rclpy.init()
    node = HandGestureDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
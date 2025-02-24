#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImagePublisher:
    def __init__(self, image_path, topic_name="/camera/image_raw"):
        # ROS 노드 초기화
        rospy.init_node('image_publisher', anonymous=True)

        # 이미지 파일 경로
        self.image_path = image_path

        # CvBridge 객체 생성
        self.bridge = CvBridge()

        # 이미지 파일을 읽어서 OpenCV 이미지로 변환
        self.image = cv2.imread(image_path)
        if self.image is None:
            rospy.logerr(f"Failed to read the image from path : {image_path}")
            return

        # 카메라 토픽에 메시지 발행할 퍼블리셔 생성
        self.pub = rospy.Publisher(topic_name, Image, queue_size=10)

        self.rate = rospy.Rate(30)  # 30 Hz로 발행 (프레임 속도 맞추기)

        while not rospy.is_shutdown():
            try:
                # OpenCV 이미지를 ROS 메시지로 변환
                ros_image = self.bridge.cv2_to_imgmsg(self.image, encoding="bgr8")

                # 이미지 메시지 발행
                self.pub.publish(ros_image)
                rospy.loginfo("Published image message")

                # 설정된 발행 속도 유지
                self.rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error occurred during image publishing: {e}")
                break

class VideoPublisher:
    def __init__(self, image_path, topic_name="/camera/image_rect_color"):
        # ROS 노드 초기화
        rospy.init_node('video_publisher', anonymous=True)

        # 비디오 파일 경로
        self.video_path = video_path

        # CvBridge 객체 생성
        self.bridge = CvBridge()

        # 비디오 캡처 객체 생성
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            rospy.logerr("Error opening video stream or file")
            rospy.signal_shutdown("Failed to open video file")

        # 카메라 토픽에 메시지 발행할 퍼블리셔 생성
        self.pub = rospy.Publisher(topic_name, Image, queue_size=10)

        # 비디오 프레임을 지속적으로 발행
        self.publish_video()

    def publish_video(self):
        rate = rospy.Rate(30)  # 30 Hz로 발행 (프레임 속도 맞추기)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.loginfo("End of video stream")
                rospy.signal_shutdown("End of video")
                break

            # OpenCV 이미지 -> ROS 메시지 변환
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            # 카메라 토픽에 이미지 발행
            self.pub.publish(ros_image)

            # 주기적으로 발행
            rate.sleep()

        # 비디오 캡처 객체 해제
        self.cap.release()

if __name__ == '__main__':
    try:
        video_path = "/home/heven/OSR/src/srlpkg/src/dataset/ww.mp4"
        img_path = "/home/heven/OSR/src/srlpkg/src/dataset/04950.jpg"
        video_publisher = VideoPublisher(video_path)
        # image_publisher = ImagePublisher(img_path)
        rospy.spin()  # ROS 노드를 계속 실행
    except rospy.ROSInterruptException:
        rospy.loginfo("Video publishing stopped.")

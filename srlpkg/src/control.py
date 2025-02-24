#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from srlpkg.msg import Lanes

class LanePIDController:
    def __init__(self):
        rospy.init_node('lane_pid_controller', anonymous=True)

        self.velocity = 1.0
        self.k_stanley = 1.0

        # 토픽 구독 및 퍼블리셔 설정
        self.lane_sub = rospy.Subscriber('/lane', Lanes, self.lane_callback)
        self.control_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        rospy.loginfo("Lane PID Controller Initialized")

    def lane_callback(self, msg):
        # 메시지 데이터 추출
        left_x = msg.left_x
        right_x = msg.right_x
        left_theta = msg.left_theta
        right_theta = msg.right_theta

        # PID x_좌표
        target_x = 320  
        current_x = (left_x + right_x) / 2.0  
        x_error = target_x - current_x
        x_control = self.PID_control(x_error)

        # PID theta
        target_theta = 90.0
        current_theta = (left_theta + right_theta) / 2.0
        theta_error = target_theta - current_theta
        theta_control = self.PID_control(theta_error)

        # Stanley x_좌표
        # x_control = self.stanley_control(x_error, theta_error)

        twist = Twist()
        twist.linear.x = self.velocity # 전진
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = math.radians(x_control) # - : CW, + : CCW

        # 제어값 퍼블리싱
        self.control_pub.publish(twist)

        rospy.loginfo(f"x_error: {x_error}, theta_error: {theta_error}, angular_z : {twist.angular.z}")
        rospy.loginfo(f"x_control: {x_control}, theta_control: {theta_control}")

    
    def PID_control(self, error):
        self.kp = 1
        self.ki = 0.1
        self.kd = 0.01
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

        self.i_error += error
        self.d_error = error - self.p_error
        self.p_error = error
        return self.kp * self.p_error + self.ki * self.i_error + self.kd * self.d_error
    
    def stanley_control(self, error, theta_e):
        control = theta_e + math.atan2(self.k_stanley * error, self.v)
        return control

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        controller = LanePIDController()
        controller.run()
    except rospy.ROSInterruptException:
        pass

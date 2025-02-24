#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import numpy as np
import torch
import model  # Assuming model is the module where enhance_net_nopool is defined

showWindow = True
combineFrames = True
beta = 0.5
sf = 8

class ZeroDCE:
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('zerodce_node', anonymous=True)

        # Set device to GPU if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load the model once during initialization
        scale_factor = sf # Set the appropriate scale factor
        self.DCE_net = model.enhance_net_nopool(scale_factor).to(self.device)
        self.DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch99.pth', map_location=self.device))
        self.DCE_net.eval()  # Set the model to evaluation mode
        
        # ROS 이미지 구독
        self.image_sub = rospy.Subscriber("/camera/image_rect_color", Image, self.callback, queue_size=1)
        
        # ROS 이미지 퍼블리셔
        self.image_pub = rospy.Publisher("/enhanced_image", Image, queue_size=1)
        
        # CvBridge 인스턴스 (ROS Image 메시지를 OpenCV 형식으로 변환)
        self.bridge = CvBridge()
      
    def callback(self, ros_image):
        if(showWindow):
            # Create named windows
            cv2.namedWindow("Original_Image")
            # cv2.namedWindow("Enhanced_Image")
            # cv2.namedWindow("Combined_Frame")

            # Move windows to fixed coordinates
            cv2.moveWindow("Original_Image", 0, 0)  # Move to (100, 100)
            # cv2.moveWindow("Enhanced_Image", 0, 360+65)  # Move to (900, 100)
            # cv2.moveWindow("Combined_Frame", 0, 720+130)  # Move to (900, 100)

        start = time.time()
        
        # ROS 메시지에서 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
        height, width, channels = cv_image.shape
        print(height, width)
        cv_image = cv2.resize(cv_image, (640, 360), interpolation=cv2.INTER_AREA) # 통신 이슈로 400*160으로 변환

        #########################################
        # cv_image = cv2.resize(cv_image, (800, 320))  # Resize to 800x320
        # cv2.INTER_LINEAR 
        # cv2.INTER_CUBIC
        # cv2.INTER_LANCZOS4 
        # cv2.INTER_AREA
        
        # Preprocess the image
        data_lowlight = (np.asarray(cv_image) / 255.0)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2, 0, 1)
        data_lowlight = data_lowlight.to(self.device).unsqueeze(0)

        # Disable gradient calculation
        with torch.no_grad():
            enhanced_image = self.DCE_net(data_lowlight)
        
        # Extract the tensor from the tuple if necessary
        if isinstance(enhanced_image, tuple):
            enhanced_image = enhanced_image[0]
    
        # Calculate and print processing time
        t2 = (time.time() - start)
        print("Processing time:", t2)
            
        # Convert the tensor to a NumPy array
        enhanced_image = enhanced_image.squeeze().permute(1, 2, 0).cpu().numpy()
        
        enhanced_image = np.clip(enhanced_image, 0, 1)

        # Convert the tensor to a NumPy array and scale to 8-bit
        enhanced_image = (enhanced_image * 255).astype(np.uint8)
        
        # Convert the enhanced image to ROS Image message
        enhanced_image_msg = self.bridge.cv2_to_imgmsg(enhanced_image, encoding="bgr8")
        
        combined_frame = cv2.addWeighted(enhanced_image, beta, cv_image, 1 - beta, 0)
        combined_frame_msg = self.bridge.cv2_to_imgmsg(combined_frame, encoding="bgr8")
        
                
        # Publish the enhanced image
        if(combineFrames):
            self.image_pub.publish(combined_frame_msg)
        else:
            self.image_pub.publish(enhanced_image_msg)
        
        
        if(showWindow):
            
            cv2.imshow("Original_Image", cv_image)
            cv2.resizeWindow("Original_Image", 1200, 720)  # Resize to 800x320
            # cv2.imshow("Enhanced_Image", enhanced_image)
            # cv2.imshow("Combined_Frame", combined_frame)
            cv2.waitKey(1)
            
        
if __name__ == '__main__':
    try:
        zerodce_node = ZeroDCE()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("ROS interrupted.")
    finally:
        cv2.destroyAllWindows()
        print("Node terminated.")
		

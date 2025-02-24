#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import numpy as np
import cv2
import math
from collections import deque
from std_msgs.msg import Float64MultiArray
from torchvision import transforms
from mmengine.config import Config
from srlane.models.registry import build_net
from mmcv.parallel import DataContainer as DC
from srlane.ops import nms
from srlane.utils.lane import Lane
from srlpkg.msg import Lanes
import csv

# 모델 로드 함수
def load_model(model_path, cfg):
    model = build_net(cfg)
    load_network(model, model_path, strict=False)
    model.eval()  
    return model

def load_network(net, model_dir, strict=False):
    weights = torch.load(model_dir)["net"]
    new_weights = {}
    for k, v in weights.items():
        new_k = k.replace("module.", '') if "module" in k else k
        new_weights[new_k] = v
    net.load_state_dict(new_weights, strict=strict)

class LaneDetectionNode(nn.Module):
    def __init__(self):
        # ROS 노드 초기화
        rospy.init_node('lane_detection_node', anonymous=True)
        super(LaneDetectionNode, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "checkpoint/baseline.pth" # 모델 파일 경로
        cfg = Config.fromfile("configs/srlane_r18.py")

        self.model = None
        self.model = load_model(model_path,cfg)
        self.model.to(self.device)
        print("Model loaded successfully")

        self.previous_left_lane = []
        self.previous_right_lane = []
        self.csv_idx = 0

        self.register_buffer(name = "prior_ys", tensor = torch.linspace(1,0, steps = self.model.cfg.n_offsets, dtype = torch.float32).to(self.device))

        # ROS 이미지 구독
        self.image_sub = rospy.Subscriber("/enhanced_image", Image, self.callback) # 토픽 이름 확인!!!

        # CvBridge 인스턴스 (ROS Image 메시지를 OpenCV 형식으로 변환)
        self.bridge = CvBridge()

        # 전처리 정의 (CULane 스타일 전처리)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.model.cfg.img_h, self.model.cfg.img_w)),  
            transforms.ToTensor(),
        ])
        self.lane_pub = rospy.Publisher('lane', Lanes, queue_size=10)
        
    def callback(self, ros_image):
        if self.model is None:
            rospy.loginfo("Model is not loaded.")
            return
        try:
            # ROS 메시지에서 OpenCV 이미지로 변환 
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")

            # 비디오 출력 (확인용) ########################################
            # cv2.namedWindow("Original_videos", cv2.WINDOW_NORMAL) # self.model.cfg.ori_img_w self.model.cfg.ori_img_h -- 로봇 카메라 픽셀 확인 후 수정
            # imS = cv2.resize(cv_image, (self.model.cfg.ori_img_w, self.model.cfg.ori_img_h), interpolation=cv2.INTER_AREA)
            # cv2.imshow("Original_videos", imS)
            # cv2.waitKey(1)
            ###########################################################

            # 이미지 전처리2 (여기서 CULane과 비슷한 방식으로 전처리)
            input_image = self.transform(cv_image).unsqueeze(0).to(self.device)

            # 전처리 이미지 출력 (확인용) ##################################
            # image_np = input_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # image_np = (image_np * 255).astype(np.uint8)
            # cv2.imshow("Preprocessed Image", image_np)
            # cv2.waitKey(1)
            ##############################################################
            with torch.no_grad():
                try:
                    predictions = self.process_output(input_image)
                    coordinates = self.convert(predictions, cv_image)
                    print(coordinates)
                    revised_coordinates = self.visualize(cv_image, coordinates)
                    self.publish_lanes(cv_image, revised_coordinates) 

                except Exception as e:
                    print(f"Error during model inference: {e}")
                
        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")

    # 모델 예측 결과를 실제 픽셀 좌표로 변환
    def convert(self, predictions, img):
        rospy.loginfo("Evaluating...")
        for idx, pred in enumerate(predictions):
            lane_points = []
            for lane in pred:
                single_lane_points=[]
                for x, y in lane.points:
                    real_x = x * self.model.cfg.ori_img_w
                    real_y = y * self.model.cfg.ori_img_h
                    single_lane_points.append((real_x, real_y))  
                lane_points.append(single_lane_points)

            # 일부 시각화
            for single_lane_points in lane_points:
                for i in range(1, len(single_lane_points)):
                    real_x, real_y = single_lane_points[i]
                    prev_x, prev_y = single_lane_points[i-1]
                    # cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (255, 255, 255), 5) # bgr
               
        
        # cv2.namedWindow("Final_Lane Detection", cv2.WINDOW_NORMAL) 
        # imS = cv2.resize(img, (self.model.cfg.ori_img_w, self.model.cfg.ori_img_h), interpolation=cv2.INTER_AREA)
        # cv2.imshow("Final_Lane Detection", imS)
        # cv2.waitKey(1)
        
        return lane_points

    def predictions_to_pred(self, predictions, img_meta):
        """
        Convert predictions to internal Lane structure for evaluation & visualization.
        """
        prior_ys = self.prior_ys.to(predictions.device)
        prior_ys = prior_ys.double()
        lanes = []

        for lane in predictions:
            lane_xs = lane[4:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.model.cfg.n_strips))),
                        self.model.cfg.n_strips)
            length = int(round(lane[3].item()))
            end = start + length - 1
            end = min(end, self.model.cfg.n_strips)
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       ).cpu().numpy()[::-1].cumprod()[::-1]).astype(bool))
            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = prior_ys[(lane_xs >= 0.) & (lane_xs <= 1.)]
            lane_xs = lane_xs[(lane_xs >= 0.) & (lane_xs <= 1.)]
            if len(lane_xs) <= 1:
                continue
            lane_xs = lane_xs.flip(0).double()
            lane_ys = lane_ys.flip(0)

            if "img_cut_height" in img_meta:
                cut_height = img_meta["img_cut_height"]
                ori_img_h = img_meta["img_size"][0]
                lane_ys = (lane_ys * (ori_img_h - cut_height) +
                           cut_height) / ori_img_h
            points = torch.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                dim=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={})
            lanes.append(lane)
        return lanes

    def get_lanes(self, output, img_metas, as_lanes=True):
        """
        Convert model output to lanes.
        """
        softmax = nn.Softmax(dim=1)
        
        decoded = []
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for predictions, img_meta in zip(output, img_metas):
            threshold = self.model.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            nms_preds = predictions.detach().clone()
            nms_preds[..., 2:4] *= self.model.cfg.n_strips
            nms_preds[..., 3] = nms_preds[..., 2] + nms_preds[..., 3] - 1
            nms_preds[..., 4:] *= self.model.cfg.img_w

            keep, num_to_keep, _ = nms(
                nms_preds,
                scores,
                overlap=self.model.cfg.test_parameters.nms_thres,
                top_k=self.model.cfg.max_lanes)
            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            predictions[:, 3] = torch.round(predictions[:, 3] * self.model.cfg.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions, img_meta)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded

    def process_output(self,input_image):
        """
        예측된 차선 정보를 처리하여 (x, y) 좌표로 변환
        """
        predictions = []
        meta_data = {"img_size" : (self.model.cfg.ori_img_h, self.model.cfg.ori_img_w), "img_cut_height" : self.model.cfg.cut_height}
        meta_data = DC(meta_data, cpu_only=True)
        print(meta_data)
        output = self.model(input_image)
        added_output = self.get_lanes(output,meta_data)
        predictions.extend(added_output)
        return predictions
    

    def visualize(self, img, coordinates):
        """
        차선 정보를 시각화
        """
        ############ 영상별 차선 ROI ############
        # 왼커브 (ww1)
        # Left lane : last_x <= 150 and last_y >= 340
        # Right lane : (last_x >= 630 and last_y >= 280) or (last_x >= 560 and last_y >= 340)
        # 오커브 (ww2)
        # Left lane : last_x <= 150 and last_y >= 340
        # Right lane : last_x >= 530 and last_y >= 340
        # 오커브 (ww3)
        # Left lane : (last_x <= 150 and last_y >= 340) and (last_x >= 50 and last_y >= 340)
        # Right lane : (last_x >= 450 and last_y >= 340) and (last_x <= 600 and last_y >= 340)
        # 직진 (b4), ww
        # Left lane : last_x <= 110 and last_y >= 340
        # Right lane : last_x >= 450 and last_y >= 340
        # straight1.bag, straight3.bag
        # Left lane : last_x <= 10 and last_y >= 270
        # Right lane : last_x >= 630 and last_y >= 270
        # hori
        # Left lane : (last_x <= 140 and last_y >= 340) and (last_x >= 50 and last_y >= 340)
        # Right lane : (last_x >= 450 and last_y >= 340) and (last_x <= 600 and last_y >= 340)
        # 로봇카메라
        # Left_lane : (last_x <= 10 and last_y >= 190) and (last_x <= 10 and last_y <= 280)
        # Right_lane : (last_x >= 630 and last_y >= 190) and (last_x >= 630 and last_y <= 320)
        #######################################
        Unnecessary_lanes = []
        c2 = []
        left_lane = []
        right_lane = []
        for idx, single_lane_points in enumerate(coordinates):
            last_x, last_y = single_lane_points[-1]
            # Left Lane
            if (last_x <= 10 and last_y >= 190) and (last_x <= 10 and last_y <= 280):
                if coordinates[idx] not in left_lane:
                    left_lane.append(coordinates[idx])
            # Right Lane
            elif (last_x >= 630 and last_y >= 200):
                if coordinates[idx] not in right_lane:
                    right_lane.append(coordinates[idx])
            # Unnecessary Lane
            else:
                Unnecessary_lanes.append(coordinates[idx])

        for single_lane_points in left_lane:
            print("Drawing Left")
            for i in range(1, len(single_lane_points)):
                real_x, real_y = single_lane_points[i]
                prev_x, prev_y = single_lane_points[i-1]
                # cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (0, 0, 255), 10) # bgr
            if left_lane not in c2:
                c2.append(left_lane[0])
        for single_lane_points in right_lane:
            print("Drawing Right")
            for i in range(1, len(single_lane_points)):
                real_x, real_y = single_lane_points[i]
                prev_x, prev_y = single_lane_points[i-1]
                # cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (0, 255, 0), 10)
            if right_lane not in c2:
                c2.append(right_lane[0])
        for single_lane_points in Unnecessary_lanes:
            # print("Drawing Unnecessary")
            for i in range(1, len(single_lane_points)):
                real_x, real_y = single_lane_points[i]
                prev_x, prev_y = single_lane_points[i-1]
                cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (255, 255, 255), 5)

        if len(left_lane) == 0 : 
            print("Left Processing...")
            left_lane = self.previous_left_lane
            c2.append(left_lane[0])
        if len(right_lane) == 0 : 
            print("Right Processing...")
            right_lane = self.previous_right_lane
            c2.append(right_lane[0])

        self.previous_left_lane = left_lane
        self.previous_right_lane = right_lane

        for single_lane_points in left_lane:
            for i in range(1, len(single_lane_points)):
                real_x, real_y = single_lane_points[i]
                prev_x, prev_y = single_lane_points[i-1]
                cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (0, 0, 255), 5)
        for single_lane_points in right_lane:
            for i in range(1, len(single_lane_points)):
                real_x, real_y = single_lane_points[i]
                prev_x, prev_y = single_lane_points[i-1]
                cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (0, 255, 0), 5)
        print("left")
        print(left_lane)
        print("right")
        print(right_lane)

        # cv2.namedWindow("Final_Lane_Detection", cv2.WINDOW_NORMAL)
        # imS = cv2.resize(img, (self.model.cfg.ori_img_w, self.model.cfg.ori_img_h), interpolation=cv2.INTER_AREA)
        # cv2.imshow("Final_Lane_Detection", imS)
        # cv2.waitKey(1)

        return c2

        # ########## 원래 쓰던 필터링 ##########
        # # print(coordinates)
        # c1=[]
        # for idx, single_lane_points in enumerate(coordinates):

        #     for i in range(1, len(single_lane_points)):
        #         real_x, real_y = single_lane_points[i]
        #         prev_x, prev_y = single_lane_points[i-1] 
        #         if real_y >= self.model.cfg.ori_img_h * 7 / 8: # abs(real_x - self.model.cfg.ori_img_w / 2) <= self.model.cfg.ori_img_w * 5 / 12 and
        #             lineindex1 = idx
        #             # cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (0, 255, 255), 9)
        #             if coordinates[lineindex1] not in c1:
        #                 c1.append(coordinates[lineindex1])
        # for single_lane_points in c1:
        #     for i in range(1, len(single_lane_points)):
        #         real_x, real_y = single_lane_points[i]
        #         prev_x, prev_y = single_lane_points[i-1]
        #         cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (255, 0, 0), 4)
        
        # # cv2.namedWindow("Final_Lane Detection", cv2.WINDOW_NORMAL) 
        # # imS = cv2.resize(img, (self.model.cfg.ori_img_w, self.model.cfg.ori_img_h), interpolation=cv2.INTER_AREA)
        # # cv2.imshow("Final_Lane Detection", imS)
        # # cv2.waitKey(1)

        # return c1

    def publish_lanes(self, img, coordinates):
        """
        차선 정보를 ROS 메시지로 발행
        """
        # print(coordinates)
        if not coordinates:
            rospy.loginfo("No coordinates to publish.")
            return
        msg = Lanes()
        msg.left_x = coordinates[0][-18][0] # 뒤에서 20번째 좌표 --- 수정 필요
        msg.right_x = coordinates[1][-18][0]
        if msg.right_x - msg.left_x < 0:
            msg.left_x, msg.right_x = msg.right_x, msg.left_x
        msg.left_theta = math.atan2(coordinates[0][-17][1] - coordinates[0][-18][1], coordinates[0][-18][0] - coordinates[0][-17][0]) * 180 / math.pi
        if msg.left_theta >= 90:
            msg.left_theta = 180 - msg.left_theta
        msg.right_theta = math.atan2(coordinates[1][-17][1] - coordinates[1][-18][1], coordinates[1][-18][0] - coordinates[1][-17][0]) * 180 / math.pi
        if msg.right_theta <= 90:
            msg.right_theta = 180 - msg.right_theta

        # csv 파일로 저장 ################################################################
        # with open('./csv/lanes.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([self.csv_idx, int(msg.left_x), int(msg.right_x)])

        # with open('./csv/angles.csv', mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([self.csv_idx, int(msg.left_theta), int(msg.right_theta)])
        ################################################################################

        self.csv_idx += 1
        # if msg.right_theta - msg.left_theta < 0:
        #     msg.left_theta, msg.right_theta = msg.right_theta, msg.left_theta
        cv2.circle(img, (int(msg.left_x), int(coordinates[0][-18][1])), 5, (255, 255, 255), -1)
        cv2.circle(img, (int(msg.right_x), int(coordinates[1][-18][1])), 5, (255, 255, 255), -1) 
        cv2.namedWindow("Final_Lane_Detection", cv2.WINDOW_NORMAL) 
        cv2.resizeWindow("Final_Lane_Detection", 1200, 720)  # 원하는 가로, 세로 크기로 조정
        imS = cv2.resize(img, (self.model.cfg.ori_img_w, self.model.cfg.ori_img_h), interpolation=cv2.INTER_AREA)
        cv2.imshow("Final_Lane_Detection", imS)
        cv2.moveWindow("Final_Lane_Detection", 800, 0)
        cv2.waitKey(1)

        try:
            rospy.loginfo("Publishing lane coordinates...")
            self.lane_pub.publish(msg)
        except Exception as e:
            rospy.logerr(f"Error in publish_lanes: {e}")
  
if __name__ == '__main__':
    try:
        lane_detection_node = LaneDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        print("ROS interrupted.")
    finally:
        cv2.destroyAllWindows()
        print("Node terminated.")

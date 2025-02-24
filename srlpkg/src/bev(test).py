#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import numpy as np
import cv2
import math
from std_msgs.msg import Float64MultiArray
from torchvision import transforms
from mmengine.config import Config
from srlane.models.registry import build_net
from mmcv.parallel import DataContainer as DC
from srlane.ops import nms
from srlane.utils.lane import Lane
from srlpkg.msg import Lanes

Width = 640
Height = 360
warp_src  = np.array([
            [Width*1/5, Height>>2],  
            [0,Height>>1],
            [Width*4/5, Height>>2],
            [Width,Height>>1]
        ], dtype=np.float32)

warp_dist = np.array([
            [Width/16,0],
            [Width*1/5,Height],
            [Width*15/16,0],
            [Width*4/5, Height]
        ], dtype=np.float32)

lx = deque()
rx = deque()
ltheta = deque()
rtheta = deque()

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

        self.register_buffer(name = "prior_ys", tensor = torch.linspace(1,0, steps = self.model.cfg.n_offsets, dtype = torch.float32).to(self.device))

        # ROS 이미지 구독
        # self.image_sub = rospy.Subscriber("/camera/image_rect_color", Image, self.callback) # 토픽 이름 확인!!!
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
                
        self.M, self.Minv = self.compute_transformation_matrices(warp_src, warp_dist)

    def compute_transformation_matrices(self, src, dst):
        """
        Compute the perspective transformation matrix and its inverse.
        """
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        return M, Minv
     
    def callback(self, ros_image):
        if self.model is None:
            rospy.loginfo("Model is not loaded.")
            return
        try:
            # ROS 메시지에서 OpenCV 이미지로 변환 
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")

            # 비디오 출력 (확인용) ########################################
            for point in warp_src:
                cv2.circle(cv_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1) 

            cv2.namedWindow("Original_videos") # self.model.cfg.ori_img_w self.model.cfg.ori_img_h -- 로봇 카메라 픽셀 확인 후 수정
            cv2.moveWindow("Original_videos", 0, 0)
            cv2.imshow("Original_videos", cv_image)
            cv2.waitKey(1)
            
            #warp_img, _, _ = self.warp_image(cv_image, warp_src, warp_dist, (640, 360))
            warp_img = cv2.warpPerspective(cv_image, self.M, (640,360), flags=cv2.INTER_LINEAR)
            #warp_img size is 640x360
            cv2.imshow("BEV", warp_img)
            cv2.waitKey(1)
            # ###########################################################

            # # 이미지 전처리1 (높이 자르기)
            # cut_image = cv_image[self.model.cfg.cut_height:, :, :]
            # cv2. namedWindow("Cut Image", cv2.WINDOW_NORMAL)
            # cc = cv2.resize(cut_image, (1920, 1080))
            # cv2.imshow("Cut Image", cc)
            # cv2.waitKey(1)

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
                    # print(predictions)
                    coordinates, transformed_coordinates = self.convert(predictions, cv_image, warp_img)
                    self.publish_lanes(cv_image, warp_img, coordinates, transformed_coordinates)  

                except Exception as e:
                    print(f"Error during model inference: {e}")
                
        except Exception as e:
            rospy.logerr(f"Error in callback: {e}")
    
    def warp_image(self, img, src, dst, size):
        """
        이미지를 원근 변환
        """
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        # 카메라에서 가져온 이미지를 버드뷰로 전환
        warp_img = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
        return warp_img, M, Minv

    # 모델 예측 결과를 실제 픽셀 좌표로 변환
    def convert(self, predictions, img, warp_img):
        left_lane_left_boundary = 640 >> 2
        left_lane_right_boundary = left_lane_left_boundary + 100
        right_lane_right_boundary = 640 - (640 >> 2)
        right_lane_left_boundary = right_lane_right_boundary - 100

        rospy.loginfo("Evaluating...")
        for _, pred in enumerate(predictions):
            lane_points = []
            transformed_points = []
            # idx로 왼쪽 차선과 오른쪽 차선을 구분: idx=0 -> 왼쪽 차선, idx=1 -> 오른쪽 차선
            for idx, lane in enumerate(pred):
                #print(lane, idx)
                single_lane_points = []
                for x, y in lane.points:
                    real_x = x * self.model.cfg.ori_img_w
                    real_y = y * self.model.cfg.ori_img_h
                    single_lane_points.append((real_x, real_y))
                
                lane_points.append(single_lane_points)

                # Apply BEV transformation to the lane points
                points_array = np.array(single_lane_points, dtype=np.float32).reshape(-1, 1, 2)
                transformed_points_array = cv2.perspectiveTransform(points_array, self.M)
                # print(np.shape(transformed_points_array))

                # Convert the points back to the original format
                transformed_single_lane_points = [(pt[0][0], pt[0][1]) for pt in transformed_points_array]
                # print(np.shape(transformed_single_lane_points))
                
                # 왼쪽 차선과 오른쪽 차선을 구분
                # x좌표로 filtering 해줌, 근데 잘 안 됨
                # 왼쪽 차선
                if idx == 0:
                    # [-1][0]에서 [-1]이 가장 아래쪽의 (x, y) 좌표!!!, [-1][0]은 x좌표, [-1][1]은 y좌표
                    # [0][0]에서 [0]이 가장 위쪽의 (x, y) 좌표
                    # print("leftlane", transformed_single_lane_points[-1][0], transformed_single_lane_points[-1][1])
                    if(left_lane_left_boundary <= int(transformed_single_lane_points[-1][0]) <= left_lane_right_boundary):
                        # print("leftlane", transformed_single_lane_points[-1][0], transformed_single_lane_points[-1][1])
                        transformed_points.append(transformed_single_lane_points)
                # 오른쪽 차선
                if idx == 1:
                    # print("rightlane", transformed_single_lane_points[-1][0], transformed_single_lane_points[-1][1])
                    if(right_lane_left_boundary <= int(transformed_single_lane_points[-1][0]) <= right_lane_right_boundary):
                        # print("rightlane", transformed_single_lane_points[-1][0], transformed_single_lane_points[-1][1])
                        transformed_points.append(transformed_single_lane_points)

                

            # Draw the original lane points
            for single_lane_points in lane_points:
                for i in range(1, len(single_lane_points)):
                    real_x, real_y = single_lane_points[i]
                    prev_x, prev_y = single_lane_points[i-1]
                    cv2.line(img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (255, 0, 0), 4)

            # Draw the transformed lane points on the warp_img
            for single_lane_points in transformed_points:
                for i in range(1, len(single_lane_points)):
                    real_x, real_y = single_lane_points[i]
                    prev_x, prev_y = single_lane_points[i-1]
                    cv2.line(warp_img, (int(prev_x), int(prev_y)), (int(real_x), int(real_y)), (255, 0, 0), 4)
                    
        cv2.namedWindow("Final_Lane Detection")
        cv2.moveWindow("Final_Lane Detection", 0, 360+100)
        cv2.imshow("Final_Lane Detection", img)
        cv2.waitKey(1)

        # BEV 이미지에서의 차선 x좌표 범위들 표시
        cv2.circle(warp_img, (left_lane_left_boundary, 360-10), 5, (0, 0, 255), -1)
        cv2.circle(warp_img, (left_lane_right_boundary, 360-10), 5, (0, 0, 255), -1)

        cv2.circle(warp_img, (right_lane_right_boundary, 360-10), 5, (0, 0, 255), -1)
        cv2.circle(warp_img, (right_lane_left_boundary, 360-10), 5, (0, 0, 255), -1)
        
        cv2.namedWindow("BEV")
        cv2.moveWindow("BEV", 0, 720+200)
        cv2.imshow("BEV", warp_img)
        cv2.waitKey(1)

        return lane_points, transformed_points
    
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
            # filter out the conf lower than conf threshold
            threshold = self.model.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            # print(scores)
            keep_inds = scores >= threshold
            # print(keep_inds)
            predictions = predictions[keep_inds]
            # print(predictions)
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            nms_preds = predictions.detach().clone()
            nms_preds[..., 2:4] *= self.model.cfg.n_strips
            nms_preds[..., 3] = nms_preds[..., 2] + nms_preds[..., 3] - 1
            nms_preds[..., 4:] *= self.model.cfg.img_w
            # print(nms_preds)
            # print(scores)

            keep, num_to_keep, _ = nms(
                nms_preds,
                scores,
                overlap=self.model.cfg.test_parameters.nms_thres,
                top_k=self.model.cfg.max_lanes)
            # print(f"keep'{keep}")
            # print(f"num_to_keep'{num_to_keep}")
            keep = keep[:num_to_keep]
            # print(f"keep'{keep}")
            predictions = predictions[keep]
            # print(f"predictions'{predictions}")

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
        # print(meta_data)
        output = self.model(input_image)
        # print(output)
        added_output = self.get_lanes(output,meta_data)
        # print(added_output)
        predictions.extend(added_output)
        return predictions

    def publish_lanes(self, img, warp_img, coordinates, transformed_coordinates):
        """
        차선 정보를 ROS 메시지로 발행
        """
        lx, ly, rx, ry = [], [], [], []
        # print(coordinates)
        if not coordinates:
            rospy.loginfo("No coordinates to publish.")
            return
        msg = Lanes()
        if(len(transformed_coordinates) == 0):
            msg.left_x = (640 >> 2) + 50
            msg.right_x = 640 - (640 >> 2) - 50
        else:
            msg.left_x = transformed_coordinates[0][-1][0]
            msg.right_x = transformed_coordinates[1][-1][0]
        

        #print("hello",np.shape(transformed_coordinates[0]))
        for i in range(len(transformed_coordinates[0])):
            lx.append(transformed_coordinates[0][i][0])
            print("lx",lx[i])
            ly.append(transformed_coordinates[0][i][1])
        for i in range(len(transformed_coordinates[1])):
            rx.append(transformed_coordinates[1][i][0])
            print("rx",lx[i])
            ry.append(transformed_coordinates[1][i][1])
        left_angle_radians = math.atan2(transformed_coordinates[0][-1][0]-transformed_coordinates[0][-10][0], transformed_coordinates[0][-1][1]-transformed_coordinates[0][-10][1])
        # left_angle_radians = math.atan2(1,math.sqrt(3))
        left_angle_degrees = math.degrees(left_angle_radians)
        #print(left_angle_degrees)

        right_angle_radians = math.atan2(transformed_coordinates[1][-1][0]-transformed_coordinates[1][-10][0], transformed_coordinates[1][-1][1]-transformed_coordinates[1][-10][1])
        right_angle_degrees = math.degrees(right_angle_radians)
        #print(right_angle_degrees)

        msg.left_theta = left_angle_degrees
        msg.right_theta = right_angle_degrees

        cv2.circle(warp_img, (int(msg.left_x), int(coordinates[0][-20][1])), 5, (255, 255, 255), -1)
        cv2.circle(warp_img, (int(msg.right_x), int(coordinates[1][-20][1])), 5, (255, 255, 255), -1) 
        cv2.imshow("BEV", warp_img)
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

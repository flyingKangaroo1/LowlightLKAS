# LowlightLKAS (Lane Keeping Assitance)
2024 동계 URP 프로젝트  

주제: 저조도 환경에서의 카메라 기반 차선 인식 및 LKAS 모듈 개발​​  
  
팀원: 황석민, 범민규, 안근호  


## Environment
- OS: Ubuntu 20.04
- Board: NVIDIA Jetson Orin, Intel Core i9 processor 14900K
- 로봇: Agilex Ranger Mini 3.0
- 카메라: Allied Vision Prosilica GT2000

## Dependency
- ROS1/Noetic
- OpenCV/4.2.0
- PyTorch
- CUDA

## 센서 실행법
- 로봇
```
roslaunch ranger_base ranger.launch 
```
- 카메라
```
roslaunch avt_vimba_camera mono_camera.launch
```

## 실행 파일들
- Zero-DCE 저조도 개선 모델
```
rosrun Zero_DCE zerodce.py
```
- SRLane 차선 인식 모델
```
rosrun srlpkg main.py
```
- 제어
```
rosrun srlpkg control.py
```

## 실행 영상
- Zero-DCE 저조도 개선 모델
![1](https://github.com/user-attachments/assets/f4dc76fc-7d61-4daf-b71e-13c10c2e1be8)
- SRLane 차선 인식 모델
![2](https://github.com/user-attachments/assets/21918ec1-7d3a-4f89-a1fd-e57f4f92832a)
- 통합 모델
![3](https://github.com/user-attachments/assets/f47a1a9b-875c-4c12-9ecd-6e1be572cfd2)

# LowlightLKAS (Lane Keeping Assitance)
![1](https://github.com/user-attachments/assets/f4dc76fc-7d61-4daf-b71e-13c10c2e1be8)
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
  

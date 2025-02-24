import cv2
import matplotlib.pyplot as plt
import os

# 이미지 파일 경로 설정
image_dir = '/home/heven/OSR/src/srlpkg/src/dataset'
image_filename = '00240.jpg'  # 원하는 이미지 파일 이름으로 변경

# 이미지 파일 경로
image_path = os.path.join(image_dir, image_filename)

# 이미지 읽기
image = cv2.imread(image_path)

# 이미지가 제대로 읽혔는지 확인
if image is None:
    print("이미지를 읽을 수 없습니다.")
else:
    # 윗부분 270만큼 자르기
    cropped_image = image[270:, :]
    
    # 800x320으로 리사이즈
    resized_image = cv2.resize(cropped_image, (800, 320))
    
    # BGR을 RGB로 변환 (matplotlib은 RGB 형식을 사용)
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # 이미지 시각화
    plt.imshow(resized_image_rgb)
    plt.axis('off')  # 축 제거
    plt.show()

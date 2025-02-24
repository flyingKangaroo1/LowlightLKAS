import csv
import matplotlib.pyplot as plt

# CSV 파일 경로
# csv_file = "/home/heven/OSR/src/srlpkg/src/csv/lanes.csv" # 수정 필요
csv_file = "/home/heven/OSR/src/srlpkg/src/csv/angles.csv" # 수정 필요

# 데이터 저장용 리스트
index = []
left_lane_x = []
right_lane_x = []

# CSV 파일 읽기
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # 첫 번째 열은 index, 두 번째 열은 left_lane_x, 세 번째 열은 right_lane_x
        index.append(int(row[0]))
        left_lane_x.append(int(row[1]))
        right_lane_x.append(int(row[2]))

# 점 그래프 그리기
# plt.scatter(index, left_lane_x, color='blue', label='Left Lane X', s=10)  # 수정 필요
# plt.scatter(index, right_lane_x, color='red', label='Right Lane X', s=10)
plt.scatter(index, left_lane_x, color='blue', label='Left Lane Angle', s=10)
plt.scatter(index, right_lane_x, color='red', label='Right Lane Angle', s=10)

# 그래프 꾸미기
# plt.title("Lane X-Coordinates (Second-Filtered)") # 수정 필요
plt.title("Lane angles (Second-Filtered)")
plt.xlabel("Frames")
# plt.ylabel("X-Coordinates")
plt.ylabel("Angles")
plt.legend()
plt.grid(True)

# 그래프 표시
plt.show()

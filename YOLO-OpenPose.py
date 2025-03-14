import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import pyzed.sl as sl
import numpy as np
import os
from sys import platform
from system_ui import Ui_MainWindow  # 确保这个导入语句匹配您的文件和类名
from ultralytics import YOLO
from numpy import array, dot, arccos, clip, linalg, pi
from scipy.signal import savgol_filter

model = YOLO('best.pt')

# 尝试导入OpenPose
try:
    dir_path = os.path.join('D:', os.sep, '4_2', 'test6', 'bin')
    if platform == "win32":
        sys.path.append(dir_path)
        os.environ['PATH'] += os.pathsep + dir_path
        import pyopenpose as op
    else:
        print("This script is only configured for Windows.")
        sys.exit(-1)
except Exception as e:
    print('Error during OpenPose library import.')
    print(e)
    sys.exit(-1)


class MainWindow(QMainWindow, Ui_MainWindow):

    WINDOW_SIZE = 7  # 例如，选择5作为窗口大小
    POLY_ORDER = 3  # 选择3作为多项式的阶数

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)  # 设置UI布局

        # 初始化ZED相机
        self.zed = self.init_zed()

        # 加载 YOLO 模型
        self.model = YOLO('best.pt')

        # 用于存储每个关节随时间变化的角度数据
        self.joint_history = {}

        # 初始化OpenPose
        params = {
            "model_folder": os.path.join('D:', os.sep, '4_2', 'test6', 'models'),
            "face": False,  # Optional: Disable face to increase processing speed
            "hand": False,  # Optional: Disable hand to increase processing speed
        }
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

        # 设置定时器用于视频流处理
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30fps

    def init_zed(self):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        zed = sl.Camera()
        if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED camera.")
            sys.exit(1)
        return zed

    def update_frame(self):
        image = sl.Mat()
        depth_map = sl.Mat()  # Added for depth information
        point_cloud = sl.Mat()
        runtime_parameters = sl.RuntimeParameters()

        if self.zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(image, sl.VIEW.LEFT)  # Get the left image
            frame = image.get_data()

            # Get depth information at the center of the image
            self.zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)  # Retrieve the depth map
            depth_value = depth_map.get_value(int(depth_map.get_width() / 2), int(depth_map.get_height() / 2))[1]
            print(f"Depth at center: {depth_value} mm")  # Print depth information

            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  # Convert ZED image to OpenCV format

            # Process with OpenPose
            datum = op.Datum()
            datum.cvInputData = frame
            self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            processed_frame = datum.cvOutputData  # Processed frame from OpenPose

            # Convert to QImage and display on video1 QLabel
            qImg = self.convert_cv_to_qimage(processed_frame)
            self.video1.setPixmap(QPixmap.fromImage(qImg))

            # OpenPose angle 确认检测到的人数大于0
            self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            keypoints = datum.poseKeypoints
            if keypoints is not None and len(datum.poseKeypoints) > 0:
                try:
                    self.process_keypoints(keypoints, point_cloud)
                except Exception as e:
                    print(f"Error processing keypoints: {e}")
            else:
                print("No person detected in the frame.")


            # Process with YOLO
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()
            qImg2 = self.convert_cv_to_qimage(annotated_frame)
            self.video2.setPixmap(QPixmap.fromImage(qImg2))

            # YOLO ID
            names_text = self.process_yolo_results(results)
            self.output2.setText(names_text)

            # 在循环结束后，将包含所有名称的字符串设置为 QLabel 的文本
            self.output2.setText(names_text.strip())  # 使用 .strip() 移除最后一个换行符

    def process_keypoints(self, keypoints, point_cloud):
        indexes = {
            "nose": 0,
            "neck": 1,
            "right_shoulder": 2,
            "right_elbow": 3,
            "right_wrist": 4,
            "left_shoulder": 5,
            "left_elbow": 6,
            "left_wrist": 7,
            "mid_hip": 8,
        }

        angle_texts = []
        point = 0

        for person in keypoints:
            keypoints_3d = {}  # 初始化存储处理后的三维坐标数据的字典
            for key, index in indexes.items():
                x, y = int(person[index][0]), int(person[index][1])
                err, point_cloud_value = point_cloud.get_value(x, y)
                if err == sl.ERROR_CODE.SUCCESS:
                    if key not in self.joint_history:
                        self.joint_history[key] = {'x': [], 'y': [], 'z': []}

                    # 累积关节点数据
                    self.joint_history[key]['x'].append(point_cloud_value[0])
                    self.joint_history[key]['y'].append(point_cloud_value[1])
                    self.joint_history[key]['z'].append(point_cloud_value[2])

                    # 检查并应用滤波器
                    if len(self.joint_history[key]['x']) >= self.WINDOW_SIZE:
                        filtered_x = self.apply_savgol_filter(self.joint_history[key]['x'])[-1]
                        filtered_y = self.apply_savgol_filter(self.joint_history[key]['y'])[-1]
                        filtered_z = self.apply_savgol_filter(self.joint_history[key]['z'])[-1]
                        # 更新keypoints_3d为滤波后的值
                        keypoints_3d[key] = (filtered_x, filtered_y, filtered_z)
                    else:
                        # 使用原始数据
                        keypoints_3d[key] = point_cloud_value[:3]

                else:
                    keypoints_3d[key] = (None, None, None)

            # 使用keypoints_3d中的数据进行角度计算和分数评估
            # 以下是假设的处理过程，您需要根据实际需求实现
            # angle = calculate_angle(keypoints_3d)
            # point += calculate_score(angle)

            # 计算颈部角度
            if all(keypoints_3d[key] is not None for key in ["nose", "neck", "mid_hip"]):
                head_direction_vector = np.array(keypoints_3d['nose']) - np.array(keypoints_3d['neck'])
                torso_vector = np.array(keypoints_3d['neck']) - np.array(keypoints_3d['mid_hip'])

                # 计算头部方向向量与躯干向量之间的角度
                cos_angle = np.dot(head_direction_vector, torso_vector) / (
                            np.linalg.norm(head_direction_vector) * np.linalg.norm(torso_vector))
                angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                angle_deg = np.degrees(angle_rad)

                if 0 < angle_deg <= 30:
                    point += 1
                elif 30 < angle_deg <= 40:
                    point += 2
                elif 40 < angle_deg <= 50:
                    point += 3
                elif angle_deg > 50:
                    point += 4

                angle_texts.append(f"Neck angle relative to torso: {angle_deg:.2f} degrees")

            # 计算左右肘部的角度
            for side in ['left', 'right']:
                if all(keypoints_3d[key] is not None for key in
                       [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]):
                    elbow_to_shoulder_vector = np.array(keypoints_3d[f"{side}_shoulder"]) - np.array(
                        keypoints_3d[f"{side}_elbow"])
                    elbow_to_wrist_vector = np.array(keypoints_3d[f"{side}_wrist"]) - np.array(
                        keypoints_3d[f"{side}_elbow"])

                    cos_angle = np.dot(elbow_to_shoulder_vector, elbow_to_wrist_vector) / (
                                np.linalg.norm(elbow_to_shoulder_vector) * np.linalg.norm(
                            elbow_to_wrist_vector))
                    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    angle_deg = np.degrees(angle_rad)

                    if 0 < angle_deg <= 60:
                        point += 2
                    elif 60 < angle_deg <= 100:
                        point += 1
                    elif angle_deg > 100:
                        point += 2

                    angle_texts.append(f"{side.capitalize()} elbow angle: {angle_deg:.2f} degrees")

            # 计算上臂相对于躯干的角度
            if all(keypoints_3d[key] is not None for key in
                   ["neck", "mid_hip", "right_shoulder", "right_elbow", "left_shoulder", "left_elbow"]):
                body_axis_vector = np.array(keypoints_3d['mid_hip']) - np.array(keypoints_3d['neck'])

                for side in ['left', 'right']:
                    arm_vector = np.array(keypoints_3d[f"{side}_elbow"]) - np.array(
                        keypoints_3d[f"{side}_shoulder"])
                    cos_angle = np.dot(arm_vector, body_axis_vector) / (
                                np.linalg.norm(arm_vector) * np.linalg.norm(body_axis_vector))
                    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                    angle_deg = np.degrees(angle_rad)

                    if 0 < angle_deg <= 20:
                        point += 1
                    elif 20 < angle_deg <= 45:
                        point += 2
                    elif 45 < angle_deg <= 90:
                        point += 3
                    elif angle_deg > 90:
                        point += 4

                    angle_texts.append(f"{side.capitalize()} arm relative to torso angle: {angle_deg:.2f} degrees")

        point *= 6


        # 更新UI等
        self.PointBar.setValue(point)
        self.output1.setText(str(point))
        self.output1.setText(angle_texts)

    def convert_cv_to_qimage(self, cv_img):
        # 首先将BGR图像转换为RGB图像
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        height, width, channels = cv_img.shape
        bytes_per_line = channels * width
        return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def process_yolo_results(self, results):
        cls = results[0].boxes.cls  # 获取包含类别ID的张量
        names_text = ""  # 初始化一个空字符串来收集所有名称
        for class_id in cls:
            id = class_id.item()  # 获取类别ID的数字值
            name = results[0].names[id]  # 获取对应的类别名称
            names_text += f"{name}\n"  # 将名称添加到字符串中，每个名称后添加换行符
        return names_text.strip()  # 使用 .strip() 移除最后一个换行符

    def apply_savgol_filter(self, data):
        # 仅当历史数据足够时才应用滤波器
        if len(data) >= self.WINDOW_SIZE:
            return savgol_filter(data, self.WINDOW_SIZE, self.POLY_ORDER).tolist()
        return data


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

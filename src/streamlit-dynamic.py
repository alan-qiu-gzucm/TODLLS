import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import os
import tempfile
import subprocess
import torch
import torchvision.transforms as transforms
from lane_detector import LaneDetector as LD
from mymodel.model import UNet
from mymodel.enetpytorch import ENet

if 'previous_inverted_mask' not in st.session_state:
    st.session_state.previous_inverted_mask = None
if 'previous_center' not in st.session_state:
    st.session_state.previous_center = None
if 'previous_time' not in st.session_state:
    st.session_state.previous_time = None
if 'speeds' not in st.session_state:
    st.session_state.speeds = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'total_distance' not in st.session_state:
    st.session_state.total_distance = 0
if 'voice_process' not in st.session_state:
    st.session_state.voice_process = None
if 'selected_ldm' not in st.session_state:
    st.session_state.selected_ldm = 'UNET'

class_dict = {
    0: '行人',
    1: '汽车',
    2: '卡车',
    3: '自行车',
    4: '交通信号灯',
    5: '限速20（禁止）',
    6: '限速30（禁止）',
    7: '限速50（禁止）',
    8: '限速60（禁止）',
    9: '限速70（禁止）',
    10: '限速80（禁止）',
    11: '限速80结束（其他）',
    12: '限速100（禁止）',
    13: '限速120（禁止）',
    14: '禁止超车（禁止）',
    15: '禁止卡车超车（禁止）',
    16: '下一交叉口优先（危险）',
    17: '优先道路（其他）',
    18: '让行（其他）',
    19: '停车（其他）',
    20: '双向禁行（禁止）',
    21: '禁止卡车通行（禁止）',
    22: '禁止进入（其他）',
    23: '危险（危险）',
    24: '向左转弯（危险）',
    25: '向右转弯（危险）',
    26: '转弯（危险）',
    27: '路面不平（危险）',
    28: '路面湿滑（危险）',
    29: '道路变窄（危险）',
    30: '施工区域（危险）',
    31: '交通信号（危险）',
    32: '人行横道（危险）',
    33: '学校路口（危险）',
    34: '自行车道口（危险）',
    35: '积雪（危险）',
    36: '动物（危险）',
    37: '限制结束（其他）',
    38: '向右行驶（强制）',
    39: '向左行驶（强制）',
    40: '直行（强制）',
    41: '向右或直行（强制）',
    42: '向左或直行（强制）',
    43: '靠右行驶（强制）',
    44: '靠左行驶（强制）',
    45: '环岛（强制）',
    46: '超车限制结束（其他）',
    47: '卡车超车限制结束（其他）'
}


def log_text_to_file(text, log_file_path):
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w', encoding='utf-8') as f:
            pass
    try:
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"{text}\n")
    except Exception as e:
        print(f"记录文本时出错: {e}")


# Load models
@st.cache_resource
def load_models():
    lane_model = UNet(in_channels=3, out_channels=1).cuda()
    lane_model.load_state_dict(torch.load('best_unet_lane_detection.pth', map_location=torch.device('cuda')))
    lane_model.eval()
    model_lane = ENet(1)
    model_lane.load_state_dict(torch.load('./best_enet_lane_detection.pth'))
    model_lane.eval()
    yolo_model = YOLO('YoloV11m_traffic_object_detection_final.pt')  #
    # yolo_model = YOLO('yolov8n.pt')

    return lane_model, model_lane, yolo_model


lane_model, model_lane, yolo_model = load_models()
detector = LD()


def moving_average_2d(data, window_size):
    ret = np.cumsum(data, axis=0, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size


def get_traffic_light_color(frame, x1, y1, x2, y2):
    light = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)

    max_pixels = max(red_pixels, yellow_pixels, green_pixels)
    if max_pixels == red_pixels:
        return "Red"
    elif max_pixels == yellow_pixels:
        return "Yellow"
    elif max_pixels == green_pixels:
        return "Green"
    else:
        return "Unknown"


def dynamic_window_size_adjustment(mask, base_window_size, min_window_size, max_window_size):
    detected_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    proportion = detected_pixels / total_pixels
    window_size = int(max_window_size * (1 - proportion) + min_window_size * proportion)

    return max(min_window_size, min(max_window_size, window_size))


def process_frame(frame, lane_model, model_lane, yolo_model, transform, yolo_conf, detection_alpha,
                  interpolation_factor, base_window_size, min_window_size, max_window_size, LDM='UNET'):
    t = time.time()
    if LDM == 'ML':
        processed = detector.process(frame)
        result = cv2.addWeighted(frame, 1, processed, 0.5, 0)
    elif LDM == 'UNET' or LDM == 'ENET':
        if LDM == 'UNET':
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_image).unsqueeze(0).to('cuda')
            with torch.no_grad():
                output = lane_model(input_tensor)
        elif LDM == 'ENET':
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_image).unsqueeze(0)
            with torch.no_grad():
                output = model_lane(input_tensor)
        inverted_mask = output.squeeze().cpu().numpy()
        inverted_mask = cv2.resize(inverted_mask, (frame.shape[1], frame.shape[0]))
        mask = 1 - inverted_mask
        # mask_filename = f"{t}_mask.png"
        # mask_path = os.path.join("outputs", mask_filename)
        # cv2.imwrite(mask_path, inverted_mask * 255)
        if st.session_state.previous_inverted_mask is not None:
            inverted_mask = cv2.addWeighted(st.session_state.previous_inverted_mask, 1 - interpolation_factor,
                                            inverted_mask, interpolation_factor, 0)

        st.session_state.previous_inverted_mask = inverted_mask.copy()

        window_size = dynamic_window_size_adjustment(mask, base_window_size, min_window_size, max_window_size)
        if window_size > 1:
            inverted_mask = moving_average_2d(inverted_mask, window_size)

        st.session_state.frame_count += 1
        red_overlay = np.zeros_like(frame)
        red_overlay[:, :, 2] = 255
        mask_3d = np.stack([inverted_mask] * 3, axis=2)
        mask_3d = cv2.resize(mask_3d, (frame.shape[1], frame.shape[0]))
        red_mask = (mask_3d * red_overlay).astype(np.uint8)

        result = cv2.addWeighted(frame, 1, red_mask, 0.9, 0)

    yolo_results = yolo_model(frame, conf=yolo_conf)
    yolo_overlay = np.zeros_like(frame, dtype=np.uint8)
    frame_height, frame_width = frame.shape[:2]
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]
            cls = int(box.cls[0])
            center_x = (x1 + x2) / 2

            if conf > yolo_conf:
                class_name = yolo_model.names[cls]
                color = (0, 255, 0)

                if class_name == "traffic light":
                    light_color = get_traffic_light_color(frame, x1, y1, x2, y2)
                    label = f'Traffic Light ({light_color}) {conf:.2f}'
                    if light_color == "Red":
                        color = (0, 0, 255)
                    elif light_color == "Yellow":
                        color = (255, 255, 0)
                    elif light_color == "Green":
                        color = (0, 255, 0)
                else:
                    label = f'{class_name} {conf:.2f}'

                cv2.rectangle(yolo_overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(yolo_overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            class_name = class_dict[cls]
            if center_x < frame_width / 3:
                position = "左侧"
            elif center_x < 2 * frame_width / 3:
                position = "正前方"
            else:
                position = "右侧"
            position_text = f"注意{position}{class_name}"
            log_text_to_file(position_text, rf".\voice_test.txt")

    cv2.addWeighted(result, 1, yolo_overlay, detection_alpha, 0, result)
    return result


def start_voice_broadcast():
    """启动语音播报脚本"""
    if st.session_state.voice_process is None or st.session_state.voice_process.poll() is not None:
        st.session_state.voice_process = subprocess.Popen(
            ['python', rf'./txt_to_speech.py', './voice_test.txt']
        )
        st.write("语音播报已启动")


def stop_voice_broadcast():
    if st.session_state.voice_process is not None and st.session_state.voice_process.poll() is None:
        st.session_state.voice_process.terminate()
        st.session_state.voice_process = None


def main():
    st.title("车道检测和目标识别应用")
    st.sidebar.title("控制台")
    uploaded_file = st.sidebar.file_uploader("选择视频文件", type=["mp4", "avi", "mov"])
    use_webcam = st.sidebar.checkbox("使用摄像头")

    ldm_options = ['UNET', 'ENET', 'ML']
    selected_ldm = st.sidebar.selectbox('选择车道模型', ldm_options, index=0)

    if use_webcam or uploaded_file is not None:
        yolo_conf = st.sidebar.slider('YOLO 置信度阈值', 0.0, 1.0, 0.5, 0.05)
        detection_alpha = st.sidebar.slider('检测透明度', 0.0, 1.0, 0.5, 0.05)
        interpolation_factor = st.sidebar.slider('插值因子', 0.0, 1.0, 0.5, 0.05)
        base_window_size = st.sidebar.slider('基础窗口大小', 1, 100, 30, 1)
        min_window_size = st.sidebar.slider('最小窗口大小', 1, 50, 10, 1)
        max_window_size = st.sidebar.slider('最大窗口大小', 1, 200, 50, 1)

        if st.sidebar.button("处理视频" if uploaded_file else "启动摄像头"):
            st.session_state.previous_inverted_mask = None
            st.session_state.previous_center = None
            st.session_state.previous_time = None
            st.session_state.speeds = []
            st.session_state.frame_count = 0
            st.session_state.total_distance = 0

            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    tmpfile.write(uploaded_file.getbuffer())
                    temp_file_name = tmpfile.name

                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])

                cap = cv2.VideoCapture(temp_file_name)
            else:
                cap = cv2.VideoCapture(0)
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])

            if uploaded_file:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                os.makedirs("outputs", exist_ok=True)
                output_file = "outputs/processed_video_raw.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

                progress_bar = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame = process_frame(frame, lane_model, model_lane, yolo_model, transform, yolo_conf,
                                                    detection_alpha, interpolation_factor, base_window_size,
                                                    min_window_size, max_window_size, LDM=selected_ldm)
                    out.write(processed_frame)

                    progress = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / frame_count
                    progress_bar.progress(progress)

                cap.release()
                out.release()
                os.unlink(temp_file_name)

                st.success("视频处理完成！")
                video_file = open(output_file, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
            else:
                stframe = st.empty()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame = process_frame(frame, lane_model, model_lane, yolo_model, transform, yolo_conf,
                                                    detection_alpha, interpolation_factor, base_window_size,
                                                    min_window_size, max_window_size, LDM=selected_ldm)
                    stframe.image(processed_frame, channels="BGR", use_container_width=True)

                cap.release()

        col1, col2 = st.sidebar.columns(2)
        if col1.button("开启语音播报"):
            start_voice_broadcast()

        if col2.button("关闭语音播报"):
            stop_voice_broadcast()


if __name__ == "__main__":
    main()

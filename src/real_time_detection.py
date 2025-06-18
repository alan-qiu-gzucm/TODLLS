import cv2
import time
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from mymodel.enetpytorch import ENet
from ultralytics import YOLO
from lane_detector import LaneDetector as LD

device = 'cuda'
model_path = "./YoloV11m_traffic_object_detection_final.pt"
model_lane = ENet(1).to('cuda')
model_lane.load_state_dict(torch.load('./model_best_enet.pth'))
model_lane.eval()
video_output_path = "./TEST_20250610.mp4"
confidence_threshold = 0.3
camera_id = 0  # Camera index
camera_fps = 5  # Frame rate, should be between 5 and 30
inference_fps = 2  # Inference frame rate
camera_width = 1280  # Camera resolution, can be 1280x960, 1280x720, 1024x768, 800x600, 640x480, 640x360, 320x240 for Logitech C270
camera_height = 960
detector = LD()
camera = cv2.VideoCapture(camera_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
camera.set(cv2.CAP_PROP_FPS, camera_fps)
model = YOLO(model_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
VIDEO_PATH = './lane1.mp4'
OUTPUT_PATH = './processed_video.mp4'
video_writer = None


def vediotest():
    print("Starting Lane Detection Pipeline")
    # Check if video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Video file not found at: {VIDEO_PATH}")
        return

    # Load video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to load video.")
        return
    print("Video opened successfully.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # processed = detector.process(frame)
        results = model.predict(frame, save=False, show=False, conf=confidence_threshold)
        for i, r in enumerate(results):
            result_image = r.plot()
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        input_tensor = transform(pil_image).unsqueeze(0).to('cuda')
        with torch.no_grad():
            output = model_lane(input_tensor)
        mask = output.squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8)  # 阈值处理生成二值掩码
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        processed = np.zeros_like(result_image)  # 创建一个与 result_image 大小相同的三通道图像
        processed[:, :, 1] = mask * 255
        combined_image = cv2.addWeighted(result_image, 1, processed, 0.5, 0)
        if video_writer is None:
            (h, w) = processed.shape[:2]
            video_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (w, h))
        video_writer.write(combined_image)
        cv2.imshow('Lane Detection', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Output saved to: {OUTPUT_PATH}")


# vediotest()
while True:
    start_time = time.time()
    ret, frame = camera.read()
    if not ret:
        print("Could not read camera feed.")
        break
    # processed = detector.process(frame)
    results = model.predict(frame, save=False, show=False, conf=confidence_threshold)
    for i, r in enumerate(results):
        result_image = r.plot()
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(pil_image).unsqueeze(0).to('cuda')
    with torch.no_grad():
        output = model_lane(input_tensor)
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)  # 阈值处理生成二值掩码
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    processed = np.zeros_like(result_image)  # 创建一个与 result_image 大小相同的三通道图像
    processed[:, :, 1] = mask * 255
    combined_image = cv2.addWeighted(result_image, 1, processed, 0.5, 0)
    if video_writer is None:
        h, w = processed.shape[:2]
        video_writer = cv2.VideoWriter(video_output_path, fourcc, inference_fps, (w, h))
    elapsed_time = time.time() - start_time
    frame_interval = 1.0 / inference_fps - elapsed_time
    if frame_interval > 0:
        time.sleep(frame_interval)

    # Check for user input to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Program terminated.")
# Clean up everything
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
print(f"Done. Output saved to: {OUTPUT_PATH}")

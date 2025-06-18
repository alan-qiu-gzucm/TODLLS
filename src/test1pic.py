import cv2
import numpy as np
from lane_detector import valLaneDetector as VLD
detector = VLD()
VIDEO_PATH = './lane4.mp4'

def vediotest():

    # Load video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to load video.")
        return
    print("Video opened successfully.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # No more frames left

        # Run lane detection on current frame
        processed = detector.process(frame)
        mask = (processed * 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 显示原始帧和掩码
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Lane Mask', mask)

        # 保存掩码为黑白图片
        cv2.imwrite(f"mask_output4.png", mask)





vediotest()
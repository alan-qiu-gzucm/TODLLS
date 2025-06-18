import cv2
import numpy as np

def region_of_interest(img):
    # Get image dimensions
    height, width = img.shape
    # Create a blank mask the same size as the image
    mask = np.zeros_like(img)

    # Define the polygon that covers the lane area
    polygon = np.array([[
        (int(0.3 * width), int(0.69 * height)),     # Bottom-left
        (int(0.54 * width), int(0.435 * height)),   # Top-left
        (int(0.59 * width), int(0.435 * height)),   # Top-right
        (int(1.37 * width), int(0.68 * height))     # Bottom-right (extends beyond width for full coverage)
    ]], np.int32)

    # Fill the polygon with white (255) on the mask
    cv2.fillPoly(mask, [polygon], 255)
    # Apply the mask to the input image using bitwise AND
    return cv2.bitwise_and(img, mask)


class valLaneDetector:
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi = region_of_interest(edges)
        lines = cv2.HoughLinesP(
            roi, 1, np.pi / 180,
            threshold=30,
            minLineLength=10,
            maxLineGap=200
        )

        left_pts, right_pts = [], []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 == x1:
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    if not (0.4 <= abs(slope) <= 1.2):
                        continue
                    if slope < 0:
                        left_pts.extend([(x1, y1), (x2, y2)])
                    else:
                        right_pts.extend([(x1, y1), (x2, y2)])

        def fit_line(points):
            if len(points) < 2:
                return None
            x, y = zip(*points)
            slope, intercept = np.polyfit(x, y, 1)
            y1 = frame.shape[0]
            y2 = int(y1 * 0.45)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return (x1, y1), (x2, y2)
        left_line = fit_line(left_pts)
        right_line = fit_line(right_pts)
        mask = np.zeros_like(gray)
        if left_line and right_line:
            cv2.line(mask, left_line[0], left_line[1], 1, 10)
            cv2.line(mask, right_line[0], right_line[1], 1, 10)
        return mask

class LaneDetector:
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        roi = region_of_interest(edges)
        lines = cv2.HoughLinesP(
            roi, 1, np.pi / 180,
            threshold=30,
            minLineLength=10,
            maxLineGap=200
        )
        left_pts, right_pts = [], []
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if x2 == x1:
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    if not (0.4 <= abs(slope) <= 1.2):
                        continue
                    if slope < 0:
                        left_pts.extend([(x1, y1), (x2, y2)])
                    else:
                        right_pts.extend([(x1, y1), (x2, y2)])

        def fit_line(points):
            if len(points) < 2:
                return None
            x, y = zip(*points)
            slope, intercept = np.polyfit(x, y, 1)
            y1 = frame.shape[0]
            y2 = int(y1 * 0.45)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return (x1, y1), (x2, y2)
        left_line = fit_line(left_pts)
        right_line = fit_line(right_pts)

        if left_line:
            cv2.line(frame, left_line[0], left_line[1], (0, 200, 255), 6)
        if right_line:
            cv2.line(frame, right_line[0], right_line[1], (0, 255, 0), 6)

        if left_line and right_line:
            polygon = np.array([[left_line[0], left_line[1], right_line[1], right_line[0]]], dtype=np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        return frame


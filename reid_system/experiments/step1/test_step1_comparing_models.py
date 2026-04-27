from typing import List, Tuple
import numpy as np
import cv2
from ultralytics import YOLO

# Configuration class to easily change string values and parameters
class config:
    INPUT_VIDEO = "assets/store_cam3.mp4"         # <--- CHANGE THIS to your input video path
    OUTPUT_VIDEO = "comparison.mp4"   # <--- CHANGE THIS to your desired output video path
    
    MODEL_N_PATH = "../models/yolov8n.pt"
    MODEL_S_PATH = "../models/yolov8s.pt"
    
    CONF_N = 0.1  # Confidence threshold for YOLOv8 Nano
    CONF_S = 0.1  # Confidence threshold for YOLOv8 Small
    
    IOU_THRESHOLD = 0.5  # Threshold to consider boxes "overlapping"
    CLASSES = [0]        # 0 = Person. Set to None to detect all classes
    DEVICE = "cuda"      # 'cuda' or 'cpu'


class YOLOModelComparator:
    """
    Module to visually compare two YOLOv8 models.
    
    - Green boxes: Detected ONLY by Model N (Nano)
    - Blue boxes:  Detected ONLY by Model S (Small)
    - Red boxes:   Detected by BOTH models (Overlapping)
    """

    def __init__(
        self,
        model_n_path: str = config.MODEL_N_PATH,
        model_s_path: str = config.MODEL_S_PATH,
        conf_n: float = config.CONF_N,
        conf_s: float = config.CONF_S,
        device: str = config.DEVICE,
    ):
        self.device = device
        self.conf_n = conf_n
        self.conf_s = conf_s

        print(f"Loading Models: {model_n_path} & {model_s_path}...")
        self.model_n = YOLO(model_n_path)
        self.model_n.to(device)
        
        self.model_s = YOLO(model_s_path)
        self.model_s.to(device)

    @staticmethod
    def _compute_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
        """Calculate the Intersection over Union (IoU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Runs detection on both models and draws the comparative bounding boxes."""
        
        # 1. Run detection for Nano
        res_n = self.model_n(frame, classes=config.CLASSES, conf=self.conf_n, device=self.device, verbose=False)
        boxes_n = res_n[0].boxes.xyxy.cpu().numpy() if res_n and res_n[0].boxes is not None else[]

        # 2. Run detection for Small
        res_s = self.model_s(frame, classes=config.CLASSES, conf=self.conf_s, device=self.device, verbose=False)
        boxes_s = res_s[0].boxes.xyxy.cpu().numpy() if res_s and res_s[0].boxes is not None else[]

        # 3. Match overlapping boxes using IoU
        matched_n = set()
        matched_s = set()
        matches =[]

        for i, bn in enumerate(boxes_n):
            best_iou = 0
            best_j = -1
            for j, bs in enumerate(boxes_s):
                if j in matched_s:
                    continue
                iou = self._compute_iou(bn, bs)
                if iou >= config.IOU_THRESHOLD and iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_j != -1:
                matched_n.add(i)
                matched_s.add(best_j)
                matches.append((bn, boxes_s[best_j]))

        # 4. Draw Boxes
        # A) Draw matched (Red) - we'll average the coordinates of both models for a clean single box
        for bn, bs in matches:
            x1, y1 = int((bn[0] + bs[0]) / 2), int((bn[1] + bs[1]) / 2)
            x2, y2 = int((bn[2] + bs[2]) / 2), int((bn[3] + bs[3]) / 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red BGR
            cv2.putText(frame, "Both", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # B) Draw unmatched Nano (Green)
        for i, box in enumerate(boxes_n):
            if i not in matched_n:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green BGR
                cv2.putText(frame, "Nano", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # C) Draw unmatched Small (Blue)
        for j, box in enumerate(boxes_s):
            if j not in matched_s:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue BGR
                cv2.putText(frame, "Small", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # 5. Add a legend to the frame
        cv2.rectangle(frame, (10, 10), (220, 100), (0, 0, 0), -1)
        cv2.putText(frame, "Nano Only", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Small Only", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Both (Overlap)", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def process_video(self, input_path: str, output_path: str):
        """Reads the input video, applies comparison per frame, and writes to output."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Get video properties for Writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print(f"Processing Video: {input_path} -> {output_path}")
        print(f"Resolution: {width}x{height} @ {fps} FPS")

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}/{total_frames}...")

            # Process frame
            annotated_frame = self.process_frame(frame)
            
            # Write to output video
            out.write(annotated_frame)

        cap.release()
        out.release()
        print("Video generation completed successfully!")


if __name__ == "__main__":
    # Initialize the comparator
    comparator = YOLOModelComparator()
    
    # Process the video specified in the config
    comparator.process_video(config.INPUT_VIDEO, config.OUTPUT_VIDEO)
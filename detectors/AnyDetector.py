import cv2
import torch
from ultralytics import YOLO
from typing import List

default_color = (255, 255, 255)


class AnyDetector:
    def __init__(self, model_weights: str, classes: List[str], colors: List[tuple] = None, confidence=0.5):
        """
        Initialize the object detector with YOLO model
        :param model_weights: path to .pt model weights file
        :param classes: list of class names for detection
        :param colors: list of colors for classes
        :param confidence: confidence threshold for prediction
        """
        self.model = YOLO(model_weights)
        self.classes = classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.confidence = confidence
        if colors is None:
            self.colors = [default_color] * len(self.classes)
        else:
            self.colors = colors

    def process_frame(self, frame):
        """Process a single frame and return detection results
        :param frame: one frame to predict
        :return: result of prediction
        """
        results = self.model.predict(frame, verbose=False)
        return results[0]

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame
        :param frame: one frame to work with
        :param detections: list of boxes and labels witch was detected
        :return: frame with labeled objects, if they were detected
        """
        for box in detections.boxes:
            class_id = int(box.cls)
            label = self.classes[class_id]
            confidence = float(box.conf)

            if confidence < confidence:  # Confidence threshold
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = self.colors[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Create text label with confidence
            text = f"{label} {confidence:.2f}"

            # Calculate text position
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Handle text position for top-edge cases
            text_y = y1 - 10 if y1 - 10 > text_height else y1 + 20

            # Draw text background
            cv2.rectangle(
                frame,
                (x1, text_y - text_height - 2),
                (x1 + text_width, text_y + 2),
                color,
                -1
            )

            # Put text
            cv2.putText(
                frame,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (30, 30, 30),  # Dark gray text
                2
            )
        return frame

    def process_video(self,
                      input_path: str,
                      output_path: str,
                      show_live: bool = False):
        """
        Process video file and save annotated results
        :param input_path: input video file path
        :param output_path: output video file path
        :param show_live: show real-time processing window
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.process_frame(frame)
            frame = self.draw_detections(frame, results)

            if show_live:
                cv2.imshow('Video Processing', frame)
                if cv2.waitKey(1) == ord('q'):
                    break

            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def real_time_processing(self, camera_id: int = 0):
        """
        Process live video stream from webcam
        :param camera_id: webcam device ID (default 0)
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError("Error connecting to camera")

        print("Real-time processing started. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.process_frame(frame)
            frame = self.draw_detections(frame, results)

            cv2.imshow('Live Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

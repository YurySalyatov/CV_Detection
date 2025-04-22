from detectors.AnyDetector import AnyDetector
import os
if __name__ == "__main__":
    # Class names should match your model's training classes
    classes = [
        "smoke", "fire"
    ]
    colors = [
        (50, 200, 50), (255, 50, 0)
    ]
    print(os.getcwd().replace('\\', '/'))
    weights_path = os.getcwd().replace('\\', '/') + "/../model/best_model_params/fire_smoke_best.pt"
    print(weights_path)
    # Initialize detector with custom model
    detector = AnyDetector(
        model_weights=weights_path,  # Path to your trained model
        classes=classes,
        colors=colors
    )

    # Process video file example
    video_dir = os.getcwd().replace('\\', '/') + "/../test_video/"
    print(video_dir)
    detector.process_video(
        input_path=video_dir + "fire-smoke.mp4",
        output_path=video_dir + "fire-smoke_detection.mp4",
        show_live=True
    )

    # Start webcam processing example
    # detector.real_time_processing()
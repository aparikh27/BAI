from backend.image_detection.yolo_detector import YOLODetector


def get_video_source():
    """
    Prompt the user for a video source.

    Returns:
        int | str:
            0 -> Default webcam
            1 -> External camera
            str -> Path to an image or video file
    """

    print("\nSelect an input source:")
    print("1. Default webcam")
    print("2. External webcam")
    print("3. Video file")

    choice = input("\n>>> ")

    if choice == "1":
        return 0

    elif choice == "2":
        return 1

    elif choice == "3":
        path = input("\nEnter the file path:\n>>> ").strip()
        return path

    else:
        print("\nInvalid option. Using default webcam.\n")
        return 0


def get_confidence_threshold():
    """
    Prompt the user for a confidence threshold.
    """

    while True:
        confidence = input(
            "\nEnter confidence threshold (0.0 - 1.0):\n>>> "
        )

        try:
            confidence = float(confidence)

            if 0 <= confidence <= 1:
                return confidence

            print("Confidence must be between 0 and 1.")

        except ValueError:
            print("Please enter a valid decimal number.")


def main():

    source = get_video_source()
    confidence = get_confidence_threshold()

    detector = YOLODetector()

    detector.detect(
        source=source,
        confidence=confidence,
    )


if __name__ == "__main__":
    main()
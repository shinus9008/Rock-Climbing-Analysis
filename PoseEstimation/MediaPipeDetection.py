import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
VisionRunningMode = mp.tasks.vision.RunningMode



baseOptions = mp.tasks.BaseOptions(
    model_asset_path='efficientdet.tflite')


options = mp.tasks.vision.ObjectDetectorOptions(
    base_options=baseOptions,
    running_mode=VisionRunningMode.VIDEO)



with ObjectDetector.create_from_options(options) as detector:
    detector.detect_for_video()

    # The detector is initialized. Use it here.
    # ...

    # Use OpenCV’s VideoCapture to load the input video.

    # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
    # You’ll need it to calculate the timestamp for each frame.

    # Loop through each frame in the video using VideoCapture#read()

    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    #mp_image = mp.Image(format=ImageFormat.SRGB, data=numpy_frame_from_opencv)

    # Calculate the timestamp of the current frame
    #frame_timestamp_ms = 1000 * frame_index / video_file_fps

    # Perform object detection on the video frame.
    #detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)




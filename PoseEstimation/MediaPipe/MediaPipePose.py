import copy
import argparse
import cv2
import mediapipe as mp
import imutils



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", default='C:/Users/Alexandr/source/_repos/Rock Climbing Analysis/.video/video1.mp4')
    parser.add_argument("--width", help='cap width', type=int,   default=180)
    parser.add_argument("--height", help='cap height', type=int, default=320)

    parser.add_argument('--plot_world_landmark', type=bool, default=False)

    args = parser.parse_args()

    return args

def plot_world_landmarks(
    plt,
    ax,
    landmarks,
    visibility_th=0.5,
):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))
            
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    
    plt.pause(.5)

    return

def main():
    args = get_args()

    cap_device  = args.device
    cap_width   = args.width
    cap_height  = args.height

    plot_world_landmark = args.plot_world_landmark


    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    help(mp_drawing.plot_landmarks)

    with mp_pose.Pose(
        static_image_mode = False, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        model_complexity = 1,
        smooth_landmarks = True) as pose:

        #
        if plot_world_landmark:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break;

            image = imutils.resize(image , width=360) 

            # Convert the BGR image to RGB and process it with MediaPipe Pose.
            # To improve performance, optionally mark the image as not writable to pass by reference. 
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
            image.flags.writeable = False

            # Make Detections
            results = pose.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



            # Draw pose landmarks.
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks, mp_pose.POSE_CONNECTIONS,     
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      
            # Plot pose world landmarks.
            if plot_world_landmark:
                if results.pose_world_landmarks is not None:
                    plot_world_landmarks(
                        plt,
                        ax,
                        results.pose_world_landmarks)

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

            cv2.imshow('MediaPipe Pose', image)           
    

    cap.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()
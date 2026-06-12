import cv2
import mediapipe as mp
import numpy as np
import time
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
        self.left_eye_idxs = [362, 385, 387, 263, 373, 380]
        self.right_eye_idxs = [33, 160, 158, 133, 153, 144]
        self.iris_idx = 468
        self.blink_threshold = 0.20
        self.both_eye_blink_threshold = 0.15
        self.close_threshold = 0.10
        self.squint_threshold = 0.15
        self.open_left_ear = 0.0
        self.open_right_ear = 0.0
        self.closed_left_ear = 0.0
        self.closed_right_ear = 0.0
        self.is_closed = False
        # Model points are standard 3D face model
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),      # Left eye corner
            (225.0, 170.0, -135.0),       # Right eye corner
            (-150.0, -150.0, -125.0),     # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

        # Initialize Kalman filter for head pose
        self.kf_rotation = self.create_kalman_filter(3)  # 3 for rotation vector (pitch, yaw, roll)
        self.kf_translation = self.create_kalman_filter(3) # 3 for translation vector (x, y, z)

    def create_kalman_filter(self, dim):
        kf = KalmanFilter(dim_x=dim, dim_z=dim)

        # State transition matrix (F):  Assume constant velocity model
        kf.F = np.eye(dim)

        # Measurement function (H):  Directly measure the state
        kf.H = np.eye(dim)

        # Process noise covariance (Q):  Small noise to allow for changes in pose
        kf.Q = Q_discrete_white_noise(dim=dim, dt=1.0, var=0.01)  # Adjust var

        # Measurement noise covariance (R):  Model the noise in the measurements
        kf.R = np.eye(dim) * 0.1  # Adjust the multiplier

        # Initial state covariance (P):  Initial uncertainty in the pose
        kf.P = np.eye(dim) * 1  # Adjust the multiplier

        return kf

    def calculate_ear(self, eye_landmarks):
        try:
            p2_p6 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            p3_p5 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            p1_p4 = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
            return ear
        except ZeroDivisionError:
            return 0.0  # Handle potential division by zero

    def get_landmarks(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]  # Assuming only one face
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in self.left_eye_idxs]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in self.right_eye_idxs]
            iris = (int(face_landmarks.landmark[self.iris_idx].x * w), int(face_landmarks.landmark[self.iris_idx].y * h))
            return left_eye, right_eye, iris, face_landmarks
        else:
            return None, None, None, None

    def calibrate_thresholds(self, cap, duration=5):
        # Collect EAR data for calibration duration
        open_ears = []
        closed_ears = []
        start_time = time.time()
        h = 0
        w = 0
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame during calibration.")
                return False  # Indicate calibration failure

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in self.left_eye_idxs]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in self.right_eye_idxs]

                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                # Prompt the user to blink
                if time.time() - start_time < duration/2:
                    print("Look straight and keep your eyes open")
                    open_ears.append((left_ear, right_ear))
                else:
                    print("Blink a couple of times")
                    closed_ears.append((left_ear, right_ear))
            else:
                print("No face detected during calibration.")
                return False # Indicate calibration failure

        if not open_ears or not closed_ears:
            print("Insufficient data during calibration.")
            return False

        open_left_ear = 0.0
        open_right_ear = 0.0
        closed_left_ear = 0.0
        closed_right_ear = 0.0

        for le, re in open_ears:
            open_left_ear += le
            open_right_ear += re
        for le, re in closed_ears:
            closed_left_ear += le
            closed_right_ear += re

        open_left_ear /= len(open_ears)
        open_right_ear /= len(open_ears)
        closed_left_ear /= len(closed_ears)
        closed_right_ear /= len(closed_ears)

        # Calculate dynamic thresholds (e.g., 70% of open EAR)
        self.blink_threshold = (open_left_ear + open_right_ear) / 2 * 0.7
        self.close_threshold = (closed_left_ear + closed_right_ear) / 2 * 1.3  # A bit higher than closed ear
        self.both_eye_blink_threshold = (closed_left_ear + closed_right_ear) / 2 * 1.1
        self.squint_threshold = (open_left_ear + open_right_ear) / 2 * 0.8 # Sometime in between open and closed EAR

        self.open_left_ear = open_left_ear
        self.open_right_ear = open_right_ear
        self.closed_left_ear = closed_left_ear
        self.closed_right_ear = closed_right_ear

        print("Calibration complete.")
        print(f"Blink threshold: {self.blink_threshold:.2f}")
        print(f"Close threshold: {self.close_threshold:.2f}")
        print(f"Squint threshold: {self.squint_threshold:.2f}")
        return True

    def draw_landmarks(self, frame, left_eye, right_eye):
        if left_eye and right_eye:
            for x, y in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1) #Green Color

    def estimate_head_pose(self, face_landmarks, frame_width, frame_height):
        # 2D image points
        image_points = np.array([
            (face_landmarks.landmark[4].x * frame_width, face_landmarks.landmark[4].y * frame_height),     # Nose tip
            (face_landmarks.landmark[152].x * frame_width, face_landmarks.landmark[152].y * frame_height),   # Chin
            (face_landmarks.landmark[226].x * frame_width, face_landmarks.landmark[226].y * frame_height),   # Left eye corner
            (face_landmarks.landmark[454].x * frame_width, face_landmarks.landmark[454].y * frame_height),   # Right eye corner
            (face_landmarks.landmark[61].x * frame_width, face_landmarks.landmark[61].y * frame_height),     # Left mouth corner
            (face_landmarks.landmark[291].x * frame_width, face_landmarks.landmark[291].y * frame_height)    # Right mouth corner
        ], dtype=np.float64)

        # Camera internals
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

        # Solve for pose
        success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            # Correctly reshape the vectors before using them
            rotation_vector = rotation_vector.reshape(-1)
            translation_vector = translation_vector.reshape(-1)

            # Use Kalman filter to smooth the pose
            rotation_vector = self.filter_pose(self.kf_rotation, rotation_vector)
            translation_vector =  self.filter_pose(self.kf_translation, translation_vector)

            return rotation_vector, translation_vector
        else:
            return None, None

    def filter_pose(self, kf, measurement):
        """
        Applies a Kalman filter to smooth the pose measurement.
        """
        kf.predict()
        kf.update(measurement)
        return kf.x  # Returns the filtered state estimate

    def detect_eyebrow_raise(self, face_landmarks, frame_height, eyebrow_raise_threshold = 10):
        # Landmark indices for eyebrows
        left_eyebrow_top = face_landmarks.landmark[105]
        left_eyebrow_bottom = face_landmarks.landmark[107]
        right_eyebrow_top = face_landmarks.landmark[336]
        right_eyebrow_bottom = face_landmarks.landmark[332]

        # Calculate vertical distance between eyebrow landmarks
        left_eyebrow_distance = (left_eyebrow_bottom.y - left_eyebrow_top.y) * frame_height
        right_eyebrow_distance = (right_eyebrow_bottom.y - right_eyebrow_top.y) * frame_height

        # Check if eyebrows are raised
        if left_eyebrow_distance > eyebrow_raise_threshold or right_eyebrow_distance > eyebrow_raise_threshold:
            return True
        else:
            return False
        
import os
import cv2
import json
import time
import pyautogui
import numpy as np
from sklearn.svm import SVC
from face_tracker import FaceTracker
from hand_tracker import HandTracker
from mouse_controller import MouseController

def run_mode(control_mode):
    # Always get the folder of *this* file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

    print("Looking for config.json at:", CONFIG_PATH)  # Debug print

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"config.json not found at: {CONFIG_PATH}")

    # Load configuration
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)


    # Initialize
    screen_w, screen_h = pyautogui.size()
    face_tracker = FaceTracker()
    hand_tracker = HandTracker()

    mouse_config = {k: v for k, v in config.items() if k in ["smoothing_window", "dwell_time", "mouse_speed_factor", "min_distance_threshold"]}
    mouse_controller = MouseController(screen_w, screen_h, **mouse_config)

    cap = cv2.VideoCapture(0)

    if not face_tracker.calibrate_thresholds(cap, duration=config["calibration_duration"]):
        print("Calibration failed. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # SVM setup
    X_train = [[0.25], [0.30], [0.10], [0.05], [0.20], [0.15]]
    y_train = [1, 1, 0, 0, 1, 0]  # 1 = Open, 0 = Closed
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(X_train, y_train)

    frame_count = 0
    start_time = time.time()
    close_time_threshold = 5
    close_start_time = None
    action_delay = 0.2
    last_action_time = 0
    scroll_sensitivity = 0.15
    scroll_region_top = 0.35
    scroll_region_bottom = 0.65
    long_blink_duration = 0.5
    blink_start_time = None
    squint_start_time = None
    squint_duration_threshold = 0.4  # Adjust as needed

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            current_time = time.time()
            frame_height, frame_width, _ = frame.shape
            if 'right_click_pause_until' in locals() and current_time < right_click_pause_until:
                cv2.imshow("Image", frame)
                continue

            # ---------- Eye Control ----------
            if control_mode == "eye":
                left_eye, right_eye, iris, face_landmarks = face_tracker.get_landmarks(frame)
                if left_eye and right_eye and iris and face_landmarks:
                    iris_x, iris_y = iris
                    left_ear = face_tracker.calculate_ear(left_eye)
                    right_ear = face_tracker.calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2
                    eye_state_svm = svm_model.predict([[avg_ear]])[0]

                    mouse_controller.move_mouse(iris_x / frame_width, iris_y / frame_height)
                    mouse_controller.handle_dwell_click(iris_x / frame_width, iris_y / frame_height)

                    if current_time - last_action_time >= action_delay:
                        if right_ear < face_tracker.blink_threshold and left_ear > face_tracker.blink_threshold:
                            print("Left Eye Blinked - Double Click")
                            mouse_controller.double_click()
                            last_action_time = current_time

                        elif left_ear < face_tracker.blink_threshold and right_ear > face_tracker.blink_threshold:
                            print("Right Eye Blinked - Right Click")
                            mouse_controller.right_click()
                            last_action_time = current_time
                            right_click_pause_until = current_time + 1.0  # Pause all actions for 1 second

                        if left_ear < face_tracker.both_eye_blink_threshold and right_ear < face_tracker.both_eye_blink_threshold:
                            if blink_start_time is None:
                                blink_start_time = time.time()
                        else:
                            if blink_start_time is not None:
                                blink_duration = time.time() - blink_start_time
                                if blink_duration >= long_blink_duration:
                                    print("Long Blink - Left Click")
                                    mouse_controller.left_click()
                                    last_action_time = current_time
                                blink_start_time = None
                        if left_ear < face_tracker.squint_threshold and right_ear < face_tracker.squint_threshold:
                            if squint_start_time is None:
                                squint_start_time = time.time()
                            elif time.time() - squint_start_time >= squint_duration_threshold:
                                print("Squint Detected - Minimizing Window")
                                pyautogui.hotkey('alt', 'space')
                                pyautogui.press('n')
                                last_action_time = current_time
                                squint_start_time = None  # Reset after action
                            else:
                                squint_start_time = None  # Reset if eyes aren't squinting anymore

                        scroll_up_threshold = 200    # if iris_y goes above this → scroll up
                        scroll_down_threshold = 270  # if iris_y goes below this → scroll down
                        scroll_amount = 20           # fixed scroll steps
                        if iris_y < scroll_up_threshold:
                            print("Scrolling Up")
                            pyautogui.scroll(scroll_amount)
                            last_action_time = current_time
                        elif iris_y > scroll_down_threshold:
                            print("Scrolling Down")
                            pyautogui.scroll(-scroll_amount)
                            last_action_time = current_time

                    face_tracker.draw_landmarks(frame, left_eye, right_eye)
                    cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    close_start_time = None

            # ---------- Hand Control ----------
            elif control_mode == "hand":
                img = hand_tracker.findHands(frame)
                lmList = hand_tracker.findPosition(frame)
                if len(lmList) != 0:
                    fingers = hand_tracker.fingersUp()
                    x1, y1 = lmList[8][1:]
                    x = np.interp(x1, (0, frame_width), (0, screen_w))
                    y = np.interp(y1, (0, frame_height), (0, screen_h))
                    pyautogui.moveTo(x, y, duration=0.05)
                    if current_time - last_action_time >= action_delay:
                        length, img, _ = hand_tracker.findDistance(4, 8, img)
                        if length < 40:
                            print("Left Click (Pinch)")
                            pyautogui.click(button='left')
                            last_action_time = current_time
                        if fingers == [0, 1, 0, 0, 0]:
                            print("Right Click (Index finger up 👆)")
                            pyautogui.click(button='right')
                            last_action_time = current_time
                        if fingers == [0, 1, 1, 0, 0]:
                            print("Double Click (Index and Middle finger up ✌️ )")
                            pyautogui.doubleClick()
                            last_action_time = current_time
                        if fingers == [0, 0, 0, 0, 0]:
                            print("Minimize Window (Closed Fist ✊)")
                            pyautogui.hotkey('alt', 'space')
                            pyautogui.press('n')
                            last_action_time = current_time
                        if fingers == [1, 0, 0, 0, 0]:
                            print("Scroll Up (Thumb up)")
                            pyautogui.scroll(10)
                            last_action_time = current_time
                        if fingers == [0, 0, 0, 0, 1]:
                            print("Scroll Down (Pinky up)")
                            pyautogui.scroll(-10)
                            last_action_time = current_time

            # ---------- Dual Control ----------
            elif control_mode == "dual":
                left_eye, right_eye, iris, face_landmarks = face_tracker.get_landmarks(frame)
                if left_eye and right_eye and iris and face_landmarks:
                    iris_x, iris_y = iris
                    mouse_controller.move_mouse(iris_x / frame_width, iris_y / frame_height)
                    face_tracker.draw_landmarks(frame, left_eye, right_eye)
                img = hand_tracker.findHands(frame)
                lmList = hand_tracker.findPosition(frame)
                if len(lmList) != 0:
                    fingers = hand_tracker.fingersUp()
                    length, img, _ = hand_tracker.findDistance(4, 8, img)
                    if length < 40:
                        print("Left Click (Pinch)")
                        pyautogui.click(button='left')
                        last_action_time = current_time
                    elif fingers == [0, 1, 0, 0, 0]:
                        print("Right Click (Index finger up 👆)")
                        pyautogui.click(button='right')
                        last_action_time = current_time
                    elif fingers == [0, 1, 1, 0, 0]:
                        print("Double Click (Index and Middle finger up ✌️ )")
                        pyautogui.doubleClick()
                        last_action_time = current_time
                    elif fingers == [0, 0, 0, 0, 0]:
                        print("Minimize Window (Closed Fist ✊)")
                        pyautogui.hotkey('alt', 'space')
                        pyautogui.press('n')
                        last_action_time = current_time
                    elif fingers == [1, 0, 0, 0, 0]:
                        print("Scroll Up (Thumb)")
                        pyautogui.scroll(10)
                        last_action_time = current_time
                    elif fingers == [0, 0, 0, 0, 1]:
                        print("Scroll Down (Pinky)")
                        pyautogui.scroll(-10)
                        last_action_time = current_time

            cv2.imshow("Image", frame)
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                break

            frame_count += 1
            if frame_count % 60 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                start_time = time.time()

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Releasing resources")
        cap.release()
        cv2.destroyAllWindows()

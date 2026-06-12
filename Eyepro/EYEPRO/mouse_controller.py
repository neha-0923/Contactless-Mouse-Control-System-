import pyautogui
import time

class MouseController:
    def __init__(self, screen_width, screen_height, smoothing_window=5, dwell_time=0.7, mouse_speed_factor = 2.0, min_distance_threshold = 0.01):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.smoothing_window = smoothing_window
        self.dwell_time = dwell_time
        self.mouse_speed_factor = mouse_speed_factor
        self.min_distance_threshold = min_distance_threshold
        self.x_history = []
        self.y_history = []
        self.last_iris_position = None
        self.dwell_start_time = None
        self.dwell_position = None

    def move_mouse(self, iris_x, iris_y):
        screen_x = int(iris_x * self.screen_width)
        screen_y = int(iris_y * self.screen_height)

        # Apply smoothing
        self.x_history.append(screen_x)
        self.y_history.append(screen_y)

        if len(self.x_history) > self.smoothing_window:
            self.x_history.pop(0)
            self.y_history.pop(0)

        avg_x = sum(self.x_history) / len(self.x_history)
        avg_y = sum(self.y_history) / len(self.y_history)

        # Check if the iris position has moved significantly
        if self.last_iris_position:
            distance = ((avg_x - self.last_iris_position[0])**2 + (avg_y - self.last_iris_position[1])**2)**0.5
            if distance > self.min_distance_threshold:
                delta_x = avg_x - self.last_iris_position[0]
                delta_y = avg_y - self.last_iris_position[1]

                # Adjust the mouse movement speed
                move_x = delta_x * self.mouse_speed_factor
                move_y = delta_y * self.mouse_speed_factor

                # Move the mouse
                pyautogui.move(move_x, move_y)
        else:
            # Initial mouse position set
            pyautogui.moveTo(avg_x, avg_y)

        # Update the last iris position
        self.last_iris_position = (avg_x, avg_y)

    def handle_dwell_click(self, iris_x, iris_y):
        screen_x = int(iris_x * self.screen_width)
        screen_y = int(iris_y * self.screen_height)

        if self.dwell_position is None:
            # Start dwell timer
            self.dwell_position = (screen_x, screen_y)
            self.dwell_start_time = time.time()
        else:
            # Check if mouse has moved significantly
            distance = ((screen_x - self.dwell_position[0])**2 + (screen_y - self.dwell_position[1])**2)**0.5
            if distance > 20:  # Moved too much, reset dwell
                self.dwell_position = (screen_x, screen_y)
                self.dwell_start_time = time.time()
            elif time.time() - self.dwell_start_time >= self.dwell_time:
                # Dwell time reached, perform click
                pyautogui.click()
                self.dwell_position = None  # Reset
                self.dwell_start_time = None

    def left_click(self):
        pyautogui.click(button='left')
        time.sleep(0.2)  # Small delay to prevent accidental multiple clicks

    def right_click(self):
        pyautogui.click(button='right')
        time.sleep(0.2)

    def double_click(self):
        pyautogui.doubleClick()
        time.sleep(0.2)
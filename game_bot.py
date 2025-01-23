import pyautogui
import cv2
import numpy as np
import mss
import time

click_button_template = cv2.imread("click_button.png", cv2.IMREAD_GRAYSCALE)
skull_template = cv2.imread("skull.png", cv2.IMREAD_GRAYSCALE)

click_button_template = cv2.resize(click_button_template, (30, 30))
skull_template = cv2.resize(skull_template, (30, 30))

CLICK_THRESHOLD = 0.392
SKULL_THRESHOLD = 0.5

last_click_time = 0
WAIT_TIME = 0.75  # 500 ms

def detect_template(template, screen_gray, threshold):
    result = cv2.matchTemplate(screen_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f"Matching Score: {max_val}")
    if max_val >= threshold:
        return max_loc, max_val
    return None, 0

def visualize_match(screenshot, position, score, label, color):
    top_left = position
    bottom_right = (top_left[0] + 30, top_left[1] + 30)
    cv2.rectangle(screenshot, top_left, bottom_right, color, 2)
    cv2.putText(screenshot, f"{label}: {score:.2f}", (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    global last_click_time
    print("Starting game bot. Press Ctrl+C to stop.")
    time.sleep(2)

    with mss.mss() as sct:
        monitor = sct.monitors[2]
        game_region = {"top": monitor["top"] + 250, "left": monitor["left"] + 300, "width": 500, "height": 200}
        game_center_x = game_region["left"] + game_region["width"] // 2
        game_center_y = game_region["top"] + game_region["height"] // 2

        while True:
            current_time = time.time()
            if current_time - last_click_time < WAIT_TIME:
                continue
            screenshot = np.array(sct.grab(game_region))
            screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
            skull_position, skull_score = detect_template(skull_template, screen_gray, SKULL_THRESHOLD)
            if skull_position:
                print(f"Skull detected at {skull_position} with confidence {skull_score}")
                visualize_match(screenshot, skull_position, skull_score, "Skull", (0, 0, 255))
                cv2.imshow("Matches", screenshot)
                #cv2.waitKey(1)
                continue

            click_position, click_score = detect_template(click_button_template, screen_gray, CLICK_THRESHOLD)
            if click_position:
                print(f"Click button detected at {click_position} with confidence {click_score}")
                visualize_match(screenshot, click_position, click_score, "Click", (0, 255, 0))

                pyautogui.click(game_center_x, game_center_y)
                last_click_time = current_time 

                time.sleep(WAIT_TIME)
                pyautogui.click(game_center_x, game_center_y)
            cv2.imshow("Matches", screenshot)
            cv2.waitKey(1)
            time.sleep(0.05)


if __name__ == "__main__":
    main()

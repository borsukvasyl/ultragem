import threading
import time

import cv2

from bot.ScreenReader import GameInterface
from game.ultragem import UltraGemGame


def run_game(gameid=1):
    game = UltraGemGame(gameid=gameid)
    game.run()


if __name__ == '__main__':
    game_thread = threading.Thread(target=run_game)
    game_thread.start()

    time.sleep(5)
    interface = GameInterface()
    while True:
        img, num_boxes = interface.get_field()

        print(num_boxes)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", img)
        if cv2.waitKey(0) & 0xFF == 27:
            cv2.destroyAllWindows()
        break

import threading
import time

from bot.Bot import Bot
from bot.ScreenReader import GameInterface
from game.ultragem import UltraGemGame


def run_game(gameid=1):
    game = UltraGemGame(gameid=gameid)
    game.run()


if __name__ == '__main__':
    game_thread = threading.Thread(target=run_game)
    game_thread.start()

    time.sleep(4)  # wait init
    interface = GameInterface()
    bot = Bot()
    last_img = interface.read_screen()
    while True:
        img = interface.read_screen()
        if (img == last_img).all():
            field = interface.get_field()
            pair1, pair2, score = bot.get_best_move(field)
            # ####
            # with open("koko.txt", "w") as f:
            #     for i in field:
            #         for j in i:
            #             f.write(j["cell_id"])
            #             f.write(" ")
            #         f.write("\n")
            #     f.write(" ".join(map(str, pair1)) + "    " + " ".join(map(str, pair2)) + "\n")
            # ####
            interface.interact(pair1, pair2)
        time.sleep(1)
        last_img = img
    game_thread.join()

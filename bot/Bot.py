from copy import deepcopy


class Bot(object):
    def __init__(self):
        pass

    def get_best_move(self, field):
        max_score = 0
        pair1, pair2 = None, None
        for i_d, j_d in [(0, 1), (1, 0)]:
            for i in range(7):
                for j in range(7):
                    b = deepcopy(field)
                    b[i][j] = b[i + i_d][j + j_d]
                    score = self.calculate_score(b)
                    if score > max_score:
                        max_score = score
                        pair1 = (i, j)
                        pair2 = (i + i_d, j + j_d)
        return pair1, pair2, max_score

    @staticmethod
    def calculate_score(board):
        score = 0
        i, j = 0, 0
        while True:
            k = 1
            while j + k < 8 and board[i][j]["cell_id"] == board[i][j + k]["cell_id"]:
                if k == 2:
                    score += 3
                elif k > 2:
                    score += 1
                k += 1
            j += k
            if j >= 8:
                j = 0
                i += 1
            if i == 8:
                break

        i, j = 0, 0
        while True:
            k = 1
            while i + k < 8 and board[i][j]["cell_id"] == board[i + k][j]["cell_id"]:
                if k == 2:
                    score += 3
                elif k > 2:
                    score += 1
                k += 1
            i += k
            if i >= 8:
                i = 0
                j += 1
            if j == 8:
                break
        return score
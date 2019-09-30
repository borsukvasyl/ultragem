import cv2
import numpy as np
import pygame


class GameInterface(object):
    def __init__(self, surface=None, min_line_len=0.5, min_line_dist=0.02):
        self.surface = surface if surface is not None else pygame.display.get_surface()
        self.min_line_len = min_line_len * self.surface.get_height()
        self.min_line_dist = min_line_dist * self.surface.get_height()

    def get_field(self):
        img = self.read_screen()
        bbox, num_boxes = self.get_field_location(img)
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]], num_boxes

    def read_screen(self):
        img = pygame.surfarray.array3d(self.surface)
        img = img.swapaxes(0, 1)
        return img

    def get_field_location(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thr = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)

        runs = self._find_runs(thr, 1)
        ys = runs[0].reshape(-1, 2)
        xs = runs[1].reshape(-1, 2)
        num_boxes = -1
        res_xs, res_ys = [], []
        for x, y in zip(xs, ys):
            if x[1] - x[0] > self.min_line_len and (num_boxes == -1 or y[0] - res_ys[-1][0] > self.min_line_dist):
                num_boxes += 1
                res_xs.append(x)
                res_ys.append(y)
        return (np.min(res_xs), np.min(res_ys), np.max(res_xs), np.max(res_ys)), (num_boxes, num_boxes)

    def _find_runs(self, a, v):
        is_v = np.pad(np.equal(a, v), ((0, 0), (1, 1)))
        absdiff = np.abs(np.diff(is_v, axis=-1))
        ranges = np.where(absdiff)
        return ranges

    def interact(self, x, y):
        pass

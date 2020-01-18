import glob
import os

import cv2
import numpy as np
import pygame
from math import ceil

from game.ultragem import BGCOLOR, GRIDCOLOR


class GameInterface(object):
    GRAPHICS_ROOT = os.path.join("game", "graphics")

    def __init__(self, surface=None, min_line_len=0.5, min_line_dist=0.02):
        self.surface = surface if surface is not None else pygame.display.get_surface()
        self.min_line_len = min_line_len * self.surface.get_height()
        self.min_line_dist = min_line_dist * self.surface.get_height()

        self._init()

    def get_field(self):
        img = self.read_screen()
        start_w, start_h = self.field_bbox[:2]
        w, h = self.bbox_shape

        boxes = []
        field = np.zeros(self.num_boxes, int)
        for i in range(self.num_boxes[0]):
            for j in range(self.num_boxes[1]):
                x1, y1 = i * w, j * h
                x2, y2 = (i + 1) * w, (j + 1) * h
                box = img[start_h + y1:start_h + y2, start_w + x1:start_w + x2]
                field[j, i] = self._get_label(box)
                boxes.append(box)
        cv2.imshow("image", cv2.cvtColor(np.hstack(boxes), cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) & 0xFF == 27:
            cv2.destroyAllWindows()
        return field

    def interact(self, x, y):
        pass

    def crop_field(self):
        img = self.read_screen()
        return img[self.field_bbox[1]:self.field_bbox[3], self.field_bbox[0]:self.field_bbox[2]]

    def read_screen(self):
        img = pygame.surfarray.array3d(self.surface)
        img = img.swapaxes(0, 1)
        return img

    def _get_label(self, img):
        emb = img[~(np.all(img == BGCOLOR, axis=-1) | np.all(img == GRIDCOLOR, axis=-1))].mean(0)
        dist = np.sqrt(np.sum((emb - self.embs) ** 2, axis=1))
        return dist.argmin()

    def _init(self):
        img = self.read_screen()
        self.field_bbox, self.num_boxes = self._get_field_location(img)
        self.bbox_shape = (ceil((self.field_bbox[2] - self.field_bbox[0]) / self.num_boxes[0]),
                           ceil((self.field_bbox[3] - self.field_bbox[1]) / self.num_boxes[1]))
        self.embs = self._read_images()

    def _get_field_location(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, thr = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY_INV)

        runs = GameInterface._find_runs(thr, 1)
        xs = runs[0].reshape(-1, 2)
        ys = runs[1].reshape(-1, 2)
        res_xs, res_ys = [], []
        for x, y in zip(xs, ys):
            if y[1] - y[0] > self.min_line_len and (not len(res_xs) or x[0] - res_xs[-1][0] > self.min_line_dist):
                res_xs.append(x)
                res_ys.append(y)
        bbox = (np.min(res_xs), np.min(res_ys), np.max(res_xs), np.max(res_ys))
        num_boxes = (len(res_xs) - 1, len(res_ys) - 1)
        return bbox, num_boxes

    @staticmethod
    def _find_runs(a, v):
        is_v = np.pad(np.equal(a, v), ((1, 1), (0, 0)))
        absdiff = np.abs(np.diff(is_v, axis=0))
        ranges = np.where(absdiff.transpose())
        return ranges

    @staticmethod
    def _read_images():
        res = []
        for i, file_path in enumerate(glob.glob("{}/gem[0-9].png".format(GameInterface.GRAPHICS_ROOT))):
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            res.append(img[img[..., -1] == 0][..., :-1].mean(0))
        return np.vstack(res)

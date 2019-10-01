import glob
import os

import cv2
import numpy as np
import pygame
from skimage.measure import compare_ssim


class GameInterface(object):
    GRAPHICS_ROOT = os.path.join("game", "graphics")

    def __init__(self, surface=None, min_line_len=0.5, min_line_dist=0.02):
        self.surface = surface if surface is not None else pygame.display.get_surface()
        self.min_line_len = min_line_len * self.surface.get_height()
        self.min_line_dist = min_line_dist * self.surface.get_height()
        self.field_bbox = None
        self.num_boxes = None
        self.bbox_shape = None
        self.imgs = None

    def get_field(self):
        field_img = self.crop_field()

        max_h, max_w = field_img.shape[:2]
        h, w = self.bbox_shape

        field = np.zeros(self.num_boxes, int)
        for i in range(self.num_boxes[0]):
            for j in range(self.num_boxes[1]):
                x1, y1 = i * h, j * w
                x2, y2 = min((i + 1) * h, max_h), min((j + 1) * w, max_w)
                box = field_img[x1:x2, y1:y2]

                best_score, best_id = 0, None
                for img_id, img in self.imgs:
                    ssim = self._get_similarity(box, img)
                    if ssim > best_score:
                        best_score = ssim
                        best_id = img_id
                field[i, j] = best_id
        return field

    def interact(self, x, y):
        pass

    def crop_field(self):
        img = self.read_screen()
        if self.field_bbox is None:
            self.field_bbox, self.num_boxes = self.get_field_location(img)
            field = img[self.field_bbox[1]:self.field_bbox[3], self.field_bbox[0]:self.field_bbox[2]]
            self.bbox_shape = (int(field.shape[0] / self.num_boxes[0]), int(field.shape[1] / self.num_boxes[1]))
            self.imgs = self._read_images(self.bbox_shape)
            return field

        return img[self.field_bbox[1]:self.field_bbox[3], self.field_bbox[0]:self.field_bbox[2]]

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
        res_xs, res_ys = [], []
        for x, y in zip(xs, ys):
            if x[1] - x[0] > self.min_line_len and (not len(res_ys) or y[0] - res_ys[-1][0] > self.min_line_dist):
                res_xs.append(x)
                res_ys.append(y)
        return (np.min(res_xs), np.min(res_ys), np.max(res_xs), np.max(res_ys)), (len(res_xs), len(res_ys))

    def _find_runs(self, a, v):
        is_v = np.pad(np.equal(a, v), ((0, 0), (1, 1)))
        absdiff = np.abs(np.diff(is_v, axis=-1))
        ranges = np.where(absdiff)
        return ranges

    def _get_similarity(self, img1, img2):
        img2 = cv2.resize(img2, img1.shape[1::-1])
        return compare_ssim(img1, img2, multichannel=True)

    @staticmethod
    def _read_images(bbox_shape):
        res = []
        for i, file_path in enumerate(glob.glob("{}/comb*.png".format(GameInterface.GRAPHICS_ROOT))):
            img = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, bbox_shape)
            res.append((i, img))
        return res

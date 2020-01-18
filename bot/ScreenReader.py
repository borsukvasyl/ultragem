import glob
import os

import cv2
import numpy as np
import pygame
from math import ceil
from patchify import patchify


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

        field = [[None for _ in range(self.num_boxes[1])] for _ in range(self.num_boxes[0])]
        for i in range(self.num_boxes[0]):
            for j in range(self.num_boxes[1]):
                x1, y1 = i * w, j * h
                x2, y2 = (i + 1) * w, (j + 1) * h
                box = img[start_h + y1:start_h + y2, start_w + x1:start_w + x2]
                field[j][i] = self._get_label(box)
        return field

    def interact(self, pair1, pair2):
        pos1 = (self.field_bbox[0] + pair1[1] * self.bbox_shape[0] + self.bbox_shape[0] // 2,
                self.field_bbox[1] + pair1[0] * self.bbox_shape[1] + self.bbox_shape[1] // 2)
        pos2 = (self.field_bbox[0] + pair2[1] * self.bbox_shape[0] + self.bbox_shape[0] // 2,
                self.field_bbox[1] + pair2[0] * self.bbox_shape[1] + self.bbox_shape[1] // 2)

        # with open("koko.txt", "a") as f:
        #     f.write(" ".join(map(str, pos1)) + "    " + " ".join(map(str, pos2)) + "\n")
        pygame.event.post(pygame.event.Event(pygame.locals.MOUSEBUTTONDOWN, {"pos": pos1}))
        pygame.event.post(pygame.event.Event(pygame.locals.MOUSEBUTTONUP, {"pos": pos1}))
        pygame.event.post(pygame.event.Event(pygame.locals.MOUSEBUTTONDOWN, {"pos": pos2}))
        pygame.event.post(pygame.event.Event(pygame.locals.MOUSEBUTTONUP, {"pos": pos1}))

    def crop_field(self):
        img = self.read_screen()
        return img[self.field_bbox[1]:self.field_bbox[3], self.field_bbox[0]:self.field_bbox[2]]

    def read_screen(self):
        img = pygame.surfarray.array3d(self.surface)
        img = img.swapaxes(0, 1)
        return img

    def _get_label(self, img):
        parts_per_side, resize, padding = self.embs['parts_per_side'], self.embs['padding'], self.embs['resize_shape']
        emb = self.get_emb(img, parts_per_side, resize, padding)
        cell_embs = np.array([e['emb'] for e in self.embs['embs']])
        result = (cell_embs - emb[None, :])
        idx = np.argmin(np.sum(np.abs(result), axis=1))
        return self.embs['embs'][idx]

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
    def read_cell(img_path):
        img = cv2.imread(img_path, -1)
        mask = (img[:, :, 3] > 100.)
        img[:, :, 0][~mask] = 252.
        img[:, :, 1][~mask] = 188.
        img[:, :, 2][~mask] = 159.
        img = img[:, :, :3]
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def get_emb(img, parts_per_side, padding, resize_shape):
        if img.shape[0] != resize_shape:
            img = cv2.resize(img, (resize_shape, resize_shape))
        img = img[padding:-padding, padding:-padding, :]
        part_size = img.shape[0] // parts_per_side
        patches = patchify(img, (part_size, part_size, 3), step=part_size).reshape(-1, part_size, part_size, 3)
        return np.mean(patches, axis=(1, 2)).reshape(-1)

    @staticmethod
    def get_analytics(img_path, parts_per_side, padding, resize_shape):
        img = GameInterface.read_cell(img_path)
        emb = GameInterface.get_emb(img, parts_per_side, padding, resize_shape)
        splitted = os.path.basename(img_path).split('-')
        cell_id, attr = splitted[1], splitted[2].split('.')[0]
        return cell_id, attr, emb

    @staticmethod
    def _read_images():
        res = {
            'parts_per_side': 3,
            'padding': 2,
            'resize_shape': 32,
            'embs': list()
        }
        img_paths = glob.glob("{}/combN*".format(GameInterface.GRAPHICS_ROOT))
        for img_path in img_paths:
            cell_id, attr, emb = GameInterface.get_analytics(
                img_path, res["parts_per_side"], res["padding"], res["resize_shape"])
            res["embs"].append({
                'cell_id': cell_id,
                'attr': attr,
                'emb': emb,
                'img_path': img_path
            })
        return res

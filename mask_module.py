import cv2
import imutils
import numpy as np


class Mask(object):
    def __init__(self, settings_o, min_area):
        print("Init mask object...")
        self.fgmask = []
        self.settings_o = settings_o
        self.min_area = min_area
        width, height = self.settings_o.get_size()
        self.accum_image = np.zeros((height, width), np.uint8)

    def get_countours(self, prev, current):
        if prev.shape == current.shape:
            mask = cv2.absdiff(prev, current)
        else:
            mask = cv2.absdiff(current, current)
        mask = cv2.threshold(mask, 25, 255, cv2.THRESH_BINARY)[1]
        self.fgmask = cv2.dilate(mask, None, iterations=2)
        countours = cv2.findContours(self.fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        countours = imutils.grab_contours(countours)
        return countours

    def get_mask(self):
        fgmask = self.fgmask.copy()
        fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2RGB)
        fgmask[np.where((fgmask == [255, 255, 255]).all(axis=2))] = [0, 0, 225]
        fgmask[np.where((fgmask == [127, 127, 127]).all(axis=2))] = [0, 0, 225]
        return fgmask

    def clear_accum(self):
        width, height = self.settings_o.get_size()
        self.accum_image = np.zeros((height, width), np.uint8)

    def update_accum(self):
        mask = self.fgmask.copy()
        mask[mask == 255] = 1
        if self.accum_image.shape == mask.shape:
            self.accum_image = cv2.add(self.accum_image, mask)
        else:
            self.accum_image = mask
        max_arr = self.accum_image.max()
        if (max_arr > 250):
            self.accum_image = np.divide(self.accum_image, 1.01)
            self.accum_image = self.accum_image.astype(np.uint8)
        #print(max_arr, self.accum_image.max())

    def get_heatmap(self):
        colormap = cv2.applyColorMap(self.accum_image, cv2.COLORMAP_JET)
        colormap_rgba = cv2.cvtColor(colormap, cv2.COLOR_RGB2RGBA)
        colormap_rgba[np.where((colormap_rgba == [128, 0, 0, 255]).all(axis=2))] = [0, 0, 0, 0]
        return colormap_rgba

    def get_min_area(self):
        return self.min_area

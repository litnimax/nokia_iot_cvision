import cv2
import imutils
import numpy as np


class Mask(object):
    def __init__(self, settings_o):
        print("Init mask object...")
        self.fgmask = []
        self.settings_o = settings_o
        try:
            self.accum_image = np.load("data/heatmap.npy")
            print("Heatmap loaded")
        except Exception as ex:
            print("No load heatmap, create new")
            self.accum_image = np.zeros(self.settings_o.get_size(), np.uint32)

    def __del__(self):
        print("Save heatmap...")
        np.save("data/heatmap", self.accum_image)


    def clear_accum(self):
        self.accum_image = np.zeros(self.settings_o.get_size(), np.uint32)

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

    def update_accum(self):
        mask = self.fgmask.copy()
        mask[mask == 255] = 1
        if self.accum_image.shape == mask.shape:
            self.accum_image = self.accum_image + mask
        else:
            self.clear_accum()

    def get_heatmap(self):
        max_arr = self.accum_image.max()
        if (max_arr > 4294967295-10):
            self.accum_image = np.divide(self.accum_image, 1.01)
            self.accum_image = self.accum_image.astype(np.uint8)

        temp_accum_image = self.accum_image.copy()
        divider = (max_arr+1)/254
        temp_accum_image = temp_accum_image/divider
        temp_accum_image = temp_accum_image.astype(np.uint8)
        colormap = cv2.applyColorMap(temp_accum_image, cv2.COLORMAP_JET)
        colormap_rgba = cv2.cvtColor(colormap, cv2.COLOR_RGB2RGBA)
        colormap_rgba[np.where((colormap_rgba == [128, 0, 0, 255]).all(axis=2))] = [0, 0, 0, 0]
        return colormap_rgba

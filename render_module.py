import cv2
import numpy as np
from shapely.geometry import Polygon

class Render():
    def __init__(self, mask_o, frame_o, settings_o):
        self.frame_o = frame_o
        self.mask_o = mask_o
        self.settings_o = settings_o

    def render_user_frame(self):
        countours = self.mask_o.get_countours(self.frame_o.get_prev_frame(), self.frame_o.get_current_frame())
        overlay_frame = self.mask_o.get_mask()
        overlay_frame = self.frame_o.render_detect_areas(overlay_frame, self.settings_o.get_areas())
        frame = cv2.cvtColor(self.frame_o.get_color_frame(), cv2.COLOR_RGB2RGBA)

        for countour in countours:
            if cv2.contourArea(countour) < self.settings_o.get_min_area():
                continue

            (x, y, w, h) = cv2.boundingRect(countour)
            countour_rect = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            cv2.polylines(overlay_frame, np.array([countour_rect]), True, (127, 255, 127), 2)
            cv2.polylines(overlay_frame, np.array([countour]), True, (127, 255, 127), 1)

            for key, detect_area in self.settings_o.get_areas().items():
                detect_area_pl = Polygon(detect_area)
                countour_area_pl = Polygon(countour_rect)
                intersect = detect_area_pl.intersects(countour_area_pl)
                if (intersect):
                    cv2.polylines(overlay_frame, np.array([detect_area]), True, (0, 0, 255), 3)

        overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGB2RGBA)
        overlay_frame[np.where((overlay_frame == [0, 0, 0, 255]).all(axis=2))] = [0, 0, 0, 0]
        #cv2.putText(frame, "FPS: %.1f" % self.frame_o.get_fps(), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        user_image = cv2.addWeighted(frame, 1, overlay_frame, 0.5, 0)
        return user_image

    def render_heatmap_frame(self):
        frame = cv2.cvtColor(self.frame_o.get_color_frame(), cv2.COLOR_RGB2RGBA)
        heatmap = self.mask_o.get_heatmap()
        heatmap_user_image = cv2.addWeighted(frame, 0.7, heatmap, 0.5, 0)
        return heatmap_user_image

    def render_real_frame(self):
        countours = self.mask_o.get_countours(self.frame_o.get_prev_frame(), self.frame_o.get_current_frame())
        real_image = self.frame_o.get_current_frame()
        real_image = cv2.cvtColor(real_image, cv2.COLOR_GRAY2RGB)
        for countour in countours:
            if cv2.contourArea(countour) < self.settings_o.get_min_area():
                continue
            cv2.polylines(real_image, np.array([countour]), True, (127, 255, 127), 1)
        return real_image

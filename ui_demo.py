import numpy as np
import cv2

# ============================================================================

CANVAS_SIZE = (600,800)

FINAL_LINE_COLOR = (255, 255, 255)
WORKING_LINE_COLOR = (127, 127, 127)

ENTER_KEYCODE = 13
N_KEYCODE = 110

# ============================================================================

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.canvas = np.zeros(CANVAS_SIZE, np.uint8)


    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
            self.canvas = np.zeros(CANVAS_SIZE, np.uint8)
            if (len(self.points) > 0):
                cv2.polylines(self.canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 1)
                cv2.line(self.canvas, self.points[-1], self.current, WORKING_LINE_COLOR)
                cv2.imshow(self.window_name, self.canvas)
        elif event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.imshow(self.window_name, np.zeros(CANVAS_SIZE, np.uint8))
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
          if cv2.waitKey(50) == ENTER_KEYCODE: # enter hit
            self.done = True

        print("Completing polygon with %d points: %s" % (len(self.points), self.points))
        self.canvas = np.zeros(CANVAS_SIZE, np.uint8)
        if (len(self.points) > 0):
            cv2.fillPoly(self.canvas, np.array([self.points]), FINAL_LINE_COLOR)
        cv2.imshow(self.window_name, self.canvas)

        key = cv2.waitKey()
        if (key == ENTER_KEYCODE):
          cv2.destroyWindow(self.window_name)
          return "end"
        elif (key == N_KEYCODE):
          return "new"

# ============================================================================

if __name__ == "__main__":
    while(PolygonDrawer("Polygon").run() != "end"):
      print("")


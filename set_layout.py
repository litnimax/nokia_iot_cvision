import numpy as np
import cv2
import argparse
import signal
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="video type")
args = vars(ap.parse_args())

def signal_handler(sig, frame):
    print('\nRelease cap..')
    cap.release()
    cv2.destroyAllWindows()
    print('Exit..')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if args.get("video", None) is None:
	cap = cv2.VideoCapture(0)
else:
	cap = cv2.VideoCapture(args["video"])
# ============================================================================

FINAL_LINE_COLOR = (127, 255, 127)
WORKING_LINE_COLOR = (127, 127, 255)

ENTER_KEYCODE = 13
N_KEYCODE = 110

# ============================================================================

ret, last_frame = cap.read()

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (20,20)
fontScale              = 0.5
fontColor              = (0,0,127)
lineType               = 1
text                   = 'LB to set new point, enter to end polygon, n to new polygon'
cv2.putText(last_frame, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

class PolygonDrawer(object):
    def __init__(self, window_name):
        self.window_name = window_name # Name for our window
        self.done = False # Flag signalling we're done
        self.current = (0, 0) # Current position, so we can draw the line-in-progress
        self.points = [] # List of points defining our polygon
        self.canvas = last_frame.copy()


    def on_mouse(self, event, x, y, buttons, user_param):
        if self.done: # Nothing more to do
            return

        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.current = (x, y)
            self.canvas = last_frame.copy()
            if (len(self.points) > 0):
                cv2.polylines(self.canvas, np.array([self.points]), False, FINAL_LINE_COLOR, 2)
                cv2.line(self.canvas, self.points[-1], self.current, WORKING_LINE_COLOR, 2)
                cv2.imshow(self.window_name, self.canvas)
        elif event == cv2.EVENT_LBUTTONDOWN:
            print("Adding point #%d with position(%d,%d)" % (len(self.points), x, y))
            self.points.append((x, y))
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self.done = True

    def run(self):
        # Let's create our working window and set a mouse callback to handle events
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(self.window_name, 20, 20)
        cv2.imshow(self.window_name, last_frame.copy())
        cv2.waitKey(1)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while(not self.done):
          if cv2.waitKey(50) == ENTER_KEYCODE: # enter hit
            self.done = True

        print("Completing polygon with %d points: %s" % (len(self.points), self.points))
        self.canvas = last_frame.copy()
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


import cv2
from screeninfo import get_monitors
import numpy as np
import math
from scipy.spatial.transform import Rotation


import ctypes  # An included library with Python install.
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

message = "    To highlight a point, press it with a double click.\n    Once you select a point, click next to the background to compare the intensity for it.\n    You can select several points. Do not forget for all of them to select a pair with background immediately.\n    After you have selected all the points, press the Enter for further processing."
# Mbox('Program usage rules', message, 1)

SELECT = 0
screen = get_monitors()
# print(screen[0].height)


for monitor in get_monitors():
    work_area = [monitor.width, monitor.height - 100]

    print(str(work_area[0]) + 'x' + str(work_area[1]))

crds = np.array([])
i=1
M = np.array([])

def mouse_action(event, x, y, flags, param):
    global crds, img, selected_img, union_img, img_ori, SELECT, I, i, M

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if crds.shape[0] == 2:
            crds = np.append(crds, [x, y])
            print("crds", crds)
            get_crds(crds)
            crds = np.array([])



        else:
            crds = np.append(crds, [x, y])  # save click coordinates
            print("crds", crds)
    cv2.imshow("Z project", img)

def calculate_I(sel_img):
    s=1

def get_crds(crds):
    x1, y1, x2, y2 = crds

    if x2<x1:
        tmp = x1
        x1 = x2
        x2 = tmp

        tmp = y1
        y1 = y2
        y2 = tmp
        
    a = (y2-y1)/(x2-x1)

    if a > -0.01 and a<0.01:
        x3=x1
        y3=y1-3
        x4=x2
        y4=y2+3
        cv2.rectangle(img, (int(x3), int(y3)), (int(x4), int(y4)), (255, 255, 0), 1)
        return True
    else:
        x3 = int(round(x1 + 3 / math.sqrt(1 + 1 / (a * a))))
        y3 = int(round(-x3 / a + x1 / a + y1))

        x4 = int(round(x2 - 3 / math.sqrt(1 + 1 / (a * a))))
        y4 = int(round(-x4 / a + x2 / a + y2))

        x5 = int(round(x1 - 3 / math.sqrt(1 + 1 / (a * a))))
        y5 = int(round(-x5 / a + x1 / a + y1))

        x6 = int(round(x2 + 3 / math.sqrt(1 + 1 / (a * a))))
        y6 = int(round(-x6 / a + x2 / a + y2))

    y_array = np.array([y3, y4, y5, y6])
    x_array = np.array([x3, x4, x5, x6])

    y_min = np.min(y_array)
    y_max = np.max(y_array)

    x_min = np.min(x_array)
    x_max = np.max(x_array)

    for yy in np.arange(y_min, y_max):
        for xx in np.arange(x_min, x_max):
            if xx > line1(yy, a, x1, y1) and xx >line3(yy, a, x4, y4) and xx < line2(yy, a, x3, y3) and xx < line4(yy, a, x2, y2):
                img[yy, xx] = (255, 255, 0)

    return True
def line1(yy, a, x1, y1):
    # x=1/a*y-1/a*b
    return -a * yy + x1 + a * y1
def line2(yy, a, x3, y3):
    return yy / a - y3 / a + x3
def line3(yy, a, x4, y4):
    return yy / a - y4 / a + x4
def line4(yy, a, x2, y2):
    return -a * yy + x2 + a * y2

#------------change file name
img_ori = cv2.imread("pics/1.jpg")

cv2.namedWindow('Z project', cv2.WINDOW_FULLSCREEN)
cv2.moveWindow('Z project', int(0.1 * work_area[0]), int(0.1 * work_area[1]))
win = cv2.getWindowImageRect("Z project")



img_ori_h, img_ori_w = img_ori.shape[0:2] # original image width and height
img = img_ori.copy()
selected_img = []
union_img = []
cv2.imshow("Z project", img)
I = np.array([])


cv2.setMouseCallback('Z project', mouse_action)


while cv2.getWindowProperty("Z project", cv2.WND_PROP_VISIBLE) > 0:
    # display the image and wait for a keypress
    # print("waiting")
    key = cv2.waitKey(0) & 0xFF
    # print(key)


    if key == ord("q"):
        cv2.destroyAllWindows()
        break
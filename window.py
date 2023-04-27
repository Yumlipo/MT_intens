import os
import cv2
from screeninfo import get_monitors
import numpy as np
import math
from scipy.spatial.transform import Rotation


import ctypes  # An included library with Python install.
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

message = "    To highlight a microtube, double click on one of its ends\n and then double click on the other.\n You can also highlight another MT.\n Press Enter for further processing."
# Mbox('Program usage rules', message, 1)
img_path = input('Input the picture path \n')
stack_path = input('Input the stack path or press Enter for auto\n')
if stack_path == "":
    stack_path = os.path.splitext(img_path)[0] + ".nd2"

video_name = os.path.splitext(os.path.basename(img_path))[0]
output_dir = "output/" + video_name + "/"
os.makedirs(output_dir, exist_ok=True)

print(output_dir, stack_path)

#img_path = "D:\\lab\\F-23\\20_100\\AVG_9_TIRF_20ms_100%_999_5x.jpg"
#stack_path = "D:\\lab\\F-23\\20_100\\9_TIRF_20ms_100%_999_5x.nd2"
#C:\\Users\\YummyPolly\\Documents\\LAB\\02-04-2023\\TIRF_10laser_1to5_labe.jpg

SELECT = 0
screen = get_monitors()
# print(screen[0].height)

for monitor in get_monitors():
    work_area = [monitor.width, monitor.height - 100]

    print(str(work_area[0]) + 'x' + str(work_area[1]))

crds_tmp = np.array([])
crds = np.array([])
i=1
M = np.array([])

#This is our rectangle
#    (x3, y3)------------------------(x6, y6)
#       |                               |
#    (x1, y1)---MT is on this line---(x2, y2)
#       |                               |
#    (x5, y5)------------------------(x4, y4)

def mouse_action(event, x, y, flags, param):
    global crds, crds_tmp, img, selected_img, union_img, img_ori, SELECT, I, i, M

    if event == cv2.EVENT_LBUTTONDBLCLK:
        if crds_tmp.shape[0] == 2:#This is second click, so we need process selected MT
            #crds_tmp store the time coordinates for only one MT, and crds store all the coordinates of all the allocated MT
            crds_tmp = np.append(crds_tmp, [x, y])
            crds = np.append(crds, [x, y])
            x3, y3, x4, y4, ang, M = get_crds(crds_tmp)
            img_rotated = cv2.warpAffine(img, M, (img_ori_w, img_ori_h))
            # cv2.imshow("Rotated by 45 Degrees", rotated)
            # copy selected area from rotated picture
            selected_img = img_rotated[y3:y4, x3:x4].copy()
            cv2.rectangle(img_rotated, (x3, y3), (x4, y4), (255, 255, 0), 1)#draw rectangle on full img

            cv2.imshow(f"Selected Image {i}", selected_img)
            cv2.imwrite(output_dir + f"selected_image_{i}.jpg", selected_img)

            # cv2.rectangle(selected_img, (0, 0), (8, 8), (255, 255, 255), 1)
            #Intensity from z-project
            I = np.append(I, calculate_I(selected_img))
            #clear crds_tmp becous we are done with this MT
            crds_tmp = np.array([])

            cv2.imshow("Img rotated " + str(i), img_rotated)
            cv2.imwrite(output_dir + f"rotated_image_{i}.jpg", img_rotated)
            #counter of MT
            i += 1

        else:#We are here if it is the first click
            crds_tmp = np.append(crds_tmp, [x, y])  # save click coordinates
            crds = np.append(crds, [x, y])
    cv2.imshow("Z project", img)

def calculate_I(sel_img):
    Intens = 0
    for i in range(sel_img.shape[0]):
        for j in range(sel_img.shape[1]):
            Intens += 0.299 * sel_img[i, j, 2] + 0.587 * sel_img[i, j, 1] + 0.114 * sel_img[i, j, 0]
    return round(Intens / sel_img.shape[1])

def get_crds(crds_tmp):#Get coordinats and ratation matrix to make the picture horizontal and draw the right rectangle on it
    x1, y1, x2, y2 = crds_tmp
    l = math.sqrt((x1-x2)**2 + (y1-y2)**2)

    if x2 < x1:
        tmp = x1
        x1 = x2
        x2 = tmp

        tmp = y1
        y1 = y2
        y2 = tmp

    ang = math.atan((y2-y1)/(x2-x1)) * 180 / math.pi
    M = cv2.getRotationMatrix2D((x1, y1), ang, 1.0)

    x2_new = x1 + l
    y2_new = y1
#return x3, y3, x4, y4, ang and M
    return int(x1), int(y1-3), int(x2_new), int(y2_new+3), ang, M


#------------change file name
img_ori = cv2.imread(img_path)

cv2.namedWindow('Z project', cv2.WINDOW_NORMAL)
cv2.moveWindow('Z project', int(0.1 * work_area[0]), int(0.1 * work_area[1]))
win = cv2.getWindowImageRect("Z project")



img_ori_h, img_ori_w = img_ori.shape[0:2] # original image width and height
img = img_ori.copy()
selected_img = []
union_img = []
cv2.imshow("Z project", img)
I = np.array([])

cv2.setMouseCallback('Z project', mouse_action)

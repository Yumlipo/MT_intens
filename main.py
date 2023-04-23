import cv2
import numpy as np
from nd2reader import ND2Reader
import matplotlib.pyplot as plt

import window
import processing

# processing.smoothing(1, 0)


#Open ND2 file
stack = ND2Reader(window.stack_path)
# plt.imshow(stack[5])
print(stack.metadata)
# plt.show()


IminusBG = np.array([])

while cv2.getWindowProperty("Z project", cv2.WND_PROP_VISIBLE) > 0:
    # display the image and wait for a keypress
    # print("waiting")
    key = cv2.waitKey(0) & 0xFF
    # print(key)

    if key == 13:  # If 'enter' is pressed calculate I(point) - I(BG)
        IminusBG = []
        I_point = []
        I_BG = []
        print("I in main")

        #for each picture in ND2 stack calculate intensity
        for img_for in stack:
            I_point_temp = np.array([])
            I_BG_temp = np.array([])

            # print("Img", img)
            for x1, y1, x2, y2 in zip(window.crds[::4], window.crds[1::4], window.crds[2::4], window.crds[3::4]):
                I_point_temp = np.append(I_point_temp, processing.int_from_rect(x1, y1, x2, y2, img_for))
                I_BG_temp = np.append(I_BG_temp, processing.int_from_rect(x1+10, y1+10, x2+10, y2+10, img_for))
            #these are full arrays of intensity depending on the time for BG and MT
            I_point += [I_point_temp]
            I_BG += [I_BG_temp]
            mean_BG = sum(I_BG) / len(I_BG)
            IminusBG += [I_point_temp - I_BG_temp]
        IminusBG_arr = np.stack(IminusBG, axis=1)
        I_point_arr = np.stack(I_point, axis=1)
        I_BG_arr = np.stack(I_BG, axis=1)
        # I_BG_arr[I_BG_arr != I_BG_arr.mean()] = I_BG_arr.mean()

        #Get grafc I(t) and get tau fron fitting
        tau, I0 = processing.draw_results_and_param(IminusBG_arr, I_point_arr, I_BG_arr)
        print("tau, I0", tau, I0)

        #Save our results in file
        np.savetxt('params.txt', (tau, I0))
        np.savetxt('crds.txt', window.crds)
        np.savetxt('I_BG(t).txt', IminusBG_arr)
        plt.show()

            # for I in IminusBG_arr:
            #     processing.hist(I)


    if key == ord("q"):
        cv2.destroyAllWindows()
        break


import cv2
import numpy as np
from nd2reader import ND2Reader
import matplotlib.pyplot as plt

import window
import processing

# processing.smoothing(1, 0)

stack = ND2Reader('pics/example.nd2')
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
        print("I in main", window.I)

        for img_for in stack:
            I_point_temp = np.array([])
            I_BG_temp = np.array([])

            # print("Img", img)
            for x1, y1, x2, y2 in zip(window.crds[::4], window.crds[1::4], window.crds[2::4], window.crds[3::4]):
                I_point_temp = np.append(I_point_temp, processing.int_from_rect([x1, y1, x2, y2], img_for))
                I_BG_temp = np.append(I_BG_temp, processing.int_from_rect([x1+10, y1, x2+10, y2], img_for))
            I_point += [I_point_temp]
            I_BG += [I_BG_temp]
            IminusBG += [I_point_temp - I_BG_temp]
        IminusBG_arr = np.stack(IminusBG, axis=1)
        I_point_arr = np.stack(I_point, axis=1)
        I_BG_arr = np.stack(I_BG, axis=1)
        print("I-BG, ", IminusBG_arr, " shape ", IminusBG_arr.shape)



        # num_of_points = IminusBG_arr.shape[1]
        t = np.linspace(0, IminusBG_arr.shape[1], IminusBG_arr.shape[1])



        # for i in range(IminusBG_arr.shape[0]):
        #     plt.plot(t, IminusBG_arr[i], label=str(i))
        # plt.xlabel('frames, x ms')
        # plt.ylabel('I(point) - I(BG)')
        # plt.title("Signal to noise from time")
        # plt.legend()
        # plt.show()
        for i in range(IminusBG_arr.shape[0]):
            IminusBG_arr[i] = processing.smoothing(IminusBG_arr[i], 1)
            I_point_arr[i] = processing.smoothing(I_point_arr[i], 0)
            I_BG_arr[i] = processing.smoothing(I_BG_arr[i],  0)

            fig, ax = plt.subplots()
            ax.plot(t, IminusBG_arr[i], label="I(point) - I(BG) " + str(i+1))
            ax.plot(t, I_point_arr[i], label="Point " + str(i + 1))
            ax.plot(t, I_BG_arr[i], label="BG " + str(i + 1))
            ax.set_xlabel('frames, x ms')
            ax.set_ylabel('I')
            ax.set_title("Signal to noise from time for №" + str(i+1) + " point")
            ax.legend()
        plt.show()

            # for I in IminusBG_arr:
            #     processing.hist(I)


    if key == ord("q"):
        cv2.destroyAllWindows()
        break


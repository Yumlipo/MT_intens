import matplotlib.animation as animation
import matplotlib.pyplot as plt
from nd2reader import ND2Reader

img = ND2Reader('pics/example.nd2')

frames = [] # for storing the generated images
fig = plt.figure()
for i in range(1999):
    frames.append([plt.imshow(img[i], animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)
plt.show()

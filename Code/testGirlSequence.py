import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
    
seq = np.load("data/girlseq.npy")
rect = [280, 152, 330, 318]

rect_arr = np.zeros((seq.shape[2],4))
rect_arr[0] = rect

for i in range(seq.shape[2]-1):
    It, It1 = seq[:,:,i], seq[:,:,i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    rect_arr[i+1] = rect
    x1, y1, x2, y2 = rect

    if (i%20 == 0 or i==1) and (i<81):
        w, h = (x2-x1), (y2-y1)

        plt.figure()
        plt.imshow(It, cmap = 'gray')
        rec = patches.Rectangle((int(rect[0]), int(rect[1])), w, h, fill = False, edgecolor = 'r', linewidth = 2)

        plt.gca().add_patch(rec)
        plt.axis('off')
        plt.title(f'Frame Number {i}')
        plt.show()

np.save('output/girlseqrects.npy', rect_arr)
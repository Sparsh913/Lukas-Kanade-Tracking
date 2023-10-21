import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
import time

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('data/aerialseq.npy')
num_fames = seq.shape[2]
time_req = 0

for i in range(seq.shape[2]-1):
    timer_start = time.time()
    It, It1 = seq[:,:,i], seq[:,:,i+1]
    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)
    movement = np.where(mask == False)
    # print("Shape mask", mask.shape)
    # print("iteration in testAerialseq", i)
    timer_end = time.time()
    time_req += timer_end - timer_start

    if ((i == 1) or (i == 29) or (i == 59) or (i == 89) or (i ==119)):
        f = plt.figure()
        plt.imshow(It1, cmap = 'gray')
        plt.axis('off')
        plt.title(f'Frame Number {i+1}')
        fig, = plt.plot(movement[1], movement[0], '.', markerfacecolor='blue', markeredgecolor='None')
        # fig.set_markerfacecolor((0, 0, 1, 1))
        # plt.show()
        f.savefig('output/aerialseq' + str(i+1) + '.png', bbox_inches = 'tight')

print("Time required to complete tracking per frame is", num_fames/time_req)

    # if i % 30 == 0:
    #     plt.figure()
    #     plt.imshow(mask, cmap = 'gray')
    #     plt.axis('off')
    #     plt.title(f'Frame Number {i}')
    #     plt.show()

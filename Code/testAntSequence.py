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

seq = np.load('data/antseq.npy')
time_req = 0
num_fames = seq.shape[2]

### Testing Code
# frame1, frame2 = seq[:,:,0], seq[:,:,1]
# mask = SubtractDominantMotion(frame1, frame2, threshold, num_iters, tolerance)
# plt.figure(1)
# plt.imshow(mask, cmap='gray')
# plt.show()

## Formalizing
for i in range(seq.shape[2]-1):
    timer_start = time.time()
    It, It1 = seq[:,:,i], seq[:,:,i+1]
    mask = SubtractDominantMotion(It, It1, threshold, num_iters, tolerance)
    movement = np.where(mask == False)
    # print("Shape mask", mask.shape)
    # print("iteration in testAntseq", i)
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
        f.savefig('output/antseq_invvomp' + str(i+1) + '.png', bbox_inches = 'tight')

print("Time required to complete tracking per frame is", num_fames/time_req)

    #     count = 0
    #     for h in range(mask.shape[0]-1):
    #         for w in range(mask.shape[1]-1):
    #             print("In the loop", count)
    #             if mask[h,w]:
    #                 print("Entering Scatter")
    #                 plt.scatter(w, h, s = 1, c = 'b', alpha=0.5)
    #             count += 1
    #     plt.show()
    # print("iteration in testAntseq", i)

    # if i % 30 ==0:
    #     fig,ax = plt.subplots(1)
    #     ax.imshow(It1)
    #     C = np.dstack((It1, It1, It1, mask))
    #     ax.imshow(C)
    #     fig, = plt.plot(movement[1], movement[0], '*')
    #     fig.set_markerfacecolor((0, 0, 1, 1))
    #     plt.axis('off')
    #     plt.show()
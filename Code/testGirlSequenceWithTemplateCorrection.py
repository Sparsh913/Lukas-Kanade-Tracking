import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=5, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = args.template_threshold
    
seq = np.load("data/girlseq.npy")
rect = [280, 152, 330, 318]

rect_arr = np.zeros((seq.shape[2],4))
rect_initial = [280, 152, 330, 318]
rect_old = np.load("output/girlseqrects.npy")

width=rect[2]-rect[0]
height=rect[3]-rect[1]
It=seq[:,:,0]

x1, y1, x2, y2 = rect
r,c = It.shape
# print("Shape It", It.shape)
N_row = int(y2 - y1)
N_col = int(x2 - x1)
# print("Shape rect", (N_row, N_col))


rect_arr[0] = rect
p0 = np.zeros(2)

for i in range(seq.shape[2]-1):
    It, It1 = seq[:,:,i], seq[:,:,i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters, p0)
    rect[0] += p[0]
    rect[1] += p[1]
    rect[2] += p[0]
    rect[3] += p[1]

    # p_n = temp_corr(T, It1, rect, threshold, num_iters)
    pn = p + [rect[0] - rect_initial[0],rect[1]-rect_initial[1]]
    # pn = p_prev
    # st()
    pstar = LucasKanade(seq[:,:,0], It1, rect_initial, threshold, num_iters, pn)

    if (np.linalg.norm(pn-pstar) < template_threshold):
        p_n= (pstar - [rect[0] - rect_initial[0],rect[1]-rect_initial[1]])
        rect[0] += p_n[0]
        rect[1] += p_n[1]
        rect[2] += p_n[0]
        rect[3] += p_n[1]
        p0 = np.zeros(2)
    else:
        p0 = p
    rect_arr[i+1] = rect

np.save('output/girlseqrects-wcrt.npy', rect_arr)

frames_to_record = [1, 20, 40, 60, 80]

for frame in frames_to_record:
    fig = plt.figure()
    cap = seq[:,:,frame]
    rect_bc = rect_old[frame,:]
    rect_ct = rect_arr[frame,:]
    plt.imshow(cap, cmap = 'gray')
    plt.axis('off')
    patch1 = patches.Rectangle((rect_bc[0],rect_bc[1]),(rect_bc[2]-rect_bc[0]),(rect_bc[3]-rect_bc[1]),edgecolor = 'b',facecolor='none',linewidth=2)
    patch2 = patches.Rectangle((rect_ct[0],rect_ct[1]),(rect_ct[2]-rect_ct[0]),(rect_ct[3]-rect_ct[1]),edgecolor = 'r',facecolor='none',linewidth=2)
    ax = plt.gca()
    ax.add_patch(patch1)
    ax.add_patch(patch2)
    fig.savefig('girlseq-wcrtframe' + str(frame) + '.png',bbox_inches='tight')
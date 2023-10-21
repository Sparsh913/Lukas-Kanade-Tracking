import numpy as np
from scipy.interpolate import RectBivariateSpline
import ipdb
st = ipdb.set_trace

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
	
    # Put your implementation here
    p = p0
    
    x1, y1, x2, y2 = rect
    r,c = It.shape
    print("Shape It", It.shape)
    r1,c1 = It1.shape
    print("Shape It1", It1.shape)
    N_row = int(y2 - y1)
    N_col = int(x2 - x1)
    print("Shape rect", (N_row, N_col))
    if (N_row == 0) or (N_col == 0):
        st()

    r_arr = np.linspace(0,r, r, endpoint=False)
    c_arr = np.linspace(0,c,c, endpoint=False)
    r1_arr = np.linspace(0,r1, r1, endpoint=False)
    c1_arr = np.linspace(0,c1,c1, endpoint=False)

    # Note that spline.ev returns the interploated value at input point (x,y). 
    # To get the image intensity value at each and every pixel, we need a total of r*c points.
    # Note that r_arr, c_arr contains all the possible y, x respectively. But how to obtain all the pairwise combinations of (x,y)??
    # Meshgrid to the rescue!  
    rr,cc = np.meshgrid(np.linspace(y1,y2, N_row), np.linspace(x1,x2, N_col))

    # Now if we input the array received from meshgrid above in the spline.ev(), it will calculate --
    # the interploated intensity value at every possible (x,y) in the meshgrid input range! 
    splinet = RectBivariateSpline(r_arr, c_arr, It)
    splinet1 = RectBivariateSpline(r1_arr, c1_arr, It1)
    T = splinet.ev(rr,cc)
    print("Shape of T", T.shape)
    J = np.array([[1,0], [0,1]]) # Jacobian

    I_y, I_x = np.gradient(It1)
    print("Shape I_y", I_y.shape)
    spline_x = RectBivariateSpline(r_arr, c_arr, I_x)
    # print("Shape spline_x", spline_x.shape)
    spline_y = RectBivariateSpline(r_arr, c_arr, I_y)

    dp = np.array([[1],[1]])
    iter = 1

    while(np.sum(np.square(dp)) > threshold and iter < num_iters):
        x1_dp = x1 + p0[0]
        y1_dp = y1 + p0[1]
        x2_dp = x2 + p0[0]
        y2_dp = y2 + p0[1]

        # Update the meshgrid
        rr_up, cc_up = np.meshgrid(np.linspace(y1_dp, y2_dp, N_row), np.linspace(x1_dp, x2_dp, N_col))
        warp = splinet1.ev(rr_up, cc_up)
        print("Shape warp", warp.shape)

        # Now we can calculate the error image: T(x) - I(W(x;p))
        err_im = T - warp
        print("Shape Error Image", err_im.shape)

        err_im = err_im.reshape(-1,1)
        print("Shape err_im after reshape", err_im.shape)
        grad_I_x = spline_x.ev(rr_up, cc_up)
        grad_I_y = spline_y.ev(rr_up, cc_up)
        print("Shape grad_I_x", grad_I_x.shape)

        # grad_I = (I_x, I_y)
        grad_I = np.hstack((grad_I_x.reshape(-1,1), grad_I_y.reshape(-1,1)))
        print("Shape grad_I_x after reshape", grad_I_x.reshape(-1,1).shape)
        print("Shape grad_I", grad_I.shape)

        # grad_I * Jacobian
        grad_I_J = grad_I @ J
        print("Shape grad_I_J", grad_I_J.shape)

        # Hessian
        H = grad_I_J.T @ grad_I_J
        print("Shape H", H.shape)

        # Solution, calculate dp
        dp = np.linalg.inv(H) @ grad_I_J.T @ err_im
        print("Shape dp", dp.shape)

        # Update p
        p0[0] += dp[0]
        p0[1] += dp[1]

        # Loop Counter increment
        iter += 1

    return p0

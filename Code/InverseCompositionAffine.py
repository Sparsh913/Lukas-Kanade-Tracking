import numpy as np
from scipy.interpolate import RectBivariateSpline
import ipdb

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    # put your implementation here
    # M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p  = np.zeros(6)
    dp = np.ones(6)

    M = np.eye(3)

    r, c = It.shape
    r1, c1 = It1.shape

    r_arr = np.linspace(0,r, r, endpoint=False)
    c_arr = np.linspace(0,c,c, endpoint=False)
    r1_arr = np.linspace(0,r1, r1, endpoint=False)
    c1_arr = np.linspace(0,c1,c1, endpoint=False)

    rr,cc = np.meshgrid(r_arr, c_arr)
    rr1,cc1 = np.meshgrid(r1_arr, c1_arr)

    splinet = RectBivariateSpline(r_arr, c_arr, It)
    splinet1 = RectBivariateSpline(r1_arr, c1_arr, It1)

    I_y, I_x = np.gradient(It)  # It or It1 ??
    spline_x = RectBivariateSpline(r_arr, c_arr, I_x)
    # print("Shape spline_x", spline_x.shape)
    spline_y = RectBivariateSpline(r_arr, c_arr, I_y)

    y_it = rr.reshape(1,-1)
    x_it = cc.reshape(1,-1)
    hom_c1 = np.vstack((x_it, y_it, np.ones((1, r*c))))

    grad_I_x = spline_x.ev(rr, cc).flatten()
    grad_I_y = spline_y.ev(rr, cc).flatten()

    T = splinet.ev(rr, cc).flatten()

    # Pre-computing gradient of I times Jacobian matrix, as both are constant
    IJ1 = np.multiply(grad_I_x, x_it)
    IJ2 = np.multiply(grad_I_x, y_it)
    IJ3 = grad_I_x.reshape(1,-1)
    IJ4 = np.multiply(grad_I_y, x_it)
    IJ5 = np.multiply(grad_I_y, y_it)
    IJ6 = grad_I_y.reshape(1,-1)

    IJ = np.vstack((IJ1,IJ2,IJ3,IJ4,IJ5,IJ6)) 
    IJ = IJ.T
    # print("Shape IJ", IJ.shape)

    # Pre-computing Hessian, as it's constant
    H = IJ.T@IJ
    # print("Shape H", H.shape)
    iter = 0

    while(np.linalg.norm(dp) > threshold and iter < num_iters):
        M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [0,0,1]])
        warp1 = M @ hom_c1
        warp_x = warp1[0]
        warp_y = warp1[1]
        # print("Shape warp_x", warp_x.shape)

        warp_im = splinet1.ev(warp_y,warp_x).flatten()
        # print("Shape warp_im", warp_im.shape)

        err_im = (T-warp_im).reshape(len(warp_x),1)
        # print("Shape err_im", err_im.shape)

        dp = np.linalg.inv(H) @ IJ.T @ err_im
        # print("Shape dp", dp.shape)

        p = (p + dp.T).ravel()
        # print("Shape p", p.shape)

        dM = np.vstack((dp.reshape(2,3), [0,0,1]))
        M = M @ np.linalg.inv(dM)

        # print("Iteration", iter)
        iter += 1
        # print(np.linalg.norm(dp))

    
    return M

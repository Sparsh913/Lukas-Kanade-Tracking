import numpy as np
from scipy.interpolate import RectBivariateSpline
import ipdb
st = ipdb.set_trace

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.zeros(6)
    dp=np.ones(6)
    r,c = It.shape
    # print("Shape It", It.shape)
    r1,c1 = It1.shape
    # print("Shape It1", It1.shape)

    x1, y1, x2, y2 = 0, 0, It.shape[1], It.shape[0]
    N_row = int(y2 - y1)
    N_col = int(x2 - x1)

    r_arr = np.linspace(0,r, r, endpoint=False)
    c_arr = np.linspace(0,c,c, endpoint=False)
    r1_arr = np.linspace(0,r1, r1, endpoint=False)
    c1_arr = np.linspace(0,c1,c1, endpoint=False)

    rr,cc = np.meshgrid(r_arr, c_arr)
    rr1,cc1 = np.meshgrid(r1_arr, c1_arr)

    splinet = RectBivariateSpline(r_arr, c_arr, It)
    splinet1 = RectBivariateSpline(r1_arr, c1_arr, It1)
    T = splinet.ev(rr,cc)

    I_y, I_x = np.gradient(It1)
    spline_x = RectBivariateSpline(r1_arr, c1_arr, I_x)
    # print("Shape spline_x", spline_x.shape)
    spline_y = RectBivariateSpline(r1_arr, c1_arr, I_y)

    M = np.eye(3)

    #Converting cartesian coordinates from It to homogeneous
    y_it = rr.reshape(1,-1)
    x_it = cc.reshape(1,-1)
    y_it1 = rr1.reshape(1,-1)
    x_it2 = cc1.reshape(1,-1) 
    hom_c1  =  np.vstack((x_it, y_it, np.ones((1, r*c))))
    hom_c2  =  np.vstack((x_it2, y_it1, np.ones((1, r1*c1))))
    # def common_im(x, y, c, r):
    #     g = np.logical_and(np.logical_and(x >= 0, x<= c-1), np.logical_and(y >= 0, y <= r-1))
    #     return g.nonzero()[0]

    iter = 0
    while(np.linalg.norm(dp) > threshold and iter < num_iters):
        M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [0,0,1]])

        warp1 = M @ hom_c1
        # warp2 = M @ hom_c2
        # print("Shape warp1", warp1.shape)
        

        ## Test code 1 ######

        # # Jacobian elements
        # # dW_dp_x = spline_x.ev(warp_y, warp_x).flatten()
        # # dW_dp_y = spline_y.ev(warp_y, warp_x).flatten()

        # # warp_final = splinet1.ev(warp_y,warp_x).flatten()
        # # T = splinet.ev(y, x).flatten()

        # x_del = (np.where(warp1[0] >= c) or np.where(warp1[0] < 0))
        # # # st()
        # y_del = (np.where(warp1[1] >= r) or np.where(warp1[1] < 0))

        # if ((np.shape(x_del)[1] == 0) and (np.shape(y_del)[1] == 0)):
        #     rem = []
        # elif ((np.shape(x_del)[1] != 0) and (np.shape(y_del)[1] == 0)):
        #     rem = x_del
        # elif ((np.shape(x_del)[1] == 0) and (np.shape(y_del)[1] != 0)):
        #     rem = y_del
        # else:
        #     rem = np.unique(np.concatenate((x_del, y_del), 0))
        # print("rem",rem)
        # x_fin = np.delete(x_it, rem)
        # print("Shape x_fin", x_fin.shape)
        # y_fin = np.delete(y_it, rem)
        # warp_x = np.delete(warp_x, rem).reshape(r,c)
        # print("Shape warp_x", warp_x.shape)
        # warp_y = np.delete(warp_y, rem).reshape(r,c)
        # grad_warp_x = splinet1.ev(warp_y, warp_x, dy=1).reshape(r,c)
        # print("Shape grad_warp_x", grad_warp_x.shape)
        # grad_warp_y = splinet1.ev(warp_y, warp_x, dx=1).reshape(r,c)
        # It1_eval = splinet1.ev(warp_y, warp_x)
        # It_eval = splinet.ev(y_fin, x_fin)

        # IJ1 = x_fin @ grad_warp_x
        # IJ2 = y_fin @ grad_warp_x
        # IJ11 = x_fin @ grad_warp_y
        # IJ22 = y_fin @ grad_warp_y
        # IJ = np.hstack((IJ1, IJ2, grad_warp_x, IJ11, IJ22, grad_warp_y))
        # print("Shape IJ", IJ.shape)

        # err_im = (It_eval - It1_eval).flatten()
        # dp = np.linalg.pinv(IJ).dot(err_im)

        # p = (p + dp.T).ravel()
        # iter += 1

        ## Test code 2 ####
        # hom_c1  =  np.vstack((x1_fin, y1_fin, np.ones((1, len(x1_fin)))))
        # hom_c2  =  np.vstack((x_it2, y_it1, np.ones((1, r1*c1))))

        # x1_dp = warp1[0]
        # print("Shape x1_dp", x1_dp.shape)
        # y1_dp = warp1[1]
        # x2_dp = warp2[0]
        # y2_dp = warp2[1]

        # val_c = common_im(warp1[0], warp1[1], c, r)   ## Comment out here
        # print("Shape val_c", val_c.shape)
        # print("val_c elements", val_c[:10])
        # print("Shape x_it", x_it.shape)

        # x1, x2 = np.min(np.unique(x_fin)), np.max(np.unique(x_fin))
        # y1, y2 = np.min(np.unique(y_fin)), np.max(np.unique(y_fin))
        # # x_fin, y_fin = x_it[val_c], y_it[val_c]
        # # print("Shape x_it", x_it.shape)
        # # print("Shape y_it", y_it.shape)
        # x1, x2 = np.min(np.unique(x_fin)), np.max(np.unique(x_fin))
        # y1, y2 = np.min(np.unique(y_fin)), np.max(np.unique(y_fin))

        # x1_dp = M[0,0] * x1 + M[0,1] * y1 + M[0,2]
        # y1_dp = M[1,0] * x1 + M[1,1] * y1 + M[1,2]
        # x2_dp = M[0,0] * x2 + M[0,1] * y2 + M[0,2]
        # y2_dp = M[1,0] * x2 + M[1,1] * y2 + M[1,2]

        # rr_up, cc_up = np.meshgrid(np.linspace(y1_dp, y2_dp, N_row), np.linspace(x1_dp, x2_dp, N_col))
        # warp = splinet1.ev(rr_up, cc_up)
        # print("Shape warp", warp.shape)

        # err_im = T - warp
        # err_im = err_im.reshape(-1,1)
        # print("Shape err_im after reshape", err_im.shape)
        # grad_I_x = spline_x.ev(rr_up, cc_up)
        # grad_I_y = spline_y.ev(rr_up, cc_up)
        # print("Shape grad_I_x", grad_I_x.shape)
        # grad_I = np.hstack((grad_I_x.reshape(-1,1), grad_I_y.reshape(-1,1)))
        # print("Shape grad_I_x after reshape", grad_I_x.reshape(-1,1).shape)
        # print("Shape grad_I", grad_I.shape)

        # I_J = np.zeros((r*c, 6))

        # for i in range(r):
        #     for j in range(c):
        #         I_pix = np.array([grad_I[i*c + j]]).reshape(1,2)
        #         J_pix = np.array([[j, 0, i, 0, 1, 0], [0, j, 0, i, 0, 1]])
        #         I_J[i*c+j] = I_pix @ J_pix

        # H = I_J.T @ I_J

        # dp = np.linalg.inv(H) @ I_J.T @ err_im
        # p[0] += dp[0,0]
        # p[1] += dp[1,0]
        # p[2] += dp[2,0]
        # p[3] += dp[3,0]
        # p[4] += dp[4,0]
        # p[5] += dp[5,0]

        # M = np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [0,0,1]])
        warp_x = warp1[0]
        warp_y = warp1[1]
        # print("Shape warp_x", warp_x.shape)

        grad_I_x = spline_x.ev(warp_y,warp_x).flatten()
        grad_I_y = spline_y.ev(warp_y,warp_x).flatten()
        # print("Shape grad_I_x", grad_I_x.shape)

        warp_im = splinet1.ev(warp_y,warp_x).flatten()
        # print("Shape warp_im", warp_im.shape)
        T = splinet.ev(rr,cc).flatten()
        # print("Shape T", T.shape)

        err_im = (T - warp_im).reshape(len(warp_x),1)
        # print("Shape err_im", err_im.shape)

        # IJ = grad_I x Jacobian
        # IJ1 = grad_I_x @ x_it
        IJ1 = np.multiply(grad_I_x, x_it)
        # IJ2 = grad_I_x @ y_it
        IJ2 = np.multiply(grad_I_x, y_it)
        IJ3 = grad_I_x.reshape(1,-1)
        # IJ4 = grad_I_y @ x_it
        IJ4 = np.multiply(grad_I_y, x_it)
        # IJ5 = grad_I_y @ y_it
        IJ5 = np.multiply(grad_I_y, y_it)
        IJ6 = grad_I_y.reshape(1,-1)
        
        IJ = np.vstack((IJ1,IJ2,IJ3,IJ4,IJ5,IJ6)) 
        IJ = IJ.T
        # print("Shape IJ", IJ.shape)

        # Hessian
        H = IJ.T@IJ
        # print("Shape H", H.shape)
        
        # Applying formula
        dp = np.linalg.inv(H) @ IJ.T @ err_im
        # print("Shape dp", dp.shape)

        p = (p + dp.T).ravel()
        # print("Shape p", p.shape)
        # print("Iteration", iter)
        iter += 1
        # print(np.linalg.norm(dp))
        
    M = np.array([[1+p[0], p[1],p[2]], [p[3],1+p[4],p[5]], [0,0,1]])

    return M

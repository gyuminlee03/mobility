import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot

###############################################################
## plot function
###############################################################
def rot_mat_2d(angle):
    """
    Create 2D rotation matrix from an angle

    Parameters
    ----------
    angle :

    Returns
    -------
    A 2D rotation matrix

    Examples
    --------
    >>> angle_mod(-4.0)


    """
    return Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]

def plot_covariance_ellipse(x, y, cov, chi2=3.0, color="-r", ax=None):
    """
    This function plots an ellipse that represents a covariance matrix. The ellipse is centered at (x, y) and its shape, size and rotation are determined by the covariance matrix.

    Parameters:
    x : (float) The x-coordinate of the center of the ellipse.
    y : (float) The y-coordinate of the center of the ellipse.
    cov : (numpy.ndarray) A 2x2 covariance matrix that determines the shape, size, and rotation of the ellipse.
    chi2 : (float, optional) A scalar value that scales the ellipse size. This value is typically set based on chi-squared distribution quantiles to achieve certain confidence levels (e.g., 3.0 corresponds to ~95% confidence for a 2D Gaussian). Defaults to 3.0.
    color : (str, optional) The color and line style of the ellipse plot, following matplotlib conventions. Defaults to "-r" (a red solid line).
    ax : (matplotlib.axes.Axes, optional) The Axes object to draw the ellipse on. If None (default), a new figure and axes are created.

    Returns:
    None. This function plots the covariance ellipse on the specified axes.
    """
    eig_val, eig_vec = np.linalg.eig(cov)

    if eig_val[0] >= eig_val[1]:
        big_ind = 0
        small_ind = 1
    else:
        big_ind = 1
        small_ind = 0
    a = math.sqrt(chi2 * eig_val[big_ind])
    b = math.sqrt(chi2 * eig_val[small_ind])
    angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
    plot_ellipse(x, y, a, b, angle, color=color, ax=ax)


def plot_ellipse(x, y, a, b, angle, color="-r", ax=None, **kwargs):
    """
    This function plots an ellipse based on the given parameters.

    Parameters
    ----------
    x : (float) The x-coordinate of the center of the ellipse.
    y : (float) The y-coordinate of the center of the ellipse.
    a : (float) The length of the semi-major axis of the ellipse.
    b : (float) The length of the semi-minor axis of the ellipse.
    angle : (float) The rotation angle of the ellipse, in radians.
    color : (str, optional) The color and line style of the ellipse plot, following matplotlib conventions. Defaults to "-r" (a red solid line).
    ax : (matplotlib.axes.Axes, optional) The Axes object to draw the ellipse on. If None (default), a new figure and axes are created.
    **kwargs: Additional keyword arguments to pass to plt.plot or ax.plot.

    Returns
    ---------
    None. This function plots the ellipse based on the specified parameters.
    """

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    px = [a * math.cos(it) for it in t]
    py = [b * math.sin(it) for it in t]
    fx = rot_mat_2d(angle) @ (np.array([px, py]))
    px = np.array(fx[0, :] + x).flatten()
    py = np.array(fx[1, :] + y).flatten()
    if ax is None:
        plt.plot(px, py, color, **kwargs)
    else:
        ax.plot(px, py, color, **kwargs)



###############################################################
## EKF function and parameters
###############################################################

# Covariance for EKF simulation
Q = np.diag([ # motion model
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance

# Observation x,y position covariance
R = np.diag([1.0, 1.0]) ** 2 # motion model




#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 50.0  # simulation time [s]

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u) # my motion model

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1) # GPS data + noise mix

    # add noise to input
    ud = u + INPUT_NOISE @ np.random.randn(2, 1) # real action data

    xd = motion_model(xd, ud) # motion model input, so we can do yeahchuk

    return xTrue, z, xd, ud # xd \ black line / ud = noise mixed motion data


def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([ # x, y result 
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """



    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -DT * v * math.sin(yaw), DT * math.cos(yaw)],
        [0.0, 1.0, DT * v * math.cos(yaw), DT * math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h(): # motion model's jacobian ("G" part)
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    
    # to-do ###########################################
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst,u)
    PPred = jF @ PEst @ jF.T + Q

    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst
    


    return xEst, PEst


###############################################################
## main
###############################################################
def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xEst = np.array([[3],[3],[0],[0]])
    print(xEst.shape)
    xTrue = np.zeros((4, 1)) # tlfwprkqt blue line
    PEst = np.eye(4) # gong bunsan

    xDR = np.zeros((4, 1))  # Dead reckoning # black line

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    while SIM_TIME >= time: # time out!  
        time += DT 
        u = calc_input() # input value y, yawrate (go, rotate vel -> fixed!)

        xTrue, z, xDR, ud = observation(xTrue, xDR, u) # just filter observation 

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud) # ekf filter-> input = chogi , noisy data ... -> more junhwak black line can make

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r")
            plot_covariance_ellipse(xEst[0, 0], xEst[1, 0], PEst)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == '__main__':
    main()

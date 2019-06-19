"""
Simulate a CTR following a 3D trajectory

Author: Izzat Kamarudzaman

Adapted from code by Python Robotics, Daniel Ingram (daniel-s-ingram)
"""

from math import cos, sin
import numpy as np
import sys
sys.path.append("../")
from mpl_toolkits.mplot3d import Axes3D
# from ConcentricTubeRobot.CTR_model import moving_CTR
from test_model import moving_CTR
from TrajectoryGenerator import TrajectoryGenerator
from controller import Jacobian


show_animation = True

def CTR_sim(x_c, y_c, z_c, total_time):
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c.
    """
    x_pos = -5
    y_pos = -5
    z_pos = 5
    x_vel = 0
    y_vel = 0
    z_vel = 0
    x_acc = 0
    y_acc = 0
    z_acc = 0

    dt = 0.1
    t = 0

    uz_0 = np.array([[0, 0, 0]]).transpose()
    model = lambda q, uz_0: moving_CTR(q, uz_0)

    i = 0
    n_run = 8
    irun = 0

    while True:
        while t <= total_time:
            des_x_pos = calculate_position(x_c[i], t)
            des_y_pos = calculate_position(y_c[i], t)
            des_z_pos = calculate_position(z_c[i], t)
            des_x_vel = calculate_velocity(x_c[i], t)
            des_y_vel = calculate_velocity(y_c[i], t)
            des_z_vel = calculate_velocity(z_c[i], t)
            des_x_acc = calculate_acceleration(x_c[i], t)
            des_y_acc = calculate_acceleration(y_c[i], t)
            des_z_acc = calculate_acceleration(z_c[i], t)

            # print(des_x_pos, des_y_pos, des_z_pos)


            # q = np.array([0, 0, 0, 0, np.pi, 0])  # inputs q
            # delta_q = np.ones(6) * 1e-3
            # x = np.zeros(3)

            # r_jac = Jacobian(delta_q, x, q, uz_0, model)
            # r_jac.jac_approx()
            # J = r_jac.J
            # J_inv = r_jac.p_inv()

            # print('J:\n', J)
            # print('\nJ_inv:\n', J_inv)
            # print('\na * a+ * a == a   -> ', np.allclose(J, np.dot(J, np.dot(J_inv, J))))
            # print('\na+ * a * a+ == a+ -> ', np.allclose(J_inv, np.dot(J_inv, np.dot(J, J_inv))))

            
            # # q = J_inv * 

            x_acc = acc[0]
            y_acc = acc[1]
            z_acc = acc[2]
            x_vel += x_acc * dt
            y_vel += y_acc * dt
            z_vel += z_acc * dt
            x_pos += x_vel * dt
            y_pos += y_vel * dt
            z_pos += z_vel * dt

            q.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw)

            t += dt

        t = 0
        i = (i + 1) % 4
        irun += 1
        if irun >= n_run:
            break

    print("Done")


def calculate_position(c, t):
    """
    Calculates a position given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial 
            trajectory generator.
        t: Time at which to calculate the position

    Returns
        Position
    """
    return c[0] * t**5 + c[1] * t**4 + c[2] * t**3 + c[3] * t**2 + c[4] * t + c[5]


def calculate_velocity(c, t):
    """
    Calculates a velocity given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial 
            trajectory generator.
        t: Time at which to calculate the velocity

    Returns
        Velocity
    """
    return 5 * c[0] * t**4 + 4 * c[1] * t**3 + 3 * c[2] * t**2 + 2 * c[3] * t + c[4]


def calculate_acceleration(c, t):
    """
    Calculates an acceleration given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial 
            trajectory generator.
        t: Time at which to calculate the acceleration

    Returns
        Acceleration
    """
    return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]


def main():
    """
    Calculates the x, y, z coefficients for the four segments 
    of the trajectory
    """
    x_coeffs = [[], []]
    y_coeffs = [[], []]
    z_coeffs = [[], []]
    #  all B (0 -> 0), all alpha (0 -> 2pi/3) 
    waypoints = [[0.042224, -0.042224, 0.269772], [0.015457, 0.057684, 0.269762]]
    total_time = 5

    for i in range(len(waypoints)):
        traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % len(waypoints)], total_time)
        traj.solve()
        x_coeffs[i] = traj.x_c
        y_coeffs[i] = traj.y_c
        z_coeffs[i] = traj.z_c

    print('x_coeffs:\n', x_coeffs)
    print('y_coeffs:\n', y_coeffs)
    print('z_coeffs:\n', z_coeffs)

    # CTR_sim(x_coeffs, y_coeffs, z_coeffs, total_time)


if __name__ == "__main__":
    main()

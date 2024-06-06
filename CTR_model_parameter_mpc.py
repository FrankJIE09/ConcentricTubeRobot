'''
    Author: Izzat Kamarudzaman
    Adaptor: Frank
    Python Version: 3.7.2
    Adapted from Matlab code by Mohsen Khadem

    Model for a three-tubed concentric tube continuum robot class.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
from scipy.optimize import minimize


class CTRobotModel(object):
    def __init__(self, no_of_tubes, tubes_length, curve_length, initial_q,
                 E, J, I, G, Ux, Uy):
        self.n = no_of_tubes
        self.tubes_length = np.array(tubes_length)  # length of tubes
        self.curve_length = np.array(curve_length)  # length of the curved part of tubes
        self.q_0 = np.array(initial_q)  # [BBBaaa]
        self.x_d = np.array([-0.1, -0.03190973, 0.3805521])
        # print(np.array([px, py, pz]))
        # self.accuracy = Tol

        # physical parameters
        self.E = np.array(E)  # E stiffness
        self.J = np.array(J)  # J second moment of inertia
        self.I = np.array(I)  # I inertia
        self.G = np.array(G)  # G torsion constant

        self.Ux = np.array(Ux)  # constant U curvature vectors for each tubes
        self.Uy = np.array(Uy)
        self.init_U = self.Ux
        self.r = np.zeros(3)

    ## main ode solver
    def moving_CTR(self, q):  # q[0:3] 伸长量  q[3:6]绕z轴转的量
        # self.Ux = np.array(Ux)  # constant U curvature vectors for each tubes
        # self.Uy = np.array(Uy)
        q = np.array(q)
        uz_0 = np.zeros(3)
        self.Ux = q[6:9] * np.cos(q[3:6])
        self.Uy = q[6:9] * np.sin(q[3:6])
        # q1 to q3 are robot base movments, q3 to q6 are robot base rotation angles.
        uz0 = uz_0.copy()  # TODO: uz_0 column check
        B = q[:self.n] + self.q_0[:self.n]  # length of tubes before template

        # initial angles
        alpha = (q[3:6] + self.q_0[-self.n:]) - B * uz0  # .transpose()  TODO????
        # alpha = alpha.flatten()  # vectorise. check again  TODO
        alpha_1 = alpha[0].copy()

        # segmenting tubes. Check all inputs must have n elements, n is number of tubes
        (L, d_tip, EE, UUx, UUy) = self.segmenting(B)  # U是初始长度
        # L中的长度分别是tube1,2,3起点的位置，预弯折的位置，末端位置及出口位置（即0）位置最小到最大排序后的差，最小值被省去
        SS = L.copy()
        for i in np.arange(len(L)):
            SS[i] = np.sum(L[:i + 1])  # Total length to each segments
        # 则SS里的分别是tube1,2,3起点的位置，预弯折的位置，末端位置及出口位置（即0）从最小位置到他们位置的差（最小到最大排序），最小值被省去
        #     plot((B(1)+SS(i))*ones(1,10),1:10,'b' ,'LineWidth',2)

        # S is segmented abssica of tube after template (removes -'s after translations)
        # S中的长度都是大于0的但的长度
        S = SS[SS + np.min(B) > 0] + np.min(B)
        E = np.zeros((self.n, len(S)))
        Ux = np.zeros((self.n, len(S)))
        Uy = np.zeros((self.n, len(S)))
        for i in np.arange(self.n):  # each (i,j) element of above matrices correspond to the jth segment of
            E[i, :] = EE[i, SS + np.min(B) > 0]  # ith tube, 1st tube is the most inner
            Ux[i, :] = UUx[i, SS + np.min(B) > 0]
            Uy[i, :] = UUy[i, SS + np.min(B) > 0]

        ## Vectors of tube abssica starting at zero
        span = np.hstack((0, S))
        Length = np.array([], dtype=np.int64).reshape(0, 1)
        r = np.array([], dtype=np.int64).reshape(0, 3)
        U_z = np.array([], dtype=np.int64).reshape(0, 3)  # solved length, curvatures, and twist angles

        # Boundary Conditions # (2)
        # U1_after=[0;0;0];                              # 1st tube initial curvature at segment beginning
        r0 = np.array([[0, 0, 0]]).transpose()
        R0 = np.array([[np.cos(alpha_1), np.sin(alpha_1), 0],
                       [-np.sin(alpha_1), np.cos(alpha_1), 0],
                       [0, 0, 1]])
        R0 = R0.reshape(9, 1, order='F')  # fortran scan order  # TODO: simplify
        # alpha=alpha-B.*uz_0'
        ## Solving ode for shape
        for seg in np.arange(len(S)):
            s_span = [span[seg], span[seg + 1] - 0.0000001]  # TODO: how was the timestep chosen?
            y0_1 = np.vstack([r0, R0])

            y0_2 = np.zeros((2 * self.n, 1))
            y0_2[self.n:2 * self.n] = np.reshape(alpha.copy(), (self.n, 1))
            y0_2[0:self.n] = np.reshape(uz0.copy(), (self.n, 1))

            y_0 = np.vstack([y0_2, y0_1]).flatten()  # shape: (18,) [u, alpha, r, R]

            EI = E[:, seg] * self.I.transpose()
            GJ = self.G * self.J
            ode_sols = solve_ivp(lambda s, y: self.ode(s, y, Ux[:, seg], Uy[:, seg], EI, GJ, self.n), s_span, y_0,
                                 method='RK23')
            s = ode_sols.t[:, np.newaxis]
            y = ode_sols.y.transpose()

            # first n elements of y are curvatures along z, e.g., y= [ u1_z  u2_z ... ]
            # last n elements of y are twist angles, alpha_i
            shape = np.array([y[:, 2 * self.n], y[:, 2 * self.n + 1], y[:, 2 * self.n + 2]]).transpose()  # r

            Length = np.vstack([Length, s])  # stack for every segments
            r = np.vstack([r, shape])
            U_z = np.vstack([U_z, y[:, 0:self.n]])

            r0 = shape[-1][:, np.newaxis]  # TODO: check relation to next segment
            R0 = y[-1, 2 * self.n + 3:2 * self.n + 12][:, np.newaxis]
            uz0 = U_z.copy()[-1]

        Uz = np.zeros((self.n, 1))
        for i in np.arange(self.n):
            index = np.argmin(np.abs(Length - d_tip[i] + 0.0001))  # get tube end position
            Uz[i] = U_z[index, i]  # .copy()?

        r1 = r.copy()
        tube2_end = np.argmin(np.abs(Length - d_tip[1]))
        r2 = np.array([r[0:tube2_end, 0], r[0:tube2_end, 1], r[0:tube2_end, 2]]).transpose()
        tube3_end = np.argmin(np.abs(Length - d_tip[2]))
        r3 = np.array([r[0:tube3_end, 0], r[0:tube3_end, 1], r[0:tube3_end, 2]]).transpose()
        self.r = r1
        return r1, r2, r3, Uz

    def ode(self, s, y, Ux, Uy, EI, GJ, n):  # dydt s>~

        dydt = np.zeros(2 * n + 12)
        # first n elements of y are curvatures along z, e.g., y= [ u1_z  u2_z ... ]  曲率描述了管道轴线在空间中的弯曲程度，通常通过曲率半径来表示。它告诉你管道轴线在某一点处的弯曲情况。
        # second n elements of y are twist angles, alpha_i
        # last 12 elements are r (position) and R (orientations), respectively
        # calculating 1st tube's curvatures in x and y direction
        ux = np.zeros(n)
        uy = np.zeros(n)

        # calculating tube's curvatures in x and y direction
        for i in np.arange(n):  # alpha to curvature                            # 1(c)
            ux[i] = (1 / (EI[0] + EI[1] + EI[2])) * (
                    EI[0] * Ux[0] * np.cos(y[n + i] - y[n + 0]) + EI[0] * Uy[0] * np.sin(y[n + i] - y[n + 0]) +
                    EI[1] * Ux[1] * np.cos(y[n + i] - y[n + 1]) + EI[1] * Uy[1] * np.sin(y[n + i] - y[n + 1]) +
                    EI[2] * Ux[2] * np.cos(y[n + i] - y[n + 2]) + EI[2] * Uy[2] * np.sin(y[n + i] - y[n + 2])
            )

            uy[i] = (1 / (EI[0] + EI[1] + EI[2])) * (
                    -EI[0] * Ux[0] * np.sin(y[n + i] - y[n + 0]) + EI[0] * Uy[0] * np.cos(y[n + i] - y[n + 0]) +
                    -EI[1] * Ux[1] * np.sin(y[n + i] - y[n + 1]) + EI[1] * Uy[1] * np.cos(y[n + i] - y[n + 1]) +
                    -EI[2] * Ux[2] * np.sin(y[n + i] - y[n + 2]) + EI[2] * Uy[2] * np.cos(y[n + i] - y[n + 2])
            )

        # odes for twist
        for i in np.arange(n):
            dydt[i] = ((EI[i]) / (GJ[i])) * (ux[i] * Uy[i] - uy[i] * Ux[i])  # ui_z  公式22d
            dydt[n + i] = y[i]  # 1(e)

        e3 = np.array([[0, 0, 1]]).transpose()
        uz = y[0:n]

        # y(1) to y(3) are position of point materials
        # r1=[y(1) y(2) y(3)]
        # y(4) to y(12) are rotation matrix elements
        R1 = np.array([[y[2 * n + 3], y[2 * n + 4], y[2 * n + 5]],
                       [y[2 * n + 6], y[2 * n + 7], y[2 * n + 8]],
                       [y[2 * n + 9], y[2 * n + 10], y[2 * n + 11]]])

        u_hat = np.array([[0, -uz[0], uy[0]],
                          [uz[0], 0, -ux[0]],
                          [-uy[0], ux[0], 0]])

        # odes
        dr1 = R1 @ e3  # 1(a)  22a
        dR1 = R1 @ u_hat.astype(float)  # 1(b)  22b

        dydt[2 * n + 0] = dr1[0]
        dydt[2 * n + 1] = dr1[1]
        dydt[2 * n + 2] = dr1[2]  # r  6-8

        dR = dR1.flatten()
        for i in np.arange(3, 12):  # R  9-17
            dydt[2 * n + i] = dR[i - 3]

        return dydt

    ## code for segmenting tubes
    def segmenting(self, B):  # -> [L,d1,E,Ux,Uy,I,G,J]

        # all vectors must be sorted, starting element belongs to the most inner tube
        # l vector of tube length
        # B  vector of tube movments with respect to template position, i.e., s=0 (always negative)
        # l_k vector of tube's curved part length
        d1 = self.tubes_length + B  # position of tip of the tubes
        d2 = d1 - self.curve_length  # position of the point where tube bending starts
        points = np.hstack((0, B, d2, d1))
        index = np.argsort(points)  # [L, index] = sort(points)
        L = points[index]
        L = 1e-5 * np.floor(1e5 * np.diff(L))  # length of each segment
        # (used floor because diff command doesn't give absolute zero sometimes)
        # for i=1:k-1
        # if B(i)>B(i+1)
        #     sprintf('inner tube is clashing into outer tubes')
        #     E=zeros(k,length(L))
        #     I=E G=E J=E Ux=E Uy=E

        EE = np.zeros((self.n, len(L)))
        II = np.zeros((self.n, len(L)))
        GG = np.zeros((self.n, len(L)))
        JJ = np.zeros((self.n, len(L)))
        UUx = np.zeros((self.n, len(L)))
        UUy = np.zeros((self.n, len(L)))

        for i in np.arange(self.n):  # 1:3
            a = np.argmin(np.abs(index - i + 1))  # find where tube begins  # find "i+1" by making it "0"
            b = np.argmin(np.abs(index - (1 * self.n + i + 1)))  # find where tube curve starts
            c = np.argmin(np.abs(index - (2 * self.n + i + 1)))  # find where tube ends
            if a == 9:
                return
            if L[a] == 0:
                a = a + 1
            if L[b] == 0:
                b = b + 1
            if c < len(L):  # <= matlab
                if L[c] == 0:
                    c = c + 1
            EE[i, a:c] = self.E[i]
            UUx[i, b:c] = self.Ux[i]
            UUy[i, b:c] = self.Uy[i]

        l = L[np.nonzero(L)]  # ~(L==0)]  # get rid of zero lengthes
        E = np.zeros((self.n, len(l)))
        Ux = np.zeros((self.n, len(l)))
        Uy = np.zeros((self.n,
                       len(l)))  # length https://stackoverflow.com/questions/30599101/translating-mathematical-functions-from-matlab-to-python
        for i in np.arange(self.n):  # remove L==0 column
            E[i, :] = EE[i, ~(L == 0)]
            Ux[i, :] = UUx[i, ~(L == 0)]
            Uy[i, :] = UUy[i, ~(L == 0)]
        L = L[np.nonzero(L)]  # (~(L==0))

        return L, d1, E, Ux, Uy  # L,d1,E,Ux,Uy,I,G,J

    # 解决MPC问题
    # 估算成本函数的雅可比矩阵
    def cost(self, q):
        global px, py, pz
        self.moving_CTR(q, )
        Error = 1000 * (self.r[-1, :].reshape(3, 1) - np.array([px, py, pz]).reshape(3, 1))
        print(px, py, pz)
        # 1000 * (self.r[-1, :].reshape(3, 1) - self.x_d.reshape(3, 1))
        C = np.sum((Error.T @ Error), axis=0)
        return C

    def jac(self, q):
        self.eps = 1e-4
        jac = np.zeros((9,))
        r = self.cost(q)
        for i in range(0, 9):
            q[i] = q[i] + self.eps
            r_perturb = (self.cost(q) - r) / self.eps
            jac[i] = r_perturb.reshape(1, )
            q[i] = q[i] - self.eps
        return jac

    def minimize(self, q_init, q):
               # 限制条件
        ineq_cons = {'type': 'ineq',
                     'jac': lambda q: np.array([[-1.0, 0, 0, 0],
                                                [0, -1.0, 0, 0],
                                                [1.0, 0.0, 0, 0],
                                                [0.0, 1.0, 0, 0],
                                                [-1.0, 1.0, 0, 0]])}
        # r = r1[-1, :].copy()
        # r_d_array = np.array([[r[0] + px, r[1] + py, r[2] + pz]])
        bounds = [
            (-50*1e-3, 50*1e-3),
            (-50*1e-3, 50*1e-3),
            (-50*1e-3, 50*1e-3),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (0, 100),
            (0, 100),
            (0, 100),
        ]
        res = minimize(self.cost, q, method='SLSQP', jac=self.jac,bounds=bounds,
                       options={'ftol': 0.75e-3})
        if res.success:
            return self.moving_CTR(res.x),res.x


def update_plot():
    ax.clear()
    global q
    global q_diff
    global U
    global theta
    global uz_0
    global r
    global initial_q
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_zlabel('Z [mm]')

    # q is defined within the loop below

    # (r1, r2, r3, _) = ctr.moving_CTR(q, uz_0, U * np.cos(theta), U * np.sin(theta))
    # r = r1[-1, :].copy()
    # r_d_array = np.array([[r[0] + px, r[1] + py, r[2] + pz]])
    # ctr.moving_CTR(q, uz_0, U * np.cos(theta), U * np.sin(theta))
    (r1, r2, r3, _),q = ctr.minimize(initial_q, q)
    # q =
    # print(q_diff)
    print("####################")
    # print(r1[-1, :])
    ax.set_box_aspect([1, 1, 1])
    # 设置绘图区间
    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.1, 0.3])
    ax.plot3D(r1[:, 0], r1[:, 1], r1[:, 2], linewidth=1, label='tube1')
    ax.plot3D(r2[:, 0], r2[:, 1], r2[:, 2], linewidth=2, label='tube2')
    ax.plot3D(r3[:, 0], r3[:, 1], r3[:, 2], linewidth=3, label='tube3')
    ax.scatter(r1[-1, 0], r1[-1, 1], r1[-1, 2],
               label='({:03f},{:03f},{:03f})'.format(r1[-1, 0], r1[-1, 1], r1[-1, 2]))
    ax.legend()
    plt.draw()


# 定义步进大小
step_size = 0.01

r = [0, 0, 0]


# 按钮回调函数
def update_px_plus(event):
    global px, py, pz
    # px, py, pz = 0., 0., 0.
    px += step_size
    update_plot()


def update_px_minus(event):
    global px, py, pz
    # px, py, pz = 0., 0., 0.
    px += -step_size
    update_plot()


def update_py_plus(event):
    global px, py, pz
    # px, py, pz = 0., 0., 0.
    py += step_size
    update_plot()


def update_py_minus(event):
    global px, py, pz
    # px, py, pz = 0., 0., 0.
    py += -step_size
    update_plot()


def update_pz_plus(event):
    global px, py, pz
    # px, py, pz = 0., 0., 0.
    pz += step_size
    update_plot()


def update_pz_minus(event):
    global px, py, pz
    # px, py, pz = 0., 0., 0.
    pz += -step_size
    update_plot()


px, py, pz = 0., -0.03190973, 0.3805521

uz_0 = np.array([0, 0, 0], dtype=float)
# # q = np.array([0, -3, -3, 0, 0, 0])  #inputs [BBBaaa]
q = np.array([0, 0, 0, 0, 0, 0, 20, 15, 3], dtype="float")
# no_of_tubes = 3  # ONLY MADE FOR 3 TUBES for now
initial_q = [-0.950, -0.85, -0.51, 0, 0, 0]
tubes_length = [1350, 1200, 810]
curve_length = [0, 40, 55]
tubes_length = 1e-3 * np.array(tubes_length)  # length of tubes
curve_length = 1e-3 * np.array(curve_length)  # length of the curved part of tubes
U = np.array([20, 15, 3], dtype=float)
theta = np.array([0, 0, np.pi / 6], dtype=float)

# physical parameters
E = np.array([6.4359738368e+10, 5.2548578304e+10, 4.7163091968e+10])  # E stiffness
J = 1.0e-11 * np.array([0.0120, 0.0653, 0.1686])  # J second moment of inertia
I = 1.0e-12 * np.array([0.0601, 0.3267, 0.8432])  # I inertia
G = np.array([2.5091302912e+10, 2.1467424256e+10, 2.9788923392e+10])  # G torsion constant

Ux = np.array([21, 13, 3])  # constant U curvature vectors for each tubes
Uy = np.array([0, 0, 0])

ctr = CTRobotModel(3, tubes_length, curve_length, initial_q, E, J, I, G, Ux, Uy)

# Create initial plot
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')

ax.view_init(elev=30, azim=45)
# Move the 3D plot to the right
pos = ax.get_position()
ax.set_position([pos.x0 - 0.2, pos.y0, pos.width * 1.5, pos.height * 1.5])
slider_width = 0.4
slider_height = 0.02

top = pos.y0 + 0.6
a = 0.03
left = pos.x0 + 0.4

# 创建按钮
button_px_plus = Button(plt.axes([0.6, 0.8, 0.1, 0.1]), '+px')
button_px_plus.on_clicked(update_px_plus)

button_px_minus = Button(plt.axes([0.75, 0.8, 0.1, 0.1]), '-px')
button_px_minus.on_clicked(update_px_minus)

button_py_plus = Button(plt.axes([0.6, 0.65, 0.1, 0.1]), '+py')
button_py_plus.on_clicked(update_py_plus)

button_py_minus = Button(plt.axes([0.75, 0.65, 0.1, 0.1]), '-py')
button_py_minus.on_clicked(update_py_minus)

button_pz_plus = Button(plt.axes([0.6, 0.5, 0.1, 0.1]), '+pz')
button_pz_plus.on_clicked(update_pz_plus)

button_pz_minus = Button(plt.axes([0.75, 0.5, 0.1, 0.1]), '-pz')
button_pz_minus.on_clicked(update_pz_minus)

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy import linalg
import logging


class ThermoSimulation:
    """
    Thermal simulation model of the contact between two bodies, using finite difference.
    See publication for more information.
    """

    def __init__(self):
        """
        Initializing the model
        """

        self.theta_start = None
        self.x_b = None
        self.x_w = None
        self.D_w = None
        self.D_b = None
        self.i_snap = None
        self.temp_list = None
        self.time_list = None
        self.Q_inv = None
        self.A_d = None
        self.C = None

        # Material properties
        self.lambda_w = 50  # w/m/K
        self.lambda_b = 250  # w/m/K

        self.rho_w = 7.72e3  # kg/m^3
        self.rho_b = 2.7e3  # kg/m^3

        self.cp_w = 221  # J/kg/K
        self.cp_b = 896  # J/kg/K

        # solver properties
        self.s = 0.2  # step size in mm
        self.n_b = 50  # number of point in sheet
        self.n_w = 350  # number of point in tool

        self.L_b = self.n_b * self.s  # mm
        self.L_w = self.n_w * self.s  # mm

        self.dt = 1e-6  # s timstep
        self.T = 150  # s Termination time
        self.NPLTC = 25

        self.h_c = 4500 * 1e3  # h_contact w/m^2/K * 1e3

        self.theta_room = 293  # K
        self.theta_blech = 102  # K

        self.i_snap = np.array(np.linspace(0, self.T, num=self.NPLTC, endpoint=True) / self.dt, int)

    def prepare(self):
        """
        prepare the matrix coefficients
        """

        # Compute the thermal diffusivity
        self.D_b = self.lambda_b / (self.rho_b * self.cp_b) * 1e6  # mm^2/s
        self.D_w = self.lambda_w / (self.rho_w * self.cp_w) * 1e6  # mm^2/s

        # Compute the coordinates of the "nodes" in the sheet and the tool (_b and _w respect.)
        self.x_b = np.linspace(-self.L_b, 0, num=self.n_b)
        self.x_w = np.linspace(0, self.L_w, num=self.n_w)

        # Set up the start temperature
        self.theta_start = np.hstack((np.ones(self.n_b) * self.theta_blech, np.ones(self.n_w) * self.theta_room))

        logging.info(self.D_b * self.dt / self.s ** 2, "<< 0.5 ?")
        if self.D_b * self.dt / self.s ** 2 > 0.5:
            raise logging.warn("Condition 1 for numerical stability is violated.")

        logging.info(self.D_w * self.dt / self.s ** 2, "<< 0.5 ?")
        if self.D_w * self.dt / self.s ** 2 > 0.5:
            raise Warning("Condition 2 for numerical stability is violated")

        logging.info(-self.h_c / self.rho_b / self.cp_b * self.dt, "<< 0.5 ?")

        if -self.h_c / self.rho_b / self.cp_b * self.dt > 0.5:
            raise Warning("Condition 3 for numerical stability is violated")

        # Set up the matrix, i this case A spase does not really help, but if B where to be zero it would.
        A = sparse.dok_array((self.n_b + self.n_w, self.n_b + self.n_w), dtype=np.float64)
        B = np.zeros((self.n_b + self.n_w), dtype=np.float64)

        # Filling the matrix.
        for i in range(self.n_b + self.n_w):
            if i == 0:
                # symetry on sheet
                A[i, i] = 1 - 2 * self.D_b * self.dt / self.s ** 2
                A[i, i + 1] = 2 * self.D_b * self.dt / self.s ** 2

            elif i < self.n_b - 1:
                # bulk of sheet
                A[i, i] = 1 - 2 * self.D_b * self.dt / self.s ** 2
                A[i, i - 1] = self.D_b * self.dt / self.s ** 2
                A[i, i + 1] = self.D_b * self.dt / self.s ** 2
            elif i == self.n_b - 1:
                # contact on sheet side
                A[i, i - 1] = 2 * self.D_b * self.dt / self.s ** 2
                A[i, i] = -self.h_c / self.rho_b / self.cp_b * self.dt + 1 - 2 * self.D_b * self.dt / self.s ** 2
                A[i, i + 1] = self.h_c / self.rho_b / self.cp_b * self.dt
            elif i == self.n_b:
                # contact on tool side
                A[i, i - 1] = self.h_c / self.rho_w / self.cp_w * self.dt
                A[i, i] = -self.h_c / self.rho_w / self.cp_w * self.dt + 1 - 2 * self.D_w * self.dt / self.s ** 2
                A[i, i + 1] = 2 * self.D_w * self.dt / self.s ** 2
            elif self.n_b < i < self.n_b + self.n_w - 1:
                # bulk of tool
                A[i, i] = 1 - 2 * self.D_w * self.dt / self.s ** 2
                A[i, i - 1] = self.D_w * self.dt / self.s ** 2
                A[i, i + 1] = self.D_w * self.dt / self.s ** 2
            elif i == self.n_b + self.n_w - 1:
                # constant temperature on boundary
                A[i, i] = 1 - 2 * self.D_w * self.dt / self.s ** 2
                A[i, i - 1] = 2 * self.D_w * self.dt / self.s ** 2
                # A[i, i - 1] = self.D_w * self.dt / self.s ** 2
                # B[i] = -self.theta_room * self.D_w * self.dt / self.s ** 2

        A = sparse.csc_array(A)

        # Finding C for efficient computation of the timesteps.
        self.C = np.linalg.inv(A - np.identity(A.shape[0])) @ B

        # Diagonalization of A to compute efficiently the power of A

        self.A_d, self.Q = linalg.eig(A.toarray())
        self.A_d = np.diag(np.abs(self.A_d))
        self.Q_inv = linalg.inv(self.Q)

    def compute(self):
        """
        Compute the model
        """

        self.time_list = [0, ]
        self.temp_list = [self.theta_start]

        for i in self.i_snap[1:]:
            self.time_list.append(i * self.dt)
            t_act = self.Q @ self.A_d ** i @ self.Q_inv @ (self.theta_start - self.C) + self.C
            self.temp_list.append(t_act)

        self.time_list = np.array(self.time_list)
        self.temp_list = np.array(self.temp_list)

    def plot_temp(self):
        """
        Plot the results
        """
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
        for i in range(len(self.temp_list)):
            ax1.plot(np.hstack((self.x_b, self.x_w)), self.temp_list[i], label="{:.2f}".format(self.time_list[i]))
            ax2.plot(np.hstack((self.x_b, self.x_w)), self.temp_list[i], label="{:.2f}".format(self.time_list[i]),
                     marker="+")

        ax1.set_xlabel("X position in mm")
        ax1.set_ylabel("Temperature in K")

        ax1.legend(ncols=3)

        ax2.set_xlabel("X position in mm")
        ax2.set_ylabel("Temperature in K")

        ax1.set_title("Temperature profile")
        ax2.set_title("Temperature profile (Zoomed and with points)")

        ax2.legend(ncols=3)
        ax2.set_xlim(-1.5, 1.5)

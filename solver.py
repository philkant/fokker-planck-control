import numpy as np
import scipy as sp

from typing import Callable, Union, Optional


class SolverFP:
    def __init__(self, volatility: float, feedback: float, control_max: float, weight:float,
                 intensity_function: Callable):
        self.volatility = volatility
        self.feedback = feedback
        self.control_max = control_max
        self.weight = weight
        self.intensity_function = intensity_function

    def solve(self, mesh_time: np.array, mesh_space: np.array, theta: float, initial_condition: np.array,
              value: Optional[np.array] = None) -> np.array:
        if value is None:
            value = -np.ones((mesh_time.size, mesh_space.size - 2))

        solution = np.zeros((mesh_time.size, mesh_space.size - 2))
        solution[0, :] = initial_condition

        delta_time = mesh_time[1] - mesh_time[0]
        delta_space = mesh_space[1] - mesh_space[0]
        intensity = self.intensity_function(mesh_space[1:-1])

        for j in range(0, mesh_time.size - 1):
            # Compute integral of intensity against current distribution.
            intensity_j = np.sum(intensity*solution[j, :]*delta_space)

            # Compute control.
            gradient = np.zeros(mesh_space.size - 2)
            gradient[1:-1] = (value[j, 2:] - value[j, :-2])/(2*delta_space)
            gradient[0] = (value[j, 1] + 1)/(2*delta_space)
            gradient[-1] = (-1 - value[j, -1])/(2*delta_space)
            control = self.control_max/(1 + np.exp(10*(gradient + self.weight)))

            # Compute system matrix for theta scheme.
            sys_mat = np.zeros((3, mesh_space.size - 2))
            sys_mat[0, 1:] = delta_time*theta*((control[:-1] - self.feedback*intensity_j)/(2*delta_space)
                                               - self.volatility**2/(2*delta_space**2))
            sys_mat[1, :] = (1 + delta_time*theta*(intensity + self.volatility**2/(delta_space**2)))
            sys_mat[2, :-1] = delta_time*theta*((-control[1:] + self.feedback*intensity_j)/(2*delta_space)
                                                - self.volatility**2/(2*delta_space**2))

            # Compute right-hand side of theta scheme.
            rhs = (1 - delta_time*(1 - theta)*(intensity + self.volatility**2/(delta_space**2)))*solution[j, :]
            rhs[:-1] = rhs[:-1] - delta_time*(1 - theta)*((control[:-1] - self.feedback*intensity_j)/(2*delta_space)
                                                          - self.volatility**2/(2*delta_space**2))*solution[j, 1:]
            rhs[1:] = rhs[1:] - delta_time*(1 - theta)*((-control[1:] + self.feedback*intensity_j)/(2*delta_space)
                                                        - self.volatility**2/(2*delta_space**2))*solution[j, :-1]

            # Solve for next time step.
            solution[j + 1, :] = sp.linalg.solve_banded((1, 1), sys_mat, rhs)

        return solution


class SolverHJB:
    def __init__(self, volatility: float, feedback: float, control_max: float, weight:float,
                 intensity_function: Callable):
        self.volatility = volatility
        self.feedback = feedback
        self.control_max = control_max
        self.weight = weight
        self.intensity_function = intensity_function

    def solve(self, mesh_time: np.array, mesh_space: np.array, theta: float,
              terminal_condition: Union[np.array, float] = -1., flow: Optional[np.array] = None) -> np.array:
        if flow is None:
            flow = np.zeros((mesh_time.size, mesh_space.size - 2))

        solution = np.zeros((mesh_time.size, mesh_space.size - 2))

        # Add one to solution to transform into problem with homogeneous Dirichlet boundary conditions.
        solution[-1, :] = terminal_condition + 1.

        delta_time = mesh_time[1] - mesh_time[0]
        delta_space = mesh_space[1] - mesh_space[0]
        intensity = self.intensity_function(mesh_space[1:-1])

        for j_back in range(0, mesh_time.size - 1):
            j = mesh_time.size - j_back - 1

            # Compute integral of intensity against current distribution.
            intensity_j = np.sum(intensity*flow[j, :]*delta_space)

            # Compute system matrix for theta scheme.
            sys_mat = np.zeros((3, mesh_space.size - 2))
            sys_mat[0, 1:] = delta_time*theta*(self.feedback*intensity_j/(2*delta_space)
                                               - self.volatility**2/(2*delta_space**2))
            sys_mat[1, :] = (1 + delta_time*theta*(intensity + self.volatility**2/(delta_space**2)))
            sys_mat[2, :-1] = delta_time*theta*(-self.feedback*intensity_j/(2*delta_space)
                                                - self.volatility**2/(2*delta_space**2))

            # Compute gradient of solution.
            gradient = np.zeros(mesh_space.size - 2)
            gradient[1:-1] = (solution[j, 2:] - solution[j, :-2])/(2*delta_space)
            gradient[0] = solution[j, 1]/(2*delta_space)
            gradient[-1] = -solution[j, -1]/(2*delta_space)

            # Compute nonlinearity and nonlocality.
            nonlinearity = self.control_max*np.maximum(-gradient - self.weight, 0)
            nonlocality = self.feedback*np.sum(gradient*flow[j, :]*delta_space)*intensity

            # Compute right-hand side of theta scheme.
            rhs = ((1 - delta_time*(1 - theta)*(intensity + self.volatility**2/(delta_space**2)))*solution[j, :]
                   - delta_time*(nonlinearity + nonlocality - intensity))
            rhs[:-1] = rhs[:-1] - delta_time*(1 - theta)*(self.feedback*intensity_j/(2*delta_space)
                                                          - self.volatility**2/(2*delta_space**2))*solution[j, 1:]
            rhs[1:] = rhs[1:] - delta_time*(1 - theta)*(-self.feedback*intensity_j/(2*delta_space)
                                                        - self.volatility**2/(2*delta_space**2))*solution[j, :-1]

            # Solve for next time step.
            solution[j - 1, :] = sp.linalg.solve_banded((1, 1), sys_mat, rhs)

        return solution - 1.


class SolverMFC:
    def __init__(self, volatility: float, feedback: float, control_max: float, weight: float,
                 intensity_function: Callable):
        self.control_max = control_max
        self.weight = weight
        self.solver_fp = SolverFP(volatility, feedback, control_max, weight, intensity_function)
        self.solver_hjb = SolverHJB(volatility, feedback, control_max, weight, intensity_function)

    def solve(self, iterations: int, mesh_time: np.array, mesh_space: np.array, theta: float,
              initial_condition: np.array, terminal_condition: Union[np.array, float] = -1.):

        flow = np.zeros((mesh_time.size, mesh_space.size - 2))
        value = -np.ones((mesh_time.size, mesh_space.size - 2))

        error = np.zeros((2, iterations - 1))

        for j in range(0, iterations):
            flow_1 = self.solver_fp.solve(mesh_time, mesh_space, theta, initial_condition, value=value)
            value_1 = self.solver_hjb.solve(mesh_time, mesh_space, theta, terminal_condition, flow=flow)

            if j > 0:
                error[0, j - 1] = np.max(np.abs(flow_1 - flow))
                error[1, j - 1] = np.max(np.abs(value_1 - value))

            flow = flow_1
            value = value_1

            # if j % 50 == 0:
            #     plt.plot(mesh_space[1:-1], flow[-1, :])
            #     plt.show()
            #
            #     plt.plot(mesh_space[1:-1], value[0, :])
            #     plt.show()

        cost = np.zeros(2)

        delta_time = mesh_time[1] - mesh_time[0]
        delta_space = mesh_space[1] - mesh_space[0]
        gradient = np.zeros((mesh_time.size, mesh_space.size - 2))
        gradient[:, 1:-1] = (value[:, 2:] - value[:, :-2])/(2*delta_space)
        gradient[:, 0] = (value[:, 1] + 1)/(2*delta_space)
        gradient[:, -1] = (-1 - value[:, -1])/(2*delta_space)
        cost[0] = np.sum(delta_time*delta_space*self.weight
                         * flow*self.control_max/(1 + np.exp(10*(gradient + self.weight))))
        cost[1] = 1 - np.sum(delta_space*flow[-1, :])

        return flow, value, cost, error

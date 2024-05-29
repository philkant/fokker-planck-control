import numpy as np
import math
from matplotlib import pyplot as plt

import solver


def convert_floats_to_strings(float_list, decimal_places=2):
    string_list = [f"{round(num, decimal_places):.{decimal_places}f}" for num in float_list]
    return string_list


if __name__ == '__main__':

    intensity_parameter = 10.

    def intensity_function(x):
        return intensity_parameter*np.maximum(-x, 0)

    volatility = .1
    feedback = 1.
    control_max = 2.
    weight = 1.
    solver_fp = solver.SolverFP(volatility, feedback, control_max, weight, intensity_function)
    solver_hjb = solver.SolverHJB(volatility, feedback, control_max, weight, intensity_function)
    solver_mfc = solver.SolverMFC(volatility, feedback, control_max, weight, intensity_function)

    timepoints = 500
    mesh_time = np.linspace(0., 1., timepoints)

    spacepoints = 5000
    mesh_space = np.linspace(-5., 5., spacepoints)
    theta = 0.5

    iterations = 40

    def gamma(x: np.array, shape: float, rate: float):
        y = np.maximum(x, 0.)
        return rate**shape*y**(shape - 1)*np.exp(-rate*y)/math.gamma(shape)

    shape = 6.
    rate = 60.

    initial_condition = gamma(mesh_space[1:-1], shape, rate)

    flow, value, cost, error = solver_mfc.solve(iterations, mesh_time, mesh_space, theta, initial_condition)

    idx_time = 50
    delta_time = mesh_time[1] - mesh_time[0]
    delta_space = mesh_space[1] - mesh_space[0]
    gradient = np.zeros((mesh_time.size, mesh_space.size - 2))
    gradient[:, 1:-1] = (value[:, 2:] - value[:, :-2])/(2*delta_space)
    gradient[:, 0] = (value[:, 1] + 1)/(2*delta_space)
    gradient[:, -1] = (-1 - value[:, -1])/(2*delta_space)

    idx = gradient[idx_time, :] <= -weight
    idx_min = np.arange(mesh_space.size - 2)[idx][0]
    idx_max = np.arange(mesh_space.size - 2)[idx][-1]
    x_min = mesh_space[1:-1][idx_min]
    y_min = value[idx_time, idx_min]
    x_max = mesh_space[1:-1][idx_max]
    y_max = value[idx_time, idx_max]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mesh_space[1:-1], value[idx_time, :])

    ax.axvline(x=x_min, ymin=(-1. + 1.05)/(0.25 + 1.05), ymax=(y_min + 1.05)/(0.25 + 1.05), linestyle='dashed',
               color='black', linewidth=.75)
    ax.axvline(x=x_max, ymin=(-1. + 1.05)/(0.25 + 1.05), ymax=(y_max + 1.05)/(0.25 + 1.05), linestyle='dashed',
               color='black', linewidth=.75)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u_t(x)$')
    ax.set_title(r'Adjoint process for $\alpha = ${}'.format(feedback))

    ax.set_xlim(-1., 1.)
    ax.set_ylim(-1.05, 0.25)

    plt.show()

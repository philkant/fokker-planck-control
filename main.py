import numpy as np
import math
from matplotlib import pyplot as plt

import solver


def convert_floats_to_strings(float_list, decimal_places=2):
    string_list = [f"{round(num, decimal_places):.{decimal_places}f}" for num in float_list]
    return string_list


if __name__ == '__main__':

    intensity_parameter = 2.5

    def intensity_function(x):
        return intensity_parameter*np.maximum(-x, 0)

    volatility = 1.
    feedback = 0.
    control_max = 5.
    weight = .25
    solver_fp = solver.SolverFP(volatility, feedback, control_max, weight, intensity_function)
    solver_hjb = solver.SolverHJB(volatility, feedback, control_max, weight, intensity_function)
    solver_mfc = solver.SolverMFC(volatility, feedback, control_max, weight, intensity_function)

    timepoints = 100
    mesh_time = np.linspace(0., 1., timepoints)

    spacepoints = 1000
    mesh_space = np.linspace(-10., 10., spacepoints)
    theta = 0.5

    iterations = 40

    # flow = (mesh_space.size*np.ones((mesh_time.size, mesh_space.size - 2))
    #         / ((mesh_space.size - 2)*(mesh_space[-1] - mesh_space[0])))

    # initial_condition = flow[0, :]
    initial_condition = np.exp(-(mesh_space[1:-1] - 1.)**2/(2*volatility**2))/(np.sqrt(2*math.pi*volatility**2))

    flow, value, cost, error = solver_mfc.solve(iterations, mesh_time, mesh_space, theta, initial_condition)

    delta_time = mesh_time[1] - mesh_time[0]
    delta_space = mesh_space[1] - mesh_space[0]
    gradient = np.zeros((mesh_time.size, mesh_space.size - 2))
    gradient[:, 1:-1] = (value[:, 2:] - value[:, :-2])/(2*delta_space)
    gradient[:, 0] = (value[:, 1] + 1)/(2*delta_space)
    gradient[:, -1] = (-1 - value[:, -1])/(2*delta_space)

    idx = gradient[0, :] <= -weight
    idx_min = np.arange(mesh_space.size - 2)[idx][0]
    idx_max = np.arange(mesh_space.size - 2)[idx][-1]
    x_min = mesh_space[1:-1][idx_min]
    y_min = value[0, idx_min]
    x_max = mesh_space[1:-1][idx_max]
    y_max = value[0, idx_max]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(mesh_space[1:-1], value[0, :])

    # x_ticks = convert_floats_to_strings(list(ax.get_xticks())[1:-1], 1)
    # ax.set_xticks(list(ax.get_xticks())[1:-1] + [x_min] + [x_max])
    # ax.set_xticklabels(x_ticks + [r'$a$'] + [r'$b$'])

    ax.axvline(x=x_min, ymin=(-1. + 1.05)/(0.25 + 1.05), ymax=(y_min + 1.05)/(0.25 + 1.05), linestyle='dashed',
               color='black', linewidth=.75)
    ax.axvline(x=x_max, ymin=(-1. + 1.05)/(0.25 + 1.05), ymax=(y_max + 1.05)/(0.25 + 1.05), linestyle='dashed',
               color='black', linewidth=.75)

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u_t(x)$')
    ax.set_title(r'Adjoint process for $\kappa = ${}'.format(feedback))

    ax.set_xlim(-7., 7.)
    ax.set_ylim(-1.05, 0.25)

    plt.show()

    print(cost)



    # solution = solver_fp.solve(mesh_time, mesh_space, theta, initial_condition, value=None)

    # plt.plot(mesh_space[1:-1], solution[-1, :])
    # plt.plot(mesh_space[1:-1], np.exp(-mesh_space[1:-1]**2/(2*volatility**2))/(np.sqrt(2*math.pi*volatility**2)))
    # plt.show()

    # solution = solver_hjb.solve(mesh_time, mesh_space, theta, flow=flow)
    #
    # plt.plot(mesh_space[1:-1], solution[0, :])
    # plt.show()

    # a = 9
    # b = 21
    # x = np.zeros(a + b)
    # x[0:a] = np.linspace(-5., -0.01, a)
    # x[a:] = np.linspace(0, 5., b)
    # y = -1/(1 + np.exp(-(1.5 - np.minimum(x, 0)/3)*x))
    # spline = sp.interpolate.CubicSpline(x, y)
    #
    # x = np.linspace(-5., 5., 100)
    # y = spline(x)
    #
    # der = (y[1:] - y[:-1])/0.1
    # idx = der < -0.3
    # x_min = x[:-1][idx][0]
    # x_max = x[:-1][idx][-1]
    #
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(x, y)
    #
    # x_ticks = convert_floats_to_strings(list(ax.get_xticks())[1:-1], 1)
    # ax.set_xticks(list(ax.get_xticks())[1:-1] + [x_min] + [x_max])
    # ax.set_xticklabels(x_ticks + [r'$a$'] + [r'$b$'])
    #
    # ax.axvline(x=x_min, ymin=0.03, ymax=0.765, linestyle='dashed', color='black', linewidth=.75)
    # ax.axvline(x=x_max, ymin=0.03, ymax=0.3, linestyle='dashed', color='black', linewidth=.75)
    #
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$u_t(x)$')
    # ax.set_title(r'Adjoint process for $\kappa = 0$')
    #
    # plt.show()
    #
    # a = 6
    # b = 21
    # x = np.zeros(a + b)
    #
    # x[0] = -5.
    # x[1] = -4.
    # x[2] = -3.
    # x[3:a] = np.linspace(-1.5, -0.01, a - 3)
    # x[a:] = np.linspace(0, 5., b)
    # y = np.zeros(a + b)
    # y[0] = 0.045
    # y[1] = 0.06
    # y[2] = 0.09
    # y[3:] = -1.1/(1 + np.exp(-(1.3 - np.minimum(x[3:], 0)/2)*x[3:])) + 0.1
    # spline = sp.interpolate.CubicSpline(x, y)
    #
    # x = np.linspace(-5., 5., 100)
    # y = spline(x)
    #
    # der = (y[1:] - y[:-1])/0.1
    # idx = der < -0.3
    # x_min = x[:-1][idx][0]
    # x_max = x[:-1][idx][-1]
    #
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(x, y)
    #
    # x_ticks = convert_floats_to_strings(list(ax.get_xticks())[1:-1], 1)
    # ax.set_xticks(list(ax.get_xticks())[1:-1] + [x_min] + [x_max])
    # ax.set_xticklabels(x_ticks + [r'$a$'] + [r'$b$'])
    #
    # ax.axvline(x=x_min, ymin=0.03, ymax=0.775, linestyle='dashed', color='black', linewidth=.75)
    # ax.axvline(x=x_max, ymin=0.03, ymax=0.31, linestyle='dashed', color='black', linewidth=.75)
    #
    # ax.set_xlabel(r'$x$')
    # ax.set_ylabel(r'$u_t(x)$')
    # ax.set_title(r'Adjoint process for $\kappa > 0$')
    #
    # plt.show()

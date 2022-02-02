import scipy.linalg
import numpy as np
from pprint import pprint
import matplotlib as mpt
import matplotlib.pyplot as plt

# mpt.use('TkAgg')
# plt.ion()
np.set_printoptions(suppress=True)


def starting_function1(x):
    return -x ** 4


def border_function1(t):
    return t ** 2 - t


def border_function2(t):
    return t ** 2 + t - t * (np.exp(1)) - 1


def f(t, x):
    return (0.017 * 12 * (x ** 2)) + np.exp(x) * (0.017 * t - 1) + x + 2 * t


def exact_function(t, x):
    return -(x ** 4) + t * x + (t ** 2) - t * np.exp(x)


def findMaxDeviation(u, u_exact):
    return np.max(np.abs(u[:][:] - u_exact[:][:]))


def create_A_matrix(x_grid_size, a, tau, h):
    iterMatrix = np.zeros((3, x_grid_size - 2))

    for k in range(x_grid_size - 2):
        for j in range(x_grid_size - 2):
            if abs(k - j) == 1:
                iterMatrix[1 + k - j][j] = -((a * tau) / (2 * h * h))
            elif j == k:
                iterMatrix[1 + k - j][j] = 1 + (a * tau) / (h * h)

    return iterMatrix


def create_B_matrix(x_grid_size, a, tau, h):
    iter_row = np.zeros(x_grid_size - 2)
    iter_row[0] = 1 - ((a * tau) / (h * h))
    iter_row[1] = ((a * tau) / (2 * h * h))
    iter_column = np.zeros(x_grid_size - 2)
    iter_column[0] = 1 - ((a * tau) / (h * h))
    iter_column[1] = ((a * tau) / (2 * h * h))
    return scipy.linalg.toeplitz(iter_column, iter_row)


def solve(tau, h, a):
    x_grid_size = int(1 / h)
    t_grid_size = int(1 / tau)

    x = np.linspace(0, 1, x_grid_size)
    t = np.linspace(0, 1, t_grid_size)


    u = np.zeros((t_grid_size, x_grid_size))

    u[0, :] = starting_function1(x[:])
    u[:, 0] = border_function1(t[:])
    u[:, x_grid_size - 1] = border_function2(t[:])

    A = create_A_matrix(x_grid_size, a, tau, h)
    B = create_B_matrix(x_grid_size, a, tau, h)

    b = np.zeros(x_grid_size - 2)

    alpha = (tau * a) / (h * h)

    for j in range(1, t_grid_size):
        b[0:x_grid_size - 2] = B.dot(u[j - 1][1:x_grid_size - 1]) + tau * f(t[j], x[1:x_grid_size - 1])

        b[0] = (alpha / 2) * (u[j - 1][0]) + (1 - alpha) * u[j - 1][1] + (alpha / 2) * (u[j - 1][2]) + tau * f(t[j],
                                                                                                               x[1]) \
               + (alpha / 2) * (u[j][0])

        b[x_grid_size - 3] = (alpha / 2) * (u[j - 1][x_grid_size - 3]) + (1 - alpha) * u[j - 1][x_grid_size - 2] \
                             + (alpha / 2) * (u[j - 1][x_grid_size - 1]) + tau * f(t[j], x[x_grid_size - 2]) + (
                                         alpha / 2) * (u[j][x_grid_size - 1])

        u[j][1:x_grid_size - 1] = scipy.linalg.solve_banded((1, 1), A, b)
    return u

    # Calculating u


h = 0.1
tau = 0.1

u = solve(tau, h, 0.017)

# Creating calculated graph
x_grid_size = int(1 / h)
t_grid_size = int(1 / tau)
x = np.linspace(0, 1, x_grid_size)
t = np.linspace(0, 1, t_grid_size)
X, T = np.meshgrid(x, t)

CalculatedGraph = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, T, u, 30, cmap='binary')
ax.plot_surface(X, T, u, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.invert_xaxis()
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.set_title('Calculated Solution')

# Creating exact graph
X_exact, T_exact = np.meshgrid(x, t)

U_exact = exact_function(T, X)

ExactGraph = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X_exact, T_exact, U_exact, 30, cmap='binary')
ax.plot_surface(X_exact, T_exact, U_exact, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
ax.invert_xaxis()
ax.set_title('Exact Solution')

# Creating both graphs
BothGraph = plt.figure()
ax_both = plt.axes(projection='3d')
ax_both.plot_surface(X_exact, T_exact, U_exact, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
ax_both.plot_wireframe(X, T, u, cmap='binary')
ax_both.set_title('Both solutions')
ax_both.set_xlabel('x')
ax_both.set_ylabel('t')
ax_both.invert_xaxis()
ax_both.set_zlabel('u')

print("actual error", findMaxDeviation(u, U_exact))
print("theoretical error", tau + h * h)
plt.show()

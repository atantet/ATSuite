import numpy as np


def RK4(field, state, parameters, dt=0.01):
    k1 = dt * field(state, parameters)
    k2 = dt * field(state + 0.5 * k1, parameters)
    k3 = dt * field(state + 0.5 * k2, parameters)
    k4 = dt * field(state + k3, parameters)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def generateRK4(dataLength, field, state, parameters, dt):
    nt = int(dataLength / dt)
    data = np.empty((state.shape[0], nt))

    for i in np.arange(nt):
        state = RK4(field, state, parameters, dt)
        data[:, i] = state

    return data


def lorenzField((x, y, z), (sigma, rho, beta)):
        return np.array([sigma * (y - x),
                            x * (rho - z) - y,
                            x * y - beta * z])


def rosslerField((x, y, z), (a, b, c)):
        return np.array([-y - z, x + a * y, b + z * (x - c)])



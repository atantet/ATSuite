import numpy as np

def EM(fieldDrift, paramDrift, fieldDiff, paramDiff, state, noiseSample, dt):
    return state + dt * fieldDrift(state, paramDrift) \
        + np.sqrt(dt) * fieldDiff(state, paramDiff, noiseSample)


def generateEM(dataLength, fieldDrift, paramDrift, fieldDiff, paramDiff,
               state, noiseSamples, dt):
    nt = int(dataLength / dt)
    data = np.empty([state.shape[0], nt])

    for i in np.arange(nt):
        state = EM(fieldDrift, paramDrift, fieldDiff, paramDiff, state, noiseSamples[i], dt)
        data[:, i] = state

    return data


def additiveWienerField(state, (S,), noiseSample):
    return np.dot(S, noiseSample)

def linearWienerField(state, (S,), noiseSample):
    return state * np.dot(S, noiseSample)

import numpy as np


def et(b0, w_i, c_s, g):
    return b0 * (w_i / c_s) ** g


def runoff(w_i, c_s, a):
    return (w_i / c_s) ** a


def predict(curr_moist, evapo, wrunoff, precip, rad):
    return curr_moist + (precip - (evapo * rad) - (wrunoff * precip))


def seasonal_sinus(n_time_steps, amplitude=.5, freq=2, phase=5, center=.8):
    """

    :param center: 0 center will be shifted to given center
    :param n_time_steps: No. time steps
    :param amplitude: Amplitude of the sinusoidal curve
    :param freq: Frequency of the sinusoidal curve (for one year)
    :param phase: Phase of the sinusoidal curve tried
    :return:
    """

    freq = freq * np.pi / 365  # Frequency of the curve for one year

    # Calculate the sinusoidal curve values based on dates
    sinus_curve = amplitude * np.sin(freq * np.arange(n_time_steps) + phase)
    sinus_curve += center  # Centered at 0.8

    return sinus_curve

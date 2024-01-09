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


def predict_ts(data, config, n_days=None):
    """Run the SMBW for given time series

    :param data: input data (pandas df) (time, lat, long, tp, sm, ro, le, snr)
    :param config: parameters
                   - water holding capacity (c_s),
                   - maximum of ET function (b0),
                   - ET function shape (g),
                   - runoff function shape (a))
    :param n_days: time series length (default: None)
    :return: soil moisture, runoff, ET (for entire ts) (numpy arrays)
    """
    n_days = data.shape[0] if n_days is None else n_days

    # initialize arrays for model outputs
    moists, runoffs, ets = np.zeros(n_days), np.zeros(n_days), np.zeros(n_days)
    curr = {}  # to temporarily store parameters in loop

    # initial moisture (90% of soil water holding capacity)
    if isinstance(config['c_s'], (float, int)):
        moists[0] = 0.9 * config['c_s']
    else:
        moists[0] = 0.9 * config['c_s'][0]

    for i in range(n_days):

        # deal with parameters: flexible for both seasonal or constant
        for k, v in config.items():
            curr[k] = v[i] if isinstance(v, np.ndarray) else v
        c_s, b0, g, a = curr['c_s'], curr['b0'], curr['g'], curr['a']

        # compute evapotrans. and runoff
        ets[i] = et(b0, moists[i], c_s, g)
        runoffs[i] = runoff(moists[i], c_s, a)

        # compute soil moisture
        if i < n_days - 1:
            moists[i + 1] = predict(moists[i], ets[i], runoffs[i],
                                    data['tp'][i], data['snr'][i])

    return moists, runoffs, ets

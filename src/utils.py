import arrow
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from src.swbm import predict_ts, seasonal_sinus


def minimize_res2df(result, opt_swbm_params):
    """Extract sinus parameters from optimizer function output and put into
    dataframe

    :param result: minimize function output
    :param opt_swbm_params: SWBM parameters that were optimized e.g. ['b0']
    :return: pandas dataframe with SWBM param. columns and sinus param. indices
    """
    # Print optimized sinus parameters
    opt_params = np.reshape(result['x'], (len(opt_swbm_params), 4))
    opt_params_df = {p: val for p, val in zip(opt_swbm_params, opt_params)}
    opt_params_df = pd.DataFrame(opt_params_df,
                                 index=['amplitude', 'freq', 'phase', 'center'])
    return opt_params_df


def eval_swbm(obs_data, model_data, swbm_param=None):
    """Compute soil moisture, runoff and evapotrans. correlation of model and
    observed time series

    :param input: SWBM input df
    :param output: SWBM output (as pandas df)
    :param swbm_param: list of SWBM parameters that were optimiezed e.g. ['b0']
    :return: pandas dataframe with parameter, corr, pval and kind (ms/ro/le)
    """
    eval_dict = {'parameter': [],
                 'kind': [],
                 'corr': [],
                 'pval': []}
    # Test correlation
    for i in ['sm', 'ro', 'le']:
        corr, p = pearsonr(model_data[i], obs_data[i])
        eval_dict['parameter'].append(swbm_param)
        eval_dict['corr'].append(corr)
        eval_dict['pval'].append(p)
        eval_dict['kind'].append(i)
    return pd.DataFrame(eval_dict)


def prepro(raw_data):
    """ Preprocess data for SWBM
    Convert runoff, latent heat flux and solar net radiation to mm.
    Convert time to date.

    :param raw_data: raw input data (pandas df):
         -snr: surface net radiation
         -tp: total precipitation
         -ro: runoff
         -sm: soil moisture at the surface
         -le: latent heat flux
    :return: pre-processed data (pandas df)
    """

    data = {'time': pd.to_datetime(raw_data['time']),
            'lat': raw_data['latitude'],
            'long': raw_data['longitude'],
            'tp': raw_data['tp_[mm]'],
            'sm': raw_data['sm_[m3/m3]'],
            'ro': raw_data['ro_[m]'] * 24000,
            'le': raw_data['le_[W/m2]'] * (86400 / 2260000),
            # 86400 (seconds) / 2260000 (latent heat of vaporization
            # of water in J/kg)
            'snr': raw_data['snr_[MJ/m2]'] * (1 / 2.26),
            }

    return pd.DataFrame(data)


def opt_swbm_corr(inits, data, params, seasonal_param):
    """ Calculates correlation between Swbm with sesonal parameter variation
    and true values

    :param inits: initial parameters for seasonal sinus function
    :param data: input (true) data (pandas df) (time, lat, long, tp, sm, ro, le,
                 snr)
    :param params: parameters for Swbm (look predict_ts), can be empty dict if
                   all parameters will be seasonal
    :param seasonal_param: parameter(s) to set seasonal (str or list)
    :return: correlation
    """
    # if single parameters is given make list
    seasonal_param = ([seasonal_param] if isinstance(seasonal_param, str)
                      else seasonal_param)

    inits = np.reshape(inits, (len(seasonal_param), 4))
    # (no. SWBM const_swbm_params. x no. sinus const_swbm_params.)

    # Make seasonal parameters
    for param, sinus_init in zip(seasonal_param, inits):
        params[param] = seasonal_sinus(len(data),
                                       amplitude=sinus_init[0],
                                       freq=sinus_init[1],
                                       phase=sinus_init[2],
                                       center=sinus_init[3],
                                       which=param)

    # Run SWBM
    out_sm, out_ro, out_et = predict_ts(data, params)
    if 'a' in seasonal_param and len(seasonal_param) == 1:
        # only optimize runoff
        score, pval = pearsonr(out_ro, data['ro'])
    else:
        corr_sm, pval = pearsonr(out_sm, data['sm'])
        corr_ro, _ = pearsonr(out_ro, data['ro'])
        corr_et, _ = pearsonr(out_et, data['le'])
        # include ro and et
        score = 1 * corr_sm + 0 * corr_ro + 0 * corr_et

    if pval > 0.05:
        print(f'No corr. P={pval}')

    return score * -1  # to get maximum


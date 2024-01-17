import arrow
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


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

    data = {'time': [arrow.get(date) for date in raw_data['time']],
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

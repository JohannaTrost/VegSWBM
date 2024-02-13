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
            'sm': raw_data['sm_[m3/m3]'] * 1000,
            'ro': raw_data['ro_[m]'] * 24000,
            'le': raw_data['le_[W/m2]'] * (86400 / 2260000),
            # 86400 (seconds) / 2260000 (latent heat of vaporization
            # of water in J/kg)
            'snr': raw_data['snr_[MJ/m2]'] * (1 / 2.26),
            }

    return pd.DataFrame(data)


def transform_evals(input_data):
    """ Transform function for evaluation """
    output_data = [
        input_data['Combination'],
        {'sum_corr': input_data['sum_corr']},
        {'sm_corr': input_data['eval_df']['corr'][0],
         'ro_corr': input_data['eval_df']['corr'][1],
         'le_corr': input_data['eval_df']['corr'][2]},
        input_data['SinusParameters']
    ]
    return output_data


def transform_data(input_data):
    output_data = [
        {'c_s': input_data['Combination'][0],
         'b0': input_data['Combination'][1],
         'g': input_data['Combination'][2],
         'a': input_data['Combination'][3]},
        {'sum_corr': input_data['sum_corr']},
        {'sm_cor': input_data['eval_df']['corr'][0],
         'ro_cor': input_data['eval_df']['corr'][1],
         'le_cor': input_data['eval_df']['corr'][2]},
        {'amplitude': input_data['SinusParameters']['b0']['amplitude'],
         'freq': input_data['SinusParameters']['b0']['freq'],
         'phase': input_data['SinusParameters']['b0']['phase'],
         'center': input_data['SinusParameters']['b0']['center']}
    ]
    return output_data


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
    out_sm, out_ro, out_et, na_counts = predict_ts(data, params)

    # Deal with NAs
    if na_counts['sm'] > 0:
        out_sm[np.isnan(out_sm)] = 0
    if na_counts['le'] > 0:
        out_et[np.isnan(out_et)] = 0
    if na_counts['ro'] > 0:
        out_ro[np.isnan(out_ro)] = 0

    # optimize for runoff correlation if only runoff shape param. seasonal
    if 'a' in seasonal_param and len(seasonal_param) == 1:
        # only optimize runoff
        score, pval = pearsonr(out_ro, data['ro'])
    elif 'b0' in seasonal_param and len(seasonal_param) == 1:
        # only optimize ET
        score, pval = pearsonr(out_et, data['le'])
    else:  # optimize for all parameters
        corr_sm, pval = pearsonr(out_sm, data['sm'])
        corr_ro, _ = pearsonr(out_ro, data['ro'])
        corr_et, _ = pearsonr(out_et, data['le'])
        # include ro and et
        score = 1 * corr_sm + 1 * corr_ro + 1 * corr_et

    return score * -1  # to get maximum

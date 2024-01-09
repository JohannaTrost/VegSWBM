# %%
# os.chdir('..')
# %%
import seaborn as sns
from scipy.stats import pearsonr
from scipy.optimize import minimize

from src.swbm import *
from src.plots import *
from src.utils import *


# %%
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
    # (no. SWBM params. x no. sinus params.)

    # Make seasonal parameters
    for param, sinus_init in zip(seasonal_param, inits):
        params[param] = seasonal_sinus(len(data),
                                       amplitude=sinus_init[0],
                                       freq=sinus_init[1],
                                       phase=sinus_init[2],
                                       center=sinus_init[3],
                                       which=param)

    # Run SWBM
    out_sm, _, _ = predict_ts(data, params)
    corr_sm, p_sm = pearsonr(out_sm, data['sm'])

    if p_sm > 0.05:
        print(f'No corr. P={p_sm}')
    else:
        print(corr_sm)

    return corr_sm * -1  # to get maximum


# %%
# Load and pre-process data
input_swbm_raw = pd.read_csv('data/Data_swbm_Germany.csv')
input_swbm = prepro(input_swbm_raw)

# %%
# initialize parameters and sinus params
init_sinus_params_all = [[0.5, 2, 5, 420],
                         # c_s (amplitude, freq, phase, center)
                         [0.5, 2, 5, 0.8],  # b0 -> max. ET
                         [0.1, 2, 5, 0.5],  # g -> ET function shape
                         [1, 2, 5, 4]]  # a -> runoff function shape
make_seasonal_all = ['c_s', 'b0', 'g', 'a']

np.random.seed(42)
res_all = minimize(opt_swbm_corr,
                   np.asarray(init_sinus_params_all).flatten(),  # has to be 1D
                   args=(input_swbm, {}, make_seasonal_all),
                   options={"maxiter": 500, "disp": True})

# print optimized sinus parameters
opt_params_all = np.reshape(res_all['x'], (len(make_seasonal_all), 4))
# (no. SWBM params. x no. sinus params.)
opt_params_all_df = {p: val for p, val in
                     zip(make_seasonal_all, opt_params_all)}
opt_params_all_df = pd.DataFrame(opt_params_all_df,
                                 index=['amplitude', 'freq', 'phase', 'center'])
print(opt_params_all_df)

# get optimized seasonal SWBM parameters
opt_sinus_all = {}
for swbm_param in opt_params_all_df:
    opt_sinus_all[swbm_param] = seasonal_sinus(
        len(input_swbm),
        amplitude=opt_params_all_df.loc['amplitude', swbm_param],
        freq=opt_params_all_df.loc['freq', swbm_param],
        phase=opt_params_all_df.loc['phase', swbm_param],
        center=opt_params_all_df.loc['center', swbm_param],
        which=swbm_param
    )

opt_sinus_all_df = pd.DataFrame(opt_sinus_all)

# plot all sinus curves
melted_df = opt_sinus_all_df.melt(var_name='SWBM parameter', value_name='Value',
                                  ignore_index=False)
# sns.relplot(
#     data=melted_df, kind='line',
#     col='region', x='timepoint', y='signal', units='subject',
#     estimator=None, color='.7'
# )
# plt.show()

# %%
# ---- Evaluation

# %%
# Set swbm params
params = {'c_s': 420, 'b0': 0.8, 'g': .5, 'a': 4}
params_seasonal = {'c_s': opt_sinus, 'b0': 0.8, 'g': .5, 'a': 4}  # TODO

# %%
# Run SWBM
moists, runoffs, ets = predict_ts(input_swbm, params)
moists_seasonal, runoffs_seasonal, ets_seasonal = predict_ts(input_swbm,
                                                             params_seasonal)

output_swbm = {'sm': moists, 'ro': runoffs, 'le': ets}
output_swbm_seasonal = {'sm': moists_seasonal,
                        'ro': runoffs_seasonal,
                        'le': ets_seasonal}

eval = {'model': [], 'kind': [], 'corr': [], 'pval': []}
for model, out_swbm in zip(['Constant', 'Seasonal Beta'],
                           [output_swbm, output_swbm_seasonal]):
    for key in ['sm', 'ro', 'le']:
        corr, p = pearsonr(out_swbm[key], input_swbm[key])

        eval['corr'].append(corr)
        eval['pval'].append(p)
        eval['model'].append(model)
        eval['kind'].append(key)

eval_df = pd.DataFrame(eval)
print(np.round(eval_df, 3))

# %%
# -- some plots

# only show one year
input_swbm['time'] = [arrow.get(date) for date in input_swbm['time']]
year_mask = [date.year == 2010 for date in input_swbm['time']]

fig, ax = plt.subplots()
ax.set_title('Seasonal Beta')
ax.scatter(moists_seasonal[year_mask],
           ets_seasonal[year_mask], label='ET/Rnet', alpha=0.5)
ax.scatter(moists_seasonal[year_mask],
           runoffs[year_mask], label='Runoff (Q)', alpha=0.5)
ax.set_xlabel('Soil moisture(mm)')
plt.legend()
plt.tight_layout()
# plt.savefig('figs/b0_seasonal_rel.pdf')

fig, ax = plt.subplots()
ax.set_title('Seasonal Beta')
plot_time_series(moists_seasonal[year_mask], ets_seasonal[year_mask],
                 runoffs[year_mask], ax)
ax.set_ylabel('Soil moisture(mm)')
plt.legend()
plt.tight_layout()
# plt.savefig('figs/b0_seasonal_ts_2010.pdf')

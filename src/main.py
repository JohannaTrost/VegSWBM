#%%
#os.chdir('..')
#%%
from scipy.stats import pearsonr
from scipy.optimize import minimize

from src.swbm import *
from src.plots import *
from src.utils import *


def opt_swbm_corr(inits, data, params, seasonal_param):
    # Set seasonal parameters
    params[seasonal_param] = seasonal_sinus(len(data['time']),
                                            amplitude=inits[0],
                                            freq=inits[1],
                                            phase=inits[2],
                                            center=inits[3])
    # Run SWBM
    out_sm, _, _ = predict_ts(data, params)
    corr_sm, p_sm = pearsonr(out_sm, data['sm'])

    if p_sm > 0.05:
        print(f'No corr. P={p_sm}')

    return corr_sm * -1  # to get maximum


# Load and pre-process data
input_swbm_raw = pd.read_csv('data/Data_swbm_Germany.csv')
input_swbm = prepro(input_swbm_raw)

# initialize parameters and sinus params
config = {'c_s': 420, 'b0': .8, 'g': .5, 'a': 4}
init_sinus_params = [0.5, 2, 5, 0.8]

np.random.seed(42)
res = minimize(opt_swbm_corr, init_sinus_params,
               args=(input_swbm, config, 'b0'),
               options={"maxiter": 100,
                        "disp": True})

print(f"Optimal beta sinus parameters:\n"
      f"\tamplitude={np.round(res['x'][0], 3)}\n"
      f"\tfreq={np.round(res['x'][1], 3)}\n"
      f"\tphase={np.round(res['x'][2], 3)}\n"
      f"\tcenter={np.round(res['x'][3], 3)}")

# ---- Evaluation
opt_sinus_b0 = seasonal_sinus(len(input_swbm),
                              amplitude=res['x'][0],
                              freq=res['x'][1],
                              phase=res['x'][2],
                              center=res['x'][3])
# Set swbm params
params = {'c_s': 420, 'b0': 0.8, 'g': .5, 'a': 4}
params_seasonal = {'c_s': 420, 'b0': opt_sinus_b0, 'g': .5, 'a': 4}

# Run SWBM
moists, runoffs, ets = predict_ts(input_swbm, params)
moists_seasonal, runoffs_seasonal, ets_seasonal = predict_ts(input_swbm,
                                                             params_seasonal)

output_swbm = {'sm': moists, 'ro': runoffs, 'et': ets}
output_swbm_seasonal = {'sm': moists_seasonal,
                        'ro': runoffs_seasonal,
                        'et': ets_seasonal}

eval = {'model': [], 'kind': [], 'corr': [], 'pval': []}
for model, out_swbm in zip(['Constant', 'Seasonal Beta'],
                           [output_swbm, output_swbm_seasonal]):
    for key in ['sm', 'ro']:
        corr, p = pearsonr(out_swbm[key], input_swbm[key])

        eval['corr'].append(corr)
        eval['pval'].append(p)
        eval['model'].append(model)
        eval['kind'].append(key)

eval_df = pd.DataFrame(eval)
print(np.round(eval_df, 3))

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

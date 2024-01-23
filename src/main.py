# %%
# os.chdir('..')
# %%
# Imports
from scipy.optimize import minimize

from src.plots import *
from src.utils import *

np.random.seed(42)

# %%
# Load and pre-process data
input_swbm_raw = pd.read_csv('data/Data_swbm_Germany.csv')
input_swbm = prepro(input_swbm_raw)

# %%
# Calibration (opt c_s, g, a, b0 seasonal)

# %%
# ---- Single parameter optimization
const_swbm_params = {'c_s': 420, 'b0': 0.8, 'g': .5, 'a': 4}

# %%
# Run SWBM without seasonal variation
moists, runoffs, ets = predict_ts(input_swbm, const_swbm_params)
eval_df = eval_swbm(input_swbm,
                    {'sm': moists, 'ro': runoffs, 'le': ets},
                    'None\nSeasonal')

# ---- Seasonal Variation for single parameter
# %%
swbm_param = 'b0'

# search different solvers

solvers = ['Nelder-Mead']
# , 'Powell', 'CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
# 'TNC', 'COBYLA', 'SLSQP', 'trust-constr', 'dogleg', 'trust-exact',
# 'trust-exact', 'trust-krylov']
max_corr = -np.inf
for solver in solvers:
    init_values = [0.5, 2, 5, 0.8]
    const_swbm_params = {'c_s': 420, 'b0': 0.8, 'g': .5, 'a': 4}

    res = minimize(opt_swbm_corr,
                   init_values,
                   args=(input_swbm, const_swbm_params, swbm_param),
                   options={"maxiter": 500, "disp": True})
    curr_opt_params_df = minimize_res2df(res, [swbm_param])

    # Set swbm const_swbm_params
    curr_params_seasonal = {
        'c_s': 420,
        'g': .5,
        'a': 4,
        'b0': seasonal_sinus(
            len(input_swbm),
            amplitude=curr_opt_params_df.loc['amplitude', swbm_param],
            freq=curr_opt_params_df.loc['freq', swbm_param],
            phase=curr_opt_params_df.loc['phase', swbm_param],
            center=curr_opt_params_df.loc['center', swbm_param],
            which=swbm_param
        )
    }

    # Run SWBM
    preds_seasonal = predict_ts(input_swbm, curr_params_seasonal)
    moists_seasonal, runoffs_seasonal, ets_seasonal = preds_seasonal

    # Test correlation
    curr_corr, _ = pearsonr(input_swbm['sm'], moists_seasonal)

    print(curr_corr)

    # check if is better
    if curr_corr > max_corr:
        max_corr = curr_corr
        b0_model_preds = {'sm': moists_seasonal,
                          'ro': runoffs_seasonal,
                          'le': ets_seasonal}
        params_seasonal = curr_params_seasonal
        opt_params_df = curr_opt_params_df

eval_df = pd.concat((eval_df, eval_swbm(input_swbm, b0_model_preds, 'b0')))

# %%
# visualize b0-model vs. constant-model vs. observed
year_mask = [date.year == 2010 or date.year == 2011
             for date in input_swbm['time']]
x_ticks = input_swbm['time'][year_mask]

fig, ax = plt.subplots(nrows=3, figsize=(9, 16))

ax[0].plot(x_ticks, input_swbm['sm'][year_mask] * 1000, label='Observed')
ax[0].plot(x_ticks, b0_model_preds['sm'][year_mask], label='B0-seasonal-model')
ax[0].plot(x_ticks, moists[year_mask], label='Constant model')
ax[0].set_title('Soil moisture')
ax[0].legend()

ax[1].plot(x_ticks, input_swbm['le'][year_mask], label='Observed')
ax[1].plot(x_ticks, b0_model_preds['le'][year_mask], label='B0-seasonal-model')
ax[1].plot(x_ticks, ets[year_mask], label='Constant model')
ax[1].set_title('Evapotranspiration')
ax[1].legend()

ax[2].plot(x_ticks, input_swbm['ro'][year_mask], label='Observed')
ax[2].plot(x_ticks, b0_model_preds['ro'][year_mask], label='B0-seasonal-model')
ax[2].plot(x_ticks, runoffs[year_mask], label='Constant model')
ax[2].set_title('Runoff')
ax[2].legend()

plt.tight_layout()
plt.show()

# %%
print(params_seasonal['b0'])

# %%
# plot our sinus for B0
plt.plot(x_ticks, params_seasonal['b0'])
plt.xticks(rotation=45)
plt.show()

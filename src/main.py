from scipy.stats import pearsonr
from scipy.optimize import minimize

from src.swbm import *
from src.plots import *
from src.utils import *


def cost_fun(amplis, data):
    ampli = amplis[0]
    # Set parameter values
    params = {'c_s': 420,
              'b0': seasonal_sinus(len(data['time']), amplitude=ampli),
              'g': .5, 'a': 4}
    # Run SWBM
    moists, runoffs, ets = predict_ts(data, params)
    out_swbm = {'sm': moists, 'ro': runoffs, 'et': ets}

    # evaluate
    #print(f'\nampli={ampli}')
    score = 0
    for key in ['sm', 'ro']:
        corr, p = pearsonr(out_swbm[key], data[key])
        #print(f'{key}: corr={corr}, P={p}')
        score += corr
    return score * - 1  # to get maximum


# Load and pre-process data
input_swbm_raw = pd.read_csv('data/Data_swbm_Germany.csv')
input_swbm = prepro(input_swbm_raw)

# vary initial value for amplitude of b0 sinus curve, try diff. random seeds
results = {'init': [], 'ampli': [], 'score': []}
for ampli_sin_b0 in [0.0000001, 0.5, 1]:
    for i in range(100):
        # run optimizer
        np.random.seed(i)
        res = minimize(cost_fun, [ampli_sin_b0], args=input_swbm,
                       options={"maxiter": 100,
                                "disp": True})
        # results
        results['init'].append(ampli_sin_b0)
        results['ampli'].append(res['x'][0])
        results['score'].append(res['fun'] * -1)

results = pd.DataFrame(results)
results.to_csv('results/seed_0-99_opt_ampli_b0.csv')

# ---- Evaluation

opt_ampli = np.mean(results['ampli'])

params = {'c_s': 420,
          'b0': 0.8,
          'g': .5, 'a': 4}
params_seasonal = {'c_s': 420,
                   'b0': seasonal_sinus(len(input_swbm), amplitude=opt_ampli),
                   'g': .5, 'a': 4}
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

eval_df = np.round(pd.DataFrame(eval), 3)
print(eval_df)
eval_df.to_csv('results/eval_seed_0-99_opt_ampli_b0.csv')

# -- some plots

# only show one year
input_swbm['time'] = [arrow.get(date) for date in input_swbm['time']]
year_mask = [date.year == 2010 for date in input_swbm['time']]

fig, ax = plt.subplots()
ax.set_title('Beta = 0.8')
plot_relation(moists[year_mask], ets[year_mask], runoffs[year_mask], ax)
ax.set_xlabel('Soil moisture(mm)')
plt.legend()
plt.tight_layout()
plt.savefig('figs/b0_0.8_rel.pdf')

fig, ax = plt.subplots()
ax.set_title('Beta = 0.8')
plot_time_series(moists[year_mask], ets[year_mask], runoffs[year_mask], ax)
ax.set_ylabel('Soil moisture(mm)')
plt.legend()
plt.tight_layout()
plt.savefig('figs/b0_0.8_ts_2010.pdf')

fig, ax = plt.subplots()
ax.set_title('Seasonal Beta')
ax.scatter(moists_seasonal[year_mask],
           ets_seasonal[year_mask], label='ET/Rnet', alpha=0.5)
ax.scatter(moists_seasonal[year_mask],
           runoffs[year_mask], label='Runoff (Q)', alpha=0.5)
ax.set_xlabel('Soil moisture(mm)')
plt.legend()
plt.tight_layout()
plt.savefig('figs/b0_seasonal_rel.pdf')

fig, ax = plt.subplots()
ax.set_title('Seasonal Beta')
plot_time_series(moists_seasonal[year_mask], ets_seasonal[year_mask],
                 runoffs[year_mask], ax)
ax.set_ylabel('Soil moisture(mm)')
plt.legend()
plt.tight_layout()
plt.savefig('figs/b0_seasonal_ts_2010.pdf')

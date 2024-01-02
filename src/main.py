from scipy.stats import pearsonr

from src.swbm import *
from src.plots import *
from src.utils import *

# Load and pre-process data
input_swbm_raw = pd.read_csv('data/Data_swbm_Germany.csv')
input_swbm = prepro(input_swbm_raw)

# Set parameter values
params_seasonal = {'c_s': 420,
                   'b0': seasonal_sinus(len(input_swbm['time']), amplitude=0.5),
                   'g': .5, 'a': 4}
params = params_seasonal.copy()
params['b0'] = 0.8

# Run SWBM
moists, runoffs, ets = predict_ts(input_swbm, params)
moists_seasonal, runoffs_seasonal, ets_seasonal = predict_ts(input_swbm,
                                                             params_seasonal)

# evaluate
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

print(np.round(pd.DataFrame(eval), 3))

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


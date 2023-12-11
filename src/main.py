import arrow
import pandas as pd

from smbw import *
from plots import *

input_swbm = pd.read_csv('data/input_swbm.csv')

input_swbm['time'] = [arrow.get(date) for date in input_swbm['time']]

# set parameter values
params = {'c_s': 420,
          'b0': seasonal_sinus(len(input_swbm['time']), amplitude=0.5),
          'g': .5, 'a': 4}

R = input_swbm['snr_[MJ/m2]'] * (1 / 2.26)
P = input_swbm['tp_[mm]']

# extract parameters
c_s, b0, g, a = params['c_s'], params['b0'], params['g'], params['a']

n_days = input_swbm.shape[0]

# -- run model for given params
moists, runoffs, ets = np.zeros(n_days), np.zeros(n_days), np.zeros(n_days)

ets_seasonal, moists_seasonal = np.zeros(n_days), np.zeros(n_days)
runoffs_seasonal = np.zeros(n_days)

# initial moisture
moists[0] = 0.9 * params['c_s']  # 90% of soil water holding capacity
moists_seasonal[0] = 0.9 * params['c_s']

for i in range(n_days):
    ets_seasonal[i] = et(b0[i], moists_seasonal[i], c_s, g)
    runoffs_seasonal[i] = runoff(moists_seasonal[i], c_s, a)

    ets[i] = et(0.8, moists[i], c_s, g)
    runoffs[i] = runoff(moists[i], c_s, a)

    if i < n_days - 1:
        moists[i + 1] = predict(moists[i], ets[i], runoffs[i], P[i], R[i])
        moists_seasonal[i + 1] = predict(moists_seasonal[i],
                                         ets_seasonal[i],
                                         runoffs_seasonal[i], P[i],
                                         R[i])

# only show one year
input_swbm['time'] = [arrow.get(date) for date in input_swbm['time']]
year_mask = [date.year == 2010 for date in input_swbm['time']]

fig, ax = plt.subplots(2, 2, figsize=(15, 9))

ax[0, 0].set_title('Beta = 0.8')
plot_relation(moists[year_mask], ets[year_mask], runoffs[year_mask], ax[0, 0])
plot_time_series(moists[year_mask], ets[year_mask], runoffs[year_mask],
                 ax[0, 1])

ax[1, 0].set_title('Seasonal Beta')
ax[1, 0].scatter(moists_seasonal[year_mask],
                 ets_seasonal[year_mask], label='ET/Rnet', alpha=0.5)
ax[1, 0].scatter(moists_seasonal[year_mask],
                 runoffs[year_mask], label='Runoff (Q)', alpha=0.5)
plot_time_series(moists_seasonal[year_mask], ets_seasonal[year_mask],
                 runoffs[year_mask], ax[1, 1])

ax[1, 1].set_ylabel('Soil moisture(mm)')
ax[0, 1].set_ylabel('Soil moisture(mm)')
ax[0, 0].set_xlabel('Soil moisture(mm)')
ax[1, 0].set_xlabel('Soil moisture(mm)')

for i in range(4):
    ax[np.unravel_index(i, (2, 2))].legend()
plt.legend()
plt.tight_layout()

# TODO save plots and vary amplitude !!

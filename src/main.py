# %%
# os.chdir('..')
# %%
# Imports
import json
import os

from scipy.optimize import minimize

from src.plots import *
from src.utils import *

np.random.seed(42)

# %%

# Paths
data_paths = [os.path.join('data', 'Data_swbm_Germany.csv'),
              os.path.join('data', 'Data_swbm_Spain.csv'),
              os.path.join('data', 'Data_swbm_Sweden.csv')]
calib_params_paths = [os.path.join('results', 'ger_output.json'),
                      os.path.join('results', 'esp_output.json'),
                      os.path.join('results', 'swe_output.json')]

for data_path, calib_params_path in zip(data_paths, calib_params_paths):
    # Load and pre-process data
    input_swbm_raw = pd.read_csv(data_path)
    input_swbm = prepro(input_swbm_raw)

    # %%

    with open(calib_params_path, 'r') as json_file:
        calib_out = json.load(json_file)

    const_swbm_params = calib_out[0].copy()
    sinus_params = calib_out[-1].copy()

    # %%
    # Run SWBM without seasonal variation
    moists, runoffs, ets, _ = predict_ts(input_swbm, const_swbm_params)
    eval_df = eval_swbm(input_swbm,
                        {'sm': moists, 'ro': runoffs, 'le': ets},
                        'None\nSeasonal')

    # %%
    # Run SWBM with seasonal variation of B0
    swbm_param = 'b0'
    params_seasonal = {
        'c_s': const_swbm_params['c_s'],
        'g': const_swbm_params['g'],
        'a': const_swbm_params['a'],
        'b0': seasonal_sinus(
            len(input_swbm),
            amplitude=sinus_params['amplitude'],
            freq=sinus_params['freq'],
            phase=sinus_params['phase'],
            center=sinus_params['center'],
            which='b0'
        )
    }

    # Run SWBM
    preds_seasonal = predict_ts(input_swbm, params_seasonal)
    b0_model_preds = {'sm': preds_seasonal[0],
                      'ro': preds_seasonal[1],
                      'le': preds_seasonal[2]}
    eval_df = pd.concat((eval_df, eval_swbm(input_swbm, b0_model_preds, 'b0')))

    print(data_path)

    pd.DataFrame(b0_model_preds).to_csv(
        os.path.join('data', 'output', os.path.basename(data_path)))

    # %%
    # visualize b0-model vs. constant-model vs. observed
    year_mask = [date.year == 2009 for date in input_swbm['time']]
    x_ticks = input_swbm['time'][year_mask]

    fig, ax = plt.subplots(2, 2, figsize=(16, 9))

    ax[0, 0].plot(x_ticks, input_swbm['sm'][year_mask], label='Observed')
    ax[0, 0].plot(x_ticks, b0_model_preds['sm'][year_mask],
                  label='B0-seasonal-model')
    ax[0, 0].plot(x_ticks, moists[year_mask], label='Constant model')
    ax[0, 0].set_title('Soil moisture')
    ax[0, 0].legend()

    ax[0, 1].plot(x_ticks, input_swbm['le'][year_mask], label='Observed')
    ax[0, 1].plot(x_ticks, b0_model_preds['le'][year_mask],
                  label='B0-seasonal-model')
    ax[0, 1].plot(x_ticks, ets[year_mask], label='Constant model')
    ax[0, 1].set_title('Evapotranspiration')
    ax[0, 1].legend()

    ax[1, 0].plot(x_ticks, input_swbm['ro'][year_mask], label='Observed')
    ax[1, 0].plot(x_ticks, b0_model_preds['ro'][year_mask],
                  label='B0-seasonal-model')
    ax[1, 0].plot(x_ticks, runoffs[year_mask], label='Constant model')
    ax[1, 0].set_title('Runoff')
    ax[1, 0].legend()

    ax[1, 1].set_visible(False)

    plt.tight_layout()
    plt.show()

    # %%
    # plot our sinus for B0
    plt.plot(x_ticks, params_seasonal['b0'][year_mask])
    plt.xticks(rotation=45)
    plt.show()

    # %%
    fig, ax = plt.subplots()
    ax.set_title('Seasonal Beta')
    ax.plot(b0_model_preds['sm'][year_mask],
           b0_model_preds['le'][year_mask] / input_swbm['snr'][year_mask],
           label='ET/Rnet', alpha=0.5)
    ax.plot(b0_model_preds['sm'][year_mask],
           b0_model_preds['ro'][year_mask] / input_swbm['tp'][year_mask],
           label='Runoff (Q/P)', alpha=0.5)
    ax.set_xlabel('Soil moisture(mm)')
    plt.legend()
    plt.tight_layout()
    plt.close('all')
    # plt.savefig('figs/b0_seasonal_rel.pdf')

    fig, ax = plt.subplots()
    ax.set_title('Seasonal Beta')
    ax, ax2 = plot_time_series(b0_model_preds['sm'][year_mask],
                               b0_model_preds['le'][year_mask],
                               b0_model_preds['ro'][year_mask], ax)
    ax.plot(range(365), input_swbm['sm'][year_mask], label='True sm',
            linestyle='dashed', color='grey', alpha=0.5)
    ax.set_ylabel('Soil moisture(mm)')
    ax.legend()
    ax2.legend()
    plt.tight_layout()
    # plt.savefig('figs/b0_seasonal_ts_2010.pdf')

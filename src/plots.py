from matplotlib import pylab as plt


def plot_time_series(soil_moist, evapo_trans, prec_runoff, ax):
    ax2 = ax.twinx()
    ax.plot(soil_moist, color='g')
    ax2.plot(evapo_trans, label='ET(mm/day)', alpha=.5)
    ax2.plot(prec_runoff, label='Runoff(mm/day)', alpha=.5)

    ax.yaxis.label.set_color('g')
    ax.tick_params(axis='y', colors='g')

    ax2.set_ylabel('mm per day')

    return ax2


def plot_relation(soil_moist, evapo_trans, prec_runoff, ax):
    ax.plot(soil_moist, evapo_trans, label='ET')
    ax.plot(soil_moist, prec_runoff, label='Runoff')

    return ax

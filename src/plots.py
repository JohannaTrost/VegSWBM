import seaborn as sns
import pandas as pd
from matplotlib import pylab as plt


# Provided parameter labels
parameter_labels = {
    'None\nSeasonal': 'None\nSeasonal',
    'b0': 'Beta0',
    'g': 'Gamma',
    'a': 'Alpha',
    'c_s': 'Water Holding Capacity',
    'not b0': '-Beta0',
    'not g': '-Gamma',
    'not a': '-Alpha',
    'not c_s': 'Without Water\nHolding Capacity',
    'all': 'All seasonal'
}


def plot_avg_scores(scores, save=None):

    # Add 'Avg' kind and calculate average scores
    avg_df = scores.groupby('parameter')['corr'].mean().reset_index()
    avg_df['kind'] = 'Avg'
    scores = pd.concat([scores, avg_df], ignore_index=True, sort=False)

    # Map parameter names to labels
    scores['parameter_label'] = scores['parameter'].map(parameter_labels)

    # Filter DataFrame to keep only 'Avg' kind
    eval_df_avg = scores[scores['kind'] == 'Avg']

    # Sort the DataFrame by the 'Avg' score
    eval_df_avg_sorted = eval_df_avg.sort_values(by='corr', ascending=False)

    # Plot single opt results
    ax = sns.barplot(data=eval_df_avg_sorted, x='corr', y='parameter_label',
                     palette='viridis')

    # Annotate the values on top of each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.2f}',
                    (p.get_width(), p.get_y() + p.get_height() / 2.),
                    ha='center',
                    va='center',
                    xytext=(10, 0),
                    textcoords='offset points',
                    fontsize=10)

    # Remove plot borders and x-axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_visible(False)

    # Set labels and title
    plt.ylabel('(Not) Optimized Parameter')
    plt.title('Average Correlation Scores')

    # Show the plot
    plt.show()

    # save plot
    if save is not None:
        plt.savefig(save)


def plot_param_importance(importances, save=None):
    # Define the parameter labels
    xtick_labs = {'b0': 'Beta0',
                  'g': 'Gamma',
                  'a': 'Alpha',
                  'c_s': 'Water Holding\nCapacity'}

    # Map parameter names to labels
    importances['parameter_label'] = importances['parameter'].map(xtick_labs)

    # Order the DataFrame by feature_importance in descending order
    importances = importances.sort_values(by='feature_importance',
                                          ascending=False)

    # Create the bar plot
    ax = sns.barplot(x='feature_importance', y='parameter_label',
                     data=importances, color='skyblue')

    # Annotate the values on top of each bar
    for index, value in enumerate(importances['feature_importance']):
        ax.text(value, index, f'{value:.4f}', va='center', fontsize=10)

    # Remove plot borders and x-axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.xaxis.set_visible(False)

    # Set labels and title
    plt.ylabel('')
    plt.xlabel('Correlation difference')
    plt.title('Importance of SWBM Parameters')

    # Show the plot
    plt.tight_layout()
    plt.show()

    # save plot
    if save is not None:
        plt.savefig(save)


def plot_time_series(soil_moist, evapo_trans, prec_runoff, ax):
    ax2 = ax.twinx()
    ax.plot(soil_moist, color='g')
    ax2.plot(evapo_trans, label='ET(mm/day)', alpha=.5)
    ax2.plot(prec_runoff, label='Runoff(mm/day)', alpha=.5)

    ax.yaxis.label.set_color('g')
    ax.tick_params(axis='y', colors='g')

    ax2.set_ylabel('mm per day')

    return ax, ax2


def plot_relation(soil_moist, evapo_trans, prec_runoff, ax):
    ax.plot(soil_moist, evapo_trans, label='ET')
    ax.plot(soil_moist, prec_runoff, label='Runoff')

    return ax

# VISUALIZE sinus parameters
# opt_sinus_all_df = pd.DataFrame(preds_seasonal_all)
# opt_sinus_all_df['time'] = [date.format('YYYY-MM-DD')
#                             for date in input_swbm['time']]
# year_mask = [arrow.get(date).year == 2010 for date in opt_sinus_all_df['time']]
#
# # plot all sinus curves
# melted_df = opt_sinus_all_df[year_mask].melt(var_name='SWBM parameter',
#                                              value_name='Value',
#                                              id_vars=['time'],
#                                              ignore_index=False)
# g = sns.relplot(data=melted_df, kind='line',
#                 col='SWBM parameter', y='Value', x='time',
#                 estimator=None, col_wrap=2,
#                 facet_kws={'sharey': False, 'sharex': True})
# g.set_xticklabels(rotation=90)
# plt.show()
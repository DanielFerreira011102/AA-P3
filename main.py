import copy
import string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import make_interp_spline, BSpline
from scipy.interpolate import CubicSpline

sns.set_style("whitegrid")


def main():
    work = 'republic'
    files = {
        'en': {'morris': f'results\{work}\en\morris.csv', 'fixed': f'results\{work}\en\\fixed.csv', 'cms': f'results\{work}\en\cms.csv', 'lossy': f'results\{work}\en\lossy.csv'},
        'pt': {'morris': f'results\{work}\pt\morris.csv', 'fixed': f'results\{work}\pt\\fixed.csv', 'cms': f'results\{work}\pt\cms.csv', 'lossy': f'results\{work}\pt\lossy.csv'}
    }

    # Define color palette
    palette = {'en': '#000', 'pt': '#aaa'}

    alg = 'cms'

    en_df = pd.read_csv(files['en'][alg])
    pt_df = pd.read_csv(files['pt'][alg])

    e_values = en_df['error_rate']
    p_values = en_df['probability_of_failure']
    en_total_bits_required = en_df['BR']
    pt_total_bits_required = pt_df['BR']
    en_total_bits_saved = en_df['BS']
    pt_total_bits_saved = pt_df['BS']
    en_mean_relative_error = en_df['mean_relative_error']
    pt_mean_relative_error = pt_df['mean_relative_error']

    alpha = 0.5
    en_composite_score = (alpha * en_total_bits_saved + (1 - en_mean_relative_error) * (1 - alpha))
    pt_composite_score = (alpha * pt_total_bits_saved + (1 - pt_mean_relative_error) * (1 - alpha))

    en_max_index = en_df.iloc[en_composite_score.idxmax()]
    pt_max_index = pt_df.iloc[pt_composite_score.idxmax()]
    print(f'(EN) p: {en_max_index["probability_of_failure"]}, e: {en_max_index["error_rate"]}, bs: {en_max_index["BS"]}, br: {en_max_index["BR"]}, mre: {en_max_index["mean_relative_error"]}, ce: {en_composite_score.max()}')
    print(f'(PT) p: {pt_max_index["probability_of_failure"]}, e: {pt_max_index["error_rate"]}, bs: {pt_max_index["BS"]}, br: {pt_max_index["BR"]}, mre: {pt_max_index["mean_relative_error"]}, ce: {pt_composite_score.max()}')

    en_value_s = en_composite_score
    pt_value_s = pt_composite_score
    y_label = r'$\gamma$'
    x_label = r'$\varepsilon$'
    bar_label = 'CE'
    filename = bar_label.lower()

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, layout="constrained", figsize=(14, 8))

    vmin = np.vstack([en_value_s,pt_value_s]).min()
    vmax = np.vstack([en_value_s,pt_value_s]).max()

    scatter_en = axes[0].scatter(en_df['error_rate'], en_df['probability_of_failure'], c=en_value_s, vmin=vmin, vmax=vmax, cmap='Greys', alpha=0.8)
    axes[0].set_ylabel(y_label, fontsize=20)
    # set axes tick label size
    axes[0].tick_params(axis='both', which='major', labelsize=18)

    # change position to 20 to move the label to the right
    axes[0].set_xlabel(x_label, fontsize=20)
    axes[0].xaxis.set_label_coords(1.025, -0.04)

    scatter_pt = axes[1].scatter(en_df['error_rate'], en_df['probability_of_failure'], c=pt_value_s, vmin=vmin, vmax=vmax, cmap='Greys', alpha=0.8)
    axes[1].tick_params(axis='both', which='major', labelsize=18)

    cbar = fig.colorbar(scatter_pt)
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel(bar_label, fontsize=20)

    axes[0].grid(True, linestyle='--')
    axes[1].grid(True, linestyle='--')

    plt.savefig(f'images/republic/cms/{filename}.png', format='png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

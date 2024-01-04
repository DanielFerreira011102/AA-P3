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

    en_composite_score = (
        alpha * en_total_bits_saved + (1 - en_mean_relative_error) * (1 - alpha)
    )
    pt_composite_score = (
        alpha * pt_total_bits_saved + (1 - pt_mean_relative_error) * (1 - alpha)
    )

    # Plotting
    plt.figure(figsize=(12, 8))

    value_t = p_values
    en_value_s = en_composite_score
    pt_value_s = pt_composite_score
    x_label = r'$\gamma$'
    y_label = 'CE'
    filename = 'ce_p'

    # Plot data points with alpha blending
    plt.scatter(value_t, en_value_s, color=palette['en'], alpha=0.05, edgecolors='none')
    plt.scatter(value_t, pt_value_s, color=palette['pt'], alpha=0.05, edgecolors='none')

    unique_value_t = np.unique(value_t)

    # Calculate average en_value_s and pt_value_s for each unique p_value
    en_avg_scores = [np.mean(en_value_s[value_t == val]) for val in unique_value_t]
    pt_avg_scores = [np.mean(pt_value_s[value_t == val]) for val in unique_value_t]

    # Fit a cubic spline for English
    en_spline = CubicSpline(unique_value_t, en_avg_scores)
    en_smoothed = en_spline(unique_value_t)
    plt.plot(unique_value_t, en_smoothed, label='English', color=palette['en'], marker='s')

    # Fit a cubic spline for Portuguese
    pt_spline = CubicSpline(unique_value_t, pt_avg_scores)
    pt_smoothed = pt_spline(unique_value_t)
    plt.plot(unique_value_t, pt_smoothed, label='Portuguese', color=palette['pt'], marker='s')

    max_en_value_s = np.max(en_value_s)
    max_pt_value_s = np.max(pt_value_s)

    print(f"Max composite score for English: {max_en_value_s:.4f}")
    print(f"Max composite score for Portuguese: {max_pt_value_s:.4f}")

    best_e_en = value_t[np.argmax(en_value_s)]
    best_e_pt = value_t[np.argmax(pt_value_s)]

    best_p_en = p_values[np.argmax(en_value_s)]
    best_p_pt = p_values[np.argmax(pt_value_s)]

    print(f"Best e for English: {best_e_en:.4f}")
    print(f"Best e for Portuguese: {best_e_pt:.4f}")

    print(f"Best p for English: {best_p_en:.4f}")
    print(f"Best p for Portuguese: {best_p_pt:.4f}")

    print(f"Bits saved for English with best alpha: {en_total_bits_saved[np.argmax(en_value_s)]:.4f}")
    print(f"Bits saved for Portuguese with best alpha: {pt_total_bits_saved[np.argmax(pt_value_s)]:.4f}")

    plt.axvline(x=best_e_en, color='blue', linestyle='--', label=r'Best ' + x_label + ' (English): ' + f'{best_e_en:.4f}')
    plt.axvline(x=best_e_pt, color='red', linestyle='--', label=r'Best ' + x_label + ' (Portuguese): ' + f'{best_e_pt:.4f}')

    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.grid(linestyle='--')
    plt.legend(loc='best', frameon=False, fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig(f"images/republic/cms/{filename}.png", format="png", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()

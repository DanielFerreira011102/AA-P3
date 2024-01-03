import copy
import string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

sns.set_style("whitegrid")


def main():
    work = 'republic'
    files = {
        'en': {'fixed': f'results\{work}\en\\fixed.csv'},
        'pt': {'fixed': f'results\{work}\pt\\fixed.csv'},
    }

    p = {'en': '#000', 'pt': '#aaa'}

    en_df = pd.read_csv(files['en']['fixed'])
    pt_df = pd.read_csv(files['pt']['fixed'])

    p_values = en_df['probability']
    en_total_bits_saved = en_df['BS']
    pt_total_bits_saved = pt_df['BS']
    en_mean_relative_error = en_df['mean_relative_error']
    pt_mean_relative_error = pt_df['mean_relative_error']

    scaler = MinMaxScaler()
    en_total_bits_saved_normalized = scaler.fit_transform(en_total_bits_saved.values.reshape(-1, 1))
    pt_total_bits_saved_normalized = scaler.fit_transform(pt_total_bits_saved.values.reshape(-1, 1))
    en_mean_relative_error_normalized = scaler.fit_transform(en_mean_relative_error.values.reshape(-1, 1))
    pt_mean_relative_error_normalized = scaler.fit_transform(pt_mean_relative_error.values.reshape(-1, 1))

    en_composite_score = (
        en_total_bits_saved_normalized - en_mean_relative_error_normalized
    )
    pt_composite_score = (
        pt_total_bits_saved_normalized - pt_mean_relative_error_normalized
    )
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    # color with s-
    plt.plot(p_values, en_composite_score, label='English', color=p['en'], marker='s')
    plt.plot(p_values, pt_composite_score, label='Portuguese', color=p['pt'], marker='s')#
    # line for best p for each language
# Vertical lines for the best ρ for each language
    best_rho_en = p_values[np.argmax(en_composite_score)]
    best_rho_pt = p_values[np.argmax(pt_composite_score)]

    plt.axvline(x=best_rho_en, color='blue', linestyle='--', label=r'Best $\rho$ (English):' + f'{best_rho_en:.2f}')
    plt.axvline(x=best_rho_pt, color='red', linestyle='--', label=r'Best $\rho$ (Portuguese):' + f'{best_rho_pt:.2f}')

    plt.xlabel(r"$\rho$", fontsize=20)
    plt.ylabel("BR", fontsize=20)
    plt.grid(linestyle='--')
    plt.legend(loc='best', frameon=False, fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig(f"images/republic/fixed_composite_score.png", bbox_inches='tight', format='png')
    plt.show()

if __name__ == "__main__":
    main()

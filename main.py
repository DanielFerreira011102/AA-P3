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
        'en': {'morris': f'results\{work}\en\morris.csv', 'fixed': f'results\{work}\en\\fixed.csv', 'cms': f'results\{work}\en\cms.csv', 'lossy': f'results\{work}\en\lossy.csv'},
        'pt': {'morris': f'results\{work}\pt\morris.csv', 'fixed': f'results\{work}\pt\\fixed.csv', 'cms': f'results\{work}\pt\cms.csv', 'lossy': f'results\{work}\pt\lossy.csv'}
    }

    p = {'en': '#000', 'pt': '#aaa'}

    alg = 'morris'

    en_df = pd.read_csv(files['en'][alg])
    pt_df = pd.read_csv(files['pt'][alg])

    p_values = en_df['alpha']
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
        alpha *pt_total_bits_saved + (1 - pt_mean_relative_error) * (1 - alpha)
    )

    
    plt.figure(figsize=(12, 8))
    plt.plot(p_values, en_composite_score, label='English', color=p['en'], marker='s')
    plt.plot(p_values, pt_composite_score, label='Portuguese', color=p['pt'], marker='s')#
    
    max_en_composite_score = np.max(en_composite_score)
    max_pt_composite_score = np.max(pt_composite_score)

    print(f"Max composite score for English: {max_en_composite_score:.4f}")
    print(f"Max composite score for Portuguese: {max_pt_composite_score:.4f}")
    
    best_rho_en = p_values[np.argmax(en_composite_score)]
    best_rho_pt = p_values[np.argmax(pt_composite_score)]

    print(f"Best alpha for English: {best_rho_en:.4f}")
    print(f"Best alpha for Portuguese: {best_rho_pt:.4f}")

    print(f"Bits saved for English with best alpha: {en_total_bits_saved[np.argmax(en_composite_score)]:.4f}")
    print(f"Bits saved for Portuguese with best alpha: {pt_total_bits_saved[np.argmax(pt_composite_score)]:.4f}")

    print(f"Mean relative error for English with best alpha: {en_mean_relative_error[np.argmax(en_composite_score)]:.4f}")
    print(f"Mean relative error for Portuguese with best alpha: {pt_mean_relative_error[np.argmax(pt_composite_score)]:.4f}")

    plt.axvline(x=best_rho_en, color='blue', linestyle='--', label=r'Best $\alpha$ (English): 12')
    plt.axvline(x=best_rho_pt, color='red', linestyle='--', label=r'Best $\alpha$ (Portuguese): 14')

    plt.xlabel(r"$\alpha$", fontsize=20)
    plt.ylabel("CE", fontsize=20)
    plt.grid(linestyle='--')
    plt.legend(loc='best', frameon=False, fontsize=20)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig(f"images/republic/morris/composite_score.png", bbox_inches='tight', format='png')
    plt.show()

if __name__ == "__main__":
    main()

import math
import random
import string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("whitegrid")

def main():
    work = 'republic'
    files = {
        'en': {'morris': f'results\{work}\en\morris.csv', 'fixed': f'results\{work}\en\\fixed.csv', 'cms': f'results\{work}\en\cms.csv', 'lossy': f'results\{work}\en\lossy.csv', 'exact': f'results\{work}\en\exact.csv'},
        'pt': {'morris': f'results\{work}\pt\morris.csv', 'fixed': f'results\{work}\pt\\fixed.csv', 'cms': f'results\{work}\pt\cms.csv', 'lossy': f'results\{work}\pt\lossy.csv', 'exact': f'results\{work}\pt\exact.csv'}
    }

    alg = 'cms'
    print(work)
    print(alg)


    en_df = pd.read_csv(files['en'][alg])
    pt_df = pd.read_csv(files['pt'][alg])

    en_total_bits_saved = en_df['BS']
    pt_total_bits_saved = pt_df['BS']
    en_ndcg_10 = en_df['ndcg@10']
    pt_ndcg_10 = pt_df['ndcg@10']
    en_ndcg_5 = en_df['ndcg@5']
    pt_ndcg_5 = pt_df['ndcg@5']
    en_ndcg_3 = en_df['ndcg@3']
    pt_ndcg_3 = pt_df['ndcg@3']

    # set ndcg@10, ndcg@5, ndcg@3 to 0 if BS == 1

    # set cre@10, cre@5, cre@3 to 0.5 if BS = 1

    print(en_df['cre@10'].max())
    print(en_df['cre@10'].idxmax())

    # print the full row
    print(en_df.iloc[en_df['cre@3'].idxmax()])
    print(pt_df.iloc[pt_df['cre@3'].idxmax()])

    return
    
    palette = {'en': '#000', 'pt': '#aaa'}

    en_value_s = en_ndcg_10
    pt_value_s = pt_ndcg_10
    y_label = r'$\sigma$'
    x_label = r'$\varepsilon$'
    bar_label = 'NDCG@10'
    filename = bar_label.lower()
    
    # print values that maximize NDGC@10

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, layout="constrained", figsize=(14, 8))

    vmin = np.vstack([en_value_s,pt_value_s]).min()
    vmax = np.vstack([en_value_s,pt_value_s]).max()

    scatter_en = axes[0].scatter(en_df['error_rate'], en_df['support_threshold'], c=en_value_s, vmin=vmin, vmax=vmax, cmap='Greys', alpha=0.8)
    axes[0].set_ylabel(y_label, fontsize=20)
    # set axes tick label size
    axes[0].tick_params(axis='both', which='major', labelsize=18)

    # change position to 20 to move the label to the right
    axes[0].set_xlabel(x_label, fontsize=20)
    axes[0].xaxis.set_label_coords(1.025, -0.04)

    scatter_pt = axes[1].scatter(pt_df['error_rate'], pt_df['support_threshold'], c=pt_value_s, vmin=vmin, vmax=vmax, cmap='Greys', alpha=0.8)
    axes[1].tick_params(axis='both', which='major', labelsize=18)

    cbar = fig.colorbar(scatter_pt)
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel(bar_label, fontsize=20)

    axes[0].grid(True, linestyle='--')
    axes[1].grid(True, linestyle='--')

    plt.savefig(f'images/{work}/{alg}/{filename}.png', format='png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

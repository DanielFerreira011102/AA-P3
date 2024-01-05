import random
import string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
sns.set_style("whitegrid")


def main():
    work = 'mere_christianity'
    files = {
        'en': {'morris': f'results\{work}\en\morris.csv', 'fixed': f'results\{work}\en\\fixed.csv', 'cms': f'results\{work}\en\cms.csv', 'lossy': f'results\{work}\en\lossy.csv', 'exact': f'results\{work}\en\exact.csv'},
        'pt': {'morris': f'results\{work}\pt\morris.csv', 'fixed': f'results\{work}\pt\\fixed.csv', 'cms': f'results\{work}\pt\cms.csv', 'lossy': f'results\{work}\pt\lossy.csv', 'exact': f'results\{work}\pt\exact.csv'}
    }

    en_exact_df = pd.read_csv(files['en']['exact'])
    pt_exact_df = pd.read_csv(files['pt']['exact'])

    en_exact_df['counter'] = en_exact_df['counter'].apply(eval)
    pt_exact_df['counter'] = pt_exact_df['counter'].apply(eval)
    en_exact_counter = en_exact_df['counter'][0]
    pt_exact_counter = pt_exact_df['counter'][0]

    print(en_exact_counter)

    alg = 'lossy'
    print(alg)

    en_df = pd.read_csv(files['en'][alg])
    pt_df = pd.read_csv(files['pt'][alg])

    s = 'counter' + ('_transformed' if alg in ('lossy', 'cms') else '')

    en_df[s] = en_df[s].apply(eval)
    pt_df[s] = pt_df[s].apply(eval)
    
    def ndcg(retrieved, relevant, k=10, max_trials=1000):
        def unrelevant_element():
            combined_characters = string.ascii_letters + string.digits
            for _ in range(max_trials):
                new_element = ''.join(random.choices(combined_characters, k=len(combined_characters) // 2))
                if new_element not in relevant:
                    return new_element

        relevant = list(sorted(relevant.keys(), key=lambda item: (-relevant[item], item)))
        retrieved = list(sorted(retrieved.keys(), key=lambda item: (-retrieved[item], relevant.index(item))))
        
        n_pred = len(retrieved)
        n_true = len(relevant)

        k = min(k, n_true)

        retrieved = retrieved[:k]

        if k > n_pred:
            fill_char = unrelevant_element()
            retrieved += [fill_char] * (k - n_pred)

        relevant = relevant[:len(retrieved)]

        hits = 0.0
        
        for i, item in enumerate(retrieved):
            if item == relevant[i]:
                hits += 1

        return hits / k
    
    en_df['acc@10'] = en_df[s].apply(lambda x: ndcg(x, en_exact_counter, k=10))
    pt_df['acc@10'] = pt_df[s].apply(lambda x: ndcg(x, pt_exact_counter, k=10))

    en_df['acc@5'] = en_df[s].apply(lambda x: ndcg(x, en_exact_counter, k=5))
    pt_df['acc@5'] = pt_df[s].apply(lambda x: ndcg(x, pt_exact_counter, k=5))

    en_df['acc@3'] = en_df[s].apply(lambda x: ndcg(x, en_exact_counter, k=3))
    pt_df['acc@3'] = pt_df[s].apply(lambda x: ndcg(x, pt_exact_counter, k=3))

    en_df.to_csv(files['en'][alg], index=False)
    pt_df.to_csv(files['pt'][alg], index=False)

    return

    e_values = en_df['error_rate']
    s_values = en_df['support_threshold']
    en_total_bits_required = en_df['BR']
    pt_total_bits_required = pt_df['BR']
    en_total_bits_saved = en_df['BS']
    pt_total_bits_saved = pt_df['BS']
    en_df['counter'] = en_df['counter'].apply(eval)
    pt_df['counter'] = pt_df['counter'].apply(eval)

    en_counter = en_df['counter']
    pt_counter = pt_df['counter']
    
    palette = {'en': '#000', 'pt': '#aaa'}

    en_value_s = en_total_bits_required
    pt_value_s = pt_total_bits_required
    y_label = r'$\sigma$'
    x_label = r'$\varepsilon$'
    bar_label = 'BR'
    filename = bar_label.lower()

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

    scatter_pt = axes[1].scatter(en_df['error_rate'], en_df['support_threshold'], c=pt_value_s, vmin=vmin, vmax=vmax, cmap='Greys', alpha=0.8)
    axes[1].tick_params(axis='both', which='major', labelsize=18)

    cbar = fig.colorbar(scatter_pt)
    cbar.ax.tick_params(labelsize=18)
    cbar.ax.set_ylabel(bar_label, fontsize=20)

    axes[0].grid(True, linestyle='--')
    axes[1].grid(True, linestyle='--')

    plt.savefig(f'images/republic/lossy/{filename}.png', format='png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()

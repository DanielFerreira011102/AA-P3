import copy
import string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")

def main():
    en_dir_ = "results/republic/en"
    pt_dir_ = "results/republic/pt"
    
    exact_en_df = pd.read_csv(en_dir_ + "/exact.csv")
    exact_pt_df = pd.read_csv(pt_dir_ + "/exact.csv")

    exact_en_df["lang"] = "en"
    exact_pt_df["lang"] = "pt"

    fixed_en_df = pd.read_csv(en_dir_ + "/fixed.csv")
    fixed_pt_df = pd.read_csv(pt_dir_ + "/fixed.csv")

    fixed_en_df["lang"] = "en"
    fixed_pt_df["lang"] = "pt"

    morris_en_df = pd.read_csv(en_dir_ + "/morris.csv")
    morris_pt_df = pd.read_csv(pt_dir_ + "/morris.csv")

    morris_en_df["lang"] = "en"
    morris_pt_df["lang"] = "pt"

    cms_en_df = pd.read_csv(en_dir_ + "/cms.csv")
    cms_pt_df = pd.read_csv(pt_dir_ + "/cms.csv")

    cms_en_df["lang"] = "en"
    cms_pt_df["lang"] = "pt"

    lossy_en_df = pd.read_csv(en_dir_ + "/lossy.csv")
    lossy_pt_df = pd.read_csv(pt_dir_ + "/lossy.csv")

    lossy_en_df["lang"] = "en"
    lossy_pt_df["lang"] = "pt"

    # barplot character-count for exact (en and pt in grey shades)

    exact_df = pd.concat([exact_en_df, exact_pt_df])
    exact_df["lang"] = exact_df["lang"].apply(lambda x: "English" if x == "en" else "Portuguese")
    exact_df["type"] = "Exact"

    # each row has counter column, which is a string of a dict with char: count
    # plot the count of each char in a barplot for each language
    en_counter = eval(exact_en_df["counter"][0])
    pt_counter = eval(exact_pt_df["counter"][0])

    # dicts can have different keys, so we need to merge the keys and fill the missing ones with 0
    # we also need to sort the keys
    en_keys = set(en_counter.keys())
    pt_keys = set(pt_counter.keys())
    all_keys = sorted(list(en_keys.union(pt_keys)))

    en_counts = [en_counter.get(k, 0) for k in all_keys]
    pt_counts = [pt_counter.get(k, 0) for k in all_keys]
        
    data = pd.DataFrame({'Language': ['English'] * len(en_counts) + ['Portuguese'] * len(pt_counts),
                        'Character': all_keys * 2,
                        'Count': en_counts + pt_counts})
    
    p = {"English": "#000000", "Portuguese": "#AAAAAA"}
    plt.figure(figsize=(16, 10))
    sns.barplot(y="Count", x="Character", hue="Language", data=data, palette=p)
    plt.yscale("log")
    plt.xlabel("char", fontsize=20)
    plt.ylabel("count", fontsize=20)
    plt.legend(loc="upper left", bbox_to_anchor=(0, 0.95), frameon=False, fontsize=20)
    # increase fontsize of ticks and labels and legend
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig("images/republic/exact_char_count.png", format="png", bbox_inches="tight")
    plt.show()

    # what to write in latex image (I like long descriptive captions). Specify that it is log scaled count for each character for the exact algorithm for both languages and for The Republic.
    """
    \begin{figure}[htbp]
        \centering
        \includegraphics[width=\columnwidth]{exact_char_count.png}
        \caption{Log scaled count for each character for the exact algorithm for both languages and for The Republic.}
        \label{fig:alg:exact:char_count}
    \end{figure}
    """





if __name__ == "__main__":
    main()
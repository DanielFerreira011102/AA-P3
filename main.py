import copy
import string
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_style("whitegrid")


def main():
    work = 'republic'
    ex_files = {
        'en': f'results\{work}\en\exact.csv',
        'pt': f'results\{work}\pt\exact.csv',
    }
    files = {
        'en': {'cms': f'results\{work}\en\cms.csv'},
        'pt': {'cms': f'results\{work}\pt\cms.csv'},
    }

    ex_en_df = pd.read_csv(ex_files['en'])
    ex_pt_df = pd.read_csv(ex_files['pt'])

    ex_en_br = ex_en_df['BR'][0]
    ex_pt_br = ex_pt_df['BR'][0]

    for lang in files:
        for method in files[lang]:
            filename = files[lang][method]
            df = pd.read_csv(filename)

            df['example_sketch'] = df['example_sketch'].apply(eval)
            df['example_sketch_map'] = df['example_sketch_map'].apply(eval)
            df['counter_transformed'] = df['counter_transformed'].apply(eval)

            columns_to_drop = ['TBR', 'TBR_transformed', 'PBS', 'PBS_transformed']

            df = df.drop(columns=columns_to_drop)

            df['BR'] = df['example_sketch'].apply(lambda x: sum([value.bit_length() for table in x for value in table]))
            df['BR_transformed'] = df['counter_transformed'].apply(
                lambda x: sum([value.bit_length() for value in x.values()]) + sum([len(key) * 8 for key in x.keys()]))

            # BS is (ex_en_br or ex_pt_br depending on the language - df['BR']) / ex_en_br or ex_pt_br depending on the language
            y = ex_en_br if lang == 'en' else ex_pt_br
            df['BS'] = df['BR'].apply(lambda x: (y - x) / y)
            df['BS_transformed'] = df['BR_transformed'].apply(lambda x: (y - x) / y)

            df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()

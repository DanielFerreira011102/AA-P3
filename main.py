import copy
import string
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from counter import ConcurrentCounter, CountMinSketchCounter, ExactCounter, FixedProbabilityCounter, LossyCountingCounter, MorrisCounter
from evaluator import CounterEvaluator

sns.set_style("whitegrid")

def main():
    files = [
        "works/orthodoxy/en.txt", 
        "works/orthodoxy/pt.txt", 
        "works/mere_christianity/en.txt", 
        "works/mere_christianity/pt.txt", 
        "works/republic/en.txt", 
        "works/republic/pt.txt"
    ]

    def filter_stopwords_and_punctuation(word):
        return word in allowed_chars
    
    def map_to_uppercase(word):
        return word.upper()
    
    for file in files:
        print(f"Processing {file}...")
        book, lang = file.split("/")[1], file.split("/")[2].split(".")[0]
        allowed_chars = string.ascii_letters + string.digits
        
        exact_counter = ExactCounter(file, filter=filter_stopwords_and_punctuation, map=map_to_uppercase).count().transform()

        error_rates = [0.025 * i for i in range(1, 41)]
        support_thresholds = [0.025 * i for i in range(1, 41)]
        total_bits_required = []
        scores = []
        counter_dicts = []
        counter_dicts_transformed = []
        total_bits_required_transformed = []
        new_error_rates = []
        new_support_thresholds = []
        counter_transformed_smoothed = []
        total_bits_required_transformed_smoothed = []
        smoothed_estimates = []
        T = len(support_thresholds) * len(error_rates)
        for i, (e, s) in enumerate([(e, s) for e in error_rates for s in support_thresholds]):
            print(f"Using error rate {e} and support threshold {s} ({i + 1}/{T})...")
            new_error_rates.append(e)
            new_support_thresholds.append(s)
            lossy_counter = LossyCountingCounter(file, filter=filter_stopwords_and_punctuation, map=map_to_uppercase, e=e, s=s).count()
            smoothed_estimate = {element: (lossy_counter[element] / lossy_counter._n + lossy_counter.s) for element in lossy_counter.counter}
            smoothed_estimates.append(smoothed_estimate)

            lossy_counter_2 = copy.deepcopy(lossy_counter)

            counter_dicts.append(str(dict(lossy_counter.counter)))
            total_bits_required.append(lossy_counter.total_bits_required())

            lossy_counter_t1 = lossy_counter.transform()
            counter_dicts_transformed.append(str(dict(lossy_counter_t1.counter)))
            total_bits_required_transformed.append(lossy_counter_t1.total_bits_required())

            lossy_counter_t2 = lossy_counter_2.transform(s=True)
            counter_transformed_smoothed.append(str(dict(lossy_counter_t2.counter)))
            total_bits_required_transformed_smoothed.append(lossy_counter_t2.total_bits_required())

        df = pd.DataFrame({
            "error_rate": new_error_rates,
            "support_threshold": new_support_thresholds,
            "TBR": total_bits_required,
            "TBR_transformed": total_bits_required_transformed,
            "TBR_transformed_smoothed": total_bits_required_transformed_smoothed,
            "PBS": [1 - (tbr / exact_counter.total_bits_required()) for tbr in total_bits_required],
            "PBS_transformed": [1 - (tbr / exact_counter.total_bits_required()) for tbr in total_bits_required_transformed],
            "PBS_transformed_smoothed": [1 - (tbr / exact_counter.total_bits_required()) for tbr in total_bits_required_transformed_smoothed],
            "counter": counter_dicts,
            "counter_transformed": counter_dicts_transformed,
            "counter_transformed_smoothed": counter_transformed_smoothed,
            "smoothed_estimates": smoothed_estimates
        })

        # list of dicts to dataframe
        scores_df = pd.DataFrame.from_dict(scores)
        
        df = pd.concat([df, scores_df], axis=1)

        df.to_csv(f"results/{book}/{lang}/lossy.csv", index=False)

        continue

        for i, label in enumerate([r"$RE$", r"$re$", r"$Relative Error$", r"$relative error$"]):
            plt.figure(figsize=(16, 8))
            plt.plot(alpha_values, relative_errors, 'ks-')
            plt.xlabel(r"$\alpha$")
            plt.ylabel(label)
            plt.grid(linestyle='--')
            plt.tight_layout()
            plt.savefig(f"images/{book}/{lang}/morris_re_p_alpha_{i}.png", format="png", bbox_inches="tight")
            # plt.show()
        
        for i, label in enumerate([r"$EVS$", r"$evs$", r"$Explained Variance Score$", r"$explained variance score$"]):
            plt.figure(figsize=(16, 8))
            plt.plot(alpha_values, explained_variance_scores, 'ks-')
            plt.xlabel(r"$\alpha$")
            plt.ylabel(label)
            plt.grid(linestyle='--')
            plt.tight_layout()
            plt.savefig(f"images/{book}/{lang}/morris_evs_p_alpha_{i}.png", format="png", bbox_inches="tight")
            # plt.show()

        for i, label in enumerate([r"$TBR$", r"$tbr$", r"$Total Bits Required$", r"$total bits required$"]):
            plt.figure(figsize=(16, 8))
            plt.plot(alpha_values, total_bits_required, 'ks-')
            plt.xlabel(r"$\alpha$")
            plt.ylabel(label)
            plt.grid(linestyle='--')
            plt.tight_layout()
            plt.savefig(f"images/{book}/{lang}/morris_tbr_p_alpha_{i}.png", format="png", bbox_inches="tight")
            # plt.show()

        for i, label in enumerate([r"$TBR$", r"$tbr$", r"$Total Bits Required$", r"$total bits required$"]):
            plt.figure(figsize=(16, 8))
            plt.plot(alpha_values, total_bits_required_transformed, 'ks-')
            plt.xlabel(r"$\alpha$")
            plt.ylabel(label)
            plt.grid(linestyle='--')
            plt.tight_layout()
            plt.savefig(f"images/{book}/{lang}/morris_transformed_tbr_p_alpha_{i}.png", format="png", bbox_inches="tight")
            # plt.show()

        for i, label in enumerate([r"$PBS$", r"$pbs$", r"$Percentage of Bits Saved$", r"$percentage of bits saved$"]):
            plt.figure(figsize=(16, 8))
            plt.plot(alpha_values, [1 - (tbr / exact_counter.total_bits_required()) for tbr in total_bits_required], 'ks-')
            plt.xlabel(r"$\alpha$")
            plt.ylabel(label)
            plt.grid(linestyle='--')
            plt.tight_layout()
            plt.savefig(f"images/{book}/{lang}/morris_pbs_p_alpha_{i}.png", format="png", bbox_inches="tight")
            # plt.show()

        for i, label in enumerate([r"$PBS$", r"$pbs$", r"$Percentage of Bits Saved$", r"$percentage of bits saved$"]):
            plt.figure(figsize=(16, 8))
            plt.plot(alpha_values, [1 - (total_bits_required[i] / total_bits_required_transformed[i]) for i in range(len(total_bits_required_transformed))], 'ks-')
            plt.xlabel(r"$\alpha$")
            plt.ylabel(label)
            plt.grid(linestyle='--')
            plt.tight_layout()
            plt.savefig(f"images/{book}/{lang}/morris_transformed_pbs_p_alpha_{i}.png", format="png", bbox_inches="tight")
            # plt.show()

if __name__ == "__main__":
    main()
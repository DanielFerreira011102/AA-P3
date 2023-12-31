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
        #"works/orthodoxy/en.txt", 
        #"works/orthodoxy/pt.txt", 
        #"works/mere_christianity/en.txt", 
        #"works/mere_christianity/pt.txt", 
        #"works/republic/en.txt", 
        "works/republic/pt.txt"
    ]

    allowed_chars = string.ascii_letters + string.digits

    def filter_stopwords_and_punctuation(word):
        return word in allowed_chars
    
    def map_to_uppercase(word):
        return word.upper()
    
    for file in files:
        print(f"Processing {file}...")
        book, lang = file.split("/")[1], file.split("/")[2].split(".")[0]
        
        exact_counter = ExactCounter(file, filter=filter_stopwords_and_punctuation, map=map_to_uppercase).count().transform()

        error_rates = [0.025 * i for i in range(1, 41)]
        probability_of_failures = [0.025 * i for i in range(1, 41)]
        total_bits_required = []
        scores = []
        example_sketches = []
        example_sketches_map = []
        widths = []
        depths = []
        counter_dicts_transformed = []
        total_bits_required_transformed = []
        new_error_rates = []
        new_probability_of_failures = []
        N = 10
        T = len(error_rates) * len(probability_of_failures)
        for i, (e, g) in enumerate([(e, g) for e in error_rates for g in probability_of_failures]):
            print(f"Using g = {g} and e = {e} ({i + 1}/{T})...")
            new_error_rates.append(e)
            new_probability_of_failures.append(g)
            sketches = [CountMinSketchCounter(file, filter=filter_stopwords_and_punctuation, map=map_to_uppercase, g=g, e=e, cache=True).count() for _ in range(N)]
            single_sketch = sketches[0]
            example_sketches.append(str(single_sketch._sketch))
            example_sketches_map.append(str(single_sketch._cache))
            widths.append(single_sketch.w)
            depths.append(single_sketch.d)
            total_bits_required.append(sum([sketch.total_bits_required() for sketch in sketches]) / N)
            total_bits_required_transformed.append(sum([sketch.transform().total_bits_required() for sketch in sketches]) / N)
            new_sketch = copy.deepcopy(single_sketch)
            # get all keys merged
            all_keys = set()
            for sketch in sketches:
                all_keys.update(sketch.keys())
            average_counter_dict = {key: round(sum([sketch[key] for sketch in sketches]) / N) for key in all_keys}
            counter_dicts_transformed.append(str(average_counter_dict))
            new_sketch.assign(average_counter_dict)
            evaluator = CounterEvaluator(exact_counter, new_sketch)
            scores.append(evaluator.lazy())

        df = pd.DataFrame({
            "error_rate": new_error_rates,
            "probability_of_failure": new_probability_of_failures,
            "width": widths,
            "depth": depths,
            "N": [N] * T,
            "TBR": total_bits_required,
            "TBR_transformed": total_bits_required_transformed,
            "PBS": [1 - (tbr / exact_counter.total_bits_required()) for tbr in total_bits_required],
            "PBS_transformed": [1 - (tbr / exact_counter.total_bits_required()) for tbr in total_bits_required_transformed],
            "example_sketch": example_sketches,
            "example_sketch_map": example_sketches_map,
            "counter_transformed": counter_dicts_transformed,
        })

        # list of dicts to dataframe
        scores_df = pd.DataFrame.from_dict(scores)
        
        df = pd.concat([df, scores_df], axis=1)

        df.to_csv(f"results/{book}/{lang}/cms.csv", index=False)

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
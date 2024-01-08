import math
import random
import string

class CounterEvaluator:
    """
    Class for evaluating the performance of a counter. The class contains methods for calculating the following metrics:
    - Explained variance score
    - Mean relative error
    - Mean bias error
    - Mean absolute error
    - Mean absolute percentage error
    - Mean squared error
    - Root mean squared error
    - Mean squared log error
    - Maximum error
    - Covariance
    - Correlation
    - Coefficient of determination
    - Pearson correlation index
    - Willmott index
    - Confidence index
    - Nash-Sutcliffe efficiency
    - Average precision
    - Normalized discounted cumulative gain
    - Accuracy
    - Kendall tau
    """
    def __init__(self, true_counter, pred_counter):
        """
        Initialize the evaluator.

        Parameters:
        - true_counter (Counter): True counter.
        - pred_counter (Counter): Predicted counter.
        """
        self.true_counter = true_counter
        self.pred_counter = pred_counter
        
        for key in self.true_counter:
            if key not in self.pred_counter:
                self.pred_counter[key] = 0

    def explained_variance_score(self):
        """
        Return the explained variance score between the counters.

        Returns:
        - float: Explained variance score between the counters.
        """
        y_diff_avg = sum([self.true_counter[key] - self.pred_counter[key] for key in self.true_counter]) / len(self.true_counter)
        numerator = sum([(self.true_counter[key] - self.pred_counter[key] - y_diff_avg) ** 2 for key in self.true_counter]) / len(self.true_counter)

        y_true_avg = self.true_counter.mean()
        denominator = sum([(self.true_counter[key] - y_true_avg) ** 2 for key in self.true_counter]) / len(self.true_counter)

        return 1 - numerator / denominator
    
    def mean_relative_error(self):
        """
        Return the mean relative error between the counters.

        Returns:
        - float: Mean relative error between the counters.
        """
        mre_values = [abs((self.true_counter[key] - self.pred_counter[key]) / self.true_counter[key]) for key in self.true_counter]
        mean_mre = sum(mre_values) / len(self.true_counter)
        return mean_mre
    
    def mean_bias_error(self):
        """
        Return the mean bias error between the counters.

        Returns:
        - float: Mean bias error between the counters.
        """
        mbe_values = [self.pred_counter[key] - self.true_counter[key] for key in self.true_counter]
        mean_mbe = sum(mbe_values) / len(self.true_counter)
        return mean_mbe
    
    def mean_absolute_error(self):
        """
        Return the mean absolute error between the counters.

        Returns:
        - float: Mean absolute error between the counters.
        """
        mae_values = [abs(self.true_counter[key] - self.pred_counter[key]) for key in self.true_counter]
        mean_mae = sum(mae_values) / len(self.true_counter)
        return mean_mae
    
    def mean_absolute_percentage_error(self):
        """
        Return the mean absolute percentage error between the counters.

        Returns:
        - float: Mean absolute percentage error between the counters.
        """
        mape_values = [abs((self.true_counter[key] - self.pred_counter[key]) / self.true_counter[key]) for key in self.true_counter]
        mean_mape = sum(mape_values) / len(self.true_counter)
        return mean_mape
    
    def mean_squared_error(self):
        """
        Return the mean squared error between the counters.

        Returns:
        - float: Mean squared error between the counters.
        """
        mse_values = [(self.true_counter[key] - self.pred_counter[key]) ** 2 for key in self.true_counter]
        mean_mse = sum(mse_values) / len(self.true_counter)
        return mean_mse

    def root_mean_squared_error(self):
        """
        Return the root mean squared error between the counters.

        Returns:
        - float: Root mean squared error between the counters.
        """
        rmse_values = [(self.true_counter[key] - self.pred_counter[key]) ** 2 for key in self.true_counter]
        mean_rmse = (sum(rmse_values) / len(self.true_counter)) ** 0.5
        return mean_rmse
    
    def mean_squared_log_error(self):
        """
        Return the mean squared log error between the counters.

        Returns:
        - float: Mean squared log error between the counters.
        """
        msle_values = [(math.log(self.true_counter[key] + 1) - math.log(self.pred_counter[key] + 1)) ** 2 for key in self.true_counter]
        mean_msle = sum(msle_values) / len(self.true_counter)
        return mean_msle
    
    def max_error(self):
        """
        Return the maximum error between the counters.

        Returns:
        - float: Maximum error between the counters.
        """
        max_error = max([abs(self.true_counter[key] - self.pred_counter[key]) for key in self.true_counter])
        return max_error
    
    def covariance(self):
        """
        Return the covariance between the counters.

        Returns:
        - float: Covariance between the counters.
        """
        mean_self = self.pred_counter.mean()
        mean_other = self.true_counter.mean()
        covariance = sum([(self.pred_counter[key] - mean_self) * (self.true_counter[key] - mean_other) for key in self.true_counter]) / len(self.true_counter)
        return covariance
    
    def correlation(self):
        """
        Return the correlation between the counters.

        Returns:
        - float: Correlation between the counters.
        """
        covariance = self.covariance()
        std_self = self.pred_counter.std()
        std_other = self.true_counter.std()
        correlation = covariance / (std_self * std_other) if std_self * std_other != 0 else 0
        return correlation
    
    def coefficient_of_determination(self):
        """
        Return the coefficient of determination between the counters.

        Returns:
        - float: Coefficient of determination between the counters.
        """
        numerator = sum([(self.true_counter[key] - self.pred_counter[key]) ** 2 for key in self.true_counter])
        denominator = sum([(self.true_counter[key] - self.true_counter.mean()) ** 2 for key in self.true_counter])
        coefficient_of_determination = 1 - numerator / denominator
        return coefficient_of_determination
    
    def pearson_correlation_coefficient(self):
        """
        Return the Pearson correlation index between the counters.

        Returns:
        - float: Pearson correlation index between the counters.
        """
        covariance = self.covariance()
        std_self = self.pred_counter.std()
        std_other = self.true_counter.std()
        pearson_correlation = covariance / (std_self * std_other) if std_self * std_other != 0 else 0
        return pearson_correlation

    def willmott_index(self):
        """
        Return the Willmott index between the counters.

        Returns:
        - float: Willmott index between the counters.
        """
        numerator = sum([(self.pred_counter[key] - self.true_counter[key]) ** 2 for key in self.true_counter])
        denominator = sum([(abs(self.pred_counter[key] - self.true_counter.mean()) +
                            abs(self.true_counter[key] - self.true_counter.mean())) ** 2 for key in self.true_counter])
        willmott_index = 1 - numerator / denominator
        return willmott_index

    def confidence_index(self):
        """
        Return the confidence index between the counters.

        Returns:
        - float: Confidence index between the counters.
        """
        return self.pearson_correlation_coefficient() * self.willmott_index()
    
    def nash_sutcliffe_efficiency(self):
        """
        Return the Nash-Sutcliffe efficiency between the counters.

        Returns:
        - float: Nash-Sutcliffe efficiency between the counters.
        """
        numerator = sum([(self.true_counter[key] - self.pred_counter[key]) ** 2 for key in self.true_counter])
        denominator = sum([(self.true_counter[key] - self.true_counter.mean()) ** 2 for key in self.true_counter])
        nash_sutcliffe_efficiency = 1 - numerator / denominator
        return nash_sutcliffe_efficiency
    
    def average_precision(self, k=10, max_trials=1000):
        """
        Return the average precision between the counters.

        Parameters:
        - k (int): Number of top elements to consider.
        - max_trials (int): Maximum number of trials to generate an unrelevant element, in case the retrieved list is smaller than k.

        Returns:
        - float: Average precision between the counters.
        """
        if len(set(self.true_counter.values())) == 1:
            return 0

        def unrelevant_element():
            combined_characters = string.ascii_letters + string.digits
            for _ in range(max_trials):
                new_element = ''.join(random.choices(combined_characters, k=len(combined_characters) // 2))
                if new_element not in relevant:
                    return new_element
                
        relevant = list(sorted(self.true_counter.keys(), key=lambda item: (-self.true_counter[item], item)))
        retrieved = list(sorted(self.pred_counter.keys(), key=lambda item: (-self.pred_counter[item], relevant.index(item))))

        n_pred = len(retrieved)
        n_true = len(relevant)

        k = min(k, n_true)

        retrieved = retrieved[:k]

        if k > n_pred:
            fill_char = unrelevant_element()
            retrieved += [fill_char] * (k - n_pred)

        relevant = relevant[:len(retrieved)]

        score = 0.0
        hits = 0.0

        for i, item in enumerate(retrieved):
            if item in relevant:
                hits += 1
                score += hits / (i + 1)

        return score / k
    
    def normalized_discounted_cumulative_gain(self, k=10, relevance_method='standard', max_trials=1000):
        """
        Return the normalized discounted cumulative gain between the counters.

        Parameters:
        - k (int): Number of top elements to consider.
        - relevance_method (str): Method for relevance scoring. Can be 'linear', 'inverse_log', 'inverse_rank' or 'standard'.
        - max_trials (int): Maximum number of trials to generate an unrelevant element, in case the retrieved list is smaller than k.

        Returns:
        - float: Normalized discounted cumulative gain between the counters.
        """

        if len(set(self.true_counter.values())) == 1:
            return 0

        def unrelevant_element():
            combined_characters = string.ascii_letters + string.digits
            for _ in range(max_trials):
                new_element = ''.join(random.choices(combined_characters, k=len(combined_characters) // 2))
                if new_element not in relevant:
                    return new_element

        def relevance_score(doc):
            if relevance_method == 'linear':
                return len(relevant) - relevant.index(doc)
            elif relevance_method == 'inverse_log':
                return 1 / math.log2(relevant.index(doc) + 2)
            elif relevance_method == 'inverse_rank':
                return 1 / (relevant.index(doc) + 1)
            elif relevance_method == 'standard':
                return 1
            raise ValueError("Invalid relevance scoring method")
        
        relevant = list(sorted(self.true_counter.keys(), key=lambda item: (-self.true_counter[item], item)))
        retrieved = list(sorted(self.pred_counter.keys(), key=lambda item: (-self.pred_counter[item], relevant.index(item))))
        
        n_pred = len(retrieved)
        n_true = len(relevant)

        k = min(k, n_true)

        retrieved = retrieved[:k]

        if k > n_pred:
            fill_char = unrelevant_element()
            retrieved += [fill_char] * (k - n_pred)

        relevant = relevant[:len(retrieved)]

        rels = [relevance_score(doc) if doc in relevant else 0 for doc in retrieved]

        dcg = rels[0] + sum([(rels[i]) / math.log2(i + 2) for i in range(1, len(rels))])

        ideal_rels = [relevance_score(doc) if doc in relevant else 0 for doc in relevant]
        idcg = ideal_rels[0] + sum([(ideal_rels[i]) / math.log2(i + 2) for i in range(1, len(ideal_rels))])
        
        ndcg = dcg / idcg if idcg > 0 else 0

        return ndcg
    
    def accuracy(self, k=10, max_trials=1000):
        """
        Return the accuracy between the counters.

        Parameters:
        - k (int): Number of top elements to consider.
        - max_trials (int): Maximum number of trials to generate an unrelevant element, in case the retrieved list is smaller than k.

        Returns:
        - float: Accuracy between the counters.
        """

        if len(set(self.true_counter.values())) == 1:
            return 0
        
        def unrelevant_element():
            combined_characters = string.ascii_letters + string.digits
            for _ in range(max_trials):
                new_element = ''.join(random.choices(combined_characters, k=len(combined_characters) // 2))
                if new_element not in relevant:
                    return new_element
                
        relevant = list(sorted(self.true_counter.keys(), key=lambda item: (-self.true_counter[item], item)))
        retrieved = list(sorted(self.pred_counter.keys(), key=lambda item: (-self.pred_counter[item], relevant.index(item))))
        
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

    def kendall_tau(self):
        """
        Return the Kendall tau between the counters.

        Returns:
        - float: Kendall tau between the counters.
        """
        relevant = list(sorted(self.true_counter.keys(), key=lambda item: (-self.true_counter[item], item)))
        retrieved = list(sorted(self.pred_counter.keys(), key=lambda item: (-self.pred_counter[item], relevant.index(item))))
    
        ranking = {item: i for i, item in enumerate(relevant)}

        retrieved = [ranking[doc] for doc in retrieved]
        relevant = [ranking[doc] for doc in relevant]

        n = len(retrieved)

        if n != len(relevant):
            return float('nan')

        concordant_pairs = 0
        discordant_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                rel_i, rel_j = relevant[i], relevant[j]
                ret_i, ret_j = retrieved[i], retrieved[j]

                concordant = (rel_i < rel_j and ret_i < ret_j) or (rel_i > rel_j and ret_i > ret_j)
                discordant = (rel_i < rel_j and ret_i > ret_j) or (rel_i > rel_j and ret_i < ret_j)

                if concordant:
                    concordant_pairs += 1
                elif discordant:
                    discordant_pairs += 1

        score = (concordant_pairs - discordant_pairs) / (concordant_pairs + discordant_pairs)

        return score
    
    def lazy(self):
        return {
            "explained_variance_score": self.explained_variance_score(),
            "mean_relative_error": self.mean_relative_error(),
            "mean_bias_error": self.mean_bias_error(),
            "mean_absolute_error": self.mean_absolute_error(),
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error(),
            "mean_squared_error": self.mean_squared_error(),
            "root_mean_squared_error": self.root_mean_squared_error(),
            "mean_squared_log_error": self.mean_squared_log_error(),
            "max_error": self.max_error(),
            "covariance": self.covariance(),
            "correlation": self.correlation(),
            "coefficient_of_determination": self.coefficient_of_determination(),
            "pearson_correlation_coefficient": self.pearson_correlation_coefficient(),
            "willmott_index": self.willmott_index(),
            "confidence_index": self.confidence_index(),
            "nash_sutcliffe_efficiency": self.nash_sutcliffe_efficiency(),
            "average_precision": self.average_precision(),
            "normalized_discounted_cumulative_gain": self.normalized_discounted_cumulative_gain(),
            "accuracy": self.accuracy(),
            "kendall_tau": self.kendall_tau(),
        }
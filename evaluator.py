import math

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
            "nash_sutcliffe_efficiency": self.nash_sutcliffe_efficiency()
        }
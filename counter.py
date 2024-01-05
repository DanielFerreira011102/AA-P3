from collections import defaultdict
from math import ceil, exp, log
from operator import itemgetter
import random
from evaluator import CounterEvaluator
from hash import GeneralHashFunctions
from stream import Stream
import matplotlib.pyplot as plt
import seaborn as sns


class Counter:
    """
    Counter is the base class for all counters. It serves as the foundation for implementing various streaming algorithms
    designed to efficiently estimate and track the occurrences of elements in a data stream.
    """

    def __init__(self, file_path, by=Stream.BY_CHAR, filter=None, map=None):
        """
        Initialize the Counter with the given file path and stream granularity.
        
        :param file_path: The path to the input file.
        :param by: The granularity of the stream (e.g., Stream.BY_CHAR for character-level stream).
        :param filter: The filter to be applied to each stream element.
        :param map: The map to be applied to each stream element.
        :return: The counter.
        """
        self.file_path = file_path
        self.by = by
        self.counter = defaultdict(int)
        self._filter = filter
        self._map = map

    def visualize(self, file_path=None):
        """
        Visualize the counter as a bar chart.

        :return: The counter.
        """
        plt.figure(figsize=(16, 8))
        items = dict(sorted(self.items(), key=itemgetter(0)))
        sns.barplot(x=list(items.keys()), y=list(items.values()))
        plt.grid()
        plt.tight_layout()
        if file_path is not None:
            plt.savefig(file_path, format="png", bbox_inches="tight")
        plt.show()
        return self

    def count(self):
        """
        Count the occurrences of elements in the data stream.
        
        :param func: A function that takes an element as an argument and returns a boolean. If func is provided, only elements for which func returns True will be considered.
        :return: The counter.
        """
        with Stream(self.file_path, self.by, filter=self._filter, map=self._map) as stream:
            for element in stream:
                self.update(element)
        return self

    def update(self, element):
        """
        Update the counter based on the given element in the data stream. This method is implemented by subclasses.

        :param element: The element in the data stream.
        """
        raise NotImplementedError("count() not implemented")

    def transform(self):
        """
        Transform the counter.

        :return: The counter.
        """
        return self

    def merge(self, other):
        """
        Merge the given counter with the current counter.

        :param other: The counter to merge.
        """
        self.counter.update(other.counter)

    def apply(self, func):
        """
        Apply the given function to the counter.

        :param func: A function that takes a key and value as arguments and returns a new value.
        """
        self.counter = {key: func(key, value) for key, value in self.counter.items()}

    def assign(self, dictionary):
        """
        Assign the given dictionary to the counter.

        :param dictionary: The dictionary to assign.
        """
        self.counter = dictionary

    def total_bits_required(self):
        """
        Total number of bits required to store the counter values.

        :return: The total number of bits required to store the counter values and the counter keys.
        """
        b_values = sum([value.bit_length() for value in self.values()])
        b_keys = sum([len(key) * 8 for key in self.keys()])
        return b_values + b_keys

    def filter(self, func):
        """
        Filter the counter based on the given function.

        :param func: A function that takes a key and value as arguments and returns a boolean.
        """
        self.counter = {key: value for key, value in self.counter.items() if func(key, value)}

    def clear(self):
        """
        Reset the counter.

        :return: The counter.
        """
        self.counter.clear()

    def max(self):
        """
        Maximum value in the counter.

        :return: The maximum value in the counter.
        """
        return max(self.values())

    def min(self):
        """
        Minimum value in the counter.

        :return: The minimum value in the counter.
        """
        return min(self.values())

    def sum(self):
        """
        Sum of the counter.

        :return: The sum of the counter.
        """
        return sum(self.values())

    def mean(self):
        """
        Mean of the counter.

        :return: The mean of the counter.
        """
        return sum(self.values()) / len(self)

    def median(self):
        """
        Median of the counter.

        :return: The median of the counter.
        """
        values = sorted(self.values())
        length = len(values)
        return (values[length // 2 - 1] + values[length // 2]) / 2 if length % 2 == 0 else values[length // 2]

    def variance(self):
        """
        Variance of the counter.

        :return: The variance of the counter.
        """
        mean = self.mean()
        return sum([(value - mean) ** 2 for value in self.values()]) / len(self)

    def std(self):
        """
        Standard deviation of the counter.

        :return: The standard deviation of the counter.
        """
        return self.variance() ** 0.5

    def values(self):
        """
        Values of the counter.

        :return: The values of the counter.
        """
        return self.counter.values()

    def keys(self):
        """
        Keys of the counter.

        :return: The keys of the counter.
        """
        return self.counter.keys()

    def items(self):
        """
        Items of the counter.

        :return: The items of the counter.
        """
        return self.counter.items()

    def config(self):
        """
        Configuration of the counter.

        :return: The configuration of the counter.
        """
        return {"file_path": self.file_path, "by": self.by, "filter": self._filter, "map": self._map}

    def __eq__(self, other):
        """
        Check if the counter is equal to the given counter.
        
        :param other: The counter to compare.
        :return: True if the counter is equal to the given counter, False otherwise.
        """
        return self.counter == other.counter

    def __enter__(self):
        """
        Enter the context of the counter.

        :return: The counter.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context of the counter.

        :param exc_type: The exception type.
        :param exc_value: The exception value.
        :param traceback: The traceback.
        """
        pass

    def __iter__(self):
        """
        Iterate through the counter.

        :return: The iterator of the counter.
        """
        return iter(self.counter)

    def __getitem__(self, key):
        """
        Return the value of the given key in the counter.

        :param key: The key in the counter.
        :return: The value of the given key in the counter.
        """
        return self.counter.get(key, 0)

    def __setitem__(self, key, value):
        """
        Set the value of the given key in the counter.

        :param key: The key in the counter.
        :param value: The value of the given key in the counter.
        """
        self.counter[key] = value

    def __len__(self):
        """
        Length of the counter.

        :return: The length of the counter.
        """
        return len(self.counter)

    def __contains__(self, key):
        """
        Check if the counter contains the given key.

        :param key: The key in the counter.
        :return: True if the counter contains the given key, False otherwise.
        """
        return key in self.counter

    def __str__(self):
        """
        String representation of the counter.

        :return: The string representation of the counter.
        """
        return str(self.counter)

    def __repr__(self):
        """
        Representation of the counter.

        :return: The representation of the counter.
        """
        return repr(self.counter)


class ConcurrentCounter:
    """
    ConcurrentCounter is a subclass of Counter that calculates the mean of multiple counters.
    """

    def __init__(self, file_path, by=Stream.BY_CHAR, filter=None, map=None, type=Counter, N=4, **kwargs):
        """
        Initialize ConcurrentCounter with the given file path, stream granularity, counter type, and number of counters.

        :param file_path: The path to the input file.
        :param by: The granularity of the stream (e.g., Stream.BY_CHAR for character-level stream).
        :param filter: The filter to be applied to each stream element.
        :param map: The map to be applied to each stream element.
        :param type: The type of counter to use.
        :param N: Number of counters to create for mean calculation.
        :return: The concurrent counter.
        """
        self.file_path = file_path
        self.by = by
        self._filter = filter
        self._map = map
        self.type = type
        self.N = N
        self.counter = None
        self._counter_dict = {}
        self._counters = [self.type(file_path, by, filter, map, **kwargs) for _ in range(self.N)]
        self._transformed = False
        self._transformed_t = False

    def count(self):
        """
        Count occurrences in each counter.

        :return: The concurrent counter.
        """
        for counter in self._counters:
            counter.count()
            self._counter_dict |= counter.counter
        return self

    def reduce(self):
        """
        Reduce the counters to calculate the mean.

        :return: The concurrent counter.
        """
        self.counter = self._counters[0]
        self.counter.assign(self._counter_dict)
        self.counter.apply(lambda key, _: round(sum(counter.counter[key] for counter in self._counters) / self.N))
        return self

    def transform(self):
        """
        Transform the counter.

        :return: The concurrent counter.
        """
        self.counter.transform()
        return self

    def get(self):
        """
        Return the counter.

        :return: The counter.
        """
        return self.counter

    def config(self):
        """
        Configuration of the concurrent counter.

        :return: The configuration of the concurrent counter.
        """
        return {"file_path": self.file_path, "by": self.by, "filter": self._filter, "map": self._map, "type": self.type,
                "N": self.N}


class ExactCounter(Counter):
    """
    ExactCounter counts the exact number of occurrences of each element in a data stream.
    It provides a baseline for evaluating the accuracy of other streaming algorithms.
    """

    def __init__(self, file_path, by=Stream.BY_CHAR, filter=None, map=None):
        """
        Initialize the Counter with the given file path and stream granularity.

        :param file_path: The path to the input file.
        :param by: The granularity of the stream (e.g., Stream.BY_CHAR for character-level stream).
        :param filter: The filter to be applied to each stream element.
        :param map: The map to be applied to each stream element.
        :return: The counter.
        """
        super().__init__(file_path, by, filter, map)

    def update(self, element):
        """
        Update the counter based on the given element in the data stream.

        :param element: The element in the data stream.
        """
        self.counter[element] += 1


class ApproximateCounter(Counter):
    """
    ApproximateCounter is a subclass of Counter that estimates the number of occurrences of each element in a data stream.
    """

    def __init__(self, file_path, by=Stream.BY_CHAR, filter=None, map=None):
        """
        Initialize the ApproximateCounter with the given file path and stream granularity.

        :param file_path: The path to the input file.
        :param by: The granularity of the stream (e.g., Stream.BY_CHAR for character-level stream).
        :param filter: The filter to be applied to each stream element.
        :param map: The map to be applied to each stream element.
        :return: The counter.
        """
        super().__init__(file_path, by, filter, map)
        self._transformed = False

    def transform(self):
        """
        Transform the counter to estimate the cardinality.

        :return: The counter.
        """
        if not self._transformed:
            self.apply(lambda x, _: self._estimate(x))
            self._transformed = True
        return self

    def _estimate(self, element):
        """
        Return the estimate of the given element in the counter.

        :param element: The element in the data stream.
        :return: The estimate of the given element in the counter.

        """
        return self.counter[element]

    def __getitem__(self, key):
        """
        Return the estimate of the given key in the counter.~

        :param key: The key in the counter.
        :return: The estimate of the given key in the counter.
        """
        if key not in self.counter:
            return 0
        return super().__getitem__(key) if self._transformed else self._estimate(key)


class FixedProbabilityCounter(ApproximateCounter):
    """
    FixedProbabilityCounter is a probabilistic counter that uses the Fixed Probability Algorithm. It is designed to
    estimate the cardinality (number of distinct elements) in a data stream efficiently with reduced memory requirements.
    """

    def __init__(self, file_path, by=Stream.BY_CHAR, filter=None, map=None, p=.01, seed=None):
        """
        Initialize the FixedProbabilityCounter with the given stream, stream granularity and probability.

        :param file_path: The path to the input file.
        :param by: The granularity of the stream (e.g., Stream.BY_CHAR for character-level stream).
        :param filter: The filter to be applied to each stream element.
        :param map: The map to be applied to each stream element.
        :param p: The probability of the algorithm.
        :param seed: Optional seed for controlling randomness.
        :return: The counter.
        """
        super().__init__(file_path, by, filter, map)
        self.p = p
        self._seed = seed
        random.seed(self._seed)

    def update(self, element):
        """
        Update the counter based on the given element in the data stream.

        :param element: The element in the data stream.
        """
        if random.random() <= self.p:
            self.counter[element] += 1

    def transform(self, r=True):
        """
        Transform the counter to estimate the cardinality.

        :param r: If True, round the estimate.
        :return: The counter.
        """
        if not self._transformed:
            self.apply(lambda x, _: self._estimate(x, r))
            self._transformed = True
        return self

    def config(self):
        """
        Configuration of the counter.

        :return: The configuration of the counter.
        """
        return {**super().config(), "p": self.p, "seed": self._seed}

    def _estimate(self, element, r=True):
        """
        Return the estimate of the given element in the counter.

        :param element: The element in the data stream.
        :param r: If True, round the estimate.
        :return: The estimate of the given element in the counter.
        """
        e = self.counter[element] / self.p
        return round(e) if r else e


class MorrisCounter(ApproximateCounter):
    """
    MorrisCounter is a probabilistic counter that uses the Morris Algorithm. It is designed to estimate the
    cardinality (number of distinct elements) in a data stream efficiently with reduced memory requirements.
    """

    def __init__(self, file_path, by=Stream.BY_CHAR, filter=None, map=None, a=.05, b=8, seed=None):
        """
        Initialize the MorrisCounter with the given stream, stream granularity.

        :param file_path: The path to the input file.
        :param by: The granularity of the stream (e.g., Stream.BY_CHAR for character-level stream).
        :param filter: The filter to be applied to each stream element.
        :param map: The map to be applied to each stream element.
        :param a: 
        :param seed: Optional seed for controlling randomness.
        :return: The counter.
        """
        super().__init__(file_path, by, filter, map)
        self.a = a
        self.b = b
        self._seed = seed
        random.seed(self._seed)

    def update(self, element):
        """
        Update the counter based on random values and a threshold.
    
        :param element: The element in the data stream.
        """
        if (self.counter[element] < self.b) or (random.random() < 1 / ((1 + 1 / self.a) ** self.counter[element])):
            self.counter[element] += 1

    def transform(self, r=True):
        """
        Transform the counter to estimate the cardinality.

        :param r: If True, round the estimate.
        :return: The counter.
        """
        if not self._transformed:
            self.apply(lambda x, _: self._estimate(x, r))
            self._transformed = True
        return self

    def config(self):
        """
        Configuration of the counter.

        :return: The configuration of the counter.
        """
        return {**super().config(), "a": self.a, "b": self.b, "seed": self._seed}

    def _estimate(self, element, r=True):
        """
        Return the estimate of the given element in the counter.

        :param element: The element in the data stream.
        :return: The estimate of the given element in the counter.
        """
        e = (self.a * ((1 + 1 / self.a) ** self.counter[element] - 1)) if self.counter[element] > self.b else self.counter[element]
        return round(e) if r else e


class CountMinSketchCounter(ApproximateCounter):
    """
    CountMinSketchCounter uses the Count-Min Sketch algorithm to estimate the frequencies of elements in a data stream.
    It is particularly useful for tracking heavy hitters in the stream.
    """

    def __init__(self, file_path, by=Stream.BY_CHAR, filter=None, map=None, e=0.1, g=0.1, cache=False, seed=None):
        """
        Initialize the CountMinSketchCounter with the given file path, stream granularity, error rate and probability of failure.

        :param file_path: The path to the input file.
        :param by: The granularity of the stream (e.g., Stream.BY_CHAR for character-level stream).
        :param filter: The filter to be applied to each stream element.
        :param map: The map to be applied to each stream element.
        :param e: The error rate of the algorithm.
        :param g: The probability for the error rate of the algorithm.
        :param seed: Optional seed for controlling randomness.
        :return: The counter.
        """
        super().__init__(file_path, by, filter, map)
        self.e = e
        self.g = g
        self.w = ceil(exp(1) / self.e)
        self.d = ceil(log(1 / self.g))
        self._cache = {} if cache else None
        self._sketch = [[0] * self.w for _ in range(self.d)]
        self._seed = self._generate_seed(seed)

    def update(self, element):
        """
        Update the counter based on the given element in the data stream.
        
        :param element: The element in the data stream.
        """
        if self._cache is not None and element not in self._cache:
            idx = set()
            for i in range(self.d):
                j = self._bucket(element, self._seed + i)
                self._sketch[i][j] += 1
                idx.add((i, j))
            self._cache[element] = idx
        else:
            for i in range(self.d):
                self._sketch[i][self._bucket(element, self._seed + i)] += 1

    def transform(self):
        """
        Transform the counter to estimate the cardinality.

        :return: The counter.
        """
        if not self._transformed:
            if self._cache:
                self.assign({element: self._estimate(element) for element in self._cache.keys()})
            else:
                with Stream(self.file_path, self.by) as stream:
                    for element in stream:
                        self.counter[element] = self._estimate(element)
            self._transformed = True
        return self
    
    def total_bits_required(self):
        """
        Total number of bits required to store the counter values.

        :return: The total number of bits required to store the counter values.
        """
        if self._transformed:
            return super().total_bits_required()
        return sum([sum([value.bit_length() for value in table]) for table in self._sketch])

    def config(self):
        """
        Configuration of the counter.

        :return: The configuration of the counter.
        """
        return {**super().config(), "e": self.e, "g": self.g, "cache": self._cache is not None, "seed": self._seed}

    def _bucket(self, element, table):
        """
        Return the bucket of the given element in the given hash table.

        :param element: The element in the data stream.
        :param table: The hash table.
        :return: The bucket of the given element in the given hash table.
        """
        return GeneralHashFunctions.murmurhash(element, b=32, seed=self._seed + table) % self.w

    def _estimate(self, element):
        """
        Return the estimate of the given element in the data stream.
        
        :param element: The element in the data stream.
        :return: The estimate of the given element in the data stream.
        """
        return min([self._sketch[i][self._bucket(element, self._seed + i)] for i in range(self.d)], default=0)
    
    def _generate_seed(self, seed):
        """
        Generate a seed for the hash function.
        
        :param seed: The seed for the hash function.
        :return: A seed for the hash function.
        """
        if seed is not None:
            if isinstance(seed, int):
                return seed
            return hash(seed) % (2 ** 32 - 1)
        return random.randint(0, 2 ** 32 - 1)


class LossyCountingCounter(ApproximateCounter):
    """
    LossyCountingCounter uses the Lossy Counting algorithm to estimate the frequencies of elements in a data stream.
    It is designed to identify and track elements with frequencies above a certain threshold.
    """

    def __init__(self, file_path, by=Stream.BY_CHAR, filter=None, map=None, e=0.001, s=0.001):
        """
        Initialize the LossyCountingCounter with the given file path, stream granularity, error rate and support threshold.
        
        :param file_path: The path to the input file.
        :param by: The granularity of the stream (e.g., Stream.BY_CHAR for character-level stream).
        :param filter: The filter to be applied to each stream element.
        :param map: The map to be applied to each stream element.
        :param e: The error rate of the algorithm.
        :param s: The support threshold of the algorithm.
        :return: The counter.
        """
        super().__init__(file_path, by, filter, map)
        self.e = e
        self.s = s
        self._w = ceil(1 / self.e)
        self._n = 0
        self._bucket = {}
        self._bucket_id = 1

    def update(self, element):
        """
        Update the counter based on the given element in the data stream.
        
        :param element: The element in the data stream.
        """
        self._n += 1

        if element not in self.counter:
            self._bucket[element] = self._bucket_id - 1
        self.counter[element] += 1

        if self._n % self._w == 0:
            self._prune()
            self._bucket_id += 1

    def transform(self, t=None, s=False, r=True):
        """
        Transform the counter to estimate the cardinality.

        :param t: The threshold for the support of an element. If t is None, use the support threshold of the algorithm.
        :param s: If True, return the smoothed estimate; otherwise, return the raw estimate.
        :return: The counter.
        """
        if not self._transformed:
            t = t if t is not None else self.s
            self.filter(lambda _, value: value >= (t - self.e) * self._n)
            self.apply(lambda x, _: self._estimate(x, s, r))
            self._transformed = True
        return self
    
    def top(self, k=10, t=None, s=False, r=True):
        """
        Return the top k elements in the counter.

        :param k: The number of elements to return.
        :param t: The threshold for the support of an element. If t is None, use the support threshold of the algorithm.
        :param s: If True, return the smoothed estimate; otherwise, return the raw estimate.
        :param r: If True, round the estimate.
        :return: The top k elements in the counter.
        """
        if self._transformed:
            res = self.counter.items()
        else:
            t = t if t is not None else self.s
            res = filter(lambda x: x[1] >= (t - self.e) * self._n, self.counter.items())
            res = map(lambda x: (x[0], self._estimate(x[0], s, r)), res)
        return sorted(res, key=itemgetter(1), reverse=True)[:k]

    def config(self):
        """
        Configuration of the counter.

        :return: The configuration of the counter.
        """
        return {**super().config(), "e": self.e, "s": self.s}

    def _prune(self):
        """
        Prune the counter.
        """
        prune = set()
        for element, count in self.counter.items():
            if count + self._bucket[element] <= self._bucket_id:
                prune.add(element)
        for element in prune:
            del self.counter[element]
            del self._bucket[element]

    def _estimate(self, element, s=False, r=True):
        """
        Return the estimate of the given element in the data stream.
        
        :param element: The element in the data stream.
        :param s: If True, return the smoothed estimate; otherwise, return the raw estimate.
        :param r: If True, round the estimate.
        :return: The estimate of the given element in the data stream.
        """
        e = self.counter[element] if not s else (self.counter[element] * (self.counter[element] / self._n + self.s))
        return round(e) if r else e


# HyperLogLog Counter
# LogLog Counter
# Randomized Counter
# Space-Saving Counter
# Frequent Counting Counter
# Rapid-Update Counter  
# Misra-Gries Counter 

def main():
    file = "works/test.txt"

    exact_counter = ExactCounter(file).count().transform()  # .visualize(file_path="images/test/exact_counter.png")
    morris_counter = MorrisCounter(file).count().transform()  # .visualize(file_path="images/test/morris_counter.png")
    count_min_sketch_counter = CountMinSketchCounter(file, w=50, d=10).count().transform()  # .visualize(file_path="images/test/count_min_sketch_counter.png")
    lossy_counting_counter = LossyCountingCounter(file).count().transform()  # .visualize(file_path="images/test/lossy_counting_counter.png")

    morris_evaluator = CounterEvaluator(exact_counter, morris_counter)
    count_min_sketch_evaluator = CounterEvaluator(exact_counter, count_min_sketch_counter)
    lossy_counting_evaluator = CounterEvaluator(exact_counter, lossy_counting_counter)

    print(lossy_counting_counter.config())

    print(morris_evaluator.explained_variance_score())
    print("EVS", count_min_sketch_evaluator.explained_variance_score())
    print("MRE",count_min_sketch_evaluator.mean_relative_error())
    print(lossy_counting_evaluator.explained_variance_score())


if __name__ == "__main__":
    main()

from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.width', None)

print("pandas version-" + pd.__version__)
example_Series_city_names = pd.Series(["San Francisco", "San Jose", "Sacramento"])
example_Series_population = pd.Series([852469, 1015785, 485199])

example_DataFrame = pd.DataFrame({"City name": example_Series_city_names, "Population": example_Series_population})

example_DataFrame_from_file = pd.read_csv("california_housing_train.csv")
# print(example_DataFrame_from_file.describe())
# print(example_DataFrame_from_file.plot.hist('housing_median_age'))
# plt.show()

print(example_DataFrame["Population"])
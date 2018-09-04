from __future__ import print_function
import pandas as pd
print(pd.__version__)
example_Series_city_names = pd.Series(["San Francisco", "San Jose", "Sacramento"])
example_Series_population = pd.Series([852469, 1015785, 485199])

example_DataFrame = pd.DataFrame({"City name": example_Series_city_names, "Population": example_Series_population})
print(example_DataFrame)
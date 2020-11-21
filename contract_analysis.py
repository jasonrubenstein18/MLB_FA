import pandas as pd
import numpy as np
import time

fg_data_full = fg_data_full
print(len(fg_data_full))
salary_data = salary_data
print(len(salary_data))

merged_data = pd.merge(fg_data_full, salary_data, left_on=['Name', 'Year'], right_on=['Player', 'Season'])
print(len(merged_data))

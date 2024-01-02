import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.swbm import seasonal_sinus

# Generating date range (for example, one year)
start_date = '2023-01-01'
end_date = '2024-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

plt.figure(figsize=(12, 6))

# Plotting the curve
plt.plot(date_range, seasonal_sinus(len(date_range)))

plt.xlabel('Date')
plt.ylabel('Value')
plt.tight_layout()
plt.grid(True)
plt.show()


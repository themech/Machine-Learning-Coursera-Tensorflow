import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read the data - two comma-separated columns with no header. We name the columns here
df = pd.read_csv('data/ex1data1.txt', names=['population', 'profit'])

# Print the header and first few rows
print df.head()

# Plot the data
sns.lmplot('population', 'profit', df, size=6, fit_reg=False)
plt.show()

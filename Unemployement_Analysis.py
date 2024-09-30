import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
df = pd.read_csv('Unemployment in India.csv')

# Displaying the first few rows of the dataset
print(df.head())

# Checking for missing values
print(df.isnull().sum())

# Getting basic statistics
print(df.describe())

# Checking data types of each column
print(df.dtypes)

# Strip any leading/trailing spaces or hidden characters in column names
df.columns = df.columns.str.strip()

# converting 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'],dayfirst=True, errors='coerce')

# Plotting unemployment rate over time
plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=df)
plt.title('Unemployment Rate Over Time in India')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()

# Plotting unemployment rate by region
plt.figure(figsize=(10,6))
sns.barplot(x='Region', y='Estimated Unemployment Rate (%)', data=df)
plt.title('Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.show()



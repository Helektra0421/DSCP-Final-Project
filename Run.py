import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 
                'acceleration', 'model_year', 'origin', 'car_name']

df = pd.read_csv(url, names=column_names, sep=r'\s+', na_values="?", comment='\t')

#2. Data Pre-Processing
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

#4. Analysis 1: Fuel Efficiency Over the Years
plt.figure(figsize=(10, 6))
yearly_avg = df.groupby('model_year')['mpg'].mean().reset_index()
sns.lineplot(x='model_year', y='mpg', data=yearly_avg, marker='o', color='b')
plt.title('Average MPG by Model Year (1970-1982)')
plt.xlabel('Model Year')
plt.ylabel('Average MPG')
plt.grid(True)
plt.show()

#5. Analysis 2: Weight vs. MPG
plt.figure(figsize=(10, 6))
sns.scatterplot(x='weight', y='mpg', data=df, alpha=0.6, color='r')
plt.title('Vehicle Weight vs. MPG')
plt.xlabel('Weight (lbs)')
plt.ylabel('MPG')
plt.grid(True)
plt.show()
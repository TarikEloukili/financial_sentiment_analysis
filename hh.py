import pandas as pd

# Load the dataset
df = pd.read_csv("data.csv")

# Drop the column (replace 'column_to_drop' with the actual column name)
df.drop(columns=["Sentiment"], inplace=True)

# Save the updated dataset to a new CSV file
df.to_csv("data3.csv", index=False)

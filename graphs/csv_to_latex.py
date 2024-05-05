import pandas as pd
import os

print(os.getcwd())
# Read the CSV file
df = pd.read_csv('./log/adult_accuracy_20%.csv')



substring_to_remove = 'accuracy score'  # Replace with the actual substring you want to remove
df.columns = df.columns.str.replace(substring_to_remove, '')

# Define a formatting function for floats
float_format = "{:0.4f}".format

# Convert the DataFrame to a LaTeX table
latex = df.to_latex(index=False, float_format=float_format)

# Print the LaTeX table
print(latex)
from textwrap import wrap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

# Read the CSV file
db = pd.read_csv('./log/experiment_german_short_titles.csv', index_col=0)

# Function to check if a string represents a float
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# Filter the DataFrame
model_names = ['LG','RF','XGB']
print(model_names)

metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for metric in metrics:
    df = db[[db.columns[0]] + [col for col in db.columns[1:] if db[col][0] == metric]]
    df = df.iloc[1:]
    df = df.applymap(lambda x: round(float(x), 4) if isinstance(x, str) and is_float(x) else x)
    df.columns = ['\n'.join(wrap(col, 13)) for col in df.columns]
    max_values = df.idxmax(axis=1)
    print(max_values)


    # Create a new figure
    fig, ax = plt.subplots(1, 1)

    # Hide axes
    ax.axis('off')

    # Create the table
    table = plt.table(cellText=df.values, colLabels=df.columns,rowLabels=model_names, cellLoc = 'center', loc='center')
    # Color each cell with the maximum value in its row green
    for i in range(len(df)):
        max_col = df.iloc[i].idxmax()
        print(max_col)
        table.get_celld()[(i+1, df.columns.get_loc(max_col))].set_facecolor('#90ee90')


    # Auto size the columns
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.savefig(f'./graphs/german_{metric}.png', bbox_inches='tight', dpi=300)
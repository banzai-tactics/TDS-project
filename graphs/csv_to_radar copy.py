import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Assuming df is your DataFrame
# df = pd.DataFrame({
#     'group': ['A','B','C'],
#     'var1': [38, 1.5, 30],
#     'var2': [29, 10, 9],
#     'var3': [8, 39, 23],
#     'var4': [7, 31, 33],
#     'var5': [28, 15, 32]
# })

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
db = pd.read_csv('./log/experiment_german_short_titles.csv', index_col=0)

df = db[[col for col in db.columns[1:] if db[col][0] == "f1"]]
df = df.iloc[1:]
df = df.applymap(lambda x: round(float(x), 4) if isinstance(x, str) and is_float(x) else x)
print(df.columns)
# print(df.T.columns)
# ------- PART 1: Create background

# number of variable
categories=list(df)[1:]
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7)
plt.ylim(0,40)

# ------- PART 2: Add plots

# Plot each individual = each line of the data
for i, row in df.iterrows():
    values = row.drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="group " + row['group'])
    ax.fill(angles, values, alpha=0.1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.show()
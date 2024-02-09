import pandas as pd

df_census = pd.read_csv("../Datasets/census_2000_2010.csv")
print(df_census.head(5))
census_2010 = df_census["pop.2010"]

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2

def Benford_Analysis(Count, EP, DIGITS, x_label=""):
    # Fill in missing digits with zero
    missing_digit = np.setdiff1d(DIGITS, np.array(Count.index, dtype=int))
    if len(missing_digit) > 0:
        Count = pd.concat([Count, pd.Series(0, index=missing_digit)])
        Count = Count.sort_index()

    # Remove any additional digit in Count not in the group DIGITS
    Count = Count[DIGITS]

    AP = Count / Count.sum()
    N = Count.sum()
    folded_z = np.abs(AP - EP) / np.sqrt(EP * (1 - EP) / N)
    p_val = 2 * norm.pdf(folded_z)  # p-values for folded z-scores
    print(DIGITS[p_val < 0.05])   # Rejected digits by Benford"s law

    # Build a bar chart for each digit
    colors = np.repeat("burlywood", len(DIGITS))
    colors[p_val < 0.05] = "red"
    bar_name = [str(d).zfill(len(str(DIGITS[-1]))) for d in DIGITS]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(bar_name, AP, color=colors, edgecolor="none", label="Actual")
    ax.plot(bar_name, EP, color="blue", linewidth=2, label="Benford's Law")
    ax.set_xlabel(x_label, fontsize=15)
    ax.set_ylabel("PROPORTION", fontsize=15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    
    handles, _ = ax.get_legend_handles_labels()
    labels = ["Actual", "Rejected", "Benford's Law"]
    handles = [plt.Rectangle((0,0),1,1, color="burlywood"),
               plt.Rectangle((0,0),1,1, color="red"),
               handles[0]]
    plt.legend(handles, labels, loc="upper right")
    
    fig.tight_layout()
    fig.savefig(f"../Picture/{x_label}.png", dpi=200)
    
    # Return the p-value of chi-squared goodness-of-fit test statistics
    return chi2.pdf(np.sum(N*(AP - EP)**2 / EP), len(DIGITS)-1)

#%%
# FIRST DIGIT
FT_1 = [int(str(d)[0]) for d in census_2010]         # Get first digit
Count_1 = pd.Series(FT_1).value_counts().sort_index()# A summary table
DIGITS_1 = np.arange(1, 10)                  # Possible first digit values
EP_1 = np.log10(1 + 1 / DIGITS_1)                    # Expected proportion
print(Benford_Analysis(Count_1, EP_1, DIGITS_1, "FIRST DIGIT"))

#%%
# SECOND DIGIT
FT_2 = [int(str(d)[1]) for d in census_2010]         # Get second digit
Count_2 = pd.Series(FT_2).value_counts().sort_index()# A summary table
DIGITS_2 = np.arange(10)               # Possible second digit values
EP_2 = np.zeros_like(DIGITS_2, dtype=float)          # Expected proportion
for i in range(1, 10):
    EP_2 += np.log10(1 + 1 / np.arange(10 * i, 10 * i + 10))
    
print(Benford_Analysis(Count_2, EP_2, DIGITS_2, "SECOND DIGIT"))

#%%
# FIRST-TWO DIGITS
FT_3 = [int(str(d)[:2]) for d in census_2010]    # Get first-two digits
Count_3 = pd.Series(FT_3).value_counts().sort_index()# A summary table
DIGITS_3 = np.arange(10, 100)        # Possible first-two digits values
EP_3 = np.log10(1 + 1 / DIGITS_3)                    # Expected proportion
print(Benford_Analysis(Count_3, EP_3, DIGITS_3, "FIRST-TWO DIGITS"))

#%%
# FIRST-THREE DIGITS
FT_4 = [int(str(d)[:3]) for d in census_2010]    # Get first-three digits
Count_4 = pd.Series(FT_4).value_counts().sort_index()# A summary table
DIGITS_4 = np.arange(100, 1000)      # Possible first-three digits values
EP_4 = np.log10(1 + 1 / DIGITS_4)                    # Expected proportion
print(Benford_Analysis(Count_4, EP_4, DIGITS_4, "FIRST-THREE DIGITS"))

#%%
# LAST-TWO DIGITS
print(pd.Series([len(str(d)) for d in census_2010]
                ).value_counts().sort_index())   # Table for digit length

census_4_digit = [d for d in census_2010 if len(str(d)) >= 4]
FT_5 = [int(str(d)[-2:]) for d in census_4_digit]    # Get last-two digits
Count_5 = pd.Series(FT_5).value_counts().sort_index()# A summary table
DIGITS_5 = np.arange(100)                # Possible last-two digits values
EP_5 = np.ones_like(DIGITS_5) / len(DIGITS_5)        # Expected proportion
print(Benford_Analysis(Count_5, EP_5, DIGITS_5, "LAST-TWO DIGITS"))
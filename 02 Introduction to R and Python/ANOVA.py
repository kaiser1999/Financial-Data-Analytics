import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', size=20); plt.rc('axes', titlesize=30, labelsize=30)
plt.rc('xtick', labelsize=25); plt.rc('ytick', labelsize=25)

#%%
SPX = pd.read_csv("../Datasets/SPX.csv", index_col=0)
HSI = pd.read_csv("../Datasets/HSI.csv", index_col=0)
FTSE = pd.read_csv("../Datasets/FTSE.csv", index_col=0)

SPX["Year"] = pd.to_datetime(SPX.index.values, format='%d/%m/%Y').year
HSI["Year"] = pd.to_datetime(HSI.index.values, format='%d/%m/%Y').year
FTSE["Year"] = pd.to_datetime(FTSE.index.values, format='%d/%m/%Y').year

SPX["log_return"] = np.log(SPX.Price) - np.log(SPX.Price.shift(1))
HSI["log_return"] = np.log(HSI.Price) - np.log(HSI.Price.shift(1))
FTSE["log_return"] = np.log(FTSE.Price) - np.log(FTSE.Price.shift(1))

SPX, HSI, FTSE = SPX.iloc[1:], HSI.iloc[1:], FTSE.iloc[1:]

def Remove_Outlier(index, outlier_factor=1.5):
    q25 = np.quantile(index.log_return, q=0.25)
    q75 = np.quantile(index.log_return, q=0.75)
    iqr = q75 - q25    # Inter-quartile range
    lower_bound = q25 - outlier_factor*iqr
    upper_bound = q75 + outlier_factor*iqr
    boolean = (lower_bound<index.log_return) & (index.log_return<upper_bound)
    return index[boolean]

# Hyperparameter
outlier_removal = True
if outlier_removal:
    SPX = Remove_Outlier(SPX)
    HSI = Remove_Outlier(HSI)
    FTSE = Remove_Outlier(FTSE)

#%%
### from 2005 to 2007 ###
#%%
start_year = 2005
end_year = 2007
year_name = " Year: " + str(start_year) + "-" + str(end_year)
Chosen_Year = range(start_year, end_year + 1)
Chosen_SPX = SPX[SPX.Year.isin(Chosen_Year)]
Chosen_HSI = HSI[HSI.Year.isin(Chosen_Year)]
Chosen_FTSE = FTSE[FTSE.Year.isin(Chosen_Year)]

for idx, Chosen_idx in zip(["SPX", "HSI", "FTSE"], 
                           [Chosen_SPX, Chosen_HSI, Chosen_FTSE]):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.set_title(f"{idx} daily log-return." + year_name)
    ax.hist(Chosen_idx.log_return, bins="sturges", ec='black')
    
    fig.tight_layout()
    fig.savefig(f"../Picture/{idx} log_return histogram {start_year}_{end_year}.png", dpi=200)

#%%
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison

fig, ax = plt.subplots(1, 1, figsize=(11,10))
plt.subplots_adjust(wspace=0.4)      # horizontal spacing between plots
ax.set_title("Boxplot." + year_name)
ax.set_xlabel("Location")
ax.set_ylabel("Daily log-return")
ax.boxplot([Chosen_SPX.log_return, Chosen_HSI.log_return, 
            Chosen_FTSE.log_return],
           widths=0.7, labels=["SPX", "HSI", "FTSE"])

fig.tight_layout()
fig.savefig(f"../Picture/Boxplot {start_year}_{end_year}.png", dpi=200)

Chosen_SPX.loc[:, "Index"] = "SPX"
Chosen_HSI.loc[:, "Index"] = "HSI"
Chosen_FTSE.loc[:, "Index"] = "FTSE"
AllIndex = pd.concat([Chosen_SPX, Chosen_HSI, Chosen_FTSE])

fig, ax = plt.subplots(1, 1, figsize=(11,10))
sns.pointplot(x='Index', y="log_return", data=AllIndex, ci=95, ax=ax)
ax.set_title("Mean Plot with 95% CI." + year_name)
ax.set_xlabel("Index")
ax.set_ylabel("Daily log-return")

fig.tight_layout()
fig.savefig(f"../Picture/Meanplot {start_year}_{end_year}.png", dpi=200)

import statsmodels.api as sm
from statsmodels.formula.api import ols

index_lm = ols('log_return ~ Index', data=AllIndex).fit()
print(sm.stats.anova_lm(index_lm, typ=2))

comp = MultiComparison(data=AllIndex.log_return, groups=AllIndex.Index, 
                       group_order=["SPX", "HSI", "FTSE"])
TurkeyHSD_result = comp.tukeyhsd()
print(TurkeyHSD_result)

fig, ax = plt.subplots(1, 1, figsize=(11,10))
TurkeyHSD_result.plot_simultaneous(figsize=(11, 10), ax=ax)
ax.set_xlabel("Daily log-return")
ax.set_ylabel("Index")

fig.tight_layout()
fig.savefig(f"../Picture/Tukey HSD {start_year}_{end_year}.png", dpi=200)

#%%
mu_SPX = np.mean(Chosen_SPX.log_return)
mu_HSI = np.mean(Chosen_HSI.log_return)
mu_FTSE = np.mean(Chosen_FTSE.log_return)

SSE_SPX = np.sum((Chosen_SPX.log_return - mu_SPX)**2)
SSE_HSI = np.sum((Chosen_HSI.log_return - mu_HSI)**2)
SSE_FTSE = np.sum((Chosen_FTSE.log_return - mu_FTSE)**2)

n_SPX = len(Chosen_SPX.log_return)
n_HSI = len(Chosen_HSI.log_return)
n_FTSE = len(Chosen_FTSE.log_return)

df = len(AllIndex.log_return) - 3
print(df)

Within_group_MSE = (SSE_SPX + SSE_HSI + SSE_FTSE) / df
print(Within_group_MSE)

SPX_HSI_SE_ANOVA = np.sqrt(Within_group_MSE * (1/n_SPX + 1/n_HSI))
FTSE_SPX_SE_ANOVA = np.sqrt(Within_group_MSE * (1/n_FTSE + 1/n_SPX))
FTSE_HSI_SE_ANOVA = np.sqrt(Within_group_MSE * (1/n_FTSE + 1/n_HSI))

# q_Tukey = sqrt(2) t
SPX_vs_HSI = np.abs(mu_SPX - mu_HSI) / SPX_HSI_SE_ANOVA * np.sqrt(2)
FTSE_vs_SPX = np.abs(mu_FTSE - mu_SPX) / FTSE_SPX_SE_ANOVA * np.sqrt(2)
FTSE_vs_HSI = np.abs(mu_FTSE - mu_HSI) / FTSE_HSI_SE_ANOVA * np.sqrt(2)

from statsmodels.stats.libqsturng import psturng

print(psturng(q=SPX_vs_HSI, r=3, v=df))
print(psturng(q=FTSE_vs_SPX, r=3, v=df))
print(psturng(q=FTSE_vs_HSI, r=3, v=df))

#%%
### from 2018 to 2020 ###
#%%
start_year = 2018
end_year = 2020
year_name = " Year: " + str(start_year) + "-" + str(end_year)
Chosen_Year = range(start_year, end_year + 1)
Chosen_SPX = SPX[SPX.Year.isin(Chosen_Year)]
Chosen_HSI = HSI[HSI.Year.isin(Chosen_Year)]
Chosen_FTSE = FTSE[FTSE.Year.isin(Chosen_Year)]

for idx, Chosen_idx in zip(["SPX", "HSI", "FTSE"], 
                           [Chosen_SPX, Chosen_HSI, Chosen_FTSE]):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    ax.set_title(f"{idx} daily log-return." + year_name)
    ax.hist(Chosen_idx.log_return, bins="sturges", ec='black')
    
    fig.tight_layout()
    fig.savefig(f"../Picture/{idx} log_return histogram {start_year}_{end_year}.png", dpi=200)

#%%
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison

fig, ax = plt.subplots(1, 1, figsize=(11,10))
plt.subplots_adjust(wspace=0.4)      # horizontal spacing between plots
ax.set_title("Boxplot." + year_name)
ax.set_xlabel("Location")
ax.set_ylabel("Daily log-return")
ax.boxplot([Chosen_SPX.log_return, Chosen_HSI.log_return, Chosen_FTSE.log_return],
           widths=0.7, labels=["SPX", "HSI", "FTSE"])

fig.tight_layout()
fig.savefig(f"../Picture/Boxplot {start_year}_{end_year}.png", dpi=200)

Chosen_SPX.loc[:, "Index"] = "SPX"
Chosen_HSI.loc[:, "Index"] = "HSI"
Chosen_FTSE.loc[:, "Index"] = "FTSE"
AllIndex = pd.concat([Chosen_SPX, Chosen_HSI, Chosen_FTSE])

fig, ax = plt.subplots(1, 1, figsize=(11,10))
sns.pointplot(x='Index', y="log_return", data=AllIndex, ci=95, ax=ax)
ax.set_title("Mean Plot with 95% CI." + year_name)
ax.set_xlabel("Index")
ax.set_ylabel("Daily log-return")

fig.tight_layout()
fig.savefig(f"../Picture/Meanplot {start_year}_{end_year}.png", dpi=200)

import statsmodels.api as sm
from statsmodels.formula.api import ols

index_lm = ols('log_return ~ Index', data=AllIndex).fit()
print(sm.stats.anova_lm(index_lm, typ=2))

comp = MultiComparison(data=AllIndex.log_return, groups=AllIndex.Index, 
                       group_order=["SPX", "HSI", "FTSE"])
TurkeyHSD_result = comp.tukeyhsd()
print(TurkeyHSD_result)

fig, ax = plt.subplots(1, 1, figsize=(11,10))
TurkeyHSD_result.plot_simultaneous(figsize=(11, 10), ax=ax)
ax.set_xlabel("Daily log-return")
ax.set_ylabel("Index")

fig.tight_layout()
fig.savefig(f"../Picture/Tukey HSD {start_year}_{end_year}.png", dpi=200)

#%%
mu_SPX = np.mean(Chosen_SPX.log_return)
mu_HSI = np.mean(Chosen_HSI.log_return)
mu_FTSE = np.mean(Chosen_FTSE.log_return)

SSE_SPX = np.sum((Chosen_SPX.log_return - mu_SPX)**2)
SSE_HSI = np.sum((Chosen_HSI.log_return - mu_HSI)**2)
SSE_FTSE = np.sum((Chosen_FTSE.log_return - mu_FTSE)**2)

n_SPX = len(Chosen_SPX.log_return)
n_HSI = len(Chosen_HSI.log_return)
n_FTSE = len(Chosen_FTSE.log_return)

df = len(AllIndex.log_return) - 3
print(df)

Within_group_MSE = (SSE_SPX + SSE_HSI + SSE_FTSE) / df
print(Within_group_MSE)

SPX_HSI_SE_ANOVA = np.sqrt(Within_group_MSE * (1/n_SPX + 1/n_HSI))
FTSE_SPX_SE_ANOVA = np.sqrt(Within_group_MSE * (1/n_FTSE + 1/n_SPX))
FTSE_HSI_SE_ANOVA = np.sqrt(Within_group_MSE * (1/n_FTSE + 1/n_HSI))

# q_Tukey = sqrt(2) t
SPX_vs_HSI = np.abs(mu_SPX - mu_HSI) / SPX_HSI_SE_ANOVA * np.sqrt(2)
FTSE_vs_SPX = np.abs(mu_FTSE - mu_SPX) / FTSE_SPX_SE_ANOVA * np.sqrt(2)
FTSE_vs_HSI = np.abs(mu_FTSE - mu_HSI) / FTSE_HSI_SE_ANOVA * np.sqrt(2)

from statsmodels.stats.libqsturng import psturng

print(psturng(q=SPX_vs_HSI, r=3, v=df))
print(psturng(q=FTSE_vs_SPX, r=3, v=df))
print(psturng(q=FTSE_vs_HSI, r=3, v=df))
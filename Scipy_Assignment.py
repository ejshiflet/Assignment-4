import pandas as pd
import numpy as np
from scipy import stats

# ====================================================================================================================

# Load Dataset

file_path = "players_stats_by_season_full_details.csv"
df = pd.read_csv(file_path)

# ====================================================================================================================

# Filter NBA Regular Season

regular_df = df[
    (df['League'] == 'NBA') &
    (df['Stage'] == 'Regular_Season')
]

# ====================================================================================================================

# Player With Most Seasons

season_counts = regular_df.groupby('Player')['Season'].nunique()
top_player = season_counts.idxmax()
top_seasons = season_counts.max()

print("Player with most seasons:", top_player)
print("Number of seasons:", top_seasons)

# ====================================================================================================================

# 3PT Accuracy Per Season

player_df = regular_df[
    regular_df['Player'] == top_player
].copy()

player_df = player_df.sort_values('Season')

player_df['ThreePtPct'] = player_df['3PM'] / player_df['3PA']
player_df.replace([np.inf, -np.inf], np.nan, inplace=True)
player_df.dropna(subset=['ThreePtPct'], inplace=True)

print("\nThree Point Accuracy by Season:")
print(player_df[['Season', 'ThreePtPct']])

# ====================================================================================================================

# Linear Regression

player_df['SeasonIndex'] = pd.factorize(player_df['Season'])[0]

years = player_df['SeasonIndex'].values
accuracy = player_df['ThreePtPct'].values

slope, intercept, r_value, p_value, std_err = stats.linregress(
    years, accuracy
)

fit_line = intercept + slope * years

print("\nRegression Results:")
print("Slope:", slope)
print("Intercept:", intercept)
print("R-value:", r_value)
print("P-value:", p_value)

# ====================================================================================================================

# Regression-Based Average (Integration)

avg_fit_accuracy = np.trapezoid(fit_line, years) / (
    years.max() - years.min()
)

actual_avg_accuracy = accuracy.mean()

print("\nAverage Accuracy (Regression):", avg_fit_accuracy)
print("Actual Average Accuracy:", actual_avg_accuracy)

# ====================================================================================================================

# Interpolation for Missing Seasons

player_df_interp = player_df.set_index('Season')

missing_seasons = ['2002-2003', '2015-2016']

for season in missing_seasons:
    if season not in player_df_interp.index:
        player_df_interp.loc[season] = np.nan

player_df_interp = player_df_interp.sort_index()
player_df_interp['ThreePtPct'] = player_df_interp['ThreePtPct'].interpolate()

print("\nInterpolated Values:")
for season in missing_seasons:
    if season in player_df_interp.index:
        print(season, ":", player_df_interp.loc[season, 'ThreePtPct'])

# ====================================================================================================================

# Statistical Metrics

fgm = regular_df['FGM']
fga = regular_df['FGA']

print("\nFGM Statistics:")
print("Mean:", fgm.mean())
print("Variance:", fgm.var())
print("Skew:", fgm.skew())
print("Kurtosis:", fgm.kurtosis())

print("\nFGA Statistics:")
print("Mean:", fga.mean())
print("Variance:", fga.var())
print("Skew:", fga.skew())
print("Kurtosis:", fga.kurtosis())

# ====================================================================================================================

# T-Tests

paired_ttest = stats.ttest_rel(fgm, fga)
independent_ttest = stats.ttest_ind(fgm, fga)

print("\nPaired T-Test (FGM vs FGA):")
print(paired_ttest)

print("\nIndependent T-Test (FGM vs FGA):")
print(independent_ttest)
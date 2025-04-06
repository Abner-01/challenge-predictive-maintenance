import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind  # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore
from statsmodels.tsa.seasonal import STL  # type: ignore
from statsmodels.tsa.stattools import adfuller  # type: ignore


# ----------------------------------------------------
# 2. CORRELATION ANALYSIS
# ----------------------------------------------------


def plot_sensor_correlation(df, sensor_columns):
    """Plot a correlation matrix (heatmap) for the specified sensor columns."""
    corr_matrix = df[sensor_columns].corr()

    plt.figure(figsize=(8, 6))
    plt.imshow(corr_matrix, cmap="coolwarm", interpolation="none")
    plt.colorbar()
    plt.xticks(range(len(sensor_columns)), sensor_columns, rotation=90)
    plt.yticks(range(len(sensor_columns)), sensor_columns)
    for i in range(len(sensor_columns)):
        for j in range(len(sensor_columns)):
            plt.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
            )
    plt.title("Correlation Matrix of Sensor Readings")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# 3. TIME SERIES ANALYSIS FOR A SINGLE MACHINE
# ----------------------------------------------------


def plot_time_series(df_machine, col_name, machine_id):
    """Plot the raw time series for a given column of a single machine."""
    plt.figure(figsize=(10, 4))
    plt.plot(df_machine[col_name], label=col_name)
    plt.title(f"Time Series of {col_name} (Machine {machine_id})")
    plt.xlabel("Datetime")
    plt.ylabel(col_name)
    plt.tight_layout()
    plt.show()


def plot_rolling_statistics(series, col_name, machine_id, window=24):
    """Plot rolling mean and rolling std for a given series."""
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()

    plt.figure(figsize=(10, 4))
    plt.plot(series, label=col_name)
    plt.plot(rolling_mean, label="Rolling Mean", color="red")
    plt.plot(rolling_std, label="Rolling Std", color="green")
    plt.title(
        f"Rolling Statistics of {col_name} (Machine {machine_id}, window={window})"
    )
    plt.xlabel("Datetime")
    plt.legend()
    plt.tight_layout()
    plt.show()


def adfuller_test(series, col_name):
    """Run Augmented Dickey-Fuller test on a series and prints results."""
    result = adfuller(series.dropna())
    print(f"ADF Test for {col_name}:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print("-" * 40)


def analyze_time_series_for_machine(df, machine_id, sensor_columns, rolling_window=24):
    """Perform the main time series analysis steps for a single machine:

    - Plots the raw time series
    - Plots rolling statistics (mean & std)
    - Runs ADF tests
    """
    df_machine = df.loc[machine_id].sort_index()

    for col in sensor_columns:
        series = df_machine[col]
        # 1. Raw Series Plot
        plot_time_series(df_machine, col, machine_id)
        # 2. Rolling Stats
        plot_rolling_statistics(series, col, machine_id, rolling_window)
        # 3. ADF Test
        adfuller_test(series, col)


# ----------------------------------------------------
# 4. SEASONAL DECOMPOSITION
# ----------------------------------------------------


def detect_spikes_in_trend(trend_series, window=24, z_threshold=2.5):
    """
    1) Computes a rolling mean and std over the trend component.
    2) Calculates z-scores (distance from mean in std units).
    3) Returns a boolean Series where True indicates a spike.
    """
    rolling_mean = trend_series.rolling(window=window).mean()
    rolling_std = trend_series.rolling(window=window).std()

    eps = 1e-6
    z_scores = (trend_series - rolling_mean) / (rolling_std + eps)

    return z_scores.abs() > z_threshold


def analyze_trend_spikes(
    df,
    machine_id,
    sensor_col="volt",
    failure_flag_col="failure_flag",
    stl_period=24,
    window_for_spike=24,
    z_threshold=2.5,
):
    """
    1) Subsets the DataFrame for a single machine.
    2) Computes time-to-failure if not present.
    3) Performs STL on `sensor_col` to extract the trend.
    4) Detects spikes in that trend using a rolling z-score approach.
    5) Correlates the spike indicator with time-to-failure and does
       a quick grouped comparison (spike vs no spike).
    """
    # 1) Subset for the chosen machine
    df_machine = df.loc[machine_id].copy().sort_index()

    # 2) Compute TTF if it doesn't exist
    if "time_to_failure" not in df_machine.columns:
        df_machine = compute_time_to_failure(df_machine, failure_flag_col)

    # 3) STL Decomposition to get the trend
    series = df_machine[sensor_col].dropna()
    stl = STL(
        series, period=stl_period, robust=True
    )  # robust=True can help with outliers
    result = stl.fit()
    trend = result.trend

    # Align the trend back to the main DataFrame
    df_machine["trend_component"] = trend

    # 4) Detect spikes
    spikes = detect_spikes_in_trend(
        df_machine["trend_component"], window=window_for_spike, z_threshold=z_threshold
    )
    df_machine["trend_spike"] = spikes.astype(
        int
    )  # Convert boolean to 0/1 for correlation

    # 5a) Correlation with TTF
    corr_value = df_machine["trend_spike"].corr(df_machine["time_to_failure"])
    print(
        f"\nCorrelation between 'trend_spike' and 'time_to_failure': {corr_value:.4f}"
    )

    # 5b) Compare TTF for spike vs. no spike (t-test or group stats)
    has_spike = df_machine.loc[
        df_machine["trend_spike"] == 1, "time_to_failure"
    ].dropna()
    no_spike = df_machine.loc[
        df_machine["trend_spike"] == 0, "time_to_failure"
    ].dropna()

    if not has_spike.empty and not no_spike.empty:
        t_stat, p_val = ttest_ind(has_spike, no_spike, equal_var=False)
        print(
            f"T-test between TTF when spike vs. no spike: t-stat={t_stat:.4f}, p-value={p_val:.4f}"
        )
    else:
        print("Not enough data in one of the spike groups for T-test.")

    # 5c) Print group means
    group_means = df_machine.groupby("trend_spike")["time_to_failure"].mean()
    print("\nMean TTF by spike indicator:\n", group_means)

    # --- Optional Visualizations ---

    # A) Plot the trend with spike markers
    plt.figure(figsize=(10, 5))
    plt.plot(df_machine.index, df_machine["trend_component"], label="Trend")
    plt.scatter(
        df_machine.index[df_machine["trend_spike"] == 1],
        df_machine.loc[df_machine["trend_spike"] == 1, "trend_component"],
        color="red",
        marker="x",
        s=50,
        label="Spikes",
    )
    plt.title(f"{sensor_col} Trend with Spikes (Machine {machine_id})")
    plt.xlabel("Datetime")
    plt.ylabel(f"{sensor_col} Trend")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # B) Scatter plot: TTF vs. trend
    plt.figure(figsize=(8, 5))
    plt.scatter(df_machine["trend_component"], df_machine["time_to_failure"], alpha=0.5)
    plt.title(f"Time-to-Failure vs. Trend (Machine {machine_id})")
    plt.xlabel("Trend Level")
    plt.ylabel("Time to Failure")
    plt.tight_layout()
    plt.show()


def plot_stl_decomposition(df, machine_id, col_to_decompose, period=24):
    """Perform and plots the STL decomposition for a specific sensor column of one machine."""
    df_machine = df.loc[machine_id].sort_index()
    series = df_machine[col_to_decompose].dropna()
    stl = STL(series, period=period)
    result = stl.fit()

    plt.figure(figsize=(10, 8))
    plt.subplot(4, 1, 1)
    plt.plot(result.observed)
    plt.title(f"STL Decomposition of {col_to_decompose} (Observed)")

    plt.subplot(4, 1, 2)
    plt.plot(result.trend)
    plt.title("Trend Component")

    plt.subplot(4, 1, 3)
    plt.plot(result.seasonal)
    plt.title("Seasonal Component")

    plt.subplot(4, 1, 4)
    plt.plot(result.resid)
    plt.title("Residual Component")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# 5. DISTRIBUTION ANALYSIS
# ----------------------------------------------------


def plot_sensor_distributions(df, machine_id, sensor_columns):
    """Plot both histograms and boxplots for the specified sensors of a single machine."""
    df_machine = df.loc[machine_id].sort_index()

    for col in sensor_columns:
        data = df_machine[col].dropna()

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.hist(data, bins=30, edgecolor="black")
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.boxplot(data, vert=False)
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)

        plt.tight_layout()
        plt.show()


# ----------------------------------------------------
# 6. AUTOCORRELATION ANALYSIS
# ----------------------------------------------------


def plot_acf_and_pacf(df, machine_id, sensor_columns, lags=50):
    """Plot the autocorrelation and partial autocorrelation for the specified sensors of one machine."""

    df_machine = df.loc[machine_id].sort_index()

    for col in sensor_columns:
        series = df_machine[col].dropna()

        fig = plt.figure(figsize=(12, 4))
        fig.suptitle(f"Autocorrelation and Partial Autocorrelation: {col}", fontsize=14)

        gs = GridSpec(1, 2, width_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0])
        plot_acf(series, lags=lags, ax=ax1, zero=False)
        ax1.set_title("ACF")

        ax2 = fig.add_subplot(gs[1])
        plot_pacf(series, lags=lags, ax=ax2, zero=False)
        ax2.set_title("PACF")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ----------------------------------------------------
# 7. WEEKEND MAINTENANCE ANALYSIS & TIME-TO-FAILURE
# ----------------------------------------------------


def compute_time_to_failure(df_machine, failure_flag_col="failure_flag"):
    """Function to compute time to failure (TTF) for each row
    based on a binary failure_flag column.

    Adjust to match your real logic.
    """
    df_machine["future_failure_time"] = df_machine.loc[
        df_machine[failure_flag_col] == 1, "datetime"
    ]

    df_machine["future_failure_time"] = df_machine["future_failure_time"].bfill()

    df_machine["time_to_failure"] = (
        df_machine["future_failure_time"] - df_machine["datetime"]
    ).dt.total_seconds() / 3600.0

    return df_machine


def compute_total_failures(df_machine, failure_flag_col="failure_flag"):
    """Example function to compute time to failure (TTF) for each row
    based on a binary failure_flag column.

    Adjust to match your real logic.
    """
    df_machine["time_to_failure"] = (
        df_machine[failure_flag_col][::-1]
        .cumsum()[::-1]
        .where(df_machine[failure_flag_col] == 1)
        .ffill()
        .sub(df_machine[failure_flag_col].cumsum())
    )

    return df_machine


def analyze_weekend_maintenance(df, machine_id, failure_flag_col="failure_flag"):
    """Investigating whether maintenance on Fri/Sat/Sun differs in subsequent time-to-failure."""
    df_machine = df.loc[machine_id].copy()
    df_machine = df_machine[~df_machine.index.duplicated(keep="first")]

    if "time_to_failure" not in df_machine.columns:
        df_machine = compute_time_to_failure(df_machine, failure_flag_col)

    df_machine["maintenance_done"] = (
        (df_machine["maint_comp1"] == 1)
        | (df_machine["maint_comp2"] == 1)
        | (df_machine["maint_comp3"] == 1)
        | (df_machine["maint_comp4"] == 1)
    )

    df_machine["day_of_week"] = df_machine.index.dayofweek
    df_machine["is_fri_weekend"] = df_machine["day_of_week"].isin([4, 5, 6])
    df_maint = df_machine[df_machine["maintenance_done"]].copy()

    summary = df_maint.groupby("is_fri_weekend")["time_to_failure"].agg(
        ["count", "mean", "median", "std"]
    )
    print("Time to Failure Summary After Maintenance:\n", summary, "\n")

    # T-test
    group_fri_weekend = df_maint.loc[
        df_maint["is_fri_weekend"], "time_to_failure"
    ].dropna()
    group_other_days = df_maint.loc[
        ~df_maint["is_fri_weekend"], "time_to_failure"
    ].dropna()
    stat, pval = ttest_ind(group_fri_weekend, group_other_days, equal_var=False)
    print("t-statistic:", stat, "| p-value:", pval)

    # Bar plot of the mean TTF
    grouped = df_maint.groupby("is_fri_weekend")["time_to_failure"]
    means = grouped.mean()
    stds = grouped.std()

    labels = ["Weekday (Mon–Thu)", "Fri or Weekend (Fri–Sun)"]
    x = [0, 1]

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=8)
    plt.xticks(x, labels)
    plt.ylabel("Mean Time to Failure (hours)")
    plt.title("Mean Time to Failure After Maintenance")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def analyze_maintenance_by_component(df, machine_id):
    """Show how time-to-failure differs by maintenance day type (Fri/weekend vs.

    others).
    """
    df_machine = df.loc[machine_id].copy()
    if "time_to_failure" not in df_machine.columns:
        df_machine = compute_time_to_failure(df_machine)

    df_machine["maintenance_done"] = (
        (df_machine["maint_comp1"] == 1)
        | (df_machine["maint_comp2"] == 1)
        | (df_machine["maint_comp3"] == 1)
        | (df_machine["maint_comp4"] == 1)
    )

    # Day-of-week: Monday=0 ... Sunday=6
    df_machine["day_of_week"] = df_machine.index.dayofweek
    df_machine["is_fri_weekend"] = df_machine["day_of_week"].isin([4, 5, 6])

    components = ["maint_comp1", "maint_comp2", "maint_comp3", "maint_comp4"]
    summary_by_comp = {}

    for comp in components:
        df_comp_maint = df_machine[df_machine[comp] == 1].copy()
        data_check = df_comp_maint.groupby("is_fri_weekend")["time_to_failure"].agg(
            ["count", "mean", "std"]
        )
        print(f"{comp} Data Summary:\n{data_check}\n")

        grouped = df_comp_maint.groupby("is_fri_weekend")["time_to_failure"]
        means = grouped.mean()
        stds = grouped.std()

        summary_by_comp[comp] = {"means": means, "stds": stds}

    # Plot in a 2x2 grid
    x_labels = ["Weekday (Mon–Thu)", "Fri or Weekend (Fri–Sun)"]
    x = [0, 1]

    _, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, comp in enumerate(components):
        data = summary_by_comp[comp]
        means = data["means"]
        stds = data["stds"]

        axs[i].bar(x, means, yerr=stds, capsize=8)
        axs[i].set_title(f"Time to Failure After {comp.upper()} Maintenance")
        axs[i].set_ylabel("Mean Time to Failure (hrs)")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(x_labels)
        axs[i].grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Time to Failure by Maintenance Day and Component", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# ----------------------------------------------------
# 8. HOUR-BASED & TIME-SIN CORRELATION ANALYSIS
# ----------------------------------------------------


def analyze_time_features(
    df,
    machine_id,
    ttf_col="time_to_failure",
    time_since_last_failure_col="time_since_last_failure_flag",
    filter_threshold=10000,
):
    """Analyze time features for a specific machine.

    1) Computes correlation of 'hour_sin' and 'hour_cos' with time-to-failure.
    2) Filters rows where 'time_since_last_failure_flag' < filter_threshold.
    3) Calculates mean, max, and hourly stats for 'time_since_last_failure_flag'.
    """
    df_machine = df.loc[machine_id].copy()
    if ttf_col not in df_machine.columns:
        df_machine = compute_time_to_failure(df_machine)

    if "hour_sin" in df_machine.columns and "hour_cos" in df_machine.columns:
        corr_sin = df_machine["hour_sin"].corr(df_machine[ttf_col])
        corr_cos = df_machine["hour_cos"].corr(df_machine[ttf_col])
        print(f"Correlation of hour_sin with {ttf_col}: {corr_sin:.4f}")
        print(f"Correlation of hour_cos with {ttf_col}: {corr_cos:.4f}")
    else:
        print("hour_sin/hour_cos columns not found for correlation analysis.")

    df_time = df_machine[
        df_machine[time_since_last_failure_col] < filter_threshold
    ].copy()

    if not df_time.empty:
        mean_val = df_time[time_since_last_failure_col].mean()
        max_val = df_time[time_since_last_failure_col].max()
        print(
            f"Mean of {time_since_last_failure_col} (< {filter_threshold}): {mean_val:.2f}"
        )
        print(
            f"Max of {time_since_last_failure_col} (< {filter_threshold}): {max_val:.2f}"
        )

        # Hourly grouping if 'hour' not already encoded
        if "hour" not in df_time.columns:
            df_time["hour"] = df_time.index.hour  # Extract hour from datetime index
        hourly_stats = df_time.groupby("hour")[time_since_last_failure_col].agg(
            ["mean", "median", "std", "count"]
        )
        print("\nHourly Statistics:\n", hourly_stats)
    else:
        print(f"No data found with {time_since_last_failure_col} < {filter_threshold}.")

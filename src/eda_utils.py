"""Script containing various EDA functions for time series analysis."""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind  # type: ignore
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore
from statsmodels.tsa.seasonal import STL  # type: ignore
from statsmodels.tsa.stattools import adfuller  # type: ignore


# ----------------------------------------------------
# 1. GENERAL DATA EXPLORATION
# ----------------------------------------------------
def check_null(df: pd.DataFrame) -> pd.Series:
    """Return percentage of rows containing missing data."""
    return df.isna().sum() * 100 / len(df)


def get_missing_dates(series, start_date, end_date, freq="H") -> pd.DatetimeIndex:
    """Return missing dates in the series."""
    return pd.date_range(start=start_date, end=end_date, freq=freq).difference(series)


def check_duplicate(df: pd.DataFrame, subset: Optional[list[str]] = None) -> int:
    """Return the number of duplicate rows in the DataFrame."""
    if subset is not None:
        return df.duplicated(subset=subset, keep=False).sum()
    else:
        return df.duplicated(keep=False).sum()


def explore_df(name: str, df: pd.DataFrame) -> None:
    """Perform basic exploration of a DataFrame."""
    print(f"\n--- {name} ---")
    print(f"Shape: {df.shape}")
    print("Column types:")
    print(df.dtypes)
    print("First 3 rows:")
    print(df.head(3))
    print("Null values (%):")
    print(check_null(df))
    print("Duplicate rows:", check_duplicate(df))
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        print("Datetime range:", df["datetime"].min(), "to", df["datetime"].max())
    if "machineID" in df.columns:
        print("Unique machines:", df["machineID"].nunique())


def general_exploration(df_dict: dict[str, pd.DataFrame]) -> None:
    """Perform general exploration on a dictionary of DataFrames."""
    for name, df in df_dict.items():
        explore_df(name, df)

    # Count number of failures by component
    if "PdM_failures" in df_dict:
        print("\n--- Failure Counts ---")
        print(df_dict["PdM_failures"]["failure"].value_counts())

    # Describe telemetry
    if "PdM_telemetry" in df_dict:
        print("\n--- Telemetry Stats ---")
        print(df_dict["PdM_telemetry"].describe())
        missing_dates = get_missing_dates(
            df_dict["PdM_telemetry"]["datetime"],
            start_date="2015-01-01 06:00:00",
            end_date="2016-01-01 06:00:00",
            freq="H",
        )
        print("Missing dates in telemetry: ", len(missing_dates))
        duplicated_rows = check_duplicate(
            df_dict["PdM_telemetry"], ["datetime", "machineID"]
        )
        print("Duplicated rows in telemetry: ", duplicated_rows)


# ----------------------------------------------------
# 2. CORRELATION ANALYSIS
# ----------------------------------------------------


def plot_correlations(
    df: pd.DataFrame,
    sensor_columns: list[str],
    title: str = "Correlation Matrix of Sensor Readings",
) -> None:
    """Plot a correlation matrix (heatmap) for the specified sensor columns."""
    corr_matrix = df[sensor_columns].corr()

    plt.figure(
        figsize=(max(10, 0.4 * len(sensor_columns)), max(8, 0.4 * len(sensor_columns)))
    )
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 6},
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# 3. TIME SERIES ANALYSIS FOR A SINGLE MACHINE
# ----------------------------------------------------


def plot_time_series(df_machine: pd.DataFrame, col_name: str, machine_id: str) -> None:
    """Plot the raw time series for a given column of a single machine."""
    plt.figure(figsize=(10, 4))
    plt.plot(df_machine[col_name], label=col_name)
    plt.title(f"Time Series of {col_name} (Machine {machine_id})")
    plt.xlabel("Datetime")
    plt.ylabel(col_name)
    plt.tight_layout()
    plt.show()


def plot_rolling_statistics(
    series: pd.Series, col_name: str, machine_id: str, window: int = 24
) -> None:
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


def adfuller_test(series: pd.Series, col_name: str) -> None:
    """Run Augmented Dickey-Fuller test on a series and prints results."""
    result = adfuller(series.dropna())
    print(f"ADF Test for {col_name}:")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print("-" * 40)


def analyze_time_series_for_machine(
    df: pd.DataFrame,
    machine_id: str,
    sensor_columns: list[str],
    rolling_window: int = 24,
) -> None:
    """Perform the main time series analysis steps for a single machine."""
    df_machine: pd.DataFrame = df.loc[machine_id].sort_index()  # type: ignore

    for col in sensor_columns:
        series = df_machine[col]
        plot_time_series(df_machine, col, machine_id)
        plot_rolling_statistics(series, col, machine_id, rolling_window)
        adfuller_test(series, col)


# ----------------------------------------------------
# 4. SEASONAL DECOMPOSITION
# ----------------------------------------------------


def detect_spikes_in_trend(
    trend_series: pd.Series, window: int = 24, z_threshold: float = 2.5
) -> pd.Series:
    """Detect spikes in a time series trend component using a rolling z-score method.

    Identify spikes in a trend series by calculating a rolling mean
    and standard deviation over a specified window, and then computing z-scores
    to determine how far each point deviates from the rolling mean in terms of
    standard deviations.

    Args:
        trend_series (pd.Series): The time series data representing the trend component.
        window (int, optional): The size of the rolling window to compute the mean
            and standard deviation. Defaults to 24.
        z_threshold (float, optional): The z-score threshold above which a point
            is considered a spike. Defaults to 2.5.

    Returns:
        pd.Series: A boolean Series where `True` indicates a spike in the trend series.
    """
    rolling_mean = trend_series.rolling(window=window).mean()
    rolling_std = trend_series.rolling(window=window).std()

    eps = 1e-6
    z_scores = (trend_series - rolling_mean) / (rolling_std + eps)

    return z_scores.abs() > z_threshold


def analyze_trend_spikes(
    df: pd.DataFrame,
    machine_id: str,
    sensor_col: str = "volt",
    failure_flag_col: str = "failure_flag",
    stl_period: int = 24,
    window_for_spike: int = 24,
    z_threshold: float = 2.5,
) -> None:
    """Analyzes trend spikes in sensor data for a specific machine.

    Applies STL to extract the trend component from the specified sensor column and
    detects spikes in the trend component using a rolling z-score approach to correlate
    the detected spikes with the time-to-failure and performs a grouped comparison
    (spike vs no spike).

    Args:
         df (pd.DataFrame): The input DataFrame containing sensor data for multiple machines.
         machine_id (str): The identifier for the machine to analyze.
         sensor_col (str, optional): The name of the column containing sensor data to analyze.
              Defaults to "volt".
         failure_flag_col (str, optional): The name of the column indicating failure events.
              Defaults to "failure_flag".
         stl_period (int, optional): The seasonal period for STL decomposition. Defaults to 24.
         window_for_spike (int, optional): The rolling window size for detecting spikes.
              Defaults to 24.
         z_threshold (float, optional): The z-score threshold for identifying spikes.
              Defaults to 2.5.

    Notes:
         - The function assumes the input DataFrame is indexed by machine IDs and timestamps.
         - The `compute_time_to_failure` and `corr_spikes` functions must be defined elsewhere
            in the codebase.
         - The `_plot_trend_spikes` function is used for visualizing the trend and detected spikes.
    """
    df_machine = df.loc[machine_id].copy().sort_index()

    if "time_to_failure" not in df_machine.columns:
        df_machine = compute_time_to_failure(df_machine, failure_flag_col)  # type: ignore

    series = df_machine[sensor_col].dropna()
    stl = STL(
        series, period=stl_period, robust=True
    )  # robust=True can help with outliers
    result = stl.fit()
    trend = result.trend

    df_machine["trend_component"] = trend
    corr_spikes(df_machine, window_for_spike, z_threshold)  # type: ignore
    _plot_trend_spikes(df_machine, sensor_col, machine_id)  # type: ignore


def corr_spikes(
    df_machine: pd.DataFrame, window_for_spike: int, z_threshold: float
) -> None:
    """Analyze the correlation between trend spikes and time to failure (TTF) in a dataset.

    Detects spikes in the trend component of the input DataFrame, calculates
    the correlation between the detected spikes and the "time_to_failure" column, performs
    a T-test to compare TTF values for rows with and without spikes, and computes the mean
    TTF for each spike group.

    Args:
        df_machine (pd.DataFrame): The input DataFrame containing at least the columns
            "trend_component" and "time_to_failure".
        window_for_spike (int): The window size used for detecting spikes in the trend component.
        z_threshold (float): The Z-score threshold for identifying spikes.

    Returns:
        None: The function prints the correlation value, T-test results, and mean TTF by
        spike indicator to the console.

    Notes:
        - The "trend_spike" column is added to the input DataFrame, where 1 indicates a
          detected spike and 0 indicates no spike.
        - If there is insufficient data in one of the spike groups, the T-test is skipped.
    """
    spikes: pd.Series = detect_spikes_in_trend(
        df_machine["trend_component"], window=window_for_spike, z_threshold=z_threshold
    )
    df_machine["trend_spike"] = spikes.astype(
        int
    )  # Convert boolean to 0/1 for correlation

    corr_value: float = df_machine["trend_spike"].corr(df_machine["time_to_failure"])
    print(
        f"\nCorrelation between 'trend_spike' and 'time_to_failure': {corr_value:.4f}"
    )

    has_spike: pd.Series = df_machine.loc[
        df_machine["trend_spike"] == 1, "time_to_failure"
    ].dropna()
    no_spike: pd.Series = df_machine.loc[
        df_machine["trend_spike"] == 0, "time_to_failure"
    ].dropna()

    if not has_spike.empty and not no_spike.empty:
        t_stat: float
        p_val: float
        t_stat, p_val = ttest_ind(has_spike, no_spike, equal_var=False)  # type: ignore
        print(
            f"T-test between TTF when spike vs. no spike: t-stat={t_stat:.4f}, p-value={p_val:.4f}"
        )
    else:
        print("Not enough data in one of the spike groups for T-test.")

    group_means: pd.Series = df_machine.groupby("trend_spike")["time_to_failure"].mean()
    print("\nMean TTF by spike indicator:\n", group_means)


def _plot_trend_spikes(
    df_machine: pd.DataFrame, sensor_col: str, machine_id: str
) -> None:
    """Plot trend and spikes."""
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

    plt.figure(figsize=(8, 5))
    plt.scatter(df_machine["trend_component"], df_machine["time_to_failure"], alpha=0.5)
    plt.title(f"Time-to-Failure vs. Trend (Machine {machine_id})")
    plt.xlabel("Trend Level")
    plt.ylabel("Time to Failure")
    plt.tight_layout()
    plt.show()


def plot_stl_decomposition(
    df: pd.DataFrame, machine_id: str, col_to_decompose: str, period: int = 24
) -> None:
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


def plot_sensor_distributions(
    df: pd.DataFrame, machine_id: str, sensor_columns: list[str]
) -> None:
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


def plot_acf_and_pacf(
    df: pd.DataFrame, machine_id: str, sensor_columns: list[str], lags: int = 50
) -> None:
    """Plot autocorrelation and partial autocorrelation for the specified sensors."""
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

        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()


# ----------------------------------------------------
# 7. WEEKEND MAINTENANCE ANALYSIS & TIME-TO-FAILURE
# ----------------------------------------------------


def compute_time_to_failure(
    df_machine: pd.DataFrame, failure_flag_col: str = "failure_flag"
) -> pd.DataFrame:
    """Compute time to failure (TTF) for each row."""
    df_machine["future_failure_time"] = df_machine.loc[
        df_machine[failure_flag_col] == 1, "datetime"
    ]

    df_machine["future_failure_time"] = df_machine["future_failure_time"].bfill()

    df_machine["time_to_failure"] = (
        df_machine["future_failure_time"] - df_machine["datetime"]
    ).dt.total_seconds() / 3600.0

    return df_machine


def compute_total_failures(
    df_machine: pd.DataFrame,
    failure_flag_col: str = "failure_flag",
) -> pd.DataFrame:
    """Compute the past failures (TTF) for each row."""
    df_machine["time_to_failure"] = (
        df_machine[failure_flag_col][::-1]
        .cumsum()[::-1]
        .where(df_machine[failure_flag_col] == 1)
        .ffill()
        .sub(df_machine[failure_flag_col].cumsum())
    )

    return df_machine


def analyze_weekend_maintenance(
    df: pd.DataFrame, machine_id: str | int, failure_flag_col: str = "failure_flag"
) -> None:
    """Investigate whether maintenance performed on Fr, Str, or Sun affects in ttf.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing machine data.
                           Must have a datetime index and columns for maintenance
                           and failure flags.
        machine_id (str or int): The identifier for the machine to analyze.
        failure_flag_col (str, optional): The column name indicating failure flags.
                                          Defaults to "failure_flag".

    Returns:
        None: Prints a summary of time-to-failure statistics and the results of
              a t-test comparing maintenance on weekends (Fri-Sun) versus other days.

    Notes:
        - The function assumes the presence of maintenance columns named
          "maint_comp1", "maint_comp2", "maint_comp3", and "maint_comp4".
        - If the "time_to_failure" column is not present, it will be computed
          using the `compute_time_to_failure` function.
        - A t-test is performed to assess the statistical significance of
          differences in time-to-failure between weekend and non-weekend maintenance.
        - A plot of time-to-failure by weekday is generated using `_plot_ttf_weekday`.

    Example:
        analyze_weekend_maintenance(df, machine_id="machine_01")
    """
    df_machine = df.loc[machine_id].copy()
    df_machine = df_machine[~df_machine.index.duplicated(keep="first")]

    if "time_to_failure" not in df_machine.columns:
        df_machine = compute_time_to_failure(df_machine, failure_flag_col)  # type: ignore

    df_machine["maintenance_done"] = (
        (df_machine["maint_comp1"] == 1)
        | (df_machine["maint_comp2"] == 1)
        | (df_machine["maint_comp3"] == 1)
        | (df_machine["maint_comp4"] == 1)
    )

    df_machine["day_of_week"] = df_machine.index.dayofweek  # type: ignore
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

    _plot_ttf_weekday(df_maint)  # type: ignore


def _plot_ttf_weekday(df_maint: pd.DataFrame) -> None:
    """Plot the mean time to failure (TTF) for maintenance done on weekdays vs.

    weekends.
    """
    # Bar plot of the mean TTF
    df_maint = df_maint[df_maint["time_to_failure"] > 0]
    grouped = df_maint.groupby("is_fri_weekend")["time_to_failure"]
    means = grouped.mean()
    stds = grouped.std()

    labels = ["Weekday (Mon-Thu)", "Fri or Weekend (Fri-Sun)"]
    x = [0, 1]

    plt.figure(figsize=(8, 5))
    plt.bar(x, means, yerr=stds, capsize=8)
    plt.xticks(x, labels)
    plt.ylabel("Mean Time to Failure (hours)")
    plt.title("Mean Time to Failure After Maintenance")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def analyze_maintenance_by_component(df: pd.DataFrame, machine_id: str | int) -> None:
    """Analyze how the time-to-failure of a machine differs based on the day of maintenance.

    Args:
        df (pd.DataFrame): The input DataFrame containing machine data. It should have
            columns for maintenance indicators (e.g., 'maint_comp1', 'maint_comp2', etc.)
            and optionally 'time_to_failure'. The index should be datetime-like.
        machine_id (str | int): The identifier for the machine to analyze. This is used
            to filter the DataFrame for the specific machine.

    Notes:
        - The function assumes the presence of maintenance indicator columns named
          'maint_comp1', 'maint_comp2', 'maint_comp3', and 'maint_comp4'.
        - The function generates a 2x2 grid plot to visualize the results for each component.
    """
    df_machine = df.loc[machine_id].copy()
    if "time_to_failure" not in df_machine.columns:
        df_machine = compute_time_to_failure(df_machine)  # type: ignore

    df_machine["maintenance_done"] = (
        (df_machine["maint_comp1"] == 1)
        | (df_machine["maint_comp2"] == 1)
        | (df_machine["maint_comp3"] == 1)
        | (df_machine["maint_comp4"] == 1)
    )

    df_machine["day_of_week"] = df_machine.index.dayofweek  # type: ignore
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

    _plot_ttf_maint(components, summary_by_comp, x, x_labels)


def _plot_ttf_maint(
    components: list[str],
    summary_by_comp: dict[str, dict[str, pd.Series]],
    x: list[int],
    x_labels: list[str],
) -> None:
    _, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for i, comp in enumerate(components):
        data = summary_by_comp[comp]
        means = data["means"]
        stds = data["stds"].abs()
        axs[i].bar(x, means, yerr=stds, capsize=8)
        axs[i].set_title(f"Time to Failure After {comp.upper()} Maintenance")
        axs[i].set_ylabel("Mean Time to Failure (hrs)")
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(x_labels)
        axs[i].grid(axis="y", linestyle="--", alpha=0.7)

    plt.suptitle("Time to Failure by Maintenance Day and Component", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()


# ----------------------------------------------------
# 8. HOUR-BASED & TIME-SIN CORRELATION ANALYSIS
# ----------------------------------------------------


def analyze_time_features(
    df: pd.DataFrame,
    machine_id: str | int,
    ttf_col: str = "time_to_failure",
    time_since_last_failure_col: str = "time_since_last_failure_flag",
    filter_threshold: int = 10000,
) -> None:
    """Analyze time features for a specific machine.

    Compute the correlation of 'hour_sin' and 'hour_cos' with the time-to-failure (TTF) column
    and Calculate and print insights for the 'time_since_last_failure_flag' column.

    Args:
        df (pd.DataFrame): The input DataFrame containing machine data.
        machine_id (str | int): The identifier for the specific machine to analyze.
        ttf_col (str, optional): The name of the column representing time-to-failure.
            Defaults to "time_to_failure".
        time_since_last_failure_col (str, optional): The name of the column representing time
            since the last failure. Defaults to "time_since_last_failure_flag".
        filter_threshold (int, optional): The threshold value for filtering rows based on the
            'time_since_last_failure_flag' column. Defaults to 10000.

    Returns:
        None: This function prints the analysis results and does not return any value.
    """
    df_machine: pd.DataFrame = df.loc[machine_id].copy()  # type: ignore
    if ttf_col not in df_machine.columns:
        df_machine = compute_time_to_failure(df_machine)

    if "hour_sin" in df_machine.columns and "hour_cos" in df_machine.columns:
        corr_sin: float = df_machine["hour_sin"].corr(df_machine[ttf_col])
        corr_cos: float = df_machine["hour_cos"].corr(df_machine[ttf_col])
        print(f"Correlation of hour_sin with {ttf_col}: {corr_sin:.4f}")
        print(f"Correlation of hour_cos with {ttf_col}: {corr_cos:.4f}")
    else:
        print("hour_sin/hour_cos columns not found for correlation analysis.")

    df_time: pd.DataFrame = df_machine[
        df_machine[time_since_last_failure_col] < filter_threshold
    ].copy()

    if not df_time.empty:
        _print_time_insight(df_time, time_since_last_failure_col, filter_threshold)
    else:
        print(f"No data found with {time_since_last_failure_col} < {filter_threshold}.")


def _print_time_insight(
    df_time: pd.DataFrame, time_since_last_failure_col: str, filter_threshold: int
) -> None:
    mean_val: float = df_time[time_since_last_failure_col].mean()
    max_val: float = df_time[time_since_last_failure_col].max()
    print(
        f"Mean of {time_since_last_failure_col} (< {filter_threshold}): {mean_val:.2f}"
    )
    print(f"Max of {time_since_last_failure_col} (< {filter_threshold}): {max_val:.2f}")

    # Hourly grouping if 'hour' not already encoded
    if "hour" not in df_time.columns:
        df_time["hour"] = df_time.index.hour  # type: ignore
    hourly_stats: pd.DataFrame = df_time.groupby("hour")[
        time_since_last_failure_col
    ].agg(["mean", "median", "std", "count"])
    print("\nHourly Statistics:\n", hourly_stats)

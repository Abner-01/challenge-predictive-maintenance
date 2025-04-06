import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns  # type: ignore
import statsmodels.formula.api as smf  # type: ignore

#! TODO:
# Automatically download the data
# Change the balancing function to avoid balance test and val
# Enforce positve horizonts
# Convert create_splits into a class to return fited scaler
# Add Random Forest

# EDA:
# - Check for missing values
# - Check for duplicates
# - Check correlations
# - Check stationarity
# - Check for seasonality
# - Check for trends
# - Check distribution
# - Checkk autocorlation for lags
# - Check causality

# Choose numeric columns (adjust this list as needed)
numeric_cols = ["volt", "rotate", "pressure", "vibration", "age", "failure_flag"]

final_df = pd.read_csv("data/final_df.csv")
# Compute correlation matrix
corr_matrix = final_df[numeric_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix for Numeric Variables")
plt.show()

# Scatter plot for Age vs Vibration
plt.figure(figsize=(8, 6))
sns.scatterplot(data=final_df, x="age", y="vibration", alpha=0.5)
plt.title("Scatter Plot: Age vs Vibration")
plt.xlabel("Age (years)")
plt.ylabel("Vibration")
plt.show()

# Box plot for Age grouped by Failure Flag
plt.figure(figsize=(8, 6))
sns.boxplot(x="failure_flag", y="age", data=final_df)
plt.title("Box Plot: Age Distribution by Failure Status")
plt.xlabel("Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Age (years)")
plt.show()

# 3. Regression Analysis

# Linear regression: effect of age on vibration
lm_model = smf.ols("vibration ~ age", data=final_df).fit()
print("Linear Regression Summary (Vibration ~ Age):")
print(lm_model.summary())

# Logistic regression: effect of age on failure probability
logit_model = smf.logit("failure_flag ~ age", data=final_df).fit()
print("\nLogistic Regression Summary (Failure ~ Age):")
print(logit_model.summary())

# 4. Causality Analysis using DoWhy
# -----------------------------------
# Install DoWhy via: pip install dowhy
try:
    from dowhy import CausalModel  # type: ignore
except ImportError:
    print("DoWhy library not installed. Please install it with: pip install dowhy")
else:
    # Define a preliminary causal graph.
    # This is an example graph; you need to refine it with domain expertise.
    causal_graph = """
    digraph {
        age -> vibration;
        model -> age;
        model -> vibration;
        # machineID might act as a confounder if machine-specific effects exist.
        machineID -> age;
        machineID -> vibration;
    }
    """

    # For computational ease, you might sample the data if the full dataset is large.
    df_sample = final_df.sample(n=10000, random_state=42)

    causal_model = CausalModel(
        data=df_sample, treatment="age", outcome="vibration", graph=causal_graph
    )

    # Visualize the causal graph (this will generate a file named 'causal_model.png')
    causal_model.view_model()

    # Identify the causal effect using back-door adjustment
    identified_estimand = causal_model.identify_effect()
    print("\nIdentified Estimand:")
    print(identified_estimand)

    # Estimate the effect using a linear regression method (as an example)
    causal_estimate = causal_model.estimate_effect(
        identified_estimand, method_name="backdoor.linear_regression"
    )
    print("\nCausal Estimate (Effect of Age on Vibration):")
    print(causal_estimate)

    # Refutation to test the robustness of the estimate
    refute_results = causal_model.refute_estimate(
        identified_estimand, causal_estimate, method_name="random_common_cause"
    )
    print("\nRefutation Results:")
    print(refute_results)

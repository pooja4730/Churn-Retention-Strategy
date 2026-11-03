import pandas as pd

ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
          "TechSupport", "StreamingTV", "StreamingMovies"]

def add_features(df):
    df = df.copy()
    df["num_services"] = (df[ADDONS] == "Yes").sum(axis=1)
    df["has_support"] = ((df["TechSupport"] == "Yes") |
                         (df["OnlineSecurity"] == "Yes")).astype(int)
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72],
                                labels=["0-12", "13-24", "25-48", "49+"]).astype(str)
    return df

"""
01_data_cleaning_eda.py
Loads the three raw brand datasets, cleans data, engineers features,
validates ROI, creates multi-label channel indicators, and generates EDA.
Run from the project root:
    python 01_data_cleaning_eda.py
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "outputs"
CLEAN_DIR = OUTPUT_DIR / "data"
REPORT_DIR = OUTPUT_DIR / "reports"

for folder in (CLEAN_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "Nykaa": "nykaa_campaign_data_with_nulls.csv",
    "Purplle": "purplle_campaign_data_with_nulls.csv",
    "Tira": "tira_campaign_data_with_nulls.csv",
}

NUMERIC_COLUMNS = [
    "Duration", "Impressions", "Clicks", "Leads", "Conversions",
    "Revenue", "Acquisition_Cost", "ROI", "Engagement_Score"
]
CATEGORICAL_COLUMNS = [
    "Campaign_Type", "Target_Audience", "Channel_Used",
    "Language", "Customer_Segment"
]

def load_data():
    frames = []
    for brand, filename in DATASETS.items():
        path = DATA_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {path}\n"
                "Place all three original CSV files inside the dataset folder."
            )
        frame = pd.read_csv(path)
        frame["Brand"] = brand
        frames.append(frame)
        print(f"Loaded {brand}: {len(frame):,} rows")
    return pd.concat(frames, ignore_index=True)

def main():
    print("\n[1/5] Loading and combining datasets")
    df = load_data()
    raw_rows = len(df)
    raw_missing = df.isna().sum()

    print("\n[2/5] Removing duplicates and converting data types")
    before = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = before - len(df)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Revenue is the regression target; rows without a known target cannot train it.
    missing_revenue_rows = int(df["Revenue"].isna().sum())
    df = df.dropna(subset=["Revenue"]).copy()

    # Convert dates and derive calendar features. Invalid/missing dates are retained
    # with neutral calendar values rather than silently dropping campaign records.
    parsed_dates = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df["Campaign_Month"] = parsed_dates.dt.month
    df["Campaign_DayOfWeek"] = parsed_dates.dt.dayofweek
    df["Campaign_Month"] = df["Campaign_Month"].fillna(0).astype(int)
    df["Campaign_DayOfWeek"] = df["Campaign_DayOfWeek"].fillna(0).astype(int)

    # Keep an audit copy of supplied ROI, then calculate ROI consistently.
    df["ROI_Original"] = df["ROI"]
    df["Total_Spend"] = df["Conversions"] * df["Acquisition_Cost"]
    valid_spend = df["Total_Spend"].where(df["Total_Spend"] > 0)
    df["ROI_Calculated"] = (df["Revenue"] - df["Total_Spend"]) / valid_spend

    # The project defines Profit/Loss from ROI. Use recalculated ROI so the label
    # is internally consistent with Revenue and campaign spend.
    df["Profit_Flag"] = (df["ROI_Calculated"] > 0).astype(int)

    # Derived performance features. Division guards prevent infinities.
    df["CTR"] = df["Clicks"] / df["Impressions"].replace(0, np.nan)
    df["Conversion_Rate"] = df["Conversions"] / df["Clicks"].replace(0, np.nan)
    df["Cost_per_Conversion"] = (
        df["Total_Spend"] / df["Conversions"].replace(0, np.nan)
    )
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Clean identifiers and categorical fields.
    if "Campaign_ID" in df.columns:
        df["Campaign_ID"] = df["Campaign_ID"].fillna("UNKNOWN").astype(str)
    for col in CATEGORICAL_COLUMNS + ["Brand"]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col].eq(""), col] = "Unknown"

    # Impute predictor numeric columns for the exported cleaned dataset.
    numeric_predictors = [
        "Duration", "Impressions", "Clicks", "Leads", "Conversions",
        "Acquisition_Cost", "Engagement_Score", "CTR", "Conversion_Rate",
        "Cost_per_Conversion"
    ]
    for col in numeric_predictors:
        median = df[col].median()
        df[col] = df[col].fillna(0 if pd.isna(median) else median)

    # Multi-label encoding: fit on the combined cleaned channel field and save
    # the encoder for consistent inference in the Streamlit app.
    channel_lists = df["Channel_Used"].apply(
        lambda value: [part.strip() for part in str(value).split(",") if part.strip()]
    ).tolist()
    mlb = MultiLabelBinarizer()
    channel_matrix = mlb.fit_transform(channel_lists)
    channel_columns = [f"Channel_{name}" for name in mlb.classes_]
    channel_df = pd.DataFrame(channel_matrix, columns=channel_columns, index=df.index)
    df = pd.concat([df, channel_df], axis=1)

    # Keep supplied ROI for audit; ROI_Calculated is the validated modelling label basis.
    cleaned_path = CLEAN_DIR / "cleaned_campaign_data.csv"
    raw_combined_path = CLEAN_DIR / "raw_combined_campaigns.csv"
    df.to_csv(cleaned_path, index=False)

    # ROI audit: report agreement only where both values exist.
    comparable = df[["ROI_Original", "ROI_Calculated"]].dropna()
    if len(comparable):
        roi_mae = float(
            np.mean(np.abs(comparable["ROI_Original"] - comparable["ROI_Calculated"]))
        )
        roi_corr = float(comparable["ROI_Original"].corr(comparable["ROI_Calculated"]))
    else:
        roi_mae, roi_corr = np.nan, np.nan

    audit = pd.DataFrame({
        "Metric": [
            "Raw rows combined", "Rows after duplicate removal",
            "Duplicates removed", "Rows removed due to missing Revenue",
            "Final cleaned rows", "Original missing values (total)",
            "ROI comparison rows", "ROI original vs calculated MAE",
            "ROI original vs calculated correlation"
        ],
        "Value": [
            raw_rows, before, duplicates_removed, missing_revenue_rows,
            len(df), int(raw_missing.sum()), len(comparable), roi_mae, roi_corr
        ]
    })
    audit.to_csv(REPORT_DIR / "data_quality_and_roi_audit.csv", index=False)
    pd.Series(mlb.classes_, name="Channel").to_csv(
        OUTPUT_DIR / "models_channel_list.csv", index=False
    )

    print(f"Combined rows: {raw_rows:,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Rows without Revenue removed: {missing_revenue_rows:,}")
    print(f"Final cleaned rows: {len(df):,}")
    print(f"ROI original/calculated MAE: {roi_mae:.4f}" if pd.notna(roi_mae) else "ROI comparison unavailable")

    print("\n[3/5] Generating EDA summaries")
    brand_summary = df.groupby("Brand").agg(
        Campaigns=("Campaign_ID", "count"),
        Avg_Revenue=("Revenue", "mean"),
        Avg_ROI=("ROI_Calculated", "mean"),
        Profit_Rate=("Profit_Flag", "mean"),
        Avg_Conversions=("Conversions", "mean")
    ).round(3)
    brand_summary.to_csv(REPORT_DIR / "brand_performance_summary.csv")

    campaign_summary = df.groupby("Campaign_Type").agg(
        Campaigns=("Campaign_ID", "count"),
        Avg_Revenue=("Revenue", "mean"),
        Avg_ROI=("ROI_Calculated", "mean"),
        Profit_Rate=("Profit_Flag", "mean")
    ).sort_values("Avg_Revenue", ascending=False).round(3)
    campaign_summary.to_csv(REPORT_DIR / "campaign_type_summary.csv")

    # Channel effectiveness, not just channel frequency.
    channel_rows = []
    for channel, col in zip(mlb.classes_, channel_columns):
        subset = df.loc[df[col] == 1]
        channel_rows.append({
            "Channel": channel,
            "Campaigns": len(subset),
            "Average_Revenue": subset["Revenue"].mean(),
            "Average_ROI": subset["ROI_Calculated"].mean(),
            "Profit_Rate": subset["Profit_Flag"].mean()
        })
    pd.DataFrame(channel_rows).sort_values(
        "Average_ROI", ascending=False
    ).to_csv(REPORT_DIR / "channel_effectiveness.csv", index=False)

    top_campaigns = df.nlargest(10, "ROI_Calculated")[
        ["Campaign_ID", "Brand", "Campaign_Type", "Revenue", "Total_Spend", "ROI_Calculated"]
    ]
    low_campaigns = df.nsmallest(10, "ROI_Calculated")[
        ["Campaign_ID", "Brand", "Campaign_Type", "Revenue", "Total_Spend", "ROI_Calculated"]
    ]
    top_campaigns.to_csv(REPORT_DIR / "top_10_campaigns.csv", index=False)
    low_campaigns.to_csv(REPORT_DIR / "bottom_10_campaigns.csv", index=False)

    print("\nBrand performance:\n", brand_summary)
    print("\nCampaign type performance:\n", campaign_summary)

    print("\n[4/5] Saving visualisations")
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="Brand", y="Revenue", hue="Brand", estimator="mean", errorbar=None, legend=False)
    plt.title("Average Revenue by Brand")
    plt.ylabel("Average Revenue")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "01_average_revenue_by_brand.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.barplot(data=df, x="Campaign_Type", y="ROI_Calculated", hue="Campaign_Type", estimator="mean", errorbar=None, legend=False)
    plt.title("Average Calculated ROI by Campaign Type")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "02_roi_by_campaign_type.png", dpi=160)
    plt.close()

    sample = df.sample(min(5000, len(df)), random_state=42)
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=sample, x="Total_Spend", y="Revenue", hue="Brand", alpha=0.5)
    plt.title("Campaign Spend vs Revenue (sample)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "03_spend_vs_revenue.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="Brand", hue="Profit_Flag")
    plt.title("Profitable vs Loss-Making Campaigns by Brand")
    plt.ylabel("Campaign count")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "04_profit_loss_by_brand.png", dpi=160)
    plt.close()

    corr_cols = [
        "Duration", "Impressions", "Clicks", "Leads", "Conversions",
        "Revenue", "Acquisition_Cost", "ROI_Calculated", "Engagement_Score"
    ]
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Campaign Metric Correlation")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "05_correlation_heatmap.png", dpi=160)
    plt.close()

    print("\n[5/5] Complete")
    print(f"Cleaned data: {cleaned_path}")
    print(f"Reports: {REPORT_DIR}")

if __name__ == "__main__":
    main()

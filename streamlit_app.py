import streamlit as st
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from io import StringIO
import warnings
import datetime

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

st.set_page_config(page_title="Compound V2 Wallet Scorer", layout="wide")
st.title("📊 Compound V2 Wallet Scoring Tool")

# Check if results are already processed to avoid repeating the file upload phase
if 'processed_scores' not in st.session_state:
    st.session_state.processed_scores = None

uploaded_files = st.file_uploader(
    "Upload CompoundV2 transaction JSON files (you can upload multiple):",
    type="json",
    accept_multiple_files=True
)

# Process the uploaded files
if st.button("🚀 Score Wallets") and uploaded_files:
    st.info("Processing uploaded files...")

    # 1. Load and structure the data
    data_frames = defaultdict(list)
    transaction_types = ["deposits", "borrows", "repays", "withdraws", "liquidations"]

    for uploaded_file in uploaded_files:
        try:
            data = json.load(uploaded_file)
            for tx_type in transaction_types:
                if tx_type in data and data[tx_type]:
                    df = pd.json_normalize(data[tx_type])
                    df['transaction_type'] = tx_type
                    data_frames[tx_type].append(df)
        except Exception as e:
            st.error(f"Error loading file {uploaded_file.name}: {e}")

    all_dfs = {}
    for tx_type, df_list in data_frames.items():
        if df_list:
            all_dfs[tx_type] = pd.concat(df_list, ignore_index=True)

    if not any(t in all_dfs for t in ["deposits", "borrows", "repays", "withdraws"]):
        st.error("Missing essential transaction types (deposits, borrows, etc). Please upload complete data.")
        st.stop()

    # 2. Clean and process each DataFrame
    def to_numeric_safe(series):
        return pd.to_numeric(series, errors='coerce')

    def extract_wallet_id(df):
        return df.get('account.id', pd.Series([None] * len(df), index=df.index))

    processed_dfs = {}
    for tx_type, df in all_dfs.items():
        temp_df = df.copy()
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='s', errors='coerce')
        temp_df['amount'] = to_numeric_safe(temp_df['amount'])
        temp_df['amountUSD'] = to_numeric_safe(temp_df['amountUSD'])
        temp_df['wallet'] = extract_wallet_id(temp_df)
        temp_df.dropna(subset=['timestamp', 'wallet', 'amountUSD'], inplace=True)
        if 'asset.symbol' in temp_df.columns:
            temp_df.rename(columns={'asset.symbol': 'asset_symbol'}, inplace=True)
        temp_df['transaction_type'] = tx_type
        processed_dfs[tx_type] = temp_df

    all_transactions_df = pd.concat(processed_dfs.values(), ignore_index=True)

    # 3. Feature engineering
    all_transactions_df['is_deposit'] = all_transactions_df['transaction_type'] == 'deposits'
    all_transactions_df['is_borrow'] = all_transactions_df['transaction_type'] == 'borrows'
    all_transactions_df['is_repay'] = all_transactions_df['transaction_type'] == 'repays'
    all_transactions_df['is_withdraw'] = all_transactions_df['transaction_type'] == 'withdraws'

    aggregations = {
        'n_deposits': ('is_deposit', 'sum'),
        'n_borrows': ('is_borrow', 'sum'),
        'n_repays': ('is_repay', 'sum'),
        'n_withdraws': ('is_withdraw', 'sum'),
        'total_deposit_usd': ('amountUSD', lambda x: x[all_transactions_df.loc[x.index, 'is_deposit']].sum()),
        'total_borrow_usd': ('amountUSD', lambda x: x[all_transactions_df.loc[x.index, 'is_borrow']].sum()),
        'total_repay_usd': ('amountUSD', lambda x: x[all_transactions_df.loc[x.index, 'is_repay']].sum()),
        'total_withdraw_usd': ('amountUSD', lambda x: x[all_transactions_df.loc[x.index, 'is_withdraw']].sum()),
        'first_tx_timestamp': ('timestamp', 'min'),
        'last_tx_timestamp': ('timestamp', 'max'),
        'total_transactions': ('id', 'count')
    }

    wallet_features = all_transactions_df.groupby('wallet').agg(**aggregations)

    # Derived metrics
    epsilon = 1e-6
    wallet_features['repay_borrow_ratio'] = wallet_features['total_repay_usd'] / (wallet_features['total_borrow_usd'] + epsilon)
    wallet_features.loc[wallet_features['total_borrow_usd'] < epsilon, 'repay_borrow_ratio'] = 1.0
    wallet_features['repay_borrow_ratio'] = wallet_features['repay_borrow_ratio'].clip(upper=1.0)

    wallet_features['borrow_deposit_ratio'] = wallet_features['total_borrow_usd'] / (wallet_features['total_deposit_usd'] + epsilon)

    wallet_features['account_lifetime_days'] = (
        (wallet_features['last_tx_timestamp'] - wallet_features['first_tx_timestamp'])
        .dt.total_seconds() / (60 * 60 * 24)
    )
    wallet_features['account_lifetime_days'].fillna(0, inplace=True)

    # 4. Scoring logic
    scores_df = wallet_features.copy()
    scores_df['raw_score'] = 0

    # Repay/Borrow Ratio - up to 35 points
    scores_df['raw_score'] += scores_df['repay_borrow_ratio'] * 35

    # Account lifetime (log-normalized) - up to 20 points
    max_log_lifetime = np.log1p(scores_df['account_lifetime_days'].max())
    if max_log_lifetime > 0:
        scores_df['lifetime_score'] = (
            np.log1p(scores_df['account_lifetime_days']) / max_log_lifetime
        ) * 20
    else:
        scores_df['lifetime_score'] = 0
    scores_df['raw_score'] += scores_df['lifetime_score']

    # Borrow/Deposit Ratio Penalty - up to -30 points
    threshold = 0.8
    penalty_factor = 30
    scores_df['borrow_risk_penalty'] = 0
    high_risk_mask = scores_df['borrow_deposit_ratio'] > threshold
    if high_risk_mask.any():
        max_ratio_observed = scores_df.loc[high_risk_mask, 'borrow_deposit_ratio'].max()
        penalty = ((scores_df.loc[high_risk_mask, 'borrow_deposit_ratio'] - threshold) /
                   (max_ratio_observed - threshold)) * penalty_factor
        scores_df.loc[high_risk_mask, 'borrow_risk_penalty'] = np.minimum(penalty, penalty_factor)

    scores_df['raw_score'] -= scores_df['borrow_risk_penalty']

    # Finalize output
    scores_df = scores_df.sort_values(by='raw_score', ascending=False).reset_index()

    st.success("✅ Wallet scoring complete!")
    st.dataframe(scores_df)

    # Save processed results in session state
    st.session_state.processed_scores = scores_df

# Allow the option to generate CSV for the top 1000 wallets only after processing
if st.session_state.processed_scores is not None:
    if st.checkbox("Generate CSV for Top 1000 Wallets"):
        top_1000 = st.session_state.processed_scores.head(1000)
        csv = top_1000.to_csv(index=False)
        st.download_button(
            label="Download Top 1000 Wallets (CSV)",
            data=csv,
            file_name="top_1000_wallets.csv",
            mime="text/csv"
        )

    # Analysis Generation (optional)
    def generate_wallet_analysis(df, output_path="wallet_analysis.txt"):
        top_5 = df.sort_values(by="raw_score", ascending=False).head(5)
        bottom_5 = df.sort_values(by="raw_score", ascending=True).head(5)

        def summarize_group(group, label):
            lines = [f"{label} (Scores: {group['raw_score'].min():.2f} - {group['raw_score'].max():.2f})\n"]
            if label.startswith("High"):
                lines.append("These wallets exhibit behaviors deemed positive by the scoring model:\n")
            else:
                lines.append("The lowest-scoring wallets in this dataset share distinct characteristics:\n")

            repay_ratios = group["repay_borrow_ratio"]
            lines.append(f"• Repayment Behavior: Mean repay_borrow_ratio = {repay_ratios.mean():.2f}\n")

            lifetimes = group["account_lifetime_days"]
            lines.append(f"• Lifetime: Mean account_lifetime_days = {lifetimes.mean():.1f} days\n")

            tx_counts = group["total_transactions"]
            lines.append(f"• Activity: Mean total_transactions = {tx_counts.mean():.1f}\n")

            volumes = group["total_deposit_usd"]
            lines.append(f"• Volume: Mean total_volume_usd = ${volumes.mean():,.2f}\n")

            borrow_ratios = group["borrow_deposit_ratio"]
            lines.append(f"• Borrow vs. Deposit: Mean borrow_deposit_ratio = {borrow_ratios.mean():.2f}\n")

            return "\n".join(lines)

        with open(output_path, "w") as f:
            f.write("Wallet Analysis (Top 5 vs. Bottom 5)\n")
            f.write("This analysis examines the characteristics of the 5 highest-scoring and 5 lowest-scoring wallets based on the developed credit scoring model.\n\n")

            f.write(summarize_group(top_5, "High-Scoring Wallets"))
            f.write("\n")
            f.write(summarize_group(bottom_5, "Low-Scoring Wallets"))

    # Generate the analysis text file and allow download
    generate_wallet_analysis(st.session_state.processed_scores)
    with open("wallet_analysis.txt", "r") as file:
        st.download_button("Download Wallet Analysis", file.read(), file_name="wallet_analysis.txt")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# For PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------
# Global / Adjustable Parameters
# ---------------------------
DATA_FOLDER = "data"       # Folder containing CSV files named "2010.csv" through "2025.csv"

TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025
END_YEAR = 2025           
THRESHOLD = 0.01          

EPOCHS = 250
BATCH_SIZE = 64

CSV_FOLDER = "predictions_csv"
PDF_FOLDER = "predictions_pdf"

# ---------------------------
# Technical Indicator Functions
# ---------------------------
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series, span_short=12, span_long=26):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    return ema_short - ema_long

# ---------------------------
# Data Loading and Preparation
# ---------------------------
def load_yearly_data(data_folder, start_year, end_year):
    data_dict = {}
    for year in range(start_year, end_year + 1):
        path = os.path.join(data_folder, f"{year}.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path, parse_dates=["Date"])
            df.sort_values("Date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            if df["Volume"].dtype == object:
                df["Volume"] = df["Volume"].str.replace(",", "").astype(float)
            data_dict[year] = df
        else:
            print(f"Warning: missing {path}")
    return data_dict

def prepare_data(df):
    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["SMA10"] = df["Close"].rolling(window=10).mean()
    df["RSI14"] = compute_RSI(df["Close"], period=14)
    df["MACD"] = compute_MACD(df["Close"], span_short=12, span_long=26)
    df["NextClose"] = df["Close"].shift(-1)
    # For years that are not the final year (TEST_YEAR), drop rows missing NextClose.
    if df["Date"].dt.year.max() < TEST_YEAR:
        df.dropna(subset=["SMA10", "RSI14", "MACD", "NextClose"], inplace=True)
    else:
        df.dropna(subset=["SMA10", "RSI14", "MACD"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def concatenate_years(data_dict, years):
    dfs = []
    for y in sorted(years):
        if y in data_dict:
            dfs.append(data_dict[y])
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values("Date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined = prepare_data(combined)
    return combined

# ---------------------------
# Feature Engineering for Classification
# ---------------------------
def make_label(cur_close, next_close, threshold=THRESHOLD):
    if pd.isna(next_close):
        return np.nan
    pct_change = (next_close - cur_close) / cur_close
    if pct_change > threshold:
        return 2  # Buy
    elif pct_change < -threshold:
        return 0  # Sell
    else:
        return 1  # Hold

def build_features_for_classification(df):
    # Create the label column
    df["Label"] = df.apply(lambda row: make_label(row["Close"], row["NextClose"]), axis=1)
    # Define feature columns
    feature_cols = ["Open", "High", "Low", "Close", "Volume", "SMA10", "RSI14", "MACD"]
    # Drop rows with missing features or label
    df.dropna(subset=feature_cols + ["Label"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df["Label"].values.astype(int)
    return X, y, df

def int_to_action(x):
    return {0: "Sell", 1: "Hold", 2: "Buy"}.get(x, "Unknown")

# ---------------------------
# Dense Neural Network Model for Classification
# ---------------------------
def build_dense_model(input_dim, num_classes=3):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ---------------------------
# PDF Generation Function
# ---------------------------
def df_to_pdf(df, pdf_filename, title=None):
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    styles = getSampleStyleSheet()
    flowables = []
    if title:
        flowables.append(Paragraph(title, styles["Title"]))
        flowables.append(Spacer(1, 12))
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.gray),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    flowables.append(table)
    doc.build(flowables)

# ---------------------------
# Main Script
# ---------------------------
def main():
    os.makedirs(CSV_FOLDER, exist_ok=True)
    os.makedirs(PDF_FOLDER, exist_ok=True)

    # Load all data from CSV files
    data_dict = load_yearly_data(DATA_FOLDER, 2010, END_YEAR)
    
    # For classification, train on 2022-2024 and test on 2025
    combined_train = concatenate_years(data_dict, [2022, 2023, 2024])
    combined_test = concatenate_years(data_dict, [TEST_YEAR])

    if combined_train.empty or combined_test.empty:
        print("Insufficient data. Exiting.")
        return

    # Build features and labels for classification
    X_train, y_train, combined_train = build_features_for_classification(combined_train)
    X_test, y_test, combined_test = build_features_for_classification(combined_test)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough training or testing samples. Exiting.")
        return

    
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Build the dense neural network model
    input_dim = X_train.shape[1]
    clf_model = build_dense_model(input_dim=input_dim, num_classes=3)
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    clf_model.fit(X_train_res, y_train_res, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[early_stop], verbose=1)

    # Predict on the test set
    y_pred_proba = clf_model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Map integer labels to action strings
    combined_test["Predicted_Label"] = y_pred
    combined_test["Actual_Action"] = combined_test["Label"].apply(int_to_action)
    combined_test["Predicted_Action"] = combined_test["Predicted_Label"].apply(int_to_action)

    # Print classification report
    rep = classification_report(combined_test["Label"], y_pred, zero_division=0, output_dict=True)
    print("Classification Report (Neural Network):")
    print(classification_report(combined_test["Label"], y_pred, zero_division=0))

    # Save predictions to CSV and PDF
    out_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "NextClose", "Label", "Predicted_Label", "Actual_Action", "Predicted_Action"]
    pred_csv = os.path.join(CSV_FOLDER, f"{TEST_YEAR}_predictions.csv")
    combined_test.to_csv(pred_csv, columns=out_cols, index=False)
    print(f"Saved {TEST_YEAR} predictions to {pred_csv}")

    pred_pdf = os.path.join(PDF_FOLDER, f"{TEST_YEAR}_predictions.pdf")
    df_to_pdf(combined_test[out_cols].head(200), pred_pdf, title=f"{TEST_YEAR} Predictions (First 200 Rows)")
    print(f"Saved {TEST_YEAR} predictions PDF to {pred_pdf}")

    # Save metrics summary
    stats = {
        "Year": TEST_YEAR,
        "Buy_precision": rep.get("2", {}).get("precision", 0),
        "Buy_recall": rep.get("2", {}).get("recall", 0),
        "Buy_f1": rep.get("2", {}).get("f1-score", 0),
        "Hold_precision": rep.get("1", {}).get("precision", 0),
        "Hold_recall": rep.get("1", {}).get("recall", 0),
        "Hold_f1": rep.get("1", {}).get("f1-score", 0),
        "Sell_precision": rep.get("0", {}).get("precision", 0),
        "Sell_recall": rep.get("0", {}).get("recall", 0),
        "Sell_f1": rep.get("0", {}).get("f1-score", 0),
        "accuracy": rep.get("accuracy", 0),
        "macro_precision": rep.get("macro avg", {}).get("precision", 0),
        "macro_recall": rep.get("macro avg", {}).get("recall", 0),
        "macro_f1": rep.get("macro avg", {}).get("f1-score", 0),
        "weighted_precision": rep.get("weighted avg", {}).get("precision", 0),
        "weighted_recall": rep.get("weighted avg", {}).get("recall", 0),
        "weighted_f1": rep.get("weighted avg", {}).get("f1-score", 0),
    }
    stats_df = pd.DataFrame([stats])
    stats_csv = os.path.join(CSV_FOLDER, "metrics_summary.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved metrics summary CSV to {stats_csv}")

    stats_pdf = os.path.join(PDF_FOLDER, "metrics_summary.pdf")
    df_to_pdf(stats_df, stats_pdf, title="Metrics Summary (Neural Network)")
    print(f"Saved metrics summary PDF to {stats_pdf}")

if __name__ == "__main__":
    main()

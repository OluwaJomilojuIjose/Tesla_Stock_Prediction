import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
import datetime
import collections

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
TEST_MONTH = 2            # February
END_YEAR = 2025           
THRESHOLD = 0.015          # Increased threshold for clearer signal

# Random Forest parameters
N_ESTIMATORS = 200         # Increased estimators
MAX_DEPTH = 15             # Increased max depth
RANDOM_STATE = 42
MIN_SAMPLES_LEAF = 5       # Added to prevent overfitting

# Trading strategy parameters
PREDICTION_HORIZON = 1     # How many days to look ahead for predictions (1 = same day close)
TREND_LOOKBACK = 10        # How many days to look back for trend analysis
BUY_CONFIDENCE_THRESHOLD = 0.60  # Threshold for buying opportunities
SELL_CONFIDENCE_THRESHOLD = 0.60 # Threshold for selling opportunities
PROFIT_TAKING_THRESHOLD = 0.03   # Take profits at 3% gain
STOP_LOSS_THRESHOLD = 0.05       # Cut losses at 5% loss

# Position sizing parameters
MAX_POSITION_PCT = 0.8     # Maximum percentage of capital to invest at once
MIN_POSITION_PCT = 0.2     # Minimum position size as percentage of available cash

CSV_FOLDER = "predictions_csv"
PDF_FOLDER = "predictions_pdf"
COMPARISON_FOLDER = "comparison"  # New folder for comparison files

# Simulation parameters
INITIAL_CAPITAL = 10000.0
TRANSACTION_FEE_PERCENTAGE = 0.01  # 1%

# ---------------------------
# Technical Indicator Functions
# ---------------------------
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 0.001)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series, span_short=12, span_long=26):
    ema_short = series.ewm(span=span_short, adjust=False).mean()
    ema_long = series.ewm(span=span_long, adjust=False).mean()
    return ema_short - ema_long

def compute_bollinger_bands(series, window=20, num_std=2):
    """Calculate Bollinger Bands for a price series."""
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def compute_support_resistance(df, window=20):
    """
    Calculate dynamic support and resistance levels based on recent highs and lows.
    Returns support and resistance as percentage distances from current price.
    """
    df = df.copy()
    # Find local maxima and minima
    df['Local_Max'] = df['High'].rolling(window=window, center=True).max()
    df['Local_Min'] = df['Low'].rolling(window=window, center=True).min()
    
    # Calculate distance from current price (as percentage)
    df['Resistance_Pct'] = (df['Local_Max'] / df['Close'] - 1) * 100
    df['Support_Pct'] = (df['Local_Min'] / df['Close'] - 1) * 100
    
    return df[['Resistance_Pct', 'Support_Pct']]

# ---------------------------
# Data Loading and Preparation
# ---------------------------
def load_yearly_data(data_folder, start_year, end_year):
    data_dict = {}
    for year in range(start_year, end_year + 1):
        path = os.path.join(data_folder, f"{year}.csv")
        if os.path.isfile(path):
            # Load the data
            df = pd.read_csv(path, parse_dates=["Date"])
            
            # Display the original date range as loaded
            print(f"Year {year} data as loaded: {len(df)} records from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            
            # Convert Volume to numeric, handling commas
            if "Volume" in df.columns and df["Volume"].dtype == object:
                df["Volume"] = df["Volume"].str.replace(",", "").astype(float)
            
            # Store without sorting (we'll sort when needed)
            data_dict[year] = df
            print(f"Loaded {len(df)} records for year {year}")
        else:
            print(f"Warning: missing {path}")
    return data_dict

def prepare_data_for_year(df):
    """
    Prepare a single year's data with technical indicators.
    Preserves all rows, handles missing values appropriately.
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Sort in ascending order for indicators (oldest to newest)
    df.sort_values("Date", inplace=True, ascending=True)
    df.reset_index(drop=True, inplace=True)
    
    # Basic technical indicators
    df["SMA10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["RSI14"] = compute_RSI(df["Close"], period=14)
    df["MACD"] = compute_MACD(df["Close"], span_short=12, span_long=26)
    
    # Bollinger Bands
    upper_band, middle_band, lower_band = compute_bollinger_bands(df["Close"])
    df["BB_Upper"] = upper_band
    df["BB_Middle"] = middle_band
    df["BB_Lower"] = lower_band
    df["BB_Width"] = (upper_band - lower_band) / middle_band * 100  # Normalized width
    df["BB_Position"] = (df["Close"] - lower_band) / (upper_band - lower_band)  # Position within bands (0-1)
    
    # Support and resistance levels
    supp_res = compute_support_resistance(df)
    df["Resistance_Pct"] = supp_res["Resistance_Pct"]
    df["Support_Pct"] = supp_res["Support_Pct"]
    
    # Distance from key moving averages
    df["Dist_From_SMA10"] = (df["Close"] / df["SMA10"] - 1) * 100
    df["Dist_From_SMA20"] = (df["Close"] / df["SMA20"] - 1) * 100
    df["Dist_From_SMA50"] = (df["Close"] / df["SMA50"] - 1) * 100
    
    # Trend direction indicators
    df["Trend_Direction"] = (df["SMA10"] > df["SMA10"].shift(1)).astype(int)  # 1 if uptrend, 0 if downtrend
    df["Price_Above_SMA20"] = (df["Close"] > df["SMA20"]).astype(int)  # 1 if price above SMA20
    
    # Volatility indicators
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Volatility_5d"] = df["Daily_Return"].rolling(window=5, min_periods=1).std()
    df["Volatility_10d"] = df["Daily_Return"].rolling(window=10, min_periods=1).std()
    df["Volatility_20d"] = df["Daily_Return"].rolling(window=20, min_periods=1).std()
    
    # Volatility ratio (current vs historical)
    df["Volatility_Ratio"] = df["Volatility_5d"] / df["Volatility_20d"]
    
    # Moving average crossovers (1 when SMA10 > SMA20, 0 otherwise)
    df["MA_Cross"] = (df["SMA10"] > df["SMA20"]).astype(int)
    df["MA_Cross_Change"] = df["MA_Cross"].diff().fillna(0)  # 1 when crossing up, -1 when crossing down
    
    # Advanced momentum indicators
    for period in [1, 3, 5, 10, 20]:
        df[f"Return_{period}d"] = df["Close"].pct_change(periods=period) * 100
        df[f"Volume_Change_{period}d"] = df["Volume"].pct_change(periods=period) * 100
    
    # Volume analysis
    df["Volume_SMA10"] = df["Volume"].rolling(window=10, min_periods=1).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_SMA10"]
    df["Volume_Trend"] = (df["Volume_SMA10"] > df["Volume_SMA10"].shift(1)).astype(int)  # 1 if volume increasing
    
    # Price and volume correlation (expanding window)
    df["Price_Volume_Corr"] = df["Close"].rolling(window=20, min_periods=5).corr(df["Volume"])
    
    # Gap analysis
    df["Gap_Up"] = ((df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) > 0.01).astype(int)  # 1% gap up
    df["Gap_Down"] = ((df["Close"].shift(1) - df["Open"]) / df["Close"].shift(1) > 0.01).astype(int)  # 1% gap down
    
    # Intraday volatility (High-Low range)
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"] * 100
    
    # Inside day pattern (today's range inside yesterday's)
    df["Inside_Day"] = ((df["High"] < df["High"].shift(1)) & (df["Low"] > df["Low"].shift(1))).astype(int)
    
    # Outside day pattern (today's range outside yesterday's)
    df["Outside_Day"] = ((df["High"] > df["High"].shift(1)) & (df["Low"] < df["Low"].shift(1))).astype(int)
    
    # Create target for next day/same day prediction
    df["NextClose"] = df["Close"].shift(-1)  # Next day's close
    df["SameDay_Return"] = (df["Close"] / df["Open"] - 1) * 100  # Same day open-to-close return
    
    # Fill any remaining NaN values in technical indicators with forward fill, backward fill, then mean
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col not in ["Date"]:  # Skip non-numeric columns
            df[col] = df[col].fillna(method="ffill")
            df[col] = df[col].fillna(method="bfill")
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mean())
    
    # Fill boolean columns
    bool_cols = ["Trend_Direction", "Price_Above_SMA20", "MA_Cross", "Gap_Up", "Gap_Down", 
                 "Volume_Trend", "Inside_Day", "Outside_Day"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df

def prepare_all_data(data_dict):
    """Prepare all data with a consistent approach."""
    prepared_dict = {}
    for year, df in data_dict.items():
        if df is not None and not df.empty:
            prepared_dict[year] = prepare_data_for_year(df)
    return prepared_dict

# ---------------------------
# Feature Engineering for Classification
# ---------------------------
def make_label(cur_price, next_price, threshold=THRESHOLD):
    """
    Create a label based on price movement:
    0 = Sell (price will fall)
    1 = Hold (price will stay relatively stable)
    2 = Buy (price will rise)
    """
    if pd.isna(next_price):
        return np.nan
    
    pct_change = (next_price - cur_price) / cur_price
    
    if pct_change > threshold:
        return 2  # Buy
    elif pct_change < -threshold:
        return 0  # Sell
    else:
        return 1  # Hold

def make_same_day_label(open_price, close_price, threshold=THRESHOLD):
    """
    Create a label based on same-day price movement (open to close):
    0 = Sell (price will fall from open to close)
    1 = Hold (price will stay relatively stable from open to close)
    2 = Buy (price will rise from open to close)
    """
    if pd.isna(open_price) or pd.isna(close_price):
        return np.nan
    
    pct_change = (close_price - open_price) / open_price
    
    if pct_change > threshold:
        return 2  # Buy
    elif pct_change < -threshold:
        return 0  # Sell
    else:
        return 1  # Hold
    
def add_labels(df):
    """Add prediction labels to the dataframe."""
    # Next-day label
    df["NextDay_Label"] = df.apply(lambda row: make_label(row["Close"], row["NextClose"]), axis=1)
    
    # Same-day label (open to close)
    df["SameDay_Label"] = df.apply(lambda row: make_same_day_label(row["Open"], row["Close"]), axis=1)
    
    # Use SameDay_Label as our primary target
    df["Label"] = df["SameDay_Label"]
    
    return df

def int_to_action(x):
    """Convert integer label to action string."""
    return {0: "Sell", 1: "Hold", 2: "Buy"}.get(x, "Unknown")

def prepare_training_data(training_data):
    """Prepare data for model training."""
    # Add labels
    training_data = add_labels(training_data)
    
    # Define feature columns - use all available indicators
    feature_cols = [
        # Price data
        "Open", "High", "Low", "Close", "Volume",
        
        # Moving averages
        "SMA10", "SMA20", "SMA50",
        
        # Oscillators
        "RSI14", "MACD",
        
        # Bollinger Bands
        "BB_Width", "BB_Position",
        
        # Support/Resistance
        "Resistance_Pct", "Support_Pct",
        
        # Distance from MAs
        "Dist_From_SMA10", "Dist_From_SMA20", "Dist_From_SMA50",
        
        # Trend direction
        "Trend_Direction", "Price_Above_SMA20", "MA_Cross", "MA_Cross_Change",
        
        # Volatility
        "Volatility_5d", "Volatility_10d", "Volatility_20d", "Volatility_Ratio", "Intraday_Range",
        
        # Returns over multiple timeframes
        "Daily_Return", "Return_3d", "Return_5d", "Return_10d", "Return_20d",
        
        # Volume
        "Volume_Ratio", "Volume_Trend", "Price_Volume_Corr",
        "Volume_Change_1d", "Volume_Change_3d", "Volume_Change_5d",
        
        # Patterns
        "Gap_Up", "Gap_Down", "Inside_Day", "Outside_Day"
    ]
    
    # Make sure all feature columns exist
    feature_cols = [col for col in feature_cols if col in training_data.columns]
    
    # Handle boolean/integer columns for scaling
    bool_cols = ["Trend_Direction", "Price_Above_SMA20", "MA_Cross", "MA_Cross_Change", 
                 "Gap_Up", "Gap_Down", "Volume_Trend", "Inside_Day", "Outside_Day"]
    for col in bool_cols:
        if col in training_data.columns:
            training_data[col] = training_data[col].astype(int)
    
    # Drop rows with missing features or labels
    valid_data = training_data.dropna(subset=feature_cols + ["Label"])
    
    # Extract features and labels
    X = valid_data[feature_cols].values
    y = valid_data["Label"].values.astype(int)
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, valid_data, scaler, feature_cols

# ---------------------------
# Random Forest Model for Classification
# ---------------------------
def build_random_forest_model(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE):
    """Build a Random Forest classifier model with optimized parameters."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=random_state,
        class_weight="balanced_subsample",  # Better handling of imbalanced classes
        bootstrap=True,
        max_features='sqrt',  # Better generalization
        n_jobs=-1  # Use all available processors
    )

# ---------------------------
# Resampling Functions
# ---------------------------
def safe_resample(X, y):
    """
    Safely apply resampling techniques based on class distribution.
    Handles cases where SMOTE might fail due to too few samples in a class.
    """
    # Check class distribution
    class_counts = collections.Counter(y)
    print(f"Original class distribution: {dict(class_counts)}")
    
    # If any class has fewer than 6 samples, SMOTE will fail
    min_samples = min(class_counts.values())
    
    if min_samples >= 6:
        # Use SMOTE with default parameters if we have enough samples
        try:
            sm = SMOTE(random_state=RANDOM_STATE)
            X_res, y_res = sm.fit_resample(X, y)
            print("Using SMOTE for resampling")
            return X_res, y_res
        except ValueError as e:
            print(f"SMOTE failed: {e}")
            # Fall back to RandomOverSampler if SMOTE fails
            pass
    
    # Use RandomOverSampler for smaller datasets or if SMOTE fails
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X, y)
    print("Using RandomOverSampler for resampling")
    
    # Report new class distribution
    new_class_counts = collections.Counter(y_res)
    print(f"Resampled class distribution: {dict(new_class_counts)}")
    
    return X_res, y_res

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
    
    # Convert DataFrame to a list of lists
    data = [df.columns.tolist()]
    for _, row in df.iterrows():
        data.append([str(val) for val in row.values])
    
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
# Trading Simulation Class
# ---------------------------
class TradingSimulation:
    def __init__(self, initial_capital=INITIAL_CAPITAL, transaction_fee_pct=TRANSACTION_FEE_PERCENTAGE):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.shares = 0
        self.transaction_fee_pct = transaction_fee_pct
        self.transactions = []
        self.portfolio_history = []
        self.balance_history = []
        self.date_history = []
        
        # For optimized trading strategy
        self.last_buy_date = None
        self.avg_buy_price = 0
        self.last_transaction_date = None
        self.last_transaction_type = None
        
    def calculate_transaction_fee(self, amount):
        return amount * self.transaction_fee_pct
    
    def buy(self, amount, price, date):
        if amount <= 0:
            print(f"Invalid buy amount: ${amount:.2f}")
            return False
        
        if amount > self.cash:
            # If trying to buy with more cash than available, use all available cash
            amount = self.cash
            
        fee = self.calculate_transaction_fee(amount)
        actual_amount = amount - fee
        shares_bought = actual_amount / price
        
        self.cash -= amount
        self.shares += shares_bought
        
        # Update tracking variables
        self.last_buy_date = date
        self.last_transaction_date = date
        self.last_transaction_type = "Buy"
        
        # Calculate average buy price (weighted)
        if self.avg_buy_price == 0:
            self.avg_buy_price = price
        else:
            # Weighted average based on number of shares
            old_value = self.avg_buy_price * (self.shares - shares_bought)
            new_value = price * shares_bought
            self.avg_buy_price = (old_value + new_value) / self.shares
        
        transaction = {
            'date': date,
            'action': 'Buy',
            'amount': amount,
            'fee': fee,
            'price': price,
            'shares': shares_bought,
            'cash_after': self.cash,
            'shares_after': self.shares,
            'portfolio_value': self.cash + (self.shares * price)
        }
        self.transactions.append(transaction)
        return True
    
    def sell(self, num_shares, price, date):
        if num_shares <= 0:
            print(f"Invalid sell shares: {num_shares:.6f}")
            return False
        
        if num_shares > self.shares:
            # If trying to sell more shares than owned, sell all shares
            num_shares = self.shares
            
        amount = num_shares * price
        fee = self.calculate_transaction_fee(amount)
        actual_amount = amount - fee
        
        self.shares -= num_shares
        self.cash += actual_amount
        
        # Update tracking variables
        self.last_transaction_date = date
        self.last_transaction_type = "Sell"
        
        # If all shares sold, reset average buy price
        if self.shares == 0 or abs(self.shares) < 1e-10:  # Account for floating point rounding
            self.shares = 0  # Reset to exactly zero
            self.avg_buy_price = 0
            self.last_buy_date = None
        
        transaction = {
            'date': date,
            'action': 'Sell',
            'amount': amount,
            'fee': fee,
            'price': price,
            'shares': num_shares,
            'cash_after': self.cash,
            'shares_after': self.shares,
            'portfolio_value': self.cash + (self.shares * price)
        }
        self.transactions.append(transaction)
        return True
    
    def hold(self, price, date):
        portfolio_value = self.cash + (self.shares * price)
        
        transaction = {
            'date': date,
            'action': 'Hold',
            'amount': 0,
            'fee': 0,
            'price': price,
            'shares': 0,
            'cash_after': self.cash,
            'shares_after': self.shares,
            'portfolio_value': portfolio_value
        }
        self.transactions.append(transaction)
        return True
    
    def update_history(self, date, price):
        portfolio_value = self.cash + (self.shares * price)
        self.portfolio_history.append(portfolio_value)
        self.balance_history.append(self.cash)
        self.date_history.append(date)
    
    def get_transactions_df(self):
        return pd.DataFrame(self.transactions)
    
    def get_portfolio_history_df(self):
        return pd.DataFrame({
            'Date': self.date_history,
            'Cash': self.balance_history,
            'Portfolio_Value': self.portfolio_history
        })
    
    def has_shares(self):
        """Check if we actually have shares to sell (handles floating point precision issues)"""
        return self.shares > 1e-10

# ---------------------------
# Trading Strategy Functions
# ---------------------------
def should_take_profits(current_price, avg_buy_price, threshold=PROFIT_TAKING_THRESHOLD):
    """Determine if we should take profits based on current gain."""
    if avg_buy_price == 0 or avg_buy_price <= 1e-10:
        return False
    
    gain_pct = (current_price - avg_buy_price) / avg_buy_price
    return gain_pct >= threshold

def should_cut_losses(current_price, avg_buy_price, threshold=STOP_LOSS_THRESHOLD):
    """Determine if we should cut losses based on current loss."""
    if avg_buy_price == 0 or avg_buy_price <= 1e-10:
        return False
    
    loss_pct = (avg_buy_price - current_price) / avg_buy_price
    return loss_pct >= threshold

def calculate_position_size(cash, confidence, trend_strength, max_pct=MAX_POSITION_PCT):
    """Calculate appropriate position size based on confidence and trend strength."""
    # Base size on confidence
    base_size = MIN_POSITION_PCT + (confidence * (max_pct - MIN_POSITION_PCT))
    
    # Adjust based on trend strength (0-1 scale)
    adjusted_size = base_size * trend_strength
    
    # Cap at max percentage
    final_size = min(adjusted_size, max_pct)
    
    return cash * final_size

# ---------------------------
# Comparison File Creation
# ---------------------------
def create_comparison_file(test_data, predictions):
    """
    Create a CSV file comparing predicted trading actions vs. actual price movements
    
    Args:
        test_data: DataFrame with test data
        predictions: List of prediction dictionaries
        
    Returns:
        DataFrame with comparison results
    """
    # Create DataFrame
    comparison_df = pd.DataFrame()
    
    # Add basic data
    comparison_df['Date'] = [p['Date'] for p in predictions]
    comparison_df['Open'] = [p['Open'] for p in predictions]
    comparison_df['Close'] = [p['Close'] for p in predictions]
    
    # Add model prediction and confidence
    comparison_df['Model_Predicted_Label'] = [p['Predicted_Label'] for p in predictions]
    comparison_df['Model_Predicted_Action'] = [p['Predicted_Action'] for p in predictions]
    comparison_df['Model_Confidence'] = [p['Prediction_Confidence'] for p in predictions]
    
    # Add executed action from simulation
    comparison_df['Executed_Action'] = [p['Executed_Action'] for p in predictions]
    
    # Add same-day return
    comparison_df['Same_Day_Return'] = [(p['Close'] - p['Open']) / p['Open'] * 100 for p in predictions]
    
    # Add actual labels
    comparison_df['Actual_Label'] = [p['Actual_Label'] if 'Actual_Label' in p and p['Actual_Label'] is not None else np.nan for p in predictions]
    comparison_df['Actual_Action'] = [p['Actual_Action'] if 'Actual_Action' in p and p['Actual_Action'] != 'Unknown' else np.nan for p in predictions]
    
    # Add evaluation
    comparison_df['Model_vs_Actual'] = comparison_df.apply(
        lambda row: 'Correct' if row['Model_Predicted_Label'] == row['Actual_Label'] else 'Incorrect' 
        if not pd.isna(row['Actual_Label']) else 'Unknown', axis=1)
    
    # Add performance metrics
    correct_count = (comparison_df['Model_vs_Actual'] == 'Correct').sum()
    incorrect_count = (comparison_df['Model_vs_Actual'] == 'Incorrect').sum()
    
    if correct_count + incorrect_count > 0:
        accuracy = correct_count / (correct_count + incorrect_count)
    else:
        accuracy = 0
    
    print(f"Model Accuracy: {accuracy:.2f}")
    
    return comparison_df

# ---------------------------
# Sequential Prediction and Trading Simulation
# ---------------------------
def run_february_trading_simulation(prepared_data_dict, train_years, test_year, test_month):
    """
    Run a sequential trading simulation for February 2025.
    Makes decisions based only on prior data and open price at 11 AM.
    """
    print(f"Starting trading simulation for month {test_month} of {test_year}...")
    
    # Combine training data from all training years
    train_dfs = [prepared_data_dict[year] for year in train_years if year in prepared_data_dict]
    if not train_dfs:
        raise ValueError(f"No training data available for years {train_years}")
    
    training_data = pd.concat(train_dfs).reset_index(drop=True)
    
    # Sort training data by date (ascending)
    training_data.sort_values("Date", inplace=True, ascending=True)
    
    # Prepare test data
    test_data_full = prepared_data_dict[test_year].copy()
    
    # Sort test data by date in ASCENDING order (oldest to newest)
    test_data_full.sort_values("Date", inplace=True, ascending=True)
    
    # Extract only February 2025 data
    test_data = test_data_full[test_data_full['Date'].dt.month == test_month].copy()
    
    # And all data before February for context
    prior_data = test_data_full[test_data_full['Date'] < test_data['Date'].min()].copy()
    
    print(f"Training data: {len(training_data)} rows from {training_data['Date'].min().date()} to {training_data['Date'].max().date()}")
    print(f"Prior 2025 data: {len(prior_data)} rows from {prior_data['Date'].min().date() if not prior_data.empty else 'N/A'} to {prior_data['Date'].max().date() if not prior_data.empty else 'N/A'}")
    print(f"February 2025 data: {len(test_data)} rows from {test_data['Date'].min().date()} to {test_data['Date'].max().date()}")
    
    # Add prior data to training data for context
    if not prior_data.empty:
        training_data = pd.concat([training_data, prior_data]).reset_index(drop=True)
        print(f"Combined training data: {len(training_data)} rows up to {training_data['Date'].max().date()}")
    
    # Prepare initial training dataset
    X_train, y_train, valid_train_data, scaler, feature_cols = prepare_training_data(training_data)
    
    # Deal with imbalanced classes using safe resampling technique
    X_train_res, y_train_res = safe_resample(X_train, y_train)
    
    # Build and train the model
    model = build_random_forest_model()
    model.fit(X_train_res, y_train_res)
    
    # Initialize trading simulation
    sim = TradingSimulation()
    
    # Lists to store results
    all_predictions = []
    
    # Running data - start with training data
    running_data = training_data.copy()
    
    # Store recent predictions for trend analysis
    recent_predictions = []
    recent_confidences = []
    
    # Process each day in February 2025
    for i, (idx, row) in enumerate(test_data.iterrows()):
        current_date = row['Date']
        current_price = row['Open']  # Use Open price at 11 AM
        
        print(f"Processing day {i+1}/{len(test_data)}: {current_date.strftime('%Y-%m-%d')}")
        
        # Get features for the current day
        try:
            current_features = row[feature_cols].values.reshape(1, -1)
        except KeyError as e:
            missing_cols = [col for col in feature_cols if col not in row.index]
            print(f"Missing columns: {missing_cols}")
            # Fall back to using available columns
            available_cols = [col for col in feature_cols if col in row.index]
            if not available_cols:
                print("No feature columns available. Skipping this day.")
                continue
            print(f"Continuing with {len(available_cols)} available features.")
            current_features = row[available_cols].values.reshape(1, -1)
        
        # Scale features
        current_features_scaled = scaler.transform(current_features)
        
        # Make prediction - will the stock go up or down by close?
        prediction = model.predict(current_features_scaled)[0]
        probability = model.predict_proba(current_features_scaled)[0]
        confidence = np.max(probability)  # How confident is the model?
        
        # Get the recommended action from the model
        action = int_to_action(prediction)
        
        # Update recent predictions
        recent_predictions.append(prediction)
        recent_confidences.append(confidence)
        
        # Keep only recent predictions (lookback window)
        if len(recent_predictions) > TREND_LOOKBACK:
            recent_predictions.pop(0)
            recent_confidences.pop(0)
        
        # Analyze recent prediction trend
        buy_signals = recent_predictions.count(2)
        sell_signals = recent_predictions.count(0)
        hold_signals = recent_predictions.count(1)
        
        # Weighted average confidence (more weight to recent predictions)
        weights = np.linspace(0.5, 1.0, len(recent_confidences))
        weighted_confidence = np.average(recent_confidences, weights=weights) if recent_confidences else 0
        
        # Calculate trend strength (0-1 scale)
        if action == "Buy":
            trend_strength = buy_signals / max(1, len(recent_predictions))
        elif action == "Sell":
            trend_strength = sell_signals / max(1, len(recent_predictions))
        else:
            trend_strength = 0.5  # Neutral
        
        # Print prediction insights
        print(f"  Model predicts: {action} (confidence: {confidence:.2f})")
        print(f"  Recent signals: Buy={buy_signals}, Hold={hold_signals}, Sell={sell_signals}")
        print(f"  Weighted confidence: {weighted_confidence:.2f}, Trend strength: {trend_strength:.2f}")
        
        # Default to Hold
        executed_action = "Hold"
        
        # Safety check - verify we have valid avg_buy_price if shares are held
        if sim.shares > 0 and sim.avg_buy_price <= 0:
            print(f"  WARNING: Shares held but invalid avg_buy_price ({sim.avg_buy_price}). Resetting to current price.")
            sim.avg_buy_price = current_price
        
        # Profit taking and stop loss logic - safety first!
        take_profits = should_take_profits(current_price, sim.avg_buy_price)
        cut_losses = should_cut_losses(current_price, sim.avg_buy_price)
        
        if sim.has_shares() and take_profits:
            # Take profits if we have a good gain
            print(f"  Taking profits: ${sim.avg_buy_price:.2f} -> ${current_price:.2f} " +
                  f"({(current_price - sim.avg_buy_price) / sim.avg_buy_price * 100:.2f}%)")
            
            sim.sell(sim.shares, current_price, current_date)
            executed_action = "Sell"
            print(f"  EXECUTED SELL (PROFIT TAKING): {sim.shares:.6f} shares at {current_price:.2f}")
        
        elif sim.has_shares() and cut_losses:
            # Cut losses if we're down too much
            loss_pct = (sim.avg_buy_price - current_price) / sim.avg_buy_price * 100
            print(f"  Cutting losses: ${sim.avg_buy_price:.2f} -> ${current_price:.2f} (-{loss_pct:.2f}%)")
            
            sim.sell(sim.shares, current_price, current_date)
            executed_action = "Sell"
            print(f"  EXECUTED SELL (STOP LOSS): {sim.shares:.6f} shares at {current_price:.2f}")
        
        else:
            # Regular trading strategy
            if action == "Buy":
                # Buy conditions
                buy_condition = confidence >= BUY_CONFIDENCE_THRESHOLD and sim.cash > 0
                
                if buy_condition:
                    # Calculate position size based on confidence and trend
                    position_amount = calculate_position_size(sim.cash, confidence, trend_strength)
                    
                    if position_amount > 100:  # Minimum meaningful trade
                        sim.buy(position_amount, current_price, current_date)
                        executed_action = "Buy"
                        print(f"  EXECUTED BUY: ${position_amount:.2f} at {current_price:.2f}")
                        print(f"  Reason: Predicted price increase by close (confidence={confidence:.2f})")
                    else:
                        sim.hold(current_price, current_date)
                        print(f"  Holding instead of making small buy (amount would be ${position_amount:.2f})")
                else:
                    sim.hold(current_price, current_date)
                    print(f"  Model suggests Buy but confidence too low: {confidence:.2f} < {BUY_CONFIDENCE_THRESHOLD}")
            
            elif action == "Sell" and sim.has_shares():
                # Sell conditions
                sell_condition = confidence >= SELL_CONFIDENCE_THRESHOLD
                
                if sell_condition:
                    # Calculate how much to sell based on confidence and trend
                    sell_pct = min(0.5 + (confidence * trend_strength * 0.5), 1.0)
                    shares_to_sell = sim.shares * sell_pct
                    
                    sim.sell(shares_to_sell, current_price, current_date)
                    executed_action = "Sell"
                    print(f"  EXECUTED SELL: {shares_to_sell:.6f} shares at {current_price:.2f}")
                    print(f"  Reason: Predicted price decrease by close (confidence={confidence:.2f})")
                else:
                    sim.hold(current_price, current_date)
                    print(f"  Model suggests Sell but confidence too low: {confidence:.2f} < {SELL_CONFIDENCE_THRESHOLD}")
            
            else:
                # Hold by default
                sim.hold(current_price, current_date)
                print(f"  EXECUTED HOLD (following model recommendation or no action needed)")
        
        # Get the actual same-day label (for evaluation)
        actual_label = make_same_day_label(row['Open'], row['Close'])
        actual_action = int_to_action(actual_label)
        
        # Update portfolio history using the Close price
        sim.update_history(current_date, row['Close'])
        
        # Store prediction results
        prediction_result = {
            'Date': current_date,
            'Open': row['Open'],
            'High': row['High'],
            'Low': row['Low'],
            'Close': row['Close'],
            'Volume': row['Volume'],
            'Same_Day_Return': (row['Close'] - row['Open']) / row['Open'] * 100,
            'Predicted_Label': prediction,
            'Predicted_Action': action, 
            'Executed_Action': executed_action,
            'Prediction_Confidence': confidence,
            'Actual_Label': actual_label,
            'Actual_Action': actual_action,
            'Recent_Buy_Signals': buy_signals,
            'Recent_Sell_Signals': sell_signals
        }
        all_predictions.append(prediction_result)
        
        # Add the current day's data to running data after we know the close
        current_row_with_label = row.copy()
        current_row_with_label['Label'] = actual_label
        running_data = pd.concat([running_data, pd.DataFrame([current_row_with_label])], ignore_index=True)
        
        # Retrain the model periodically
        if (i + 1) % 5 == 0 and i > 0:
            # Prepare updated training data
            X_train, y_train, valid_train_data, scaler, feature_cols = prepare_training_data(running_data)
            
            # Apply safe resampling technique
            try:
                X_train_res, y_train_res = safe_resample(X_train, y_train)
                
                # Retrain model
                model = build_random_forest_model()
                model.fit(X_train_res, y_train_res)
                print(f"  Model retrained after day {i+1}")
            except Exception as e:
                print(f"Error retraining model: {e}")
                print("Continuing with previous model")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_predictions)
    
    # Create comparison file
    comparison_df = create_comparison_file(test_data, all_predictions)
    
    # Add formatted trading advice
    def format_advice(row, sim_transactions):
        # Find the transaction for this date
        date_str = row['Date'].strftime('%Y-%m-%d')
        matching_transactions = [t for t in sim_transactions if t['date'].strftime('%Y-%m-%d') == date_str]
        
        if not matching_transactions:
            return "No transaction"
        
        transaction = matching_transactions[0]
        action = transaction['action']
        
        if action == "Buy":
            return f"Buy: ${transaction['amount']:.2f}"
        elif action == "Sell":
            return f"Sell: {transaction['shares']:.6f} shares"
        else:
            return "Hold"
    
    transactions_list = sim.transactions
    results_df['Trading_Advice'] = results_df.apply(
        lambda row: format_advice(row, transactions_list), axis=1)
    
    # Add strategy information
    results_df['Executed_vs_Predicted'] = results_df.apply(
        lambda row: "Same" if row['Executed_Action'] == row['Predicted_Action'] else "Different", 
        axis=1
    )
    
    # Calculate metrics for trading performance
    if not results_df.empty:
        y_true = results_df['Actual_Label'].dropna().astype(int)
        y_pred = results_df.loc[y_true.index, 'Predicted_Label'].astype(int)
        
        if len(y_true) > 0 and len(y_pred) > 0:
            try:
                # Generate classification report
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                
                # Create metrics summary
                metrics = {
                    'Year': test_year,
                    'Month': test_month,
                    'Buy_precision': report.get('2', {}).get('precision', 0),
                    'Buy_recall': report.get('2', {}).get('recall', 0),
                    'Buy_f1': report.get('2', {}).get('f1-score', 0),
                    'Hold_precision': report.get('1', {}).get('precision', 0),
                    'Hold_recall': report.get('1', {}).get('recall', 0),
                    'Hold_f1': report.get('1', {}).get('f1-score', 0),
                    'Sell_precision': report.get('0', {}).get('precision', 0),
                    'Sell_recall': report.get('0', {}).get('recall', 0),
                    'Sell_f1': report.get('0', {}).get('f1-score', 0),
                    'accuracy': report.get('accuracy', 0),
                    'macro_precision': report.get('macro avg', {}).get('precision', 0),
                    'macro_recall': report.get('macro avg', {}).get('recall', 0),
                    'macro_f1': report.get('macro avg', {}).get('f1-score', 0),
                    'weighted_precision': report.get('weighted avg', {}).get('precision', 0),
                    'weighted_recall': report.get('weighted avg', {}).get('recall', 0),
                    'weighted_f1': report.get('weighted avg', {}).get('f1-score', 0),
                }
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                metrics = {col: 0 for col in ['Year', 'Month', 'Buy_precision', 'Buy_recall', 'Buy_f1', 
                                             'Hold_precision', 'Hold_recall', 'Hold_f1',
                                             'Sell_precision', 'Sell_recall', 'Sell_f1',
                                             'accuracy', 'macro_precision', 'macro_recall', 
                                             'macro_f1', 'weighted_precision', 'weighted_recall',
                                             'weighted_f1']}
                metrics['Year'] = test_year
                metrics['Month'] = test_month
        else:
            print("No matching predictions and actuals for metrics calculation")
            metrics = {col: 0 for col in ['Year', 'Month', 'Buy_precision', 'Buy_recall', 'Buy_f1', 
                                         'Hold_precision', 'Hold_recall', 'Hold_f1',
                                         'Sell_precision', 'Sell_recall', 'Sell_f1',
                                         'accuracy', 'macro_precision', 'macro_recall', 
                                         'macro_f1', 'weighted_precision', 'weighted_recall',
                                         'weighted_f1']}
            metrics['Year'] = test_year
            metrics['Month'] = test_month
    else:
        print("Empty results DataFrame, can't calculate metrics")
        metrics = {col: 0 for col in ['Year', 'Month', 'Buy_precision', 'Buy_recall', 'Buy_f1', 
                                     'Hold_precision', 'Hold_recall', 'Hold_f1',
                                     'Sell_precision', 'Sell_recall', 'Sell_f1',
                                     'accuracy', 'macro_precision', 'macro_recall', 
                                     'macro_f1', 'weighted_precision', 'weighted_recall',
                                     'weighted_f1']}
        metrics['Year'] = test_year
        metrics['Month'] = test_month
    
    metrics_df = pd.DataFrame([metrics])
    
    # Get portfolio performance metrics
    portfolio_history = sim.get_portfolio_history_df()
    final_portfolio_value = portfolio_history['Portfolio_Value'].iloc[-1] if not portfolio_history.empty else 0
    roi_percentage = ((final_portfolio_value / INITIAL_CAPITAL) - 1) * 100
    
    performance_metrics = {
        'Initial_Capital': INITIAL_CAPITAL,
        'Final_Portfolio_Value': final_portfolio_value,
        'ROI_Percentage': roi_percentage,
        'Final_Cash': sim.cash,
        'Final_Shares': sim.shares,
        'Last_Close_Price': test_data['Close'].iloc[-1] if not test_data.empty else 0,
        'Total_Transactions': len(sim.transactions),
        'Total_Fees_Paid': sum(t['fee'] for t in sim.transactions),
        'Transaction_Count_Buy': sum(1 for t in sim.transactions if t['action'] == 'Buy'),
        'Transaction_Count_Sell': sum(1 for t in sim.transactions if t['action'] == 'Sell'),
        'Transaction_Count_Hold': sum(1 for t in sim.transactions if t['action'] == 'Hold'),
    }
    
    performance_df = pd.DataFrame([performance_metrics])
    
    return results_df, sim.get_transactions_df(), portfolio_history, metrics_df, performance_df, comparison_df

# ---------------------------
# Main Script
# ---------------------------
def main():
    os.makedirs(CSV_FOLDER, exist_ok=True)
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(COMPARISON_FOLDER, exist_ok=True)

    print(f"Tesla Trading Agent - February {TEST_YEAR} Optimization")
    print("===========================================")
    
    # Load all data from CSV files
    print("Loading data...")
    data_dict = load_yearly_data(DATA_FOLDER, 2010, END_YEAR)
    
    # Verify we have the 2025 test data
    if TEST_YEAR not in data_dict or data_dict[TEST_YEAR] is None or data_dict[TEST_YEAR].empty:
        print(f"ERROR: No data available for test year {TEST_YEAR}")
        return
    
    # Check if February data exists
    feb_data = data_dict[TEST_YEAR][data_dict[TEST_YEAR]['Date'].dt.month == TEST_MONTH]
    if feb_data.empty:
        print(f"ERROR: No data for month {TEST_MONTH} in year {TEST_YEAR}")
        return
    
    print(f"Month {TEST_MONTH} of {TEST_YEAR} has {len(feb_data)} trading days")
    
    # Prepare all data with technical indicators
    print("Preparing data with technical indicators...")
    prepared_data_dict = prepare_all_data(data_dict)
    
    # Print summary of prepared data
    for year, df in prepared_data_dict.items():
        if df is not None:
            print(f"Year {year}: {len(df)} trading days from {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # Verify again that test data is still intact after preparation
    if TEST_YEAR not in prepared_data_dict or prepared_data_dict[TEST_YEAR] is None or prepared_data_dict[TEST_YEAR].empty:
        print(f"ERROR: Test year {TEST_YEAR} data was lost during preparation")
        return
    
    feb_data = prepared_data_dict[TEST_YEAR][prepared_data_dict[TEST_YEAR]['Date'].dt.month == TEST_MONTH]
    if feb_data.empty:
        print(f"ERROR: Month {TEST_MONTH} data was lost during preparation")
        return
    
    print(f"Prepared month {TEST_MONTH} of {TEST_YEAR} has {len(feb_data)} trading days")
    
    try:
        # Run the trading simulation for February 2025
        print(f"\nStarting trading simulation for February {TEST_YEAR}...")
        results_df, transactions_df, portfolio_history, metrics_df, performance_df, comparison_df = run_february_trading_simulation(
            prepared_data_dict, TRAIN_YEARS, TEST_YEAR, TEST_MONTH)
        
        # Save trading results to CSV and PDF
        results_csv = os.path.join(CSV_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_trading_results.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} trading results to {results_csv}")
        
        # Limit PDF to prevent memory issues
        pdf_results = results_df.head(50) if len(results_df) > 50 else results_df
        results_pdf = os.path.join(PDF_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_trading_results.pdf")
        df_to_pdf(pdf_results, results_pdf, title=f"Month {TEST_MONTH}/{TEST_YEAR} Trading Results")
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} trading results PDF to {results_pdf}")
        
        # Save comparison data
        comparison_csv = os.path.join(COMPARISON_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_predictions_vs_actual.csv")
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} predictions vs. actual to {comparison_csv}")
        
        # Limit PDF to prevent memory issues
        pdf_comparison = comparison_df.head(50) if len(comparison_df) > 50 else comparison_df
        columns_to_include = ['Date', 'Open', 'Close', 'Same_Day_Return', 'Model_Predicted_Action', 
                              'Executed_Action', 'Actual_Action', 'Model_vs_Actual']
        columns_to_include = [col for col in columns_to_include if col in pdf_comparison.columns]
        
        comparison_pdf = os.path.join(PDF_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_predictions_vs_actual.pdf")
        df_to_pdf(pdf_comparison[columns_to_include], 
                comparison_pdf, title=f"Month {TEST_MONTH}/{TEST_YEAR} Predictions vs. Actual")
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} predictions vs. actual PDF to {comparison_pdf}")
        
        # Save transaction history to CSV and PDF
        transactions_csv = os.path.join(CSV_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_transactions.csv")
        transactions_df.to_csv(transactions_csv, index=False)
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} transactions to {transactions_csv}")
        
        transactions_pdf = os.path.join(PDF_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_transactions.pdf")
        df_to_pdf(transactions_df, transactions_pdf, title=f"Month {TEST_MONTH}/{TEST_YEAR} Transaction History")
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} transactions PDF to {transactions_pdf}")
        
        # Save portfolio history to CSV and PDF
        portfolio_csv = os.path.join(CSV_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_portfolio_history.csv")
        portfolio_history.to_csv(portfolio_csv, index=False)
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} portfolio history to {portfolio_csv}")
        
        portfolio_pdf = os.path.join(PDF_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_portfolio_history.pdf")
        df_to_pdf(portfolio_history, portfolio_pdf, title=f"Month {TEST_MONTH}/{TEST_YEAR} Portfolio History")
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} portfolio history PDF to {portfolio_pdf}")
        
        # Save metrics
        metrics_csv = os.path.join(CSV_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_metrics.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} metrics to {metrics_csv}")
        
        metrics_pdf = os.path.join(PDF_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_metrics.pdf")
        df_to_pdf(metrics_df, metrics_pdf, title=f"Month {TEST_MONTH}/{TEST_YEAR} Metrics")
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} metrics PDF to {metrics_pdf}")
        
        # Save performance metrics
        performance_csv = os.path.join(CSV_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_performance_metrics.csv")
        performance_df.to_csv(performance_csv, index=False)
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} performance metrics to {performance_csv}")
        
        performance_pdf = os.path.join(PDF_FOLDER, f"{TEST_YEAR}_{TEST_MONTH:02d}_performance_metrics.pdf")
        df_to_pdf(performance_df, performance_pdf, title=f"Month {TEST_MONTH}/{TEST_YEAR} Performance Metrics")
        print(f"Saved {TEST_MONTH}/{TEST_YEAR} performance metrics PDF to {performance_pdf}")
        
        # Print final performance
        final_value = performance_df['Final_Portfolio_Value'].iloc[0]
        roi = performance_df['ROI_Percentage'].iloc[0]
        
        print(f"\nTesla Trading Simulation Results for February {TEST_YEAR}:")
        print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Final Cash: ${performance_df['Final_Cash'].iloc[0]:.2f}")
        print(f"Final Shares: {performance_df['Final_Shares'].iloc[0]:.6f}")
        print(f"Last Tesla Price: ${performance_df['Last_Close_Price'].iloc[0]:.2f}")
        print(f"Total Transactions: {performance_df['Total_Transactions'].iloc[0]}")
        print(f"  - Buy transactions: {performance_df['Transaction_Count_Buy'].iloc[0]}")
        print(f"  - Sell transactions: {performance_df['Transaction_Count_Sell'].iloc[0]}")
        print(f"  - Hold transactions: {performance_df['Transaction_Count_Hold'].iloc[0]}")
        print(f"Total Fees Paid: ${performance_df['Total_Fees_Paid'].iloc[0]:.2f}")
        
        # Print metrics summary
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {metrics_df['accuracy'].iloc[0]:.4f}")
        print(f"Macro F1 Score: {metrics_df['macro_f1'].iloc[0]:.4f}")
        
        # Display all transactions
        print("\nFebruary 2025 Transactions:")
        for i, (_, row) in enumerate(transactions_df.iterrows()):
            date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else row['date']
            action = row['action']
            price = row['price']
            
            if action == 'Buy':
                amount = row['amount']
                shares = row['shares']
                print(f"{date_str}: {action} ${amount:.2f} ({shares:.6f} shares at ${price:.2f})")
            elif action == 'Sell':
                shares = row['shares']
                amount = row['amount']
                print(f"{date_str}: {action} {shares:.6f} shares at ${price:.2f} (${amount:.2f})")
            else:  # Hold
                print(f"{date_str}: {action}")
            
    except Exception as e:
        import traceback
        print(f"ERROR in simulation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep

df = pd.read_csv('data/Ecommerce_Fraud_Detection.csv')
df

# data cleaning
df.isnull()

df.fillna(df.mean, inplace= True)
df.dropna(inplace= True)

# remove duplicates
df.drop_duplicates(inplace= True)

# fix data type
df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors= 'coerce')

# Check if any values became 'NaT' (Not a Time)
invalid_dates =df['transaction_time'].isna().sum()
if invalid_dates > 0:
    print(f"Waring :{invalid_dates} rows could not be parsed and are now NaT.")


def apply_missing_value_strategy(df):
    # Create a copy to avoid setting with copy waring:
    df = df.copy()

    # 1. Amount mandontary remove missing value
    df = df.dropna(subset=['amount'])

    # 2. 'avg_amt_user': Replace with user's typical spend
    
    if 'user_id' in df.columns:
        df['avg_amount_user'] = df.groupby('user_id')['avg_amount_user'].transform(lambda x: x.fillna(x.mean()))

        # 3. 'shipping_distanse_km': Replace with median or 0 (digital)
    median_val = df['shipping_distance_km'].median()

    # Logic: If digital, fill with 0. Otherwise, fill with median.
    # Assuming 'is_digital' is a boolean or 1/0 column

    df['shipping_distance_km'] = np.where(
        df['merchant_category'] == True,
        df['shipping_distance_km'].fillna(0),
        df['shipping_distance_km'].fillna(median_val)
    )

    # 4. 'Security flags': Label as "Not Checked"
    df['three_ds_flag'] = df['three_ds_flag'].fillna('Not Checked')

    return df
apply_missing_value_strategy(df)


def standardize_data(df):
    df = df.copy()

    # 1. Convert inconsistent values (Yes/Y/1 -> Yes, No/N/0 -> No)
    bool_map = {
        'Yes': "Yes",'Y': "Yes",'1':"Yes", 1: "Yes",
        "No": "No",'N': 'No','0': "No", 0: 'No'
    }
    # Apply to columns that should be Yes/No (Example: 'is_member')
    if 'is_fraud' in df.columns:
        df['is_fraud'] = df['is_fraud'].map(bool_map)

    # 2. Standardize country names (Title Case & strip spaces)
    if 'country' in df.columns:
        df['country'] = df['country'].astype(str).str.strip().str.title()

    # 3. Fix spelling/format in categories
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str).str.strip().str.title()

    return df
standardize_data(df)    


# remove outliers
Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1

# Define Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify Outliers
outliers = df[(df['amount'] < lower_bound) | (df['amount'] > upper_bound)]

# Filter them out keep only normal Data
df_clean = df[(df['amount'] >= lower_bound) & (df['amount'] <= upper_bound)]

# Assuming you have already calculated lower_bound and upper_bound

print(f"Total Outliers detected: {len(outliers)}")
print("-" * 30)
print("Top 5 Highest Outliers:")
print(outliers['amount'].sort_values(ascending=False).head())

print("\nTop 5 Lowest Outliers:")
print(outliers['amount'].sort_values(ascending=True).head())

sns.boxplot(x = df['amount'])
plt.title("Outliers Detection For Amount using Boxplot")
plt.show()

# log transformation
df['amount_log'] = np.log1p(df['amount'])
print("Log Transformation Value:")
print(df['amount_log'].head())

# Calculate Outliers
Q1_log = df['amount_log'].quantile(0.25)
Q3_log = df['amount_log'].quantile(0.75)
IQR_log = Q3_log - Q1_log

# Define bonds identify Ouliers
lower_limit = Q1_log - 1.5 * IQR_log
upper_limit= Q3_log + 1.5 * IQR_log

# Apply the Capping
df['amount_log_final'] = df['amount_log'].clip(lower= lower_limit, upper= upper_limit)

final_outliers = df[(df['amount_log_final'] < lower_limit) | (df['amount_log_final'] > upper_limit)]
print(f"Final Outliers: {len(final_outliers)}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['amount_log_final'], kde=True, color='blue')
plt.title('Final Data Distribution')

# remove ouliers in shipping distance km
# Calculate Ouartiles
Q1 = df['shipping_distance_km'].quantile(0.25)
Q3 = df['shipping_distance_km'].quantile(0.75)
IQR = Q3 - Q1

# Define Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify Outliers
outliers = df[(df['shipping_distance_km'] < lower_bound) | (df['shipping_distance_km'] > upper_bound)]

# Filter them out keep only normal Data
df_clean = df[(df['shipping_distance_km'] >= lower_bound) & (df['shipping_distance_km'] <= upper_bound)]

# Assuming you have already calculated lower_bound and upper_bound

print(f"Total Outliers detected: {len(outliers)}")
print("-" * 30)
print("Top 5 Highest Outliers:")
print(outliers['shipping_distance_km'].sort_values(ascending=False).head())

print("\nTop 5 Lowest Outliers:")
print(outliers['shipping_distance_km'].sort_values(ascending=True).head())

sns.stripplot(x = df['shipping_distance_km'])
plt.title("Outliers Detection shipping distance km using Boxplot")
plt.show()

# log transformation
df['shipping_distance_km_log'] = np.log1p(df['shipping_distance_km'])
print("Log Transformation Value:")
print(df['shipping_distance_km_log'].head())

# 1. Calculate Outliers
Q1 = df['shipping_distance_km_log'].quantile(0.25)
Q3 = df['shipping_distance_km_log'].quantile(0.75)
IQR = Q3 - Q1

# 2. Limits Set
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

# 3. Apply the clip for new column add.
df['shipping_distance_final'] = df['shipping_distance_km_log'].clip(lower=lower_limit, upper=upper_limit)

# 4. Final Outliers Check for New column.
final_outliers = ((df['shipping_distance_final'] < lower_limit) | (df['shipping_distance_final'] > upper_limit)).sum()

print(f"Lower Limit: {lower_limit}, Upper Limit: {upper_limit}")
print(f"Final Outliers in New Column: {final_outliers}")

# 5. Verification: 
print(f"Max value after capping: {df['shipping_distance_final'].max()}")

sns.stripplot(x = df['shipping_distance_final'])
plt.title('Fixed Output in shipping distance km')
plt.show()

# EDA


# Fraud Distribution
def Analyze_fraud_distribution(df):
    if 'is_fraud' in df.columns:
        counts = df['is_fraud'].value_counts(normalize= True) * 100
        print("--- Fraud Distrbution ---")
        print(f"Non Fraud: {counts.get("NO", 0):.2f}%")
        print(f"Fraud : {counts.get("YES", 1):.2f}%")
        print("Business Insight: Imbalance is expected as fraud is rare.")
    else:
        print("Error: 'is_fraud' column not found for distribution analysis.")
Analyze_fraud_distribution(df)

plt.figure(figsize=(8, 5))
sns.countplot(x = 'is_fraud', data = df, palette=['#4CAF50', '#FF5252'])
plt.title("Critical Data Imbalance: Fraudulent Transactions Represent Only 1% of Total Volume")
plt.xlabel("Fraud (0 = No, 1 = Yes)")
plt.show()


# transaction behaviour analysis
def analyze_transaction_behavior(df):
    print("--- 4.2 Transaction Behavior Analysis ---")

    # 1. Create 'Amount Buckets' (Low, Medium, High)
    # This splits your 'amt' column into 3 equal-sized groups
    df['amt_bucket'] = pd.qcut(df['amount'], q=3, labels=['Low', 'Medium', 'High'])

    # 2. Calculate Fraud Rate for each Bucket
    fraud_col = 'is_fraud' 
    
    # We convert the fraud column to numbers (1 for Yes, 0 for No) to get the average
    df['fraud_numeric'] = df[fraud_col].apply(lambda x: 1 if str(x) in ['Yes', '1', 'True'] else 0)
    
    print("\nFraud Rate by Amount Level:")
    print(df.groupby('amt_bucket')['fraud_numeric'].mean() * 100)

    # 3. Compare Current Amount vs User's Average
    # We calculate how many times larger the current transaction is compared to their normal
    if 'avg_amt_user' in df.columns:
        df['spend_ratio'] = df['amount'] / df['avg_amount_user']
        
        print("\nSpike Analysis (Current Amt / Avg User Amt):")
        # Show the average 'spike' for Non-Fraud vs Fraud
        print(df.groupby(fraud_col)['spend_ratio'].mean())
        print("\nInsight: If the Fraud number is higher, it means fraudsters spend unusually high amounts.")

analyze_transaction_behavior(df)

fraud_rate_bucket = df.groupby('amt_bucket')['fraud_numeric'].mean().reset_index()

plt.figure(figsize= (8,5))
sns.barplot(x = 'amt_bucket', y = 'fraud_numeric', data= fraud_rate_bucket, palette='viridis')
plt.title('High-Value Transactions and Micro-Charges Identified as Primary Targets for Fraudulent Activity')
plt.ylabel('Fraud Rate')
plt.xlabel('Amount Level')
plt.show()

# shipping distance analysis
def analyze_shipping_risk(df):
    print("--- Shipping Distance Analysis ---")

    # 1. Create 'Local' vs 'Long Distance' Groups
    # We split the data into 2 equal parts: the lower half is 'Local', upper half is 'Long Distance'
    try:
        df['shipping_type'] = pd.qcut(df['shipping_distance_km'], q=2, labels=['Local', 'Long Distance'])
    except ValueError:
        # Fallback if there isn't enough data variance
        df['shipping_type'] = 'Unknown'

    # 2. Calculate Fraud Rate for each Group
    # REPLACE 'is_fraud' with your actual fraud column name
    fraud_col = 'is_fraud'
    
    # Convert fraud column to numbers (1 for Yes, 0 for No) if needed
    # (If you already ran the previous step, this 'fraud_numeric' column might already exist)
    if 'fraud_numeric' not in df.columns:
        df['fraud_numeric'] = df[fraud_col].apply(lambda x: 1 if str(x) in ['Yes', '1', 'True'] else 0)

    print("\nFraud Rate by Shipping Distance:")
    print(df.groupby('shipping_type')['fraud_numeric'].mean() * 100)
    
    print("\nInsight: If 'Long Distance' has a higher percentage, fraudsters are ordering items to faraway places.")

# --- How to run it ---
analyze_shipping_risk(df)

shipping_risk = df.groupby('shipping_type')['fraud_numeric'].mean().reset_index()

plt.figure(figsize= (8, 5))
sns.barplot(x = 'shipping_type', y = 'fraud_numeric', data= shipping_risk, palette= 'coolwarm')
plt.title('Logistics and Transaction Value Correlate to Risk: Long-Distance Shipping and High-Value Orders Drive Fraud Peak')
plt.ylabel('Fraud Rate')
plt.xlabel('shipping type')
plt.show()

# time analysis
def analyze_time_patterns(df):
    print("--- Time Analysis: Business Hours vs Late Night ---")

    # 1. Convert your time column to DateTime objects
    # REPLACE 'trans_time' with your actual time column (e.g., 'trans_date_trans_time', 'timestamp')
    time_col = 'transaction_time' 
    
    # This line ensures Python understands the dates/times correctly
    df[time_col] = pd.to_datetime(df[time_col])

    # 2. Extract the Hour (0 to 23)
    df['hour'] = df[time_col].dt.hour

    # 3. Create Categories: 'Business Hours' (9 AM - 9 PM) vs 'Late Night' (Everything else)
    # logic: if hour is between 9 and 21 (9pm), it's Business. Else, Late Night.
    df['time_of_day'] = df['hour'].apply(lambda x: 'Business Hours' if 9 <= x <= 21 else 'Late Night')

    # 4. Calculate Fraud Rate by Time Category
    fraud_col = 'is_fraud' # Update this if your column name is different
    
    if 'fraud_numeric' not in df.columns:
         df['fraud_numeric'] = df[fraud_col].apply(lambda x: 1 if str(x) in ['Yes', '1', 'True'] else 0)

    print("\nFraud Rate by Time of Day:")
    print(df.groupby('time_of_day')['fraud_numeric'].mean() * 100)
    
    print("\nInsight: A higher rate in 'Late Night' suggests automated bots or international fraudsters.")

# --- How to run it ---
analyze_time_patterns(df)

time_analysis = df.groupby('hour')['fraud_numeric'].mean().reset_index()

plt.figure(figsize=(8,5))
sns.lineplot(x = 'hour', y = 'fraud_numeric', data= time_analysis)
plt.title('"Late-Night Risk Surge: Peak Fraud Rates Identified During Off-Hours, Signaling Automated Bot Activity"')
plt.ylabel('Fraud Rate')
plt.xlabel('Hour')
plt.show()


# security checks analysis
def analyze_security_checks(df):
    print("--- Security Checks Analysis ---")

    # 1. Setup: Define your actual column names here
    avs_col = 'avs_match'  # avs: address Match
    cvv_col = 'cvv_result'    # cvv: cvv check
    three_col = 'three_ds_flag' # 3d_secure: verified by visa
    
    # Define our Fraud Column (make sure this matches your data)
    fraud_col = 'is_fraud'

    # Ensure we have the numeric fraud column for calculations
    if 'fraud_numeric' not in df.columns:
        df['fraud_numeric'] = df[fraud_col].apply(lambda x: 1 if str(x) in ['Yes', '1', 'True'] else 0)

    # 2. Analyze AVS (Address Verification)
    if avs_col in df.columns:
        print(f"\nFraud Rate by AVS ({avs_col}):")
        print(df.groupby(avs_col)['fraud_numeric'].mean() * 100)
    else:
        print(f"\nSkipping AVS: Column '{avs_col}' not found.")

    # 3. Analyze CVV (Card Security Code)
    if cvv_col in df.columns:
        print(f"\nFraud Rate by CVV ({cvv_col}):")
        print(df.groupby(cvv_col)['fraud_numeric'].mean() * 100)
    else:
        print(f"\nSkipping CVV: Column '{cvv_col}' not found.")

    # 4. Analyze 3DS (3D Secure / OTP)
    if three_col in df.columns:
        print(f"\nFraud Rate by 3D Secure ({three_col}):")
        print(df.groupby(three_col)['fraud_numeric'].mean() * 100)
    else:
        print(f"\nSkipping 3DS: Column '{three_col}' not found.")

    print("\nInsight: You expect to see much higher fraud rates when these checks 'Fail' or are 'Disabled'.")

# --- How to run it ---
analyze_security_checks(df)

df_grouped = pd.DataFrame({
    "Check": ['AVS','AVS','CVV','CVV','3D Secure','3D Secure'],
    "Status": ['Failed','Passed','Failed','Passed','Not Used','Used'],
    "Fraud_Rate": [9.666124, 0.764103, 10.605823, 0.974492, 6.752068, 0.958165]
})

plt.figure(figsize=(8,5))
sns.barplot(x= "Check", y= "Fraud_Rate", hue= "Status", data= df_grouped)
plt.title("Critical Vulnerability Identified: Fraud Rates Skyrocket Tenfold When AVS and CVV Security Checks Fail")
plt.ylabel("Fraud Rate (%)")
plt.tight_layout()
plt.show()

# combined EDA
def analyze_fraud_intersections(df):

    # 1. define "High Amount" threshold 

    high_amt_threshold = df['amount'].quantile(0.90)

    # Define the Intersection
    # Intersection 1: High amount + No 3ds 
    mask1 = (df['amount'] >= high_amt_threshold) & (df['three_ds_flag'] == 0)

    # Intersection 2: shipping type (e.g long distance) + cvv failed
    mask2 = (df['shipping_type'] == 'Long Distance') & (df['cvv_result'] == 0)

    # Intersection 3: Late Night (e.g 11 PM to 4 AM) + weak security
    mask3 = (df['hour'].isin([23,0,1,2,3,4])) & (df['three_ds_flag'])


    # calculate fraud rates for each
    results= []
    for name, mask in zip(['High Amt + No 3DS', 'Long Ship + CVV Fail','Late Night + Weak Security'],
                          [mask1, mask2, mask3]):
        subset = df[mask]
        fraud_rate = subset['is_fraud'].mean() * 100 if len(subset) > 0 else 0
        results.append({
            "Intersection": name,
            "Count": len(subset),
            "Fraud Rate (%)": round(fraud_rate, 2)
        })
    return pd.DataFrame(results)    

analyze_fraud_intersections(df)

# statitical analysis
def run_statistical_fraud_analysis(df):
    # 1. Secured vs Unsecured Analysis
    df['security_status'] = np.where(
        (df['three_ds_flag'] == 1) | (df['cvv_result'] == 1),
        'Secured',
        'Unsecured')
    
    security_stats = df.groupby('security_status')['is_fraud'].agg(['count','mean'])
    security_stats['mean'] = (security_stats['mean'] * 100).round(2)
    security_stats.columns = ['Total Transeactions', 'Fraud Rate (%)']


    # Normal vs Abnormal Amount Analysis

    Q1 = df['amount'].quantile(0.25)
    Q3 = df['amount'].quantile(0.95)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df['amount_type'] = np.where(
        (df['amount'] < lower_bound) | (df['amount'] > upper_bound),
        'Abnormal',
        'Normal'
    )

    amount_stats = df.groupby('amount_type')['is_fraud'].agg(['count', 'mean'])
    amount_stats['mean'] = (amount_stats['mean'] * 100).round(2)
    amount_stats.columns = ['Total Transaction', 'Fraud Rate (%)']

    return security_stats, amount_stats

security_stats, amount_stats = run_statistical_fraud_analysis(df)

# Graph 1: Fraud rate by Security Status----
plt.figure(figsize=(8,5))
sns.barplot(x= security_stats.index, y= security_stats['Fraud Rate (%)'], palette= 'coolwarm')

plt.title('Fraud Rate: Secured vs. Unsecured Transactions', fontsize= 14)
plt.ylabel('Fraud Rate (&)')
plt.xlabel('Security Status')
plt.show()

# Graph 2: Fraud Rate by Amount Type (Normal vs Abnormal) ---
plt.figure(figsize=(8, 5))
# we plot the index (Normal/abnormal) as fraud rate column
sns.barplot(x= amount_stats.index, y=amount_stats['Fraud Rate (%)'], palette='magma')
plt.title('Fraud Rate: Normal vs. Abnormal Transaction Amounts', fontsize= 14)
plt.ylabel('Fraud Rate (%)')
plt.xlabel('Transaction Amount Type')
plt.show()

def plot_fraud_analysis(df):
    plt.figure(figsize=(16,10))

    # Graph 1: Barplot for security
    plt.subplot(1,2,1)
    sns.barplot(x = 'security_status', y= 'is_fraud', data= df, estimator=lambda x: sum(x)/len(x) * 100, palette= 'viridis')
    plt.title('High Risk Profile Alert: Unsecured Abnorml Transactions and Failed Security Checks Drive Fraud Rates up to 40 %')
    plt.ylabel('Fraud Rate (%)')

    # Graph 2: Boxplot for amounts
    plt.subplot(1,2,2)
    sns.boxplot(x = 'is_fraud', y = 'amount', data= df, palette= 'viridis')
    plt.yscale('log')

    plt.tight_layout()
    plt.show()

plot_fraud_analysis(df)    

# hypothesis testing
def test_security_hypothesis(df):
    """
    Performs a Z-test to compare fraud proportions between secured and unsecured groups.
    """
    # 1. Prepare data groups
    secured = df[df['security_status'] == 'Secured']
    unsecured = df[df['security_status'] == 'Unsecured']
    
    # 2. Get counts of fraud (successes) and total observations (n)
    fraud_counts = [secured['is_fraud'].sum(), unsecured['is_fraud'].sum()]
    total_counts = [len(secured), len(unsecured)]
    
    # 3. Perform Z-test for proportions
    # H1: fraud in secured < fraud in unsecured ('smaller')
    z_stat, p_value = proportions_ztest(count=fraud_counts, nobs=total_counts, alternative='smaller')
    
    # 4. Calculate 95% Confidence Interval for the difference
    ci_low, ci_high = confint_proportions_2indep(
        fraud_counts[0], total_counts[0], 
        fraud_counts[1], total_counts[1], 
        method='wald'
    )
    
    # Results Summary
    print(f"--- Hypothesis Testing Results ---")
    print(f"Z-statistic: {z_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"95% CI for difference: [{ci_low:.4f}, {ci_high:.4f}]")
    
    if p_value < 0.05:
        print("Result: Reject H0. There is a significant difference in fraud rates.")
    else:
        print("Result: Fail to reject H0. No statistically significant difference.")


test_security_hypothesis(df)

# corrlelatio analysis
# Create the list for correlaion check
corr = ['amount','three_ds_flag','is_fraud']

# calculation correlation
correlation_matrix = df[corr].corr()

print("Correlation Matrix")
print(correlation_matrix)

# Visualize the Relationship
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot= True, cmap= 'coolwarm', fmt= ".2f")
plt.title("The strongest correlation with fraudulent activity is found with larger transaction amounts.Fraudulent actions persist despite the use of strong security measures.")
plt.show()

# feature engineering
# Transaction_rik_features
def apply_transaction_risk_feature(df):
    """
    Applies Business Logic for Transaction Risk Feature
    """
    # 1. Amount band (low, Medium, high)
    df['amount_band'] = pd.qcut(df['amount'], q=3, labels=['Low','Medium','High'])

    # 2. Amount Deviation Flag amt > 2* avg_amt_user
    # Calculate the average amount user per user
    avg_amt_user = df.groupby('user_id')['amount'].transform('mean')
    df['amount_deviation_flag'] = (df['amount'] > 2 * avg_amt_user).astype(int)

    # 3. Shipping Risk Level (Short, Medium, Long)
    # Assumes 'shipping_distance' exists; uses 'cut' to create 3 distance-based bins
    if 'shipping_distance_km' in df.columns:
        df['shipping_risk_level'] = pd.cut(df['shipping_distance_km'],
        bins= 3,
        labels= ['Short','Medium','Long'])

    # Time Risk Bucket (Normal/Night)
    if 'transaction_time' in df.columns:
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        hours = df['transaction_time'].dt.hour
        df['time_risk_bucket'] = np.where((hours >= 23) | (hours <= 5), 'Night','Normal')

    return df

apply_transaction_risk_feature(df)

# --- Graph 1: Fraud happen more at night ?
plt.figure(figsize=(8, 5))

sns.barplot(data= df, x= 'time_risk_bucket', y= 'is_fraud', palette= 'magma')

plt.title('Fraud Probability by time of day', fontsize= 14)
plt.ylabel('Fraud Probability (0.0 - 1.0)')
plt.xlabel('Time Risk Bucket')
plt.show()

# ---- Graph 2: Do higher Amounts mean higher fraud ?
plt.figure(figsize=(8, 5))
# amount_band (low/medium/high) vs is_fraud
sns.barplot(data= df, x= 'amount_band', y='is_fraud', palette='viridis')
plt.title('Fraud Probability by Amount Band', fontsize=14)
plt.ylabel('Fraud Probability (0.0 - 1.0)')
plt.xlabel('Amount Band')
plt.show()

# security strentght feature
# Security Strenght feature
# Assuming column are nameed 'avs_check' , 'cvv_result', 'three_ds_flag'
sec_security = ['avs_match','cvv_result','three_ds_flag']
security_score = df[sec_security].sum(axis= 1)

conditions = [
    (security_score == 3), # AVS + CVV + THREE_DS_FLAG
    (security_score == 2), # AVS + CVV
    (security_score == 1), # One Check
    (security_score == 0)  # None
]

choices = ['Strong','Medium','Weak','None']

df['security_level'] = np.select(conditions, choices, default= True)
df['security_level']

# risk segmantation logic
def classify_risk_segments(df):
    # High Amount: high amount combined with weak security
    conditions = [
        (df['amount_band'] == "High") & (df['security_level'].isin(['Weak','None'])),
    # Medium Risk: Abnormal behavior with only partial security checks 
        (df['amount_deviation_flag'] == 1) & (df['security_level'] == 'Medium'),
    # Low Risk: Normal behavior and strong security checks passed
        (df['amount_band'] == 'Low') & (df['security_level'] == 'Strong')    
    ]

    segments = ["High Risk", "Medium Risk", "Low Risk"]
    
    # Defaulting to 'Medium Risk' if it doesn't strictly meet Low/High criteria
    df['risk_segments'] = np.select(conditions, segments, default= 'Medium Risk')

    return df

classify_risk_segments(df)
# Define order and colors for a logical flow (Green -> Orange -> Red)
risk_order = ['Low Risk', 'Medium Risk', 'High Risk']
risk_colors = {'Low Risk': '#2ecc71', 'Medium Risk': '#f1c40f', 'High Risk': '#e74c3c'}

# --- Graph 1: How many transactions are in each Risk Segment? ---
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='risk_segments', order=risk_order, palette=risk_colors)

plt.title('Transaction Volume by Risk Segment', fontsize=15)
plt.xlabel('Risk Level')
plt.ylabel('Number of Transactions')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

# --- Graph 2: Did the "High Risk" segment actually catch the fraud? ---
plt.figure(figsize=(10, 5))
# This calculates the average of 'is_fraud' (0 or 1) to get the percentage
sns.barplot(data=df, x='risk_segments', y='is_fraud', order=risk_order, palette=risk_colors)

plt.title('Actual Fraud Rate per Risk Segment', fontsize=15)
plt.xlabel('Risk Level')
plt.ylabel('Fraud Probability (0 to 1)')
plt.show()


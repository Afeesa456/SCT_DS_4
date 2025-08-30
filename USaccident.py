import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("🔍 Downloading US Accidents dataset from Kaggle...")

try:
    import kagglehub
    
    path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
    print(f"✅ Dataset downloaded to: {path}")
    
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if csv_files:
        filename = os.path.join(path, csv_files[0])  # Use the first CSV file
        print(f"📄 Using file: {csv_files[0]}")
    else:
        print("❌ No CSV files found in downloaded dataset")
        exit()
        
except ImportError:
    print("❌ kagglehub not installed. Installing now...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

    path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
    print(f"✅ Dataset downloaded to: {path}")
    
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    filename = os.path.join(path, csv_files[0])
    print(f"📄 Using file: {csv_files[0]}")

except Exception as e:
    print(f"❌ Error downloading dataset: {e}")
    print("💡 Make sure you have Kaggle API credentials set up")
    exit()

    #1. Load Data

print("Loading dataset with sampling to save RAM...")

use_cols = [
    'ID','Start_Time','End_Time','Severity','State','Weather_Condition',
    'Temperature(F)','Visibility(mi)','Wind_Speed(mph)',
    'Start_Lat','Start_Lng'
]

dtype_dict = {
    'ID': 'category',
    'State': 'category', 
    'Weather_Condition': 'category',
    'Temperature(F)': 'float32',
    'Visibility(mi)': 'float32',
    'Wind_Speed(mph)': 'float32',
    'Start_Lat': 'float32',
    'Start_Lng': 'float32'
}

try:
    df = pd.read_csv(filename, usecols=use_cols, nrows=100000, dtype=dtype_dict)
    
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    df['Hour'] = df['Start_Time'].dt.hour
    df['Day_of_Week'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month_name()
    df['Year'] = df['Start_Time'].dt.year
    df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    
    df['Duration_Minutes'] = df['Duration_Minutes'].clip(0, 1440)  # Max 24 hours
    
    print(f"✅ Data sample loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"📊 Date range: {df['Start_Time'].min()} to {df['Start_Time'].max()}")
    print(f"🏛 States covered: {df['State'].nunique()}")
    print(f"🌤 Weather conditions: {df['Weather_Condition'].nunique()}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    print("💡 The file might have different column names or format")
    exit()

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'


# 2. Plot Accident Frequency by Hour
plt.figure(figsize=(12,6))
hourly_counts = df['Hour'].value_counts().sort_index()
sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette="viridis")
plt.title("🕐 Accidents by Hour of Day", fontsize=16, pad=20)
plt.xlabel("Hour of Day")
plt.ylabel("Number of Accidents")
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Plot Accidents by Day of Week
plt.figure(figsize=(12,6))
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
sns.countplot(data=df, x='Day_of_Week', order=day_order, palette="Set2")
plt.title("📅 Accidents by Day of Week", fontsize=16, pad=20)
plt.xlabel("Day of Week")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# 4. Plot Accidents by Month

plt.figure(figsize=(14,6))
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
sns.countplot(data=df, x='Month', order=month_order, palette="coolwarm")
plt.title("📆 Accidents by Month", fontsize=16, pad=20)
plt.xlabel("Month")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# 5. Severity Distribution

plt.figure(figsize=(10,6))
severity_counts = df['Severity'].value_counts().sort_index()
colors = ['#2ecc71', '#f39c12', '#e74c3c', '#8e44ad']
plt.pie(severity_counts.values, labels=[f'Severity {i}' for i in severity_counts.index], 
        autopct='%1.1f%%', colors=colors[:len(severity_counts)])
plt.title("⚠️ Accident Severity Distribution", fontsize=16, pad=20)
plt.axis('equal')
plt.tight_layout()
plt.show()


# 6. Weather Impact

plt.figure(figsize=(14,8))
# Clean weather data
df_clean_weather = df.dropna(subset=['Weather_Condition'])
top_weather = df_clean_weather['Weather_Condition'].value_counts().nlargest(10).index

sns.countplot(data=df_clean_weather[df_clean_weather['Weather_Condition'].isin(top_weather)], 
              y='Weather_Condition', order=top_weather, palette="cubehelix")
plt.title("🌦 Top 10 Weather Conditions in Accidents", fontsize=16, pad=20)
plt.xlabel("Number of Accidents")
plt.ylabel("Weather Condition")
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()


# 7. Correlation Heatmap

plt.figure(figsize=(10,8))
# Select numeric columns for correlation
numeric_cols = ['Severity','Temperature(F)','Visibility(mi)','Wind_Speed(mph)','Duration_Minutes']
correlation_data = df[numeric_cols].dropna()

if not correlation_data.empty:
    correlation_matrix = correlation_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title("🔗 Feature Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Not enough numeric data for correlation analysis")

# 8. Summary Statistics

print("\n📈 ANALYSIS SUMMARY")
print("="*50)
print(f"🔥 Peak accident hour: {df['Hour'].mode().iloc[0]}:00")
print(f"📅 Most dangerous day: {df['Day_of_Week'].mode().iloc[0]}")
print(f"🗓 Most dangerous month: {df['Month'].mode().iloc[0]}")
print(f"⚠️ Most common severity: Level {df['Severity'].mode().iloc[0]}")
if not df['Weather_Condition'].mode().empty:
    print(f"🌤 Most common weather: {df['Weather_Condition'].mode().iloc[0]}")
print(f"⏱️ Average accident duration: {df['Duration_Minutes'].mean():.1f} minutes")
print("="*50)

print("✅ Analysis complete! All visualizations generated successfully.")
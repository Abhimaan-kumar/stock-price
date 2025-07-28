import os
import pandas as pd

# Path to the stock_data folder
data_folder = "stock_data"

# Output folder to store processed files
merged_folder = "merged_data"
os.makedirs(merged_folder, exist_ok=True)

# Merge all CSVs into individual stock-specific cleaned files
def merge_and_clean():
    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_folder, filename)
            df = pd.read_csv(filepath)

            # Keep only useful columns and rename
            if 'Date' in df.columns and 'Close' in df.columns:
                df = df[['Date', 'Close']].dropna()
                df.columns = ['Date', 'Close']
                df['Date'] = pd.to_datetime(df['Date'])
                df.sort_values('Date', inplace=True)
                
                stock_name = filename.replace('.csv', '')
                output_path = os.path.join(merged_folder, f"{stock_name}.csv")
                df.to_csv(output_path, index=False)
                print(f"Merged and saved: {output_path}")
            else:
                print(f"Skipped: {filename} (Missing Date/Close columns)")

if __name__ == "__main__":
    merge_and_clean()

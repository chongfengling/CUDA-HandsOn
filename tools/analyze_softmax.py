import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def merge_csv_files(target_dir):
    csv_files = glob.glob(os.path.join(target_dir, "*.raw.csv"))
    
    if not csv_files:
        print("No .raw.csv files found.")
        return None

    all_data = []

    for file in csv_files:
        # Extract shape from filename: M1024_N8192.raw.csv -> M1024_N8192
        basename = os.path.basename(file)
        shape = basename.replace(".raw.csv", "")
        
        try:
            # Read the CSV
            df = pd.read_csv(file)
            
            # Filter out context rows
            df = df[df['section'] != 'context']
            
            # Keep only name, unit, and value
            df = df[['name', 'unit', 'value']].copy()
            
            # Create a combined metric name (e.g., "metric_name (unit)")
            # Some units might be NaN
            df['metric'] = df.apply(lambda row: f"{row['name']} ({row['unit']})" if pd.notna(row['unit']) and row['unit'] != '' else row['name'], axis=1)
            
            # Set metric as index
            df = df[['metric', 'value']].set_index('metric')
            
            # Rename value column to shape
            df.columns = [shape]
            
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if not all_data:
        print("No valid data extracted.")
        return None

    # Merge all dataframes on the metric index
    merged_df = pd.concat(all_data, axis=1)
    
    # Sort columns (shapes) alphabetically or numerically if possible
    # Let's just sort alphabetically for now
    merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

    # 1. Save merged table
    output_csv = os.path.join(target_dir, "merged_metrics.csv")
    merged_df.to_csv(output_csv)
    print(f"Saved merged metrics to {output_csv}")
    
    return merged_df

def plot_metrics(target_dir, merged_df):
    if merged_df is None or merged_df.empty:
        print("No data available for plotting.")
        return

    # 2. Select important metrics for plotting
    important_metrics = [
        "gpu__time_duration.sum (us)",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed (%)",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed (%)",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed (%)",
        "dram__bytes.sum (byte)" # Unit might be different, need to check
    ]

    # Find the actual names in the merged_df index since units might differ slightly
    actual_metrics_to_plot = []
    for m in important_metrics:
        # try to find a match
        matching_indices = [idx for idx in merged_df.index if str(idx).startswith(m.split(' ')[0])]
        if matching_indices:
            actual_metrics_to_plot.append(matching_indices[0])

    sns.set_theme(style="whitegrid")

    for metric in actual_metrics_to_plot:
        # Extract row
        row = merged_df.loc[metric]
        
        # Plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=row.index, y=row.values, color="skyblue")
        
        plt.title(f"Comparison of {metric} across Shapes", pad=20)
        plt.xlabel("Shape (M x N)")
        plt.ylabel("Value")
        plt.xticks(rotation=45, ha='right')
        
        # Add values on top of bars
        for i, v in enumerate(row.values):
            try:
                val = float(v)
                # Format to 2 decimal places if it's a float, otherwise integer
                label = f"{val:.2f}" if val % 1 != 0 else f"{int(val)}"
                ax.text(i, val, label, ha='center', va='bottom')
            except ValueError:
                pass
                
        plt.tight_layout()
        
        # Clean filename
        safe_metric_name = metric.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")
        plot_file = os.path.join(target_dir, f"plot_{safe_metric_name}.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved plot for {metric} to {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Softmax Profile Results")
    parser.add_argument("--dir", type=str, help="Directory containing the profile results")
    args = parser.parse_args()

    if args.dir:
        target_dir = args.dir
    else:
        # Find the latest directory
        base_dir = "kernels/softmax/profile_results"
        dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not dirs:
            print("No profile results found.")
            return
        target_dir = max(dirs) # Sorts by name which is a timestamp
    
    print(f"Analyzing directory: {target_dir}")

    merged_df = merge_csv_files(target_dir)
    
    # Temporarily disabling plot execution
    # if merged_df is not None:
    #     plot_metrics(target_dir, merged_df)

if __name__ == "__main__":
    main()

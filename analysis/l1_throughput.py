import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def analyze_l1_jitter():
    l1_dir = Path("output_data/L1_band_power_capture")
    output_dir = Path("output_data/analysis_reports")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    csvs = list(l1_dir.glob("out_*.csv"))
    if not csvs:
        print("No L1 csvs found.")
        return
        
    all_diffs = []
    
    for csv in csvs:
        df = pd.read_csv(csv)
        if "timestamp" not in df.columns or "data_origin" not in df.columns:
            continue
            
        origin = df["data_origin"].iloc[0]
        label = "Emulador Virtual" if origin == "virtual" else "Dispositivo Emotiv (Cortex API)"
            
        # Calculate time difference between consecutive packets in milliseconds
        diff = df["timestamp"].diff().dropna() * 1000
        
        # Remove massive outliers (like initial startup delay if any, > 1000ms)
        diff = diff[diff < 1000]
        
        df_diff = pd.DataFrame({
            "Intervalo (ms)": diff,
            "Fuente de Datos": label
        })
        all_diffs.append(df_diff)

    if not all_diffs:
        return
        
    merged = pd.concat(all_diffs)
    
    # Plotting
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(9, 6))
    
    # Target line at 125ms (8Hz)
    plt.axhline(125, color='red', linestyle='--', label='Objetivo Ideal (8 Hz / 125 ms)', alpha=0.7)
    
    sns.violinplot(
        data=merged, 
        x="Fuente de Datos", 
        y="Intervalo (ms)", 
        inner="quartile",
        scale="width"
    )
    
    plt.title("Estabilidad de Muestreo (Jitter): Virtual vs Emotiv", fontsize=14, pad=15)
    plt.ylabel("Intervalo entre muestras (ms)", fontsize=12)
    plt.xlabel("Origen de Datos L1", fontsize=12)
    plt.legend()
    plt.tight_layout()
    
    out_path = output_dir / "l1_jitter_analysis.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")
    
    # Print stats
    print("\nL1 Sampling Stats:")
    for label in merged["Fuente de Datos"].unique():
        subset = merged[merged["Fuente de Datos"] == label]["Intervalo (ms)"]
        print(f"{label}: Mean = {subset.mean():.2f}ms, Std = {subset.std():.2f}ms")

if __name__ == "__main__":
    analyze_l1_jitter()

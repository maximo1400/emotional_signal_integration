import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
UNITY_REPORTS_DIR = BASE_DIR.parent / "Afective-Museum" / "AffectiveReports"
OUTPUT_DATA_DIR = BASE_DIR / "output_data"
ANALYSIS_OUTPUT_DIR = OUTPUT_DATA_DIR / "analysis_reports"

# Configuration
CONFIG_IN_RANGE = (0.0, 2.0)  # L3 outputs 0, 1, 2
CONFIG_OUT_RANGE = (-1.0, 1.0)  # Map to -1 to 1


def map_va(
    val,
    in_min=CONFIG_IN_RANGE[0],
    in_max=CONFIG_IN_RANGE[1],
    out_min=CONFIG_OUT_RANGE[0],
    out_max=CONFIG_OUT_RANGE[1],
):
    return (val - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def copy_unity_reports():
    """Copies Unity AffectiveReports to Python's output_data."""
    if not UNITY_REPORTS_DIR.exists():
        print(f"Warning: Unity reports directory not found at {UNITY_REPORTS_DIR}")
        return

    target_dir = OUTPUT_DATA_DIR / "AffectiveReports"
    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(UNITY_REPORTS_DIR, target_dir)
    print(f"Copied Unity reports to {target_dir}")
    return target_dir


def find_matching_sessions():
    """Finds matching sessions across L1, L2, L3 and Unity."""
    l1_dir = OUTPUT_DATA_DIR / "L1_band_power_capture"
    l2_dir = OUTPUT_DATA_DIR / "L2_emot_state_estimation"
    l3_dir = OUTPUT_DATA_DIR / "L3_va_data_adaptation"
    unity_dir = OUTPUT_DATA_DIR / "AffectiveReports"

    # Collect all timestamps from L3 (the main layer we need for V-A)
    if not l3_dir.exists():
        print("Missing L3 directory.")
        return []

    l3_files = list(l3_dir.glob("out_*.csv"))
    matches = []
    processed_sessions = set()

    for l3_file in l3_files:
        try:
            ts_str = l3_file.stem.split("_")[1]
            session_id = ts_str
        except Exception:
            continue
            
        processed_sessions.add(session_id)

        # Match L1
        l1_csv = l1_dir / l3_file.name if l1_dir.exists() else None
        if l1_csv and not l1_csv.exists():
            l1_csv = None

        # Match L2
        l2_csv = l2_dir / l3_file.name if l2_dir.exists() else None
        if l2_csv and not l2_csv.exists():
            l2_csv = None

        # Match Unity
        unity_csv = None
        audio_csv = None
        light_csv = None
        data_start_ts = float(session_id)
        if unity_dir.exists():
            for unity_session in unity_dir.glob("Session_*"):
                all_ucsvs = list(unity_session.glob("out_*.csv"))
                ucsvs = [f for f in all_ucsvs if "audio" not in f.name and "light" not in f.name]
                if ucsvs:
                    try:
                        df_u = pd.read_csv(ucsvs[0], skipinitialspace=True)
                        df_u.columns = df_u.columns.str.strip()
                        if (
                            abs(
                                df_u["data_starting_timestamp"].iloc[0]
                                - float(session_id)
                            )
                            < 10
                        ):
                            unity_csv = ucsvs[0]
                            data_start_ts = df_u["data_starting_timestamp"].iloc[0]
                            
                            audio_files = list(unity_session.glob("audio_out_*.csv"))
                            if audio_files: audio_csv = audio_files[0]
                            
                            light_files = list(unity_session.glob("light_out_*.csv"))
                            if light_files: light_csv = light_files[0]
                            break
                    except Exception:
                        pass

        matches.append({
            "session_id": session_id,
            "l1_csv": l1_csv,
            "l2_csv": l2_csv,
            "l3_csv": l3_file,
            "unity_csv": unity_csv,
            "audio_csv": audio_csv,
            "light_csv": light_csv,
            "data_start_ts": data_start_ts,
        })
        print(
            f"Discovered Session {session_id} (L1: {bool(l1_csv)}, Unity: {bool(unity_csv)}, Audio: {bool(audio_csv)})"
        )
        
    # Also add Unity sessions that have no L3 data
    if unity_dir.exists():
        for unity_session in unity_dir.glob("Session_*"):
            ts_str = unity_session.name.split("_")[1]
            if ts_str in processed_sessions:
                continue
            
            unity_csv = None
            audio_csv = None
            light_csv = None
            data_start_ts = float(ts_str)
            
            all_ucsvs = list(unity_session.glob("out_*.csv"))
            ucsvs = [f for f in all_ucsvs if "audio" not in f.name and "light" not in f.name]
            if ucsvs:
                try:
                    df_u = pd.read_csv(ucsvs[0], skipinitialspace=True)
                    df_u.columns = df_u.columns.str.strip()
                    unity_csv = ucsvs[0]
                    data_start_ts = df_u["data_starting_timestamp"].iloc[0]
                except Exception:
                    pass
                    
            audio_files = list(unity_session.glob("audio_out_*.csv"))
            if audio_files: audio_csv = audio_files[0]
            
            light_files = list(unity_session.glob("light_out_*.csv"))
            if light_files: light_csv = light_files[0]
            
            matches.append({
                "session_id": ts_str,
                "l1_csv": None,
                "l2_csv": None,
                "l3_csv": None,
                "unity_csv": unity_csv,
                "audio_csv": audio_csv,
                "light_csv": light_csv,
                "data_start_ts": data_start_ts,
            })
            print(
                f"Discovered Unity-only Session {ts_str} (Audio: {bool(audio_csv)}, Light: {bool(light_csv)})"
            )

    return matches


def calculate_metrics_and_plot(session_match, session_label):
    session_id = session_match["session_id"]
    out_dir = ANALYSIS_OUTPUT_DIR / f"Session_{session_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_start = session_match["data_start_ts"]

    layers = []
    latencies = []

    # If L3 is missing, just do Audio/Light plots and return
    if session_match["l3_csv"] is None:
        plot_audio_light(session_match, session_label, out_dir, session_id, data_start)
        print(f"Generated Unity-only analysis for {session_label} (ID: {session_id})")
        return {"session_label": session_label, "layers": layers, "latencies": latencies}

    # Load data
    df_l3 = pd.read_csv(session_match["l3_csv"])
    df_l2 = pd.read_csv(session_match["l2_csv"]) if session_match["l2_csv"] else None

    # Handle L1 and ground truth
    df_l1 = None
    is_virtual = False
    if session_match["l1_csv"]:
        df_l1 = pd.read_csv(session_match["l1_csv"])
        if "data_origin" in df_l1.columns and df_l1["data_origin"].iloc[0] == "virtual":
            is_virtual = True

    # Handle Unity
    df_unity = None
    if session_match["unity_csv"]:
        df_unity = pd.read_csv(session_match["unity_csv"], skipinitialspace=True)
        df_unity.columns = df_unity.columns.str.strip()

    # Map L3 values
    for col in ["raw_valence", "raw_arousal", "smoothed_valence", "smoothed_arousal"]:
        if col in df_l3.columns:
            df_l3[col] = df_l3[col].apply(map_va)

    # Align by exact timestamp (merge L1 and L3 if virtual)
    if is_virtual and df_l1 is not None and "valence" in df_l1.columns:
        # The queues are strictly FIFO so row counts match exactly (no dropped frames in simulator).
        df_l1_renamed = df_l1[["valence", "arousal", "emot_state"]].rename(
            columns={"valence": "valence_true", "arousal": "arousal_true"}
        )
        merged_df = pd.concat(
            [df_l3.reset_index(drop=True), df_l1_renamed.reset_index(drop=True)], axis=1
        )
    else:
        merged_df = df_l3

    merged_df["time_relative"] = merged_df["timestamp"] - data_start

    # Basic Metrics
    metrics = {
        "Session": session_id,
        "Data_Origin": "Virtual" if is_virtual else "BCI Real",
        "Raw_Valence_Variance": merged_df["raw_valence"].var(),
        "Smoothed_Valence_Variance": merged_df["smoothed_valence"].var(),
        "Raw_Arousal_Variance": merged_df["raw_arousal"].var(),
        "Smoothed_Arousal_Variance": merged_df["smoothed_arousal"].var(),
        "Variance_Reduction_V_%": (
            merged_df["raw_valence"].var() - merged_df["smoothed_valence"].var()
        )
        / merged_df["raw_valence"].var()
        * 100
        if merged_df["raw_valence"].var() > 0
        else 0,
        "Variance_Reduction_A_%": (
            merged_df["raw_arousal"].var() - merged_df["smoothed_arousal"].var()
        )
        / merged_df["raw_arousal"].var()
        * 100
        if merged_df["raw_arousal"].var() > 0
        else 0,
    }

    # L2 Confidence
    if df_l2 is not None and "confidence" in df_l2.columns:
        metrics["Mean_Confidence"] = df_l2["confidence"].mean()
        metrics["Confidence_Std"] = df_l2["confidence"].std()
        
    layers = []
    latencies = []

    if df_l2 is not None:
        df_l2["latency_l1_to_l2"] = (
            df_l2["timestamp"] - df_l2["previous_layer_timestamp"]
        ) * 1000
        metrics["Latency_L1_to_L2_ms"] = df_l2["latency_l1_to_l2"].mean()
        layers.append("L1 -> L2")
        latencies.append(metrics["Latency_L1_to_L2_ms"])

    merged_df["latency_l2_to_l3"] = (
        merged_df["timestamp"] - merged_df["previous_layer_timestamp"]
    ) * 1000
    metrics["Latency_L2_to_L3_ms"] = merged_df["latency_l2_to_l3"].mean()
    layers.append("L2 -> L3")
    latencies.append(metrics["Latency_L2_to_L3_ms"])

    if df_unity is not None:
        df_unity["time_relative_data"] = df_unity["data_timestamp"] - data_start
        df_unity["latency_l3_to_unity"] = (
            df_unity["unity_timestamp"] - df_unity["data_timestamp"]
        ) * 1000
        metrics["Latency_L3_to_Unity_ms"] = df_unity["latency_l3_to_unity"].mean()
        metrics["Total_Adaptations"] = len(df_unity)
        layers.append("L3 -> Unity")
        latencies.append(metrics["Latency_L3_to_Unity_ms"])

    # Accuracy Metrics for Virtual
    if is_virtual and "valence_true" in merged_df.columns:
        metrics["MAE_Raw_Valence"] = (
            (merged_df["raw_valence"] - merged_df["valence_true"]).abs().mean()
        )
        metrics["MAE_Smoothed_Valence"] = (
            (merged_df["smoothed_valence"] - merged_df["valence_true"]).abs().mean()
        )
        metrics["MAE_Raw_Arousal"] = (
            (merged_df["raw_arousal"] - merged_df["arousal_true"]).abs().mean()
        )
        metrics["MAE_Smoothed_Arousal"] = (
            (merged_df["smoothed_arousal"] - merged_df["arousal_true"]).abs().mean()
        )

    pd.DataFrame([metrics]).to_csv(out_dir / "metrics_summary.csv", index=False)

    # --- Plot 2: Separate V and A Trajectories ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Valence - Swapped styles: raw is thick/solid, smoothed is thin/transparent
    ax1.plot(
        merged_df["time_relative"],
        merged_df["raw_valence"],
        alpha=0.9,
        label="Valence Cruda",
        color="darkblue",
        linewidth=2,
    )
    ax1.plot(
        merged_df["time_relative"],
        merged_df["smoothed_valence"],
        alpha=0.3,
        label="Valence Suavizada",
        color="blue",
    )
    if is_virtual and "valence_true" in merged_df.columns:
        ax1.plot(
            merged_df["time_relative"],
            merged_df["valence_true"],
            alpha=1.0,
            label="Ground Truth (Real)",
            color="limegreen",
            linestyle="--",
            linewidth=2,
        )
    ax1.set_ylabel("Valencia [-1, 1]")
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_title(f"Trayectoria de Valencia - {session_label}")
    ax1.legend(loc="upper right")
    ax1.grid(True)

    # Arousal - Swapped styles: raw is thick/solid, smoothed is thin/transparent
    ax2.plot(
        merged_df["time_relative"],
        merged_df["raw_arousal"],
        alpha=0.9,
        label="Arousal Cruda",
        color="darkred",
        linewidth=2,
    )
    ax2.plot(
        merged_df["time_relative"],
        merged_df["smoothed_arousal"],
        alpha=0.3,
        label="Arousal Suavizada",
        color="red",
    )
    if is_virtual and "arousal_true" in merged_df.columns:
        ax2.plot(
            merged_df["time_relative"],
            merged_df["arousal_true"],
            alpha=1.0,
            label="Ground Truth (Real)",
            color="limegreen",
            linestyle="--",
            linewidth=2,
        )
    ax2.set_ylabel("Activación [-1, 1]")
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_xlabel("Tiempo (s)")
    ax2.legend(loc="upper right")
    ax2.grid(True)

    # Mark Unity events
    if df_unity is not None:
        for idx, row in df_unity.iterrows():
            t = row["time_relative_data"]
            ax1.axvline(x=t, color="black", linestyle=":", alpha=0.5)
            ax1.text(t, 1.05, row["event"], rotation=90, color="black", fontsize=8)
            ax2.axvline(x=t, color="black", linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_dir / "va_trajectories_separated.png")
    plt.close()

    # --- Plot 3: Bubble Chart 2D en el Plano V-A ---
    plt.figure(figsize=(8, 8))

    # Agrupar datos discretos para el Bubble Chart
    l3_counts = (
        merged_df
        .groupby(["smoothed_valence", "smoothed_arousal"])
        .size()
        .to_frame(name="counts")
        .reset_index()
    )

    # Scatter plot con tamaño basado en la frecuencia (Bubble chart)
    sc = plt.scatter(
        l3_counts["smoothed_valence"],
        l3_counts["smoothed_arousal"],
        s=l3_counts["counts"] * 50,
        c=l3_counts["counts"],
        cmap="Blues",
        alpha=0.7,
        edgecolors="blue",
        label="Frecuencia L3",
        zorder=2,
    )

    plt.colorbar(sc, label="Número de Predicciones")

    if is_virtual and "valence_true" in merged_df.columns:
        true_counts = (
            merged_df
            .groupby(["valence_true", "arousal_true"])
            .size()
            .to_frame(name="counts")
            .reset_index()
        )
        plt.scatter(
            true_counts["valence_true"],
            true_counts["arousal_true"],
            s=true_counts["counts"] * 50,
            color="limegreen",
            marker="X",
            zorder=3,
            label="Frecuencia Ground Truth",
        )

    plt.axhline(0, color="black", linewidth=1, zorder=1)
    plt.axvline(0, color="black", linewidth=1, zorder=1)
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.xlabel("Valencia Suavizada")
    plt.ylabel("Activación Suavizada")
    plt.title(
        f"Distribución de Estados en el Plano V-A (Bubble Chart)\n{session_label}"
    )

    # Cuadrantes
    plt.text(
        0.5,
        0.5,
        "Q1 (Feliz/Excitado)",
        ha="center",
        va="center",
        alpha=0.3,
        fontsize=12,
    )
    plt.text(
        -0.5,
        0.5,
        "Q2 (Enojado/Frustrado)",
        ha="center",
        va="center",
        alpha=0.3,
        fontsize=12,
    )
    plt.text(
        -0.5,
        -0.5,
        "Q3 (Triste/Aburrido)",
        ha="center",
        va="center",
        alpha=0.3,
        fontsize=12,
    )
    plt.text(
        0.5,
        -0.5,
        "Q4 (Relajado/Calmado)",
        ha="center",
        va="center",
        alpha=0.3,
        fontsize=12,
    )

    # Custom legend for markers since scatter size varies
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Predicciones L3",
            markerfacecolor="blue",
            markersize=10,
            alpha=0.5,
        ),
    ]
    if is_virtual:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="X",
                color="w",
                label="Ground Truth",
                markerfacecolor="limegreen",
                markersize=10,
            )
        )
    plt.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()
    plt.savefig(
        out_dir / "va_density_heatmap.png"
    )  # Mantener nombre para no romper markdown
    plt.close()

    plot_audio_light(session_match, session_label, out_dir, session_id, data_start)

    print(f"Generated advanced analysis for {session_label} (ID: {session_id})")
    return {"session_label": session_label, "layers": layers, "latencies": latencies}

def plot_audio_light(session_match, session_label, out_dir, session_id, data_start):
    audio_csv = session_match.get("audio_csv")
    light_csv = session_match.get("light_csv")
    if not audio_csv and not light_csv:
        return
        
    fig, axes = plt.subplots(2 if (audio_csv and light_csv) else 1, 1, figsize=(15, 8 if (audio_csv and light_csv) else 4), sharex=True)
    if not isinstance(axes, np.ndarray): axes = [axes]
    ax_idx = 0
    
    cmap = plt.get_cmap("Set3")
    temple_colors = {}
    color_idx = 0
    
    # Load data
    df_audio = pd.read_csv(audio_csv) if audio_csv else pd.DataFrame()
    df_light = pd.read_csv(light_csv) if light_csv else pd.DataFrame()
    if not df_audio.empty: df_audio["time_relative"] = df_audio["data_timestamp"] - data_start
    if not df_light.empty: df_light["time_relative"] = df_light["data_timestamp"] - data_start
    
    df_ref = df_light if not df_light.empty else df_audio
    in_temple_times = df_ref.loc[(df_ref['temple'].notna()) & (df_ref['temple'] != 'None'), 'time_relative'].values
    
    if len(in_temple_times) == 0:
        return
        
    PADDING_SEC = 2.0
    intervals = [(t - PADDING_SEC, t + PADDING_SEC) for t in in_temple_times]
    merged = []
    for it in sorted(intervals):
        if not merged:
            merged.append(it)
        else:
            last = merged[-1]
            if it[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], it[1]))
            else:
                merged.append(it)
                
    GAP_VISUAL_WIDTH = 1.5
    warped_starts = []
    gap_centers = []
    current_w = 0.0
    
    t_min = min(df_audio['time_relative'].min() if not df_audio.empty else float('inf'),
                df_light['time_relative'].min() if not df_light.empty else float('inf'))
                
    if merged[0][0] - t_min > 2.0:
        gap_centers.append(GAP_VISUAL_WIDTH / 2)
        current_w = GAP_VISUAL_WIDTH
        
    for i, (start, end) in enumerate(merged):
        warped_starts.append(current_w)
        if i < len(merged) - 1:
            prev_w_end = current_w + (end - start)
            gap_centers.append(prev_w_end + GAP_VISUAL_WIDTH / 2)
            current_w = prev_w_end + GAP_VISUAL_WIDTH
            
    def apply_warp(df):
        warped = []
        keep_mask = []
        for t in df['time_relative'].values:
            in_int = False
            for i, (start, end) in enumerate(merged):
                if start <= t <= end:
                    in_int = True
                    w_t = warped_starts[i] + (t - start)
                    warped.append(w_t)
                    keep_mask.append(True)
                    break
            if not in_int:
                warped.append(None)
                keep_mask.append(False)
        df_out = df[keep_mask].copy()
        df_out['time_warped'] = [w for w, k in zip(warped, keep_mask) if k]
        return df_out

    def break_lines_at_gaps(df):
        if df.empty: return df
        df = df.reset_index(drop=True)
        diffs = df['time_warped'].diff()
        gap_idx = df.index[diffs > (GAP_VISUAL_WIDTH * 0.8)].tolist()
        if not gap_idx: return df
        nan_rows = []
        for idx in gap_idx:
            row = df.loc[idx].copy()
            for col in row.index:
                if col != 'time_warped':
                    row[col] = np.nan
            row['time_warped'] = df.loc[idx-1, 'time_warped'] + GAP_VISUAL_WIDTH / 2.0
            nan_rows.append((idx - 0.5, row))
        for float_idx, row in nan_rows:
            df.loc[float_idx] = row
        return df.sort_index().reset_index(drop=True)

    if not df_audio.empty: 
        df_audio = apply_warp(df_audio)
        df_audio = break_lines_at_gaps(df_audio)
    if not df_light.empty: 
        df_light = apply_warp(df_light)
        df_light = break_lines_at_gaps(df_light)
        
    def add_temple_backgrounds_and_gaps(ax, df):
        nonlocal color_idx
        for gc in gap_centers:
            ax.axvline(gc - 0.2, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
            ax.axvline(gc + 0.2, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
            ax.text(gc, 0.5, "//", transform=ax.get_xaxis_transform(), ha='center', va='center', fontsize=16, color='black', alpha=0.6, rotation=45)

        if 'temple' not in df.columns: return
        df['temple_block'] = (df['temple'] != df['temple'].shift()).cumsum()
        for block_id, group in df.groupby('temple_block'):
            temple_name = group['temple'].iloc[0]
            if pd.isna(temple_name) or temple_name == 'None': continue
            if temple_name not in temple_colors:
                temple_colors[temple_name] = cmap(color_idx % 12)
                color_idx += 1
            color = temple_colors[temple_name]
            ax.axvspan(group['time_warped'].min(), group['time_warped'].max(), color=color, alpha=0.4, label=f"Templo: {temple_name}")
            mid_point = group['time_warped'].min() + (group['time_warped'].max() - group['time_warped'].min()) / 2
            ax.text(mid_point, 1.05, temple_name, ha='center', va='bottom', fontsize=10, fontweight='bold')

    if not df_audio.empty:
        ax = axes[ax_idx]
        add_temple_backgrounds_and_gaps(ax, df_audio)
        
        ax.plot(df_audio["time_warped"], df_audio["global_volume"], label="Audio Global", linewidth=2, color='black')
        ax.plot(df_audio["time_warped"], df_audio["storm_volume"], label="Audio Storm (Q2)", alpha=0.7)
        ax.plot(df_audio["time_warped"], df_audio["aysor_volume"], label="Audio Aysor (Q1)", alpha=0.7)
        ax.plot(df_audio["time_warped"], df_audio["memoir_volume"], label="Audio Memoir (Q3/Q4)", alpha=0.7)
        
        ax.set_ylabel("Volumen Audio")
        ax.set_ylim(0, 1.2)
        ax.set_title(f"Dinámica de Audio en Entorno Virtual - {session_label}", pad=20)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(True, alpha=0.5)
        ax_idx += 1
        
    if not df_light.empty:
        ax = axes[ax_idx]
        add_temple_backgrounds_and_gaps(ax, df_light)
        
        def mix_color(r, g, b, i):
            if r == 0 and g == 0 and b == 0:
                return (i * 0.7, i * 0.7, i * 0.7)
            return (r * i, g * i, b * i)
            
        colors = [mix_color(r, g, b, i) for r,g,b,i in zip(df_light['target_color_r'], df_light['target_color_g'], df_light['target_color_b'], df_light['target_intensity'])]
        
        ax.scatter(df_light["time_warped"], df_light["target_intensity"], c=colors, s=50, label='Color Filtro', edgecolor='black', linewidth=0.5, zorder=3)
        ax.plot(df_light["time_warped"], df_light["target_intensity"], color='gray', alpha=0.3, zorder=2)
        
        ax.set_ylabel("Valor de Señal Lumínica")
        ax.set_ylim(-0.05, 1.2)
        ax.set_title(f"Dinámica de Luz (Color Ambiental) - {session_label}", pad=20)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.grid(True, alpha=0.5)
        
    df_ref_warped = df_light if not df_light.empty else df_audio
    if not df_ref_warped.empty:
        idx_samples = np.linspace(0, len(df_ref_warped)-1, min(10, len(df_ref_warped)), dtype=int)
        ticks_warped = df_ref_warped['time_warped'].iloc[idx_samples].values
        ticks_real = df_ref_warped['time_relative'].iloc[idx_samples].values
        axes[-1].set_xticks(ticks_warped)
        axes[-1].set_xticklabels([f"{t:.1f}s" for t in ticks_real])
        axes[-1].set_xlabel("Tiempo (s)")
        
    plt.tight_layout()
    plt.savefig(out_dir / "audio_light_dynamics.png", bbox_inches='tight')
    plt.close()


def main():
    print("Starting Output Analysis...")
    copy_unity_reports()
    matches = find_matching_sessions()

    if not matches:
        print("No matches found to analyze.")
        return

    virt_idx = 1
    emot_idx = 1

    all_latencies = []
    for match in matches:
        # Determine if virtual quickly
        is_virtual = False
        if match["l1_csv"]:
            try:
                df_l1_head = pd.read_csv(match["l1_csv"], nrows=1)
                if (
                    "data_origin" in df_l1_head.columns
                    and df_l1_head["data_origin"].iloc[0] == "virtual"
                ):
                    is_virtual = True
            except:
                pass

        if is_virtual:
            label = f"Virtual-{virt_idx}"
            virt_idx += 1
        else:
            label = f"Emotiv-{emot_idx}"
            emot_idx += 1

        lat_data = calculate_metrics_and_plot(match, label)
        if lat_data["latencies"]:
            for l, v in zip(lat_data["layers"], lat_data["latencies"]):
                all_latencies.append({
                    "Session": lat_data["session_label"],
                    "Capa": l,
                    "Latencia (ms)": v,
                })

    # --- Unified Latency Plot ---
    if all_latencies:
        df_lat = pd.DataFrame(all_latencies)
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df_lat,
            x="Capa",
            y="Latencia (ms)",
            hue="Session",
            marker="o",
            linewidth=2,
            markersize=8,
        )
        plt.title("Comparativa de Tiempos de Respuesta (Latencia) por Sesión")
        plt.ylabel("Latencia Promedio (ms)")
        plt.xlabel("Flujo de Datos")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(ANALYSIS_OUTPUT_DIR / "unified_latency_breakdown.png")
        plt.close()
        print("Generated unified latency plot.")

    print(f"Analysis complete. Reports saved to {ANALYSIS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()

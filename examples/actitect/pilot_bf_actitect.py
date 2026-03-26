import pandas as pd
import json
import matplotlib.pyplot as plt    
from pathlib import Path
from datetime import timedelta
from actitect.api import load, plot, process, compute_sleep_motor_features
# from actitect.processing.sleep import SleepDetector
# from actitect.utils import extract_segments

# DIRECTORY CONFIGURATION
project_root = Path("~/Documents/GitHub/actitect").expanduser()
raw_data_dir = project_root / "data" / "raw" / "pilot"
processed_dir = project_root / "data" / "processed"/ "pilot"
metadata_path = project_root / "data" / "meta" / "metadata_pilot.csv"
df_meta = pd.read_csv(metadata_path, sep=';')

error_log_path = processed_dir / "processing_errors.txt"
# all_summaries = []
# all_features = []

# MAIN PROCESSING LOOP
for _, row in df_meta.iterrows():
    subject_id = str(row['ID'])
    filename = str(row['filename'])

    sub_proc_dir = processed_dir / subject_id
    features_csv = sub_proc_dir / f"{subject_id}_motor_features.csv"
    # summary_csv = sub_proc_dir / f"{subject_id}_sleep_summary.csv"

    # RECOVERY LOGIC: Load existing data if it exists
    if features_csv.exists():
        print(f"Skipping {subject_id}: Analysis already exists. Loading...")
        # try:
        #     ext_feat = pd.read_csv(features_csv)
        #     ext_sum = pd.read_csv(summary_csv)
            
        #     if 'subject_id' not in ext_feat.columns: ext_feat.insert(0, 'subject_id', subject_id)
        #     if 'subject_id' not in ext_sum.columns: ext_sum.insert(0, 'subject_id', subject_id)
            
        #     all_features.append(ext_feat)
        #     all_summaries.append(ext_sum)
        # except Exception as e:
        #     print(f"Warning: Failed to reload {subject_id}: {e}")
        continue
    
    # PROCESSING NEW SUBJECTS
    subject_raw_folder = raw_data_dir / subject_id
    cwa_file = subject_raw_folder / filename
    sub_proc_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n>>> Starting Subject: {subject_id}")

    try:
        if cwa_file.exists():
            print(f"Loading .cwa for {subject_id}...")
            raw_df, _ = load(cwa_file, subject_id=subject_id)
        else:
            print(f"File not found: {cwa_file}")
            continue
        
        # PROCESS & SEGMENT
        print(f"Running ActiTect segmentation...")
        raw_df, _ = load(cwa_file, subject_id=subject_id)
        processed_df, info = process(raw_df, subject_id=subject_id)

        # # --- DETAILED SLEEP SUMMARY (SE, WASO, etc.) ---
        # print(f"Calculating detailed sleep stats...")
        # sptws_df = extract_segments(processed_df, column='sptw', condition=True)
        # sleep_bouts_df = extract_segments(processed_df, column='sleep_bout', condition=True)

        # if not sptws_df.empty:
        #     # The detailed calculator
        #     summary_df = SleepDetector.sptw_stats(sptws_df, sleep_bouts_df)
            
        #     # Merge selection flags from info
        #     sptw_meta = pd.DataFrame.from_dict(info['processing']['sleep_segmentation']['sptws'], orient='index')
        #     summary_df = summary_df.merge(sptw_meta[['overnight', 'selected']], left_index=True, right_index=True)
            
        #     summary_df.insert(0, 'subject_id', subject_id)
        #     summary_df.to_csv(summary_csv)
        #     all_summaries.append(summary_df)

        # SAVE METADATA
        with open(sub_proc_dir / f"{subject_id}_info.json", 'w') as f:
            json.dump(info, f, indent=4, default=str)

        # VISUALIZE (7-day subset)
        print(f"Generating 7-day plot...")
        plot_start = processed_df.index.min()
        plot_end = plot_start + timedelta(days=7)
        
        fig = plot(processed_df.loc[plot_start:plot_end])
        fig.savefig(sub_proc_dir / f"{subject_id}_7day_visualization.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

        # MOTOR FEATURES
        print(f"Computing motor features...")
        features = compute_sleep_motor_features(processed_df)
        features.to_csv(sub_proc_dir / f"{subject_id}_motor_features.csv", index=False)
        
        # MEMORY CLEANUP
        del raw_df, processed_df
        print(f"Success: Subject {subject_id} complete.")

    except Exception as e:
        error_msg = f"!!! ERROR processing {subject_id}: {e}"
        print(error_msg)
        with open(error_log_path, "a") as f:
            f.write(f"{subject_id}: {str(e)}\n")

# # --- 4. CREATE PILOT MASTER FILES ---
# print("\n--- Generating Master Tables ---")

# if all_summaries and all_features:
#     master_sleep = pd.concat(all_summaries, ignore_index=True)
#     master_feat = pd.concat(all_features, ignore_index=True)

#     # Filter for quality nights only
#     clean_sleep = master_sleep[master_sleep['selected'] == True].copy()
    
#     # Match features to the clean sleep windows
#     valid_starts = clean_sleep['start_time'].astype(str).tolist()
#     clean_feat = master_feat[master_feat['start_time'].astype(str).isin(valid_starts)].copy()

#     clean_sleep.to_csv(processed_dir / "PILOT_sleep_metrics.csv", index=False)
#     clean_feat.to_csv(processed_dir / "PILOT_motor_features.csv", index=False)
    
#     print(f"Saved: PILOT_sleep_metrics.csv and PILOT_motor_features.csv")

print("\n--- ALL SUBJECTS FINISHED ---")
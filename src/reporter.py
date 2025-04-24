import pandas as pd
import matplotlib.pyplot as plt
import os

class ReportManager:
    def __init__(self, output_dir="data/output"):
        self.records = []  # list of dicts
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def log_event(self, vehicle_id, frame_idx, timestamp, traffic_light_state):
        self.records.append({
            "vehicle_id": vehicle_id,
            "frame": frame_idx,
            "timestamp_sec": round(timestamp, 2),
            "traffic_light": traffic_light_state
        })

    def save_csv(self, filename="vehicle_log.csv"):
        df = pd.DataFrame(self.records)
        csv_path = os.path.join(self.output_dir, filename)
        df.to_csv(csv_path, index=False)
        print(f"[✔] CSV saved to: {csv_path}")
        return df

    def generate_report(self, df, filename="report.png"):
        df['minute'] = (df['timestamp_sec'] // 60).astype(int)
        counts_per_minute = df.groupby('minute').size()

        plt.figure(figsize=(10, 5))
        counts_per_minute.plot(kind='bar')
        plt.title("Vehicles per Minute")
        plt.xlabel("Minute")
        plt.ylabel("Number of Vehicles")
        plt.tight_layout()

        report_path = os.path.join(self.output_dir, filename)
        plt.savefig(report_path)
        print(f"[✔] Report saved to: {report_path}")

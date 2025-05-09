import numpy as np
import os

# ---------- CONFIG ----------
SAVE_DIR = "CSP_ATTEMPTS/data"
filename = input("Enter .npz filename (without extension): ").strip()
file_name = filename
if not filename.endswith(".npz"):
    filename += ".npz"
npz_path = os.path.join(SAVE_DIR, filename)
# ----------------------------

# Load and inspect data
data = np.load(npz_path, allow_pickle=True)

print(f"\n📦 Contents of {filename}:")
for key in data.files:
    arr = data[key]
    print(f"  🔹 {key}: shape={arr.shape}, dtype={arr.dtype}")

# Try to export 2D arrays to CSV
for key in data.files:
    arr = data[key]
    if key == "events":
        # Export events as text CSV
        csv_filename = f"{file_name}_events.csv"
        csv_path = os.path.join(SAVE_DIR, csv_filename)
        print(f"\n🗂 Exporting events to CSV: {csv_path}")
        with open(csv_path, 'w') as f:
            f.write("label,counter_start,duration\n")
            for evt in arr:
                f.write(f"{evt[0]},{int(evt[1])},{float(evt[2])}\n")
        print(f"✅ Saved {len(arr)} events")

    elif arr.ndim == 2 and arr.dtype.kind in "fi":  # float or int
        csv_filename = f"{file_name}_{key}.csv"
        csv_path = os.path.join(SAVE_DIR, csv_filename)

        # Try to get matching column names
        if "channel_names" in data and len(data["channel_names"]) == arr.shape[1]:
            header = ",".join(data["channel_names"])
        else:
            header = ",".join([f"ch{i+1}" for i in range(arr.shape[1])])

        print(f"\n📄 Saving 2D array '{key}' to CSV: {csv_path}")
        np.savetxt(csv_path, arr, delimiter=",", fmt="%.6f", header=header, comments="")
        print(f"✅ Saved {arr.shape[0]} rows × {arr.shape[1]} cols")

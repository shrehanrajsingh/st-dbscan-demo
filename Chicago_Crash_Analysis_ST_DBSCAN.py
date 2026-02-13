import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time as time_module
import colorsys
import folium
from folium.plugins import HeatMap

plt.rcParams.update(
    {
        "figure.figsize": (12, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)
sns.set_style("whitegrid")
sns.set_palette("deep")

print("All libraries loaded successfully.")

df = pd.read_csv("crashes_jan_2024_2025.csv")
print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

df.head()
df.info()
df.describe()

# Missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
    "Missing %", ascending=False
)
print(f"Columns with missing values: {len(missing_df)}")

# Duplicates
dup_count = df.duplicated(subset=["crash_record_id"]).sum()
print(f"Duplicate crash_record_id entries: {dup_count}")
full_dup = df.duplicated().sum()
print(f"Fully duplicated rows: {full_dup}")


df["crash_date"] = pd.to_datetime(df["crash_date"], format="mixed", errors="coerce")
df["date_police_notified"] = pd.to_datetime(
    df["date_police_notified"], format="mixed", errors="coerce"
)

invalid_dates = df["crash_date"].isnull().sum()
print(f"Invalid/unparseable crash_date entries: {invalid_dates}")
print(f"Date range: {df['crash_date'].min()} to {df['crash_date'].max()}")

# Filter valid coordinates within Chicago bounding box
print(f"Records with missing latitude: {df['latitude'].isnull().sum()}")
print(f"Records with missing longitude: {df['longitude'].isnull().sum()}")

df_geo = df.dropna(subset=["latitude", "longitude"]).copy()
print(f"\nRecords with valid coordinates: {len(df_geo)} / {len(df)}")

chicago_mask = (
    (df_geo["latitude"] >= 41.6)
    & (df_geo["latitude"] <= 42.1)
    & (df_geo["longitude"] >= -88.0)
    & (df_geo["longitude"] <= -87.5)
)
outside_chicago = (~chicago_mask).sum()
print(f"Records outside Chicago bounding box: {outside_chicago}")

df_geo = df_geo[chicago_mask].copy()
print(f"Final geospatial records: {len(df_geo)}")


# Outlier detection (informational only, outliers retained as genuine severe crashes)
def detect_outliers_iqr(series, name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    print(
        f"{name}: Q1={Q1}, Q3={Q3}, IQR={IQR}, bounds=[{lower:.1f}, {upper:.1f}], outliers={len(outliers)}"
    )
    return outliers


for col in ["injuries_total", "injuries_fatal", "num_units"]:
    if col in df_geo.columns:
        detect_outliers_iqr(df_geo[col].dropna(), col)

# Feature engineering
df_geo["hour"] = df_geo["crash_hour"]
df_geo["day_of_week"] = df_geo["crash_day_of_week"]
df_geo["day_name"] = df_geo["crash_date"].dt.day_name()
df_geo["month"] = df_geo["crash_month"]
df_geo["date_only"] = df_geo["crash_date"].dt.date
df_geo["is_weekend"] = df_geo["day_of_week"].isin([1, 7]).astype(int)
df_geo["is_peak"] = df_geo["hour"].apply(
    lambda h: 1 if (7 <= h <= 9) or (16 <= h <= 19) else 0
)
df_geo["response_time_min"] = (
    df_geo["date_police_notified"] - df_geo["crash_date"]
).dt.total_seconds() / 60

print("Temporal features engineered:")
print(f"  - Weekend crashes: {df_geo['is_weekend'].sum()}")
print(f"  - Peak hour crashes: {df_geo['is_peak'].sum()}")
print(f"  - Median response time: {df_geo['response_time_min'].median():.1f} minutes")


chicago_center = [df_geo["latitude"].mean(), df_geo["longitude"].mean()]

df_cluster = (
    df_geo[
        [
            "latitude",
            "longitude",
            "crash_date",
            "crash_hour",
            "injuries_total",
            "first_crash_type",
        ]
    ]
    .dropna()
    .copy()
)

# Convert crash_date to numeric (hours since earliest crash)
t_min = df_cluster["crash_date"].min()
df_cluster["time_hours"] = (
    df_cluster["crash_date"] - t_min
).dt.total_seconds() / 3600.0

print(f"\nClustering dataset: {len(df_cluster)} records")
print(
    f"Time span: {df_cluster['time_hours'].min():.1f} to {df_cluster['time_hours'].max():.1f} hours"
)
print(
    f"Spatial extent: lat [{df_cluster['latitude'].min():.4f}, {df_cluster['latitude'].max():.4f}]"
)
print(
    f"                lon [{df_cluster['longitude'].min():.4f}, {df_cluster['longitude'].max():.4f}]"
)


def st_dbscan(lat, lon, time, eps1, eps2, min_samples):
    """
    ST-DBSCAN: Spatio-Temporal DBSCAN clustering.

    Parameters
    ----------
    lat : array-like, latitude values
    lon : array-like, longitude values
    time : array-like, temporal values (numeric, e.g., hours)
    eps1 : float, spatial epsilon in degrees
    eps2 : float, temporal epsilon in same units as time
    min_samples : int, minimum points to form a cluster

    Returns
    -------
    labels : array of cluster labels (-1 = noise)

    Algorithm
    ---------
    For each unvisited point p:
      1. Find neighbors within eps1 (spatial) AND eps2 (temporal)
      2. If |neighbors| >= min_samples, p is a core point -> expand cluster
      3. Otherwise, label p as noise (may later be claimed by a cluster)
    """
    n = len(lat)
    labels = np.full(n, -1)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    lat_arr = np.array(lat)
    lon_arr = np.array(lon)
    time_arr = np.array(time)

    def get_neighbors(idx):
        """Find all points within eps1 (spatial) and eps2 (temporal) of point idx."""
        spatial_dist = np.sqrt(
            (lat_arr - lat_arr[idx]) ** 2 + (lon_arr - lon_arr[idx]) ** 2
        )
        temporal_dist = np.abs(time_arr - time_arr[idx])
        mask = (spatial_dist <= eps1) & (temporal_dist <= eps2)
        return np.where(mask)[0]

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = get_neighbors(i)

        if len(neighbors) < min_samples:
            continue

        labels[i] = cluster_id
        seed_set = list(neighbors)
        seed_set.remove(i)

        j = 0
        while j < len(seed_set):
            q = seed_set[j]
            if not visited[q]:
                visited[q] = True
                q_neighbors = get_neighbors(q)
                if len(q_neighbors) >= min_samples:
                    for nn in q_neighbors:
                        if nn not in seed_set:
                            seed_set.append(nn)
            if labels[q] == -1:
                labels[q] = cluster_id
            j += 1

        cluster_id += 1

    return labels


print("ST-DBSCAN function defined successfully.")


EPS1 = 0.01  # ~1.1 km spatial radius
EPS2 = 4.0  # 4-hour temporal window
MIN_SAMPLES = 8

print(f"Running ST-DBSCAN with eps1={EPS1}, eps2={EPS2}, min_samples={MIN_SAMPLES}...")
print(f"Dataset size: {len(df_cluster)} points")

start = time_module.time()
cluster_labels = st_dbscan(
    df_cluster["latitude"].values,
    df_cluster["longitude"].values,
    df_cluster["time_hours"].values,
    eps1=EPS1,
    eps2=EPS2,
    min_samples=MIN_SAMPLES,
)
elapsed = time_module.time() - start

df_cluster["cluster"] = cluster_labels
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = (cluster_labels == -1).sum()

print(f"\nCompleted in {elapsed:.1f} seconds")
print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
print(
    f"Clustered points: {len(cluster_labels) - n_noise} ({(len(cluster_labels)-n_noise)/len(cluster_labels)*100:.1f}%)"
)

# Cluster summary
if n_clusters > 0:
    cluster_summary = (
        df_cluster[df_cluster["cluster"] != -1]
        .groupby("cluster")
        .agg(
            count=("latitude", "size"),
            mean_lat=("latitude", "mean"),
            mean_lon=("longitude", "mean"),
            mean_hour=("crash_hour", "mean"),
            total_injuries=("injuries_total", "sum"),
            time_span_hrs=("time_hours", lambda x: x.max() - x.min()),
            top_crash_type=(
                "first_crash_type",
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A",
            ),
        )
        .sort_values("count", ascending=False)
    )

    print("\nCluster Summary (sorted by size):")
    print("=" * 90)
    for idx, row in cluster_summary.iterrows():
        print(
            f"Cluster {idx}: {row['count']} crashes | "
            f"Center: ({row['mean_lat']:.4f}, {row['mean_lon']:.4f}) | "
            f"Avg Hour: {row['mean_hour']:.1f} | "
            f"Injuries: {row['total_injuries']:.0f} | "
            f"Span: {row['time_span_hrs']:.1f}h | "
            f"Type: {row['top_crash_type']}"
        )
else:
    print(
        "No clusters found. Consider adjusting parameters (larger eps1/eps2 or smaller min_samples)."
    )


clustered = df_cluster[df_cluster["cluster"] != -1]
noise = df_cluster[df_cluster["cluster"] == -1]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Spatial view
axes[0].scatter(
    noise["longitude"], noise["latitude"], c="lightgray", s=5, alpha=0.3, label="Noise"
)
if len(clustered) > 0:
    scatter = axes[0].scatter(
        clustered["longitude"],
        clustered["latitude"],
        c=clustered["cluster"],
        cmap="tab20",
        s=15,
        alpha=0.7,
    )
    plt.colorbar(scatter, ax=axes[0], label="Cluster ID")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].set_title("ST-DBSCAN Clusters (Spatial View)")
axes[0].legend(loc="upper right")

# Temporal view
axes[1].scatter(
    noise["time_hours"], noise["latitude"], c="lightgray", s=5, alpha=0.3, label="Noise"
)
if len(clustered) > 0:
    scatter2 = axes[1].scatter(
        clustered["time_hours"],
        clustered["latitude"],
        c=clustered["cluster"],
        cmap="tab20",
        s=15,
        alpha=0.7,
    )
    plt.colorbar(scatter2, ax=axes[1], label="Cluster ID")
axes[1].set_xlabel("Time (hours since first crash)")
axes[1].set_ylabel("Latitude")
axes[1].set_title("ST-DBSCAN Clusters (Temporal View)")
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()


m_clusters = folium.Map(
    location=chicago_center, zoom_start=11, tiles="CartoDB positron"
)


def get_cluster_colors(n):
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(
            "#{:02x}{:02x}{:02x}".format(
                int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
            )
        )
    return colors


if n_clusters > 0:
    cluster_colors = get_cluster_colors(n_clusters)

    # Noise points (small, gray)
    for _, row in noise.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="gray",
            fill=True,
            fill_opacity=0.2,
        ).add_to(m_clusters)

    # Clustered points
    for _, row in clustered.iterrows():
        cid = int(row["cluster"])
        color = cluster_colors[cid % len(cluster_colors)]
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Cluster {cid} | Hour: {row['crash_hour']} | Type: {row['first_crash_type']}",
        ).add_to(m_clusters)

    # Cluster center markers
    for cid, row in cluster_summary.iterrows():
        color = cluster_colors[int(cid) % len(cluster_colors)]
        folium.Marker(
            location=[row["mean_lat"], row["mean_lon"]],
            icon=folium.Icon(color="black", icon="info-sign"),
            popup=f"<b>Cluster {cid}</b><br>{row['count']} crashes<br>Avg Hour: {row['mean_hour']:.0f}<br>Injuries: {row['total_injuries']:.0f}",
        ).add_to(m_clusters)
else:
    for _, row in df_cluster.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            color="blue",
            fill=True,
            fill_opacity=0.3,
        ).add_to(m_clusters)

m_clusters.save("st_dbscan_clusters_map.html")
print("Folium cluster map saved to: st_dbscan_clusters_map.html")

from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
    try:
        from shutil import which
        import imageio_ffmpeg
        _HAS_IMAGEIO_FFMPEG = which("ffmpeg") is not None
    except ImportError:
        _HAS_IMAGEIO_FFMPEG = False
except ImportError:
    _HAS_IMAGEIO = False
    _HAS_IMAGEIO_FFMPEG = False

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a monthly GIF of shark trajectories and ocean eddies.")
    parser.add_argument("--sharks", required=True, type=Path, help="Path to the raw shark tracking CSV file.")
    parser.add_argument("--eddy-grid", required=True, type=Path, help="Path to the monthly gridded eddy data CSV.")
    parser.add_argument("--output-gif", required=True, type=Path, help="Output path for the generated GIF file.")
    parser.add_argument("--output-mp4", type=Path, default=None, help="Optional output path for an MP4 video file.")
    parser.add_argument("--mode", choices=["snapshot", "window"], default="snapshot", help="Frame mode: 'snapshot' for current month, 'window' for a trailing number of months.")
    parser.add_argument("--step-months", type=int, default=1, help="Number of months to advance per frame.")
    parser.add_argument("--tail-months", type=int, default=3, help="Window size in months for 'window' mode.")
    parser.add_argument("--lon-range", type=float, nargs=2, default=(-98.0, -80.0), metavar=("MIN_LON", "MAX_LON"), help="Longitude range for the map.")
    parser.add_argument("--lat-range", type=float, nargs=2, default=(18.0, 31.0), metavar=("MIN_LAT", "MAX_LAT"), help="Latitude range for the map.")
    parser.add_argument("--fps", type=int, default=4, help="Frames per second for the output animation.")
    parser.add_argument("--dpi", type=int, default=130, help="Dots per inch for rendering each frame.")
    parser.add_argument("--figsize", type=float, nargs=2, default=(12.0, 9.0), metavar=("WIDTH", "HEIGHT"), help="Figure size in inches.")
    parser.add_argument("--strict-whale-shark", action="store_true", help="Filter data strictly for 'Rhincodon typus' if taxonomy columns exist.")
    return parser.parse_args()

def filter_shark_species(df: pd.DataFrame, strict_whale_shark: bool) -> pd.DataFrame:
    taxonomy_cols = [c for c in ["individual-taxon-canonical-name", "taxon", "species", "study-name"] if c in df.columns]
    if not taxonomy_cols:
        return df

    text_content = df[taxonomy_cols].astype(str).agg(" ".join, axis=1)
    manta_mask = text_content.str.contains(r"\b(manta|mobula|devil\s*ray|mantarraya)\b", case=False, regex=True)
    filtered_df = df.loc[~manta_mask].copy()

    if strict_whale_shark:
        rhincodon_mask = filtered_df[taxonomy_cols].astype(str).agg(" ".join, axis=1).str.contains(r"\bRhincodon\s+typus\b", case=False, regex=True)
        filtered_df = filtered_df.loc[rhincodon_mask].copy()

    return filtered_df

def load_shark_data(csv_path: Path, bbox: Tuple[float, float, float, float], strict_whale_shark: bool) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    df = filter_shark_species(df, strict_whale_shark)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["location-lat"] = pd.to_numeric(df["location-lat"], errors="coerce")
    df["location-long"] = pd.to_numeric(df["location-long"], errors="coerce")
    df = df.dropna(subset=["timestamp", "location-lat", "location-long"]).copy()

    id_col = "individual-local-identifier" if "individual-local-identifier" in df.columns else "tag-local-identifier"
    df = df.sort_values([id_col, "timestamp"])
    
    min_lon, max_lon, min_lat, max_lat = bbox
    df = df[(df["location-long"].between(min_lon, max_lon)) & (df["location-lat"].between(min_lat, max_lat))].copy()
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    
    return df, id_col

def load_eddy_grid_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Eddy grid file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if "month" not in df.columns:
        raise ValueError("Eddy CSV must contain a 'month' column.")
        
    return df

def get_month_sequence(df_sharks: pd.DataFrame, step_months: int) -> list[str]:
    min_month = df_sharks["month"].min()
    max_month = df_sharks["month"].max()
    
    sequence = pd.period_range(min_month, max_month, freq="M")
    
    if step_months > 1:
        sequence = sequence[::step_months]
        if sequence[-1] != pd.Period(max_month):
            sequence = sequence.append(pd.PeriodIndex([pd.Period(max_month)]))
            
    return sequence.astype(str).tolist()

def convert_figure_to_image(fig: plt.Figure, dpi: int) -> np.ndarray:
    if not _HAS_IMAGEIO:
        return np.array([])
        
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        image_array = imageio.imread(buf)
        
    return image_array

def draw_monthly_frame(ax: plt.Axes, sharks_data: pd.DataFrame, id_col: str, eddies_data: pd.DataFrame, bbox: Tuple[float, float, float, float], title: str):
    for _, group in sharks_data.groupby(id_col, sort=True):
        if len(group) >= 2:
            ax.plot(group["location-long"].values, group["location-lat"].values, linewidth=1, alpha=0.95)
            ax.scatter(group["location-long"].values[-1], group["location-lat"].values[-1], s=16, alpha=1.0)
        else:
            ax.scatter(group["location-long"].values, group["location-lat"].values, s=16, alpha=1.0)

    if not eddies_data.empty:
        if {"eddy_count_cyclonic", "eddy_count_anticyclonic"}.issubset(eddies_data.columns):
            cyclonic_eddies = eddies_data[eddies_data["eddy_count_cyclonic"] > 0]
            if not cyclonic_eddies.empty:
                sizes = 10 + 2.0 * np.sqrt(cyclonic_eddies["eddy_count_cyclonic"].to_numpy())
                ax.scatter(cyclonic_eddies["lon"].to_numpy(), cyclonic_eddies["lat"].to_numpy(), s=sizes, alpha=0.35, marker="v", linewidths=0.0, c="blue")
            
            anticyclonic_eddies = eddies_data[eddies_data["eddy_count_anticyclonic"] > 0]
            if not anticyclonic_eddies.empty:
                sizes = 10 + 2.0 * np.sqrt(anticyclonic_eddies["eddy_count_anticyclonic"].to_numpy())
                ax.scatter(anticyclonic_eddies["lon"].to_numpy(), anticyclonic_eddies["lat"].to_numpy(), s=sizes, alpha=0.35, marker="^", linewidths=0.0, c="red")
        else:
            total_eddies = eddies_data[eddies_data.get("eddy_count_total", pd.Series(0)) > 0]
            if not total_eddies.empty:
                sizes = 10 + 2.0 * np.sqrt(total_eddies["eddy_count_total"].to_numpy())
                ax.scatter(total_eddies["lon"].to_numpy(), total_eddies["lat"].to_numpy(), s=sizes, alpha=0.35, linewidths=0.0)

    min_lon, max_lon, min_lat, max_lat = bbox
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

def main():
    args = parse_arguments()
    bbox = (args.lon_range[0], args.lon_range[1], args.lat_range[0], args.lat_range[1])

    shark_df, id_col = load_shark_data(args.sharks, bbox, args.strict_whale_shark)
    eddy_df = load_eddy_grid_data(args.eddy_grid)

    months = get_month_sequence(shark_df, args.step_months)

    if not _HAS_IMAGEIO:
        raise RuntimeError("The 'imageio' library is required to create animations.")

    args.output_gif.parent.mkdir(parents=True, exist_ok=True)
    if args.output_mp4:
        args.output_mp4.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(str(args.output_gif), mode="I", duration=1.0 / args.fps, loop=0) as gif_writer:
        mp4_writer = None
        if args.output_mp4:
            try:
                mp4_writer = imageio.get_writer(str(args.output_mp4), fps=args.fps, macro_block_size=1, codec="libx264" if _HAS_IMAGEIO_FFMPEG else None)
            except Exception:
                print("Warning: Could not initialize MP4 writer. Continuing with GIF only.")
                mp4_writer = None

        for month in months:
            if args.mode == "snapshot":
                sharks_in_frame = shark_df[shark_df["month"] == month]
            else:
                current_period = pd.Period(month)
                start_period = (current_period - (args.tail_months - 1)).strftime("%Y-%m")
                window_months = pd.period_range(start_period, month, freq="M").astype(str)
                sharks_in_frame = shark_df[shark_df["month"].isin(window_months)]

            eddies_in_frame = eddy_df[eddy_df["month"] == month] if "month" in eddy_df.columns else pd.DataFrame()

            fig = plt.figure(figsize=tuple(args.figsize))
            ax = fig.add_subplot(111)

            num_individuals = sharks_in_frame[id_col].nunique()
            num_points = len(sharks_in_frame)
            num_eddies = int(eddies_in_frame.get("eddy_count_total", 0).sum())
            if num_eddies == 0:
                num_eddies = int(eddies_in_frame.get("eddy_count_cyclonic", 0).sum()) + int(eddies_in_frame.get("eddy_count_anticyclonic", 0).sum())
            
            if args.mode == "snapshot":
                title = f"Month: {month} | Individuals: {num_individuals} | Points: {num_points} | Eddies: {num_eddies}"
            else:
                title = f"{args.tail_months}-Month Window ending {month} | Individuals: {num_individuals} | Points: {num_points} | Eddies: {num_eddies}"

            draw_monthly_frame(ax, sharks_in_frame, id_col, eddies_in_frame, bbox, title)

            frame_image = convert_figure_to_image(fig, dpi=args.dpi)
            plt.close(fig)

            if frame_image.size > 0:
                gif_writer.append_data(frame_image)
                if mp4_writer:
                    try:
                        mp4_writer.append_data(frame_image)
                    except Exception as e:
                        print(f"Warning: Failed to write MP4 frame. Disabling MP4 output. Error: {e}")
                        mp4_writer.close()
                        mp4_writer = None
        
        if mp4_writer:
            mp4_writer.close()

    print(f"GIF successfully created: {args.output_gif}")
    if args.output_mp4:
        print(f"MP4 successfully created: {args.output_mp4} (requires ffmpeg)")

if __name__ == "__main__":
    main()

# How to run (example on Windows/CMD or PowerShell):
# python shark_eddy_animator.py ^
#   --sharks "Whale shark movements in Gulf of Mexico.csv" ^
#   --eddy-grid ".\out_monthly\monthly_eddy_grid.csv" ^
#   --output-gif ".\out_monthly\sharks_eddies_monthly.gif" ^
#   --mode snapshot ^
#   --step-months 1 ^
#   --tail-months 3 ^
#   --lon-range -98 -80 ^
#   --lat-range 18 31 ^
#   --fps 4 ^
#   --dpi 130
#
# Note: the script expects --output-gif (not --outgif) and
#       --lon-range / --lat-range (not --gulf-lon / --gulf-lat).

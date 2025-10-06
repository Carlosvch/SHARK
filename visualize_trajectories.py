import json
import subprocess
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def load_and_prepare_data(csv_path: Path) -> tuple[pd.DataFrame, str]:
    if not csv_path.exists():
        print(f"Warning: File '{csv_path}' not found. Creating a sample file.")
        dummy_data = {
            'timestamp': ['2010-08-15 12:00:00', '2010-08-15 18:00:00', '2010-08-16 12:00:00', '2010-08-15 14:00:00', '2010-08-16 09:00:00'],
            'location-lat': [28.5, 28.6, 28.7, 27.1, 27.2],
            'location-long': [-88.0, -88.2, -88.3, -89.5, -89.6],
            'individual-local-identifier': ['A01', 'A01', 'A01', 'B02', 'B02']
        }
        pd.DataFrame(dummy_data).to_csv(csv_path, index=False)
    
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['latitude'] = pd.to_numeric(df.get('location-lat'), errors='coerce')
    df['longitude'] = pd.to_numeric(df.get('location-long'), errors='coerce')
    df.dropna(subset=['latitude', 'longitude', 'timestamp'], inplace=True)

    id_col = 'individual-local-identifier' if 'individual-local-identifier' in df.columns else 'tag-local-identifier'
    df.sort_values([id_col, 'timestamp'], inplace=True)
    
    return df, id_col


def plot_all_trajectories(df: pd.DataFrame, id_col: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, group in df.groupby(id_col, sort=True):
        if len(group) >= 2:
            ax.plot(group['longitude'].values, group['latitude'].values, linewidth=1, alpha=0.7)
        ax.scatter(group['longitude'].values, group['latitude'].values, s=6, alpha=0.6)

    ax.set_title(f'Trajectories of {df[id_col].nunique()} Whale Sharks')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def create_individual_pdf(df: pd.DataFrame, id_col: str, output_path: Path):
    with PdfPages(output_path) as pdf:
        for individual_id, group in df.groupby(id_col, sort=True):
            fig, ax = plt.subplots(figsize=(9, 7))
            if len(group) >= 2:
                ax.plot(group['longitude'].values, group['latitude'].values, linewidth=1, alpha=0.8, marker='o', markersize=2)
            else:
                ax.scatter(group['longitude'].values, group['latitude'].values, s=20, alpha=0.8, marker='o')
            
            start_date = group['timestamp'].iloc[0]
            end_date = group['timestamp'].iloc[-1]
            ax.set_title(f'Trajectory for Individual {individual_id}\n{len(group)} points | {start_date.date()} to {end_date.date()}')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.2)
            ax.set_aspect('equal', adjustable='box')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def generate_geojson(df: pd.DataFrame, id_col: str, output_path: Path):
    features = []
    for individual_id, group in df.groupby(id_col, sort=True):
        coords = list(zip(group['longitude'].tolist(), group['latitude'].tolist()))
        
        if len(coords) >= 2:
            geometry = {"type": "LineString", "coordinates": coords}
        else:
            geometry = {"type": "Point", "coordinates": coords[0]}
        
        properties = {
            "individual_id": str(individual_id),
            "n_points": len(coords),
            "start_timestamp": group['timestamp'].iloc[0].isoformat(),
            "end_timestamp": group['timestamp'].iloc[-1].isoformat(),
        }
        features.append({"type": "Feature", "geometry": geometry, "properties": properties})

    feature_collection = {"type": "FeatureCollection", "features": features}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(feature_collection, f, indent=2)


def create_trajectory_animation(df: pd.DataFrame, id_col: str, output_path: Path) -> str:
    animation_df = df.sort_values('timestamp').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    lon_min, lon_max = animation_df['longitude'].min() - 0.5, animation_df['longitude'].max() + 0.5
    lat_min, lat_max = animation_df['latitude'].min() - 0.5, animation_df['latitude'].max() + 0.5
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    title = ax.set_title('')
    
    unique_ids = animation_df[id_col].unique()
    shark_plots = {sid: ax.plot([], [], marker='o', markersize=3, label=sid)[0] for sid in unique_ids}
    shark_data = {sid: {'x': [], 'y': []} for sid in unique_ids}

    def init():
        for plot in shark_plots.values():
            plot.set_data([], [])
        title.set_text('')
        return list(shark_plots.values()) + [title]

    def update(frame_num):
        record = animation_df.iloc[frame_num]
        sid = record[id_col]
        
        shark_data[sid]['x'].append(record['longitude'])
        shark_data[sid]['y'].append(record['latitude'])
        shark_plots[sid].set_data(shark_data[sid]['x'], shark_data[sid]['y'])
        
        current_date = record['timestamp'].strftime('%Y-%m-%d')
        title.set_text(f'Trajectory Evolution | Date: {current_date}')
        
        return list(shark_plots.values()) + [title]

    num_frames = len(animation_df)
    anim = animation.FuncAnimation(fig, update, frames=num_frames, init_func=init, blit=True, interval=50)

    try:
        anim.save(str(output_path), writer='ffmpeg', dpi=150, progress_callback=lambda i, n: print(f'Saving video: frame {i+1}/{n}', end='\r'))
        print() 
        return f"MP4 video (chronological animation): {output_path}"
    except FileNotFoundError:
        return "MP4 video not generated: `ffmpeg` was not found. Please install it and ensure it's in your system's PATH."
    except Exception as e:
        return f"MP4 video not generated: An error occurred -> {e}"
    finally:
        plt.close(fig)


def main():
    INPUT_CSV = Path('Whale shark movements in Gulf of Mexico.csv')
    OUTPUT_DIR = Path('./shark_trajectories_outputs')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, id_col = load_and_prepare_data(INPUT_CSV)

    png_path = OUTPUT_DIR / 'all_trajectories.png'
    plot_all_trajectories(df, id_col, png_path)

    pdf_path = OUTPUT_DIR / 'individual_trajectories.pdf'
    create_individual_pdf(df, id_col, pdf_path)

    geojson_path = OUTPUT_DIR / 'individual_trajectories.geojson'
    generate_geojson(df, id_col, geojson_path)
    
    video_path = OUTPUT_DIR / 'animated_trajectories.mp4'
    video_message = create_trajectory_animation(df, id_col, video_path)

    print("\n=== TRAJECTORY SUMMARY ===")
    print(f"Input file: {INPUT_CSV.name}")
    print(f"Individuals: {df[id_col].nunique()}")
    print(f"Total valid positions: {len(df)}")
    print(f"\nOutputs generated in '{OUTPUT_DIR}' directory:")
    print(f"- PNG (all trajectories): {png_path}")
    print(f"- PDF (one plot per individual): {pdf_path}")
    print(f"- GeoJSON (LineString/Point per individual): {geojson_path}")
    print(f"- {video_message}")


if __name__ == "__main__":
    main()
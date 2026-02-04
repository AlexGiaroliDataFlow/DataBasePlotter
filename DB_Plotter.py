"""
DB_Plotter - SQLite Database Visualization Tool
A Streamlit application for visualizing flowsense sensor data.
"""

import streamlit as st
import sqlite3
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
import numpy as np
from datetime import timedelta
import json
import math

# Page configuration
st.set_page_config(
    page_title="DB Plotter",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #5b9bd5;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px 20px;
        color: #31333F;
        transition: color 0.3s ease;
    }
    
    /* Generic selected tab style (fallback for all tabs including subtabs) */
    .stTabs [aria-selected="true"] {
        background-color: #7eb8da;
        color: white !important;
    }

    /* --- Main Tabs Specific Colors (6 main tabs) --- */
    
    /* Tab 1: Compare - Pastel Blue (Requested) */
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(1)[aria-selected="true"] {
        background-color: #7EB8DA !important;
        color: white !important;
    }
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(1):not([aria-selected="true"]):hover {
        color: #7EB8DA !important;
    }
    
    /* Tab 2: Sensors - Pastel Yellow */
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(2)[aria-selected="true"] {
        background-color: #F4D03F !important;
        color: black !important;
    }
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(2):not([aria-selected="true"]):hover {
        color: #F4D03F !important;
    }
    
    /* Tab 3: Power Analyzer - Vivid Orange */
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(3)[aria-selected="true"] {
        background-color: #E67E22 !important;
        color: white !important;
    }
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(3):not([aria-selected="true"]):hover {
        color: #E67E22 !important;
    }
    
    /* Tab 4: FFT - Pastel Blue */
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(4)[aria-selected="true"] {
        background-color: #5DADE2 !important;
        color: white !important;
    }
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(4):not([aria-selected="true"]):hover {
        color: #5DADE2 !important;
    }
    
    /* Tab 5: Harmonics - Pastel Purple */
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(5)[aria-selected="true"] {
        background-color: #AF7AC5 !important;
        color: white !important;
    }
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(5):not([aria-selected="true"]):hover {
        color: #AF7AC5 !important;
    }
    
    /* Tab 6: GPS - Pastel Green */
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(6)[aria-selected="true"] {
        background-color: #58D68D !important;
        color: white !important;
    }
    .stTabs > div > [data-baseweb="tab-list"] > button:nth-of-type(6):not([aria-selected="true"]):hover {
        color: #58D68D !important;
    }

    /* Hide red indicator under tabs */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Multiselect chips color - Blue to match Compare Tab */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #7EB8DA !important;
        color: white !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }
        /* --- Global Interactive Component Styling (Blue Theme #7EB8DA) --- */
    
    /* 1. SLIDERS */
    /* Track (Gray) */
    .stSlider > div > div > div > div {
        background: #808080 !important;
    }
    /* Thumb/Handle */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #7EB8DA !important;
        border-color: #7EB8DA !important;
    }
    /* Value Text (Timestamp/Number above slider) */
    .stSlider [data-testid="stMarkdownContainer"] p {
        color: #7EB8DA !important;
    }
    /* Ensure all text inside the slider widget (like ticks or direct values) uses the blue */
    .stSlider [data-baseweb="slider"] div {
        color: #7EB8DA !important;
    }

    /* 2. MULTISELECT & SELECTBOX */
    /* Borders */
    .stMultiSelect [data-baseweb="select"] > div,
    .stSelectbox [data-baseweb="select"] > div {
        border-color: #7EB8DA !important;
    }
    /* Focus State */
    .stMultiSelect [data-baseweb="select"]:focus-within > div,
    .stSelectbox [data-baseweb="select"]:focus-within > div {
        border-color: #7EB8DA !important;
        box-shadow: 0 0 0 1px #7EB8DA !important;
    }
    /* Dropdown Icons */
    .stMultiSelect [data-baseweb="select"] [data-testid="stIcon"],
    .stSelectbox [data-baseweb="select"] [data-testid="stIcon"] {
        color: #7EB8DA !important;
    }
    /* Selected Chips (Multiselect) */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #7EB8DA !important;
        color: white !important;
    }
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }

    /* 3. TOGGLES & CHECKBOXES */
    /* Toggle Switch (Active) */
    .stToggle [aria-checked="true"] {
        background-color: #7EB8DA !important;
        color: white !important;
    }
    /* Checkbox (Active) */
    .stCheckbox [data-baseweb="checkbox"] [aria-checked="true"] {
        background-color: #7EB8DA !important;
        border-color: #7EB8DA !important;
    }
    
    /* 4. NUMBER INPUT */
    .stNumberInput [data-baseweb="input"] {
        border-color: #7EB8DA !important;
    }
    .stNumberInput [data-baseweb="input"]:focus-within {
        box-shadow: 0 0 0 1px #7EB8DA !important;
    }
    /* Plus/Minus buttons */
    .stNumberInput button {
        color: #7EB8DA !important;
        border-color: #7EB8DA !important;
    }
    .stNumberInput button:hover {
        background-color: #7EB8DA !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DEFAULT_DATABASE_FOLDER = Path(__file__).parent / "Database"
EXCLUDE_METADATA_COLS = ['id_day', 'id', 'status', 'human_timestamp', 'unix_timestamp', 'datetime']

# Vivid pastel color palette (more saturated)
PASTEL_COLORS = [
    '#F4D03F',  # vivid yellow
    '#E74C3C',  # vivid red
    '#58D68D',  # vivid green
    '#5DADE2',  # vivid blue
    '#AF7AC5',  # vivid purple
    '#E67E22',  # vivid orange
    '#3498DB',  # strong blue
    '#EC7063',  # coral red
    '#45B39D',  # teal green
    '#F5B041',  # amber
    '#9B59B6',  # purple
    '#EB984E',  # orange
]

def adjust_color_brightness(hex_color, factor):
    """
    Adjusts the brightness of a HEX color.
    factor > 1 lightens the color.
    factor < 1 darkens the color.
    """
    if not isinstance(hex_color, str) or not hex_color.startswith('#'):
        return hex_color
        
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        r = int(max(0, min(255, r * factor)))
        g = int(max(0, min(255, g * factor)))
        b = int(max(0, min(255, b * factor)))
        
        return f'#{r:02x}{g:02x}{b:02x}'
    return f"#{hex_color}"

# Reordered palette for sensors/power analyzer (starts with blue instead of yellow)
# Expanded with many more distinct colors to avoid repetition
SENSOR_COLORS = [
    '#5DADE2',  # vivid blue
    '#E74C3C',  # vivid red
    '#58D68D',  # vivid green
    '#AF7AC5',  # vivid purple
    '#E67E22',  # vivid orange
    '#3498DB',  # strong blue
    '#EC7063',  # coral red
    '#45B39D',  # teal green
    '#F5B041',  # amber
    '#9B59B6',  # purple
    '#EB984E',  # orange
    '#48C9B0',  # turquoise
    '#F1948A',  # light coral
    '#85C1E2',  # sky blue
    '#F8B739',  # golden yellow
    '#BB8FCE',  # lavender
    '#52BE80',  # emerald
    '#F0B27A',  # light orange
    '#5499C7',  # steel blue
    '#CD6155',  # dark coral
    '#7DCEA0',  # mint green
    '#D7BDE2',  # pale purple
    '#F7DC6F',  # bright yellow
    '#76D7C4',  # aquamarine
    '#EC407A',  # pink
    '#AED6F1',  # powder blue
    '#F4D03F',  # vivid yellow (moved to end)
]

PASTEL_BLUE = '#5DADE2'
PASTEL_RED = '#E74C3C'
PASTEL_GREEN = '#58D68D'

class MqttStats:
    def __init__(self):
        self.sources = {}

    def add(self, source, bytes_count, packets_count, duration_str="N/A"):
        if source not in self.sources:
            self.sources[source] = {'bytes': bytes_count, 'packets': packets_count, 'duration': duration_str}
        else:
            self.sources[source]['bytes'] += bytes_count
            self.sources[source]['packets'] += packets_count
            # Update duration if available and not set or just overwrite
            if duration_str != "N/A":
                self.sources[source]['duration'] = duration_str

    def get_total_mb(self):
        return sum(s['bytes'] for s in self.sources.values()) / (1024 * 1024)

    def get_total_4kb_packets(self):
        return math.ceil(sum(s['bytes'] for s in self.sources.values()) / 4096)
        
    def get_breakdown(self):
        return self.sources

def get_database_files(folder_path: Path) -> list:
    """Get list of .db files in the specified folder, sorted by modification time (newest first)."""
    if folder_path.exists():
        files = list(folder_path.glob("*.db"))
        # Sort by modification time, descending
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return [f.name for f in files]
    return []


def load_database(db_path: str) -> sqlite3.Connection:
    """Load SQLite database and return connection."""
    return sqlite3.connect(db_path)


def analyze_transmission_quality(df: pd.DataFrame, time_col: str = 'datetime', column_name: str = None) -> tuple:
    """
    Analyze data transmission quality by detecting gaps.
    
    Args:
        df: DataFrame containing the data.
        time_col: Name of the time column.
        column_name: (Optional) Specific column to check for local invalid values (NaN/Non-numeric).
        
    Returns:
        (stats_dict, global_gaps_list, local_gaps_list)
    """
    stats = {
        'expected': 0,
        'actual': len(df),
        'global_lost': 0,
        'local_lost': 0,
        'total_lost': 0,
        'success_rate': 100.0
    }
    global_gaps = [] # Transmission gaps (missing packets)
    local_gaps = []  # Sensor errors (invalid values in received packets)
    
    if df.empty:
        return stats, global_gaps, local_gaps

    # --- 1. Global Gap Detection (Transmission Loss) ---
    # Try to use 'id' column for precise gap detection
    if 'id' in df.columns and df['id'].is_unique:
        df_sorted = df.sort_values('id')
        ids = df_sorted['id'].values
        times = df_sorted[time_col].values
        
        # Calculate diffs
        id_diffs = np.diff(ids)
        
        # Where diff > 1, there is a gap
        gap_indices = np.where(id_diffs > 1)[0]
        
        total_global_lost = 0
        for idx in gap_indices:
            start_id = ids[idx]
            end_id = ids[idx+1]
            lost_count = end_id - start_id - 1
            total_global_lost += lost_count
            
            # Record global gap time range
            global_gaps.append({
                'start': times[idx],
                'end': times[idx+1],
                'count': lost_count,
                'type': 'transmission_loss'
            })
            
        stats['global_lost'] = total_global_lost
        
    else:
        # Fallback to time-based detection
        if time_col not in df.columns:
            # Cannot detect gaps without time
            return stats, global_gaps, local_gaps
            
        df_sorted = df.sort_values(time_col)
        times = df_sorted[time_col].values
        
        # Calculate time diffs in seconds
        time_diffs = np.diff(times).astype('timedelta64[ms]').astype(float) / 1000.0
        
        if len(time_diffs) > 0:
            # Estimate expected interval (median)
            median_interval = np.median(time_diffs)
            if median_interval > 0:
                # Threshold for gap (e.g., > 1.5x median)
                gap_threshold = median_interval * 1.5
                
                gap_indices = np.where(time_diffs > gap_threshold)[0]
                
                total_global_lost = 0
                for idx in gap_indices:
                    gap_duration = time_diffs[idx]
                    # Estimate lost packets
                    lost_count = int(round(gap_duration / median_interval)) - 1
                    if lost_count > 0:
                        total_global_lost += lost_count
                        global_gaps.append({
                            'start': times[idx],
                            'end': times[idx+1],
                            'count': lost_count,
                            'type': 'transmission_loss'
                        })
                stats['global_lost'] = total_global_lost
    
    # --- 2. Local Gap Detection (Sensor Faults) ---
    if column_name and column_name in df.columns:
        # Sort by time to ensure grouping works
        df_local = df.sort_values(time_col)
        
        # Check for NaN or infinite values
        series = pd.to_numeric(df_local[column_name], errors='coerce')
        
        # Identify invalid indices
        invalid_mask = series.isna()
        local_lost_count = invalid_mask.sum()
        stats['local_lost'] = int(local_lost_count)
        
        # Identify ranges/points of local loss for visualization
        if local_lost_count > 0:
            # We want to find contiguous blocks of NaNs to report as gaps, or individual points
            # Get timestamps where data is invalid
            invalid_times = df.loc[invalid_mask, time_col]
            
            for t in invalid_times:
                 local_gaps.append({
                    'start': t,
                    'end': t, # Point gap
                    'count': 1,
                    'type': 'sensor_fault'
                })

    # --- 3. Final Stats ---
    # Expected packets = Actual received + Global Lost
    stats['expected'] = int(stats['actual']) + int(stats['global_lost'])
    
    # Total Lost = Global Lost (not received) + Local Lost (received but invalid)
    # Note: 'Actual' includes the 'Local Lost' rows because they exist in the DB.
    # So valid packets = Actual - Local Lost
    # Success Rate = Valid Packets / Total Expected
    
    valid_packets = stats['actual'] - stats['local_lost']
    stats['total_lost'] = stats['global_lost'] + stats['local_lost']
    
    if stats['expected'] > 0:
        stats['success_rate'] = (valid_packets / stats['expected']) * 100.0
        
    return stats, global_gaps, local_gaps




def get_table_data(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    """Load data from a table into a DataFrame."""
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        if 'human_timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['human_timestamp'], format='%d/%m/%Y - %H:%M:%S', errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading table {table_name}: {e}")
        return pd.DataFrame()


def check_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None


def get_empty_columns(df: pd.DataFrame, exclude_cols: list = None) -> list:
    """Identify columns that are present but contain no data."""
    if df.empty:
        return []
    
    if exclude_cols is None:
        exclude_cols = []
        
    empty_cols = []
    # Define placeholder values that indicate missing data
    empty_placeholders = {'no value', 'no sensor', 'nan', 'none', 'n/a', ''}
    
    for col in df.columns:
        if col in exclude_cols:
            continue
        
        # Check if all values are null/nan
        if df[col].isna().all():
            empty_cols.append(col)
            continue
            
        # Check if all values are empty placeholder strings
        # Convert to string and lowercase for comparison
        col_values = df[col].astype(str).str.lower().str.strip()
        unique_values = set(col_values.unique())
        
        # If all unique values are in the empty placeholders set, mark as empty
        if unique_values.issubset(empty_placeholders):
            empty_cols.append(col)
            
    return empty_cols


def create_date_range_slider(df: pd.DataFrame, key_prefix: str):
    """Create a date range slider and return filtered dataframe."""
    if 'datetime' not in df.columns or df['datetime'].isna().all():
        return df.copy(), 'id'
    
    min_date = df['datetime'].min()
    max_date = df['datetime'].max()
    
    # Date range slider with second-level resolution
    date_range = st.slider(
        "Select Date/Time Range",
        min_value=min_date.to_pydatetime(),
        max_value=max_date.to_pydatetime(),
        value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        step=timedelta(seconds=1),
        format="DD/MM/YY HH:mm:ss",
        key=f"{key_prefix}_range"
    )
    
    # Filter data based on range
    mask = (df['datetime'] >= date_range[0]) & (df['datetime'] <= date_range[1])
    return df[mask].copy(), 'datetime'


def plot_sensor_data(df_filtered: pd.DataFrame, x_axis: str, show_quality: bool = True, show_mqtt_calc: bool = True, mqtt_interval: int = 1, mqtt_stats: 'MqttStats' = None, df_comparison: pd.DataFrame = None, primary_label: str = "", comparison_label: str = ""):
    """Create interactive time series plots for sensor data."""
    if df_filtered is None or df_filtered.empty:
        st.warning("No sensor data available.")
        return None, [], None
    
    # Get sensor columns (excluding metadata columns)
    sensor_cols = [col for col in df_filtered.columns if col not in EXCLUDE_METADATA_COLS]
    
    if not sensor_cols:
        st.warning("No sensor columns found.")
        return None, [], None
    
    # Filter columns that are all NaN after filtering
    valid_cols = []
    for col in sensor_cols:
        # Convert to numeric, handling any text values
        if df_filtered[col].dtype == 'object':
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        if not df_filtered[col].isna().all():
            valid_cols.append(col)
            
    if not valid_cols:
        st.warning("No valid data in selected range.")
        return df_filtered, [], x_axis

    # Prepare comparison data if provided - ALIGN TIMESTAMPS
    df_comp_aligned = None
    if df_comparison is not None and not df_comparison.empty:
        df_comp_aligned = df_comparison.copy()
        
        # Align timestamps: shift comparison data to start at primary data's start time
        if x_axis == 'datetime' and 'datetime' in df_comp_aligned.columns:
            primary_start = df_filtered[x_axis].min()
            comp_start = df_comp_aligned['datetime'].min()
            
            # Calculate time offset and shift comparison timestamps
            time_offset = primary_start - comp_start
            df_comp_aligned['datetime'] = df_comp_aligned['datetime'] + time_offset
            
            # Get primary time range for filtering
            primary_end = df_filtered[x_axis].max()
            
            # Filter to only include data within primary's range
            mask = (df_comp_aligned['datetime'] >= primary_start) & (df_comp_aligned['datetime'] <= primary_end)
            df_comp_aligned = df_comp_aligned[mask].copy()

    # Create individual plots for each sensor
    for idx, col in enumerate(valid_cols):
        try:
            line_color = SENSOR_COLORS[idx % len(SENSOR_COLORS)]
            # Use a darker shade for comparison (more distinct)
            comp_color = adjust_color_brightness(line_color, 0.7)
            
            # Create separate figure
            fig = go.Figure()
            
            # Primary DB name suffix
            primary_suffix = f" [{primary_label}]" if primary_label else ""
            comp_suffix = f" [{comparison_label}]" if comparison_label else " [Comp]"
            
            # Trace 1: Raw Data (Primary)
            fig.add_trace(go.Scatter(
                x=df_filtered[x_axis],
                y=df_filtered[col],
                mode='lines',
                name=f'{col}{primary_suffix}',
                line=dict(color=line_color, width=1),
                opacity=0.7,
                hovertemplate=f'{col}{primary_suffix}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
            ))
            
            # Calculate Moving Average (Primary)
            window_size = max(10, len(df_filtered) // 50)
            col_avg = df_filtered[col].rolling(window=window_size, center=True).mean()
            
            # Trace 2: Average (Primary)
            fig.add_trace(go.Scatter(
                x=df_filtered[x_axis],
                y=col_avg,
                mode='lines',
                name=f'{col} Avg{primary_suffix}',
                line=dict(color=line_color, width=2.5),
                hovertemplate=f'{col} Avg{primary_suffix}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
            ))
            
            # Comparison DB traces (aligned timestamps, solid lines)
            if df_comp_aligned is not None and not df_comp_aligned.empty and col in df_comp_aligned.columns:
                # Convert comparison column to numeric if needed
                comp_col_data = df_comp_aligned[col].copy()
                if comp_col_data.dtype == 'object':
                    comp_col_data = pd.to_numeric(comp_col_data, errors='coerce')
                
                if not comp_col_data.isna().all():
                    # Trace 3: Raw Data (Comparison) - dashed line for clear distinction
                    fig.add_trace(go.Scatter(
                        x=df_comp_aligned[x_axis],
                        y=comp_col_data,
                        mode='lines',
                        name=f'{col}{comp_suffix}',
                        line=dict(color=comp_color, width=1.5, dash='dot'),
                        hovertemplate=f'{col}{comp_suffix}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
                    ))
                    
                    # Calculate Moving Average (Comparison)
                    window_size_c = max(10, len(df_comp_aligned) // 50)
                    col_avg_c = comp_col_data.rolling(window=window_size_c, center=True).mean()
                    
                    # Trace 4: Average (Comparison) - dashed line for clear distinction
                    fig.add_trace(go.Scatter(
                        x=df_comp_aligned[x_axis],
                        y=col_avg_c,
                        mode='lines',
                        name=f'{col} Avg{comp_suffix}',
                        line=dict(color=comp_color, width=2.5, dash='dot'),
                        hovertemplate=f'{col} Avg{comp_suffix}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title=col.replace('_', ' ').title(),
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=True,
                hovermode='x unified',
                yaxis_title=col
            )

            # Transmission Quality Analysis per sensor
            if show_quality:
                # Debug: Ensure column exists
                if col not in df_filtered.columns:
                     st.error(f"Column {col} missing from dataframe")
                     continue
                     
                stats, global_gaps, local_gaps = analyze_transmission_quality(df_filtered, x_axis, column_name=col)
                
                # 1. Visualize Global Gaps (Transmission Loss) - Full Height Red Zones
                for gap in global_gaps:
                    fig.add_vrect(
                        x0=gap['start'],
                        x1=gap['end'],
                        fillcolor="red",
                        opacity=0.1,
                        layer="below",
                        line_width=0
                    )
                    fig.add_annotation(
                        x=gap['start'],
                        y=1,
                        yref="paper",
                        text="No Signal",
                        showarrow=False,
                        xanchor="left",
                        yanchor="top",
                        font=dict(size=8, color="red")
                    )
                
                # 2. Visualize Local Gaps (Sensor Faults) - Markers or specific indications
                # Since local gaps are specific points (or ranges) where data exists but is invalid
                if local_gaps:
                    # Collect timestamps
                    fault_times = [g['start'] for g in local_gaps]
                    # Determine Y position (use min of data or 0)
                    y_pos = df_filtered[col].min() if not pd.isna(df_filtered[col].min()) else 0
                    
                    fig.add_trace(go.Scatter(
                        x=fault_times,
                        y=[y_pos] * len(fault_times), 
                        mode='markers',
                        marker=dict(symbol='x', color='orange', size=8),
                        name='Invalid Value',
                        hoverinfo='skip'
                    ))


                st.plotly_chart(fig, width="stretch", key=f"sensor_plot_{idx}")
                
                # Metrics Row
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Success Rate", f"{stats['success_rate']:.2f}%")
                with m2:
                    st.metric("Valid Packets", stats['actual'] - stats['local_lost'])
                with m3:
                    st.metric("Transmission Loss", stats['global_lost'], help="Packets not received (network gap)")
                with m4:
                    st.metric("Sensor Faults", stats['local_lost'], help="Packets received but value is invalid (NaN/Text)")
                with m5:
                    st.metric("Total Expected", stats['expected'])
                    
                if stats['success_rate'] < 95.0:
                     st.error(f"Issue detected with {col}: {stats['total_lost']} total lost packets.")
                
                if idx < len(valid_cols) - 1:
                    st.markdown("---") # Separator between sensors
            else:
                st.plotly_chart(fig, width="stretch", key=f"sensor_plot_{idx}")
                
        except Exception as e:
            st.error(f"Error plotting {col}: {e}")


    return df_filtered, valid_cols, x_axis




def plot_power_analyzer_data(df: pd.DataFrame, show_quality: bool = True, show_mqtt_calc: bool = True, mqtt_interval: int = 1, mqtt_stats: 'MqttStats' = None, df_comparison: pd.DataFrame = None, primary_label: str = "", comparison_label: str = ""):
    """Create interactive time series plots for power analyzer data."""
    if df.empty:
        st.warning("No power analyzer data available.")
        return
    
    # Get power analyzer columns
    power_cols = [col for col in df.columns if col not in EXCLUDE_METADATA_COLS]
    
    if not power_cols:
        st.warning("No power analyzer columns found.")
        return
    
    # Power columns that are in Kilo
    power_cols_kilo = ['Psys', 'Qsys', 'Ssys']
    
    # Columns that need absolute value (currents and power system)
    abs_cols = ['A1', 'A2', 'A3', 'Asys', 'Asys_MAX', 'Psys', 'Qsys', 'Ssys', 'Wh', 'VAh']
    
    # Apply absolute value to specified columns
    df_abs = df.copy()
    for col in abs_cols:
        if col in df_abs.columns:
            df_abs[col] = df_abs[col].abs()
    
    # Date range slider
    df_filtered, x_axis = create_date_range_slider(df_abs, "power")
    
    # Prepare comparison data with time alignment
    df_comp_aligned = None
    if df_comparison is not None and not df_comparison.empty:
        df_comp_aligned = df_comparison.copy()
        # Apply absolute value to comparison data too
        for col in abs_cols:
            if col in df_comp_aligned.columns:
                df_comp_aligned[col] = df_comp_aligned[col].abs()
        
        # Align timestamps: shift comparison data to start at primary data's start time
        if x_axis == 'datetime' and 'datetime' in df_comp_aligned.columns:
            primary_start = df_filtered[x_axis].min()
            comp_start = df_comp_aligned['datetime'].min()
            time_offset = primary_start - comp_start
            df_comp_aligned['datetime'] = df_comp_aligned['datetime'] + time_offset
            
            # Filter to primary range
            primary_end = df_filtered[x_axis].max()
            mask = (df_comp_aligned['datetime'] >= primary_start) & (df_comp_aligned['datetime'] <= primary_end)
            df_comp_aligned = df_comp_aligned[mask].copy()
    
    # Group related metrics
    current_cols = [c for c in power_cols if c.startswith('A') and not c.startswith('Asys')]
    current_cols += [c for c in power_cols if c.startswith('Asys')]
    
    # Voltage + Frequency (no VAh here, it goes in Power System)
    voltage_cols = [c for c in power_cols if c.startswith('V') and c != 'VAh']
    if 'f' in power_cols and 'f' not in voltage_cols:
        voltage_cols.append('f')
    
    # Power System + Wh + VAh
    power_system_cols = [c for c in power_cols if c in ['Psys', 'Qsys', 'Ssys', 'TPFsys']]
    if 'Wh' in power_cols and 'Wh' not in power_system_cols:
        power_system_cols.append('Wh')
    if 'VAh' in power_cols and 'VAh' not in power_system_cols:
        power_system_cols.append('VAh')
    
    thd_cols = [c for c in power_cols if c.startswith('THD')]
    
    # Other cols: exclude all grouped cols
    grouped_cols = set(current_cols + voltage_cols + power_system_cols + thd_cols)
    other_cols = [c for c in power_cols if c not in grouped_cols]
    
    # Plot grouped metrics
    groups = [
        ("Current", current_cols),
        ("Voltage", voltage_cols),
        ("Power", power_system_cols),
        ("Total Harmonic Distortion", thd_cols),
        ("Other Measurements", other_cols)
    ]
    
    # Collect all valid columns to plot in order
    plot_definitions = []
    
    for group_name, cols in groups:
        if cols:
             # Check distinct columns in this group that have data
            group_valid_cols = [c for c in cols if c in df_filtered.columns and not df_filtered[c].isna().all()]
            if group_valid_cols:
                for col in group_valid_cols:
                    plot_definitions.append({
                        'col': col,
                        'group': group_name
                    })

    if not plot_definitions:
        st.warning("No valid power data found in selected range.")
        return

    num_plots = len(plot_definitions)
    

    
    for idx, definition in enumerate(plot_definitions):
        try:
            col = definition['col']
            
            row_num = idx + 1
            
            # Determine labels and titles
            group_title = definition['group']
            
            if col == 'f':
                display_name = "Frequency"
                plot_title = "Frequency"
                y_label = "Frequency"
            else:
                display_name = col
                plot_title = f"{col} ({group_title})"
                
                if col in power_cols_kilo:
                    if col == 'Psys':
                        unit = 'kW'
                    elif col == 'Qsys':
                         unit = 'kVAR'
                    else:
                         unit = 'kVA'
                    y_label = f"{col} ({unit})"
                else:
                    y_label = col
                
            line_color = SENSOR_COLORS[idx % len(SENSOR_COLORS)]
            comp_color = adjust_color_brightness(line_color, 0.7)  # Darker for comparison
            
            # Label suffixes
            primary_suffix = f" [{primary_label}]" if primary_label else ""
            comp_suffix = f" [{comparison_label}]" if comparison_label else " [Comp]"
            
            # New Figure
            fig = go.Figure()

            # Trace 1: Raw (Primary)
            fig.add_trace(go.Scatter(
                x=df_filtered[x_axis],
                y=df_filtered[col],
                mode='lines',
                name=f'{display_name}{primary_suffix}',
                line=dict(color=line_color, width=1),
                opacity=0.7,
                hovertemplate=f'{display_name}{primary_suffix}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
            ))
            
            # Calculate Moving Average (Primary)
            window_size = max(10, len(df_filtered) // 50)
            col_avg = df_filtered[col].rolling(window=window_size, center=True).mean()
            
            # Trace 2: Average (Primary)
            fig.add_trace(go.Scatter(
                x=df_filtered[x_axis],
                y=col_avg,
                mode='lines',
                name=f'{display_name} Avg{primary_suffix}',
                line=dict(color=line_color, width=2.5),
                hovertemplate=f'{display_name} Avg{primary_suffix}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
            ))
            
            # Comparison DB traces (if available)
            if df_comp_aligned is not None and not df_comp_aligned.empty and col in df_comp_aligned.columns:
                comp_col_data = df_comp_aligned[col].copy()
                if comp_col_data.dtype == 'object':
                    comp_col_data = pd.to_numeric(comp_col_data, errors='coerce')
                
                if not comp_col_data.isna().all():
                    # Trace 3: Raw (Comparison) - dashed for clear distinction
                    fig.add_trace(go.Scatter(
                        x=df_comp_aligned[x_axis],
                        y=comp_col_data,
                        mode='lines',
                        name=f'{display_name}{comp_suffix}',
                        line=dict(color=comp_color, width=1.5, dash='dot'),
                        hovertemplate=f'{display_name}{comp_suffix}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
                    ))
                    
                    # Calculate Moving Average (Comparison)
                    window_size_c = max(10, len(df_comp_aligned) // 50)
                    col_avg_c = comp_col_data.rolling(window=window_size_c, center=True).mean()
                    
                    # Trace 4: Average (Comparison) - dashed for clear distinction
                    fig.add_trace(go.Scatter(
                        x=df_comp_aligned[x_axis],
                        y=col_avg_c,
                        mode='lines',
                        name=f'{display_name} Avg{comp_suffix}',
                        line=dict(color=comp_color, width=2.5, dash='dot'),
                        hovertemplate=f'{display_name} Avg{comp_suffix}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
                    ))
            
            fig.update_layout(
                 title=plot_title,
                 height=300,
                 margin=dict(l=20, r=20, t=40, b=20),
                 showlegend=True,
                 hovermode='x unified',
                 yaxis_title=y_label
            )
            
             # Transmission Quality
            if show_quality:
                # Debug: Ensure column exists
                if col not in df_filtered.columns:
                     st.error(f"Column {col} missing from dataframe")
                     continue

                stats, global_gaps, local_gaps = analyze_transmission_quality(df_filtered, x_axis, column_name=col)
                
                # Global Gaps
                for gap in global_gaps:
                    fig.add_vrect(
                        x0=gap['start'],
                        x1=gap['end'],
                        fillcolor="red",
                        opacity=0.1,
                        layer="below",
                        line_width=0
                    )
                    fig.add_annotation(
                        x=gap['start'],
                        y=1,
                        yref="paper",
                        text="No Signal",
                        showarrow=False,
                        xanchor="left",
                        yanchor="top",
                        font=dict(size=8, color="red")
                    )
                
                # Local Gaps
                if local_gaps:
                    fault_times = [g['start'] for g in local_gaps]
                    y_pos = df_filtered[col].min() if not pd.isna(df_filtered[col].min()) else 0
                    
                    fig.add_trace(go.Scatter(
                        x=fault_times,
                        y=[y_pos] * len(fault_times), 
                        mode='markers',
                        marker=dict(symbol='x', color='orange', size=8),
                        name='Invalid Value',
                        hoverinfo='skip'
                    ))


                st.plotly_chart(fig, width="stretch", key=f"power_plot_{idx}")
                
                 # Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Success Rate", f"{stats['success_rate']:.2f}%")
                with m2:
                    st.metric("Valid Packets", stats['actual'] - stats['local_lost'])
                with m3:
                    st.metric("Transmission Loss", stats['global_lost'])
                with m4:
                    st.metric("Sensor Faults", stats['local_lost'])
                with m5:
                    st.metric("Total Expected", stats['expected'])
                    
                if stats['success_rate'] < 95.0:
                      st.error(f"Issue detected with {col}: {stats['total_lost']} total lost packets.")
                
                if idx < len(plot_definitions) - 1:
                    st.markdown("---")
            else:
                 st.plotly_chart(fig, width="stretch", key=f"power_plot_{idx}")

        except Exception as e:
            st.error(f"Error plotting {definition['col']}: {e}")

    # MQTT Packet Weight Analysis
    if show_mqtt_calc and not df_filtered.empty:
        st.markdown("---")
        st.subheader("MQTT Transmission Simulation")
        
        # 1. Frequency Slider
        # Calculate Duration
        time_min = df_filtered[x_axis].min()
        time_max = df_filtered[x_axis].max()
        duration_sec = 0.0
        
        if isinstance(time_min, pd.Timestamp):
            duration_sec = (time_max - time_min).total_seconds()
            
            # Formatting duration string
            td = timedelta(seconds=duration_sec)
            days = td.days
            hours, remainder = divmod(td.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            parts = []
            if days > 0: parts.append(f"{days} days")
            if hours > 0: parts.append(f"{hours} hours")
            if minutes > 0: parts.append(f"{minutes} minutes")
            parts.append(f"{seconds} seconds")
            duration_str = ", ".join(parts) if parts else "0 seconds"
            
            st.info(f"Selected Time Range Duration: **{duration_str}**")
        else:
            duration_sec = float(len(df_filtered))
            st.info(f"Selected Range: {len(df_filtered)} samples")
            
        sim_interval_power = mqtt_interval
        
        # Ensure duration is at least 1s
        duration_sec = max(1.0, duration_sec)
        
        # 3. Construct all Payloads
        visible_cols_set = set()
        visible_cols = []
        for d in plot_definitions:
            if d['col'] not in visible_cols_set:
                visible_cols.append(d['col'])
                visible_cols_set.add(d['col'])
        
        # Create mapping for optimized keys (incremental numbers)
        col_map = {col: str(i+1) for i, col in enumerate(visible_cols)}

        all_payloads_power = []

        if isinstance(time_min, pd.Timestamp):
            current_time = time_min
            while current_time <= time_max:
                payload = {'ts': int(current_time.timestamp())}
                
                # Find nearest row
                idx = df_filtered['datetime'].searchsorted(current_time)
                if idx >= len(df_filtered): idx = len(df_filtered) - 1
                row = df_filtered.iloc[idx]

                for col in visible_cols:
                    if col in row:
                        val = row[col]
                        key = col_map[col]
                        if pd.isna(val) or val is None:
                            payload[key] = float('nan')
                        else:
                            try: payload[key] = round(float(val), 2)
                            except: payload[key] = str(val)
                
                all_payloads_power.append(payload)
                current_time += timedelta(seconds=sim_interval_power)
        else:
            # Step-based
            for i in range(0, len(df_filtered), max(1, int(sim_interval_power))):
                row = df_filtered.iloc[i]
                payload = {'ts': int(row.get('unix_timestamp', i))}
                for col in visible_cols:
                    if col in row:
                        val = row[col]
                        key = col_map[col]
                        if pd.isna(val) or val is None: payload[key] = float('nan')
                        else:
                            try: payload[key] = round(float(val), 2)
                            except: payload[key] = str(val)
                all_payloads_power.append(payload)

        if not all_payloads_power:
            st.info("No packets to display.")
            return

        # 4. Calculate Weight
        full_json_sequence = "\n".join([json.dumps(p, separators=(',', ':'), allow_nan=True) for p in all_payloads_power])
        total_size_bytes = len(full_json_sequence.replace("\n", "")) # JSON weight is without extra chars usually, but sequence is fine
        # Re-calc precisely
        total_size_bytes = sum(len(json.dumps(p, separators=(',', ':'), allow_nan=True)) for p in all_payloads_power)
        
        total_packets = len(all_payloads_power)
        avg_packet_size = int(total_size_bytes / total_packets) if total_packets > 0 else 0
        total_size_mb = total_size_bytes / (1024 * 1024)
        packets_4kb = math.ceil(total_size_bytes / 4096)
        
        if mqtt_stats:
            mqtt_stats.add("Power Analyzer", total_size_bytes, total_packets, duration_str)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
             st.metric("4KB Packets Needed", f"{packets_4kb:,}")
        with c2:
            st.metric("Total Packets (JSON)", f"{total_packets:,}")
        with c3:
             st.metric("Avg Packet Size", f"{avg_packet_size} bytes")
        with c4:
             st.metric("Total Transmission Size (Power)", f"{total_size_mb:.2f} MB")
             
        with st.expander("View Json packet (first 10 rows)", expanded=False):
            # Show just the first 10 packets for preview
            preview_json_sequence = "\n".join([json.dumps(p, separators=(',', ':'), allow_nan=True) for p in all_payloads_power[:10]])
            st.code(preview_json_sequence, language='json')

def plot_tilt_data(df_filtered: pd.DataFrame, x_axis: str, show_quality: bool = True, show_mqtt_calc: bool = True, mqtt_interval: int = 1, mqtt_stats: 'MqttStats' = None, df_comparison: pd.DataFrame = None, primary_label: str = "", comparison_label: str = ""):
    """Create interactive time series plot for tilt data."""
    if df_filtered is None or df_filtered.empty:
        st.warning("No tilt data available.")
        return None, []
    
    if 'tilt_angle' not in df_filtered.columns:
        return df_filtered, []

    # Use a unique color from the extended palette (turquoise - not used by typical sensors)
    tilt_color = SENSOR_COLORS[11]  # turquoise
    comp_tilt_color = adjust_color_brightness(tilt_color, 0.7)  # Darker for comparison
    
    # Prepare comparison data with time alignment
    df_comp_aligned = None
    if df_comparison is not None and not df_comparison.empty and 'tilt_angle' in df_comparison.columns:
        df_comp_aligned = df_comparison.copy()
        
        # Align timestamps
        if x_axis == 'datetime' and 'datetime' in df_comp_aligned.columns:
            primary_start = df_filtered[x_axis].min()
            comp_start = df_comp_aligned['datetime'].min()
            time_offset = primary_start - comp_start
            df_comp_aligned['datetime'] = df_comp_aligned['datetime'] + time_offset
            
            # Filter to primary range
            primary_end = df_filtered[x_axis].max()
            mask = (df_comp_aligned['datetime'] >= primary_start) & (df_comp_aligned['datetime'] <= primary_end)
            df_comp_aligned = df_comp_aligned[mask].copy()
    
    # Label suffixes
    primary_suffix = f" [{primary_label}]" if primary_label else ""
    comp_suffix = f" [{comparison_label}]" if comparison_label else " [Comp]"
    
    try:
        # Calculate moving average
        window_size = max(10, len(df_filtered) // 50)
        df_filtered['tilt_angle_avg'] = df_filtered['tilt_angle'].rolling(window=window_size, center=True).mean()
        
        fig = go.Figure()
        
        # Trace 1: Raw Data (Primary)
        fig.add_trace(go.Scatter(
            x=df_filtered[x_axis],
            y=df_filtered['tilt_angle'],
            mode='lines',
            name=f'Tilt Angle{primary_suffix}',
            line=dict(color=tilt_color, width=1),
            opacity=0.5,
            hovertemplate=f'Tilt Angle{primary_suffix}: %{{y:.2f}} deg<br>Time: %{{x}}<extra></extra>'
        ))
        
        # Trace 2: Average (Primary)
        fig.add_trace(go.Scatter(
            x=df_filtered[x_axis],
            y=df_filtered['tilt_angle_avg'],
            mode='lines',
            name=f'Tilt Angle Avg{primary_suffix}',
            line=dict(color=tilt_color, width=3),
            hovertemplate=f'Tilt Angle Avg{primary_suffix}: %{{y:.2f}} deg<br>Time: %{{x}}<extra></extra>'
        ))
        
        # Comparison traces
        if df_comp_aligned is not None and not df_comp_aligned.empty:
            # Calculate moving average for comparison
            window_size_c = max(10, len(df_comp_aligned) // 50)
            df_comp_aligned['tilt_angle_avg'] = df_comp_aligned['tilt_angle'].rolling(window=window_size_c, center=True).mean()
            
            # Trace 3: Raw Data (Comparison) - dashed for clear distinction
            fig.add_trace(go.Scatter(
                x=df_comp_aligned[x_axis],
                y=df_comp_aligned['tilt_angle'],
                mode='lines',
                name=f'Tilt Angle{comp_suffix}',
                line=dict(color=comp_tilt_color, width=1.5, dash='dot'),
                hovertemplate=f'Tilt Angle{comp_suffix}: %{{y:.2f}} deg<br>Time: %{{x}}<extra></extra>'
            ))
            
            # Trace 4: Average (Comparison) - dashed for clear distinction
            fig.add_trace(go.Scatter(
                x=df_comp_aligned[x_axis],
                y=df_comp_aligned['tilt_angle_avg'],
                mode='lines',
                name=f'Tilt Angle Avg{comp_suffix}',
                line=dict(color=comp_tilt_color, width=3, dash='dash'),
                hovertemplate=f'Tilt Angle Avg{comp_suffix}: %{{y:.2f}} deg<br>Time: %{{x}}<extra></extra>'
            ))

        fig.update_layout(
            title="Tilt Angle (Calculated)",
            xaxis_title="Time" if x_axis == 'datetime' else "Sample ID",
            yaxis_title="Tilt Angle (deg)",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True,
            hovermode='x unified'
        )
        
        # Add transmission quality visualization
        if show_quality:
            # Check purely based on gaps + local validity of tilt_angle
            stats, global_gaps, local_gaps = analyze_transmission_quality(df_filtered, x_axis, column_name='tilt_angle')
            
            # Add global gaps
            for gap in global_gaps:
                fig.add_vrect(
                    x0=gap['start'],
                    x1=gap['end'],
                    fillcolor="red",
                    opacity=0.1,
                    layer="below",
                    line_width=0
                )
                fig.add_annotation(
                    x=gap['start'],
                    y=1,
                    yref="paper",
                    text="No Signal",
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=8, color="red")
                )
            
            # Add local gaps (invalid tilt values)
            if local_gaps:
                fault_times = [g['start'] for g in local_gaps]
                y_pos = df_filtered['tilt_angle'].min() if not pd.isna(df_filtered['tilt_angle'].min()) else 0
                
                fig.add_trace(go.Scatter(
                    x=fault_times,
                    y=[y_pos] * len(fault_times),
                    mode='markers',
                    marker=dict(symbol='x', color='orange', size=8),
                    name='Invalid Value',
                    hoverinfo='skip'
                ))

        st.plotly_chart(fig, width="stretch", key="tilt_main_plot")
        
        # Show stats metrics (Quality)
        if show_quality:
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Success Rate", f"{stats['success_rate']:.2f}%")
            with m2:
                st.metric("Valid Packets", stats['actual'] - stats['local_lost'])
            with m3:
                st.metric("Transmission Loss", stats['global_lost'])
            with m4:
                st.metric("Sensor Faults", stats['local_lost'])
            with m5:
                st.metric("Total Expected", stats['expected'])
                
            if stats['success_rate'] < 95.0:
                 st.error(f"Low quality detected: {stats['success_rate']:.2f}%")
                     
    except Exception as e:
        st.error(f"Error plotting tilt data: {e}")
                 
    return df_filtered, ['tilt_angle'] if 'tilt_angle' in df_filtered.columns else []

def render_fault_guide():
    """Renders a hidden-by-default guide for pump fault analysis."""
    with st.expander("Diagnostic Guide (Spectral Signatures)", expanded=False):
        st.markdown("""
        <style>
        .fault-card {
            background-color: #F0F2F6;
            border-left: 6px solid #ccc;
            padding: 12px;
            margin-bottom: 12px;
            border-radius: 6px;
        }
        .fault-title {
            font-size: 1.05em;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }
        .fault-desc {
            font-size: 0.9em;
            color: #333;
            line-height: 1.4;
        }
        .fault-cause {
            font-size: 0.85em;
            color: #555;
            font-style: italic;
            margin-top: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="fault-card" style="border-left-color: #3498DB;">
                <div class="fault-title" style="color: #3498DB;">Unbalance</div>
                <div class="fault-desc">
                    Dominant peak at <b>1x RPM</b> (Rotation Speed).<br>
                    Prevalent in Radial direction.
                </div>
                <div class="fault-cause">Cause: Dirt accumulation, impeller wear, missing weights.</div>
            </div>
            
            <div class="fault-card" style="border-left-color: #E67E22;">
                <div class="fault-title" style="color: #E67E22;">Misalignment</div>
                <div class="fault-desc">
                    Peaks at <b>1x and 2x RPM</b> (often 2x > 1x).<br>
                    Strong Axial component.
                </div>
                <div class="fault-cause">Cause: Shafts not aligned, soft foot.</div>
            </div>
            
             <div class="fault-card" style="border-left-color: #F1C40F;">
                <div class="fault-title" style="color: #F1C40F;">Electrical Faults</div>
                <div class="fault-desc">
                   Peak at line frequency (50Hz) and harmonics.<br>
                   Sidebands around 1x RPM.
                </div>
                <div class="fault-cause">Cause: Broken rotor bars, stator eccentricity.</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("""
            <div class="fault-card" style="border-left-color: #E74C3C;">
                <div class="fault-title" style="color: #E74C3C;">Looseness</div>
                <div class="fault-desc">
                    Series of harmonics (<b>1x, 2x, 3x... 10x</b>).<br>
                    Elevated noise floor.
                </div>
                <div class="fault-cause">Cause: Loose bolts, structural cracks, excessive clearance.</div>
            </div>
            
            <div class="fault-card" style="border-left-color: #9B59B6;">
                <div class="fault-title" style="color: #9B59B6;">Bearing Faults</div>
                <div class="fault-desc">
                    Non-synchronous peaks at <b>high frequency</b>.<br>
                    Energy "mound" or noise carpet.
                </div>
                <div class="fault-cause">Cause: Race/ball wear, poor lubrication.</div>
            </div>
            
            <div class="fault-card" style="border-left-color: #2ECC71;">
                <div class="fault-title" style="color: #2ECC71;">Cavitation / Flow</div>
                <div class="fault-desc">
                    Broadband noise (random) at medium-high frequencies.<br>
                    Unstable/fluctuating amplitudes.
                </div>
                <div class="fault-cause">Cause: Operation off-curve, suction problems.</div>
            </div>
            """, unsafe_allow_html=True)

def plot_fft_data(df: pd.DataFrame, show_quality: bool = True, show_mqtt_calc: bool = True, mqtt_stats: 'MqttStats' = None, df_comparison: pd.DataFrame = None, primary_db_name: str = "", comparison_db_name: str = ""):
    """Create interactive bar charts for FFT data."""
    if df.empty:
        st.warning("No FFT data available.")
        return
    
    # Get FFT columns (p_0 to p_999)
    fft_cols = [col for col in df.columns if col.startswith('p_')]
    # Ensure columns are sorted numerically (p_0, p_1, ..., p_10, ...)
    fft_cols = sorted(fft_cols, key=lambda x: int(x.split('_')[1]))
    
    if not fft_cols:
        st.warning("No FFT columns found.")
        return
    
    # Calculate Primary Dropdown Options & Defaults FIRST (to determine active FFT sample for plot settings)
    dropdown_options = []
    primary_prefix = f"[{primary_db_name}] " if primary_db_name else ""
    
    for idx, row in df.iterrows():
        axis = row.get('axis', 'N/A')  # X, Y, or Z
        max_amplitude = row.get('max_amplitude_g', 'N/A')  # Amplitude in G
        fft_type = row.get('type', 'N/A')  # acceleration or velocity
        num_points = int(row.get('number_of_points', len(fft_cols)))
        interval = row.get('human_interval_of_analysis', 'N/A')
        
        # Format amplitude
        amplitude_str = f"{max_amplitude} G"
        
        label = f"{primary_prefix}{axis} | {amplitude_str} | {fft_type} | {num_points} Hz | {interval}"
        dropdown_options.append(('primary', idx, label))
    
    # Determine default indices for X Acc and X Vel (from primary only)
    idx_x_acc = None
    idx_x_vel = None
    
    for i, (source, orig_idx, _) in enumerate(dropdown_options):
        r = df.iloc[orig_idx]
        r_axis = r.get('axis', 'N/A')
        r_type = r.get('type', 'N/A')
        
        if idx_x_acc is None and r_axis == 'X' and r_type == 'acceleration':
            idx_x_acc = i
        
        if idx_x_vel is None and r_axis == 'X' and r_type == 'velocity':
            idx_x_vel = i
            
        if idx_x_acc is not None and idx_x_vel is not None:
            break
    
    if idx_x_acc is None: idx_x_acc = 0
    if idx_x_vel is None: idx_x_vel = 1 if len(dropdown_options) > 1 else 0

    # Determine currently selected index (if previously selected in session state)
    current_selected_idx = idx_x_acc # Default is X Acc
    if "fft_selector_1" in st.session_state:
        # Check if the saved index is valid for current dropdown (range check)
        saved_idx = st.session_state["fft_selector_1"]
        if isinstance(saved_idx, int) and 0 <= saved_idx < len(dropdown_options):
            current_selected_idx = saved_idx
            
    # Get active number of points from the ACTUALLY SELECTED sample
    if 0 <= current_selected_idx < len(dropdown_options):
        _, active_row_idx, _ = dropdown_options[current_selected_idx]
        active_row = df.iloc[active_row_idx]
        active_num_points = int(active_row.get('number_of_points', len(fft_cols))) if pd.notna(active_row.get('number_of_points')) else len(fft_cols)
        fft_num_points = active_num_points
    else:
        fft_num_points = len(fft_cols)
    
    # User-adjustable number of frequencies to plot (shared across all FFT subtabs)
    # Default is fft_num_points (from selected row), max is len(fft_cols)
    max_freq_available = len(fft_cols)

    # Key must be dynamic based on selected index AND DB to force reset when context changes
    dynamic_key_freq = f"fft_freq_to_plot_{primary_db_name}_{current_selected_idx}"

    user_freq_to_plot = st.number_input(
        "Frequencies to Plot (Hz)",
        min_value=1,
        max_value=max_freq_available,
        value=min(fft_num_points, max_freq_available),
        step=50,
        key=dynamic_key_freq,
        help=f"Number of frequencies to display. Default from database: {fft_num_points} Hz. Max available: {max_freq_available} Hz. NaN values are treated as 0."
    )

    c_div, c_cut, c_cut_input, c_cut_label, c_space = st.columns([0.15, 0.15, 0.08, 0.07, 0.55])
    with c_div:
        st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True) # visual alignment
        divide_by_2 = st.toggle("Divide / 2", value=False, key="divide_by_2_toggle", help="Divides all amplitudes by 2. Except 0Hz")
    with c_cut:
        st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True) # visual alignment
        cut_low_freq = st.toggle("Cut Low Freq", value=False, key="cut_low_freq_toggle", help="Removes low frequencies.")
    with c_cut_input:
        st.markdown('<div style="margin-top: 14px;"></div>', unsafe_allow_html=True) # visual alignment
        cut_threshold = st.number_input("Cut Hz", min_value=1, value=3, step=1, key="cut_thresh_input", label_visibility="collapsed")
    with c_cut_label:
        st.markdown('<div style="margin-top: 22px; white-space: nowrap;">Cut Hz</div>', unsafe_allow_html=True)

    # Create subtabs
    fft_tab1, fft_tab2, fft_tab3 = st.tabs(["FFT", "FFT in Time", "Advanced Analysis"])
    
    with fft_tab1:
        
        # Build dropdown options from COMPARISON DB if available
        comp_prefix = f"[{comparison_db_name}] " if comparison_db_name else "[Comp] "
        comparison_dropdown_options = []
        if df_comparison is not None and not df_comparison.empty:
            # Get comparison FFT columns
            comp_fft_cols = [col for col in df_comparison.columns if col.startswith('p_')]
            comp_fft_cols = sorted(comp_fft_cols, key=lambda x: int(x.split('_')[1]))
            
            if comp_fft_cols:
                for idx, row in df_comparison.iterrows():
                    axis = row.get('axis', 'N/A')
                    max_amplitude = row.get('max_amplitude_g', 'N/A')
                    fft_type = row.get('type', 'N/A')
                    num_points = int(row.get('number_of_points', len(comp_fft_cols)))
                    interval = row.get('human_interval_of_analysis', 'N/A')
                    
                    amplitude_str = f"{max_amplitude} G"
                    label = f"{comp_prefix}{axis} | {amplitude_str} | {fft_type} | {num_points} Hz | {interval}"
                    comparison_dropdown_options.append(('comparison', idx, label))
        
        # Percentile selector - Uses value from sidebar slider (defined in main breakdown)
        percentile_value = st.session_state.get("percentile_slider", 90)
        
        # Helper function to plot FFT with optional comparison (supports cross-database)
        def plot_fft_comparison(primary_source: str, primary_idx: int, comp_source: str = None, comparison_idx: int = None, update_global_stats: bool = False, p_col=PASTEL_COLORS[3], c_col='#E67E22', p_label="Primary", c_label="Comparison"):
            # --- Primary Data ---
            if primary_source == 'primary':
                row = df.iloc[primary_idx]
            else:
                row = df_comparison.iloc[primary_idx] if df_comparison is not None else None
            
            if row is None:
                st.error("Primary data not available")
                return
                
            num_points = row.get('number_of_points', fft_num_points)
            # Use user_freq_to_plot (user-adjustable) for X-axis, handle NaN as 0
            freq_count = min(user_freq_to_plot, len(fft_cols))
            fft_values = [row[col] if pd.notna(row[col]) else 0 for col in fft_cols[:freq_count]]
            frequencies = np.arange(freq_count)
            
            # ═══ FFT UNIT CONVERSIONS ═══
            # ACCELERATION: Database stores G (gravitational units)
            #   → 1 G = 9.81 m/s²
            #   → Formula: m/s² = G × 9.81
            # VELOCITY: Firmware normalizes to (max_amplitude_g / 10) m/s
            #   → To get mm/s: raw × (max_amplitude_g / 10) × 1000 = raw × max_amplitude_g × 100
            #   → Formula: mm/s = raw × max_amplitude_g × 100
            fft_type = row.get('type', 'acceleration')
            if fft_type == 'acceleration':
                fft_values = [v * 9.81 for v in fft_values]  # G → m/s²
            elif fft_type == 'velocity':
                max_amp_g = row.get('max_amplitude_g', 16)
                if pd.notna(max_amp_g):
                    fft_values = [v * float(max_amp_g) * 100 for v in fft_values]  # raw → mm/s
            
            if cut_low_freq:
                if len(fft_values) > cut_threshold:
                    fft_values = fft_values[cut_threshold:]
                    frequencies = frequencies[cut_threshold:]
                else:
                    fft_values = []
                    frequencies = np.array([])
            
            if divide_by_2:
                # Divide by 2 ONLY if frequency is not 0Hz
                fft_values = [v / 2.0 if f != 0 else v for v, f in zip(fft_values, frequencies)]
            
            # --- Comparison Data ---
            comp_row = None
            comp_fft_values = []
            if comparison_idx is not None and comparison_idx >= 0:
                # Get comparison row from correct database
                if comp_source == 'primary':
                    comp_row = df.iloc[comparison_idx]
                elif comp_source == 'comparison' and df_comparison is not None:
                    comp_row = df_comparison.iloc[comparison_idx]
                
                if comp_row is not None:
                    # Handle NaN as 0 for comparison as well
                    comp_fft_values = [comp_row[col] if pd.notna(comp_row[col]) else 0 for col in fft_cols[:freq_count]]

                    # Align lengths if needed (though freq axis is index based 1Hz)
                    if len(comp_fft_values) > len(frequencies):
                        comp_fft_values = comp_fft_values[:len(frequencies)]
                    elif len(comp_fft_values) < len(frequencies):
                        comp_fft_values += [0] * (len(frequencies) - len(comp_fft_values))
                    
                    # Apply same unit conversion to comparison data (see formulas above)
                    comp_fft_type = comp_row.get('type', 'acceleration')
                    if comp_fft_type == 'acceleration':
                        comp_fft_values = [v * 9.81 for v in comp_fft_values]  # G → m/s²
                    elif comp_fft_type == 'velocity':
                        comp_max_amp_g = comp_row.get('max_amplitude_g', 16)
                        if pd.notna(comp_max_amp_g):
                            comp_fft_values = [v * float(comp_max_amp_g) * 100 for v in comp_fft_values]  # raw → mm/s
                    
                    if cut_low_freq:
                        if len(comp_fft_values) > cut_threshold:
                            comp_fft_values = comp_fft_values[cut_threshold:]
                        else:
                            comp_fft_values = []
                        
                    if divide_by_2:
                        # Divide by 2 ONLY if frequency is not 0Hz
                        # Note: frequencies array matches length of aligned data
                        comp_fft_values = [v / 2.0 if f != 0 else v for v, f in zip(comp_fft_values, frequencies)]

            # Calculate percentile thresholds
            primary_threshold = np.percentile(fft_values, percentile_value)
            comp_threshold = np.percentile(comp_fft_values, percentile_value) if comp_fft_values else None
            
            # Determine amplitude unit for hover templates
            amplitude_unit = 'm/s²' if fft_type == 'acceleration' else 'mm/s'
            comp_fft_type = comp_row.get('type', 'acceleration') if comp_row is not None else 'acceleration'
            comp_amplitude_unit = 'm/s²' if comp_fft_type == 'acceleration' else 'mm/s'
            
            # Create Figure
            fig = go.Figure()

            # --- Primary Colors ---
            primary_colors = p_col
            if show_mqtt_calc:
                # Use dynamic darker color for peaks
                darker_p = adjust_color_brightness(p_col, 0.6) 
                primary_colors = [darker_p if v > primary_threshold else p_col for v in fft_values]

            # Plot Primary
            fig.add_trace(go.Bar(
                x=frequencies,
                y=fft_values,
                name=p_label,
                marker_color=primary_colors,
                hovertemplate=f'<b>{p_label}</b><br>Freq: %{{x:.0f}} Hz<br>Amp: %{{y:.4f}} {amplitude_unit}<extra></extra>'
            ))

            # --- Comparison Colors ---
            if comp_fft_values:
                comp_colors = c_col
                if show_mqtt_calc:
                    darker_c = adjust_color_brightness(c_col, 0.6)
                    comp_colors = [darker_c if v > comp_threshold else c_col for v in comp_fft_values]

                fig.add_trace(go.Bar(
                    x=frequencies,
                    y=comp_fft_values,
                    name=c_label,
                    marker_color=comp_colors,
                    opacity=0.75,
                    hovertemplate=f'<b>{c_label}</b><br>Freq: %{{x:.0f}} Hz<br>Amp: %{{y:.4f}} {comp_amplitude_unit}<extra></extra>'
                ))

            # Add horizontal line for primary percentile
            if show_mqtt_calc:
                fig.add_hline(
                    y=primary_threshold,
                    line_dash="dot",
                    line_color=p_col,
                    annotation_text=f"Primary {percentile_value}th: {primary_threshold:.4f}",
                    annotation_position="top right"
                )
                if comp_threshold is not None:
                    fig.add_hline(
                        y=comp_threshold,
                        line_dash="dot",
                        line_color=c_col,
                        annotation_text=f"Comp {percentile_value}th: {comp_threshold:.4f}",
                        annotation_position="bottom right"
                    )
            
            # amplitude_unit already calculated above for hover templates
            
            title_text = f"FFT Spectrum - Sample {primary_idx + 1}"
            if comp_row is not None:
                title_text += f" vs Sample {comparison_idx + 1}"

            # X-AXIS LABELS
            target_dtick = 25

            fig.update_layout(
                title=title_text,
                xaxis=dict(
                    title="Frequency (Hz)",
                    tickmode='linear',
                    tick0=0,
                    dtick=target_dtick
                ),
                yaxis_title=f"Amplitude ({amplitude_unit})",
                height=600,
                margin=dict(l=50, r=50, t=50, b=50),
                barmode='overlay', # Overlay bars
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                hovermode='x unified'
            )
            
            # Explicitly mark the start frequency as requested (ONLY if cut is enabled)
            if cut_low_freq and len(frequencies) > 0:
                start_f = frequencies[0]
                fig.add_vline(
                    x=start_f,
                    line_width=1,
                    line_dash="dot",
                    line_color="#888",
                    opacity=0.7,
                    annotation_text=f"{int(start_f)}Hz",
                    annotation_position="top right",
                    annotation_font=dict(size=12, color="#555")
                )
            
            st.plotly_chart(fig, key=f"fft_plot_combined", width="stretch")
            
            # --- Statistics ---
            # Helper for stats
            def calc_stats(vals, thresh):
                ground_vals = [v for v in vals if v <= thresh]
                g_avg = sum(ground_vals) / len(ground_vals) if ground_vals else 0
                peaks_abv = sum(1 for v in vals if v > thresh)
                return {
                    'max': max(vals) if vals else 0,
                    'min': min(vals) if vals else 0,
                    'mean': sum(vals)/len(vals) if vals else 0,
                    'ground': g_avg,
                    'peaks_count': peaks_abv
                }

            stats_prim = calc_stats(fft_values, primary_threshold)
            stats_comp = calc_stats(comp_fft_values, comp_threshold) if comp_fft_values else None

            # Display Stats
            # We color code: Blue for Primary, Orange for Comparison
            
            st.markdown("### Statistics")
            
            # Metrics Columns
            # We will show: Label | Primary | Comparison
            
            cols = st.columns(5)
            labels = ["Max Amp", "Min Amp", "Mean Amp", "Ground Avg"]
            keys = ['max', 'min', 'mean', 'ground']
            
            if show_mqtt_calc:
                labels.append(f"Peaks > {percentile_value}th")
                keys.append('peaks_count')

            for i, (label, key) in enumerate(zip(labels, keys)):
                with cols[i]:
                    # Format logic: integer for peaks_count, float for others with unit
                    if key == 'peaks_count':
                         p_val_str = f"{int(stats_prim[key])}"
                    else:
                         p_val_str = f"{stats_prim[key]:.4f} {amplitude_unit}"

                    c_val_str = "-"
                    
                    if stats_comp:
                        if key == 'peaks_count':
                             c_val_str = f"{int(stats_comp[key])}"
                        else:
                             c_val_str = f"{stats_comp[key]:.4f} {amplitude_unit}"
                    
                    # Create a card-like container
                    card_html = f"""
                    <div style="
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 10px;
                        background-color: rgba(245, 245, 245, 0.2);
                        text-align: center;
                        margin-bottom: 10px;
                    ">
                        <div style="
                            color: #555; 
                            font-size: 0.9rem; 
                            margin-bottom: 8px; 
                            padding-bottom: 5px; 
                            font-weight: 500; 
                            border-bottom: 1px solid #ddd;
                        ">{label}</div>
                        <!-- Primary -->
                        <div style="color:{p_col}; font-size:2.2rem; line-height:1.2;">{p_val_str}</div>
                        <!-- Comparison -->
                        <div style="color:{c_col}; font-size:2.2rem; line-height:1.2; margin-top: 5px;">{c_val_str if stats_comp else '<span style="color:#ccc; font-size:1.5rem;">-</span>'}</div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

            


            # --- Top 5 Peaks Display ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='color:black; font-size:2.0rem; margin-bottom:10px;'>Dominant Frequencies</div>", unsafe_allow_html=True)
            
            def get_top_peaks(vals, freqs):
                vals_np = np.array(vals)
                l_max = []
                if len(vals_np) >= 3:
                    for i in range(1, len(vals_np) - 1):
                        if vals_np[i] > vals_np[i-1] and vals_np[i] > vals_np[i+1]:
                            l_max.append(i)
                    if vals_np[0] > vals_np[1]: l_max.append(0)
                    if vals_np[-1] > vals_np[-2]: l_max.append(len(vals_np)-1)
                else:
                    l_max = list(range(len(vals_np)))
                
                l_max.sort(key=lambda x: vals_np[x], reverse=True)
                return  l_max[:5]

            p_peaks = get_top_peaks(fft_values, frequencies)
            c_peaks = get_top_peaks(comp_fft_values, frequencies) if comp_fft_values else []

            pk_cols = st.columns(5)
            for i, col in enumerate(pk_cols):
                with col:
                    # Prepare content
                    peak_label = f"Peak {i+1}"
                    
                    # Primary Content
                    p_content = "-"
                    if i < len(p_peaks):
                        p_idx = p_peaks[i]
                        p_content = f"{int(frequencies[p_idx])} Hz<br><span style='font-size:1.0rem; opacity:0.8;'>({fft_values[p_idx]:.3f} {amplitude_unit})</span>"
                    
                    # Comparison Content
                    c_content = "-"
                    if stats_comp:
                        if i < len(c_peaks):
                            c_idx = c_peaks[i]
                            c_content = f"{int(frequencies[c_idx])} Hz<br><span style='font-size:1.0rem; opacity:0.8;'>({comp_fft_values[c_idx]:.3f} {amplitude_unit})</span>"
                        else:
                            c_content = "-" # Placeholder if comp exists but no peak this far
                    elif i == 0 and not stats_comp: # Only show dash for first if no comp, or handled by logic below
                         pass

                    # Build Card
                    card_html = f"""
                    <div style="
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        padding: 10px;
                        background-color: rgba(245, 245, 245, 0.2);
                        text-align: center;
                    ">
                        <div style="
                            color: #555; 
                            font-size: 0.9rem; 
                            margin-bottom: 8px; 
                            padding-bottom: 5px;
                            font-weight: 500; 
                            border-bottom: 1px solid #ddd;
                        ">{peak_label}</div>
                        <!-- Primary -->
                        <div style="color:{p_col}; font-size:1.6rem; line-height:1.2;">{p_content}</div>
                        <!-- Comparison -->
                        <div style="color:{c_col}; font-size:1.6rem; line-height:1.2; margin-top: 5px;">
                            {c_content if stats_comp else '<span style="color:#ccc; font-size:1.5rem; display:none;">-</span>'}
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

            if show_mqtt_calc:
                # MQTT Calc logic (Primary Only)
                st.markdown("---")
                st.subheader("MQTT Analysis")
                
                # Construct optimized payload
                ts_val = 0
                if 'unix_start' in row and pd.notna(row['unix_start']):
                     ts_val = int(row['unix_start'])

                peaks_list = []
                for i, val in enumerate(fft_values):
                    if val > primary_threshold:
                        peaks_list.append([int(frequencies[i]), float(round(val, 3))])
                
                payload = {
                    "type": "acc" if row.get('type') == 'acceleration' else ("vel" if row.get('type') == 'velocity' else row.get('type', 'N/A')),
                    "points": int(num_points),
                    "axis": row.get('axis', 'N/A'),
                    "ts": ts_val,
                    "avg": float(round(stats_prim['ground'], 2)),
                    "peaks": peaks_list
                }
                
                json_str = json.dumps(payload, separators=(',', ':'))
                payload_size = len(json_str)
                
                # Create binary mask: 1 if peak exceeds threshold, 0 otherwise
                bit_mask = ''.join(['1' if val > primary_threshold else '0' for val in fft_values])
                
                # Convert binary to hex (groups of 4 bits)
                # Pad the binary mask to be a multiple of 4
                padded_length = ((len(bit_mask) + 3) // 4) * 4
                padded_mask = bit_mask.ljust(padded_length, '0')
                
                # Convert to hex, 4 bits at a time
                hex_chars = []
                for i in range(0, len(padded_mask), 4):
                    nibble = padded_mask[i:i+4]
                    hex_char = format(int(nibble, 2), 'X')
                    hex_chars.append(hex_char)
                hex_mask = ''.join(hex_chars)
                
                # Compact the hexmask: replace consecutive zeros (>2) with -N-
                def compact_hex_mask(hex_str):
                    result = []
                    i = 0
                    while i < len(hex_str):
                        if hex_str[i] == '0':
                            zero_count = 0
                            j = i
                            while j < len(hex_str) and hex_str[j] == '0':
                                zero_count += 1
                                j += 1
                            if zero_count > 2:
                                result.append(f"-{zero_count}-")
                            else:
                                result.append('0' * zero_count)
                            i = j
                        else:
                            result.append(hex_str[i])
                            i += 1
                    return ''.join(result)
                
                compacted_mask = compact_hex_mask(hex_mask)
                
                # Calculate sizes
                bit_mask_peaks_count = bit_mask.count('1')
                original_json_size = payload_size
                
                # Build NEW optimized JSON with compacted mask + amplitudes
                # Extract only the amplitudes of peaks that exceed threshold
                peak_amplitudes = [float(round(val, 3)) for val in fft_values if val > primary_threshold]
                
                optimized_payload = {
                    "type": payload.get("type", "N/A"),
                    "points": payload.get("points", 0),
                    "axis": payload.get("axis", "N/A"),
                    "ts": payload.get("ts", 0),
                    "avg": payload.get("avg", 0),
                    "mask": compacted_mask,
                    "amps": peak_amplitudes
                }
                optimized_json_str = json.dumps(optimized_payload, separators=(',', ':'))
                optimized_json_size = len(optimized_json_str)
                
                # --- Side by side: Original JSON vs Optimized JSON ---
                col_old, col_new = st.columns(2)
                
                with col_old:
                    st.markdown("**Original JSON Payload**")
                    st.code(json_str, language='json')
                    st.caption(f"Size: **{original_json_size} bytes**")
                
                with col_new:
                    st.markdown("**Optimized JSON (Mask + Amplitudes)**")
                    st.code(optimized_json_str, language='json')
                    st.caption(f"Size: **{optimized_json_size} bytes**")
                
                # --- Mask Details (expandable) ---
                with st.expander("View Bit Mask Details", expanded=False):
                    st.markdown("**Bit Mask (Binary)**")
                    st.code(padded_mask, language=None)
                    st.markdown("**HexMask**")
                    st.code(hex_mask, language=None)
                    st.markdown("**Compacted Mask**")
                    st.code(compacted_mask, language=None)
                
                # --- Final Analysis ---
                st.markdown("##### Compression Analysis")
                
                savings = original_json_size - optimized_json_size
                savings_pct = (savings / original_json_size * 100) if original_json_size > 0 else 0
                
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                with col_a1:
                    st.metric("Peaks Detected", bit_mask_peaks_count)
                with col_a2:
                    st.metric("Original JSON", f"{original_json_size} B")
                with col_a3:
                    st.metric("Optimized JSON", f"{optimized_json_size} B")
                with col_a4:
                    if savings > 0:
                        st.metric("Savings", f"{savings_pct:.1f}%", delta=f"-{savings} bytes", delta_color="inverse")
                    else:
                        st.metric("Overhead", f"{abs(savings_pct):.1f}%", delta=f"+{abs(savings)} bytes", delta_color="normal")

                if update_global_stats and mqtt_stats:
                     total_fft_samples = len(df)
                     total_fft_bytes = optimized_json_size * total_fft_samples
                     mqtt_stats.add("FFT", total_fft_bytes, total_fft_samples, f"{total_fft_samples} Samples | P{percentile_value} | {bit_mask_peaks_count} Peaks | Optimized")

        # --- UI Selection ---
        st.subheader("FFT Analysis")
        
        # Primary Selector (only from primary DB)
        col_p1, col_p2 = st.columns([0.85, 0.15])
        with col_p1:
            selected_idx_1 = st.selectbox(
                "Primary FFT Sample",
                options=range(len(dropdown_options)),
                format_func=lambda x: dropdown_options[x][2],  # Use label from 3-tuple
                key="fft_selector_1",
                index=idx_x_acc
            )
        with col_p2:
            p_color_choice = st.color_picker("Primary", value=PASTEL_COLORS[3], key="p_color_picker", label_visibility="hidden")
        
        # Get primary source and index
        primary_source, primary_row_idx, primary_label_str = dropdown_options[selected_idx_1]
        
        # Comparison Selector - include both DBs
        # Format: (source, idx, label) where source is 'primary' or 'comparison'
        none_option = [('none', -1, "None")]
        all_comp_options = none_option + dropdown_options + comparison_dropdown_options
        
        # Only show Comparison Selector if df_comparison is active
        if df_comparison is not None and not df_comparison.empty:
            col_c1, col_c2 = st.columns([0.85, 0.15])
            with col_c1:
                selected_comp_idx = st.selectbox(
                    "Comparison FFT Sample",
                    options=range(len(all_comp_options)),
                    format_func=lambda x: all_comp_options[x][2],  # Use label from 3-tuple
                    key="fft_selector_2",
                    index=0  # Default to None
                )
            with col_c2:
                c_color_choice = st.color_picker("Compare", value='#E67E22', key="c_color_picker", label_visibility="hidden")
        else:
             selected_comp_idx = 0 # Default to 'None' option (index 0 in all_comp_options which starts with none)
             c_color_choice = '#E67E22'
        
        # Get comparison source and index
        comp_source, comp_row_idx, comp_label_str = all_comp_options[selected_comp_idx]
        if comp_source == 'none':
            comp_row_idx = None
            comp_source = None

        # Plot
        plot_fft_comparison(primary_source, primary_row_idx, comp_source, comp_row_idx, update_global_stats=True, p_col=p_color_choice, c_col=c_color_choice, p_label=primary_label_str, c_label=comp_label_str)
    # Common filters for subtabs 2 and 3
    # Get available axes and types
    available_axes = df['axis'].dropna().unique().tolist() if 'axis' in df.columns else ['X', 'Y', 'Z']
    available_types = df['type'].dropna().unique().tolist() if 'type' in df.columns else ['acceleration', 'velocity']

    with fft_tab2:
        st.subheader("FFT Heatmap Over Time")
        
        # Default defaults for Heatmap
        try:
             def_axis_idx = available_axes.index('X')
        except ValueError:
             def_axis_idx = 0
             
        try:
             def_type_idx = available_types.index('acceleration')
        except ValueError:
             def_type_idx = 0
        
        col1, col2 = st.columns(2)
        with col1:
            selected_axis_hm = st.selectbox("Select Axis", options=available_axes, key="heatmap_axis", index=def_axis_idx)
        with col2:
            selected_type_hm = st.selectbox("Select Type", options=available_types, key="heatmap_type", index=def_type_idx)
        
        # Filter dataframe
        df_hm = df.copy()
        if 'axis' in df_hm.columns:
            df_hm = df_hm[df_hm['axis'] == selected_axis_hm]
        if 'type' in df_hm.columns:
            df_hm = df_hm[df_hm['type'] == selected_type_hm]
            
        # Slider for number of samples (only show if more than 1 sample)
        if not df_hm.empty:
            if len(df_hm) > 1:
                num_samples_hm = st.slider(
                    "Number of FFT Samples to Plot",
                    min_value=1,
                    max_value=len(df_hm),
                    value=len(df_hm),
                    step=1,
                    key="fft_count_slider",
                    help="Select number of samples to display, starting from the oldest."
                )
                df_hm = df_hm.iloc[:num_samples_hm]
        
        if df_hm.empty:
            st.warning(f"No FFT data found for Axis: {selected_axis_hm}, Type: {selected_type_hm}")
        else:
            heatmap_data = []
            y_labels = []
            
            # Determine Frequency Axis for Heatmap
            # Use user_freq_to_plot (user-adjustable, shared across tabs)
            max_freq_points = min(user_freq_to_plot, len(fft_cols))
            
            freqs_hm = np.arange(max_freq_points)
            if cut_low_freq:
                 if len(freqs_hm) > 3:
                      freqs_hm = freqs_hm[3:]
                 else:
                      freqs_hm = np.array([])
            
            cols_hm = fft_cols[:max_freq_points]
            
            for idx, row in df_hm.iterrows():
                fft_vals = [row[col] if pd.notna(row[col]) else 0 for col in cols_hm]
                
                # Unit conversion: Acc G→9.81→m/s² | Vel raw×(max_amp_g×100)→mm/s
                row_type = row.get('type', 'acceleration')
                if row_type == 'acceleration':
                    fft_vals = [v * 9.81 for v in fft_vals]  # G → m/s²
                elif row_type == 'velocity':
                    max_amp_g = row.get('max_amplitude_g', 16)
                    if pd.notna(max_amp_g):
                        fft_vals = [v * float(max_amp_g) * 100 for v in fft_vals]  # raw → mm/s
                
                if cut_low_freq:
                    if len(fft_vals) > 3:
                        fft_vals = fft_vals[3:]
                    else:
                        fft_vals = []
                    fft_vals = [v / 2.0 for v in fft_vals]
                
                heatmap_data.append(fft_vals)
                interval = row.get('human_interval_of_analysis', f'Sample {idx}')
                y_labels.append(str(interval))
            
            custom_blue_scale = [
                [0.0, "rgb(15, 25, 50)"],    # Deep Navy base
                [0.4, "rgb(30, 80, 180)"],  # Smooth transition starts later
                [0.6, "rgb(60, 140, 230)"],  # Rich Blue
                [1.0, "rgb(160, 225, 255)"]  # Bright Blue highlight
            ]
            
            # Determine unit for heatmap hover
            heatmap_amplitude_unit = 'm/s²' if selected_type_hm == 'acceleration' else 'mm/s'
            
            # Calculate zmax from current (filtered) data for proper color scaling
            heatmap_zmax = max(max(row) for row in heatmap_data) if heatmap_data else 1
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                zmin=0,
                zmax=heatmap_zmax,
                x=freqs_hm,
                y=y_labels,
                colorscale=custom_blue_scale,
                colorbar=dict(
                    title='Amplitude',
                    thickness=20,
                    len=0.8,
                    ticks='outside'
                ),
                hovertemplate=f'Frequency: %{{x:.0f}} Hz<br>Time: %{{y}}<br>Amplitude: %{{z:.4f}} {heatmap_amplitude_unit}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"FFT Spectrogram Over Time - Axis: {selected_axis_hm}, Type: {selected_type_hm}",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Time Interval",
                height=max(500, len(heatmap_data) * 25),
                margin=dict(l=50, r=50, t=60, b=50),
                hovermode='closest'
            )
            st.plotly_chart(fig, key=f"fft_heatmap_{selected_axis_hm}_{selected_type_hm}", width="stretch")
            
            st.markdown("---")
            
            with st.expander("3D Surface Evolution", expanded=False):
                # Prepare data for 3D plot (using same data as heatmap)
                # Limit samples for 3D performance if too many, but allow navigation
                MAX_3D_SAMPLES = 50
                total_heatmap_samples = len(heatmap_data)
                
                if total_heatmap_samples > MAX_3D_SAMPLES:
                    # Slider inside expander
                    start_index_3d = st.slider(
                        "Navigate 3D History (Start Sample)",
                        min_value=0,
                        max_value=total_heatmap_samples - MAX_3D_SAMPLES,
                        value=max(0, total_heatmap_samples - MAX_3D_SAMPLES), # Default to latest
                        step=1,
                        key="fft_3d_slider",
                        help=f"Select the starting sample for the 3D plot. Shows {MAX_3D_SAMPLES} samples."
                    )
                    end_index_3d = start_index_3d + MAX_3D_SAMPLES
                    
                    st.info(f"Displaying samples {start_index_3d} to {end_index_3d} (of {total_heatmap_samples})")
                    
                    z_3d = heatmap_data[start_index_3d:end_index_3d]
                    y_3d = y_labels[start_index_3d:end_index_3d]
                else:
                    z_3d = heatmap_data
                    y_3d = y_labels
                    st.info(f"Showing all {total_heatmap_samples} samples. (Slider appears if > {MAX_3D_SAMPLES})")

                fig_3d = go.Figure(data=[go.Surface(
                    z=z_3d,
                    x=freqs_hm,
                    y=y_3d,
                    colorscale=custom_blue_scale,
                    contours_z=dict(
                        show=True,
                        usecolormap=True,
                        project_z=True,
                        highlightcolor="white",
                        highlightwidth=2
                    )
                )])
                
                fig_3d.update_layout(
                    title=f'3D FFT Evolution - Axis: {selected_axis_hm}, Type: {selected_type_hm}',
                    scene = dict(
                        xaxis_title='Frequency (Hz)',
                        yaxis_title='Time',
                        zaxis_title='Amplitude',
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                    ),
                    height=700,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                st.plotly_chart(fig_3d, width="stretch", key="3d_fft_tab2")
            
            # Fault Guide
            render_fault_guide()

    with fft_tab3:
        st.subheader("Advanced FFT Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_axis_adv = st.selectbox("Select Axis", options=available_axes, key="adv_axis")
        with col2:
            selected_type_adv = st.selectbox("Select Type", options=available_types, key="adv_type")
            
        # Filter dataframe
        df_adv = df.copy()
        if 'axis' in df_adv.columns:
            df_adv = df_adv[df_adv['axis'] == selected_axis_adv]
        if 'type' in df_adv.columns:
            df_adv = df_adv[df_adv['type'] == selected_type_adv]
            
        if df_adv.empty:
            st.warning("No data for selected filters.")
        else:
            # Prepare data commonly used
            timestamps = []
            spectra = []
            
            # Use user_freq_to_plot (user-adjustable, shared across tabs) for frequency axis
            freqs_adv = np.arange(min(user_freq_to_plot, len(fft_cols)))
            if cut_low_freq:
                 if len(freqs_adv) > 3:
                      freqs_adv = freqs_adv[3:]
                 else:
                      freqs_adv = np.array([])
            hz_per_bin = 1.0
            
            # Calculate default values for energy bands based on user_freq_to_plot
            # For 500 points: Low = 100Hz, Medium = 250Hz
            # For 1000 points: Low = 200Hz, Medium = 500Hz
            default_low_band = int(user_freq_to_plot * 0.2)   # 20% of max frequency
            default_med_band = int(user_freq_to_plot * 0.5)   # 50% of max frequency
            
            # Limit columns to the frequency range determined by user_freq_to_plot
            cols_adv = fft_cols[:min(user_freq_to_plot, len(fft_cols))]
            
            for idx, row in df_adv.iterrows():
                vals = [row[col] if pd.notna(row[col]) else 0 for col in cols_adv]
                
                # Unit conversion: Acc G→9.81→m/s² | Vel raw×(max_amp_g×100)→mm/s
                row_type = row.get('type', 'acceleration')
                if row_type == 'acceleration':
                    vals = [v * 9.81 for v in vals]  # G → m/s²
                elif row_type == 'velocity':
                    max_amp_g = row.get('max_amplitude_g', 16)
                    if pd.notna(max_amp_g):
                        vals = [v * float(max_amp_g) * 100 for v in vals]  # raw → mm/s

                if cut_low_freq:
                    if len(vals) > 3:
                        vals = vals[3:]
                    else:
                        vals = []
                    vals = [v / 2.0 for v in vals]
                
                spectra.append(vals)
                ts = row.get('human_interval_of_analysis', f'Sample {idx}')
                timestamps.append(str(ts))
            
            # --- 1. Peak Tracking ---
            st.markdown("#### 1. Dominant Peak Tracking")
            st.info("**How to read**: This chart tracks the frequency (Y-axis) and amplitude (bubble size) of the most dominant peaks over time. It helps identify if fault frequencies are stable or shifting (e.g. speed changes).")
            
            top_n = st.slider("Number of Peaks to Track", 1, 5, 3, key="top_n_peaks")
            
            peak_data = []
            for i, spec in enumerate(spectra):
                # Find indices of top N peaks
                # use np.argsort to get indices of top elements, then take last N and reverse
                indices = np.argsort(spec)[-top_n:][::-1]
                for p_idx in indices:
                    peak_data.append({
                        'Time': timestamps[i],
                        'Frequency': freqs_adv[p_idx],
                        'Amplitude': spec[p_idx],
                        'Rank': f"Peak {list(indices).index(p_idx) + 1}"
                    })
            
            df_peaks = pd.DataFrame(peak_data)
            
            fig_peaks = px.scatter(
                df_peaks,
                x='Time',
                y='Frequency',
                size='Amplitude',
                color='Rank',
                title=f"Top {top_n} Frequencies Over Time",
                color_discrete_sequence=PASTEL_COLORS,
                hover_data=['Amplitude']
            )
            
            fig_peaks.update_layout(height=450)
            st.plotly_chart(fig_peaks, width="stretch", key="peak_tracking")
            
            st.markdown("---")
            
            # --- 2. Energy Bands ---
            st.markdown("#### 2. Energy Bands Analysis")
            st.info("**How to read**: This chart displays the total vibration energy summed up within specific frequency bands (Low, Medium, High). It is useful for distinguishing between different types of faults (e.g. Unbalance in Low vs Bearing faults in Medium/High).")
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                low_band_max = st.number_input("Low Band Max (Hz)", value=default_low_band, step=10)
            with col_b2:
                med_band_max = st.number_input("Medium Band Max (Hz)", value=default_med_band, step=50)
                
            # Convert Hz thresholds to indices
            low_idx = int(low_band_max / hz_per_bin)
            med_idx = int(med_band_max / hz_per_bin)
            
            if cut_low_freq:
                low_idx = max(0, low_idx - 3)
                med_idx = max(low_idx, med_idx - 3)
            
            # Clamp indices to the frequency range
            max_freq_idx = len(freqs_adv)
            low_idx = min(max(0, low_idx), max_freq_idx)
            med_idx = min(max(low_idx, med_idx), max_freq_idx)
            
            # Calculate energies
            energies = []
            for spec in spectra:
                # Ensure indices don't exceed spectrum length
                spec_len = len(spec)
                l_idx = min(low_idx, spec_len)
                m_idx = min(med_idx, spec_len)
                low_energy = sum(spec[:l_idx])
                med_energy = sum(spec[l_idx:m_idx])
                high_energy = sum(spec[m_idx:])
                energies.append({
                    'Low Band': low_energy,
                    'Medium Band': med_energy,
                    'High Band': high_energy
                })
            
            df_energy = pd.DataFrame(energies, index=timestamps)
            
            fig_energy = go.Figure()
            # Plot each band
            for band_name, color in zip(['Low Band', 'Medium Band', 'High Band'], [PASTEL_BLUE, PASTEL_GREEN, PASTEL_RED]):
                fig_energy.add_trace(go.Scatter(
                    x=timestamps,
                    y=df_energy[band_name],
                    mode='lines+markers',
                    name=band_name,
                    line=dict(color=color)
                ))
                
            fig_energy.update_layout(
                title=f"Vibration Energy in Frequency Bands (Low < {low_band_max}Hz | Med {low_band_max}-{med_band_max}Hz | High > {med_band_max}Hz)",
                xaxis_title="Time",
                yaxis_title="Total Energy (Sum of Amplitudes)",
                height=450,
                hovermode='x unified'
            )
            st.plotly_chart(fig_energy, width="stretch", key="energy_bands")

    # Transmission Quality Analysis at the bottom
    if show_quality:
        st.markdown("---")
        st.subheader("Transmission Quality")
        # Ensure we have a time column or ID for analysis
        time_col = 'datetime'
        if 'datetime' not in df.columns:
            # Try to use unix_start if available
            if 'unix_start' in df.columns:
                 df['datetime'] = pd.to_datetime(df['unix_start'], unit='s')
            elif 'unix_timestamp' in df.columns:
                 df['datetime'] = pd.to_datetime(df['unix_timestamp'], unit='s')
        
        # Use 'max_amplitude_g' for local validity check if available
        check_col = 'max_amplitude_g' if 'max_amplitude_g' in df.columns else None
        
        stats, global_gaps, local_gaps = analyze_transmission_quality(df, time_col, column_name=check_col)
        
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Success Rate", f"{stats['success_rate']:.2f}%")
        with m2:
            st.metric("Valid Packets", stats['actual'] - stats['local_lost'])
        with m3:
            st.metric("Transmission Loss", stats['global_lost'], help="Packets not received (network gap)")
        with m4:
            st.metric("Sensor Faults", stats['local_lost'], help="Packets received but value is invalid")
        with m5:
            st.metric("Total Expected", stats['expected'])
            
        if stats['success_rate'] < 95.0:
             st.error(f"Issue detected with FFT Data: {stats['total_lost']} total lost packets.")



def plot_comparison_data(conn: sqlite3.Connection, comp_conn: sqlite3.Connection = None, primary_label: str = "Primary", comparison_label: str = "Comparison"):
    """Create normalized interactive comparison graphs for all time series data."""
    
    # Initialize session state for toggles
    if 'compare_show_raw' not in st.session_state:
        st.session_state.compare_show_raw = False
    if 'compare_disable_norm' not in st.session_state:
        st.session_state.compare_disable_norm = False
    
    # Read toggle values from session state
    show_raw = st.session_state.compare_show_raw
    disable_norm = st.session_state.compare_disable_norm
    
    st.subheader("Data Comparison")
    
    # Collect all available data sources
    available_data = {}
    
    # Helper to process dataframe columns
    def process_columns(key_prefix, table_name, df, source_prefix=""):
        if df.empty: return
        
        # 1. Coerce object columns to numeric where possible
        for col in df.columns:
            if col not in EXCLUDE_METADATA_COLS and df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 2. Filter valid numeric columns
        numeric_cols = [col for col in df.columns 
                      if col not in EXCLUDE_METADATA_COLS 
                      and pd.api.types.is_numeric_dtype(df[col])
                      and df[col].notna().any()] # Must have at least one value
        
        for col in numeric_cols:
            # Create a unique key including source prefix if applicable
            full_key = f"{source_prefix}{key_prefix}: {col}"
            available_data[full_key] = (table_name, col, df)

    # Determine prefixes for display
    p_prefix = f"{primary_label} - " if comp_conn else ""
    c_prefix = f"{comparison_label} - " if comp_conn else ""

    # Load and process PRIMARY data
    if check_table_exists(conn, 'sensor_data'):
        process_columns("Sensor", 'sensor_data', get_table_data(conn, 'sensor_data'), p_prefix)
        
    if check_table_exists(conn, 'power_analyzer_data'):
        process_columns("Power", 'power_analyzer_data', get_table_data(conn, 'power_analyzer_data'), p_prefix)
            
    if check_table_exists(conn, 'tilt_data'):
        process_columns("Tilt", 'tilt_data', get_table_data(conn, 'tilt_data'), p_prefix)
    
    # Load and process COMPARISON data
    if comp_conn:
        if check_table_exists(comp_conn, 'sensor_data'):
            process_columns("Sensor", 'sensor_data', get_table_data(comp_conn, 'sensor_data'), c_prefix)
            
        if check_table_exists(comp_conn, 'power_analyzer_data'):
            process_columns("Power", 'power_analyzer_data', get_table_data(comp_conn, 'power_analyzer_data'), c_prefix)
                
        if check_table_exists(comp_conn, 'tilt_data'):
            process_columns("Tilt", 'tilt_data', get_table_data(comp_conn, 'tilt_data'), c_prefix)
    
    if not available_data:
        st.warning("No time series data available for comparison.")
        return
    
    # --- Calculate Global Start Times (for relative alignment) ---
    primary_start_time = None
    comp_start_time = None
    
    # Helper to find min time across a list of dataframes
    def get_global_min_time(keys_filter):
        mins = []
        checked_ids = set()
        for k, (_, _, df) in available_data.items():
            if keys_filter(k):
                if id(df) in checked_ids: continue
                checked_ids.add(id(df))
                if 'datetime' in df.columns and not df['datetime'].isna().all():
                    mins.append(df['datetime'].min())
        return min(mins) if mins else None

    # Filter logic using the prefixes p_prefix and c_prefix defined earlier
    if comp_conn:
        primary_start_time = get_global_min_time(lambda k: k.startswith(p_prefix))
        comp_start_time = get_global_min_time(lambda k: k.startswith(c_prefix))
    else:
        # Single database mode
        primary_start_time = get_global_min_time(lambda k: True)

    
    # --- Time Range Slider ---
    all_mins = []
    all_maxs = []
    checked_dfs = set()
    
    for _, _, df in available_data.values():
        if id(df) in checked_dfs: continue
        checked_dfs.add(id(df))
        if 'datetime' in df.columns and not df['datetime'].isna().all():
            all_mins.append(df['datetime'].min())
            all_maxs.append(df['datetime'].max())
            
    if not all_mins or not all_maxs:
        st.warning("No valid timestamps found in data.")
        return # Cannot plot without time
    else:
        min_date_global = min(all_mins)
        max_date_global = max(all_maxs)
        
        date_range = st.slider(
            "Select Date/Time Range",
            min_value=min_date_global.to_pydatetime(),
            max_value=max_date_global.to_pydatetime(),
            value=(min_date_global.to_pydatetime(), max_date_global.to_pydatetime()),
            step=timedelta(seconds=1),
            format="DD/MM/YY HH:mm:ss",
            key="comparison_range_slider"
        )
    
    
    # --- Data Selection UI ---
    selected_series = []
    
    # Helper to clean labels for display (removes the long prefix)
    # The prefix is already evident from the section header
    def clean_label(k, prefix_to_remove):
        # Remove the source prefix first
        s = k.replace(prefix_to_remove, "")
        # Remove the type identifier "Sensor: " or "Tilt: ", etc
        s = s.replace("Sensor: ", "").replace("Tilt: ", "").replace("Power: ", "")
        return s

    if comp_conn:
        # --- Split View: Primary vs Comparison ---
        
        # 1. Primary DB Section
        st.markdown(f"#### **{primary_label}** (Primary)")
        c1a, c1b = st.columns(2)
        
        # Filter keys for Primary
        p_sensor_keys = [k for k in available_data.keys() if k.startswith(p_prefix) and ("Sensor:" in k or "Tilt:" in k)]
        p_power_keys = [k for k in available_data.keys() if k.startswith(p_prefix) and "Power:" in k]
        
        with c1a:
            sel_p_sens = st.multiselect(
                "Sensors",
                options=p_sensor_keys,
                format_func=lambda k: clean_label(k, p_prefix),
                key="p_multiselect_sensors"
            )
            selected_series.extend(sel_p_sens)
            
        with c1b:
            sel_p_pow = st.multiselect(
                "Power Analyzer",
                options=p_power_keys,
                format_func=lambda k: clean_label(k, p_prefix),
                key="p_multiselect_power"
            )
            selected_series.extend(sel_p_pow)
            
        # 2. Comparison DB Section
        st.markdown(f"#### **{comparison_label}** (Comparison)")
        c2a, c2b = st.columns(2)
        
        # Filter keys for Comparison
        c_sensor_keys = [k for k in available_data.keys() if k.startswith(c_prefix) and ("Sensor:" in k or "Tilt:" in k)]
        c_power_keys = [k for k in available_data.keys() if k.startswith(c_prefix) and "Power:" in k]
        
        with c2a:
            sel_c_sens = st.multiselect(
                "Sensors",
                options=c_sensor_keys,
                format_func=lambda k: clean_label(k, c_prefix),
                key="c_multiselect_sensors"
            )
            selected_series.extend(sel_c_sens)
            
        with c2b:
            sel_c_pow = st.multiselect(
                "Power Analyzer",
                options=c_power_keys,
                format_func=lambda k: clean_label(k, c_prefix),
                key="c_multiselect_power"
            )
            selected_series.extend(sel_c_pow)
            
    else:
        # --- Single Database View (Legacy) ---
        sensor_keys = [k for k in available_data.keys() if "Sensor:" in k or "Tilt:" in k]
        power_keys = [k for k in available_data.keys() if "Power:" in k]
        
        c1, c2 = st.columns(2)
        with c1:
            sel_sensors = st.multiselect(
                "Sensors",
                options=sensor_keys,
                format_func=lambda k: clean_label(k, ""),
                key="multiselect_sensors"
            )
            selected_series.extend(sel_sensors)
        with c2:
            sel_power = st.multiselect(
                "Power Analyzer",
                options=power_keys,
                format_func=lambda k: clean_label(k, ""),
                key="multiselect_power"
            )
            selected_series.extend(sel_power)

    # Analysis Toggles - moved below chart

    if not selected_series:
        st.markdown("""
        <div style="
            display: flex; 
            flex-direction: column;
            justify-content: center; 
            align_items: center; 
            text-align: center;
            width: 100%;
            height: 800px; 
            border: 2px dashed #e0e0e0; 
            border-radius: 10px; 
            color: #b0b0b0; 
            font-size: 2rem; 
            font-weight: 500;
            background-color: #fafafa;
            opacity: 0.7;
        ">
            Select tags to compare data
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Build the normalized comparison data
    fig = go.Figure()
    
    comparison_colors = [
        '#7EB8DA', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6', 
        '#34495E', '#16A085', '#D35400', '#7F8C8D', '#C0392B'
    ]
    
    legend_info = []
    
    for idx, series_name in enumerate(selected_series):
        table_name, col_name, df = available_data[series_name]
        
        if 'datetime' not in df.columns: continue
        
        # Filter by Date Range
        mask = (df['datetime'] >= date_range[0]) & (df['datetime'] <= date_range[1])
        df_filtered = df[mask].copy()
            
        if df_filtered.empty: continue
        
        x_data = df_filtered['datetime']
        y_data = df_filtered[col_name]
        
        original_values = y_data.copy()
        
        # Apply Smoothing if requested (otherwise keep raw)
        if not show_raw:
            # Automatic Smoothing
            window_size = max(10, len(y_data) // 50)
            y_data = y_data.rolling(window=window_size, min_periods=1, center=True).mean()
        
        if disable_norm:
             y_normalized = y_data
        else:
            # Normalize
            y_min = y_data.min()
            y_max = y_data.max()
            if y_max - y_min > 0:
                y_normalized = (y_data - y_min) / (y_max - y_min)
            else:
                y_normalized = y_data * 0
        
        color = comparison_colors[idx % len(comparison_colors)]

        # Calculate relative time (0 = start of the respective DB recording)
        dataset_start_time = None
        if comp_conn and series_name.startswith(c_prefix):
             dataset_start_time = comp_start_time
        else:
             dataset_start_time = primary_start_time
             
        # Fallback if something went wrong (shouldn't happen if data exists)
        if dataset_start_time is None: dataset_start_time = df['datetime'].min()

        x_relative = (x_data - dataset_start_time).dt.total_seconds()
        
        # Custom Hover text
        hover_text = [
            f"<b>{col_name}</b><br>Value: {val:.4f}<br>Time: {x:%d/%m/%Y - %H:%M:%S}"
            for x, val in zip(x_data, y_data)
        ]
        
        fig.add_trace(go.Scatter(
            x=x_relative,
            y=y_normalized,
            mode='lines',
            name=series_name, # Display full name in legend (including prefix)
            line=dict(color=color, width=1.5 if show_raw else 2.5),
            hovertext=hover_text,
            hoverinfo='text',
            customdata=original_values.values,
            opacity=0.8 if show_raw else 1.0
        ))
        
        legend_info.append({'name': series_name, 'color': color})
    
    # Dynamic layout settings
    y_title = "Value" if disable_norm else "Normalized Value (0-1)"
    chart_title = "Data Comparison" if disable_norm else "Normalized Data Comparison"
    y_axis_config = dict(tickformat='.2f') # Allow auto-range if not normalized
    if not disable_norm:
         y_axis_config['range'] = [-0.05, 1.05]

    fig.update_layout(
        title=chart_title,
        xaxis_title="Time (seconds relative to start)",
        yaxis_title=y_title,
        height=800,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=y_axis_config
    )
    
    st.plotly_chart(fig, width="stretch", key="comparison_chart")
    
    # Custom Legend below
    if legend_info:
        st.markdown("**Active Series:**")
        color_html = " ".join([
            f'<span style="display:inline-block; width:12px; height:12px; background-color:{item["color"]}; border-radius:50%; margin-right:4px;"></span>'
            f'<span style="margin-right:12px; font-size:0.9em; color:#555;">{item["name"]}</span>'
            for item in legend_info
        ])
        st.markdown(color_html, unsafe_allow_html=True)
    
    # Analysis Toggles below chart
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.toggle("Show Raw Data", key="compare_show_raw", help="Toggle to see raw data instead of smoothed mean.")
    with t_col2:
        st.toggle("Disable Normalization", key="compare_disable_norm", help="Toggle to show actual values instead of normalized 0-1 range.")



def plot_harmonics_data(conn: sqlite3.Connection, available_tables: list, show_quality: bool = True):
    """Create interactive bar charts for harmonics data (Voltage and Current harmonics)."""
    
    st.subheader("Harmonics Analysis")
    
    # Blue accent info box (consistent with app theme)
    st.markdown("""
    <div style="background-color: rgba(93, 173, 226, 0.2); border-left: 4px solid #5DADE2; padding: 12px 16px; border-radius: 4px; margin-bottom: 16px;">
        <strong style="color: #5DADE2;">Description:</strong> 55 harmonics of the 50Hz fundamental frequency (50Hz to 2750Hz).
    </div>
    """, unsafe_allow_html=True)
    
    # Separate voltage and current tables
    v_tables = [t for t in available_tables if t.startswith('V_')]
    i_tables = [t for t in available_tables if t.startswith('I_')]
    
    # Create subtabs for Voltage and Current
    harm_tab1, harm_tab2 = st.tabs(["Voltage Harmonics", "Current Harmonics"])
    
    def get_harmonics_df(table_name: str) -> pd.DataFrame:
        """Load harmonics data from a table."""
        df = get_table_data(conn, table_name)
        # Convert datetime if present
        if 'human_timestamp' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['human_timestamp'], format='%d/%m/%Y - %H:%M:%S')
            except:
                pass
        return df
    
    def get_harmonic_columns(df: pd.DataFrame, prefix: str) -> list:
        """Get harmonics columns in order (fundamental, 2nd, 3rd, ..., 55th)."""
        # Pattern: V_L1_f (fundamental), V_L1_2, V_L1_3, ..., V_L1_55
        # or: I_L1_f, I_L1_2, etc.
        harmonic_cols = []
        # First the fundamental
        f_col = f"{prefix}_f"
        if f_col in df.columns:
            harmonic_cols.append(f_col)
        # Then 2 through 55
        for i in range(2, 56):
            col = f"{prefix}_{i}"
            if col in df.columns:
                harmonic_cols.append(col)
        return harmonic_cols
    
    def plot_harmonics_bar(df: pd.DataFrame, table_name: str, prefix: str, unit: str, color: str, selected_idx: int):
        """Plot harmonics as bar chart similar to FFT."""
        harmonic_cols = get_harmonic_columns(df, prefix)
        
        if not harmonic_cols:
            st.warning(f"No harmonic columns found for {table_name}")
            return
        
        if selected_idx >= len(df):
            st.warning(f"Selected index out of range for {table_name}")
            return
        
        # Get selected row
        row = df.iloc[selected_idx]
        
        # Extract harmonic values
        harmonic_values = [row[col] if pd.notna(row[col]) else 0 for col in harmonic_cols]
        
        # Generate frequencies (50Hz = fundamental, then 100Hz, 150Hz, ... up to 2750Hz)
        frequencies = [50 * (i + 1) for i in range(len(harmonic_values))]  # 50, 100, 150, ..., 2750
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=frequencies,
            y=harmonic_values,
            name=table_name,
            marker_color=color,
            hovertemplate=f'<b>{table_name}</b><br>Freq: %{{x:.0f}} Hz<br>Amplitude: %{{y:.4f}} {unit}<extra></extra>'
        ))
        
        # Extract phase from table name (L1, L2, or L3)
        phase = table_name.split('_')[-1] if '_' in table_name else ''
        harm_type = "Voltage" if table_name.startswith('V_') else "Current"
        
        fig.update_layout(
            title=f'{harm_type} Harmonics - Phase {phase}',
            xaxis_title='Frequency (Hz)',
            yaxis_title=f'Amplitude ({unit})',
            height=450,
            xaxis=dict(
                tickmode='linear',
                dtick=250,  # Show tick every 250Hz
                range=[0, 2800]
            ),
            hovermode='x unified',
            bargap=0.1
        )
        
        st.plotly_chart(fig, width="stretch", key=f"harmonics_chart_{table_name}")
        
        # Show statistics
        with st.expander(f"Harmonics Statistics - {table_name}", expanded=False):
            # Calculate stats
            fundamental = harmonic_values[0] if harmonic_values else 0
            thd = 0
            if fundamental > 0 and len(harmonic_values) > 1:
                # THD = sqrt(sum of squares of harmonics) / fundamental * 100%
                harmonic_sum_sq = sum(v**2 for v in harmonic_values[1:])
                thd = (harmonic_sum_sq ** 0.5) / fundamental * 100
            
            # Find top 5 harmonics (excluding fundamental)
            if len(harmonic_values) > 1:
                indexed_harmonics = [(i+2, frequencies[i+1], harmonic_values[i+1]) for i in range(len(harmonic_values)-1)]
                top_harmonics = sorted(indexed_harmonics, key=lambda x: x[2], reverse=True)[:5]
            else:
                top_harmonics = []
            
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Fundamental (50Hz)", f"{fundamental:.4f} {unit}")
            with col_s2:
                st.metric("THD", f"{thd:.2f}%", help="Total Harmonic Distortion")
            with col_s3:
                st.metric("Total Harmonics", len(harmonic_values))
            
            if top_harmonics:
                st.markdown("**Top 5 Harmonics (excl. fundamental):**")
                for h_num, h_freq, h_val in top_harmonics:
                    st.caption(f"• {h_num}th harmonic ({h_freq}Hz): {h_val:.4f} {unit}")
    
    # Voltage Harmonics Tab
    with harm_tab1:
        if v_tables:
            # Load all dataframes first to get timestamps
            v_dataframes = {}
            all_timestamps = []
            
            for v_table in sorted(v_tables):
                df_v = get_harmonics_df(v_table)
                if not df_v.empty:
                    v_dataframes[v_table] = df_v
                    if not all_timestamps:  # Get timestamps from first table
                        for idx, row in df_v.iterrows():
                            ts = row.get('human_timestamp', f'Sample {idx}')
                            all_timestamps.append((idx, ts))
            
            if v_dataframes and all_timestamps:
                # Single selector for all voltage phases
                selected_idx = st.selectbox(
                    "Select Sample (All Voltage Phases)",
                    options=range(len(all_timestamps)),
                    format_func=lambda i: all_timestamps[i][1],
                    key="harmonic_voltage_selector"
                )
                
                st.markdown("---")
                
                # Create columns for each phase
                v_cols = st.columns(len(v_dataframes))
                
                # Plot data for all voltage tables with same index
                for i, (v_table, df_v) in enumerate(v_dataframes.items()):
                    with v_cols[i]:
                        # Extract prefix (e.g., V_L1 from V_harmonic_L1)
                        parts = v_table.split('_')  # ['V', 'harmonic', 'L1']
                        prefix = f"{parts[0]}_{parts[-1]}"  # V_L1
                        # Use different shades of blue for phases
                        colors = ['#5DADE2', '#3498DB', '#2E86C1']
                        color = colors[i % len(colors)]
                        plot_harmonics_bar(df_v, v_table, prefix, "V", color, selected_idx)
            else:
                st.warning("No voltage data found")
        else:
            st.warning("No voltage harmonics tables found.")
    
    # Current Harmonics Tab
    with harm_tab2:
        if i_tables:
            # Load all dataframes first to get timestamps
            i_dataframes = {}
            all_timestamps = []
            
            for i_table in sorted(i_tables):
                df_i = get_harmonics_df(i_table)
                if not df_i.empty:
                    i_dataframes[i_table] = df_i
                    if not all_timestamps:  # Get timestamps from first table
                        for idx, row in df_i.iterrows():
                            ts = row.get('human_timestamp', f'Sample {idx}')
                            all_timestamps.append((idx, ts))
            
            if i_dataframes and all_timestamps:
                # Single selector for all current phases
                selected_idx = st.selectbox(
                    "Select Sample (All Current Phases)",
                    options=range(len(all_timestamps)),
                    format_func=lambda i: all_timestamps[i][1],
                    key="harmonic_current_selector"
                )
                
                st.markdown("---")
                
                # Create columns for each phase
                i_cols = st.columns(len(i_dataframes))
                
                # Plot data for all current tables with same index
                for i, (i_table, df_i) in enumerate(i_dataframes.items()):
                    with i_cols[i]:
                        # Extract prefix (e.g., I_L1 from I_harmonic_L1)
                        parts = i_table.split('_')  # ['I', 'harmonic', 'L1']
                        prefix = f"{parts[0]}_{parts[-1]}"  # I_L1
                        # Use different shades of blue for current phases
                        colors = ['#5DADE2', '#3498DB', '#2E86C1']
                        color = colors[i % len(colors)]
                        plot_harmonics_bar(df_i, i_table, prefix, "A", color, selected_idx)
            else:
                st.warning("No current data found")
        else:
            st.warning("No current harmonics tables found.")


def plot_gps_data(df: pd.DataFrame):
    """Display GPS data on an interactive map."""
    if df.empty:
        st.warning("No GPS data available.")
        return
    
    # Check for valid coordinates
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        st.warning("GPS data does not contain latitude/longitude columns.")
        return
    
    # Filter out null coordinates
    df_valid = df.dropna(subset=['latitude', 'longitude'])
    df_valid = df_valid[(df_valid['latitude'] != 0) & (df_valid['longitude'] != 0)]
    
    if df_valid.empty:
        st.info("No valid GPS coordinates found in the data.")
        
        # Show raw data anyway
        st.subheader("Raw GPS Data")
        st.dataframe(df[['human_timestamp', 'latitude', 'longitude']])
        return
    
    # Sort by datetime to ensure correct path order
    if 'datetime' in df_valid.columns:
        df_valid = df_valid.sort_values('datetime')
    
    # Create map using graph_objects for more control over lines
    fig = go.Figure(go.Scattermap(
        lat=df_valid['latitude'],
        lon=df_valid['longitude'],
        mode='lines+markers',
        marker=dict(size=8, color=PASTEL_COLORS[1]),
        line=dict(width=2, color=PASTEL_COLORS[1]),
        hovertext=df_valid['human_timestamp'] if 'human_timestamp' in df_valid.columns else None,
        hoverinfo='text+lat+lon',
        name='Movement Path'
    ))
    
    # Calculate map center and dynamic zoom
    center_lat = df_valid['latitude'].mean()
    center_lon = df_valid['longitude'].mean()
    
    # Estimate zoom based on coordinate spread
    lat_span = df_valid['latitude'].max() - df_valid['latitude'].min()
    lon_span = df_valid['longitude'].max() - df_valid['longitude'].min()
    max_span = max(lat_span, lon_span)
    
    # Heuristic for zoom level
    if max_span < 0.005:
        zoom = 16
    elif max_span < 0.05:
        zoom = 13
    elif max_span < 0.5:
        zoom = 10
    else:
        zoom = 6
    
    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    
    st.plotly_chart(fig, key="gps_map", width="stretch")
    
    # Show coordinate table
    st.subheader("GPS Coordinates Table")
    display_cols = ['human_timestamp', 'latitude', 'longitude']
    available_cols = [col for col in display_cols if col in df_valid.columns]
    st.dataframe(df_valid[available_cols])


def display_combined_mqtt_simulation(df_sensor, cols_sensor, df_tilt, cols_tilt, x_axis_name, mqtt_interval: int = 1, mqtt_stats: 'MqttStats' = None):
    """
    Display MQTT simulation stats for combined sensor and tilt data.
    """
    st.markdown("---")
    st.subheader("MQTT Transmission Simulation (Sensors + Tilt)")
    
    # Needs at least one dataframe
    if (df_sensor is None or df_sensor.empty) and (df_tilt is None or df_tilt.empty):
        st.info("No data available for simulation.")
        return

    # Use Sensor DF for time calculation if available, otherwise Tilt
    main_df = df_sensor if (df_sensor is not None and not df_sensor.empty) else df_tilt
    main_axis = x_axis_name if x_axis_name else ('datetime' if 'datetime' in main_df.columns else main_df.columns[0])
    
    # 1. Frequency Slider & Duration
    time_min = main_df[main_axis].min()
    time_max = main_df[main_axis].max()
    duration_sec = 0.0
    
    if isinstance(time_min, pd.Timestamp):
        duration_sec = (time_max - time_min).total_seconds()
        
        # Formatting
        td = timedelta(seconds=duration_sec)
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        parts = []
        if days > 0: parts.append(f"{days} days")
        if hours > 0: parts.append(f"{hours} hours")
        if minutes > 0: parts.append(f"{minutes} minutes")
        parts.append(f"{seconds} seconds")
        duration_str = ", ".join(parts) if parts else "0 seconds"
        
        st.info(f"Selected Time Range Duration: **{duration_str}**")
    else:
        duration_sec = float(len(main_df))
        st.info(f"Selected Range: {len(main_df)} samples")
        
    sim_interval = mqtt_interval
    
    duration_sec = max(1.0, duration_sec)
    
    # 3. Construct all Payloads
    all_payloads = []
    
    # Create mapping for columns to incremental numbers
    all_cols = []
    if cols_sensor: all_cols.extend(cols_sensor)
    if cols_tilt: all_cols.extend(cols_tilt)
    
    # Remove duplicates while preserving order if any
    unique_cols = list(dict.fromkeys(all_cols))
    col_map = {col: str(i+1) for i, col in enumerate(unique_cols)}
    
    # We'll sample the dataframe based on the sim_interval
    if isinstance(time_min, pd.Timestamp):
        current_time = time_min
        while current_time <= time_max:
            # Find nearest row in sensor and tilt
            payload = {'ts': int(current_time.timestamp())}
            
            # Helper to find nearest row
            def get_nearest_row(df, target_time):
                if df is None or df.empty: return None
                # Assuming 'datetime' exists and is sorted
                idx = df['datetime'].searchsorted(target_time)
                if idx >= len(df): idx = len(df) - 1
                return df.iloc[idx]

            if df_sensor is not None and not df_sensor.empty and cols_sensor:
                row_s = get_nearest_row(df_sensor, current_time)
                if row_s is not None:
                    for col in cols_sensor:
                        val = row_s.get(col)
                        key = col_map[col]
                        if pd.isna(val) or val is None:
                            payload[key] = float('nan')
                        else:
                            try: payload[key] = round(float(val), 2)
                            except: payload[key] = str(val)

            if df_tilt is not None and not df_tilt.empty and cols_tilt:
                row_t = get_nearest_row(df_tilt, current_time)
                if row_t is not None:
                    for col in cols_tilt:
                        val = row_t.get(col)
                        key = col_map[col]
                        if pd.isna(val) or val is None:
                            payload[key] = float('nan')
                        else:
                            try: payload[key] = round(float(val), 2)
                            except: payload[key] = str(val)
            
            all_payloads.append(payload)
            current_time += timedelta(seconds=sim_interval)
    else:
        # Step-based sampling
        for i in range(0, len(main_df), max(1, int(sim_interval))):
            payload = {}
            row_main = main_df.iloc[i]
            payload['ts'] = int(row_main.get('unix_timestamp', i))
            
            # Simple assumption: index matches if time doesn't exist
            if df_sensor is not None and i < len(df_sensor) and cols_sensor:
                row_s = df_sensor.iloc[i]
                for col in cols_sensor:
                    val = row_s.get(col)
                    key = col_map[col]
                    if pd.isna(val) or val is None: payload[key] = float('nan')
                    else:
                        try: payload[key] = round(float(val), 2)
                        except: payload[key] = str(val)
            
            if df_tilt is not None and i < len(df_tilt) and cols_tilt:
                row_t = df_tilt.iloc[i]
                for col in cols_tilt:
                    val = row_t.get(col)
                    key = col_map[col]
                    if pd.isna(val) or val is None: payload[key] = float('nan')
                    else:
                        try: payload[key] = round(float(val), 2)
                        except: payload[key] = str(val)
            
            all_payloads.append(payload)

    if not all_payloads:
        st.info("No packets to display.")
        return

    # Use first packet for size calculation (average case)
    sample_json = json.dumps(all_payloads[0], separators=(',', ':'), allow_nan=True)
    
    # 4. Calculate Weight
    packet_size_bytes = len(sample_json)
    total_packets = len(all_payloads)
    total_size_bytes = sum(len(json.dumps(p, separators=(',', ':'), allow_nan=True)) for p in all_payloads)
    total_size_mb = total_size_bytes / (1024 * 1024)
    packets_4kb = math.ceil(total_size_bytes / 4096)
    
    if mqtt_stats:
        mqtt_stats.add("Sensors & Tilt", total_size_bytes, total_packets, duration_str)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
            st.metric("4KB Packets Needed", f"{packets_4kb:,}")
    with c2:
        st.metric("Total Packets (JSON)", f"{total_packets:,}")
    with c3:
            st.metric("Avg Packet Size", f"{packet_size_bytes} bytes")
    with c4:
            st.metric("Total Transmission Size", f"{total_size_mb:.2f} MB")
            
    with st.expander("View Json packet (first 10 rows)", expanded=False):
        # Join first 10 as separate lines for preview
        preview_json_sequence = "\n".join([json.dumps(p, separators=(',', ':'), allow_nan=True) for p in all_payloads[:10]])
        st.code(preview_json_sequence, language='json')


def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">Flowsense database analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar logo
    logo_path = Path(__file__).parent / "Data-Flow Logo PNG.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), width="stretch")

    # Sidebar for database selection
    st.sidebar.header("Database in folder")
    
    # Default database folder
    db_folder = DEFAULT_DATABASE_FOLDER
    
    # Show available databases
    available_dbs = get_database_files(db_folder)
    
    selected_db = None
    db_path = None
    folder_success_placeholder = None

    if available_dbs:
        # Add a None option to prevent auto-loading the first database
        db_options = ["(None)"] + available_dbs
        
        selected_db = st.sidebar.selectbox(
            "Select a database:",
            options=db_options,
            help="Select a database from the Database folder"
        )
        
        if selected_db and selected_db != "(None)":
            db_path = db_folder / selected_db
            folder_success_placeholder = st.sidebar.empty()
    else:
        st.sidebar.info("No databases found in the default folder.")
    
    # stats_placeholder removed - breakdown now renders directly in sidebar after toggles
    mqtt_stats = MqttStats() if "show_mqtt_calc" in st.session_state and st.session_state.show_mqtt_calc else None
    
    # File uploader for custom database
    uploaded_file = st.sidebar.file_uploader(
        "Upload .db file",
        type=['db'],
        help="Upload a SQLite database file"
    )
    upload_success_placeholder = st.sidebar.empty()
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("temp_uploaded.db")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        db_path = temp_path
    
    # --- Comparison Database Section ---
    # Initialize enable_comparison from session state (toggle defined later in Analysis Settings)
    enable_comparison = st.session_state.get("enable_db_comparison", False)
    
    comp_db_path = None
    comp_conn = None
    primary_db_name = ""
    comp_db_name = ""
    
    # Extract primary DB name
    if db_path:
        if uploaded_file is not None:
            primary_db_name = Path(uploaded_file.name).stem
        elif selected_db:
            primary_db_name = Path(selected_db).stem
    
    if enable_comparison:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Comparison Database")
        # Comparison database selection from folder
        comp_available_dbs = [db for db in available_dbs if db != selected_db] if available_dbs else []
        
        comp_selected_db = None
        if comp_available_dbs:
            comp_selected_db = st.sidebar.selectbox(
                "Select comparison database:",
                options=["(None)"] + comp_available_dbs,
                help="Select a database from the Database folder to compare",
                key="comp_db_selector"
            )
            if comp_selected_db and comp_selected_db != "(None)":
                comp_db_path = db_folder / comp_selected_db
                comp_db_name = Path(comp_selected_db).stem
        
        # Comparison database upload
        comp_uploaded_file = st.sidebar.file_uploader(
            "Or upload comparison .db file",
            type=['db'],
            help="Upload a SQLite database file for comparison",
            key="comp_file_uploader"
        )
        comp_upload_placeholder = st.sidebar.empty()
        
        if comp_uploaded_file is not None:
            # Save uploaded comparison file temporarily
            comp_temp_path = Path("temp_comparison.db")
            with open(comp_temp_path, 'wb') as f:
                f.write(comp_uploaded_file.getvalue())
            comp_db_path = comp_temp_path
            comp_db_name = Path(comp_uploaded_file.name).stem
        
        # Load comparison database
        if comp_db_path and Path(comp_db_path).exists():
            try:
                comp_conn = load_database(str(comp_db_path))
                if comp_uploaded_file is not None:
                    comp_upload_placeholder.success(f"Comparison: {comp_uploaded_file.name}")
                elif comp_selected_db and comp_selected_db != "(None)":
                    st.sidebar.success(f"Comparison: {comp_selected_db}")
            except Exception as e:
                st.sidebar.error(f"Failed to load comparison DB: {e}")
                comp_conn = None
    
    if db_path is None or not Path(db_path).exists():
        st.warning("Please select or upload a database to visualize.")
        return
    
    # Load database
    try:
        conn = load_database(str(db_path))
        if uploaded_file is not None:
            upload_success_placeholder.success(f"Connected: {uploaded_file.name}")
        elif available_dbs:
            folder_success_placeholder.success(f"Connected: {selected_db}")
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return
    
    # Initialize variables for usage in tabs before the sidebar toggles are defined at the end
    show_quality = st.session_state.get("show_quality_toggle", False)
    show_mqtt_calc = st.session_state.get("show_mqtt_calc_toggle", False)
    mqtt_interval = st.session_state.get("mqtt_interval_slider", 1)
    if show_mqtt_calc and mqtt_stats is None:
        mqtt_stats = MqttStats()

    # Create tabs for different data types
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Compare",
        "Sensors",
        "Power Analyzer",
        "FFT",
        "Harmonics",
        "GPS"
    ])
    
    # Tab 1: Compare
    with tab1:
        plot_comparison_data(
            conn,
            comp_conn=comp_conn if enable_comparison else None,
            primary_label=primary_db_name,
            comparison_label=comp_db_name
        )
    
    # Tab 2: Sensors
    with tab2:
        df_sensor_res = None
        cols_sensor_res = []
        x_axis_res = None
        
        df_tilt_res = None
        cols_tilt_res = []

        # Load both first to determine common range
        df_sensors_raw = get_table_data(conn, 'sensor_data') if check_table_exists(conn, 'sensor_data') else pd.DataFrame()
        df_tilt_raw = get_table_data(conn, 'tilt_data') if check_table_exists(conn, 'tilt_data') else pd.DataFrame()
        
        # Load comparison data if enabled
        df_sensors_comp = None
        df_tilt_comp = None
        if enable_comparison and comp_conn:
            if check_table_exists(comp_conn, 'sensor_data'):
                df_sensors_comp = get_table_data(comp_conn, 'sensor_data')
            if check_table_exists(comp_conn, 'tilt_data'):
                df_tilt_comp = get_table_data(comp_conn, 'tilt_data')

        if not df_sensors_raw.empty:
            # Check for empty sensors
            empty_sensors = get_empty_columns(df_sensors_raw, EXCLUDE_METADATA_COLS)
            if empty_sensors:
                st.warning(f"⚠️ **Disconnected Sensors (No Data):** {', '.join(empty_sensors)}")

            # Use sensors as primary for range
            df_sensors_filtered, x_axis_res = create_date_range_slider(df_sensors_raw, "sensor_unified")
            
            # Apply same filter to tilt if it exists
            if not df_tilt_raw.empty and 'datetime' in df_sensors_filtered.columns and 'datetime' in df_tilt_raw.columns:
                key = "sensor_unified_range"
                if key in st.session_state:
                    date_range = st.session_state[key]
                    mask_t = (df_tilt_raw['datetime'] >= date_range[0]) & (df_tilt_raw['datetime'] <= date_range[1])
                    df_tilt_filtered = df_tilt_raw[mask_t].copy()
                else:
                    df_tilt_filtered = df_tilt_raw.copy()
            else:
                df_tilt_filtered = df_tilt_raw.copy()

            # Plot Sensors with comparison overlay
            res = plot_sensor_data(
                df_sensors_filtered, x_axis_res, show_quality, False, mqtt_interval, mqtt_stats,
                df_comparison=df_sensors_comp,
                primary_label=primary_db_name if enable_comparison and comp_conn else "",
                comparison_label=comp_db_name if enable_comparison and comp_conn else ""
            )
            if res:
                df_sensor_res, cols_sensor_res, x_axis_res = res
            
            # Plot Tilt with comparison overlay
            if not df_tilt_filtered.empty:
                st.markdown("---") # Separator
                res_t = plot_tilt_data(
                    df_tilt_filtered, x_axis_res, show_quality, show_mqtt_calc, mqtt_interval, mqtt_stats,
                    df_comparison=df_tilt_comp,
                    primary_label=primary_db_name if enable_comparison and comp_conn else "",
                    comparison_label=comp_db_name if enable_comparison and comp_conn else ""
                )
                if res_t:
                    df_tilt_res, cols_tilt_res = res_t
            
            # Combined MQTT
            if show_mqtt_calc:
                 display_combined_mqtt_simulation(df_sensor_res, cols_sensor_res, df_tilt_res, cols_tilt_res, x_axis_res, mqtt_interval, mqtt_stats)
        else:
            if df_tilt_raw.empty:
                st.warning("No sensor or tilt data found.")
            else:
                # Fallback: only tilt exists
                df_tilt_filtered, x_axis_res = create_date_range_slider(df_tilt_raw, "tilt_only")
                plot_tilt_data(df_tilt_filtered, x_axis_res, show_quality, show_mqtt_calc, mqtt_interval, mqtt_stats)
    
    # Tab 3: Power Analyzer
    with tab3:
        if check_table_exists(conn, 'power_analyzer_data'):
            df_power = get_table_data(conn, 'power_analyzer_data')
            
            # Check for empty parameters
            empty_power_cols = get_empty_columns(df_power, EXCLUDE_METADATA_COLS)
            if empty_power_cols:
                st.warning(f"⚠️ **Empty Parameters (No Data):** {', '.join(empty_power_cols)}")
            
            # Load comparison power analyzer data if enabled
            df_power_comp = None
            if enable_comparison and comp_conn:
                if check_table_exists(comp_conn, 'power_analyzer_data'):
                    df_power_comp = get_table_data(comp_conn, 'power_analyzer_data')
            
            plot_power_analyzer_data(
                df_power, show_quality, show_mqtt_calc, mqtt_interval, mqtt_stats,
                df_comparison=df_power_comp,
                primary_label=primary_db_name if enable_comparison and comp_conn else "",
                comparison_label=comp_db_name if enable_comparison and comp_conn else ""
            )
        else:
            st.warning("Power analyzer data table not found in database.")
    
    # Tab 4: FFT
    with tab4:
        if check_table_exists(conn, 'fft_data'):
            df_fft = get_table_data(conn, 'fft_data')
            
            # Load comparison FFT data if enabled
            df_fft_comp = None
            if enable_comparison and comp_conn:
                if check_table_exists(comp_conn, 'fft_data'):
                    df_fft_comp = get_table_data(comp_conn, 'fft_data')
            
            plot_fft_data(
                df_fft, show_quality, show_mqtt_calc, mqtt_stats,
                df_comparison=df_fft_comp,
                primary_db_name=primary_db_name if enable_comparison and comp_conn else "",
                comparison_db_name=comp_db_name if enable_comparison and comp_conn else ""
            )
        else:
            st.warning("FFT data table not found in database.")
            
    # Tab 5: Harmonics
    with tab5:
        # Check for any harmonics tables
        harmonics_tables = ['V_harmonic_L1', 'V_harmonic_L2', 'V_harmonic_L3', 
                           'I_harmonic_L1', 'I_harmonic_L2', 'I_harmonic_L3']
        available_harmonics = [t for t in harmonics_tables if check_table_exists(conn, t)]
        
        if available_harmonics:
            plot_harmonics_data(conn, available_harmonics, show_quality)
        else:
            st.warning("No harmonics data tables found in database.")
    
    # Tab 6: GPS
    with tab6:
        if check_table_exists(conn, 'gps_data'):
            df_gps = get_table_data(conn, 'gps_data')
            plot_gps_data(df_gps)
        else:
            st.warning("GPS data table not found in database.")

    # Analysis Settings at the very bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Settings")
    show_quality = st.sidebar.toggle("Transmission Quality", value=show_quality, help="Highlight missing data and show success rate.", key="show_quality_toggle")
    show_mqtt_calc = st.sidebar.toggle("MQTT Packets", value=show_mqtt_calc, help="Calculate and show optimized MQTT JSON payload size.", key="show_mqtt_calc_toggle")
    enable_comparison = st.sidebar.toggle("DB Comparison", value=enable_comparison, help="Enable comparison with a second database.", key="enable_db_comparison")
    
    if show_mqtt_calc:
        mqtt_interval = st.sidebar.slider(
            "Sampling Interval (Seconds)",
            min_value=1,
            max_value=60,
            value=mqtt_interval,
            step=1,
            help="Simulate sending a packet every N seconds.",
            key="mqtt_interval_slider"
        )

    # Update sidebar stats if mqtt enabled - NOW APPEARS AFTER THE TOGGLE
    if show_mqtt_calc and mqtt_stats:
        total_bytes = sum(s['bytes'] for s in mqtt_stats.sources.values())
        total_pkts = mqtt_stats.get_total_4kb_packets()
        
        # Format total size
        if total_bytes < 1024:
            size_str = f"{total_bytes} B"
        elif total_bytes < 1024 * 1024:
            size_str = f"{total_bytes / 1024:.2f} KB"
        else:
            size_str = f"{total_bytes / (1024 * 1024):.2f} MB"
        
        st.sidebar.metric("Total Transmission", size_str)
        st.sidebar.metric("Total 4KB Packets", f"{total_pkts:,}")
        
        st.sidebar.markdown("### Contribution Breakdown")
        breakdown = mqtt_stats.get_breakdown()
        for source, data in breakdown.items():
            b_val = data['bytes']
            dur = data['duration']
            
            if b_val < 1024:
                s_str = f"{b_val} B"
            elif b_val < 1024 * 1024:
                s_str = f"{b_val / 1024:.2f} KB"
            else:
                s_str = f"{b_val / (1024 * 1024):.2f} MB"
                
            st.sidebar.markdown(f"**{source}**")
            if source == "FFT":
                st.sidebar.caption(f"Size: {s_str} | {dur}")
            else:
                st.sidebar.caption(f"Size: {s_str} | Time: {dur}")
            
            if source == "FFT":
                st.sidebar.slider(
                    "Percentile Threshold",
                    min_value=50,
                    max_value=99,
                    value=st.session_state.get("percentile_slider", 90),
                    step=1,
                    key="percentile_slider",
                    label_visibility="collapsed"
                )
    

    
    # Clean up temporary files if they exist
    temp_path = Path("temp_uploaded.db")
    if temp_path.exists() and uploaded_file is None:
        temp_path.unlink()
    
    comp_temp_path = Path("temp_comparison.db")
    if comp_temp_path.exists():
        # Only clean up if comparison is disabled or no file was uploaded
        comp_file_exists = 'comp_uploaded_file' in dir() and comp_uploaded_file is not None
        if not enable_comparison or not comp_file_exists:
            try:
                comp_temp_path.unlink()
            except:
                pass  # File might be in use


if __name__ == "__main__":
    main()

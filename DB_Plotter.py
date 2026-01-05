"""
DB_Plotter - SQLite Database Visualization Tool
A Streamlit application for visualizing flowsense sensor data.
"""

import streamlit as st
import sqlite3
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    }
    
    /* Generic selected tab style (fallback) */
    .stTabs [aria-selected="true"] {
        background-color: #7eb8da;
        color: white;
    }

    /* --- Main Tabs Specific Colors --- */
    
    /* Tab 1: Sensors - Pastel Blue */
    .stTabs [data-baseweb="tab-list"] button:nth-of-type(1)[aria-selected="true"] {
        background-color: #5DADE2 !important;
        color: white !important;
    }
    
    /* Tab 2: Power Analyzer - Vivid Orange */
    .stTabs [data-baseweb="tab-list"] button:nth-of-type(2)[aria-selected="true"] {
        background-color: #E67E22 !important;
        color: white !important;
    }
    
    /* Tab 3: Tilt Data - Pastel Red */
    .stTabs [data-baseweb="tab-list"] button:nth-of-type(3)[aria-selected="true"] {
        background-color: #E74C3C !important;
        color: white !important;
    }
    
    /* Tab 4: FFT - Pastel Yellow */
    .stTabs [data-baseweb="tab-list"] button:nth-of-type(4)[aria-selected="true"] {
        background-color: #F4D03F !important;
        color: black !important;
    }
    
    /* Tab 5: GPS - Pastel Green */
    .stTabs [data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"] {
        background-color: #58D68D !important;
        color: white !important;
    }

    /* --- Nested/Sub Tabs (FFT) --- */
    /* This target allows styling subtabs inside any parent tab container to match the parent if needed.
       Since we only have subtabs in FFT, we force them to match FFT's Yellow. */
    .stTabs .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #F4D03F !important;
        color: black !important;
    }

    /* Hide red indicator under tabs */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    /* Gray slider track */
    .stSlider > div > div > div > div {
        background: #808080 !important;
    }
</style>
""", unsafe_allow_html=True)

# Constants
DEFAULT_DATABASE_FOLDER = Path(__file__).parent / "Database"
EXCLUDE_METADATA_COLS = ['id_day', 'id', 'status', 'human_timestamp', 'unix_timestamp', 'datetime']

# Vivid pastel color palette (more saturated)
PASTEL_COLORS = [
    '#5DADE2',  # vivid blue
    '#E74C3C',  # vivid red
    '#58D68D',  # vivid green
    '#F4D03F',  # vivid yellow
    '#AF7AC5',  # vivid purple
    '#E67E22',  # vivid orange
    '#3498DB',  # strong blue
    '#EC7063',  # coral red
    '#45B39D',  # teal green
    '#F5B041',  # amber
    '#9B59B6',  # purple
    '#EB984E',  # orange
]

PASTEL_BLUE = '#5DADE2'
PASTEL_RED = '#E74C3C'
PASTEL_GREEN = '#58D68D'


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
        # Check for NaN or infinite values
        # Convert to numeric first to be safe (coercing errors to NaN)
        series = pd.to_numeric(df[column_name], errors='coerce')
        
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


def plot_sensor_data(df: pd.DataFrame, show_quality: bool = True, show_mqtt_calc: bool = True):

    """Create interactive time series plots for sensor data."""
    if df.empty:
        st.warning("No sensor data available.")
        return
    
    # Get sensor columns (excluding metadata columns)
    sensor_cols = [col for col in df.columns if col not in EXCLUDE_METADATA_COLS]
    
    if not sensor_cols:
        st.warning("No sensor columns found.")
        return
    
    # Date range slider
    df_filtered, x_axis = create_date_range_slider(df, "sensor")
    
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
        return

    # Create individual plots for each sensor
    for idx, col in enumerate(valid_cols):
        line_color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]
        
        # Create separate figure
        fig = go.Figure()
        
        # Trace 1: Raw Data
        fig.add_trace(go.Scatter(
            x=df_filtered[x_axis],
            y=df_filtered[col],
            mode='lines',
            name=col,
            line=dict(color=line_color, width=1),
            opacity=0.7,
            hovertemplate=f'{col}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
        ))
        
        # Calculate Moving Average
        window_size = max(10, len(df_filtered) // 50)
        col_avg = df_filtered[col].rolling(window=window_size, center=True).mean()
        
        # Trace 2: Average
        fig.add_trace(go.Scatter(
            x=df_filtered[x_axis],
            y=col_avg,
            mode='lines',
            name=f'{col} Avg',
            line=dict(color=line_color, width=2.5),
            hovertemplate=f'{col} Avg: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
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
            
            st.markdown("---") # Separator between sensors
        else:
            st.plotly_chart(fig, width="stretch", key=f"sensor_plot_{idx}")

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
             # Fallback if x_axis is not datetime
            duration_sec = float(len(df_filtered))
            st.info(f"Selected Range: {len(df_filtered)} samples")
            
        sim_interval = st.slider(
            "Simulated Sampling Interval (Seconds)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Simulate sending a packet every N seconds.",
            key="sensor_interval_slider"
        )
        
        # Ensure duration is at least 1s to avoid divide by zero or weirdness
        duration_sec = max(1.0, duration_sec)
        
        # 3. Construct Payload
        # We need "feature presenti nei grafici" -> valid_cols
        # Create a sample payload from the FIRST valid row (to simulate size)
        # OR use a row with NaNs if we want to test that. 
        # Let's take the last row as a sample.
        sample_row = df_filtered.iloc[-1]
        
        payload = {}
        # Add timestamp
        if 'unix_timestamp' in sample_row:
             payload['ts'] = int(sample_row['unix_timestamp'])
        else:
             import time
             payload['ts'] = int(time.time())

        for col in valid_cols:
            val = sample_row[col]
            # Handle NaN
            if pd.isna(val) or val is None:
                payload[col] = float('nan')
            else:
                try:
                    payload[col] = round(float(val), 2)
                except:
                    payload[col] = str(val)
        
        # Optimized JSON (no whitespace)
        # allow_nan=True is default, but explicit for clarity
        json_str = json.dumps(payload, separators=(',', ':'), allow_nan=True)
        
        # 4. Calculate Weight
        packet_size_bytes = len(json_str)
        # Total packets = Duration / Interval
        total_packets = int(duration_sec / sim_interval)
        total_size_bytes = total_packets * packet_size_bytes
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # Calculate 4KB packets
        packets_4kb = math.ceil(total_size_bytes / 4096)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
             st.metric("4KB Packets Needed", f"{packets_4kb:,}", help="Number of 4KB blocks required to transmit total data.")
        with c2:
            st.metric("Total Packets (JSON)", f"{total_packets:,}")
        with c3:
             st.metric("Packet Size", f"{packet_size_bytes} bytes")
        with c4:
             st.metric("Total Transmission Size", f"{total_size_mb:.2f} MB")
             
        with st.expander("View Sample JSON Packet", expanded=False):
            st.code(json_str, language='json')




def plot_power_analyzer_data(df: pd.DataFrame, show_quality: bool = True, show_mqtt_calc: bool = True):

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
    # Plot grouped metrics
    groups = [
        ("Current Measurements", current_cols),
        ("Voltage and Frequency", voltage_cols),
        ("Power System (kW/kVA/kVAR)", power_system_cols),
        ("Total Harmonic Distortion (THD)", thd_cols),
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
    
    # Add transmission quality visualization
    if show_quality:
        # Use first col of group as proxy for 'local' checks if we want, or just global. 
        # For grouped plots, specific local faults might be messy. Let's stick to global gaps primarily, 
        # and maybe check the FIRST column of the group for local faults as a representative?
        # A safer bet is just Global Gaps for the big plots, but the user asked for "valid value" checks.
        # Let's check ALL columns in the group. If ANY is invalid, mark it?
        # Simple approach: Check the primary column (first one) for local quality.
        
        # Actually, grouped plots display multiple lines. Let's just track global gaps for the whole group (missing rows)
        # And if specific columns have NaNs, we can't easily show 5 different stats rows.
        # So for Power Analyzer (which is grouped), we will only show Global Gaps and a generic success rate.
        # OR we can iterate per plot definition like sensors. 
        # Given "apply similar logic", let's Refactor Power Analyzer to be Iterate-based too!
        
        # Wait, Power Analyzer has Grouped plots (Currents together, Voltages together).
        # We can keep them grouped but show stats below each GROUP.
        pass

    # REFACTORING Power Analyzer Loop to render chart + stats individually per GROUP
    
    for idx, definition in enumerate(plot_definitions):
        col = definition['col'] # This was for single-col loop, but wait...
        # The previous code logic was: plot_definitions had one entry per COLUMN, but they were all added to ONE make_subplots figure.
        # We want to break that figure apart.
        
        # Re-creating the figure for just THIS column (or group?)
        # The 'groups' logic in previous code was just to select columns. 
        # But 'plot_definitions' flattened it into a list of columns to plot in order.
        # So essentially it WAS plotting one column per subplot row.
        # So we can just make individual charts! Perfect.
        
        row_num = idx + 1
        
         # Determine unit label
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
            
        line_color = PASTEL_COLORS[idx % len(PASTEL_COLORS)]
        
        # New Figure
        fig = go.Figure()

        # Trace 1: Raw
        fig.add_trace(go.Scatter(
            x=df_filtered[x_axis],
            y=df_filtered[col],
            mode='lines',
            name=col,
            line=dict(color=line_color, width=1),
            opacity=0.7,
            hovertemplate=f'{col}: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
        ))
        
        # Calculate Moving Average
        window_size = max(10, len(df_filtered) // 50)
        col_avg = df_filtered[col].rolling(window=window_size, center=True).mean()
        
        # Trace 2: Average
        fig.add_trace(go.Scatter(
            x=df_filtered[x_axis],
            y=col_avg,
            mode='lines',
            name=f'{col} Avg',
            line=dict(color=line_color, width=2.5),
            hovertemplate=f'{col} Avg: %{{y}}<br>{x_axis}: %{{x}}<extra></extra>'
        ))
        
        group_title = definition['group']
        # Maybe use Group + Col name
        
        fig.update_layout(
             title=f"{col} ({group_title})",
             height=300,
             margin=dict(l=20, r=20, t=40, b=20),
             showlegend=True,
             hovermode='x unified',
             yaxis_title=y_label
        )
        
         # Transmission Quality
        if show_quality:
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
            
            st.markdown("---")
        else:
             st.plotly_chart(fig, width="stretch", key=f"power_plot_{idx}")

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
            
        sim_interval_power = st.slider(
            "Simulated Sampling Interval (Seconds)",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Simulate sending a packet every N seconds.",
            key="power_interval_slider"
        )
        
        # Ensure duration is at least 1s
        
        duration_sec = max(1.0, duration_sec)
        
        # 3. Construct Payload
        # Features visible in graphs: plot_definitions contains all 'col' keys
        # We need to construct a unique set of columns
        visible_cols = set(d['col'] for d in plot_definitions)
        
        sample_row = df_filtered.iloc[-1]
        
        payload = {}
        if 'unix_timestamp' in sample_row:
             payload['ts'] = int(sample_row['unix_timestamp'])
        else:
             import time
             payload['ts'] = int(time.time()) # Mock if missing
             
        for col in visible_cols:
            if col in sample_row:
                val = sample_row[col]
                if pd.isna(val) or val is None:
                    payload[col] = float('nan')
                else:
                    try:
                        payload[col] = round(float(val), 2)
                    except:
                        payload[col] = str(val)
                        
        json_str = json.dumps(payload, separators=(',', ':'), allow_nan=True)
        
        # 4. Calculate Weight
        packet_size_bytes = len(json_str)
        # Total packets = Duration / Interval
        total_packets = int(duration_sec / sim_interval_power)
        total_size_bytes = total_packets * packet_size_bytes
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # Calculate 4KB packets
        packets_4kb = math.ceil(total_size_bytes / 4096)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
             st.metric("4KB Packets Needed", f"{packets_4kb:,}", help="Number of 4KB blocks required to transmit total data.")
        with c2:
            st.metric("Total Packets (JSON)", f"{total_packets:,}")
        with c3:
             st.metric("Packet Size", f"{packet_size_bytes} bytes")
        with c4:
             st.metric("Total Transmission Size", f"{total_size_mb:.2f} MB")
             
        with st.expander("View Sample JSON Packet", expanded=False):
            st.code(json_str, language='json')




def plot_tilt_data(df: pd.DataFrame, show_quality: bool = True, show_mqtt_calc: bool = True):

    """Create interactive time series plot for tilt data."""
    if df.empty:
        st.warning("No tilt data available.")
        return
    
    # Date range slider
    df_filtered, x_axis = create_date_range_slider(df, "tilt")
    
    if 'tilt_angle' not in df_filtered.columns:
        return

    tilt_color = PASTEL_RED  # vivid red
    
    # Calculate moving average
    window_size = max(10, len(df_filtered) // 50)
    df_filtered['tilt_angle_avg'] = df_filtered['tilt_angle'].rolling(window=window_size, center=True).mean()
    
    fig = go.Figure()
    
    # Trace 1: Raw Data
    fig.add_trace(go.Scatter(
        x=df_filtered[x_axis],
        y=df_filtered['tilt_angle'],
        mode='lines',
        name='Tilt Angle',
        line=dict(color=tilt_color, width=1),
        opacity=0.5,
        hovertemplate='Tilt Angle: %{y:.2f} deg<br>Time: %{x}<extra></extra>'
    ))
    
    # Trace 2: Average
    fig.add_trace(go.Scatter(
        x=df_filtered[x_axis],
        y=df_filtered['tilt_angle_avg'],
        mode='lines',
        name='Tilt Angle Avg',
        line=dict(color=tilt_color, width=3),
        hovertemplate='Tilt Angle Avg: %{y:.2f} deg<br>Time: %{x}<extra></extra>'
    ))

    fig.update_layout(
        title="Tilt Angle Analysis",
        xaxis_title="Time" if x_axis == 'datetime' else "Sample ID",
        yaxis_title="Tilt Angle (deg)",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
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
                
            sim_interval_tilt = st.slider(
                "Simulated Sampling Interval (Seconds)",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                help="Simulate sending a packet every N seconds.",
                key="tilt_interval_slider"
            )
            
            # Ensure duration is at least 1s
            duration_sec = max(1.0, duration_sec)
            
            # 3. Construct Payload
            # For Tilt, we usually have 'tilt_angle'. Let's check other columns too.
            # We'll use whatever columns are remaining after excluding metadata.
            tilt_cols = [col for col in df.columns if col not in EXCLUDE_METADATA_COLS]
            valid_cols = [c for c in tilt_cols if c in df_filtered.columns]

            sample_row = df_filtered.iloc[-1]
            
            payload = {}
            if 'unix_timestamp' in sample_row:
                 payload['ts'] = int(sample_row['unix_timestamp'])
            else:
                 import time
                 payload['ts'] = int(time.time()) # Mock if missing
                 
            for col in valid_cols:
                val = sample_row[col]
                if pd.isna(val) or val is None:
                    payload[col] = float('nan')
                else:
                    try:
                        payload[col] = round(float(val), 2)
                    except:
                        payload[col] = str(val)
                            
            json_str = json.dumps(payload, separators=(',', ':'), allow_nan=True)
            
            # 4. Calculate Weight
            packet_size_bytes = len(json_str)
            # Total packets = Duration / Interval
            total_packets = int(duration_sec / sim_interval_tilt)
            total_size_bytes = total_packets * packet_size_bytes
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            # Calculate 4KB packets
            packets_4kb = math.ceil(total_size_bytes / 4096)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                 st.metric("4KB Packets Needed", f"{packets_4kb:,}", help="Number of 4KB blocks required to transmit total data.")
            with c2:
                st.metric("Total Packets (JSON)", f"{total_packets:,}")
            with c3:
                 st.metric("Packet Size", f"{packet_size_bytes} bytes")
            with c4:
                 st.metric("Total Transmission Size", f"{total_size_mb:.2f} MB")
                 
            with st.expander("View Sample JSON Packet", expanded=False):
                st.code(json_str, language='json')




def plot_fft_data(df: pd.DataFrame, show_mqtt_calc: bool = True):
    """Create interactive bar charts for FFT data."""
    if df.empty:
        st.warning("No FFT data available.")
        return
    
    # Get FFT columns (p_0 to p_999)
    fft_cols = [col for col in df.columns if col.startswith('p_')]
    
    if not fft_cols:
        st.warning("No FFT columns found.")
        return
    
    # Create subtabs
    fft_tab1, fft_tab2, fft_tab3 = st.tabs(["FFT", "FFT in Time", "Advanced Analysis"])
    
    with fft_tab1:
        # Build dropdown options with metadata
        # Format: axis, amplitude (G), type, Number of points, interval of analysis
        dropdown_options = []
        for idx, row in df.iterrows():
            axis = row.get('axis', 'N/A')  # X, Y, or Z
            max_amplitude = row.get('max_amplitude_g', 'N/A')  # Amplitude in G
            fft_type = row.get('type', 'N/A')  # acceleration or velocity
            num_points = row.get('number_of_points', len(fft_cols))
            interval = row.get('human_interval_of_analysis', 'N/A')
            
            # Format amplitude
            amplitude_str = f"{max_amplitude} G"
            
            label = f"{axis} | {amplitude_str} | {fft_type} | {num_points} Hz | {interval}"
            dropdown_options.append((idx, label))
        
        # Percentile selector
        percentile_value = st.slider(
            "Percentile Threshold for Peak Detection",
            min_value=50,
            max_value=99,
            value=90,
            step=1,
            key="percentile_slider"
        )
        
        # Helper function to plot a single FFT
        def plot_single_fft(selected_idx: int, key_suffix: str, title_prefix: str):
            row = df.iloc[selected_idx]
            
            # Extract FFT values
            fft_values = [row[col] for col in fft_cols if pd.notna(row[col])]
            fft_indices = list(range(len(fft_values)))
            
            # Calculate percentile threshold
            percentile_threshold = np.percentile(fft_values, percentile_value)
            
            # Assign colors: vivid blue for normal, vivid red for above percentile
            colors = [
                PASTEL_RED if val > percentile_threshold else PASTEL_BLUE
                for val in fft_values
            ]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=fft_indices,
                    y=fft_values,
                    marker_color=colors,
                    hovertemplate='Frequency: %{x} Hz<br>Magnitude: %{y:.4f}<extra></extra>'
                )
            ])
            
            # Add horizontal line for percentile
            fig.add_hline(
                y=percentile_threshold,
                line_dash="dash",
                line_color="#e74c3c",
                annotation_text=f"{percentile_value}th percentile: {percentile_threshold:.4f}",
                annotation_position="top right"
            )
            
            fft_type = row.get('type', 'acceleration')
            amplitude_unit = 'G' if fft_type == 'acceleration' else 'mm/s'
            
            fig.update_layout(
                title=f"{title_prefix}FFT Spectrum - Sample {selected_idx + 1}",
                xaxis_title="Frequency (Hz)",
                yaxis_title=f"Amplitude ({amplitude_unit})",
                height=450,
                margin=dict(l=50, r=50, t=50, b=50),
                bargap=0
            )
            
            st.plotly_chart(fig, key=f"fft_plot_{key_suffix}")
            
            # Show statistics
            peaks_above = sum(1 for val in fft_values if val > percentile_threshold)
            
            # Ground average: mean of values below or equal to the percentile threshold
            ground_values = [val for val in fft_values if val <= percentile_threshold]
            ground_average = sum(ground_values) / len(ground_values) if ground_values else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Max Amplitude", f"{max(fft_values):.4f}")
            with col2:
                st.metric("Min Amplitude", f"{min(fft_values):.4f}")
            with col3:
                st.metric("Mean Amplitude", f"{sum(fft_values)/len(fft_values):.4f}")
            with col4:
                st.metric("Ground Average", f"{ground_average:.4f}")
            with col5:
                st.metric(f"Peaks > {percentile_value}th", peaks_above)
                
            if show_mqtt_calc:
                # Construct optimized payload
                # Format: {"ts": <unix_ts or 0>, "avg": <val>, "peaks": [[idx, val], ...]}
                
                # Get timestamps from row if available, else 0
                ts_val = 0
                if 'unix_start' in row and pd.notna(row['unix_start']):
                     ts_val = int(row['unix_start'])

                # Find peaks (val > percentile, returning [freq_index, amplitude])
                peaks_list = []
                for i, val in enumerate(fft_values):
                    if val > percentile_threshold:
                        peaks_list.append([i, round(val, 2)])
                
                payload = {
                    "type": "acc" if row.get('type') == 'acceleration' else ("vel" if row.get('type') == 'velocity' else row.get('type', 'N/A')),
                    "points": int(row.get('number_of_points', len(fft_cols))),
                    "axis": row.get('axis', 'N/A'),
                    "ts": ts_val,
                    "avg": round(ground_average, 2),
                    "peaks": peaks_list
                }
                
                json_str = json.dumps(payload, separators=(',', ':'))
                payload_size = len(json_str)
                
                st.markdown("---")
                st.subheader("MQTT Analysis")
                c1, c2 = st.columns([1, 3])
                with c1:
                    st.metric("Est. Payload Size", f"{payload_size} bytes")
                    st.caption(f"Peaks sent: {len(peaks_list)}")
                with c2:
                    with st.expander("View JSON Payload", expanded=False):
                        st.text("Raw Compact JSON (One Line):")
                        st.code(json_str, language='json', line_numbers=False)
        
        # First FFT viewer
        st.subheader("Primary FFT")
        selected_idx_1 = st.selectbox(
            "Select FFT Sample",
            options=range(len(dropdown_options)),
            format_func=lambda x: dropdown_options[x][1],
            key="fft_selector_1"
        )
        plot_single_fft(selected_idx_1, "1", "")
        
        # Separator
        st.markdown("---")
        
        # Second FFT viewer for comparison
        st.subheader("Comparison FFT")
        selected_idx_2 = st.selectbox(
            "Select FFT Sample for Comparison",
            options=range(len(dropdown_options)),
            format_func=lambda x: dropdown_options[x][1],
            key="fft_selector_2",
            index=min(1, len(dropdown_options) - 1)  # Default to second sample if available
        )
        plot_single_fft(selected_idx_2, "2", "Comparison: ")
    
    # Common filters for subtabs 2 and 3
    # Get available axes and types
    available_axes = df['axis'].dropna().unique().tolist() if 'axis' in df.columns else ['X', 'Y', 'Z']
    available_types = df['type'].dropna().unique().tolist() if 'type' in df.columns else ['acceleration', 'velocity']

    with fft_tab2:
        st.subheader("FFT Heatmap Over Time")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_axis_hm = st.selectbox("Select Axis", options=available_axes, key="heatmap_axis")
        with col2:
            selected_type_hm = st.selectbox("Select Type", options=available_types, key="heatmap_type")
        
        # Filter dataframe
        df_hm = df.copy()
        if 'axis' in df_hm.columns:
            df_hm = df_hm[df_hm['axis'] == selected_axis_hm]
        if 'type' in df_hm.columns:
            df_hm = df_hm[df_hm['type'] == selected_type_hm]
            
        # Slider for number of samples
        if not df_hm.empty:
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
            for idx, row in df_hm.iterrows():
                fft_vals = [row[col] if pd.notna(row[col]) else 0 for col in fft_cols]
                heatmap_data.append(fft_vals)
                interval = row.get('human_interval_of_analysis', f'Sample {idx}')
                y_labels.append(str(interval))
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data,
                x=list(range(len(fft_cols))),
                y=y_labels,
                colorscale='Viridis',
                colorbar=dict(title='Amplitude'),
                hovertemplate='Frequency: %{x} Hz<br>Time: %{y}<br>Amplitude: %{z:.4f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"FFT Heatmap - Axis: {selected_axis_hm}, Type: {selected_type_hm}",
                xaxis_title="Frequency (Hz)",
                yaxis_title="Time Interval",
                height=max(400, len(heatmap_data) * 25),
                margin=dict(l=50, r=50, t=50, b=50)
            )
            st.plotly_chart(fig, key="fft_heatmap", width="stretch")
            
            st.markdown("---")
            st.subheader("3D Surface Evolution")
            
            # Prepare data for 3D plot (using same data as heatmap)
            # Limit samples for 3D performance if too many
            MAX_3D_SAMPLES = 60
            if len(heatmap_data) > MAX_3D_SAMPLES:
                st.info(f"Displaying last {MAX_3D_SAMPLES} samples for 3D performance.")
                z_3d = heatmap_data[-MAX_3D_SAMPLES:]
                y_3d = y_labels[-MAX_3D_SAMPLES:]
            else:
                z_3d = heatmap_data
                y_3d = y_labels
                
            fig_3d = go.Figure(data=[go.Surface(
                z=z_3d,
                x=list(range(len(fft_cols))),
                y=y_3d,
                colorscale='Viridis'
            )])
            
            fig_3d.update_layout(
                title=f'3D FFT Evolution - Axis: {selected_axis_hm}, Type: {selected_type_hm}',
                scene = dict(
                    xaxis_title='Frequency (Hz)',
                    yaxis_title='Time',
                    zaxis_title='Amplitude'
                ),
                height=600,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_3d, width="stretch", key="3d_fft_tab2")

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
            for idx, row in df_adv.iterrows():
                vals = [row[col] if pd.notna(row[col]) else 0 for col in fft_cols]
                spectra.append(vals)
                ts = row.get('human_interval_of_analysis', f'Sample {idx}')
                timestamps.append(str(ts))
            
            freqs = list(range(len(fft_cols)))
            
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
                        'Frequency': freqs[p_idx],
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
                low_band_max = st.number_input("Low Band Max (Hz)", value=200, step=10)
            with col_b2:
                med_band_max = st.number_input("Medium Band Max (Hz)", value=1000, step=50)
                
            # Calculate energies
            energies = []
            for spec in spectra:
                low_energy = sum(spec[:min(low_band_max, len(spec))])
                med_energy = sum(spec[min(low_band_max, len(spec)):min(med_band_max, len(spec))])
                high_energy = sum(spec[min(med_band_max, len(spec)):])
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


def main():
    """Main application function."""
    st.markdown('<h1 class="main-header">Flowsense database plotter</h1>', unsafe_allow_html=True)
    
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
    
    if available_dbs:
        selected_db = st.sidebar.selectbox(
            "Select a database:",
            options=available_dbs,
            help="Select a database from the Database folder"
        )
        db_path = db_folder / selected_db
    else:
        st.sidebar.info("No databases found in the default folder.")
        selected_db = None
        db_path = None
    
    # Transmission Quality Toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Settings")
    show_quality = st.sidebar.toggle("Show Transmission Quality", value=True, help="Highlight missing data and show success rate.")
    show_mqtt_calc = st.sidebar.toggle("Show MQTT Optimization", value=True, help="Calculate and show optimized MQTT JSON payload size.")
    
    # File uploader for custom database

    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload a Database")
    uploaded_file = st.sidebar.file_uploader(
        "Upload .db file",
        type=['db'],
        help="Upload a SQLite database file"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("temp_uploaded.db")
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        db_path = temp_path
        st.sidebar.success(f"Loaded: {uploaded_file.name}")
    
    if db_path is None or not Path(db_path).exists():
        st.warning("Please select or upload a database to visualize.")
        return
    
    # Load database
    try:
        conn = load_database(str(db_path))
        st.sidebar.success("Connected to database")
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        return
    
    # Create tabs for different data types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Sensors",
        "Power Analyzer",
        "Tilt Data",
        "FFT",
        "GPS"
    ])
    
    with tab1:
        if check_table_exists(conn, 'sensor_data'):
            df_sensors = get_table_data(conn, 'sensor_data')
            plot_sensor_data(df_sensors, show_quality, show_mqtt_calc)
        else:
            st.warning("Sensor data table not found in database.")
    
    with tab2:
        if check_table_exists(conn, 'power_analyzer_data'):
            df_power = get_table_data(conn, 'power_analyzer_data')
            plot_power_analyzer_data(df_power, show_quality, show_mqtt_calc)
        else:
            st.warning("Power analyzer data table not found in database.")
    
    with tab3:
        if check_table_exists(conn, 'tilt_data'):
            df_tilt = get_table_data(conn, 'tilt_data')
            plot_tilt_data(df_tilt, show_quality, show_mqtt_calc)
        else:
            st.warning("Tilt data table not found in database.")

    
    with tab4:
        if check_table_exists(conn, 'fft_data'):
            df_fft = get_table_data(conn, 'fft_data')
            plot_fft_data(df_fft, show_mqtt_calc)
        else:
            st.warning("FFT data table not found in database.")
    
    with tab5:
        if check_table_exists(conn, 'gps_data'):
            df_gps = get_table_data(conn, 'gps_data')
            plot_gps_data(df_gps)
        else:
            st.warning("GPS data table not found in database.")
    
    # Close connection
    conn.close()
    
    # Clean up temporary file if it exists
    temp_path = Path("temp_uploaded.db")
    if temp_path.exists() and uploaded_file is None:
        temp_path.unlink()


if __name__ == "__main__":
    main()

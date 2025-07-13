import marimo

__generated_with = "0.14.10"
app = marimo.App()

with app.setup:
    import marimo as mo
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import seaborn as sns
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta


@app.cell
def load_wpilog_data():
    """Load and parse the FRC robot log data from wpilog file"""

    import pathlib
    from wpiutil.log import DataLogReader, DataLogRecord

    # Use the CSV filename pattern to derive the wpilog filename
    log_file = pathlib.Path.home() / 'Documents/logs/FRC_20250629_041238_NSMAC_E2.wpilog'

    def read_wpilog_to_dataframe(filename):
        """Read wpilog file and convert to pandas DataFrame"""
        data_records = []

        reader = DataLogReader(str(filename))

        if not reader.isValid():
            raise ValueError(f"Invalid log file: {filename}")

        entries = {}  # Maps entry ID to metadata

        for record in reader:
            if record.isStart():
                # Store entry metadata
                start_data = record.getStartData()
                entries[start_data.entry] = start_data

            elif not record.isControl():
                # Process data record
                entry_info = entries.get(record.getEntry(), None)
                if entry_info is not None:
                    timestamp = record.getTimestamp() / 1000000.0  # Convert to seconds

                    # Extract value based on type
                    try:
                        if entry_info.type == "double":
                            value = str(record.getDouble())
                        elif entry_info.type == "float":
                            value = str(record.getFloat())
                        elif entry_info.type == "int64":
                            value = str(record.getInteger())
                        elif entry_info.type == "boolean":
                            value = str(record.getBoolean())
                        elif entry_info.type == "string":
                            value = record.getString()
                        elif entry_info.type == "json":
                            value = record.getString()
                        else:
                            # For arrays or unknown types, convert to string
                            value = str(record.getRaw())

                        data_records.append({
                            'Timestamp': timestamp,
                            'Key': entry_info.name,
                            'Value': value
                        })
                    except Exception as e:
                        # Skip corrupted records
                        continue

        return pd.DataFrame(data_records)

    df = read_wpilog_to_dataframe(log_file)

    # Convert Key column to category for memory efficiency
    df['Key'] = df['Key'].astype('category')

    mo.md(f"""
    ## Data Overview

    **Log file:** `{log_file}`  
    **Total records:** {len(df):,}  
    **Columns:** {', '.join(df.columns)}  
    **Time range:** {df['Timestamp'].min():.2f}s to {df['Timestamp'].max():.2f}s  
    **Duration:** {(df['Timestamp'].max() - df['Timestamp'].min()):.2f}s  
    **Unique keys:** {df['Key'].nunique()}
    """)

    return (df,)


@app.cell
def extract_states(df):
    """Extract intake and shooting state data"""

    # Filter for relevant state keys
    intake_states = df[
        (df['Key'].isin([
            'NT:/components/reef_intake/state/current_state',
            'NT:/components/floor_intake/state/current_state'
        ])) & 
        (df['Value'] == 'intaking')
    ].copy()

    shooting_states = df[
        (df['Key'] == 'NT:/components/algae_shooter/state/current_state') & 
        (df['Value'] == 'shooting')
    ].copy()

    # Add component type for intake states
    intake_states['component'] = intake_states['Key'].apply(
        lambda x: 'reef' if 'reef_intake' in x else 'floor'
    )

    # Create events dataframe
    events = pd.concat([
        intake_states[['Timestamp', 'component']].assign(event_type='intake'),
        shooting_states[['Timestamp']].assign(event_type='shooting', component='algae_shooter')
    ]).sort_values('Timestamp').reset_index(drop=True)

    mo.md(f"""
    ## State Events Summary

    **Intake events:** {len(intake_states)} (Reef: {len(intake_states[intake_states['component'] == 'reef'])}, Floor: {len(intake_states[intake_states['component'] == 'floor'])})  
    **Shooting events:** {len(shooting_states)}  
    **Total events:** {len(events)}
    """)

    return (events,)


@app.cell
def calculate_cycles(events):
    """Calculate cycle times between intake and shooting events"""

    cycles = []
    last_intake_time = None
    last_shooting_time = None

    for _, event in events.iterrows():
        if event['event_type'] == 'intake':
            if last_shooting_time is not None:
                # Time from last shooting to this intake
                cycles.append({
                    'start_time': last_shooting_time,
                    'end_time': event['Timestamp'],
                    'duration': event['Timestamp'] - last_shooting_time,
                    'cycle_type': 'shooting-to-intake',
                    'start_component': 'algae_shooter',
                    'end_component': event['component']
                })
            last_intake_time = event['Timestamp']
        else:  # shooting
            if last_intake_time is not None:
                # Time from last intake to this shooting
                cycles.append({
                    'start_time': last_intake_time,
                    'end_time': event['Timestamp'],
                    'duration': event['Timestamp'] - last_intake_time,
                    'cycle_type': 'intake-to-shooting',
                    'start_component': 'intake',
                    'end_component': 'algae_shooter'
                })
            last_shooting_time = event['Timestamp']

    cycles_df = pd.DataFrame(cycles)

    # Calculate statistics
    intake_to_shooting = cycles_df[cycles_df['cycle_type'] == 'intake-to-shooting']
    shooting_to_intake = cycles_df[cycles_df['cycle_type'] == 'shooting-to-intake']

    stats = {
        'total_cycles': len(cycles_df),
        'intake_to_shooting_count': len(intake_to_shooting),
        'shooting_to_intake_count': len(shooting_to_intake),
        'avg_intake_to_shooting': intake_to_shooting['duration'].mean() if len(intake_to_shooting) > 0 else 0,
        'avg_shooting_to_intake': shooting_to_intake['duration'].mean() if len(shooting_to_intake) > 0 else 0,
        'min_intake_to_shooting': intake_to_shooting['duration'].min() if len(intake_to_shooting) > 0 else 0,
        'max_intake_to_shooting': intake_to_shooting['duration'].max() if len(intake_to_shooting) > 0 else 0,
        'min_shooting_to_intake': shooting_to_intake['duration'].min() if len(shooting_to_intake) > 0 else 0,
        'max_shooting_to_intake': shooting_to_intake['duration'].max() if len(shooting_to_intake) > 0 else 0,
    }

    return cycles_df, intake_to_shooting, shooting_to_intake, stats


@app.cell
def display_stats(stats):
    """Display cycle time statistics"""

    mo.md(f"""
    ## Cycle Time Statistics

    | Metric | Intake → Shooting | Shooting → Intake |
    |--------|-------------------|-------------------|
    | **Count** | {stats['intake_to_shooting_count']} | {stats['shooting_to_intake_count']} |
    | **Average** | {stats['avg_intake_to_shooting']:.2f}s | {stats['avg_shooting_to_intake']:.2f}s |
    | **Minimum** | {stats['min_intake_to_shooting']:.2f}s | {stats['min_shooting_to_intake']:.2f}s |
    | **Maximum** | {stats['max_intake_to_shooting']:.2f}s | {stats['max_shooting_to_intake']:.2f}s |

    **Total Cycle Time:** {stats['avg_intake_to_shooting'] + stats['avg_shooting_to_intake']:.2f}s average
    """)
    return


@app.cell
def create_timeline_plot(events):
    """Create timeline visualization of events"""

    fig_timeline = go.Figure()

    # Add intake events
    intake_events = events[events['event_type'] == 'intake']
    fig_timeline.add_trace(go.Scatter(
        x=intake_events['Timestamp'],
        y=[1] * len(intake_events),
        mode='markers',
        marker=dict(size=10, color='green', symbol='triangle-up'),
        name='Intake Events',
        hovertemplate='<b>Intake</b><br>Time: %{x:.2f}s<br>Component: %{customdata}<extra></extra>',
        customdata=intake_events['component']
    ))

    # Add shooting events
    shooting_events = events[events['event_type'] == 'shooting']
    fig_timeline.add_trace(go.Scatter(
        x=shooting_events['Timestamp'],
        y=[2] * len(shooting_events),
        mode='markers',
        marker=dict(size=10, color='red', symbol='circle'),
        name='Shooting Events',
        hovertemplate='<b>Shooting</b><br>Time: %{x:.2f}s<extra></extra>'
    ))

    fig_timeline.update_layout(
        title='FRC Robot Events Timeline',
        xaxis_title='Time (seconds)',
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2],
            ticktext=['Intake', 'Shooting'],
            range=[0.5, 2.5]
        ),
        height=300,
        showlegend=True
    )

    return


@app.cell
def create_cycle_analysis(cycles_df):
    """Create comprehensive cycle analysis plots"""

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cycle Duration Distribution', 'Cycle Times Over Time', 
                       'Cycle Type Comparison', 'Duration Histogram'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # 1. Cycle duration scatter plot
    colors = {'intake-to-shooting': 'green', 'shooting-to-intake': 'orange'}
    for cycle_type in cycles_df['cycle_type'].unique():
        data = cycles_df[cycles_df['cycle_type'] == cycle_type]
        fig.add_trace(
            go.Scatter(
                x=data['start_time'],
                y=data['duration'],
                mode='markers',
                name=cycle_type.replace('-', ' → '),
                marker=dict(color=colors[cycle_type], size=8),
                hovertemplate=f'<b>{cycle_type.replace("-", " → ")}</b><br>Start: %{{x:.2f}}s<br>Duration: %{{y:.2f}}s<extra></extra>'
            ),
            row=1, col=1
        )

    # 2. Cycle times over sequence
    cycles_df_indexed = cycles_df.reset_index()
    for cycle_type in cycles_df['cycle_type'].unique():
        data = cycles_df_indexed[cycles_df_indexed['cycle_type'] == cycle_type]
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['duration'],
                mode='lines+markers',
                name=f'{cycle_type.replace("-", " → ")} Trend',
                line=dict(color=colors[cycle_type]),
                marker=dict(color=colors[cycle_type], size=6),
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. Box plot comparison
    for cycle_type in cycles_df['cycle_type'].unique():
        data = cycles_df[cycles_df['cycle_type'] == cycle_type]
        fig.add_trace(
            go.Box(
                y=data['duration'],
                name=cycle_type.replace('-', ' → '),
                marker=dict(color=colors[cycle_type]),
                showlegend=False
            ),
            row=2, col=1
        )

    # 4. Histogram
    for cycle_type in cycles_df['cycle_type'].unique():
        data = cycles_df[cycles_df['cycle_type'] == cycle_type]
        fig.add_trace(
            go.Histogram(
                x=data['duration'],
                name=cycle_type.replace('-', ' → '),
                opacity=0.7,
                marker=dict(color=colors[cycle_type]),
                showlegend=False
            ),
            row=2, col=2
        )

    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_yaxes(title_text="Duration (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Cycle Number", row=1, col=2)
    fig.update_yaxes(title_text="Duration (seconds)", row=1, col=2)
    fig.update_xaxes(title_text="Cycle Type", row=2, col=1)
    fig.update_yaxes(title_text="Duration (seconds)", row=2, col=1)
    fig.update_xaxes(title_text="Duration (seconds)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.update_layout(
        title='Comprehensive Cycle Analysis',
        height=800,
        showlegend=True
    )

    return


@app.cell
def performance_analysis(cycles_df, stats):
    """Analyze performance patterns"""

    # Calculate performance metrics
    cycles_df['performance'] = cycles_df.apply(
        lambda row: 'Above Average' if (
            row['cycle_type'] == 'intake-to-shooting' and 
            row['duration'] < stats['avg_intake_to_shooting']
        ) or (
            row['cycle_type'] == 'shooting-to-intake' and 
            row['duration'] < stats['avg_shooting_to_intake']
        ) else 'Below Average',
        axis=1
    )

    # Performance summary
    performance_summary = cycles_df.groupby(['cycle_type', 'performance']).size().unstack(fill_value=0)

    mo.md(f"""
    ## Performance Analysis

    ### Above vs Below Average Performance

    | Cycle Type | Above Average | Below Average | Success Rate |
    |------------|---------------|---------------|--------------|
    | **Intake → Shooting** | {performance_summary.loc['intake-to-shooting', 'Above Average'] if 'Above Average' in performance_summary.columns else 0} | {performance_summary.loc['intake-to-shooting', 'Below Average'] if 'Below Average' in performance_summary.columns else 0} | {(performance_summary.loc['intake-to-shooting', 'Above Average'] / (performance_summary.loc['intake-to-shooting', 'Above Average'] + performance_summary.loc['intake-to-shooting', 'Below Average']) * 100):.1f}% |
    | **Shooting → Intake** | {performance_summary.loc['shooting-to-intake', 'Above Average'] if 'Above Average' in performance_summary.columns else 0} | {performance_summary.loc['shooting-to-intake', 'Below Average'] if 'Below Average' in performance_summary.columns else 0} | {(performance_summary.loc['shooting-to-intake', 'Above Average'] / (performance_summary.loc['shooting-to-intake', 'Above Average'] + performance_summary.loc['shooting-to-intake', 'Below Average']) * 100):.1f}% |
    """)

    return


@app.cell
def create_data_table(cycles_df):
    """Create detailed data table"""

    # Prepare data for display
    display_df = cycles_df.copy()
    display_df['cycle_number'] = range(1, len(display_df) + 1)
    display_df['cycle_type_display'] = display_df['cycle_type'].str.replace('-', ' → ')
    display_df['start_time_formatted'] = display_df['start_time'].apply(lambda x: f"{x:.2f}s")
    display_df['end_time_formatted'] = display_df['end_time'].apply(lambda x: f"{x:.2f}s")
    display_df['duration_formatted'] = display_df['duration'].apply(lambda x: f"{x:.2f}s")

    # Select and reorder columns for display
    table_df = display_df[[
        'cycle_number', 'cycle_type_display', 'duration_formatted', 
        'start_time_formatted', 'end_time_formatted', 'performance'
    ]].rename(columns={
        'cycle_number': 'Cycle #',
        'cycle_type_display': 'Type',
        'duration_formatted': 'Duration',
        'start_time_formatted': 'Start Time',
        'end_time_formatted': 'End Time',
        'performance': 'Performance'
    })

    mo.md("""
    ## Detailed Cycle Data

    Complete breakdown of all cycles with performance indicators:
    """)

    return


@app.cell
def generate_insights(
    cycles_df,
    intake_to_shooting,
    shooting_to_intake,
    stats,
):
    """Generate insights and recommendations"""

    # Calculate variability
    intake_to_shooting_std = intake_to_shooting['duration'].std()
    shooting_to_intake_std = shooting_to_intake['duration'].std()

    # Best and worst cycles
    best_intake_to_shooting = intake_to_shooting.loc[intake_to_shooting['duration'].idxmin()]
    worst_intake_to_shooting = intake_to_shooting.loc[intake_to_shooting['duration'].idxmax()]
    best_shooting_to_intake = shooting_to_intake.loc[shooting_to_intake['duration'].idxmin()]
    worst_shooting_to_intake = shooting_to_intake.loc[shooting_to_intake['duration'].idxmax()]

    mo.md(f"""
    ## Key Insights & Recommendations

    ### 🔍 Analysis Summary

    **Cycle Consistency:**
    - Intake → Shooting variability: ±{intake_to_shooting_std:.2f}s
    - Shooting → Intake variability: ±{shooting_to_intake_std:.2f}s

    **Best Performances:**
    - Fastest Intake → Shooting: {best_intake_to_shooting['duration']:.2f}s at {best_intake_to_shooting['start_time']:.2f}s
    - Fastest Shooting → Intake: {best_shooting_to_intake['duration']:.2f}s at {best_shooting_to_intake['start_time']:.2f}s

    **Areas for Improvement:**
    - Slowest Intake → Shooting: {worst_intake_to_shooting['duration']:.2f}s at {worst_intake_to_shooting['start_time']:.2f}s
    - Slowest Shooting → Intake: {worst_shooting_to_intake['duration']:.2f}s at {worst_shooting_to_intake['start_time']:.2f}s

    ### 💡 Recommendations

    1. **Focus on Intake → Shooting Optimization**: This cycle takes {stats['avg_intake_to_shooting']/stats['avg_shooting_to_intake']:.1f}x longer than Shooting → Intake
    2. **Improve Consistency**: Work on reducing variability in cycle times
    3. **Target Time**: Aim for sub-{stats['avg_intake_to_shooting']:.0f}s intake-to-shooting cycles
    4. **Analyze Best Cycles**: Study the conditions during your fastest cycles ({best_intake_to_shooting['duration']:.2f}s and {best_shooting_to_intake['duration']:.2f}s)

    ### 📊 Performance Metrics

    - **Overall Robot Efficiency**: {(len(cycles_df[cycles_df['performance'] == 'Above Average']) / len(cycles_df) * 100):.1f}% cycles above average
    - **Theoretical Max Rate**: {60 / (stats['avg_intake_to_shooting'] + stats['avg_shooting_to_intake']):.1f} cycles per minute
    - **Room for Improvement**: {((worst_intake_to_shooting['duration'] - best_intake_to_shooting['duration']) / worst_intake_to_shooting['duration'] * 100):.1f}% potential time savings
    """)
    return


@app.cell
def export_data(cycles_df):
    """Export analyzed data"""

    export_button = mo.ui.button(
        label="Export Cycle Data to CSV",
        on_click=lambda: cycles_df.to_csv('frc_cycle_analysis.csv', index=False)
    )

    mo.md(f"""
    ## Data Export

    Export the analyzed cycle data for further analysis or sharing with your team.

    {export_button}

    The exported file will contain all {len(cycles_df)} cycles with:
    - Cycle types and durations
    - Start and end times
    - Performance indicators
    - Component information
    """)
    return


if __name__ == "__main__":
    app.run()

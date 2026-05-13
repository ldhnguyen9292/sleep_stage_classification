import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    'Sleep stage W': '#aaaaaa', 'Sleep stage 1': '#ffcccc',
    'Sleep stage 2': '#ff8800', 'Sleep stage 3': '#990099',
    'Sleep stage 4': '#990099', 'Sleep stage R': '#ff0000',
    'Sleep stage ?': '#000000'
}


def plot_signals(raw, selected_channels, start_sec, sfreq):
    data, times = raw.get_data(picks=selected_channels,
                               start=int(start_sec * sfreq),
                               stop=int((start_sec + 30) * sfreq),
                               return_times=True)

    fig = make_subplots(rows=len(selected_channels), cols=1,
                        shared_xaxes=True, vertical_spacing=0.05)
    rel_times = times - times[0]

    for i, ch in enumerate(selected_channels):
        fig.add_trace(go.Scatter(
            x=rel_times, y=data[i]*1e6, name=ch), row=i+1, col=1)
        fig.update_yaxes(title_text="µV", row=i+1, col=1)

    fig.update_layout(height=200*len(selected_channels),
                      showlegend=True, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def plot_hypnogram_timeline(df_ann):
    fig = go.Figure()
    for stage, group in df_ann.groupby('Stage'):
        for i, row in group.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Start'], row['End']], y=[1, 1],
                mode='lines', line=dict(width=40, color=COLORS.get(stage, '#333')),
                name=stage,
                showlegend=False if i > group.index[0] else True,
                hovertemplate=f"<b>{stage}</b><br>Bắt đầu: %{{x|%H:%M:%S}}<extra></extra>"
            ))
    fig.update_layout(height=200, yaxis=dict(showticklabels=False, range=[
                      0.5, 1.5]), margin=dict(l=10, r=10, t=30, b=10))
    return fig

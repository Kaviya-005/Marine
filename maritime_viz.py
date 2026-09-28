import plotly.graph_objects as go
import numpy as np
import math

def create_maritime_map(env, trajectory=None, step_index=None):
    """
    Renders a 2D Plotly maritime navigation map with a COMPLETELY FIXED VIEWPORT.
    All axes, ranges, dimensions, and legend positions remain 100% stationary throughout simulation steps.
    Legend uses clear, icon-free textual names matching the requested exact order.
    """
    fig = go.Figure()

    map_w = env.map_width
    map_h = env.map_height
    theta = np.linspace(0, 2*np.pi, 100)

    # Determine current playback positions & status
    if trajectory is not None and len(trajectory['own_x']) > 0:
        idx = step_index if (step_index is not None and step_index < len(trajectory['own_x'])) else -1
        own_x = trajectory['own_x'][idx]
        own_y = trajectory['own_y'][idx]
        own_h = trajectory['own_heading'][idx]
        own_v = trajectory['own_speed'][idx]
        current_status = trajectory['step_status'][idx] if ('step_status' in trajectory and idx < len(trajectory['step_status'])) else 'IN_PROGRESS'
        targets_at_step = trajectory['target_positions'][idx] if 'target_positions' in trajectory else []
        max_idx = step_index + 1 if (step_index is not None and step_index < len(trajectory['own_x'])) else len(trajectory['own_x'])
    else:
        own_x = env.own_x
        own_y = env.own_y
        own_h = math.degrees(env.own_heading)
        own_v = env.own_speed
        current_status = 'IN_PROGRESS'
        targets_at_step = [(ts['x'], ts['y']) for ts in env.target_ships]
        max_idx = 0

    # Colors for target vessels
    target_colors = ['#c0392b', '#8e44ad', '#16a085', '#d35400', '#34495e']

    # -------------------------------------------------------------------------
    # TRACE 1: Own Ship (SAC Controlled)
    # -------------------------------------------------------------------------
    own_rad = math.radians(own_h)
    arrow_len_own = map_w * 0.06
    own_head_x = own_x + arrow_len_own * math.cos(own_rad)
    own_head_y = own_y + arrow_len_own * math.sin(own_rad)

    own_color = '#2980b9'
    if current_status == 'COLLISION_RISK_DETECTED':
        own_color = '#e67e22'
    elif current_status == 'AVOIDANCE_SUCCESSFUL':
        own_color = '#2ecc71'
    elif current_status == 'ACTUAL_COLLISION':
        own_color = '#e74c3c'

    fig.add_trace(go.Scatter(
        x=[own_x, own_head_x], y=[own_y, own_head_y],
        mode='lines+markers',
        line=dict(color=own_color, width=4),
        marker=dict(symbol=['diamond', 'arrow-bar-up'], size=[14, 10], color=[own_color, '#16a085']),
        name='Own Ship (SAC Controlled)',
        hovertext=f'Own Ship (SAC)<br>Pos: ({own_x:.1f}, {own_y:.1f})<br>Heading: {own_h:.1f}°<br>Speed: {own_v:.1f} m/s'
    ))

    # -------------------------------------------------------------------------
    # TRACE 2: Own Ship Trajectory
    # -------------------------------------------------------------------------
    if trajectory is not None and len(trajectory['own_x']) > 0:
        own_x_path = trajectory['own_x'][:max_idx]
        own_y_path = trajectory['own_y'][:max_idx]
        fig.add_trace(go.Scatter(
            x=own_x_path, y=own_y_path,
            mode='lines',
            line=dict(color='#2980b9', width=3),
            name='Own Ship Trajectory'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='#2980b9', width=3),
            name='Own Ship Trajectory'
        ))

    # -------------------------------------------------------------------------
    # TRACE 3: Own Ship Safety Boundary
    # -------------------------------------------------------------------------
    own_safe_x = own_x + env.safe_distance * np.cos(theta)
    own_safe_y = own_y + env.safe_distance * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=own_safe_x, y=own_safe_y,
        mode='lines',
        line=dict(color='rgba(52, 152, 219, 0.4)', width=1.5, dash='dash'),
        fill='toself',
        fillcolor='rgba(52, 152, 219, 0.05)',
        name='Own Ship Safety Boundary',
        hoverinfo='skip'
    ))

    # -------------------------------------------------------------------------
    # TRACE 4 to 9: Target Ships & Target Ship Trajectories (Up to 3 max in legend)
    # -------------------------------------------------------------------------
    for i in range(min(3, len(env.target_ships))):
        color = target_colors[i % len(target_colors)]

        # Target Vessel Position & Marker
        if i < len(targets_at_step):
            tx, ty = targets_at_step[i][0], targets_at_step[i][1]
            ts_orig = env.target_ships[i]
            t_rad = ts_orig.get('heading', 0) if isinstance(ts_orig.get('heading'), float) else 0
            t_arrow_len = map_w * 0.04
            t_head_x = tx + t_arrow_len * math.cos(t_rad)
            t_head_y = ty + t_arrow_len * math.sin(t_rad)
            t_h_deg = math.degrees(t_rad)

            fig.add_trace(go.Scatter(
                x=[tx, t_head_x], y=[ty, t_head_y],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(symbol=['circle', 'arrow'], size=[10, 8], color=color),
                name=f'Target Ship {i+1}',
                hovertext=f'Target Ship {i+1}<br>Pos: ({tx:.1f}, {ty:.1f})<br>Heading: {t_h_deg:.1f}°'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(symbol='circle', size=10, color=color),
                name=f'Target Ship {i+1}'
            ))

        # Target Vessel Trajectory
        if trajectory is not None and 'target_positions' in trajectory and len(trajectory['target_positions']) > 0:
            tx_path = [trajectory['target_positions'][step][i][0] for step in range(max_idx) if i < len(trajectory['target_positions'][step])]
            ty_path = [trajectory['target_positions'][step][i][1] for step in range(max_idx) if i < len(trajectory['target_positions'][step])]
            fig.add_trace(go.Scatter(
                x=tx_path, y=ty_path,
                mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=f'Target Ship {i+1} Trajectory'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color=color, width=2, dash='dot'),
                name=f'Target Ship {i+1} Trajectory'
            ))

    # -------------------------------------------------------------------------
    # TRACE 10: Destination
    # -------------------------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=[env.goal_x], y=[env.goal_y],
        mode='markers+text',
        marker=dict(symbol='star', size=18, color='#2ecc71', line=dict(color='#27ae60', width=2)),
        text=['DESTINATION'],
        textposition='top center',
        name='Destination'
    ))

    # -------------------------------------------------------------------------
    # TRACE 11: Goal Acceptance Zone
    # -------------------------------------------------------------------------
    circle_x = env.goal_x + env.goal_radius * np.cos(theta)
    circle_y = env.goal_y + env.goal_radius * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode='lines',
        line=dict(color='rgba(46, 204, 113, 0.7)', width=2, dash='dash'),
        fill='toself',
        fillcolor='rgba(46, 204, 113, 0.15)',
        name='Goal Acceptance Zone'
    ))

    # -------------------------------------------------------------------------
    # TRACE 12: Collision Warning Zone
    # -------------------------------------------------------------------------
    if len(targets_at_step) > 0:
        tx0, ty0 = targets_at_step[0][0], targets_at_step[0][1]
        warn_x = tx0 + env.collision_warning_radius * np.cos(theta)
        warn_y = ty0 + env.collision_warning_radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=warn_x, y=warn_y,
            mode='lines',
            line=dict(color='rgba(230, 126, 34, 0.7)', width=1.5, dash='dot'),
            fill='toself',
            fillcolor='rgba(243, 156, 18, 0.08)',
            name='Collision Warning Zone',
            hoverinfo='skip'
        ))
        # Add warning circles for remaining targets without duplicate legend items
        for t_idx in range(1, len(targets_at_step)):
            tx_i, ty_i = targets_at_step[t_idx][0], targets_at_step[t_idx][1]
            fig.add_trace(go.Scatter(
                x=tx_i + env.collision_warning_radius * np.cos(theta),
                y=ty_i + env.collision_warning_radius * np.sin(theta),
                mode='lines',
                line=dict(color='rgba(230, 126, 34, 0.7)', width=1.5, dash='dot'),
                fill='toself',
                fillcolor='rgba(243, 156, 18, 0.08)',
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='rgba(230, 126, 34, 0.7)', width=1.5, dash='dot'),
            name='Collision Warning Zone'
        ))

    # -------------------------------------------------------------------------
    # TRACE 13: Actual Collision Zone
    # -------------------------------------------------------------------------
    if len(targets_at_step) > 0:
        tx0, ty0 = targets_at_step[0][0], targets_at_step[0][1]
        coll_x = tx0 + env.collision_radius * np.cos(theta)
        coll_y = ty0 + env.collision_radius * np.sin(theta)
        fig.add_trace(go.Scatter(
            x=coll_x, y=coll_y,
            mode='lines',
            line=dict(color='rgba(231, 76, 60, 0.8)', width=1.5, dash='dash'),
            fill='toself',
            fillcolor='rgba(231, 76, 60, 0.15)',
            name='Actual Collision Zone',
            hoverinfo='skip'
        ))
        for t_idx in range(1, len(targets_at_step)):
            tx_i, ty_i = targets_at_step[t_idx][0], targets_at_step[t_idx][1]
            fig.add_trace(go.Scatter(
                x=tx_i + env.collision_radius * np.cos(theta),
                y=ty_i + env.collision_radius * np.sin(theta),
                mode='lines',
                line=dict(color='rgba(231, 76, 60, 0.8)', width=1.5, dash='dash'),
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.15)',
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='rgba(231, 76, 60, 0.8)', width=1.5, dash='dash'),
            name='Actual Collision Zone'
        ))

    # -------------------------------------------------------------------------
    # TRACE 14: Avoidance Path
    # -------------------------------------------------------------------------
    if trajectory is not None and 'step_status' in trajectory and max_idx > 0:
        warn_indices = [i for i in range(max_idx) if trajectory['step_status'][i] in ('COLLISION_RISK_DETECTED', 'AVOIDANCE_SUCCESSFUL')]
        if len(warn_indices) > 0:
            own_x_path = trajectory['own_x'][:max_idx]
            own_y_path = trajectory['own_y'][:max_idx]
            warn_x = [own_x_path[i] for i in warn_indices]
            warn_y = [own_y_path[i] for i in warn_indices]
            fig.add_trace(go.Scatter(
                x=warn_x, y=warn_y,
                mode='lines+markers',
                line=dict(color='#e67e22', width=5),
                marker=dict(color='#d35400', size=6),
                name='Avoidance Path'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='lines',
                line=dict(color='#e67e22', width=5),
                name='Avoidance Path'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='#e67e22', width=5),
            name='Avoidance Path'
        ))

    # -------------------------------------------------------------------------
    # TRACE 15: Ocean Current
    # -------------------------------------------------------------------------
    n_grid = 7
    xs = np.linspace(map_w * 0.1, map_w * 0.9, n_grid)
    ys = np.linspace(map_h * 0.1, map_h * 0.9, n_grid)
    curr_speed = env.ocean_current_speed
    curr_rad = math.radians(env.ocean_current_direction)
    arrow_len = map_w * 0.05 * (curr_speed / max(curr_speed, 1.0))

    for x in xs:
        for y in ys:
            cdx = x + arrow_len * math.cos(curr_rad)
            cdy = y + arrow_len * math.sin(curr_rad)
            fig.add_annotation(
                x=cdx, y=cdy, ax=x, ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor="rgba(0, 150, 255, 0.35)"
            )

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='rgba(0, 150, 255, 0.6)', width=2),
        name='Ocean Current'
    ))

    # -------------------------------------------------------------------------
    # MAP ANNOTATIONS (Pinned directly to Own Ship position during risk/avoidance)
    # -------------------------------------------------------------------------
    if current_status == 'COLLISION_RISK_DETECTED':
        fig.add_annotation(
            x=own_x, y=own_y + 40,
            text="COLLISION RISK DETECTED<br>Initiating SAC Avoidance",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#e67e22",
            font=dict(size=12, color="white", family="Arial Black"),
            bgcolor="#e67e22",
            bordercolor="#d35400",
            borderwidth=2,
            borderpad=6
        )
    elif current_status == 'AVOIDANCE_SUCCESSFUL':
        fig.add_annotation(
            x=own_x, y=own_y + 40,
            text="AVOIDANCE SUCCESSFUL<br>Resuming Path to Destination",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#2ecc71",
            font=dict(size=12, color="white", family="Arial Black"),
            bgcolor="#2ecc71",
            bordercolor="#27ae60",
            borderwidth=2,
            borderpad=6
        )

    # -------------------------------------------------------------------------
    # RIGID VIEWPORT & FIXED LAYOUT CONFIGURATION
    # -------------------------------------------------------------------------
    fixed_pad = 50.0
    x_min = -fixed_pad
    x_max = map_w + fixed_pad
    y_min = -fixed_pad
    y_max = map_h + fixed_pad

    fig.update_layout(
        title=dict(text="MASS Autonomous Surface Ship Navigation Map", font=dict(size=18, color='#1E3A8A')),
        xaxis=dict(
            range=[x_min, x_max],
            autorange=False,
            fixedrange=True,
            title="X Position (meters)",
            gridcolor='#e1e8ed',
            zeroline=False
        ),
        yaxis=dict(
            range=[y_min, y_max],
            autorange=False,
            fixedrange=True,
            title="Y Position (meters)",
            gridcolor='#e1e8ed',
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        ),
        width=850,
        height=650,
        autosize=False,
        plot_bgcolor='#f4f8fb',
        paper_bgcolor='white',
        legend=dict(
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#bdc3c7',
            borderwidth=1,
            font=dict(size=11)
        ),
        margin=dict(l=50, r=230, t=60, b=50)
    )

    return fig

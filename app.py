import streamlit as st
import numpy as np
import pandas as pd
import math
import time
import os
import plotly.graph_objects as go
import plotly.express as px

from maritime_env import MaritimeEnv
from sac_agent import create_sac_model, train_sac_agent, save_sac_model, load_sac_model
from maritime_viz import create_maritime_map

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="MASS - Maritime Autonomous Ship SAC Navigation",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 400;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .status-success {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.1rem;
    }
    .status-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.1rem;
    }
    .status-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.1rem;
    }
    .status-info {
        background-color: #E0F2FE;
        color: #0369A1;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">MASS Autonomous Surface Ship Navigation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Soft Actor-Critic (SAC) Path Planning, Ocean Current Drift Compensation & Sequential Collision-Risk Avoidance</div>', unsafe_allow_html=True)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "maritime_sac.zip")

# ---------------------------------------------------------
# Initialize Session State
# ---------------------------------------------------------
if 'env_config' not in st.session_state:
    st.session_state['env_config'] = {
        'map_width': 1000.0,
        'map_height': 1000.0,
        'own_ship_init_x': 100.0,
        'own_ship_init_y': 100.0,
        'own_ship_init_heading': 45.0,
        'own_ship_init_speed': 5.0,
        'own_ship_max_speed': 12.0,
        'own_ship_max_turn_rate': 15.0,
        'goal_x': 900.0,
        'goal_y': 900.0,
        'goal_radius': 40.0,
        'ocean_current_speed': 1.5,
        'ocean_current_direction': 90.0,
        'max_steps': 350,
        'dt': 1.0,
        'safe_distance': 220.0,
        'collision_warning_radius': 160.0,
        'collision_radius': 40.0,
        'target_ships': [
            {'x': 260.0, 'y': 260.0, 'speed': 3.5, 'heading': 225.0},
            {'x': 650.0, 'y': 550.0, 'speed': 3.0, 'heading': 200.0},
            {'x': 300.0, 'y': 700.0, 'speed': 3.0, 'heading': 315.0}
        ]
    }

if 'env' not in st.session_state:
    st.session_state['env'] = MaritimeEnv(st.session_state['env_config'])

if 'sac_model' not in st.session_state:
    st.session_state['sac_model'] = None

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH[:-4] if MODEL_PATH.endswith('.zip') else MODEL_PATH)
    if st.session_state['model_trained']:
        try:
            st.session_state['sac_model'] = load_sac_model(MODEL_PATH, env=st.session_state['env'])
        except Exception:
            st.session_state['model_trained'] = False

if 'sim_results' not in st.session_state:
    st.session_state['sim_results'] = None

if 'training_stats' not in st.session_state:
    st.session_state['training_stats'] = None


# ---------------------------------------------------------
# Sidebar Configuration Controls
# ---------------------------------------------------------
st.sidebar.header("Environment Configuration")

# 1. Map & Collision Thresholds
with st.sidebar.expander("Map & Collision Thresholds", expanded=True):
    map_w = st.number_input("Map Width (m)", min_value=500.0, max_value=5000.0, value=st.session_state['env_config']['map_width'], step=100.0)
    map_h = st.number_input("Map Height (m)", min_value=500.0, max_value=5000.0, value=st.session_state['env_config']['map_height'], step=100.0)
    max_steps = st.number_input("Max Simulation Steps", min_value=50, max_value=1000, value=st.session_state['env_config']['max_steps'], step=50)

    st.markdown("**Collision Risk & Safety Hierarchy:**")
    safe_dist = st.number_input("Outer Safety Boundary (m)", min_value=50.0, max_value=300.0, value=st.session_state['env_config']['safe_distance'], step=10.0)
    warn_radius = st.number_input("Collision Warning Radius (m)", min_value=30.0, max_value=250.0, value=st.session_state['env_config']['collision_warning_radius'], step=5.0)
    coll_radius = st.number_input("Actual Collision Radius (m)", min_value=10.0, max_value=100.0, value=st.session_state['env_config']['collision_radius'], step=5.0)

    if not (safe_dist > warn_radius > coll_radius):
        st.error("Error: Must maintain hierarchy: Outer Safety > Warning Radius > Collision Radius")

# 2. Own Vessel Params
with st.sidebar.expander("Own Vessel Parameters", expanded=True):
    own_x = st.number_input("Initial X (m)", min_value=0.0, max_value=map_w, value=st.session_state['env_config']['own_ship_init_x'], step=20.0)
    own_y = st.number_input("Initial Y (m)", min_value=0.0, max_value=map_h, value=st.session_state['env_config']['own_ship_init_y'], step=20.0)
    own_h = st.slider("Initial Heading (°)", min_value=0.0, max_value=360.0, value=float(st.session_state['env_config']['own_ship_init_heading']))
    own_v = st.number_input("Initial Speed (m/s)", min_value=0.0, max_value=20.0, value=st.session_state['env_config']['own_ship_init_speed'], step=0.5)
    own_v_max = st.number_input("Max Speed (m/s)", min_value=5.0, max_value=30.0, value=st.session_state['env_config']['own_ship_max_speed'], step=1.0)
    own_turn_max = st.number_input("Max Turn Rate (°/s)", min_value=5.0, max_value=45.0, value=st.session_state['env_config']['own_ship_max_turn_rate'], step=1.0)

# 3. Destination (Goal) Params
with st.sidebar.expander("Destination Goal", expanded=True):
    goal_x = st.number_input("Goal X (m)", min_value=0.0, max_value=map_w, value=st.session_state['env_config']['goal_x'], step=20.0)
    goal_y = st.number_input("Goal Y (m)", min_value=0.0, max_value=map_h, value=st.session_state['env_config']['goal_y'], step=20.0)
    goal_r = st.number_input("Goal Acceptance Radius (m)", min_value=10.0, max_value=150.0, value=st.session_state['env_config']['goal_radius'], step=5.0)

# 4. Ocean Current Params
with st.sidebar.expander("Ocean Current Effect", expanded=True):
    curr_speed = st.slider("Current Speed (m/s)", min_value=0.0, max_value=6.0, value=float(st.session_state['env_config']['ocean_current_speed']), step=0.2)
    curr_dir = st.slider("Current Flow Direction (°)", min_value=0.0, max_value=360.0, value=float(st.session_state['env_config']['ocean_current_direction']), step=15.0)

# 5. Target Vessels Params
with st.sidebar.expander("Target Vessels Configuration", expanded=False):
    num_targets = st.slider("Number of Target Ships", min_value=1, max_value=5, value=len(st.session_state['env_config']['target_ships']))
    target_ships_config = []

    for i in range(num_targets):
        st.markdown(f"**Target Vessel {i+1}**")
        default_ts = st.session_state['env_config']['target_ships'][i] if i < len(st.session_state['env_config']['target_ships']) else {'x': 400.0, 'y': 400.0, 'speed': 3.0, 'heading': 180.0}
        tc1, tc2 = st.columns(2)
        tx = tc1.number_input(f"TS{i+1} X", 0.0, map_w, float(default_ts['x']), key=f"tx_{i}")
        ty = tc2.number_input(f"TS{i+1} Y", 0.0, map_h, float(default_ts['y']), key=f"ty_{i}")
        tc3, tc4 = st.columns(2)
        tv = tc3.number_input(f"TS{i+1} Speed", 0.0, 15.0, float(default_ts['speed']), key=f"tv_{i}")
        th = tc4.number_input(f"TS{i+1} Heading", 0.0, 360.0, float(default_ts['heading']), key=f"th_{i}")
        target_ships_config.append({'x': tx, 'y': ty, 'speed': tv, 'heading': th})

# Apply & Create Environment Button
if st.sidebar.button("Create / Reset Environment", type="primary", use_container_width=True):
    new_config = {
        'map_width': map_w,
        'map_height': map_h,
        'own_ship_init_x': own_x,
        'own_ship_init_y': own_y,
        'own_ship_init_heading': own_h,
        'own_ship_init_speed': own_v,
        'own_ship_max_speed': own_v_max,
        'own_ship_max_turn_rate': own_turn_max,
        'goal_x': goal_x,
        'goal_y': goal_y,
        'goal_radius': goal_r,
        'ocean_current_speed': curr_speed,
        'ocean_current_direction': curr_dir,
        'max_steps': max_steps,
        'dt': 1.0,
        'safe_distance': safe_dist,
        'collision_warning_radius': warn_radius,
        'collision_radius': coll_radius,
        'target_ships': target_ships_config
    }
    st.session_state['env_config'] = new_config
    st.session_state['env'] = MaritimeEnv(new_config)
    if os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH[:-4] if MODEL_PATH.endswith('.zip') else MODEL_PATH):
        try:
            st.session_state['sac_model'] = load_sac_model(MODEL_PATH, env=st.session_state['env'])
            st.session_state['model_trained'] = True
        except Exception:
            st.session_state['model_trained'] = False
    st.session_state['sim_results'] = None
    st.success("Maritime environment reset with updated parameters & thresholds!")


# ---------------------------------------------------------
# Main Interface Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Maritime Environment",
    "Train SAC Agent",
    "Run Autonomous Simulation",
    "Trajectory Analytics & State Data"
])

# ---------------------------------------------------------
# TAB 1: Environment Overview
# ---------------------------------------------------------
with tab1:
    st.subheader("Interactive Soft Actor-Critic Maritime Simulation Map")

    c_map1, c_map2 = st.columns([3, 1])

    with c_map1:
        map_fig = create_maritime_map(st.session_state['env'])
        st.plotly_chart(map_fig, use_container_width=True)

    with c_map2:
        st.markdown("### Active Configuration")
        env_obj = st.session_state['env']
        st.metric("Algorithm", "Soft Actor-Critic (SAC)")
        st.metric("Own Ship Start", f"({env_obj.own_ship_init_x:.0f}, {env_obj.own_ship_init_y:.0f})")
        st.metric("Destination Goal", f"({env_obj.goal_x:.0f}, {env_obj.goal_y:.0f})")
        st.metric("Ocean Current", f"{env_obj.ocean_current_speed} m/s @ {env_obj.ocean_current_direction}°")

        st.markdown("**Collision Thresholds:**")
        st.write(f"- Outer Safety: **{env_obj.safe_distance} m**")
        st.write(f"- Risk Warning: **{env_obj.collision_warning_radius} m**")
        st.write(f"- Actual Collision: **{env_obj.collision_radius} m**")

        model_status_text = "🟢 SAC Model Loaded" if st.session_state['model_trained'] else "🔴 Untrained SAC Model"
        st.markdown(f"**Agent Status:** {model_status_text}")


# ---------------------------------------------------------
# TAB 2: SAC Agent Training
# ---------------------------------------------------------
with tab2:
    st.subheader("Train Soft Actor-Critic (SAC) Agent")
    st.markdown("""
    Train a **Soft Actor-Critic (SAC)** continuous-control DRL agent using **Stable-Baselines3**.
    The SAC agent learns an optimal policy to navigate to the goal while dynamically adjusting course to avoid moving vessels.
    """)

    tc1, tc2, tc3 = st.columns(3)
    timesteps = tc1.number_input("Total Training Timesteps", min_value=2000, max_value=100000, value=15000, step=2000)
    learning_rate = tc2.select_slider("Learning Rate", options=[1e-4, 3e-4, 5e-4, 1e-3], value=3e-4)
    gamma = tc3.slider("Discount Factor (Gamma)", min_value=0.90, max_value=0.999, value=0.99, step=0.005)

    if st.button("Start SAC Training", type="primary", use_container_width=True):
        st.info("Initializing Soft Actor-Critic (SAC) Training Loop with Stable-Baselines3...")

        progress_bar = st.progress(0)
        status_text = st.empty()
        chart_placeholder = st.empty()

        env = st.session_state['env']
        model = create_sac_model(env, learning_rate=learning_rate, gamma=gamma)

        def on_step_complete(step, total_step, ep_reward, loss, outcome, stats):
            progress = min(1.0, step / total_step)
            progress_bar.progress(progress)
            status_text.markdown(f"**Timestep {step}/{total_step}** | Latest Episode Reward: **{ep_reward:.1f}** | Status: `{outcome}`")

            if len(stats['episodes']) > 0 and len(stats['episodes']) % 3 == 0:
                df_stats = pd.DataFrame({
                    'Episode': stats['episodes'],
                    'Reward': stats['rewards']
                })
                fig_tr = px.line(df_stats, x='Episode', y='Reward', title="SAC Episode Reward Curve", color_discrete_sequence=['#2ecc71'])
                fig_tr.add_trace(go.Scatter(
                    x=df_stats['Episode'],
                    y=df_stats['Reward'].rolling(window=5, min_periods=1).mean(),
                    mode='lines', line=dict(color='#e74c3c', width=2), name='Moving Average'
                ))
                chart_placeholder.plotly_chart(fig_tr, use_container_width=True)

        model, stats = train_sac_agent(env, model=model, total_timesteps=timesteps, progress_callback=on_step_complete)

        # Save trained SAC model
        save_sac_model(model, MODEL_PATH)
        st.session_state['sac_model'] = model
        st.session_state['model_trained'] = True
        st.session_state['training_stats'] = stats

        st.success(f"SAC Agent successfully trained for {timesteps} timesteps! Model saved to `{MODEL_PATH}`.")

    # Show training history if available
    if st.session_state['training_stats'] is not None:
        st.markdown("---")
        st.subheader("SAC Training History Summary")
        stats = st.session_state['training_stats']
        df_stats = pd.DataFrame(stats)

        goal_rate = (df_stats['outcomes'] == 'GOAL_REACHED').mean() * 100.0 if len(df_stats) > 0 else 0.0
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Episodes", len(df_stats))
        m2.metric("Goal Reach Success Rate", f"{goal_rate:.1f}%")
        m3.metric("Avg Reward (Last 10)", f"{df_stats['rewards'].tail(10).mean():.1f}" if len(df_stats) > 0 else "N/A")


# ---------------------------------------------------------
# TAB 3: Run Autonomous Simulation
# ---------------------------------------------------------
with tab3:
    st.subheader("Execute Autonomous SAC Simulation (Sequential Collision-Risk -> Avoidance)")

    if not st.session_state['model_trained'] or st.session_state['sac_model'] is None:
        st.warning("⚠️ No trained SAC model found! Please train the agent in Tab 2 or click Create Environment.")
    else:
        st.success("🟢 Soft Actor-Critic (SAC) Model Loaded and Ready.")

        col_sim1, col_sim2 = st.columns([1, 1])
        playback_speed = col_sim1.select_slider("Simulation Speed", options=["Fast (0.01s)", "Normal (0.05s)", "Step-by-Step (0.15s)"], value="Normal (0.05s)")
        delay_map = {"Fast (0.01s)": 0.01, "Normal (0.05s)": 0.05, "Step-by-Step (0.15s)": 0.15}

        if st.button("Run Simulation Now", type="primary", use_container_width=True):
            env = st.session_state['env']
            model = st.session_state['sac_model']

            obs, info = env.reset()

            # Initialize fixed placeholders in exact top-to-bottom layout order
            banner_placeholder = st.empty()
            sim_plot_placeholder = st.empty()
            metrics_placeholder = st.empty()

            step_logs = []
            total_reward = 0.0
            step_count = 0
            terminated = False
            truncated = False

            warning_event_occurred = False
            avoidance_event_occurred = False

            while not (terminated or truncated):
                # Predict action using SAC model
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                step_count += 1
                curr_status = info['status']

                if curr_status == 'COLLISION_RISK_DETECTED':
                    warning_event_occurred = True
                elif curr_status == 'AVOIDANCE_SUCCESSFUL':
                    avoidance_event_occurred = True

                step_logs.append({
                    'Step': step_count,
                    'Own_X': info['own_x'],
                    'Own_Y': info['own_y'],
                    'Own_Heading_Deg': info['own_heading_deg'],
                    'Own_Speed': info['own_speed'],
                    'Action_Turn': action[0],
                    'Action_Speed': action[1],
                    'Reward': reward,
                    'Dist_To_Goal': info['distance_to_goal'],
                    'Min_Target_Dist': info['min_target_distance'],
                    'Status': curr_status
                })

                # Display Banner Notifications for Sequential Events in fixed top banner placeholder
                if curr_status == 'COLLISION_RISK_DETECTED':
                    banner_placeholder.markdown(f'<div class="status-warning">COLLISION RISK DETECTED! Target vessel within warning radius ({env.collision_warning_radius:.0f}m). SAC initiating avoidance maneuver...</div>', unsafe_allow_html=True)
                elif curr_status == 'AVOIDANCE_SUCCESSFUL':
                    banner_placeholder.markdown('<div class="status-info">AVOIDANCE SUCCESSFUL! Own vessel successfully altered heading and expanded safe distance. Resuming navigation to destination.</div>', unsafe_allow_html=True)
                elif curr_status == 'GOAL_REACHED':
                    banner_placeholder.markdown('<div class="status-success">SUCCESS: Destination Goal Reached Safely!</div>', unsafe_allow_html=True)
                elif curr_status in ('ACTUAL_COLLISION', 'BOUNDARY_COLLISION'):
                    banner_placeholder.markdown(f'<div class="status-danger">TERMINAL FAILURE: {curr_status}</div>', unsafe_allow_html=True)

                # Live Map Update in fixed plot placeholder (fixed aspect ratio and bounds)
                if step_count % 2 == 0 or terminated or truncated or curr_status in ('COLLISION_RISK_DETECTED', 'AVOIDANCE_SUCCESSFUL'):
                    fig_live = create_maritime_map(env, trajectory=info['history'], step_index=step_count-1)
                    sim_plot_placeholder.plotly_chart(fig_live, use_container_width=True)

                    with metrics_placeholder.container():
                        sm1, sm2, sm3, sm4 = st.columns(4)
                        sm1.metric("Current Step", f"{step_count} / {env.max_steps}")
                        sm2.metric("Distance to Goal", f"{info['distance_to_goal']:.1f} m")
                        sm3.metric("Min Target Distance", f"{info['min_target_distance']:.1f} m")
                        sm4.metric("Cumulative Reward", f"{total_reward:.1f}")

                obs = next_obs
                time.sleep(delay_map[playback_speed])

            # Save simulation results
            final_status = info['status']
            st.session_state['sim_results'] = {
                'status': final_status,
                'total_reward': total_reward,
                'steps': step_count,
                'final_dist_goal': info['distance_to_goal'],
                'min_target_dist': info['min_target_distance'],
                'warning_occurred': warning_event_occurred,
                'avoidance_occurred': avoidance_event_occurred,
                'step_logs': pd.DataFrame(step_logs),
                'history': info['history']
            }

        # Render static completed map if available
        elif st.session_state['sim_results'] is not None:
            res = st.session_state['sim_results']
            fig_final = create_maritime_map(st.session_state['env'], trajectory=res['history'])
            st.plotly_chart(fig_final, use_container_width=True)

            status_str = res['status']
            if status_str == 'GOAL_REACHED':
                st.markdown('<div class="status-success">SUCCESS: Destination Goal Reached Safely!</div>', unsafe_allow_html=True)
            elif status_str in ('ACTUAL_COLLISION', 'BOUNDARY_COLLISION'):
                st.markdown(f'<div class="status-danger">TERMINAL FAILURE: {status_str}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-warning">SIMULATION ENDED: {status_str}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------
# TAB 4: Trajectory Analytics & Empirical SAC Data
# ---------------------------------------------------------
with tab4:
    st.subheader("Simulation Trajectory & SAC Empirical Results")

    if st.session_state['sim_results'] is None:
        st.info("No simulation run performed yet. Click 'Run Simulation Now' in Tab 3 to view detailed empirical results.")
    else:
        res = st.session_state['sim_results']
        df_logs = res['step_logs']

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Final Status", res['status'])
        kpi2.metric("Total Steps", res['steps'])
        kpi3.metric("Cumulative Reward", f"{res['total_reward']:.2f}")
        kpi4.metric("Collision Risk Event", "Yes (Triggered)" if res['warning_occurred'] else "No")
        kpi5.metric("Avoidance Event", "Yes (Successful)" if res['avoidance_occurred'] else "No")

        st.markdown("---")

        c_an1, c_an2 = st.columns(2)

        with c_an1:
            fig_dist = px.line(df_logs, x='Step', y='Dist_To_Goal', title="Distance to Destination Goal Over Time (m)", color_discrete_sequence=['#27ae60'])
            fig_dist.add_hline(y=st.session_state['env'].goal_radius, line_dash="dash", line_color="red", annotation_text="Goal Acceptance Radius")
            st.plotly_chart(fig_dist, use_container_width=True)

        with c_an2:
            fig_safe = px.line(df_logs, x='Step', y='Min_Target_Dist', title="Minimum Distance to Target Vessels (m)", color_discrete_sequence=['#e74c3c'])
            fig_safe.add_hline(y=st.session_state['env'].collision_warning_radius, line_dash="dot", line_color="#e67e22", annotation_text=f"Warning Threshold ({st.session_state['env'].collision_warning_radius:.0f}m)")
            fig_safe.add_hline(y=st.session_state['env'].collision_radius, line_dash="dash", line_color="red", annotation_text=f"Actual Collision Threshold ({st.session_state['env'].collision_radius:.0f}m)")
            st.plotly_chart(fig_safe, use_container_width=True)

        c_an3, c_an4 = st.columns(2)

        with c_an3:
            fig_act = px.line(df_logs, x='Step', y=['Action_Turn', 'Action_Speed'], title="SAC Control Commands (Rudder Turn & Throttle Speed)")
            st.plotly_chart(fig_act, use_container_width=True)

        with c_an4:
            fig_rew = px.line(df_logs, x='Step', y='Reward', title="Step Reward Signal Received from Environment", color_discrete_sequence=['#8e44ad'])
            st.plotly_chart(fig_rew, use_container_width=True)

        st.markdown("### Empirical Step-by-Step Environment & SAC Execution Log")
        st.dataframe(df_logs, use_container_width=True, height=300)

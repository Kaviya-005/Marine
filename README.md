# Maritime Autonomous Surface Ship (MASS) Navigation using Soft Actor-Critic (SAC)

An interactive, self-contained, end-to-end Python Streamlit application for Maritime Autonomous Surface Ship path planning, ocean current drift compensation, and sequential collision-risk avoidance using **Soft Actor-Critic (SAC)** with **Stable-Baselines3**.

---

## 🌟 Key Features

1. **Soft Actor-Critic (SAC) Reinforcement Learning**:
   - Built using Stable-Baselines3 `SAC("MlpPolicy", env)` for continuous control.
   - Real-time training directly from the Streamlit UI with progress metrics, episode reward curves, and loss tracking.
   - Model weights saved and loaded from `models/maritime_sac.zip`.

2. **Sequential Collision-Risk -> Avoidance Demonstration**:
   - **Scene 1**: Target vessel approaches close enough to cross the `COLLISION_WARNING_RADIUS` (75m), triggering a `COLLISION_RISK_DETECTED` warning banner.
   - **Scene 2**: SAC Agent executes continuous rudder steering and throttle speed adjustments to alter heading away from the target vessel (`AVOIDANCE_SUCCESSFUL`), then resumes navigation towards the destination goal.
   - Physical collision threshold (`COLLISION_RADIUS` at 35m) triggers terminal failure only on physical contact.

3. **Fixed Camera View & Clear Legend**:
   - Plotly 2D interactive map with dynamic aspect-ratio camera zoom (`scaleanchor="x", scaleratio=1`) ensuring all vessels, safety boundaries, ocean current field, and trajectories fit without cropping.
   - Distinct, color-coded legend identifying Own Vessel (SAC), Target Vessels, Goal Acceptance Circle, Collision Warning Zone, Actual Collision Boundary, and Avoidance Pathways.

---

## 🚀 Quick Start Guide

### Step 1: Open Project in VS Code
```bash
cd k:/Projects/Finalyear
```

### Step 2: Activate Virtual Environment
- **Windows PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
Finalyear/
├── app.py               # Streamlit Dashboard UI and sequential event handlers
├── maritime_env.py      # Gymnasium Maritime Ship Environment with SAC dynamics
├── sac_agent.py         # Stable-Baselines3 SAC Agent creation, training, & callback
├── maritime_viz.py      # Plotly 2D map renderer with camera zoom fix & visual legend
├── requirements.txt     # Python package dependencies
├── models/              # Directory for saved SAC model weights (maritime_sac.zip)
└── README.md            # Project documentation and setup guide
```

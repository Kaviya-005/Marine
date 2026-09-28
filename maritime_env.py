import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

class MaritimeEnv(gym.Env):
    """
    Maritime Autonomous Surface Ship (MASS) Environment for Soft Actor-Critic (SAC).
    Simulates Own Ship dynamics under ocean current influence, moving Target Ships,
    sequential collision-risk warning -> avoidance maneuvers, and goal navigation.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, config=None):
        super(MaritimeEnv, self).__init__()
        if config is None:
            config = {}

        # Map Dimensions (meters)
        self.map_width = float(config.get('map_width', 1000.0))
        self.map_height = float(config.get('map_height', 1000.0))
        self.max_diagonal = math.sqrt(self.map_width**2 + self.map_height**2)

        # Own Ship Initial Config
        self.own_ship_init_x = float(config.get('own_ship_init_x', 100.0))
        self.own_ship_init_y = float(config.get('own_ship_init_y', 100.0))
        self.own_ship_init_heading = float(config.get('own_ship_init_heading', 45.0)) # degrees
        self.own_ship_init_speed = float(config.get('own_ship_init_speed', 5.0)) # m/s
        self.own_ship_max_speed = float(config.get('own_ship_max_speed', 12.0)) # m/s
        self.own_ship_max_turn_rate = float(config.get('own_ship_max_turn_rate', 15.0)) # deg/s

        # Goal Config
        self.goal_x = float(config.get('goal_x', 900.0))
        self.goal_y = float(config.get('goal_y', 900.0))
        self.goal_radius = float(config.get('goal_radius', 40.0))

        # Ocean Current Config
        self.ocean_current_speed = float(config.get('ocean_current_speed', 1.5)) # m/s
        self.ocean_current_direction = float(config.get('ocean_current_direction', 90.0)) # deg (flow direction)

        # Simulation Parameters & Collision Thresholds
        # Hierarchy: SAFE_DISTANCE > COLLISION_WARNING_RADIUS > COLLISION_RADIUS
        self.max_steps = int(config.get('max_steps', 350))
        self.dt = float(config.get('dt', 1.0)) # 1 second per step
        self.safe_distance = float(config.get('safe_distance', 220.0)) # Outer safe margin
        self.collision_warning_radius = float(config.get('collision_warning_radius', 160.0)) # Risk warning threshold
        self.collision_radius = float(config.get('collision_radius', 40.0)) # Actual physical collision

        # Target Ships Config
        default_targets = [
            {'x': 260.0, 'y': 260.0, 'speed': 3.5, 'heading': 225.0},
            {'x': 650.0, 'y': 550.0, 'speed': 3.0, 'heading': 200.0},
            {'x': 300.0, 'y': 700.0, 'speed': 3.0, 'heading': 315.0}
        ]
        self.init_target_ships = config.get('target_ships', default_targets)
        self.max_target_ships = 5

        # Gymnasium Spaces definition
        self.observation_space_dim = 4 + 3 + 2 + (self.max_target_ships * 5)
        self.action_space_dim = 2 # continuous: [turn_cmd (-1..1), speed_cmd (-1..1)]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_space_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_dim,), dtype=np.float32)

        # Dynamic State Variables
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.own_x = self.own_ship_init_x
        self.own_y = self.own_ship_init_y
        self.own_heading = math.radians(self.own_ship_init_heading) # in radians
        self.own_speed = self.own_ship_init_speed
        self.in_warning_zone = False
        self.has_avoided = False

        # Setup Target Ships
        self.target_ships = []
        for t in self.init_target_ships[:self.max_target_ships]:
            self.target_ships.append({
                'x': float(t['x']),
                'y': float(t['y']),
                'heading': math.radians(float(t['heading'])),
                'speed': float(t['speed'])
            })

        # Precalculate current vectors
        curr_rad = math.radians(self.ocean_current_direction)
        self.u_curr = self.ocean_current_speed * math.cos(curr_rad)
        self.v_curr = self.ocean_current_speed * math.sin(curr_rad)

        self.prev_dist_to_goal = math.hypot(self.goal_x - self.own_x, self.goal_y - self.own_y)
        self.history = {
            'own_x': [self.own_x],
            'own_y': [self.own_y],
            'own_heading': [math.degrees(self.own_heading)],
            'own_speed': [self.own_speed],
            'target_positions': [[(ts['x'], ts['y']) for ts in self.target_ships]],
            'step_status': ['IN_PROGRESS']
        }

        obs = self._get_obs()
        info = self._get_info('IN_PROGRESS')
        return obs, info

    def _normalize_angle(self, angle):
        """Wrap angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _get_obs(self):
        obs = []

        # 1. Goal relative features
        dx_g = self.goal_x - self.own_x
        dy_g = self.goal_y - self.own_y
        dist_g = math.hypot(dx_g, dy_g)
        goal_bearing = math.atan2(dy_g, dx_g)
        bearing_diff = self._normalize_angle(goal_bearing - self.own_heading)

        obs.append(dx_g / self.map_width)
        obs.append(dy_g / self.map_height)
        obs.append(dist_g / self.max_diagonal)
        obs.append(bearing_diff / math.pi)

        # 2. Own ship state
        obs.append(self.own_speed / self.own_ship_max_speed)
        obs.append(math.cos(self.own_heading))
        obs.append(math.sin(self.own_heading))

        # 3. Ocean current relative
        max_curr = max(self.ocean_current_speed, 5.0)
        obs.append(self.u_curr / max_curr)
        obs.append(self.v_curr / max_curr)

        # 4. Target ships features
        for i in range(self.max_target_ships):
            if i < len(self.target_ships):
                ts = self.target_ships[i]
                dx_t = ts['x'] - self.own_x
                dy_t = ts['y'] - self.own_y
                dist_t = math.hypot(dx_t, dy_t)

                obs.append(dx_t / self.map_width)
                obs.append(dy_t / self.map_height)
                obs.append(dist_t / self.max_diagonal)
                obs.append(math.cos(ts['heading']))
                obs.append(math.sin(ts['heading']))
            else:
                obs.extend([1.0, 1.0, 1.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _get_info(self, status):
        min_target_dist = float('inf')
        for ts in self.target_ships:
            d = math.hypot(ts['x'] - self.own_x, ts['y'] - self.own_y)
            if d < min_target_dist:
                min_target_dist = d

        curr_dist_goal = math.hypot(self.goal_x - self.own_x, self.goal_y - self.own_y)

        return {
            'status': status,
            'step': self.current_step,
            'own_x': self.own_x,
            'own_y': self.own_y,
            'own_heading_deg': math.degrees(self.own_heading) % 360,
            'own_speed': self.own_speed,
            'distance_to_goal': curr_dist_goal,
            'min_target_distance': min_target_dist if min_target_dist != float('inf') else 9999.0,
            'history': self.history
        }

    def step(self, action):
        """
        action: array-like of shape (2,) with continuous values in [-1, 1]
        action[0]: turn command (-1 = full left, +1 = full right)
        action[1]: speed command (-1 = decelerate, +1 = accelerate)
        """
        self.current_step += 1
        turn_cmd = float(np.clip(action[0], -1.0, 1.0))
        speed_cmd = float(np.clip(action[1], -1.0, 1.0))

        # Update Heading (yaw rate in rad/s)
        max_turn_rad = math.radians(self.own_ship_max_turn_rate)
        yaw_rate = turn_cmd * max_turn_rad
        self.own_heading = self._normalize_angle(self.own_heading + yaw_rate * self.dt)

        # Update Speed
        speed_accel = speed_cmd * 1.0 # max acceleration 1.0 m/s^2
        self.own_speed = float(np.clip(self.own_speed + speed_accel * self.dt, 1.0, self.own_ship_max_speed))

        # Calculate Own Ship Ground Velocity (Ship Dynamics + Ocean Current Drift)
        vx_ship = self.own_speed * math.cos(self.own_heading) + self.u_curr
        vy_ship = self.own_speed * math.sin(self.own_heading) + self.v_curr

        # Update Own Ship Position
        self.own_x += vx_ship * self.dt
        self.own_y += vy_ship * self.dt

        # Update Target Ships Positions (Moving vessels)
        for ts in self.target_ships:
            vx_t = ts['speed'] * math.cos(ts['heading'])
            vy_t = ts['speed'] * math.sin(ts['heading'])
            ts['x'] += vx_t * self.dt
            ts['y'] += vy_t * self.dt

            # Boundary handling for target ships
            if ts['x'] < 0 or ts['x'] > self.map_width:
                ts['heading'] = self._normalize_angle(math.pi - ts['heading'])
                ts['x'] = float(np.clip(ts['x'], 0, self.map_width))
            if ts['y'] < 0 or ts['y'] > self.map_height:
                ts['heading'] = self._normalize_angle(-ts['heading'])
                ts['y'] = float(np.clip(ts['y'], 0, self.map_height))

        # Distances calculation
        curr_dist_to_goal = math.hypot(self.goal_x - self.own_x, self.goal_y - self.own_y)
        min_target_dist = float('inf')
        for ts in self.target_ships:
            dist_t = math.hypot(ts['x'] - self.own_x, ts['y'] - self.own_y)
            if dist_t < min_target_dist:
                min_target_dist = dist_t

        # -------------------------------------------------------------------
        # Sequential Collision-Risk -> Avoidance Demonstration Logic
        # -------------------------------------------------------------------
        terminated = False
        truncated = False
        status = 'IN_PROGRESS'

        # 1. Physical Collision Check (Inner Collision Radius)
        if min_target_dist <= self.collision_radius:
            terminated = True
            status = 'ACTUAL_COLLISION'

        # 2. Boundary Collision Check
        elif (self.own_x < 0 or self.own_x > self.map_width or
              self.own_y < 0 or self.own_y > self.map_height):
            terminated = True
            status = 'BOUNDARY_COLLISION'

        # 3. Goal Reached Check
        elif curr_dist_to_goal <= self.goal_radius:
            terminated = True
            status = 'GOAL_REACHED'

        # 4. Collision Warning Zone (Middle Collision Warning Radius)
        # DOES NOT TERMINATE SIMULATION! Enables SAC avoidance maneuver demonstration.
        elif min_target_dist <= self.collision_warning_radius:
            self.in_warning_zone = True
            status = 'COLLISION_RISK_DETECTED'

        # 5. Avoidance Successful Transition
        elif self.in_warning_zone and min_target_dist > self.collision_warning_radius:
            self.in_warning_zone = False
            self.has_avoided = True
            status = 'AVOIDANCE_SUCCESSFUL'

        # 6. Max Steps Truncation
        elif self.current_step >= self.max_steps:
            truncated = True
            status = 'MAX_STEPS'

        # Record Trajectory History
        self.history['own_x'].append(self.own_x)
        self.history['own_y'].append(self.own_y)
        self.history['own_heading'].append(math.degrees(self.own_heading))
        self.history['own_speed'].append(self.own_speed)
        self.history['target_positions'].append([(ts['x'], ts['y']) for ts in self.target_ships])
        self.history['step_status'].append(status)

        # -------------------------------------------------------------------
        # Reward Function for SAC Agent
        # -------------------------------------------------------------------
        reward = 0.0

        # Distance reduction reward towards goal
        dist_progress = self.prev_dist_to_goal - curr_dist_to_goal
        reward += dist_progress * 4.0
        self.prev_dist_to_goal = curr_dist_to_goal

        # Goal bearing alignment bonus
        dx_g = self.goal_x - self.own_x
        dy_g = self.goal_y - self.own_y
        goal_bearing = math.atan2(dy_g, dx_g)
        bearing_diff = abs(self._normalize_angle(goal_bearing - self.own_heading))
        reward += math.cos(bearing_diff) * 0.5

        # Proximity warning penalty
        if min_target_dist < self.collision_warning_radius:
            risk = 1.0 - (min_target_dist / self.collision_warning_radius)
            reward -= risk * 12.0

        # Avoidance bonus when escaping warning zone
        if status == 'AVOIDANCE_SUCCESSFUL':
            reward += 60.0

        # Terminal Rewards / Penalties
        if status == 'GOAL_REACHED':
            reward += 350.0
        elif status in ('ACTUAL_COLLISION', 'BOUNDARY_COLLISION'):
            reward -= 300.0

        # Small step penalty to encourage fastest safe path
        reward -= 0.1

        obs = self._get_obs()
        info = self._get_info(status)

        return obs, float(reward), terminated, truncated, info

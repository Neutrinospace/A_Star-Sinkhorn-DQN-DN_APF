import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import heapq
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import math
import os
import warnings
import itertools
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from datetime import datetime
import json
import copy

warnings.filterwarnings('ignore')
matplotlib.rc('font', family="SimSun")

# ==================== 参数设置 ====================
DEFAULT_SPACE_SIZE = (20, 20, 20)
DEFAULT_GRID_RESOLUTION = 1.0
DEFAULT_OBSTACLES = [((5,5,5),2), ((15,10,10),3), ((8,15,8),2)]
DEFAULT_N_DRONES = 4
DEFAULT_N_TARGETS = 4
MAX_STEPS_PER_EPISODE = 200
DT = 0.05
SAFE_DIST_DRONE = 2.0
SAFE_DIST_OBS = 3.0
MODEL_SAVE_PATH = "dqn_model.pth"
ANALYSIS_DIR = "analysis"

# ==================== 障碍物生成（贴地）====================
def generate_non_overlapping_obstacles(n, space_size, min_size=1.0, max_size=3.0, min_dist=1.0, max_attempts=1000):
    """
    生成底部在z=0的立方体障碍物（从xoy平面向上生长）
    """
    obstacles = []
    attempts = 0
    while len(obstacles) < n and attempts < max_attempts:
        half_size = random.uniform(min_size, max_size)
        # 强制中心z = half_size，使底部在z=0
        center = np.array([
            random.uniform(half_size, space_size[0] - half_size),
            random.uniform(half_size, space_size[1] - half_size),
            half_size
        ])
        overlap = False
        for (c, hs) in obstacles:
            if (abs(center[0] - c[0]) < half_size + hs and
                abs(center[1] - c[1]) < half_size + hs and
                abs(center[2] - c[2]) < half_size + hs):
                overlap = True
                break
        if not overlap:
            obstacles.append((tuple(center), half_size))
        attempts += 1
    if len(obstacles) < n:
        print(f"警告：仅生成 {len(obstacles)} 个不重叠障碍物，尝试 {max_attempts} 次后停止。")
    return obstacles

# ==================== 环境类 ====================
class Obstacle:
    def __init__(self, center, half_size):
        self.center = np.array(center)
        self.half_size = half_size

    def contains(self, point):
        return np.all(np.abs(point - self.center) <= self.half_size)

    def draw_boundary(self, ax, color='black', linewidth=2):
        c = self.center
        s = self.half_size
        vertices = [
            [c[0]-s, c[1]-s, c[2]-s],
            [c[0]+s, c[1]-s, c[2]-s],
            [c[0]+s, c[1]+s, c[2]-s],
            [c[0]-s, c[1]+s, c[2]-s],
            [c[0]-s, c[1]-s, c[2]+s],
            [c[0]+s, c[1]-s, c[2]+s],
            [c[0]+s, c[1]+s, c[2]+s],
            [c[0]-s, c[1]+s, c[2]+s]
        ]
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        for (i,j) in edges:
            ax.plot3D([vertices[i][0], vertices[j][0]],
                      [vertices[i][1], vertices[j][1]],
                      [vertices[i][2], vertices[j][2]],
                      color=color, linewidth=linewidth)

class Drone:
    def __init__(self, pos, goal=None):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.zeros(3)
        self.goal = goal
        self.path = []
        self.sub_goal_idx = 0
        self.color = np.random.rand(3)
        self.trail = []
        self.collision_count = 0
        self.q_value_history = []

class Environment:
    def __init__(self, space_size, obstacles):
        self.space_size = space_size
        self.obstacles = [Obstacle(c, h) for c, h in obstacles]
        self.drones = []
        self.targets = []

    def add_drone(self, pos):
        self.drones.append(Drone(pos))

    def set_targets(self, targets):
        self.targets = [np.array(t) for t in targets]

    def check_collision(self, pos):
        for obs in self.obstacles:
            if obs.contains(pos):
                return True
        if np.any(pos < 0) or np.any(pos > np.array(self.space_size)):
            return True
        return False

    def random_free_position(self, avoid_positions=None, min_dist=0.0, z_range=(0, 2.0)):
        """
        生成随机自由位置，z限制在z_range内（低空）
        """
        while True:
            pos = np.random.rand(3) * np.array(self.space_size)
            pos[2] = random.uniform(z_range[0], z_range[1])
            if not self.check_collision(pos):
                if avoid_positions is None:
                    return pos
                valid = True
                for p in avoid_positions:
                    if np.linalg.norm(pos - p) < min_dist:
                        valid = False
                        break
                if valid:
                    return pos

    def reset(self, n_drones, n_targets, min_dist=5.0, z_range=(0,2.0)):
        self.drones = []
        self.targets = []
        target_positions = []
        for _ in range(n_targets):
            pos = self.random_free_position(z_range=z_range)
            target_positions.append(pos)
        self.set_targets(target_positions)

        for i in range(n_drones):
            goal = random.choice(target_positions)
            pos = self.random_free_position(avoid_positions=[goal], min_dist=min_dist, z_range=z_range)
            self.add_drone(pos)
        return [d.pos for d in self.drones], target_positions

# ==================== A* 规划器 ====================
class AStarPlanner:
    def __init__(self, env, grid_res=1.0):
        self.env = env
        self.grid_res = grid_res
        self.grid_shape = tuple(max(1, int(s / grid_res)) for s in env.space_size)
        self.obstacle_grid = np.zeros(self.grid_shape, dtype=bool)
        for x in range(self.grid_shape[0]):
            for y in range(self.grid_shape[1]):
                for z in range(self.grid_shape[2]):
                    point = np.array([x, y, z]) * grid_res + grid_res/2
                    if env.check_collision(point):
                        self.obstacle_grid[x, y, z] = True

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b)) * self.grid_res

    def get_neighbors(self, node):
        x, y, z = node
        neighbors = []
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                for dz in [-1,0,1]:
                    if dx==0 and dy==0 and dz==0: continue
                    nx, ny, nz = x+dx, y+dy, z+dz
                    if 0<=nx<self.grid_shape[0] and 0<=ny<self.grid_shape[1] and 0<=nz<self.grid_shape[2]:
                        if not self.obstacle_grid[nx, ny, nz]:
                            neighbors.append((nx, ny, nz))
        return neighbors

    def plan(self, start, goal):
        start_grid = tuple(int(s / self.grid_res) for s in start)
        goal_grid = tuple(int(g / self.grid_res) for g in goal)
        start_grid = tuple(max(0, min(self.grid_shape[i]-1, start_grid[i])) for i in range(3))
        goal_grid = tuple(max(0, min(self.grid_shape[i]-1, goal_grid[i])) for i in range(3))
        if self.obstacle_grid[start_grid] or self.obstacle_grid[goal_grid]:
            return [start, goal]

        open_set = []
        heapq.heappush(open_set, (self.heuristic(start_grid, goal_grid), start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == goal_grid:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                path.reverse()
                world_path = [(p[0]*self.grid_res + self.grid_res/2,
                               p[1]*self.grid_res + self.grid_res/2,
                               p[2]*self.grid_res + self.grid_res/2) for p in path]
                return world_path

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal_grid)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return [start, goal]

# ==================== 人工势场控制器 ====================
class APFController:
    def __init__(self, env, k_att=3.0, k_rep_obs=150.0, k_rep_drone=150.0,
                 rep_range_obs=2.0, rep_range_drone=1.5, max_force=50.0):
        self.env = env
        self.k_att = k_att
        self.k_rep_obs = k_rep_obs
        self.k_rep_drone = k_rep_drone
        self.rep_range_obs = rep_range_obs
        self.rep_range_drone = rep_range_drone
        self.max_force = max_force

    def compute_force(self, drone, target, is_final_goal=False):
        pos = drone.pos
        dist_to_target = np.linalg.norm(pos - target)

        k_att_eff = self.k_att
        if is_final_goal and dist_to_target < 5.0:
            k_att_eff = self.k_att * (1.0 + 2.0 / (dist_to_target + 0.1))
        f_att = k_att_eff * (target - pos)

        f_rep_obs = np.zeros(3)
        for obs in self.env.obstacles:
            dir_vec = pos - obs.center
            dist = np.linalg.norm(dir_vec)
            if dist < obs.half_size + self.rep_range_obs and dist > 1e-3:
                magnitude = self.k_rep_obs * (1/(dist+1e-5) - 1/(obs.half_size+self.rep_range_obs)) / (dist**2 + 1e-5)
                magnitude = min(magnitude, self.max_force)
                f_rep_obs += magnitude * (dir_vec / dist)

        f_rep_drone = np.zeros(3)
        if not is_final_goal or dist_to_target > 2.0:
            for other in self.env.drones:
                if other is drone: continue
                diff = drone.pos - other.pos
                dist = np.linalg.norm(diff)
                if 0 < dist < self.rep_range_drone:
                    magnitude = self.k_rep_drone * (1/(dist+1e-5) - 1/self.rep_range_drone) / (dist**2 + 1e-5)
                    magnitude = min(magnitude, self.max_force)
                    f_rep_drone += magnitude * (diff / dist)

        force = f_att + f_rep_obs + f_rep_drone
        force_norm = np.linalg.norm(force)
        if force_norm > self.max_force:
            force = force / force_norm * self.max_force

        drone.vel += force * DT
        max_speed = 2.0
        if np.linalg.norm(drone.vel) > max_speed:
            drone.vel = drone.vel / np.linalg.norm(drone.vel) * max_speed
        drone.pos += drone.vel * DT

        for i in range(3):
            if drone.pos[i] < 0:
                drone.pos[i] = 0
                drone.vel[i] = 0
            elif drone.pos[i] > self.env.space_size[i]:
                drone.pos[i] = self.env.space_size[i]
                drone.vel[i] = 0
        return drone.pos

# ==================== DQN 网络与智能体 ====================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def forward(self, x):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, weight, bias)

class DuelingNoisyDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_value = NoisyLinear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        self.noisy_advantage = NoisyLinear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy1(x))
        v = F.relu(self.noisy_value(x))
        v = self.value(v)
        a = F.relu(self.noisy_advantage(x))
        a = self.advantage(a)
        q = v + (a - a.mean(dim=1, keepdim=True))
        return q

    def reset_noise(self):
        self.noisy1.weight_epsilon.normal_()
        self.noisy1.bias_epsilon.normal_()
        self.noisy_value.weight_epsilon.normal_()
        self.noisy_value.bias_epsilon.normal_()
        self.noisy_advantage.weight_epsilon.normal_()
        self.noisy_advantage.bias_epsilon.normal_()

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=5e-4, gamma=0.99, buffer_size=10000,
                 batch_size=64, target_update=100, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=500):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingNoisyDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingNoisyDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self.steps_done = 0
        self.loss_history = []
        self.reward_history = []
        self.episode_lengths = []
        self.step_rewards = []

        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state, eval_mode=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return q_values.argmax().item()

    def get_max_q(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.max().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        current_q = self.policy_net(states).gather(1, actions)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        self.steps_done += 1
        self.epsilon = max(self.epsilon_end, self.epsilon_start * math.exp(-1. * self.steps_done / self.epsilon_decay))

        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ==================== 改进的Sinkhorn分配器（支持供需向量、自适应松弛）====================
class SinkhornAllocator:
    def __init__(self, epsilon=0.5, max_iter=200, adaptive_epsilon=True):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.adaptive_epsilon = adaptive_epsilon

    def allocate(self, cost_matrix, supply=None, demand=None):
        """
        cost_matrix: (n_d, n_t) 形状
        supply: (n_d,) 每个无人机的供给（默认全1）
        demand: (n_t,) 每个目标的需求（默认全1）
        返回: assignments (n_d,) 每个无人机分配的目标索引
        """
        n_d, n_t = cost_matrix.shape
        if supply is None:
            supply = np.ones(n_d)
        if demand is None:
            demand = np.ones(n_t)

        # 保证供需平衡
        total_supply = np.sum(supply)
        total_demand = np.sum(demand)
        if abs(total_supply - total_demand) > 1e-6:
            # 调整需求，使总和相等
            demand = demand * (total_supply / total_demand)
            total_demand = total_supply

        K = np.exp(-cost_matrix / self.epsilon)
        K = np.maximum(K, 1e-12)

        u = np.ones(n_d)
        v = np.ones(n_t)

        for i in range(self.max_iter):
            u = supply / (K @ v + 1e-12)
            v = demand / (K.T @ u + 1e-12)
            if self.adaptive_epsilon and i % 10 == 0 and i > 0:
                self.epsilon *= 0.95
                K = np.exp(-cost_matrix / self.epsilon)
                K = np.maximum(K, 1e-12)

        P = np.diag(u) @ K @ np.diag(v)  # 软分配矩阵

        # 将软分配转为硬分配（每行取最大列）
        assignments = np.argmax(P, axis=1)
        # 确保每个目标至少被分配一次（处理 u>t 时可能某些目标无分配）
        for j in range(n_t):
            if j not in assignments:
                # 找到对该目标概率最大的行（未分配其他目标的行）
                candidates = np.argsort(P[:, j])[::-1]
                for i in candidates:
                    if assignments[i] != j:  # 可重新分配
                        assignments[i] = j
                        break
        return assignments

# ==================== 融合规划环境 ====================
class PathPlanningEnv:
    def __init__(self, env, planner, apf):
        self.env = env
        self.planner = planner
        self.apf = apf
        self.step_count = 0
        self.dones = []
        self.n_drones = 0
        self.reward_history = []
        self.collision_events = []
        self.episode_reward = 0

    def _get_state(self, drone_idx):
        drone = self.env.drones[drone_idx]
        if drone.sub_goal_idx < len(drone.path):
            sub_goal = np.array(drone.path[drone.sub_goal_idx])
            dist_to_subgoal = np.linalg.norm(drone.pos - sub_goal)
        else:
            dist_to_subgoal = 0.0
        min_obs_dist = float('inf')
        for obs in self.env.obstacles:
            dist = np.linalg.norm(drone.pos - obs.center) - obs.half_size
            if dist < min_obs_dist:
                min_obs_dist = dist
        min_drone_dist = float('inf')
        for other in self.env.drones:
            if other is drone: continue
            dist = np.linalg.norm(drone.pos - other.pos)
            if dist < min_drone_dist:
                min_drone_dist = dist
        speed = np.linalg.norm(drone.vel)
        state = [
            min(1.0, dist_to_subgoal/20.0),
            min(1.0, min_obs_dist/10.0),
            min(1.0, min_drone_dist/5.0),
            speed/2.0
        ]
        return np.array(state, dtype=np.float32)

    def step(self, actions):
        next_states = []
        rewards = []
        dones = []
        info = {}
        step_collisions = False

        for i, drone in enumerate(self.env.drones):
            if self.dones[i]:
                next_states.append(self._get_state(i))
                rewards.append(0)
                dones.append(True)
                continue

            if actions[i] == 0:
                self.apf.k_rep_obs = 60.0
                self.apf.k_rep_drone = 80.0
            elif actions[i] == 1:
                self.apf.k_rep_obs = 100.0
                self.apf.k_rep_drone = 120.0
            elif actions[i] == 2:
                self.apf.k_rep_obs = 150.0
                self.apf.k_rep_drone = 180.0

            if drone.sub_goal_idx < len(drone.path):
                sub_goal = drone.path[drone.sub_goal_idx]
                is_final = (drone.sub_goal_idx == len(drone.path)-1)
            else:
                sub_goal = drone.goal
                is_final = True

            old_pos = drone.pos.copy()
            prev_dist_goal = np.linalg.norm(old_pos - drone.goal)

            self.apf.compute_force(drone, np.array(sub_goal), is_final_goal=is_final)
            new_pos = drone.pos
            new_dist_goal = np.linalg.norm(new_pos - drone.goal)

            collision = self.env.check_collision(new_pos)
            drone_collision = False
            for other in self.env.drones:
                if other is drone: continue
                if np.linalg.norm(new_pos - other.pos) < SAFE_DIST_DRONE:
                    drone_collision = True
                    step_collisions = True
                    drone.collision_count += 1
                    other.collision_count += 1
                    break

            reward = 0.0
            dist_change = prev_dist_goal - new_dist_goal
            reward += dist_change * 20.0

            if drone.sub_goal_idx < len(drone.path) and np.linalg.norm(new_pos - sub_goal) < 1.0:
                reward += 30
                drone.sub_goal_idx += 1

            if not collision and new_dist_goal < 1.0:
                reward += 500
                self.dones[i] = True

            if collision:
                reward -= 300
                self.dones[i] = True
                drone.collision_count += 1
                step_collisions = True
            elif drone_collision:
                reward -= 50

            min_obs_dist = float('inf')
            for obs in self.env.obstacles:
                dist = np.linalg.norm(drone.pos - obs.center) - obs.half_size
                if dist < min_obs_dist:
                    min_obs_dist = dist
            if min_obs_dist < SAFE_DIST_OBS:
                reward -= 5.0 * (SAFE_DIST_OBS - min_obs_dist)

            reward -= 0.2

            rewards.append(reward)
            next_states.append(self._get_state(i))
            dones.append(self.dones[i])

        self.step_count += 1
        self.reward_history.append(rewards)
        self.collision_events.append(step_collisions)
        self.episode_reward += sum(rewards)

        if self.step_count >= MAX_STEPS_PER_EPISODE:
            for i in range(len(dones)):
                if not dones[i]:
                    dones[i] = True
                    rewards[i] -= 100
            self.episode_reward -= 100 * (len(dones) - sum(self.dones))

        self.dones = dones
        return next_states, rewards, dones, info

    # ------------------ 新增：融合代价矩阵构建 ------------------
    def simulate_pair(self, drone_pos, target_pos, steps=20):
        """
        模拟无人机从drone_pos到target_pos的短时运动，返回三个代价分量：
        - length: 模拟路径长度（近似）
        - risk: 最小距离倒数（碰撞风险）
        - control: 平均斥力大小（控制难度）
        """
        # 创建临时无人机和APF（使用当前环境，但复制障碍物）
        temp_drone = Drone(drone_pos)
        temp_drone.goal = target_pos
        # 临时APF使用默认参数（低斥力）
        temp_apf = APFController(self.env, k_rep_obs=60.0, k_rep_drone=80.0,
                                 rep_range_obs=2.0, rep_range_drone=1.5)
        path_length = 0.0
        min_dist = float('inf')
        total_rep = 0.0
        count = 0
        for _ in range(steps):
            # 计算力（只考虑吸引和斥力，不更新真实环境）
            force = temp_apf.compute_force(temp_drone, target_pos, is_final_goal=True)
            # 累计路径长度
            path_length += np.linalg.norm(temp_drone.vel * DT)
            # 记录与障碍物的最小距离
            for obs in self.env.obstacles:
                dist = np.linalg.norm(temp_drone.pos - obs.center) - obs.half_size
                if dist < min_dist:
                    min_dist = dist
            # 记录斥力大小
            total_rep += np.linalg.norm(force)
            count += 1
            # 提前终止若到达目标
            if np.linalg.norm(temp_drone.pos - target_pos) < 1.0:
                break
        risk = 1.0 / (min_dist + 0.1)  # 距离越小风险越大
        control = total_rep / (count + 1e-6)
        return path_length, risk, control

    def build_cost_matrix(self, agent, alpha=0.4, beta=0.3, gamma=0.3):
        """
        构建融合代价矩阵（α+β+γ=1）
        - α: A*路径长度归一化
        - β: DQN碰撞风险（用最大Q值的负值归一化）
        - γ: APF控制难度（模拟步数得到的平均斥力）
        返回: cost_matrix (n_d, n_t)
        """
        n_d = len(self.env.drones)
        n_t = len(self.env.targets)
        # 初始化三个分量矩阵
        L = np.zeros((n_d, n_t))
        Q = np.zeros((n_d, n_t))
        C = np.zeros((n_d, n_t))

        for i, drone in enumerate(self.env.drones):
            for j, target in enumerate(self.env.targets):
                # 1. A*路径长度
                path = self.planner.plan(drone.pos, target)
                length = 0.0
                for k in range(len(path)-1):
                    length += np.linalg.norm(np.array(path[k+1]) - np.array(path[k]))
                L[i,j] = length

                # 2. DQN碰撞风险：用最大Q值的负值（Q值大表示风险低，所以取负）
                # 构造包含目标相对位置的状态
                rel_target = target - drone.pos
                rel_target_norm = np.linalg.norm(rel_target)
                state_ext = [
                    min(1.0, rel_target_norm/20.0),   # 到目标距离
                    min(1.0, 0),                      # 障碍物距离占位
                    min(1.0, 0),                      # 其他无人机距离占位
                    0                                 # 速度占位
                ]
                q_val = agent.get_max_q(state_ext) if agent is not None else 0.5
                # 风险 = 1 - (q归一化) 或直接用 -q
                Q[i,j] = -q_val   # 因为q越大越容易，代价应越小

                # 3. APF控制难度（模拟20步）
                length_sim, risk_sim, control_sim = self.simulate_pair(drone.pos, target, steps=20)
                C[i,j] = control_sim

        # 归一化各分量（min-max归一化）
        L_norm = (L - L.min()) / (L.max() - L.min() + 1e-6)
        Q_norm = (Q - Q.min()) / (Q.max() - Q.min() + 1e-6)
        C_norm = (C - C.min()) / (C.max() - C.min() + 1e-6)

        # 加权融合
        cost_matrix = alpha * L_norm + beta * Q_norm + gamma * C_norm
        return cost_matrix

    def allocate_and_set_goals(self, agent, sinkhorn_allocator, alpha=0.4, beta=0.3, gamma=0.3):
        """
        构建代价矩阵，进行Sinkhorn分配，并为每个无人机设置目标
        """
        cost_matrix = self.build_cost_matrix(agent, alpha, beta, gamma)
        n_d = len(self.env.drones)
        n_t = len(self.env.targets)

        # 处理u≠t：当u>=t时，需求设为1 + (u-t)/t，使每个目标至少1，多余均匀分配
        if n_d >= n_t:
            supply = np.ones(n_d)
            demand = np.ones(n_t) + (n_d - n_t) / n_t
        else:
            # u < t：无法满足每个目标至少1，提示并让需求=1，部分目标无人
            print(f"警告：无人机数量({n_d})小于目标数量({n_t})，部分目标将无无人机覆盖。")
            supply = np.ones(n_d)
            demand = np.ones(n_t)

        assignments = sinkhorn_allocator.allocate(cost_matrix, supply, demand)
        for i, drone in enumerate(self.env.drones):
            drone.goal = self.env.targets[assignments[i]]
            drone.path = self.planner.plan(drone.pos, drone.goal)
            drone.sub_goal_idx = 1 if len(drone.path) > 1 else 0
            drone.vel = np.zeros(3)
            drone.trail = [drone.pos.copy()]
            drone.collision_count = 0
            drone.q_value_history = []
        return assignments

    def reset(self, n_drones, n_targets, agent, sinkhorn_allocator, alpha=0.4, beta=0.3, gamma=0.3):
        """
        重置环境并执行分配（后置）
        """
        drone_positions, target_positions = self.env.reset(n_drones, n_targets, min_dist=5.0, z_range=(0,2.0))
        self.n_drones = n_drones
        self.step_count = 0
        self.dones = [False]*n_drones
        self.reward_history = []
        self.collision_events = []
        self.episode_reward = 0
        # 执行分配
        self.allocate_and_set_goals(agent, sinkhorn_allocator, alpha, beta, gamma)
        return [self._get_state(i) for i in range(n_drones)]

# ==================== 科学分析函数（完整保留）====================
def generate_dqn_performance_analysis(drones, agent, save_dir, timestamp):
    if not drones or not agent:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('改进DQN性能分析', fontsize=16, fontweight='bold')
    colors = plt.cm.tab10.colors

    ax1 = axes[0, 0]
    has_q = False
    for i, drone in enumerate(drones):
        if drone.q_value_history and len(drone.q_value_history) > 1:
            steps = range(len(drone.q_value_history))
            ax1.plot(steps, drone.q_value_history, color=colors[i % len(colors)], label=f'无人机{i}', alpha=0.7)
            has_q = True
    if not has_q:
        ax1.text(0.5, 0.5, 'Q值数据不足', ha='center', va='center')
    ax1.set_title('Q值变化趋势')
    ax1.set_xlabel('步数')
    ax1.set_ylabel('最大Q值')
    if has_q:
        ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if agent.reward_history and len(agent.reward_history) > 1:
        ax2.plot(agent.reward_history, color='red', label='回合总奖励', linewidth=2)
        ax2.legend()
    elif agent.step_rewards and len(agent.step_rewards) > 1:
        ax2.plot(agent.step_rewards, color='red', label='步进总奖励', linewidth=2)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, '奖励数据不足', ha='center', va='center')
    ax2.set_title('训练奖励曲线')
    ax2.set_xlabel('回合/步数')
    ax2.set_ylabel('总奖励')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    drone_names = [f'无人机{i}' for i in range(len(drones))]
    collision_counts = [drone.collision_count for drone in drones]
    if any(collision_counts):
        n = len(drone_names)
        ax3.bar(range(n), collision_counts, color=colors[:n], alpha=0.7)
        ax3.set_xticks(range(n))
        ax3.set_xticklabels(drone_names[:n])
        for i, (bar, count) in enumerate(zip(ax3.patches, collision_counts)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count),
                    ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, '无碰撞数据', ha='center', va='center')
    ax3.set_title('碰撞次数统计')
    ax3.set_xlabel('无人机')
    ax3.set_ylabel('碰撞次数')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if agent.reward_history and len(agent.reward_history) > 1:
        cum_rewards = np.cumsum(agent.reward_history)
        ax4.plot(cum_rewards, color='green', linewidth=2)
    elif agent.step_rewards and len(agent.step_rewards) > 1:
        cum_rewards = np.cumsum(agent.step_rewards)
        ax4.plot(cum_rewards, color='green', linewidth=2)
    else:
        ax4.text(0.5, 0.5, '奖励数据不足', ha='center', va='center')
    ax4.set_title('累积奖励')
    ax4.set_xlabel('回合/步数')
    ax4.set_ylabel('累积奖励')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'dqn_performance_{timestamp}.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def generate_path_statistics_analysis(drones, obstacles, save_dir, timestamp):
    if not drones:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('无人机路径科学统计分析', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]
    path_lengths = []
    for drone in drones:
        if len(drone.trail) > 1:
            length = sum(np.linalg.norm(np.array(drone.trail[i+1]) - np.array(drone.trail[i]))
                        for i in range(len(drone.trail)-1))
            path_lengths.append(length)
    if path_lengths and len(path_lengths) > 1:
        ax1.hist(path_lengths, bins=min(10, len(path_lengths)), alpha=0.7, color='skyblue', edgecolor='black')
        stats_text = f'均值: {np.mean(path_lengths):.2f}\n标准差: {np.std(path_lengths):.2f}'
        ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, '路径长度数据不足', ha='center', va='center')
    ax1.set_title('路径长度分布')
    ax1.set_xlabel('路径长度')
    ax1.set_ylabel('频数')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    efficiencies = []
    for drone in drones:
        if len(drone.trail) > 1:
            start = drone.trail[0]
            end = drone.trail[-1]
            straight = np.linalg.norm(end - start)
            actual = sum(np.linalg.norm(np.array(drone.trail[i+1]) - np.array(drone.trail[i]))
                        for i in range(len(drone.trail)-1))
            eff = straight / actual if actual > 0 else 0
            efficiencies.append(eff)
    if efficiencies:
        n = len(efficiencies)
        ax2.bar(range(n), efficiencies, color='lightgreen', alpha=0.7)
        ax2.set_xticks(range(n))
        labels = [f'无人机{i}' for i in range(n)]
        ax2.set_xticklabels(labels[:n])
    else:
        ax2.text(0.5, 0.5, '效率数据不足', ha='center', va='center')
    ax2.set_title('路径效率分析')
    ax2.set_xlabel('无人机')
    ax2.set_ylabel('效率比')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    if obstacles:
        obs_array = np.array([obs.center for obs in obstacles])
        if len(obs_array) > 1:
            scatter = ax3.scatter(obs_array[:, 0], obs_array[:, 1], c=obs_array[:, 2],
                                 cmap='viridis', alpha=0.6)
            plt.colorbar(scatter, ax=ax3)
        else:
            ax3.text(0.5, 0.5, '障碍物数据不足', ha='center', va='center')
    else:
        ax3.text(0.5, 0.5, '无障碍物数据', ha='center', va='center')
    ax3.set_title('障碍物空间分布 (XY平面)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    curvatures = []
    for drone in drones:
        if len(drone.trail) >= 3:
            path_array = np.array(drone.trail)
            curvature = 0
            for i in range(1, len(path_array)-1):
                v1 = path_array[i] - path_array[i-1]
                v2 = path_array[i+1] - path_array[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    curvature += 1 - cos_angle
            curvatures.append(curvature / (len(path_array)-2))
    if curvatures and len(curvatures) > 1:
        ax4.boxplot(curvatures)
        ticks = ax4.get_xticks()
        n_ticks = len(ticks)
        labels = [f'无人机{i}' for i in range(min(len(curvatures), n_ticks))]
        ax4.set_xticklabels(labels)
    else:
        ax4.text(0.5, 0.5, '曲率数据不足', ha='center', va='center')
    ax4.set_title('路径曲率分布')
    ax4.set_ylabel('平均曲率')
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    has_convergence = False
    for i, drone in enumerate(drones):
        if len(drone.trail) > 1:
            dist_to_goal = [np.linalg.norm(pos - drone.goal) for pos in drone.trail]
            ax5.plot(dist_to_goal, label=f'无人机{i}', alpha=0.7)
            has_convergence = True
    if has_convergence:
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, '收敛数据不足', ha='center', va='center')
    ax5.set_title('路径收敛性分析')
    ax5.set_xlabel('步数')
    ax5.set_ylabel('到目标距离')
    ax5.grid(True, alpha=0.3)

    ax6 = axes[1, 2]
    all_points = []
    for drone in drones:
        all_points.extend(drone.trail)
    if len(all_points) >= 5:
        all_points = np.array(all_points)
        try:
            clustering = DBSCAN(eps=3, min_samples=2).fit(all_points)
            labels = clustering.labels_
            scatter = ax6.scatter(all_points[:, 0], all_points[:, 1], c=labels, cmap='tab10', alpha=0.6)
            plt.colorbar(scatter, ax=ax6)
        except Exception as e:
            ax6.text(0.5, 0.5, f'聚类错误: {str(e)[:20]}', ha='center', va='center')
    else:
        ax6.text(0.5, 0.5, '数据点不足 (需要≥5)', ha='center', va='center')
    ax6.set_title('路径点聚类分析')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f'path_statistics_{timestamp}.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def generate_path_deviation_analysis(drones, save_dir, timestamp):
    if not drones:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('无人机路径偏离度分析 (相对于起点终点直线)', fontsize=16, fontweight='bold')
    colors = plt.cm.tab10.colors

    all_deviations_data = {}
    for idx, drone in enumerate(drones):
        path = drone.trail
        if len(path) < 2:
            continue
        start = path[0]
        end = path[-1]
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-6:
            deviations = [0.0] * len(path)
        else:
            deviations = []
            for point in path:
                point_vec = point - start
                t = np.dot(point_vec, line_vec) / (line_len * line_len)
                t = max(0, min(1, t))
                projection = start + t * line_vec
                dist = np.linalg.norm(point - projection)
                deviations.append(dist)
        if len(deviations) > 1:
            all_deviations_data[idx] = deviations

    if not all_deviations_data:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, '偏离数据不足', ha='center', va='center')
        plt.tight_layout()
        path = os.path.join(save_dir, f'path_deviation_{timestamp}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    ax1 = axes[0, 0]
    for idx, deviations in all_deviations_data.items():
        ax1.plot(deviations, color=colors[idx % len(colors)], label=f'无人机{idx}', linewidth=2)
    ax1.set_title('每个路径点的垂直距离')
    ax1.set_xlabel('步数')
    ax1.set_ylabel('垂直距离')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for idx, deviations in all_deviations_data.items():
        cum_avg = np.cumsum(deviations) / np.arange(1, len(deviations)+1)
        ax2.plot(cum_avg, color=colors[idx % len(colors)], label=f'无人机{idx}', linewidth=2)
    ax2.set_title('累计平均垂直距离')
    ax2.set_xlabel('步数')
    ax2.set_ylabel('累计平均距离')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if all_deviations_data:
        max_len = max(len(dev) for dev in all_deviations_data.values())
        overall_avg = []
        overall_std = []
        for step in range(max_len):
            step_vals = [dev[step] for dev in all_deviations_data.values() if step < len(dev)]
            if step_vals:
                overall_avg.append(np.mean(step_vals))
                overall_std.append(np.std(step_vals))
            else:
                break
        if overall_avg:
            steps = range(len(overall_avg))
            ax3.plot(steps, overall_avg, color='red', linewidth=3, label='平均偏离')
            ax3.fill_between(steps,
                             np.array(overall_avg) - np.array(overall_std),
                             np.array(overall_avg) + np.array(overall_std),
                             color='red', alpha=0.2, label='标准差范围')
            stats_text = f'全局平均: {np.mean(overall_avg):.3f}\n标准差: {np.std(overall_avg):.3f}'
            ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax3.legend()
    ax3.set_title('所有无人机平均垂直距离')
    ax3.set_xlabel('步数')
    ax3.set_ylabel('平均垂直距离')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if all_deviations_data:
        matrix = []
        indices = sorted(all_deviations_data.keys())
        max_len = max(len(dev) for dev in all_deviations_data.values())
        for idx in indices:
            dev = all_deviations_data[idx]
            padded = dev + [np.nan] * (max_len - len(dev))
            matrix.append(padded)
        matrix = np.array(matrix)
        im = ax4.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        ax4.set_title('偏离度热力图')
        ax4.set_xlabel('步数')
        ax4.set_ylabel('无人机')
        yticks = range(len(indices))
        ax4.set_yticks(yticks)
        ylabels = [f'无人机{i}' for i in indices][:len(yticks)]
        ax4.set_yticklabels(ylabels)
        plt.colorbar(im, ax=ax4, label='垂直距离')
    else:
        ax4.text(0.5, 0.5, '无偏离数据', ha='center', va='center')

    plt.tight_layout()
    path = os.path.join(save_dir, f'path_deviation_{timestamp}.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def generate_astar_deviation_analysis(drones, planner, save_dir, timestamp):
    if not drones:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('无人机与A*路径偏离度分析', fontsize=16, fontweight='bold')
    colors = plt.cm.tab10.colors

    all_deviations = {}

    for idx, drone in enumerate(drones):
        if len(drone.trail) < 2 or len(drone.path) < 2:
            continue
        astar_path = np.array(drone.path)
        deviations = []
        for point in drone.trail:
            min_dist = float('inf')
            for i in range(len(astar_path)-1):
                seg_start = astar_path[i]
                seg_end = astar_path[i+1]
                seg_vec = seg_end - seg_start
                seg_len = np.linalg.norm(seg_vec)
                if seg_len < 1e-6:
                    dist = np.linalg.norm(point - seg_start)
                else:
                    t = np.dot(point - seg_start, seg_vec) / (seg_len * seg_len)
                    t = max(0, min(1, t))
                    proj = seg_start + t * seg_vec
                    dist = np.linalg.norm(point - proj)
                if dist < min_dist:
                    min_dist = dist
            deviations.append(min_dist)
        if len(deviations) > 1:
            all_deviations[idx] = deviations

    if not all_deviations:
        for ax in axes.flatten():
            ax.text(0.5, 0.5, '偏离数据不足', ha='center', va='center')
        plt.tight_layout()
        path = os.path.join(save_dir, f'astar_deviation_{timestamp}.png')
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    ax1 = axes[0, 0]
    for idx, dev in all_deviations.items():
        ax1.plot(dev, color=colors[idx % len(colors)], label=f'无人机{idx}', linewidth=2)
    ax1.set_title('每个轨迹点与A*路径的距离')
    ax1.set_xlabel('步数')
    ax1.set_ylabel('距离')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    for idx, dev in all_deviations.items():
        cum_avg = np.cumsum(dev) / np.arange(1, len(dev)+1)
        ax2.plot(cum_avg, color=colors[idx % len(colors)], label=f'无人机{idx}', linewidth=2)
    ax2.set_title('累计平均距离')
    ax2.set_xlabel('步数')
    ax2.set_ylabel('累计平均距离')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    if all_deviations:
        max_len = max(len(dev) for dev in all_deviations.values())
        overall_avg = []
        overall_std = []
        for step in range(max_len):
            step_vals = [dev[step] for dev in all_deviations.values() if step < len(dev)]
            if step_vals:
                overall_avg.append(np.mean(step_vals))
                overall_std.append(np.std(step_vals))
            else:
                break
        if overall_avg:
            steps = range(len(overall_avg))
            ax3.plot(steps, overall_avg, color='red', linewidth=3, label='平均偏离')
            ax3.fill_between(steps,
                             np.array(overall_avg) - np.array(overall_std),
                             np.array(overall_avg) + np.array(overall_std),
                             color='red', alpha=0.2, label='标准差范围')
            stats_text = f'全局平均: {np.mean(overall_avg):.3f}\n标准差: {np.std(overall_avg):.3f}'
            ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax3.legend()
    ax3.set_title('所有无人机平均偏离A*路径')
    ax3.set_xlabel('步数')
    ax3.set_ylabel('平均距离')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if all_deviations:
        matrix = []
        indices = sorted(all_deviations.keys())
        max_len = max(len(dev) for dev in all_deviations.values())
        for idx in indices:
            dev = all_deviations[idx]
            padded = dev + [np.nan] * (max_len - len(dev))
            matrix.append(padded)
        matrix = np.array(matrix)
        im = ax4.imshow(matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        ax4.set_title('偏离度热力图 (相对于A*路径)')
        ax4.set_xlabel('步数')
        ax4.set_ylabel('无人机')
        yticks = range(len(indices))
        ax4.set_yticks(yticks)
        ylabels = [f'无人机{i}' for i in indices][:len(yticks)]
        ax4.set_yticklabels(ylabels)
        plt.colorbar(im, ax=ax4, label='距离')
    else:
        ax4.text(0.5, 0.5, '无偏离数据', ha='center', va='center')

    plt.tight_layout()
    path = os.path.join(save_dir, f'astar_deviation_{timestamp}.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path

def generate_performance_text_report(env, plan_env, agent, save_dir, timestamp):
    path_lengths = []
    for drone in env.drones:
        if len(drone.trail) > 1:
            length = sum(np.linalg.norm(np.array(drone.trail[i+1]) - np.array(drone.trail[i]))
                        for i in range(len(drone.trail)-1))
            path_lengths.append(length)
    avg_path_length = np.mean(path_lengths) if path_lengths else 0
    success_rate = sum(plan_env.dones) / len(env.drones) if env.drones else 0
    total_collisions = sum(d.collision_count for d in env.drones)

    all_q = [q for d in env.drones for q in d.q_value_history]
    avg_q = np.mean(all_q) if all_q else 0
    avg_loss = np.mean(agent.loss_history) if agent.loss_history else 0

    stats = f"""===== 路径规划统计报告 =====
生成时间: {timestamp}
无人机数量: {len(env.drones)}
目标点数量: {len(env.targets)}
障碍物数量: {len(env.obstacles)}
总步数: {plan_env.step_count}
成功到达目标: {sum(plan_env.dones)} / {len(env.drones)}
平均路径长度: {avg_path_length:.2f}
总碰撞次数: {total_collisions}
平均Q值: {avg_q:.3f}
平均损失: {avg_loss:.4f}
"""
    report_path = os.path.join(save_dir, f'report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write(stats)
    return stats, report_path

def generate_academic_report(env, plan_env, agent, save_dir=ANALYSIS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 路径对比图
    try:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, env.space_size[0])
        ax.set_ylim(0, env.space_size[1])
        ax.set_zlim(0, env.space_size[2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('A*规划路径与实际轨迹对比')

        for obs in env.obstacles:
            obs.draw_boundary(ax, color='black', linewidth=2)

        for i, target in enumerate(env.targets):
            ax.scatter(*target, color='green', marker='*', s=200, label='目标点' if i==0 else "")

        for i, drone in enumerate(env.drones):
            if len(drone.path) > 1:
                path = np.array(drone.path)
                ax.plot(path[:,0], path[:,1], path[:,2], 'b--', linewidth=1.5, alpha=0.7, label='A*路径' if i==0 else "")

        colors = ['r', 'g', 'c', 'm', 'y', 'orange', 'purple']
        for i, drone in enumerate(env.drones):
            trail = np.array(drone.trail)
            if len(trail) > 1:
                ax.plot(trail[:,0], trail[:,1], trail[:,2], color=colors[i%len(colors)], linewidth=2, label=f'无人机{i}')

        ax.legend()
        plt.tight_layout()
        path_compare_file = os.path.join(save_dir, f'path_comparison_{timestamp}.png')
        plt.savefig(path_compare_file, dpi=300)
        plt.close()
    except Exception as e:
        print(f"路径对比图生成失败: {e}")
        path_compare_file = None

    # 2. 训练收敛曲线
    convergence_file = None
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if agent.step_rewards and len(agent.step_rewards) > 1:
            axes[0].plot(agent.step_rewards, color='red')
            axes[0].set_xlabel('步数')
            axes[0].set_ylabel('步总奖励')
            axes[0].set_title('单次仿真步进奖励')
        elif agent.reward_history and len(agent.reward_history) > 1:
            axes[0].plot(agent.reward_history, color='red')
            axes[0].set_xlabel('回合')
            axes[0].set_ylabel('回合总奖励')
            axes[0].set_title('训练奖励曲线')
        else:
            axes[0].text(0.5, 0.5, '奖励数据不足', ha='center', va='center')
        axes[0].grid(True)

        if agent.loss_history and len(agent.loss_history) > 1:
            axes[1].plot(agent.loss_history, color='red')
            axes[1].set_xlabel('更新步数')
            axes[1].set_ylabel('损失')
            axes[1].set_title('损失下降曲线')
        else:
            axes[1].text(0.5, 0.5, '损失数据不足\n(需多回合训练)', ha='center', va='center')
        axes[1].grid(True)

        if agent.episode_lengths and len(agent.episode_lengths) > 1:
            axes[2].plot(agent.episode_lengths, color='red')
            axes[2].set_xlabel('回合')
            axes[2].set_ylabel('路径长度 (步数)')
            axes[2].set_title('路径长度收敛')
        else:
            axes[2].text(0.5, 0.5, '回合长度数据不足', ha='center', va='center')
        axes[2].grid(True)

        plt.tight_layout()
        convergence_file = os.path.join(save_dir, f'convergence_{timestamp}.png')
        plt.savefig(convergence_file, dpi=300)
        plt.close()
    except Exception as e:
        print(f"收敛曲线生成失败: {e}")

    # 3. 性能指标柱状图
    bars_file = None
    try:
        path_lengths = []
        for drone in env.drones:
            if len(drone.trail) > 1:
                length = 0
                for i in range(1, len(drone.trail)):
                    length += np.linalg.norm(drone.trail[i] - drone.trail[i-1])
                path_lengths.append(length)
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        success_rate = sum(plan_env.dones) / len(env.drones) if env.drones else 0
        total_collisions = sum(d.collision_count for d in env.drones)

        metrics = {
            '改进DQN': {
                '成功率': success_rate,
                '平均路径长度': avg_path_length,
                '碰撞次数': total_collisions
            }
        }
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        x = np.arange(len(metrics))
        tick_labels = list(metrics.keys())
        for i, (name, data) in enumerate(metrics.items()):
            axes[0].bar(i, data['成功率'], label=name)
            axes[1].bar(i, data['平均路径长度'], label=name)
            axes[2].bar(i, data['碰撞次数'], label=name)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(tick_labels[:len(x)])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(tick_labels[:len(x)])
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(tick_labels[:len(x)])
        axes[0].set_ylabel('成功率')
        axes[0].set_title('成功率')
        axes[1].set_ylabel('平均路径长度')
        axes[1].set_title('路径长度')
        axes[2].set_ylabel('碰撞次数')
        axes[2].set_title('碰撞次数')
        plt.tight_layout()
        bars_file = os.path.join(save_dir, f'performance_bars_{timestamp}.png')
        plt.savefig(bars_file, dpi=300)
        plt.close()
    except Exception as e:
        print(f"柱状图生成失败: {e}")

    # 4. Q值热力图（所有无人机）
    heatmap_file = None
    if env.drones:
        try:
            n_drones = len(env.drones)
            cols = min(3, n_drones)
            rows = (n_drones + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            for idx, drone in enumerate(env.drones):
                if idx >= len(axes):
                    break
                ax = axes[idx]
                if len(drone.trail) < 5:
                    ax.text(0.5, 0.5, f'无人机{idx}轨迹点不足', ha='center', va='center')
                    continue
                q_values = []
                positions = []
                for pos in drone.trail:
                    dist_to_goal = np.linalg.norm(pos - drone.goal)
                    state = [min(1.0, dist_to_goal/20.0), 0, 0, 0]
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                    with torch.no_grad():
                        q = agent.policy_net(state_t).max().item()
                    q_values.append(q)
                    positions.append(pos[:2])
                positions = np.array(positions)
                q_values = np.array(q_values)
                xi = np.linspace(0, env.space_size[0], 50)
                yi = np.linspace(0, env.space_size[1], 50)
                xi, yi = np.meshgrid(xi, yi)
                zi = griddata(positions, q_values, (xi, yi), method='cubic', fill_value=0)
                cf = ax.contourf(xi, yi, zi, levels=20, cmap='viridis')
                ax.scatter(positions[:,0], positions[:,1], c='red', s=10)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'无人机{idx} Q值热力图')
                plt.colorbar(cf, ax=ax, label='最大Q值')
            for j in range(len(env.drones), len(axes)):
                axes[j].set_visible(False)
            plt.tight_layout()
            heatmap_file = os.path.join(save_dir, f'q_heatmaps_all_{timestamp}.png')
            plt.savefig(heatmap_file, dpi=300)
            plt.close()
        except Exception as e:
            print(f"热力图生成失败: {e}")

    # 5. DQN性能分析
    try:
        dqn_perf_file = generate_dqn_performance_analysis(env.drones, agent, save_dir, timestamp)
    except Exception as e:
        print(f"DQN性能分析生成失败: {e}")
        dqn_perf_file = None

    # 6. 路径统计分析
    try:
        path_stats_file = generate_path_statistics_analysis(env.drones, env.obstacles, save_dir, timestamp)
    except Exception as e:
        print(f"路径统计分析生成失败: {e}")
        path_stats_file = None

    # 7. 路径偏离度分析（相对于直线）
    try:
        path_dev_file = generate_path_deviation_analysis(env.drones, save_dir, timestamp)
    except Exception as e:
        print(f"路径偏离度分析生成失败: {e}")
        path_dev_file = None

    # 8. 与A*路径偏离度分析
    try:
        astar_dev_file = generate_astar_deviation_analysis(env.drones, plan_env.planner, save_dir, timestamp)
    except Exception as e:
        print(f"A*路径偏离度分析生成失败: {e}")
        astar_dev_file = None

    # 9. 文本报告
    stats_text, report_path = generate_performance_text_report(env, plan_env, agent, save_dir, timestamp)

    summary = f"""学术报告生成完成，包含以下文件：
1. 路径对比图: {os.path.basename(path_compare_file) if path_compare_file else '失败'}
2. 训练收敛曲线: {os.path.basename(convergence_file) if convergence_file else '失败/无数据'}
3. 性能柱状图: {os.path.basename(bars_file) if bars_file else '失败'}
4. Q值热力图(所有无人机): {os.path.basename(heatmap_file) if heatmap_file else '失败/无数据'}
5. DQN性能分析: {os.path.basename(dqn_perf_file) if dqn_perf_file else '失败/无数据'}
6. 路径统计分析: {os.path.basename(path_stats_file) if path_stats_file else '失败/无数据'}
7. 路径偏离度分析(相对于直线): {os.path.basename(path_dev_file) if path_dev_file else '失败/无数据'}
8. 与A*路径偏离度分析: {os.path.basename(astar_dev_file) if astar_dev_file else '失败/无数据'}
9. 文本报告: {os.path.basename(report_path)}
保存在目录: {save_dir}
"""
    print(summary)
    return stats_text

# ==================== GUI 应用程序 ====================
class DronePathPlanningGUI:
    def __init__(self, master):
        self.master = master
        master.title("无人机集群路径规划 - 融合算法版")
        master.geometry("1400x850")

        self.space_size = DEFAULT_SPACE_SIZE
        self.n_drones = tk.IntVar(value=DEFAULT_N_DRONES)
        self.n_targets = tk.IntVar(value=DEFAULT_N_TARGETS)
        self.obstacle_density = tk.DoubleVar(value=0.3)
        self.min_dist = tk.DoubleVar(value=5.0)
        self.train_episodes = tk.IntVar(value=50)
        self.alpha = tk.DoubleVar(value=0.4)   # A*代价权重
        self.beta = tk.DoubleVar(value=0.3)    # DQN风险权重
        self.gamma = tk.DoubleVar(value=0.3)   # APF难度权重
        self.custom_obstacles = []

        self.env = None
        self.planner = None
        self.sinkhorn = None
        self.apf = None
        self.plan_env = None
        self.agent = None
        self.running = False
        self.training = False
        self.animation_after_id = None
        self.episode_reward = 0

        self.create_widgets()
        self.init_environment()

    def create_widgets(self):
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=320)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        control_frame.pack_propagate(False)

        row = 0
        ttk.Label(control_frame, text="无人机数量:").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(control_frame, textvariable=self.n_drones, width=10).grid(row=row, column=1, pady=2)
        row += 1

        ttk.Label(control_frame, text="目标点数量:").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(control_frame, textvariable=self.n_targets, width=10).grid(row=row, column=1, pady=2)
        row += 1

        ttk.Label(control_frame, text="最小起点-目标距离:").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(control_frame, textvariable=self.min_dist, width=10).grid(row=row, column=1, pady=2)
        row += 1

        ttk.Label(control_frame, text="障碍物密集度 (0-1):").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(control_frame, textvariable=self.obstacle_density, width=10).grid(row=row, column=1, pady=2)
        row += 1

        ttk.Label(control_frame, text="训练回合数:").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(control_frame, textvariable=self.train_episodes, width=10).grid(row=row, column=1, pady=2)
        row += 1

        ttk.Label(control_frame, text="α (A*路径):").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(control_frame, textvariable=self.alpha, width=10).grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(control_frame, text="β (DQN风险):").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(control_frame, textvariable=self.beta, width=10).grid(row=row, column=1, pady=2)
        row += 1
        ttk.Label(control_frame, text="γ (APF难度):").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(control_frame, textvariable=self.gamma, width=10).grid(row=row, column=1, pady=2)
        row += 1

        ttk.Button(control_frame, text="随机生成", command=self.random_generate).grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        ttk.Button(control_frame, text="开始仿真", command=self.start_simulation).grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        ttk.Button(control_frame, text="开始训练", command=self.start_training).grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        ttk.Button(control_frame, text="停止", command=self.stop).grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1
        ttk.Button(control_frame, text="生成学术报告", command=self.generate_report).grid(row=row, column=0, columnspan=2, pady=5, sticky='ew')
        row += 1

        ttk.Button(control_frame, text="导出配置", command=self.export_config).grid(row=row, column=0, pady=5, sticky='ew')
        ttk.Button(control_frame, text="导入配置", command=self.import_config).grid(row=row, column=1, pady=5, sticky='ew')
        row += 1

        ttk.Button(control_frame, text="保存模型", command=self.save_model).grid(row=row, column=0, pady=5, sticky='ew')
        ttk.Button(control_frame, text="加载模型", command=self.load_model).grid(row=row, column=1, pady=5, sticky='ew')
        row += 1

        ttk.Button(control_frame, text="保存训练数据", command=self.save_training_data).grid(row=row, column=0, pady=5, sticky='ew')
        ttk.Button(control_frame, text="加载训练数据", command=self.load_training_data).grid(row=row, column=1, pady=5, sticky='ew')
        row += 1

        self.status_label = ttk.Label(control_frame, text="状态: 就绪", relief=tk.SUNKEN)
        self.status_label.grid(row=row, column=0, columnspan=2, pady=10, sticky='ew')

        plot_frame = ttk.LabelFrame(main_frame, text="三维可视化")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.drone_scatter = None
        self.target_scatter = None
        self.trail_lines = []
        self.path_lines = []

    def init_environment(self):
        obstacles = DEFAULT_OBSTACLES + self.custom_obstacles
        self.env = Environment(self.space_size, obstacles)
        self.planner = AStarPlanner(self.env, grid_res=DEFAULT_GRID_RESOLUTION)
        self.sinkhorn = SinkhornAllocator(epsilon=0.5, max_iter=50, adaptive_epsilon=True)
        self.apf = APFController(self.env, k_att=3.0, k_rep_obs=150.0, k_rep_drone=150.0,
                                  rep_range_obs=2.0, rep_range_drone=1.5)
        self.plan_env = PathPlanningEnv(self.env, self.planner, self.apf)

        state_dim = 4
        action_dim = 3
        if self.agent is None:
            self.agent = DQNAgent(state_dim, action_dim)

    def random_generate(self):
        n_d = self.n_drones.get()
        n_t = self.n_targets.get()
        if n_d < n_t:
            messagebox.showwarning("警告", f"无人机数量({n_d})小于目标数量({n_t})，将导致部分目标无无人机覆盖。")
        density = self.obstacle_density.get()
        min_d = self.min_dist.get()

        # 生成不重叠的障碍物（贴地）
        n_extra = int(density * 20)
        extra_obstacles = generate_non_overlapping_obstacles(
            n_extra, self.space_size, min_size=1.0, max_size=3.0, min_dist=1.0, max_attempts=2000
        )
        self.custom_obstacles = extra_obstacles

        self.init_environment()
        # 重置环境但不分配（分配将在后续由reset调用完成）
        self.env.reset(n_d, n_t, min_dist=min_d, z_range=(0,2.0))
        # 手动调用分配（因为reset后需要分配）
        self.plan_env.allocate_and_set_goals(self.agent, self.sinkhorn,
                                             alpha=self.alpha.get(),
                                             beta=self.beta.get(),
                                             gamma=self.gamma.get())
        self.update_plot()
        self.status_label.config(text=f"状态: 已生成 {n_d} 无人机, {n_t} 目标, {len(self.env.obstacles)} 障碍物")

    def update_plot(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.space_size[0])
        self.ax.set_ylim(0, self.space_size[1])
        self.ax.set_zlim(0, self.space_size[2])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('无人机路径规划 (蓝色虚线为A*规划路径)')

        for obs in self.env.obstacles:
            obs.draw_boundary(self.ax, color='black', linewidth=2)

        if self.env.targets:
            targets = np.array(self.env.targets)
            self.target_scatter = self.ax.scatter(targets[:,0], targets[:,1], targets[:,2],
                                                  c='green', marker='*', s=200, label='目标点')

        if self.env.drones:
            positions = np.array([d.pos for d in self.env.drones])
            colors = [d.color for d in self.env.drones]
            self.drone_scatter = self.ax.scatter(positions[:,0], positions[:,1], positions[:,2],
                                                 c=colors, s=80, label='无人机')

        self.path_lines = []
        for drone in self.env.drones:
            if len(drone.path) > 1:
                path = np.array(drone.path)
                line, = self.ax.plot(path[:,0], path[:,1], path[:,2], 'b--', linewidth=1, alpha=0.6)
                self.path_lines.append(line)

        self.trail_lines = []
        for drone in self.env.drones:
            line, = self.ax.plot([], [], [], c=drone.color, alpha=0.7, linewidth=2)
            self.trail_lines.append(line)

        self.ax.legend(loc='upper left')
        self.canvas.draw()

    def animation_step(self):
        if not self.running:
            return

        if self.drone_scatter is None:
            self.update_plot()

        if all(self.plan_env.dones) or self.plan_env.step_count >= MAX_STEPS_PER_EPISODE:
            self.running = False
            if self.episode_reward != 0:
                self.agent.reward_history.append(self.episode_reward)
                self.agent.episode_lengths.append(self.plan_env.step_count)
            self.episode_reward = 0
            self.status_label.config(text="状态: 回合结束")
            if self.training:
                self.master.after(500, self.run_training_episode)
            return

        states = [self.plan_env._get_state(i) for i in range(len(self.env.drones))]
        eval_mode = not self.training
        actions = [self.agent.select_action(s, eval_mode=eval_mode) for s in states]

        for i, drone in enumerate(self.env.drones):
            q = self.agent.get_max_q(states[i])
            drone.q_value_history.append(q)

        next_states, rewards, dones, info = self.plan_env.step(actions)
        self.episode_reward += sum(rewards)

        if self.training:
            for i in range(len(self.env.drones)):
                if not self.plan_env.dones[i]:
                    self.agent.store_transition(states[i], actions[i], rewards[i], next_states[i], dones[i])
            loss = self.agent.update()
            if loss > 0:
                self.agent.loss_history.append(loss)

        positions = np.array([d.pos for d in self.env.drones])
        if self.drone_scatter is not None:
            self.drone_scatter._offsets3d = (positions[:,0], positions[:,1], positions[:,2])
        else:
            self.update_plot()

        for i, drone in enumerate(self.env.drones):
            drone.trail.append(drone.pos.copy())
            if len(drone.trail) > 100:
                drone.trail.pop(0)
            trail = np.array(drone.trail)
            if len(trail) > 1 and i < len(self.trail_lines):
                self.trail_lines[i].set_data(trail[:,0], trail[:,1])
                self.trail_lines[i].set_3d_properties(trail[:,2])

        self.ax.set_title(f'步数: {self.plan_env.step_count}/{MAX_STEPS_PER_EPISODE}  ε: {self.agent.epsilon:.3f}')
        self.canvas.draw()

        self.animation_after_id = self.master.after(50, self.animation_step)

    def start_simulation(self):
        if self.running or self.training:
            return
        self.running = True
        self.training = False
        self.episode_reward = 0
        if not self.env.drones:
            self.random_generate()
        self.animation_step()

    def start_training(self):
        if self.running or self.training:
            return
        self.training = True
        self.current_episode = 0
        self.total_episodes = self.train_episodes.get()
        self.status_label.config(text=f"状态: 训练开始，共{self.total_episodes}回合")
        self.agent.memory.clear()
        self.agent.loss_history.clear()
        self.agent.reward_history.clear()
        self.agent.episode_lengths.clear()
        self.agent.step_rewards.clear()
        self.run_training_episode()

    def run_training_episode(self):
        if not self.training or self.current_episode >= self.total_episodes:
            self.training = False
            self.running = False
            self.status_label.config(text="状态: 训练完成")
            return
        self.current_episode += 1
        self.status_label.config(text=f"状态: 训练回合 {self.current_episode}/{self.total_episodes}")
        n_d = self.n_drones.get()
        n_t = self.n_targets.get()
        # 重置并分配（使用当前DQN）
        self.plan_env.reset(n_d, n_t, self.agent, self.sinkhorn,
                            alpha=self.alpha.get(), beta=self.beta.get(), gamma=self.gamma.get())
        self.episode_reward = 0
        self.update_plot()
        self.running = True
        self.animation_step()

    def stop(self):
        self.running = False
        self.training = False
        if self.animation_after_id:
            self.master.after_cancel(self.animation_after_id)
            self.animation_after_id = None
        self.status_label.config(text="状态: 已停止")

    def export_config(self):
        """导出当前配置到JSON文件"""
        if not self.env.drones or not self.env.targets:
            messagebox.showwarning("警告", "没有可导出的配置（请先生成场景）")
            return
        config = {
            "space_size": self.space_size,
            "obstacles": [(obs.center.tolist(), obs.half_size) for obs in self.env.obstacles],
            "drones": [drone.pos.tolist() for drone in self.env.drones],
            "targets": [target.tolist() for target in self.env.targets]
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=4)
                messagebox.showinfo("成功", f"配置已导出到 {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {e}")

    def import_config(self):
        """从JSON文件导入配置并重建场景"""
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            required = ["space_size", "obstacles", "drones", "targets"]
            if not all(k in config for k in required):
                raise ValueError("配置文件缺少必要字段")
            self.space_size = tuple(config["space_size"])
            self.custom_obstacles = [(tuple(center), half_size) for center, half_size in config["obstacles"]]
            self.init_environment()
            self.env.drones = []
            for pos in config["drones"]:
                self.env.add_drone(pos)
            self.env.set_targets(config["targets"])
            # 重新分配目标
            self.plan_env.allocate_and_set_goals(self.agent, self.sinkhorn,
                                                 alpha=self.alpha.get(),
                                                 beta=self.beta.get(),
                                                 gamma=self.gamma.get())
            self.n_drones.set(len(self.env.drones))
            self.n_targets.set(len(self.env.targets))
            self.update_plot()
            self.status_label.config(text=f"状态: 已导入配置 (无人机{len(self.env.drones)}, 目标{len(self.env.targets)}, 障碍物{len(self.env.obstacles)})")
            messagebox.showinfo("成功", "配置导入成功")
        except Exception as e:
            messagebox.showerror("错误", f"导入失败: {e}")

    def save_model(self):
        if self.agent is None:
            messagebox.showwarning("警告", "没有可保存的模型")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pth",
                                                 filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if file_path:
            try:
                self.agent.save(file_path)
                messagebox.showinfo("成功", f"模型已保存到 {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存模型失败: {e}")

    def load_model(self):
        if self.agent is None:
            self.init_environment()
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if file_path:
            try:
                self.agent.load(file_path)
                messagebox.showinfo("成功", f"模型已从 {file_path} 加载")
            except Exception as e:
                messagebox.showerror("错误", f"加载模型失败: {e}")

    def save_training_data(self):
        if self.agent is None:
            messagebox.showwarning("警告", "没有训练数据")
            return
        data = {
            "reward_history": self.agent.reward_history,
            "loss_history": self.agent.loss_history,
            "episode_lengths": self.agent.episode_lengths,
            "step_rewards": self.agent.step_rewards
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json",
                                                 filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                messagebox.showinfo("成功", f"训练数据已保存到 {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存训练数据失败: {e}")

    def load_training_data(self):
        if self.agent is None:
            self.init_environment()
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.agent.reward_history = data.get("reward_history", [])
            self.agent.loss_history = data.get("loss_history", [])
            self.agent.episode_lengths = data.get("episode_lengths", [])
            self.agent.step_rewards = data.get("step_rewards", [])
            messagebox.showinfo("成功", f"训练数据已从 {file_path} 加载")
        except Exception as e:
            messagebox.showerror("错误", f"加载训练数据失败: {e}")

    def generate_report(self):
        if not self.env.drones:
            messagebox.showwarning("警告", "没有仿真数据")
            return
        self.stop()
        try:
            stats = generate_academic_report(self.env, self.plan_env, self.agent)
            messagebox.showinfo("成功", f"学术报告已生成，保存在 {ANALYSIS_DIR} 文件夹\n\n{stats}")
        except Exception as e:
            messagebox.showerror("错误", f"生成报告时出错: {e}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = DronePathPlanningGUI(root)
    root.mainloop()
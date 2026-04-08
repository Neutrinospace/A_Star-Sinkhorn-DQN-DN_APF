import sys
import os
import random
import time
import threading
import json
import math
from collections import deque
from heapq import heappush, heappop
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev

import matplotlib
matplotlib.use('Qt5Agg')          # GUI 交互式后端
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QFileDialog, QMessageBox
)
from PyQt5.QtCore import QTimer, pyqtSignal, QObject

import warnings
warnings.filterwarnings('ignore')
matplotlib.rc('font', family="SimSun")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

# 创建目录
os.makedirs('analysis_figures', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('models', exist_ok=True)


# ---------------------------- 环境模块（支持矿山地形与障碍物）----------------------------------
class MineEnvironment:
    def __init__(self, grid_size=(80, 80, 25), resolution=1.0):
        self.res = resolution
        self.x_len, self.y_len, self.z_len = grid_size
        self.x = np.arange(0, self.x_len, self.res)
        self.y = np.arange(0, self.y_len, self.res)
        self.z = np.arange(0, self.z_len, self.res)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.height_map = None
        self.obstacle_map = None

    def generate_mine(self, num_cones=4, cone_height_range=(5, 20), cone_radius_range=(5, 15)):
        height_map = np.zeros((self.x_len, self.y_len))
        for _ in range(num_cones):
            cx = random.uniform(10, self.x_len - 10)
            cy = random.uniform(10, self.y_len - 10)
            h = random.uniform(*cone_height_range)
            r = random.uniform(*cone_radius_range)
            dx = self.X - cx
            dy = self.Y - cy
            dist = np.sqrt(dx**2 + dy**2)
            mask = dist < r
            z_cone = h * (1 - dist[mask] / r)
            height_map[mask] = np.maximum(height_map[mask], z_cone)
        self.height_map = gaussian_filter(height_map, sigma=2)
        self._rebuild_obstacle_map()

    def import_terrain(self, height_map):
        if height_map.shape != (self.x_len, self.y_len):
            raise ValueError("地形尺寸不匹配")
        self.height_map = height_map
        self._rebuild_obstacle_map()

    def _rebuild_obstacle_map(self):
        self.obstacle_map = np.zeros((self.x_len, self.y_len, self.z_len), dtype=np.int8)
        for i in range(self.x_len):
            for j in range(self.y_len):
                terrain_z = int(np.ceil(self.height_map[i, j])) + 1
                if terrain_z >= self.z_len:
                    terrain_z = self.z_len - 1
                self.obstacle_map[i, j, :terrain_z] = 1

    def generate_random_obstacles(self, num_obstacles=15, size_range=(1, 3)):
        for _ in range(num_obstacles):
            cx = random.uniform(0, self.x_len)
            cy = random.uniform(0, self.y_len)
            cz = random.uniform(2, self.z_len / 2)
            r = random.uniform(*size_range)
            r_cells = int(np.ceil(r / self.res)) + 1
            x0 = max(0, int(cx) - r_cells)
            x1 = min(self.x_len, int(cx) + r_cells + 1)
            y0 = max(0, int(cy) - r_cells)
            y1 = min(self.y_len, int(cy) + r_cells + 1)
            z0 = max(0, int(cz) - r_cells)
            z1 = min(self.z_len, int(cz) + r_cells + 1)
            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            zs = np.arange(z0, z1)
            XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing='ij')
            dist2 = (XX - cx)**2 + (YY - cy)**2 + (ZZ - cz)**2
            mask = dist2 < r**2
            self.obstacle_map[x0:x1, y0:y1, z0:z1][mask] = 1

    def sample_free_position(self, near_mountain=True, avoid_positions=None, min_dist=0.0):
        """采样自由位置，near_mountain=True 时倾向于高海拔区域"""
        if near_mountain and self.height_map is not None:
            flat_height = self.height_map.flatten()
            prob = flat_height / (flat_height.sum() + 1e-6)
            idx = np.random.choice(len(flat_height), p=prob)
            ix = idx // self.y_len
            iy = idx % self.y_len
            x = ix * self.res + random.uniform(-0.5, 0.5) * self.res
            y = iy * self.res + random.uniform(-0.5, 0.5) * self.res
            x = np.clip(x, 0, self.x_len - 1e-3)
            y = np.clip(y, 0, self.y_len - 1e-3)
            ix = int(x / self.res)
            iy = int(y / self.res)
            terrain_z = self.height_map[ix, iy]
            z = terrain_z + random.uniform(1, 5)
            if z < self.z_len and self.obstacle_map[ix, iy, int(z)] == 0:
                if avoid_positions is not None:
                    if all(np.linalg.norm(np.array([x, y, z]) - p) >= min_dist for p in avoid_positions):
                        return np.array([x, y, z])
                    else:
                        return self.sample_free_position(near_mountain, avoid_positions, min_dist)
                return np.array([x, y, z])
        for _ in range(1000):
            x = random.uniform(0, self.x_len)
            y = random.uniform(0, self.y_len)
            ix = int(x / self.res)
            iy = int(y / self.res)
            if 0 <= ix < self.x_len and 0 <= iy < self.y_len:
                terrain_z = self.height_map[ix, iy] if self.height_map is not None else 0
                z = terrain_z + random.uniform(1, 5)
                if z < self.z_len and self.obstacle_map[ix, iy, int(z)] == 0:
                    if avoid_positions is not None:
                        if all(np.linalg.norm(np.array([x, y, z]) - p) >= min_dist for p in avoid_positions):
                            return np.array([x, y, z])
                    else:
                        return np.array([x, y, z])
        return np.array([self.x_len / 2, self.y_len / 2, 5.0])

    def get_pointcloud(self, pos, radius=10.0):
        if self.obstacle_map is None:
            return np.zeros((0, 3))
        ix = int(pos[0] / self.res)
        iy = int(pos[1] / self.res)
        iz = int(pos[2] / self.res)
        r_cells = int(radius / self.res) + 1
        x0 = max(0, ix - r_cells)
        x1 = min(self.x_len, ix + r_cells + 1)
        y0 = max(0, iy - r_cells)
        y1 = min(self.y_len, iy + r_cells + 1)
        z0 = max(0, iz - r_cells)
        z1 = min(self.z_len, iz + r_cells + 1)
        sub_map = self.obstacle_map[x0:x1, y0:y1, z0:z1]
        idx = np.argwhere(sub_map == 1)
        if len(idx) == 0:
            return np.zeros((0, 3))
        points = (idx + np.array([x0, y0, z0])) * self.res
        return points.astype(np.float32)


# ---------------------------- A* 规划器（带路径平滑）----------------------------------
class AStarPlanner:
    def __init__(self, env):
        self.env = env
        self.res = env.res
        self.grid = env.obstacle_map
        self.x_len, self.y_len, self.z_len = self.grid.shape[:3]
        self.neighbors = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                          if not (dx == 0 and dy == 0 and dz == 0)]

    @staticmethod
    def heuristic(a, b, w):
        h = np.linalg.norm(np.array(a) - np.array(b))
        return np.log(1 + h / (w + 0.01)) * h

    def a_star(self, start, goal):
        sx = int(start[0] / self.res)
        sy = int(start[1] / self.res)
        sz = int(start[2] / self.res)
        gx = int(goal[0] / self.res)
        gy = int(goal[1] / self.res)
        gz = int(goal[2] / self.res)

        total_cells = self.x_len * self.y_len * self.z_len
        w = np.sum(self.grid) / total_cells if total_cells > 0 else 0.1

        open_set = []
        heappush(open_set, (0, 0, 0, sx, sy, sz, None))
        closed_set = set()
        g_cost = {(sx, sy, sz): 0}
        parent = {}

        while open_set:
            f, _, g, x, y, z, p = heappop(open_set)
            node = (x, y, z)
            if node in closed_set:
                continue
            closed_set.add(node)
            parent[node] = p
            if node == (gx, gy, gz):
                path = []
                cur = node
                while cur is not None:
                    path.append([cur[0] * self.res, cur[1] * self.res, cur[2] * self.res])
                    cur = parent.get(cur)
                return path[::-1]

            for dx, dy, dz in self.neighbors:
                nx, ny, nz = x + dx, y + dy, z + dz
                if not (0 <= nx < self.x_len and 0 <= ny < self.y_len and 0 <= nz < self.z_len):
                    continue
                if self.grid[nx, ny, nz] == 1:
                    continue
                cost = np.linalg.norm([dx * self.res, dy * self.res, dz * self.res])
                new_g = g + cost
                h_val = self.heuristic((nx, ny, nz), (gx, gy, gz), w)
                f_val = new_g + h_val
                if (nx, ny, nz) not in closed_set and new_g < g_cost.get((nx, ny, nz), np.inf):
                    g_cost[(nx, ny, nz)] = new_g
                    heappush(open_set, (f_val, new_g, new_g, nx, ny, nz, node))
        return None

    def smooth_path(self, path, s=10.0):
        """B 样条平滑路径，s 控制平滑程度"""
        if len(path) < 4:
            return path
        path = np.array(path)
        tck, u = splprep([path[:, 0], path[:, 1], path[:, 2]], s=s)
        u_new = np.linspace(0, 1, max(200, len(path)))
        x_new, y_new, z_new = splev(u_new, tck)
        return np.vstack([x_new, y_new, z_new]).T.tolist()


# ---------------------------- 改进的 Sinkhorn 分配器（自适应松弛因子 + 融合代价矩阵）----------------------------------
class ImprovedSinkhornAllocator:
    def __init__(self, epsilon0=0.8, epsilon_min=0.05, decay=0.95, max_iter=80, tol=1e-6):
        self.epsilon0 = epsilon0
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.max_iter = max_iter
        self.tol = tol

    def allocate(self, cost_matrix, a=None, b=None):
        """
        cost_matrix: (N_u, N_t) 融合代价矩阵
        a: 无人机供给向量（默认全1）
        b: 任务需求向量（默认全1）
        返回 assignment: (N_u,) 每个无人机分配的任务索引
        """
        N, M = cost_matrix.shape
        if a is None:
            a = np.ones(N) / N
        if b is None:
            b = np.ones(M) / M
        a = a / a.sum()
        b = b / b.sum()

        epsilon = self.epsilon0
        u = np.ones(N)
        v = np.ones(M)
        P_prev = None

        for it in range(self.max_iter):
            K = np.exp(-cost_matrix / epsilon)
            # Sinkhorn 迭代（带阻尼）
            for _ in range(10):
                v_new = b / (K.T @ u + 1e-12)
                v = 0.5 * v + 0.5 * v_new
                u_new = a / (K @ v + 1e-12)
                u = 0.5 * u + 0.5 * u_new
            P = np.diag(u) @ K @ np.diag(v)
            if P_prev is not None and np.linalg.norm(P - P_prev) < self.tol:
                break
            P_prev = P
            # 自适应松弛因子衰减
            epsilon = max(self.epsilon_min, epsilon * self.decay)

        # 硬分配
        assignment = np.argmax(P, axis=1)
        used_tasks = set(assignment)
        for j in range(M):
            if j not in used_tasks:
                candidates = np.argsort(P[:, j])[::-1]
                for i in candidates:
                    if assignment[i] != j:
                        assignment[i] = j
                        used_tasks.add(j)
                        break
        return assignment


# ---------------------------- 自适应 APF（Q 值调制参数 + 盘旋逃离）----------------------------------
class AdaptiveAPF:
    def __init__(self, env, base_xi=1.5, base_eta=20.0, d0=6.0, eta_inter=10.0, safe_dist=2.5,
                 eta_task=5.0, escape_force=100.0, stuck_threshold=15):
        self.env = env
        self.base_xi = base_xi
        self.base_eta = base_eta
        self.d0 = d0
        self.eta_inter = eta_inter
        self.safe_dist = safe_dist
        self.eta_task = eta_task
        self.escape_force = escape_force
        self.stuck_threshold = stuck_threshold

    def get_force(self, pos, goal, other_positions, other_task_positions,
                  min_q_value, stuck_counter):
        d2goal = np.linalg.norm(pos - goal)
        # 引力系数：Q 值低时减小引力（避免盲目），同时随距离自适应
        xi = self.base_xi * (0.5 + 0.5 * min_q_value) * (1 + d2goal / 50.0)
        F_att = -xi * (pos - goal)

        # 斥力系数：Q 值低时增大斥力（更警觉）
        eta = self.base_eta * (1.5 - min_q_value)

        F_rep = np.zeros(3)
        ix = int(pos[0] / self.env.res)
        iy = int(pos[1] / self.env.res)
        iz = int(pos[2] / self.env.res)

        # 静态障碍物斥力（局部搜索窗口 5x5x5）
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                for dz in range(-5, 6):
                    nx, ny, nz = ix + dx, iy + dy, iz + dz
                    if not (0 <= nx < self.env.x_len and 0 <= ny < self.env.y_len and 0 <= nz < self.env.z_len):
                        continue
                    if self.env.obstacle_map[nx, ny, nz] == 1:
                        obs_pos = np.array([nx * self.env.res, ny * self.env.res, nz])
                        d = np.linalg.norm(pos - obs_pos)
                        if d < self.d0:
                            F_rep += eta * (1/d - 1/self.d0) * (1/d**2) * (pos - obs_pos) / (d + 1e-6)

        # 其他任务点斥力
        if other_task_positions is not None:
            for task_pos in other_task_positions:
                if np.linalg.norm(task_pos - goal) < 0.5:
                    continue
                d = np.linalg.norm(pos - task_pos)
                if d < self.d0:
                    F_rep += self.eta_task * (1/d - 1/self.d0) * (1/d**2) * (pos - task_pos) / (d + 1e-6)

        # 无人机间互斥
        F_inter = np.zeros(3)
        if other_positions is not None:
            for op in other_positions:
                d = np.linalg.norm(pos - op)
                if d < self.safe_dist:
                    F_inter += self.eta_inter * (1/d - 1/self.safe_dist) * (pos - op) / (d + 1e-6)

        force = F_att + F_rep + F_inter

        # 盘旋逃离机制
        if d2goal < 12.0 and stuck_counter > self.stuck_threshold:
            escape_dir = goal - pos
            if np.linalg.norm(escape_dir) < 0.1:
                escape_dir = np.random.randn(3)
                escape_dir[2] = 0
            escape_dir = escape_dir / (np.linalg.norm(escape_dir) + 1e-6)
            force += self.escape_force * escape_dir
            force += np.random.randn(3) * 8.0   # 随机扰动

        return force

    def compute_desired_velocity(self, pos, goal, other_positions, other_task_positions,
                                 min_q_value, stuck_counter, max_speed=3.5):
        F = self.get_force(pos, goal, other_positions, other_task_positions, min_q_value, stuck_counter)
        norm = np.linalg.norm(F)
        if norm > 1e-6:
            v = (F / norm) * max_speed
        else:
            v = np.zeros(3)
        return v


# ---------------------------- DQN 模块（Dueling + Noisy）----------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def forward(self, x):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return nn.functional.linear(x, weight, bias)


class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        x = x.max(dim=1)[0]
        return x


class DuelingNoisyNetDQN(nn.Module):
    def __init__(self, state_dim_target, n_actions, hidden_dim=256):
        super().__init__()
        self.pointcloud_encoder = PointCloudEncoder(input_dim=3, output_dim=64)
        self.fc = nn.Sequential(
            nn.Linear(64 + state_dim_target, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, n_actions)
        )

    def forward(self, pointcloud, target_state):
        pc_feat = self.pointcloud_encoder(pointcloud)
        x = torch.cat([pc_feat, target_state], dim=1)
        x = self.fc(x)
        V = self.value_stream(x)
        A = self.advantage_stream(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


class DQNAgent:
    def __init__(self, state_dim_target, action_dim, lr=2e-4, gamma=0.97,
                 buffer_size=150000, batch_size=128, target_update=250,
                 device='cpu', max_points=500):
        self.state_dim_target = state_dim_target
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device
        self.max_points = max_points
        self.steps = 0
        self.loss_history = []
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9993

        self.q_net = DuelingNoisyNetDQN(state_dim_target, action_dim).to(device)
        self.target_net = DuelingNoisyNetDQN(state_dim_target, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)

    def preprocess_pointcloud(self, pc):
        if len(pc) > self.max_points:
            idx = np.random.choice(len(pc), self.max_points, replace=False)
            pc = pc[idx]
        elif len(pc) < self.max_points:
            pad = np.zeros((self.max_points - len(pc), 3))
            pc = np.vstack([pc, pad])
        return torch.FloatTensor(pc).unsqueeze(0).to(self.device)

    def select_action(self, pointcloud, target_state, eval_mode=False):
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            pc_tensor = self.preprocess_pointcloud(pointcloud)
            ts_tensor = torch.FloatTensor(target_state).unsqueeze(0).to(self.device)
            q_values = self.q_net(pc_tensor, ts_tensor)
        return q_values.argmax().item()

    def store_transition(self, pc, ts, action, reward, next_pc, next_ts, done):
        self.memory.append((pc, ts, action, reward, next_pc, next_ts, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        pcs, tss, actions, rewards, next_pcs, next_tss, dones = zip(*batch)

        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        pc_tensor = torch.stack([self.preprocess_pointcloud(pc).squeeze(0) for pc in pcs])
        next_pc_tensor = torch.stack([self.preprocess_pointcloud(npc).squeeze(0) for npc in next_pcs])
        ts_tensor = torch.FloatTensor(tss).to(self.device)
        next_ts_tensor = torch.FloatTensor(next_tss).to(self.device)

        q_values = self.q_net(pc_tensor, ts_tensor).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_pc_tensor, next_ts_tensor).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def get_min_q_value(self, pointcloud, target_state):
        if len(pointcloud) == 0:
            return 0.5
        pc_tensor = self.preprocess_pointcloud(pointcloud)
        ts_tensor = torch.FloatTensor(target_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.q_net(pc_tensor, ts_tensor)
        return qvals.min().item()

    def get_max_q_value(self, pointcloud, target_state):
        if len(pointcloud) == 0:
            return 0.5
        pc_tensor = self.preprocess_pointcloud(pointcloud)
        ts_tensor = torch.FloatTensor(target_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            qvals = self.q_net(pc_tensor, ts_tensor)
        return qvals.max().item()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


# ---------------------------- 仿真主类（集成 Sinkhorn + DQN + APF）----------------------------------
class MineSimulation:
    def __init__(self, env, num_uavs, num_tasks, max_steps=300, step_time=0.08,
                 replan_interval=30):
        self.env = env
        self.num_uavs = num_uavs
        self.num_tasks = num_tasks
        self.max_steps = max_steps
        self.step_time = step_time
        self.replan_interval = replan_interval

        self.uavs = None
        self.tasks = None
        self.velocities = None
        self.astar = AStarPlanner(env)
        self.sinkhorn = ImprovedSinkhornAllocator(epsilon0=0.8, epsilon_min=0.05, decay=0.95)
        self.apf = AdaptiveAPF(env, base_xi=1.5, base_eta=20.0, stuck_threshold=12)
        self.state_dim_target = 4
        self.action_dim = 9
        self.agent = DQNAgent(self.state_dim_target, self.action_dim,
                              device='cuda' if torch.cuda.is_available() else 'cpu')
        self.episode_rewards = []
        self.collision_rates = []
        self.path_lengths = []
        self.success_rates = []
        self.current_targets = None
        self.cost_matrix = None
        self.trajectories = None
        self.collision_log = []
        self.stuck_counters = None

    def regenerate_points(self):
        min_d = 5.0
        tasks = []
        for _ in range(self.num_tasks):
            pos = self.env.sample_free_position(near_mountain=True, avoid_positions=tasks, min_dist=min_d)
            tasks.append(pos)
        uavs = []
        for _ in range(self.num_uavs):
            avoid = tasks + uavs
            pos = self.env.sample_free_position(near_mountain=True, avoid_positions=avoid, min_dist=min_d)
            uavs.append(pos)
        return uavs, tasks

    def compute_fused_cost_matrix(self):
        """
        融合代价矩阵：C = α * L_norm + β * (1 - Q_norm) + γ * Cctrl_norm
        """
        N, M = self.num_uavs, self.num_tasks
        L = np.zeros((N, M))
        Q_risk = np.zeros((N, M))
        C_ctrl = np.zeros((N, M))

        # 1. A* 路径长度
        for i, uav_pos in enumerate(self.uavs):
            for j, task_pos in enumerate(self.tasks):
                path = self.astar.a_star(uav_pos, task_pos)
                if path is None:
                    L[i, j] = 1e6
                else:
                    length = sum(np.linalg.norm(np.array(path[k+1]) - np.array(path[k]))
                                 for k in range(len(path)-1))
                    L[i, j] = length
        L_norm = (L - L.min()) / (L.max() - L.min() + 1e-6)

        # 2. DQN 风险 (1 - 归一化最大 Q 值)
        for i, uav_pos in enumerate(self.uavs):
            for j, task_pos in enumerate(self.tasks):
                delta = task_pos - uav_pos
                d = np.linalg.norm(delta)
                theta = np.arctan2(delta[1], delta[0])
                dz = delta[2]
                in_view = 1 if np.abs(theta) < np.pi/2 else 0
                ts = np.array([d, theta, dz, in_view], dtype=np.float32)
                pc = self.env.get_pointcloud(uav_pos)
                q_max = self.agent.get_max_q_value(pc, ts) if len(pc) > 0 else 0.5
                q_norm = (q_max + 10) / 20.0   # 假设 Q 值范围 [-10,10]
                risk = 1.0 - q_norm
                Q_risk[i, j] = risk
        Q_risk_norm = (Q_risk - Q_risk.min()) / (Q_risk.max() - Q_risk.min() + 1e-6)

        # 3. APF 控制代价（短时模拟平均斥力）
        for i, uav_pos in enumerate(self.uavs):
            for j, task_pos in enumerate(self.tasks):
                temp_pos = uav_pos.copy()
                total_rep = 0.0
                for _ in range(8):
                    ix = int(temp_pos[0] / self.env.res)
                    iy = int(temp_pos[1] / self.env.res)
                    iz = int(temp_pos[2] / self.env.res)
                    rep = 0.0
                    for dx in range(-3,4):
                        for dy in range(-3,4):
                            for dz in range(-3,4):
                                nx, ny, nz = ix+dx, iy+dy, iz+dz
                                if 0<=nx<self.env.x_len and 0<=ny<self.env.y_len and 0<=nz<self.env.z_len:
                                    if self.env.obstacle_map[nx,ny,nz]==1:
                                        obs_pos = np.array([nx*self.env.res, ny*self.env.res, nz])
                                        d = np.linalg.norm(temp_pos - obs_pos)
                                        if d < self.apf.d0:
                                            rep += self.apf.base_eta * (1/d - 1/self.apf.d0) * (1/d**2)
                    total_rep += rep
                    # 简单向目标移动
                    dir_to_goal = task_pos - temp_pos
                    if np.linalg.norm(dir_to_goal) > 0:
                        temp_pos += dir_to_goal / np.linalg.norm(dir_to_goal) * 0.5
                C_ctrl[i, j] = total_rep / 8.0
        C_ctrl_norm = (C_ctrl - C_ctrl.min()) / (C_ctrl.max() - C_ctrl.min() + 1e-6)

        alpha, beta, gamma = 0.4, 0.35, 0.25
        cost_matrix = alpha * L_norm + beta * Q_risk_norm + gamma * C_ctrl_norm
        return cost_matrix

    def assign_tasks(self):
        self.cost_matrix = self.compute_fused_cost_matrix()
        # 检查任务可达性
        for j in range(self.num_tasks):
            if np.all(self.cost_matrix[:, j] > 1e5):
                self.tasks[j] = self.env.sample_free_position(near_mountain=True)
                return self.assign_tasks()
        assignment = self.sinkhorn.allocate(self.cost_matrix)
        self.current_targets = [self.tasks[j] for j in assignment]
        return self.current_targets

    def get_target_state(self, uav_idx):
        pos = self.uavs[uav_idx]
        target = self.current_targets[uav_idx]
        d = np.linalg.norm(pos - target)
        delta = target - pos
        theta = np.arctan2(delta[1], delta[0])
        dz = delta[2]
        in_view = 1 if np.abs(theta) < np.pi/2 else 0
        return np.array([d, theta, dz, in_view], dtype=np.float32)

    def step(self, actions, episode_num, step_num):
        new_uavs = []
        rewards = []
        dones = []
        all_positions = self.uavs.copy()
        for i in range(self.num_uavs):
            if hasattr(self, 'dones') and self.dones[i]:
                new_uavs.append(self.uavs[i])
                rewards.append(0.0)
                dones.append(True)
                continue

            pos = self.uavs[i]
            target = self.current_targets[i]
            others = [all_positions[j] for j in range(self.num_uavs) if j != i]
            other_tasks = [self.tasks[j] for j in range(self.num_tasks) if not np.array_equal(self.current_targets[i], self.tasks[j])]

            pc = self.env.get_pointcloud(pos)
            ts = self.get_target_state(i)
            min_q = self.agent.get_min_q_value(pc, ts) if len(pc) > 0 else 0.5

            dist_to_target = np.linalg.norm(pos - target)
            if dist_to_target < 12.0 and np.linalg.norm(self.velocities[i]) < 0.7:
                self.stuck_counters[i] += 1
            else:
                self.stuck_counters[i] = max(0, self.stuck_counters[i] - 1)

            # 动作解析
            if actions[i] == 8:
                delta = target - pos
                norm = np.linalg.norm(delta)
                v = delta / norm * 4.0 if norm > 0 else np.zeros(3)
            else:
                vel_inc = np.zeros(3)
                if actions[i] == 0:      vel_inc[0] = 2.0
                elif actions[i] == 1:    vel_inc[0] = -2.0
                elif actions[i] == 2:    vel_inc[1] = -2.0
                elif actions[i] == 3:    vel_inc[1] = 2.0
                elif actions[i] == 4:    vel_inc[2] = 2.0
                elif actions[i] == 5:    vel_inc[2] = -2.0
                elif actions[i] == 6:    vel_inc[0] = -1.5; vel_inc[1] = 1.5
                elif actions[i] == 7:    vel_inc[0] = 1.5; vel_inc[1] = -1.5
                v = vel_inc

            v_apf = self.apf.compute_desired_velocity(pos, target, others, other_tasks,
                                                      min_q, self.stuck_counters[i], max_speed=3.5)
            v_final = 0.5 * v + 0.5 * v_apf
            self.velocities[i] = v_final
            new_pos = pos + v_final * self.step_time

            # 碰撞检测
            collided = False
            collision_type = None
            if (new_pos[0] < 0 or new_pos[0] > self.env.x_len or
                new_pos[1] < 0 or new_pos[1] > self.env.y_len or
                new_pos[2] < 0 or new_pos[2] > self.env.z_len):
                collided = True
                collision_type = "boundary"
            else:
                ix = int(new_pos[0] / self.env.res)
                iy = int(new_pos[1] / self.env.res)
                iz = int(new_pos[2] / self.env.res)
                if 0 <= ix < self.env.x_len and 0 <= iy < self.env.y_len and 0 <= iz < self.env.z_len:
                    if self.env.obstacle_map[ix, iy, iz] == 1:
                        collided = True
                        collision_type = "static_obstacle"
                for other in others:
                    if np.linalg.norm(new_pos - other) < 1.2:
                        collided = True
                        collision_type = "inter_uav"
                        break

            if collided:
                reward = -80.0
                new_uavs.append(pos)
                dones.append(True)
                self.collision_log.append((episode_num, step_num, i, collision_type, new_pos.tolist()))
            else:
                new_uavs.append(new_pos)
                old_dist = np.linalg.norm(pos - target)
                new_dist = np.linalg.norm(new_pos - target)
                reward = 0.0
                # 距离减少奖励（密集）
                reward += 6.0 * (old_dist - new_dist)
                # 方向对齐奖励
                if old_dist > 1.0:
                    dir_to_target = (target - pos) / old_dist
                    vel_dir = v_final / (np.linalg.norm(v_final)+1e-6)
                    alignment = np.dot(dir_to_target, vel_dir)
                    reward += 2.5 * max(0, alignment)
                if new_dist < 1.0:
                    reward += 120.0
                    dones.append(True)
                else:
                    dones.append(False)
                # 盘旋惩罚
                if dist_to_target < 12.0 and self.stuck_counters[i] > self.apf.stuck_threshold:
                    reward -= 12.0
                reward -= 0.15   # 步数惩罚
            rewards.append(reward)

        self.uavs = new_uavs
        self.dones = dones
        if self.trajectories is None:
            self.trajectories = [[] for _ in range(self.num_uavs)]
        for i in range(self.num_uavs):
            self.trajectories[i].append(self.uavs[i].copy())
        return self.get_states(), rewards, dones

    def get_states(self):
        states = []
        for i in range(self.num_uavs):
            pc = self.env.get_pointcloud(self.uavs[i])
            ts = self.get_target_state(i)
            states.append((pc, ts))
        return states

    def reset(self):
        self.uavs, self.tasks = self.regenerate_points()
        self.velocities = [np.zeros(3) for _ in range(self.num_uavs)]
        self.stuck_counters = [0] * self.num_uavs
        self.assign_tasks()
        self.dones = [False] * self.num_uavs
        self.trajectories = [[] for _ in range(self.num_uavs)]
        return self.get_states()

    def train_episode(self, episode_num, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps
        states = self.reset()
        total_reward = 0.0
        for step in range(max_steps):
            if step % self.replan_interval == 0 and step > 0:
                self.assign_tasks()
            actions = []
            for i in range(self.num_uavs):
                if self.dones[i]:
                    actions.append(0)
                    continue
                pc, ts = states[i]
                act = self.agent.select_action(pc, ts, eval_mode=False)
                actions.append(act)
            next_states, rewards, dones = self.step(actions, episode_num, step)
            for i in range(self.num_uavs):
                if not self.dones[i] or dones[i]:
                    self.agent.store_transition(states[i][0], states[i][1], actions[i],
                                                rewards[i], next_states[i][0], next_states[i][1], dones[i])
            self.agent.update()
            states = next_states
            total_reward += sum(rewards)
            if all(dones):
                break

        self.episode_rewards.append(total_reward)
        collisions = sum(1 for i in range(self.num_uavs) if self.dones[i] and rewards[i] == -80)
        self.collision_rates.append(collisions / self.num_uavs)
        path_len = 0.0
        for i in range(self.num_uavs):
            if len(self.trajectories[i]) > 1:
                path_len += sum(np.linalg.norm(np.array(self.trajectories[i][k+1]) - np.array(self.trajectories[i][k]))
                                for k in range(len(self.trajectories[i])-1))
        self.path_lengths.append(path_len / self.num_uavs)
        success = all(np.linalg.norm(self.uavs[i] - self.current_targets[i]) < 1.0 for i in range(self.num_uavs))
        self.success_rates.append(1.0 if success else 0.0)
        return total_reward

    def run_simulation(self, episodes=100):
        for ep in range(episodes):
            self.train_episode(ep)
            if ep % 10 == 0:
                print(f"Episode {ep}: Reward={self.episode_rewards[-1]:.2f}, "
                      f"Collision={self.collision_rates[-1]:.2f}, Success={self.success_rates[-1]:.2f}")
        self.agent.save('models/dqn_model.pth')
        np.save('logs/rewards.npy', self.episode_rewards)
        np.save('logs/collisions.npy', self.collision_rates)
        np.save('logs/path_lengths.npy', self.path_lengths)
        np.save('logs/success.npy', self.success_rates)
        if self.trajectories is not None:
            np.save('logs/final_trajectory.npy', self.trajectories)
        np.save('logs/collision_log.npy', self.collision_log)

    def smooth_final_trajectories(self):
        if self.trajectories is None:
            return
        smoothed = []
        for traj in self.trajectories:
            if len(traj) < 4:
                smoothed.append(traj)
                continue
            traj_arr = np.array(traj)
            try:
                tck, u = splprep([traj_arr[:,0], traj_arr[:,1], traj_arr[:,2]], s=1.0)
                u_new = np.linspace(0, 1, len(traj_arr))
                x_new, y_new, z_new = splev(u_new, tck)
                smoothed_traj = np.vstack([x_new, y_new, z_new]).T.tolist()
                smoothed.append(smoothed_traj)
            except:
                smoothed.append(traj)
        self.trajectories = smoothed

    def test_planning(self, model_path=None, max_steps=None):
        if model_path and os.path.exists(model_path):
            self.agent.load(model_path)
        elif not model_path:
            model_path = 'models/dqn_model.pth'
            if os.path.exists(model_path):
                self.agent.load(model_path)
        if max_steps is None:
            max_steps = self.max_steps
        states = self.reset()
        for step in range(max_steps):
            if step % self.replan_interval == 0 and step > 0:
                self.assign_tasks()
            actions = []
            for i in range(self.num_uavs):
                if self.dones[i]:
                    actions.append(0)
                    continue
                pc, ts = states[i]
                act = self.agent.select_action(pc, ts, eval_mode=True)
                actions.append(act)
            next_states, _, dones = self.step(actions, episode_num=0, step_num=step)
            states = next_states
            if all(dones):
                break
        self.smooth_final_trajectories()
        return self.trajectories, self.current_targets


# ---------------------------- 科学分析工具类（修复图像保存问题）----------------------------------
def safe_save_figure(fig, path, dpi=300):
    """安全保存图像，不切换后端，直接保存"""
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

class SimpleAnalysisTools:
    @staticmethod
    def plot_training_curves(rewards, collisions, path_lengths, success, save_dir='analysis_figures'):
        os.makedirs(save_dir, exist_ok=True)
        episodes = np.arange(len(rewards))
        window = 10

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(episodes, rewards, 'b-', alpha=0.7, label='Episode Reward')
        if len(rewards) > window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2, label='Moving Average')
        ax.set_xlabel('Episode'); ax.set_ylabel('Total Reward'); ax.set_title('Training Reward Curve')
        ax.legend(); ax.grid(True)
        safe_save_figure(fig, os.path.join(save_dir, 'reward_curve.png'))

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(episodes, collisions, 'g-', alpha=0.7)
        if len(collisions) > window:
            moving_avg = np.convolve(collisions, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Collision Rate'); ax.set_title('Collision Rate Curve')
        ax.grid(True)
        safe_save_figure(fig, os.path.join(save_dir, 'collision_curve.png'))

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(episodes, path_lengths, 'c-', alpha=0.7)
        if len(path_lengths) > window:
            moving_avg = np.convolve(path_lengths, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Avg Path Length (m)'); ax.set_title('Path Length Curve')
        ax.grid(True)
        safe_save_figure(fig, os.path.join(save_dir, 'path_length_curve.png'))

        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(episodes, success, 'm-', alpha=0.7)
        if len(success) > window:
            moving_avg = np.convolve(success, np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2)
        ax.set_xlabel('Episode'); ax.set_ylabel('Success Rate'); ax.set_title('Task Completion Rate')
        ax.grid(True)
        safe_save_figure(fig, os.path.join(save_dir, 'success_rate_curve.png'))

    @staticmethod
    def plot_path_deviation(trajectories, astar_planner, save_dir='analysis_figures'):
        """相对于 A* 平滑路径的偏离度分析"""
        if not trajectories or len(trajectories[0]) < 2:
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('路径偏离度分析 (相对于 A* 平滑路径)', fontsize=16)
        colors = plt.cm.tab10.colors
        all_deviations = {}
        for idx, traj in enumerate(trajectories):
            if len(traj) < 2:
                continue
            start = traj[0]
            end = traj[-1]
            ref_path = astar_planner.a_star(start, end)
            if ref_path is None or len(ref_path) < 2:
                continue
            ref_smooth = astar_planner.smooth_path(ref_path, s=5.0)
            ref_arr = np.array(ref_smooth)
            s_ref = np.linspace(0, 1, len(ref_arr))
            s_act = np.linspace(0, 1, len(traj))
            ref_interp = np.array([np.interp(s_act, s_ref, ref_arr[:, dim]) for dim in range(3)]).T
            deviations = np.linalg.norm(ref_interp - np.array(traj), axis=1)
            all_deviations[idx] = deviations
        if not all_deviations:
            for ax in axes.flatten():
                ax.text(0.5, 0.5, '无偏离数据', ha='center', va='center')
        else:
            ax1 = axes[0,0]
            for idx, dev in all_deviations.items():
                ax1.plot(dev, color=colors[idx%len(colors)], label=f'UAV{idx}', linewidth=2)
            ax1.set_title('每个轨迹点的偏离距离')
            ax1.legend()
            ax2 = axes[0,1]
            for idx, dev in all_deviations.items():
                cum_avg = np.cumsum(dev) / np.arange(1, len(dev)+1)
                ax2.plot(cum_avg, color=colors[idx%len(colors)], label=f'UAV{idx}')
            ax2.set_title('累计平均偏离距离')
            ax2.legend()
            ax3 = axes[1,0]
            max_len = max(len(dev) for dev in all_deviations.values())
            overall_avg, overall_std = [], []
            for step in range(max_len):
                step_vals = [dev[step] for dev in all_deviations.values() if step < len(dev)]
                if step_vals:
                    overall_avg.append(np.mean(step_vals))
                    overall_std.append(np.std(step_vals))
            if overall_avg:
                ax3.plot(overall_avg, 'r-', linewidth=3, label='平均偏离')
                ax3.fill_between(range(len(overall_avg)), np.array(overall_avg)-np.array(overall_std),
                                np.array(overall_avg)+np.array(overall_std), color='r', alpha=0.2)
                ax3.legend()
            ax3.set_title('所有无人机平均偏离')
            ax4 = axes[1,1]
            matrix = []
            indices = sorted(all_deviations.keys())
            max_len = max(len(dev) for dev in all_deviations.values())
            for idx in indices:
                dev = all_deviations[idx]
                # 修复：将 dev 转换为列表后再拼接，避免 NumPy 广播错误
                padded = list(dev) + [np.nan] * (max_len - len(dev))
                matrix.append(padded)
            matrix = np.array(matrix)
            im = ax4.imshow(matrix, cmap='viridis', aspect='auto')
            ax4.set_title('偏离度热力图')
            ax4.set_xlabel('步数'); ax4.set_ylabel('无人机')
            plt.colorbar(im, ax=ax4)
        plt.tight_layout()
        safe_save_figure(fig, os.path.join(save_dir, 'path_deviation.png'))

    @staticmethod
    def plot_q_heatmap(agent, env, sim, num_samples=100, save_dir='analysis_figures'):
        if not hasattr(sim, 'uavs') or sim.uavs is None:
            return
        q_samples = []
        for _ in range(num_samples):
            idx = np.random.randint(len(sim.uavs))
            pc = env.get_pointcloud(sim.uavs[idx])
            if len(pc) == 0:
                continue
            ts = sim.get_target_state(idx)
            pc_tensor = agent.preprocess_pointcloud(pc)
            ts_tensor = torch.FloatTensor(ts).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                qvals = agent.q_net(pc_tensor, ts_tensor).cpu().numpy().flatten()
            q_samples.append(qvals)
        if q_samples:
            q_samples = np.array(q_samples)
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(q_samples[:50], annot=False, cmap='viridis',
                        xticklabels=['F','B','L','R','U','D','YawL','YawR','Goal'], ax=ax)
            ax.set_xlabel('Action'); ax.set_ylabel('State Sample'); ax.set_title('Q-Value Heatmap')
            plt.tight_layout()
            safe_save_figure(fig, os.path.join(save_dir, 'q_heatmap.png'))

    @staticmethod
    def plot_collision_reasons(collision_log, save_dir='analysis_figures'):
        if not collision_log:
            return
        types = [log[3] for log in collision_log]
        counts = {t: types.count(t) for t in set(types)}
        fig, ax = plt.subplots(figsize=(6,6))
        ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%')
        ax.set_title('Collision Reasons')
        safe_save_figure(fig, os.path.join(save_dir, 'collision_reasons.png'))


# ---------------------------- GUI 主窗口 ----------------------------------
class SimThread(QObject):
    update_plot = pyqtSignal(object, object, object, object, object, object)
    finished = pyqtSignal()

    def __init__(self, sim, episodes=100):
        super().__init__()
        self.sim = sim
        self.episodes = episodes
        self.running = True
        self.paused = False

    def run(self):
        for ep in range(self.episodes):
            if not self.running:
                break
            while self.paused and self.running:
                time.sleep(0.1)
            if not self.running:
                break
            self.sim.train_episode(ep)
            self.update_plot.emit(self.sim.trajectories, self.sim.tasks,
                                  self.sim.episode_rewards, self.sim.collision_rates,
                                  self.sim.success_rates, self.sim.collision_log)
            time.sleep(0.01)
        self.finished.emit()

    def stop(self): self.running = False
    def pause(self): self.paused = True
    def resume(self): self.paused = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("矿山无人机集群路径规划系统 - Sinkhorn 科研优化版")
        self.setGeometry(100, 100, 1500, 900)

        self.grid_x = 80
        self.grid_y = 80
        self.grid_z = 25
        self.num_cones = 4
        self.num_obstacles = 15
        self.num_uavs = 3
        self.num_tasks = 3
        self.env = None
        self.sim = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.fig_3d = Figure(figsize=(10, 7))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.canvas_3d = FigureCanvas(self.fig_3d)
        left_layout.addWidget(self.canvas_3d)

        self.fig_curve = Figure(figsize=(10, 3))
        self.ax_curve = self.fig_curve.add_subplot(111)
        self.canvas_curve = FigureCanvas(self.fig_curve)
        left_layout.addWidget(self.canvas_curve)
        main_layout.addWidget(left_panel, stretch=3)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        mine_group = QGroupBox("矿山参数")
        mine_layout = QFormLayout()
        self.x_spin = QSpinBox(); self.x_spin.setRange(50,200); self.x_spin.setValue(80)
        self.y_spin = QSpinBox(); self.y_spin.setRange(50,200); self.y_spin.setValue(80)
        self.z_spin = QSpinBox(); self.z_spin.setRange(15,50); self.z_spin.setValue(25)
        self.cones_spin = QSpinBox(); self.cones_spin.setRange(1,10); self.cones_spin.setValue(4)
        self.obs_spin = QSpinBox(); self.obs_spin.setRange(0,50); self.obs_spin.setValue(15)
        mine_layout.addRow("X范围(m):", self.x_spin)
        mine_layout.addRow("Y范围(m):", self.y_spin)
        mine_layout.addRow("Z范围(m):", self.z_spin)
        mine_layout.addRow("锥体数:", self.cones_spin)
        mine_layout.addRow("障碍物数:", self.obs_spin)
        self.generate_btn = QPushButton("生成矿山")
        self.import_terrain_btn = QPushButton("导入地形(npy)")
        self.export_terrain_btn = QPushButton("导出地形(npy)")
        mine_layout.addRow(self.generate_btn)
        mine_layout.addRow(self.import_terrain_btn)
        mine_layout.addRow(self.export_terrain_btn)
        mine_group.setLayout(mine_layout)
        right_layout.addWidget(mine_group)

        points_group = QGroupBox("无人机与任务点")
        points_layout = QFormLayout()
        self.num_uavs_spin = QSpinBox(); self.num_uavs_spin.setRange(1,10); self.num_uavs_spin.setValue(3)
        self.num_tasks_spin = QSpinBox(); self.num_tasks_spin.setRange(1,10); self.num_tasks_spin.setValue(3)
        self.min_dist_spin = QDoubleSpinBox(); self.min_dist_spin.setRange(0.5,20.0); self.min_dist_spin.setValue(5.0)
        points_layout.addRow("无人机数量:", self.num_uavs_spin)
        points_layout.addRow("任务点数量:", self.num_tasks_spin)
        points_layout.addRow("起点-目标最小距离:", self.min_dist_spin)
        points_group.setLayout(points_layout)
        right_layout.addWidget(points_group)

        train_group = QGroupBox("训练参数")
        train_layout = QFormLayout()
        self.episodes_spin = QSpinBox(); self.episodes_spin.setRange(1,1000); self.episodes_spin.setValue(150)
        train_layout.addRow("训练回合数:", self.episodes_spin)
        self.save_data_btn = QPushButton("保存训练数据")
        self.load_data_btn = QPushButton("加载训练数据")
        train_layout.addRow(self.save_data_btn)
        train_layout.addRow(self.load_data_btn)
        self.start_train_btn = QPushButton("开始训练")
        self.stop_train_btn = QPushButton("停止训练")
        train_layout.addRow(self.start_train_btn)
        train_layout.addRow(self.stop_train_btn)
        train_group.setLayout(train_layout)
        right_layout.addWidget(train_group)

        plan_group = QGroupBox("规划控制")
        plan_layout = QVBoxLayout()
        self.load_model_btn = QPushButton("加载模型")
        self.save_model_btn = QPushButton("保存模型")
        self.single_plan_btn = QPushButton("单次规划（加载模型后）")
        self.analyze_btn = QPushButton("生成科学分析图")
        plan_layout.addWidget(self.load_model_btn)
        plan_layout.addWidget(self.save_model_btn)
        plan_layout.addWidget(self.single_plan_btn)
        plan_layout.addWidget(self.analyze_btn)
        plan_group.setLayout(plan_layout)
        right_layout.addWidget(plan_group)

        self.status_label = QLabel("就绪")
        right_layout.addWidget(self.status_label)
        main_layout.addWidget(right_panel, stretch=1)

        # 信号连接
        self.generate_btn.clicked.connect(self.generate_environment)
        self.import_terrain_btn.clicked.connect(self.import_terrain)
        self.export_terrain_btn.clicked.connect(self.export_terrain)
        self.save_data_btn.clicked.connect(self.save_training_data)
        self.load_data_btn.clicked.connect(self.load_training_data)
        self.start_train_btn.clicked.connect(self.start_training)
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.load_model_btn.clicked.connect(self.load_model)
        self.save_model_btn.clicked.connect(self.save_model)
        self.single_plan_btn.clicked.connect(self.run_single_planning)
        self.analyze_btn.clicked.connect(self.generate_analysis)

        self.sim_thread = None
        self.thread = None
        self.is_training = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_3d_view)
        self.timer.start(500)

    def generate_environment(self):
        self.grid_x = self.x_spin.value()
        self.grid_y = self.y_spin.value()
        self.grid_z = self.z_spin.value()
        self.num_cones = self.cones_spin.value()
        self.num_obstacles = self.obs_spin.value()
        self.env = MineEnvironment(grid_size=(self.grid_x, self.grid_y, self.grid_z))
        self.env.generate_mine(num_cones=self.num_cones)
        self.env.generate_random_obstacles(num_obstacles=self.num_obstacles)
        self.num_uavs = self.num_uavs_spin.value()
        self.num_tasks = self.num_tasks_spin.value()
        if self.num_uavs != self.num_tasks:
            self.num_tasks = self.num_uavs
            self.num_tasks_spin.setValue(self.num_uavs)
        self.sim = MineSimulation(self.env, self.num_uavs, self.num_tasks, max_steps=300, step_time=0.08, replan_interval=30)
        self.sim.reset()
        self.update_3d_view()
        self.status_label.setText("矿山已生成，无人机和任务点已随机初始化")

    def save_model(self):
        if self.sim is None:
            QMessageBox.warning(self, "警告", "请先生成矿山")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "保存模型", "models", "PyTorch models (*.pth)")
        if fname:
            self.sim.agent.save(fname)
            self.status_label.setText("模型已保存")

    def load_model(self):
        if self.sim is None:
            QMessageBox.warning(self, "警告", "请先生成矿山")
            return
        fname, _ = QFileDialog.getOpenFileName(self, "加载模型", "models", "PyTorch models (*.pth)")
        if fname:
            self.sim.agent.load(fname)
            self.status_label.setText("模型已加载")

    def start_training(self):
        if self.is_training:
            return
        if self.sim is None:
            QMessageBox.warning(self, "警告", "请先生成矿山")
            return
        self.sim.agent.memory.clear()
        if hasattr(self.sim.agent, 'loss_history'):
            self.sim.agent.loss_history.clear()
        self.sim.episode_rewards = []
        self.sim.collision_rates = []
        self.sim.path_lengths = []
        self.sim.success_rates = []
        self.sim.collision_log = []

        episodes = self.episodes_spin.value()
        self.sim_thread = SimThread(self.sim, episodes=episodes)
        self.sim_thread.update_plot.connect(self.on_update_plot)
        self.sim_thread.finished.connect(self.on_training_finished)
        self.thread = threading.Thread(target=self.sim_thread.run)
        self.thread.daemon = True
        self.thread.start()
        self.is_training = True
        self.status_label.setText("训练中...")
        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)

    def stop_training(self):
        if self.sim_thread and self.is_training:
            self.sim_thread.stop()
            self.is_training = False
            self.status_label.setText("已停止")
            self.start_train_btn.setEnabled(True)
            self.stop_train_btn.setEnabled(False)

    def run_single_planning(self):
        if self.sim is None:
            QMessageBox.warning(self, "警告", "请先生成矿山")
            return
        model_path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "models", "PyTorch models (*.pth)")
        if not model_path:
            return
        self.sim.test_planning(model_path=model_path)
        self.update_3d_view()
        self.status_label.setText("单次规划完成，路径已平滑")

    def generate_analysis(self):
        if self.sim is None or len(self.sim.episode_rewards) == 0:
            QMessageBox.warning(self, "警告", "无训练数据，请先完成训练")
            return
        save_dir = 'analysis_figures'
        SimpleAnalysisTools.plot_training_curves(
            self.sim.episode_rewards, self.sim.collision_rates,
            self.sim.path_lengths, self.sim.success_rates, save_dir)
        if self.sim.trajectories and len(self.sim.trajectories[0]) > 1:
            SimpleAnalysisTools.plot_path_deviation(self.sim.trajectories, self.sim.astar, save_dir)
        SimpleAnalysisTools.plot_q_heatmap(self.sim.agent, self.env, self.sim, save_dir=save_dir)
        if self.sim.collision_log:
            SimpleAnalysisTools.plot_collision_reasons(self.sim.collision_log, save_dir)
        self.status_label.setText("科学分析图已生成，保存在 analysis_figures 目录")

    def on_training_finished(self):
        self.is_training = False
        self.status_label.setText("训练完成")
        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

    def import_terrain(self):
        if self.env is None:
            QMessageBox.warning(self, "警告", "请先点击「生成矿山」")
            return
        fname, _ = QFileDialog.getOpenFileName(self, "导入地形", "", "Numpy files (*.npy)")
        if fname:
            try:
                terrain = np.load(fname)
                if terrain.shape != (self.env.x_len, self.env.y_len):
                    raise ValueError("尺寸不匹配")
                self.env.import_terrain(terrain)
                self.update_3d_view()
                self.status_label.setText("地形导入成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", str(e))

    def export_terrain(self):
        if self.env is None or self.env.height_map is None:
            QMessageBox.warning(self, "警告", "没有地形数据")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "导出地形", "", "Numpy files (*.npy)")
        if fname:
            np.save(fname, self.env.height_map)
            self.status_label.setText("地形导出成功")

    def save_training_data(self):
        if self.sim is None:
            QMessageBox.warning(self, "警告", "无训练数据")
            return
        data = {
            "episode_rewards": self.sim.episode_rewards,
            "collision_rates": self.sim.collision_rates,
            "path_lengths": self.sim.path_lengths,
            "success_rates": self.sim.success_rates,
            "loss_history": self.sim.agent.loss_history,
            "collision_log": self.sim.collision_log
        }
        fname, _ = QFileDialog.getSaveFileName(self, "保存训练数据", "logs", "JSON (*.json)")
        if fname:
            with open(fname, 'w') as f:
                json.dump(data, f)
            self.status_label.setText("训练数据已保存")

    def load_training_data(self):
        if self.sim is None:
            QMessageBox.warning(self, "警告", "请先生成矿山")
            return
        fname, _ = QFileDialog.getOpenFileName(self, "加载训练数据", "logs", "JSON (*.json)")
        if fname:
            with open(fname, 'r') as f:
                data = json.load(f)
            self.sim.episode_rewards = data.get("episode_rewards", [])
            self.sim.collision_rates = data.get("collision_rates", [])
            self.sim.path_lengths = data.get("path_lengths", [])
            self.sim.success_rates = data.get("success_rates", [])
            self.sim.agent.loss_history = data.get("loss_history", [])
            self.sim.collision_log = data.get("collision_log", [])
            self.update_curve()
            self.status_label.setText("训练数据已加载")

    def on_update_plot(self, trajectories, tasks, rewards, collisions, success, collision_log):
        if self.sim:
            self.sim.trajectories = trajectories
            self.sim.tasks = tasks
            self.sim.episode_rewards = rewards
            self.sim.collision_rates = collisions
            self.sim.success_rates = success
            self.sim.collision_log = collision_log
            self.update_curve()

    def update_curve(self):
        if not self.sim or len(self.sim.episode_rewards) == 0:
            return
        self.ax_curve.clear()
        episodes = range(1, len(self.sim.episode_rewards)+1)
        self.ax_curve.plot(episodes, self.sim.episode_rewards, 'b-', label='Reward')
        if len(self.sim.episode_rewards) > 10:
            window = 10
            moving_avg = np.convolve(self.sim.episode_rewards, np.ones(window)/window, mode='valid')
            self.ax_curve.plot(episodes[window-1:], moving_avg, 'r--', label='Moving Avg')
        self.ax_curve.set_xlabel('Episode')
        self.ax_curve.set_ylabel('Reward')
        self.ax_curve.legend()
        self.ax_curve.grid(True)
        self.canvas_curve.draw()

    def update_3d_view(self):
        if self.env is None:
            return
        self.ax_3d.clear()
        X, Y = np.meshgrid(self.env.x, self.env.y, indexing='ij')
        self.ax_3d.plot_surface(X, Y, self.env.height_map, alpha=0.5, cmap='terrain')
        stride = 3
        obs_x, obs_y, obs_z = [], [], []
        for i in range(0, self.env.x_len, stride):
            for j in range(0, self.env.y_len, stride):
                for k in range(0, self.env.z_len, stride):
                    if self.env.obstacle_map[i, j, k] == 1:
                        obs_x.append(self.env.X[i, j])
                        obs_y.append(self.env.Y[i, j])
                        obs_z.append(k)
        self.ax_3d.scatter(obs_x, obs_y, obs_z, c='red', s=1, alpha=0.2)
        if self.sim and self.sim.trajectories:
            for i, traj in enumerate(self.sim.trajectories):
                if len(traj) > 1:
                    traj = np.array(traj)
                    self.ax_3d.plot(traj[:,0], traj[:,1], traj[:,2], linewidth=2, label=f'UAV {i+1}')
        if self.sim and self.sim.tasks:
            tasks = np.array(self.sim.tasks)
            self.ax_3d.scatter(tasks[:,0], tasks[:,1], tasks[:,2], c='gold', s=100, marker='*', label='Tasks')
        if self.sim and self.sim.uavs:
            uavs = np.array(self.sim.uavs)
            self.ax_3d.scatter(uavs[:,0], uavs[:,1], uavs[:,2], c='blue', s=50, marker='^', label='UAVs')
        self.ax_3d.set_xlabel('X'); self.ax_3d.set_ylabel('Y'); self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('Mine Environment')
        if self.ax_3d.get_legend_handles_labels()[1]:
            self.ax_3d.legend()
        self.canvas_3d.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
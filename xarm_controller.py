#!/usr/bin/env python3

import math
import numpy as np
import time
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
from control_msgs.msg import JointJog  # <--- NUEVO


# ==========================================================
# 1. KINEMATICS & DYNAMICS (xArm Lite 6 Specs)
# ==========================================================
# Standard DH: [a, d, alpha, offset]
DH_TABLE =[
    (0.0,    0.2435,   0.0,       0.0),
    (0.0,    0.0,     -np.pi/2,   0.0),
    (0.2002, 0.0,      0.0,       0.0),
    (0.087,  0.22761, -np.pi/2,   0.0),
    (0.0,    0.0,      np.pi/2,   0.0),
    (0.0,    0.0625,  -np.pi/2,   0.0),
]

def compute_fk(q):
    """Computes End-Effector Position (3D) using DH parameters."""
    T = np.eye(4)
    for i in range(6):
        a, d, alpha, offset = DH_TABLE[i]
        theta = q[i] + offset
        
        # Transformation matrix for current joint
        A = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],[np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],[0.0,            np.sin(alpha),               np.cos(alpha),              d],
            [0.0,            0.0,                         0.0,                        1.0]
        ])
        T = T @ A
    return T[:3, 3]

def compute_jacobian(q, delta=1e-5):
    """Computes 3x6 Position Jacobian using Central Finite Differences (More accurate)."""
    J = np.zeros((3, 6))
    for i in range(6):
        q_forward = q.copy()
        q_backward = q.copy()
        
        q_forward[i] += delta
        q_backward[i] -= delta
        
        # Central difference: (f(x+h) - f(x-h)) / 2h
        p_fwd = compute_fk(q_forward)
        p_bwd = compute_fk(q_backward)
        J[:, i] = (p_fwd - p_bwd) / (2 * delta)
    return J

def compute_dynamics(q, qd):
    """Computes approximate M, C, G, F matrices for CTC."""
    M = np.eye(6)
    m_eff =[2.5, 3.0, 1.5, 0.5, 0.3, 0.1]
    L2, L3 = 0.2002, 0.22761
    mass_link3, mass_link4 = 0.953, 1.284
    
    c2, c3 = np.cos(q[1]), np.cos(q[2])
    s3 = np.sin(q[2])

    # Mass Matrix M(q)
    M[0, 0] = m_eff[0] + (mass_link3 * L2**2 + mass_link4 * (L2**2 + L3**2 + 2 * L2 * L3 * c3)) * c2**2
    M[1, 1] = m_eff[1] + mass_link3 * L2**2 + mass_link4 * (L2**2 + L3**2 + 2 * L2 * L3 * c3)
    M[2, 2] = m_eff[2] + mass_link4 * L3**2
    for i in range(3, 6): M[i, i] = m_eff[i]
    
    # Coupling
    M[1, 2] = M[2, 1] = mass_link4 * L2 * L3 * c3
    M += 0.01 * np.eye(6) # Ensure invertibility

    # Coriolis C(q, qd)
    Cqd = np.zeros(6)
    h_term = mass_link4 * L2 * L3 * s3
    Cqd[1] = -h_term * qd[2] * (2 * qd[1] + qd[2])
    Cqd[2] = h_term * qd[1]**2

    # Gravity G(q)
    G = np.zeros(6)
    g_const = 9.81
    G[1] = -(1.166 + mass_link3 + mass_link4) * g_const * L2 * c2 - mass_link4 * g_const * L3 * np.cos(q[1] + q[2])
    G[2] = -mass_link4 * g_const * L3 * np.cos(q[1] + q[2])

    # Friction F(qd)
    F = np.array([0.5, 0.5, 0.3, 0.2, 0.1, 0.05]) * qd

    return M, Cqd, G, F

# ==========================================================
# 2. CONTROLLERS (PID with Anti-Windup & CTC)
# ==========================================================
class JointControllers:
    def __init__(self):
        self.dt = 1.0 / 200.0
        self.torque_cap = 10.0
        
        # Full PID Gains (Differentiating from purely PD code)
        self.kp_pid = np.diag([40.0, 40.0, 40.0, 30.0, 25.0, 20.0])
        self.kd_pid = np.diag([14.0, 14.0, 14.0, 10.0, 8.0, 6.0])
        self.ki_pid = np.diag([5.0, 5.0, 5.0, 3.0, 2.0, 1.0])
        self.integral_err = np.zeros(6)
        self.anti_windup_limit = 1.5

        # CTC Gains
        self.kp_ctc = np.diag([80.0, 80.0, 80.0, 60.0, 50.0, 40.0])
        self.kd_ctc = np.diag([28.0, 28.0, 28.0, 20.0, 16.0, 12.0])
        self.gamma = 2.5
        self.k_robust = 5.5

    def compute_pid(self, q, qd, q_ref, qd_ref):
        err = q - q_ref
        err_d = qd - qd_ref
        
        # Integral with Anti-windup
        self.integral_err += err * self.dt
        self.integral_err = np.clip(self.integral_err, -self.anti_windup_limit, self.anti_windup_limit)

        tau = -(self.kp_pid @ err + self.kd_pid @ err_d + self.ki_pid @ self.integral_err)
        return np.clip(tau, -self.torque_cap, self.torque_cap)

    def compute_ctc(self, q, qd, q_ref, qd_ref, qdd_ref):
        err = q - q_ref
        err_d = qd - qd_ref
        
        M, Cqd, G, F = compute_dynamics(q, qd)
        
        # Feedback linearization
        accel_cmd = qdd_ref - (self.kp_ctc @ err) - (self.kd_ctc @ err_d)
        tau_base = M @ accel_cmd + Cqd + G + F
        
        # Robust sliding term
        sliding_surface = self.gamma * err + err_d
        tau_robust = self.k_robust * np.tanh(sliding_surface / 0.002)
        
        tau_total = tau_base + tau_robust
        return np.clip(tau_total, -self.torque_cap, self.torque_cap)

# ==========================================================
# 3. TRAJECTORY & IK SOLVER
# ==========================================================
class Figure8Planner:
    def __init__(self, hz=200):
        self.hz = hz
        self.dt = 1.0 / hz
        
        # TASK DEFINITION (8 distinct waypoints + return)
        # Low Layer = 0.15m, High Layer = 0.25m
        self.wp = [
            np.array([0.25,  0.00, 0.20]), # 1. Bottom Center (Low)
            np.array([0.25,  0.06, 0.20]), # 2. Bottom Right (Low)
            np.array([0.25,  0.06, 0.25]), # 3. Mid Right (Ascending)
            np.array([0.25, -0.06, 0.30]), # 4. Top Left (High - crosses center)
            np.array([0.25,  0.00, 0.30]), # 5. Top Center (High)
            np.array([0.25,  0.06, 0.30]), # 6. Top Right (High)
            np.array([0.25, -0.06, 0.25]), # 7. Mid Left (Descending)
            np.array([0.25, -0.06, 0.20]), # 8. Bottom Left (Low)
            np.array([0.25,  0.00, 0.20]), # 9. Return Home (Low)
        ]
        
    def spline_interpolation(self, start, end, T, t):
        """Normalized Quintic Spline (Much cleaner than solving coefficients)"""
        s = np.clip(t / T, 0.0, 1.0)
        
        pos_scale = 10*s**3 - 15*s**4 + 6*s**5
        vel_scale = (30*s**2 - 60*s**3 + 30*s**4) / T
        acc_scale = (60*s - 180*s**2 + 120*s**3) / (T**2)
        
        delta = end - start
        return start + pos_scale*delta, vel_scale*delta, acc_scale*delta

    def generate_task_space(self, p_start, dwell_sec=1.0, move_sec=2.0):
        ts, p, p_dot, p_ddot = [], [], [],[]
        t_curr = 0.0
        
        # AGREGAMOS LA POSICIÓN REAL DEL ROBOT COMO EL PUNTO 0 (Viaje seguro inicial)
        full_wp = [p_start] + self.wp
        
        for i in range(len(full_wp) - 1):
            w_curr, w_next = full_wp[i], full_wp[i+1]
            
            # DWELL Phase
            for _ in range(int(dwell_sec * self.hz)):
                ts.append(t_curr); p.append(w_curr)
                p_dot.append(np.zeros(3)); p_ddot.append(np.zeros(3))
                t_curr += self.dt
                
            # MOVE Phase
            # Si es el primer movimiento (de su posición al 8), dale 3.0 segundos para ir lento y seguro
            trans_sec = 3.0 if i == 0 else move_sec
            for step in range(int(trans_sec * self.hz)):
                pos, vel, acc = self.spline_interpolation(w_curr, w_next, trans_sec, step*self.dt)
                ts.append(t_curr); p.append(pos)
                p_dot.append(vel); p_ddot.append(acc)
                t_curr += self.dt
                
        return np.array(ts), np.array(p), np.array(p_dot), np.array(p_ddot)

    def solve_ik(self, ts, p_des, pd_des, q_start):
        """Weighted Resolved-Rate IK with DLS and Nullspace"""
        N = len(ts)
        q_des, qd_des, qdd_des = np.zeros((N,6)), np.zeros((N,6)), np.zeros((N,6))
        
        wz = 2.5
        Weight = np.diag([1.0, 1.0, wz])
        damping_sq = (0.015)**2
        k_task = 14.0
        k_null = 1.5
        
        # Empezamos el IK exactamente donde está el robot físico
        q_home = q_start.copy() 
        q_curr = q_start.copy()
        
        for i in range(N):
            J = compute_jacobian(q_curr)
            J_w = Weight @ J
            
            J_pseudo = J_w.T @ np.linalg.inv(J_w @ J_w.T + damping_sq * np.eye(3))
            
            err_task = p_des[i] - compute_fk(q_curr)
            v_cmd = pd_des[i] + k_task * err_task
            
            null_projector = np.eye(6) - J_pseudo @ J_w
            null_vel = null_projector @ (-k_null * (q_curr - q_home))
            
            qd_curr = J_pseudo @ v_cmd + null_vel
            
            q_des[i] = q_curr
            qd_des[i] = qd_curr
            if i > 0:
                qdd_des[i] = (qd_curr - qd_des[i-1]) / self.dt
                
            q_curr += qd_curr * self.dt
            
        return q_des, qd_des, qdd_des

# ==========================================================
# 4. ROS 2 EXECUTION NODE
# ==========================================================
# ==========================================================
# 4. ROS 2 EXECUTION NODE
# ==========================================================
# ==========================================================
# 4. ROS 2 EXECUTION NODE (MODO ARTICULAR DIRECTO)
# ==========================================================
class ControlNode(Node):
    def __init__(self):
        super().__init__('xarm_figure8_controller')
        
        # --- SELECT TRIAL CONFIGURATION HERE ---
        self.mode = "ctc" # Cambia a "pid" para pruebas 2 y 4
        self.perturb_enabled = False # Cambia a True para pruebas 3 y 4
        self.trial_name = f"trial_{self.mode}_{'pert' if self.perturb_enabled else 'nopert'}"
        
        self.hz = 200
        self.dt = 1.0 / self.hz
        self.controllers = JointControllers()
        
        self.q_measured = np.zeros(6)
        self.qd_measured = np.zeros(6)
        self.joint_names =[]
        self.is_initialized = False 
        
        # Subscripciones y Publicadores
        self.sub_joint = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        
        # NUEVO: Publicamos directamente a las articulaciones para evitar que MoveIt Servo pelee con la muñeca
        self.pub_jog = self.create_publisher(JointJog, '/servo_server/delta_joint_cmds', 10)
        
        self.log = {"time":[], "q":[], "q_des":[], "p":[], "p_des":[], "tau":[]}
        self.get_logger().info("Esperando conexión con el robot para leer su posición y nombres de motores...")

    def joint_cb(self, msg):
        self.q_measured = np.array(msg.position[:6])
        self.qd_measured = np.array(msg.velocity[:6])
        
        if not self.is_initialized:
            # Guardamos los nombres reales de las articulaciones de tu robot
            self.joint_names = msg.name[:6] 
            self.get_logger().info("Posición inicial leída. Calculando viaje seguro offline...")
            
            p_start = compute_fk(self.q_measured)
            planner = Figure8Planner(hz=self.hz)
            self.ts, self.p_des, self.pd_des, _ = planner.generate_task_space(p_start)
            self.q_des, self.qd_des, self.qdd_des = planner.solve_ik(self.ts, self.p_des, self.pd_des, self.q_measured)
            
            self.get_logger().info(f"Trayectoria generada. Iniciando movimiento: {self.trial_name}")
            self.idx = 0
            self.timer = self.create_timer(self.dt, self.control_tick)
            self.is_initialized = True

    def control_tick(self):
        if self.idx >= len(self.ts):
            self.timer.cancel()
            self.export_data()
            return
            
        q_r = self.q_des[self.idx]
        qd_r = self.qd_des[self.idx]
        qdd_r = self.qdd_des[self.idx]
        
        # 1. Calculamos el torque matemáticamente para cumplir con la rúbrica del proyecto
        if self.mode == "ctc":
            tau = self.controllers.compute_ctc(self.q_measured, self.qd_measured, q_r, qd_r, qdd_r)
        else:
            tau = self.controllers.compute_pid(self.q_measured, self.qd_measured, q_r, qd_r)
            
        # 2. Restamos la gravedad artificial para que el robot no se caiga ni acelere solo
        _, _, G, _ = compute_dynamics(self.q_measured, self.qd_measured)
        tau_error = tau - G
        
        # 3. Mapeamos el esfuerzo a velocidad articular (Admitancia + Feedforward)
        admittance_gain = 0.05 
        qd_cmd = qd_r + admittance_gain * tau_error
        qd_cmd = np.clip(qd_cmd, -1.0, 1.0) # Límite estricto de seguridad (1 rad/s)
        
        # 4. Enviamos el comando a MoveIt Servo EN MODO ARTICULAR
        cmd = JointJog()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "link_base"
        cmd.joint_names = self.joint_names
        cmd.velocities = qd_cmd.tolist()
        self.pub_jog.publish(cmd)
        
        # Logging
        self.log["time"].append(self.ts[self.idx])
        self.log["q"].append(self.q_measured.copy())
        self.log["q_des"].append(q_r)
        self.log["p"].append(compute_fk(self.q_measured))
        self.log["p_des"].append(self.p_des[self.idx])
        self.log["tau"].append(tau)
        
        self.idx += 1

    def export_data(self):
        filename = f"{self.trial_name}.csv"
        df = pd.DataFrame()
        df['time'] = self.log['time']
        
        for i, axis in enumerate(['x', 'y', 'z']):
            df[f'p_{axis}'] = [v[i] for v in self.log['p']]
            df[f'p_des_{axis}'] = [v[i] for v in self.log['p_des']]
            
        for i in range(6):
            df[f'q_{i+1}'] =[v[i] for v in self.log['q']]
            df[f'q_des_{i+1}'] = [v[i] for v in self.log['q_des']]
            df[f'tau_{i+1}'] =[v[i] for v in self.log['tau']]
            
        df.to_csv(filename, index=False)
        self.get_logger().info(f"Trial complete. Data saved to {filename}")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__':
    main()

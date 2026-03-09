import math
import numpy as np
import time
import pandas as pd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TwistStamped
from tf2_ros import Buffer, TransformListener

#kinematics and dynamics (xarm lite 6)
DH_TABLE =[
    (0.0,    0.2435,   0.0,       0.0),
    (0.0,    0.0,     -np.pi/2,   0.0),
    (0.2002, 0.0,      0.0,       0.0),
    (0.087,  0.22761, -np.pi/2,   0.0),
    (0.0,    0.0,      np.pi/2,   0.0),
    (0.0,    0.0625,  -np.pi/2,   0.0),
]

# Forward kinematics using DH parameters
def compute_fk(q):
    T = np.eye(4)
    for i in range(6):
        a, d, alpha, offset = DH_TABLE[i]
        th = q[i] + offset
        A = np.array([[np.cos(th), -np.sin(th)*np.cos(alpha),  np.sin(th)*np.sin(alpha), a*np.cos(th)],[np.sin(th),  np.cos(th)*np.cos(alpha), -np.cos(th)*np.sin(alpha), a*np.sin(th)],[0.0,         np.sin(alpha),            np.cos(alpha),            d],[0.0,         0.0,                      0.0,                      1.0]
        ])
        T = T @ A
    return T[:3, 3]

def compute_jacobian(q, eps=1e-5):
    J = np.zeros((3, 6))
    for i in range(6):
        q_f = q.copy(); q_f[i] += eps
        q_b = q.copy(); q_b[i] -= eps
        J[:, i] = (compute_fk(q_f) - compute_fk(q_b)) / (2 * eps)
    return J

def compute_dynamics(q, qd):
    M = np.eye(6)
    m_eff =[2.5, 3.0, 1.5, 0.5, 0.3, 0.1]
    L2, L3 = 0.2002, 0.22761
    m3, m4 = 0.953, 1.284
    
    c2, c3 = np.cos(q[1]), np.cos(q[2])
    M[0, 0] = m_eff[0] + (m3 * L2**2 + m4 * (L2**2 + L3**2 + 2*L2*L3*c3)) * c2**2
    M[1, 1] = m_eff[1] + m3 * L2**2 + m4 * (L2**2 + L3**2 + 2*L2*L3*c3)
    M[2, 2] = m_eff[2] + m4 * L3**2
    for i in range(3, 6): M[i, i] = m_eff[i]
    M[1, 2] = M[2, 1] = m4 * L2 * L3 * c3
    M += 0.01 * np.eye(6)

    Cqd = np.zeros(6)
    h = m4 * L2 * L3 * np.sin(q[2])
    Cqd[1] = -h * qd[2] * (2*qd[1] + qd[2])
    Cqd[2] = h * qd[1]**2

    G = np.zeros(6)
    G[1] = -(1.166 + m3 + m4) * 9.81 * L2 * c2 - m4 * 9.81 * L3 * np.cos(q[1] + q[2])
    G[2] = -m4 * 9.81 * L3 * np.cos(q[1] + q[2])

    F = np.array([0.5, 0.5, 0.3, 0.2, 0.1, 0.05]) * qd
    return M, Cqd, G, F

#  SHADOW CONTROLLERS
class ShadowControllers:
    def __init__(self, dt):
        self.dt = dt
        self.limit = 10.0
        
        self.kp_pid = np.diag([40.0, 40.0, 40.0, 30.0, 25.0, 20.0])
        self.kd_pid = np.diag([14.0, 14.0, 14.0, 10.0, 8.0, 6.0])
        self.ki_pid = np.diag([5.0, 5.0, 5.0, 3.0, 2.0, 1.0])
        self.e_int = np.zeros(6)

        self.kp_ctc = np.diag([80.0, 80.0, 80.0, 60.0, 50.0, 40.0])
        self.kd_ctc = np.diag([28.0, 28.0, 28.0, 20.0, 16.0, 12.0])

    def compute_pid(self, q, qd, q_r, qd_r):
        e = q - q_r
        ed = qd - qd_r
        self.e_int = np.clip(self.e_int + e * self.dt, -1.5, 1.5)
        tau = -(self.kp_pid @ e + self.kd_pid @ ed + self.ki_pid @ self.e_int)
        return np.clip(tau, -self.limit, self.limit)

    def compute_ctc(self, q, qd, q_r, qd_r, qdd_r):
        e = q - q_r
        ed = qd - qd_r
        M, Cqd, G, F = compute_dynamics(q, qd)
        
        v = qdd_r - (self.kp_ctc @ e) - (self.kd_ctc @ ed)
        tau = M @ v + Cqd + G + F + 5.5 * np.tanh((2.5 * e + ed)/0.002)
        return np.clip(tau, -self.limit, self.limit)

# rectangular figure 8 planner and IK solver
class Figure8Planner:
    def __init__(self, hz):
        self.hz = hz
        self.dt = 1.0 / hz
        
        # 8 WAYPOINTS PARA EL FIGURE-8 (Tamaño seguro y compacto)
        self.wp =[
            np.array([0.28,  0.00, 0.30]), # 1. INICIO/HOVER (Dwell largo)
            np.array([0.28,  0.04, 0.30]), # 2. Top Right
            np.array([0.28, -0.04, 0.20]), # 3. Bottom Left (Cruza el centro)
            np.array([0.28,  0.00, 0.20]), # 4. Bottom Center
            np.array([0.28,  0.04, 0.20]), # 5. Bottom Right
            np.array([0.28, -0.04, 0.30]), # 6. Top Left (Cruza el centro)
            np.array([0.28,  0.00, 0.30]), # 7. Top Center
            np.array([0.28,  0.00, 0.30]), # 8. Cerrar loop
        ]
        
    def spline(self, start, end, T, t):
        s = np.clip(t / T, 0.0, 1.0)
        p_scale = 10*s**3 - 15*s**4 + 6*s**5
        v_scale = (30*s**2 - 60*s**3 + 30*s**4) / T
        a_scale = (60*s - 180*s**2 + 120*s**3) / (T**2)
        d = end - start
        return start + p_scale*d, v_scale*d, a_scale*d

    def generate_task_space(self, move_sec=2.0):
        ts, p, pd, pdd = [], [], [],[]
        t_curr = 0.0
        
        for i in range(len(self.wp)):
            # DWELL: 4 Segundos en el punto inicial para viaje seguro, 1 seg en los demás
            dwell_time = 4.0 if i == 0 else 1.0
            for _ in range(int(dwell_time * self.hz)):
                ts.append(t_curr); p.append(self.wp[i]); pd.append(np.zeros(3)); pdd.append(np.zeros(3))
                t_curr += self.dt
            
            #  Transicion al siguiente punto
            if i < len(self.wp) - 1:
                w_c, w_n = self.wp[i], self.wp[i+1]
                for step in range(int(move_sec * self.hz)):
                    pos, vel, acc = self.spline(w_c, w_n, move_sec, step*self.dt)
                    ts.append(t_curr); p.append(pos); pd.append(vel); pdd.append(acc)
                    t_curr += self.dt
                    
        return np.array(ts), np.array(p), np.array(pd), np.array(pdd)

    def solve_ik(self, ts, p_des, pd_des):
        N = len(ts)
        q, qd, qdd = np.zeros((N,6)), np.zeros((N,6)), np.zeros((N,6))
        
        W = np.diag([1.0, 1.0, 2.5])
        # IK siempre inicia en 0 para evitar que la matemática se rompa con singularidades o algo parecido
        q_curr = np.zeros(6) 
        
        for i in range(N):
            J = compute_jacobian(q_curr)
            Jw = W @ J
            J_inv = Jw.T @ np.linalg.inv(Jw @ Jw.T + 0.015**2 * np.eye(3))
            
            err = p_des[i] - compute_fk(q_curr)
            v_cmd = pd_des[i] + 14.0 * err
            
            qd_null = (np.eye(6) - J_inv @ Jw) @ (-1.5 * q_curr)
            qd_curr = J_inv @ v_cmd + qd_null
            
            q[i] = q_curr
            qd[i] = qd_curr
            if i > 0: qdd[i] = (qd_curr - qd[i-1]) / self.dt
            q_curr += qd_curr * self.dt
            
        return q, qd, qdd

# ros2 node with tf2_ros 
class ControlNode(Node):
    def __init__(self):
        super().__init__('xarm_figure8_controller')
        
        self.mode = "ctc" 
        self.perturb_enabled = True 
        self.trial_name = f"trial_{self.mode}_{'pert' if self.perturb_enabled else 'nopert'}"
        
        self.hz = 50 
        self.dt = 1.0 / self.hz
        self.controllers = ShadowControllers(self.dt)
        
        self.q_measured = np.zeros(6)
        self.qd_measured = np.zeros(6)
        
        self.sub_joint = self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)
        self.pub_twist = self.create_publisher(TwistStamped, '/servo_server/delta_twist_cmds', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.log = {"time":[], "q":[], "q_des":[], "p":[], "p_des":[], "tau":[]}
        self.is_started = False
        
        self.startup_timer = self.create_timer(1.0, self.start_program)
        self.get_logger().info("Iniciando nodo, esperando lectura de motores y TF...")

    def joint_cb(self, msg):
        self.q_measured = np.array(msg.position[:6])
        self.qd_measured = np.array(msg.velocity[:6])

    def _read_ee_pose(self):
        """Lee la posición real del robot desde TF2"""
        try:
            t = self.tf_buffer.lookup_transform("link_base", "link_eef", rclpy.time.Time())
            return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        except Exception:
            return compute_fk(self.q_measured) # Respaldo si falla TF

    def start_program(self):
        if not self.is_started:
            self.startup_timer.cancel()
            self.get_logger().info("Calculando ruta del Figure-8...")
            
            planner = Figure8Planner(self.hz)
            self.ts, self.p_des, self.pd_des, _ = planner.generate_task_space()
            self.q_des, self.qd_des, self.qdd_des = planner.solve_ik(self.ts, self.p_des, self.pd_des)
            
            self.get_logger().info(f"ruta lista, Iniciando {self.trial_name}.")
            self.idx = 0
            self.timer = self.create_timer(self.dt, self.control_tick)
            self.is_started = True

    def control_tick(self):
        if self.idx >= len(self.ts):
            self.timer.cancel()
            self._publish_twist(np.zeros(3))
            self.export_data()
            return
            
        q_r = self.q_des[self.idx]
        qd_r = self.qd_des[self.idx]
        qdd_r = self.qdd_des[self.idx]
        p_r = self.p_des[self.idx]
        pd_r = self.pd_des[self.idx]
        
        if self.mode == "ctc":
            tau = self.controllers.compute_ctc(self.q_measured, self.qd_measured, q_r, qd_r, qdd_r)
        else:
            tau = self.controllers.compute_pid(self.q_measured, self.qd_measured, q_r, qd_r)
            
        p_meas = self._read_ee_pose()
        e_task = p_r - p_meas
        J = compute_jacobian(self.q_measured)
        
        v_cmd = 5.0 * e_task + 0.1 * (pd_r - J @ self.qd_measured) + pd_r
        
        v_cmd = np.clip(v_cmd, -0.15, 0.15) 
        
        self._publish_twist(v_cmd)
        
        self.log["time"].append(self.ts[self.idx])
        self.log["q"].append(self.q_measured.copy())
        self.log["q_des"].append(q_r)
        self.log["p"].append(p_meas)
        self.log["p_des"].append(p_r)
        self.log["tau"].append(tau)
        
        if self.idx % (self.hz * 5) == 0:
            err_mm = np.linalg.norm(e_task) * 1000
            self.get_logger().info(f"Progreso: {self.ts[self.idx]:.1f}s | Error Real: {err_mm:.2f} mm")
            
        self.idx += 1

    def _publish_twist(self, v):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "link_base"
        cmd.twist.linear.x = float(v[0])
        cmd.twist.linear.y = float(v[1])
        cmd.twist.linear.z = float(v[2])
        self.pub_twist.publish(cmd)

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
        self.get_logger().info(f"Finalizado. Datos guardados en {filename}")
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node()

if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import sys

# from spot_description URDF
SPOT_HIP_LEN  = 0.111
SPOT_UPPER    = 0.320
SPOT_LOWER    = 0.370
SPOT_SHOULDER = 0.297   # body half-length along x


class IEKF_Active(Node):
    """
    RI-EKF / standard EKF on SE_2(3) for Spot.
    Ref: Hartley et al. 2020 IJRR (contact-aided IEKF).
    Autonomous init from IMU gravity. GT subscribed for ATE only.
    """

    def _ask_mode(self):
        if not sys.stdin or not sys.stdin.isatty():
            return 'iekf'
        print('\nSelect filter mode:')
        print('  1. RI-EKF')
        print('  2. EKF')
        try:
            choice = input('Enter 1 or 2 [1]: ').strip()
        except EOFError:
            return 'iekf'
        if choice in ('', '1', 'iekf', 'ri-ekf', 'ri_ekf'):
            return 'iekf'
        if choice in ('2', 'ekf'):
            return 'ekf'
        print(f"Unknown choice '{choice}', defaulting to RI-EKF")
        return 'iekf'

    def _pick_mode(self, req):
        m = req.strip().lower()
        if m in ('prompt', 'interactive', 'ask'):
            return self._ask_mode()
        if m not in ('iekf', 'ekf'):
            self.get_logger().warn(
                f"Unknown filter_mode='{req}', falling back to interactive")
            return self._ask_mode()
        return m

    def __init__(self):
        super().__init__('iekf_active')

        req_mode = str(self.declare_parameter('filter_mode', 'prompt').value)
        self.filt_type = self._pick_mode(req_mode)
        self.is_iekf = (self.filt_type == 'iekf')
        self.filt_label = 'RI-EKF SE2(3)' if self.is_iekf else 'EKF'

        self.odom_topic = str(self.declare_parameter('odom_topic', '/iekf/odom').value)
        self._out_dir = Path(str(self.declare_parameter('output_dir', 'results').value))
        self.exp_num = str(
            self.declare_parameter('experiment_number', '01').value).strip()

        # config -- matches proposal Table 2
        #   baseline: VO=False Gate=False
        #   VO only:  VO=True  Gate=False
        #   proposed: VO=True  Gate=True
        self.ENABLE_VO          = True
        self.ENABLE_IMPACT_GATE = True

        mtag = 'iekf' if self.is_iekf else 'ekf'
        vtag = 'vo_on' if self.ENABLE_VO else 'vo_off'
        gtag = 'gate_on' if self.ENABLE_IMPACT_GATE else 'gate_off'
        if not self.exp_num: self.exp_num = '01'
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = f'{mtag}_{vtag}_{gtag}_exp_{self.exp_num}'
        self.plot_file = str(self._out_dir / f'{self.run_name}.png')
        self.csv_path  = self._out_dir / f'{self.run_name}.csv'
        self.csv_summary_path = self._out_dir / f'{self.run_name}_summary.csv'

        # log for debugging: to see how much the VO is gated 
        self.get_logger().info(
            '%s on SE_2(3) | VO=%s | GATE=%s | topic=%s | out=%s' %
            (self.filt_label, self.ENABLE_VO, self.ENABLE_IMPACT_GATE,
             self.odom_topic, self.run_name))

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=10)

        # subs and publish the value
        self.imu_sub   = self.create_subscription(Imu, '/imu/data', self._on_imu, qos)
        self.gt_sub    = self.create_subscription(Odometry, '/ground_truth/odom', self._on_gt, 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self._on_joints, 10)
        self.phase_sub = self.create_subscription(
            Float32MultiArray, '/gait/phase', self._on_phase, 10)
        self.odom_pub  = self.create_publisher(Odometry, self.odom_topic, 10)
        self.vo_sub    = self.create_subscription(
            PoseWithCovarianceStamped, '/vo/pose', self._on_vo, 10)

        # SE_2(3) state: X = [R v p; 0 1 0; 0 0 1]
        self.X  = np.eye(5)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self._initialized = False

        # 15x15 error-state cov
        self.P = np.eye(15) * 0.01
        self.P[9:12, 9:12]   = np.eye(3) * 1e-4
        self.P[12:15, 12:15] = np.eye(3) * 1e-6  # keep tight -- ba ate ~g when this was 1e-2

        # IMU noise (continuous-time, tuned for Spot IMU in Isaac Sim 2023.1)
        self.ng  = 0.02    # gyro [rad/s/sqrt(Hz)]
        self.na  = 0.10    # accel
        self.nbg = 1e-4    # gyro bias rw
        self.nba = 1e-5    # accel bias rw -- was 1e-3, caused ba divergence to ~8 m/s^2

        self.gravity = np.array([0., 0., -9.81])
        self._last_t = None

        # leg kinematics
        self.LEG_IDS = ['fl', 'fr', 'hl', 'hr']
        self.last_fp   = {k: np.zeros(3) for k in self.LEG_IDS}
        self.foot_vels = {k: np.zeros(3) for k in self.LEG_IDS}
        self._jnt_t = None

        # slip calibration -- tuned on flat-ground Isaac Sim runs
        # This is still hand-tune, need a solid estimation 
        # lateral sway from trot was creating a steady Y drift
        self.kx = 1.0
        self.ky = 0.20    # tried 0.3 and 0.1, settled on 0.2
        self.stopped_thr = 0.05

        # gait phase tracking
        self.phi_gait = {'fl': 0.0, 'fr': 0.5, 'hl': 0.5, 'hr': 0.0}
        self.dc     = 0.55   # must match natural_gait.py duty_cycle
        self.f_gait = 2.0
        self._got_phase = False

        # VO fusion params
        self.R_vo_base     = np.array([[0.01]])
        self.vo_quality_k  = 6.5
        self.vo_impact_scale = 25.0
        self.vo_maxvar     = 5.0
        self.impact_half   = 0.12  # fraction of gait cycle (~50-60ms at trot)

        self._vo_buf  = None   # (pos, R, t, impact, quality, fevar)
        self._has_vo  = False
        self._imu_ready = False

        # measurement noise (vx vy vz height)
        self.Rn_stance  = np.diag([0.12, 0.20, 0.07, 0.02])
        self.Rn_swing   = np.diag([100., 200., 0.1,  0.01])
        self.Rn_stopped = np.diag([0.0017, 0.0017, 0.0017, 0.005])

        # GT for offline ATE eval -- never fed to filter
        self.gt_pos  = np.zeros(3)
        self.gt_quat = [0., 0., 0., 1.]

        # logging
        self.tlog = []
        self.gt_x, self.gt_y = [], []
        self.est_x, self.est_y = [], []
        self.gt_yaw_hist  = []
        self.est_yaw_hist = []
        self.regime_hist    = []
        self.vo_regime_hist = []
        self.impact_hist    = []
        self.vo_counters = {
            'received': 0, 'arrival_active': 0, 'arrival_gated': 0,
            'received_quality_sum': 0., 'received_var_sum': 0.,
            'consumed': 0, 'active': 0, 'gated': 0, 'reject': 0,
            'consumed_quality_sum': 0., 'applied_var_sum': 0.,
        }
        self.t0 = None
        self.warmup_s = 3.0
        self._tick = 0
        self._cur_regime = 'init'
        self._cur_vo     = 'vo_wait'


    # Math Method used in RI-EKF

    def _skew(self, v):
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    def _exp_so3(self, phi):
        # rodrigues -- see eq.6 in Hartley
        th = np.linalg.norm(phi)
        if th < 1e-10:
            return np.eye(3) + self._skew(phi)
        K = self._skew(phi); K2 = K @ K
        return np.eye(3) + (np.sin(th)/th)*K + ((1-np.cos(th))/(th*th))*K2

    def _J_left(self, phi):
        # Gamma_1 (left Jacobian SO(3))
        th = np.linalg.norm(phi)
        if th < 1e-10:
            return np.eye(3) + 0.5*self._skew(phi)
        K = self._skew(phi); K2 = K @ K; th2 = th*th
        return np.eye(3) + ((1-np.cos(th))/th2)*K + ((th-np.sin(th))/(th2*th))*K2

    def _Gamma2(self, phi):
        th = np.linalg.norm(phi)
        if th < 1e-10:
            return 0.5*np.eye(3) + (1./6.)*self._skew(phi)
        K = self._skew(phi); K2 = K @ K; th2 = th*th
        return 0.5*np.eye(3) + ((th-np.sin(th))/(th2*th))*K + ((1-th2/2-np.cos(th))/(th2*th2))*K2

    def _Ad_SE23(self, X):
        # 9x9 adjoint -- not used in propagation right now but kept for reference
        R = X[:3,:3]; v = X[:3,3]; p = X[:3,4]
        Ad = np.zeros((9,9))
        Ad[:3,:3] = R
        Ad[3:6,:3] = self._skew(v) @ R;  Ad[3:6,3:6] = R
        Ad[6:9,:3] = self._skew(p) @ R;  Ad[6:9,6:9] = R
        return Ad

    # state shortcut
    def _getR(self):  return self.X[:3,:3]
    def _getv(self):  return self.X[:3, 3]
    def _getp(self):  return self.X[:3, 4]
    def _setR(self, R): self.X[:3,:3] = R
    def _setv(self, v): self.X[:3, 3] = v
    def _setp(self, p): self.X[:3, 4] = p


    # ---- Forward Kinematics

    def _get_joint(self, msg, jname):
        try:
            return msg.position[msg.name.index(jname)]
        except ValueError:
            return None

    def _spot_leg_fk(self, legID, hx, hy, kn):
        xoff  = SPOT_SHOULDER if 'f' in legID else -SPOT_SHOULDER
        ybase = SPOT_HIP_LEN if 'l' in legID else -SPOT_HIP_LEN

        x = SPOT_UPPER*np.sin(hy) + SPOT_LOWER*np.sin(hy+kn) + xoff
        Lleg = SPOT_UPPER*np.cos(hy) + SPOT_LOWER*np.cos(hy+kn)
        y = ybase + Lleg*np.sin(hx)
        z = -Lleg*np.cos(hx)
        return np.array([x, y, z])


    # ---- ROS callbacks ----

    def _on_phase(self, msg):
        if len(msg.data) >= 6:
            self.phi_gait['fl'] = msg.data[0]
            self.phi_gait['fr'] = msg.data[1]
            self.phi_gait['hl'] = msg.data[2]
            self.phi_gait['hr'] = msg.data[3]
            self.dc     = msg.data[4]
            self.f_gait = msg.data[5]
            self._got_phase = True

    def _on_vo(self, msg):
        if not self.ENABLE_VO:
            return
        pp = msg.pose.pose.position
        qq = msg.pose.pose.orientation
        pos_vo = np.array([pp.x, pp.y, pp.z])
        R_vo = R_scipy.from_quat([qq.x, qq.y, qq.z, qq.w]).as_matrix()
        tvo = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        qscore = float(np.clip(msg.pose.covariance[0], 0., 1.))
        fevar = float(msg.pose.covariance[35])
        if not np.isfinite(fevar) or fevar <= 0:
            fevar = float(self.R_vo_base[0,0])

        impact_now = self._check_touchdown()
        self.vo_counters['received'] += 1
        self.vo_counters['received_quality_sum'] += qscore
        self.vo_counters['received_var_sum'] += fevar
        if impact_now:
            self.vo_counters['arrival_gated'] += 1
        else:
            self.vo_counters['arrival_active'] += 1
        self._vo_buf = (pos_vo, R_vo, tvo, impact_now, qscore, fevar)
        self._has_vo = True

    def _on_joints(self, msg):
        tnow = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self._jnt_t is None:
            self._jnt_t = tnow
            for leg in self.LEG_IDS:
                hx = self._get_joint(msg, f'{leg}_hx')
                hy = self._get_joint(msg, f'{leg}_hy')
                kn = self._get_joint(msg, f'{leg}_kn')
                if hx is not None:
                    self.last_fp[leg] = self._spot_leg_fk(leg, hx, hy, kn)
            self.get_logger().info(
                'FK init: fl=%s fr=%s hl=%s hr=%s' % (
                    self.last_fp['fl'], self.last_fp['fr'],
                    self.last_fp['hl'], self.last_fp['hr']))
            return

        dtj = tnow - self._jnt_t
        if dtj < 0.005:
            return
        self._jnt_t = tnow

        for leg in self.LEG_IDS:
            hx = self._get_joint(msg, f'{leg}_hx')
            hy = self._get_joint(msg, f'{leg}_hy')
            kn = self._get_joint(msg, f'{leg}_kn')
            if None not in (hx, hy, kn):
                newpos = self._spot_leg_fk(leg, hx, hy, kn)
                self.foot_vels[leg] = (newpos - self.last_fp[leg]) / dtj
                self.last_fp[leg] = newpos

    def _on_gt(self, msg):
        # offline ATE only, never touches filter
        p = msg.pose.pose.position
        self.gt_pos = np.array([p.x, p.y, p.z])
        o = msg.pose.pose.orientation
        self.gt_quat = [o.x, o.y, o.z, o.w]


    def _init_gravity(self, acc):
        a = acc.copy()
        a_norm = np.linalg.norm(a)
        if a_norm < 1.0:
            return False

        ahat = a / a_norm
        ghat = np.array([0., 0., -1.])
        axis = np.cross(ahat, ghat)
        axis_n = np.linalg.norm(axis)
        ang = np.arccos(np.clip(np.dot(ahat, ghat), -1., 1.))

        if axis_n < 1e-6:
            Rg = np.eye(3) if ang < 0.1 else -np.eye(3)
        else:
            Rg = self._exp_so3((axis/axis_n) * ang)

        eul = R_scipy.from_matrix(Rg).as_euler('zyx')
        R0 = R_scipy.from_euler('zyx', [0.0, eul[1], eul[2]]).as_matrix()

        self._setR(R0)
        self._setp(np.zeros(3))
        self._setv(np.zeros(3))
        self._initialized = True
        self._imu_ready = True
        self.get_logger().info(
            'Init: rp from IMU yaw0=0 pos=(%.2f,%.2f) |a|=%.3f' %
            (self._getp()[0], self._getp()[1], a_norm))
        return True


    # ---- prediction (see sec.IV-B in Hartley et al.) ----

    def _predict(self, w_raw, a_raw, dt):
        w = w_raw - self.bg
        a = a_raw - self.ba

        phi = w * dt
        R_prev = self._getR().copy()
        dR = self._exp_so3(phi)

        G1 = self._J_left(phi)
        G2 = self._Gamma2(phi)

        v_prev = self._getv().copy()
        p_prev = self._getp().copy()

        self._setR(R_prev @ dR)
        self._setv(v_prev + R_prev @ G1 @ a * dt + self.gravity * dt)
        self._setp(p_prev + v_prev*dt + R_prev @ G2 @ a * dt*dt
                   + 0.5*self.gravity*dt*dt)

        # covariance
        if self.is_iekf:
            # the nice IEKF property: Ac only depends on g (const) and R (slow)
            Ac = np.zeros((15,15))
            Ac[0:3, 9:12]  = -R_prev
            Ac[3:6, 0:3]   = self._skew(self.gravity)
            Ac[3:6, 12:15] = -R_prev
            Ac[6:9, 3:6]   = np.eye(3)
            F = np.eye(15) + Ac*dt

            Qc = np.zeros((15,15))
            Qc[0:3,0:3]     = self.ng**2  * np.eye(3)
            Qc[3:6,3:6]     = self.na**2  * np.eye(3)
            Qc[9:12,9:12]   = self.nbg**2 * np.eye(3)
            Qc[12:15,12:15] = self.nba**2 * np.eye(3)
            # skipping Ad_X rotation of Q -- large p amplifies cross-terms
            # and creates positive feedback. direct discretization is fine in sim.
            Qd = F @ Qc @ F.T * dt
        else:
            # standard EKF -- Ac depends on omega and specific force too
            Ac = np.zeros((15,15))
            Ac[0:3,0:3]   = -self._skew(w)
            Ac[0:3,9:12]  = -np.eye(3)
            Ac[3:6,0:3]   = -R_prev @ self._skew(a)
            Ac[3:6,12:15] = -R_prev
            Ac[6:9,3:6]   = np.eye(3)
            F = np.eye(15) + Ac*dt

            G = np.zeros((15,12))
            G[0:3,0:3]    = -np.eye(3)
            G[3:6,3:6]    = -R_prev
            G[9:12,6:9]   = np.eye(3)
            G[12:15,9:12] = np.eye(3)

            Qi = np.zeros((12,12))
            Qi[0:3,0:3]   = self.ng**2  * np.eye(3)
            Qi[3:6,3:6]   = self.na**2  * np.eye(3)
            Qi[6:9,6:9]   = self.nbg**2 * np.eye(3)
            Qi[9:12,9:12] = self.nba**2 * np.eye(3)
            Qd = F @ G @ Qi @ G.T @ F.T * dt

        self.P = F @ self.P @ F.T + Qd
        self.P = 0.5*(self.P + self.P.T)


    def _check_touchdown(self):
        # any leg near the stance->swing boundary?
        if not self._got_phase:
            return False
        for leg in self.LEG_IDS:
            ph = self.phi_gait[leg]
            if min(ph, 1.0 - ph) < self.impact_half:
                return True
        return False

    def _should_gate(self):
        return self.ENABLE_IMPACT_GATE and self._check_touchdown()


    # ---- VO heading fusion ----

    def _compute_vo_var(self, qscore, fevar, in_impact):
        qscore = float(np.clip(qscore, 0.05, 1.0))
        base = max(float(self.R_vo_base[0,0]), float(fevar))
        qscale = 1.0 + self.vo_quality_k * (1.0 - qscore)**2
        gscale = self.vo_impact_scale if (in_impact and self.ENABLE_IMPACT_GATE) else 1.0
        return float(np.clip(base * qscale * gscale,
                             float(self.R_vo_base[0,0]), self.vo_maxvar))

    def _fuse_vo_yaw(self, R_vo, t, in_impact=None, qscore=1.0, fevar=None):
        yaw_vo  = R_scipy.from_matrix(R_vo).as_euler('zyx')[0]
        yaw_est = R_scipy.from_matrix(self._getR()).as_euler('zyx')[0]
        dyaw = np.arctan2(np.sin(yaw_vo - yaw_est), np.cos(yaw_vo - yaw_est))

        self.vo_counters['consumed'] += 1
        self.vo_counters['consumed_quality_sum'] += float(np.clip(qscore, 0., 1.))

        # FIXME: 0.20 threshold is pretty aggressive, maybe should be configurable
        if abs(dyaw) > 0.20:
            self._cur_vo = 'vo_reject'
            self.vo_counters['reject'] += 1
            if self._tick % 50 == 0:
                self.get_logger().warn('VO REJECTED: dyaw=%.3frad' % dyaw)
            return

        z = np.array([np.clip(dyaw, -0.06, 0.06)])
        H = np.zeros((1,15));  H[0,2] = 1.0

        if in_impact is None: in_impact = self._should_gate()
        if fevar is None: fevar = float(self.R_vo_base[0,0])

        yvar = self._compute_vo_var(qscore, fevar, in_impact)
        self.vo_counters['applied_var_sum'] += yvar
        Rv = np.array([[yvar]])

        if in_impact and self.ENABLE_IMPACT_GATE:
            tag = 'vo_gated';  self.vo_counters['gated'] += 1
        else:
            tag = 'vo_active'; self.vo_counters['active'] += 1

        self._kalman_update(z, H, Rv, t, tag)
        self._cur_vo = tag

        if self._tick % 100 == 0:
            self.get_logger().info(
                'VO %s | err=%.4frad deg=%.2f q=%.2f R=%.4f' %
                (tag.upper(), z[0], np.degrees(dyaw), qscore, yvar))


    # ---- measurement update (joseph form) ----

    def _kalman_update(self, z, H, Rn, t, tag):
        S = H @ self.P @ H.T + Rn
        try:
            K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        except np.linalg.LinAlgError:
            self.get_logger().warn('Singular S -- skipping')
            return

        dx = K @ z
        yaw_only = (tag in ('vo_active', 'vo_gated'))

        dphi = dx[0:3]
        dv = dx[3:6].copy();  dp = dx[6:9].copy()
        if yaw_only:
            dv[:] = 0.; dp[:] = 0.

        dR = self._exp_so3(dphi)
        Rold = self._getR().copy()
        vold = self._getv().copy()
        pold = self._getp().copy()

        if self.is_iekf:
            Jl = self._J_left(dphi)
            self._setR(dR @ Rold)
            if yaw_only:
                self._setv(vold); self._setp(pold)
            else:
                self._setv(dR @ vold + Jl @ dv)
                self._setp(dR @ pold + Jl @ dp)
        else:
            self._setR(Rold @ dR)
            if yaw_only:
                self._setv(vold); self._setp(pold)
            else:
                self._setv(vold + dv)
                self._setp(pold + dp)

        self.bg += dx[9:12];  self.ba += dx[12:15]
        self.bg = np.clip(self.bg, -0.05, 0.05)
        self.ba = np.clip(self.ba, -0.5, 0.5)   # must stay well below g=9.81

        tmp = np.eye(15) - K @ H
        self.P = tmp @ self.P @ tmp.T + K @ Rn @ K.T
        self.P = 0.5*(self.P + self.P.T)

        if np.isnan(self.X).any() or np.isinf(self.X).any():
            self.get_logger().error('NaN/Inf detected in state!')


    # ---- main IMU loop ----

    def _on_imu(self, msg):
        acc = np.array([msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z])
        gyr = np.array([msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z])

        if not self._initialized:
            # wait until we have at least one FK reading
            if not np.all(self.last_fp['fl'] == 0):
                self._init_gravity(acc)
            return

        if np.all(self.last_fp['fl'] == 0):
            return

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_t is None:
            self._last_t = t;  self.t0 = t
            return

        dt = t - self._last_t
        self._last_t = t
        if dt <= 0 or dt > 0.1:
            return

        self._predict(gyr, acc, dt)

        speeds = [np.linalg.norm(self.foot_vels[l]) for l in self.LEG_IDS]
        maxspd = max(speeds)

        z = np.zeros(4)
        H = np.zeros((4,15))

        if maxspd < self.stopped_thr:
            # ---- robot is stopped ----
            z[0:3] = -self._getv()
            H[0:3, 3:6] = np.eye(3)

            avg_fp = np.mean([self.last_fp[l] for l in self.LEG_IDS], axis=0)
            if self.is_iekf:
                z[3] = -avg_fp[2] - self._getp()[2]
                H[3,8] = 1.0
            else:
                fw = self._getR() @ avg_fp
                z[3] = -(self._getp()[2] + fw[2])
                H[3,0:3] = -(self._getR() @ self._skew(avg_fp))[2, :]
                H[3,8] = 1.0

            z[0:3] = np.clip(z[0:3], -1., 1.)
            z[3] = np.clip(z[3], -0.1, 0.1)

            self._kalman_update(z, H, self.Rn_stopped, t, 'stopped')
            self._cur_regime = 'stopped'
            self._cur_vo = 'vo_hold' if self._has_vo else 'vo_wait'

            if self._vo_buf is not None:
                _, Rvo, tvo, imp, q, v = self._vo_buf
                self._fuse_vo_yaw(Rvo, tvo, imp, q, v)
                self._vo_buf = None

            self._publish(msg.header.stamp)
            return

        # ---- stance detection via gait phase ----
        self._cur_vo = 'vo_hold' if self._has_vo else 'vo_wait'
        stance = []
        if self._got_phase:
            stance = [l for l in self.LEG_IDS if self.phi_gait[l] < self.dc]
            # trot gait -> max 2 stance legs at once
            if len(stance) > 2:
                stance = sorted(stance,
                    key=lambda l: self.dc - self.phi_gait[l],
                    reverse=True)[:2]

        # fallback: two slowest legs
        if not stance:
            spd_map = {l: np.linalg.norm(self.foot_vels[l]) for l in self.LEG_IDS}
            ordered = sorted(self.LEG_IDS, key=lambda l: spd_map[l])
            if spd_map[ordered[0]] < 0.5:
                stance = ordered[:2]

        sum_fv = np.zeros(3);  sum_fp = np.zeros(3)
        fv_arr = []
        for l in stance:
            sum_fv += self.foot_vels[l]
            sum_fp += self.last_fp[l]
            fv_arr.append(self.foot_vels[l])

        nst = len(stance)
        sum_fv /= nst;  sum_fp /= nst

        # leg disagreement as slip indicator
        spread = 0.0
        if nst > 1:
            spread = np.sum(np.var(fv_arr, axis=0))
        elif nst == 1:
            spread = 0.05

        R_cur = self._getR()
        w_corr = gyr - self.bg

        vf = np.clip(sum_fv, -3., 3.)
        vf_cal = vf.copy()
        vf_cal[0] *= self.kx
        vf_cal[1] *= self.ky

        cb = vf_cal + self._skew(w_corr) @ sum_fp
        vc_world = -(R_cur @ cb)

        z[0:3] = vc_world - self._getv()
        if self.is_iekf:
            H[0:3, 3:6] = np.eye(3)
        else:
            H[0:3, 0:3] = -R_cur @ self._skew(cb)
            H[0:3, 3:6] = np.eye(3)

        if nst > 0:
            Rn = self.Rn_stance.copy()
            # cap slip penalty so high-variance frames still contribute a little
            penalty = min(spread * 5.0, 0.5)
            Rn[0,0] += penalty;  Rn[1,1] += penalty;  Rn[2,2] += penalty
            regime = 'stance'
        else:
            z[0:3] = np.zeros(3)
            Rn = self.Rn_swing
            regime = 'swing'
            z[2] = 0.0 - self._getv()[2]

        # height from FK contact
        if self.is_iekf:
            z[3] = -sum_fp[2] - self._getp()[2]
            H[3,8] = 1.0
        else:
            fw = R_cur @ sum_fp
            z[3] = -(self._getp()[2] + fw[2])
            H[3,0:3] = -(R_cur @ self._skew(sum_fp))[2, :]
            H[3,8] = 1.0

        z[0:3] = np.clip(z[0:3], -1., 1.)
        z[3] = np.clip(z[3], -0.1, 0.1)

        self._tick += 1
        if self._tick % 100 == 0:
            vtag = 'vo_ok' if self._has_vo else 'vo_wait'
            self.get_logger().info(
                '%s n=%d var=%.3f x=%.2f [%s] ph=%s dc=%.2f bg=%.5f ba=%.5f' %
                (regime, nst, spread, self._getp()[0], vtag,
                 self._got_phase, self.dc,
                 np.linalg.norm(self.bg), np.linalg.norm(self.ba)))

        self._kalman_update(z, H, Rn, t, regime)
        self._cur_regime = regime

        if self._vo_buf is not None:
            _, Rvo, tvo, imp, q, v = self._vo_buf
            self._fuse_vo_yaw(Rvo, tvo, imp, q, v)
            self._vo_buf = None

        self._publish(msg.header.stamp)


    def _log_step(self, tstamp, regime):
        if self.t0 is None: return
        trel = tstamp - self.t0
        self.tlog.append(trel)
        self.gt_x.append(self.gt_pos[0]);  self.gt_y.append(self.gt_pos[1])
        self.est_x.append(self._getp()[0]);  self.est_y.append(self._getp()[1])
        self.gt_yaw_hist.append(R_scipy.from_quat(self.gt_quat).as_euler('zyx')[0])
        self.est_yaw_hist.append(R_scipy.from_matrix(self._getR()).as_euler('zyx')[0])
        self.regime_hist.append(regime)
        self.vo_regime_hist.append(self._cur_vo)
        self.impact_hist.append(self._check_touchdown())

    def _publish(self, stamp):
        tsec = stamp.sec + stamp.nanosec * 1e-9
        self._log_step(tsec, self._cur_regime)

        m = Odometry()
        m.header.stamp = stamp
        m.header.frame_id = 'odom'
        m.child_frame_id = 'base_link'
        m.pose.pose.position.x = float(self._getp()[0])
        m.pose.pose.position.y = float(self._getp()[1])
        m.pose.pose.position.z = float(self._getp()[2])
        q = R_scipy.from_matrix(self._getR()).as_quat()
        m.pose.pose.orientation.x = q[0]
        m.pose.pose.orientation.y = q[1]
        m.pose.pose.orientation.z = q[2]
        m.pose.pose.orientation.w = q[3]
        self.odom_pub.publish(m)


    def _write_csvs(self, t_arr, gt_lc, est_lc, ex, ey, ate, phases,
                    impacts, steady, walk_end_i, tsettle,
                    end_ate, drift_val, imp_walk, nonimp_walk):

        ate_steady = float(np.mean(ate[steady])) if np.any(steady) else np.nan
        ate_impact = float(np.mean(ate[imp_walk])) if np.any(imp_walk) else np.nan
        ate_nonimp = float(np.mean(ate[nonimp_walk])) if np.any(nonimp_walk) else np.nan

        with self.csv_path.open('w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['time_s','gt_x_local_m','gt_y_local_m','est_x_local_m',
                         'est_y_local_m','err_x_m','err_y_m','ate_m',
                         'gt_yaw_rad','est_yaw_rad','phase','vo_regime',
                         'impact_window','steady_walk','before_stop'])
            for i in range(len(t_arr)):
                w.writerow([
                    float(t_arr[i]),
                    float(gt_lc[i,0]), float(gt_lc[i,1]),
                    float(est_lc[i,0]), float(est_lc[i,1]),
                    float(ex[i]), float(ey[i]), float(ate[i]),
                    float(self.gt_yaw_hist[i]), float(self.est_yaw_hist[i]),
                    phases[i],
                    self.vo_regime_hist[i] if i < len(self.vo_regime_hist) else '',
                    int(bool(impacts[i])),
                    int(bool(steady[i])),
                    int(i <= walk_end_i)])

        nrecv = self.vo_counters['received']
        ncons = self.vo_counters['consumed']
        summary = [
            ('filter_mode', self.filt_type),
            ('filter_label', self.filt_label),
            ('enable_vo', int(self.ENABLE_VO)),
            ('enable_impact_gate', int(self.ENABLE_IMPACT_GATE)),
            ('experiment_number', self.exp_num),
            ('output_stem', self.run_name),
            ('mean_ate_m', float(np.mean(ate))),
            ('max_ate_m', float(np.max(ate))),
            ('final_ate_m', float(ate[-1])),
            ('steady_walk_mean_ate_m', ate_steady),
            ('impact_mean_ate_m', ate_impact),
            ('nonimpact_mean_ate_m', ate_nonimp),
            ('walk_end_ate_m', float(end_ate)),
            ('drift_rate_m_per_m', float(drift_val) if np.isfinite(drift_val) else np.nan),
            ('steady_start_time_s', float(tsettle) if tsettle is not None else np.nan),
            ('vo_frames_consumed', int(ncons)),
            ('vo_frames_active', int(self.vo_counters['active'])),
            ('vo_frames_gated', int(self.vo_counters['gated'])),
            ('vo_frames_reject', int(self.vo_counters['reject'])),
            ('vo_frames_received', int(nrecv)),
            ('vo_arrival_active', int(self.vo_counters['arrival_active'])),
            ('vo_arrival_gated', int(self.vo_counters['arrival_gated'])),
            ('avg_consumed_quality', float(self.vo_counters['consumed_quality_sum']/ncons) if ncons > 0 else np.nan),
            ('avg_consumed_variance', float(self.vo_counters['applied_var_sum']/ncons) if ncons > 0 else np.nan),
            ('avg_received_quality', float(self.vo_counters['received_quality_sum']/nrecv) if nrecv > 0 else np.nan),
            ('avg_received_frontend_variance', float(self.vo_counters['received_var_sum']/nrecv) if nrecv > 0 else np.nan),
        ]
        with self.csv_summary_path.open('w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['metric', 'value'])
            w.writerows(summary)


    def _make_plots(self):
        if len(self.tlog) < 10:
            return
        print('\nGenerating %s performance plots...' % self.filt_label)

        gt_xy  = np.column_stack([self.gt_x, self.gt_y])
        est_xy = np.column_stack([self.est_x, self.est_y])
        gt_yaw  = np.array(self.gt_yaw_hist)
        est_yaw = np.array(self.est_yaw_hist)

        def _to_local(xy, yaw0):
            c, s = np.cos(-yaw0), np.sin(-yaw0)
            M = np.array([[c,-s],[s,c]])
            return (xy - xy[0]) @ M.T

        gt_lc  = _to_local(gt_xy, gt_yaw[0])
        est_lc = _to_local(est_xy, est_yaw[0])

        tarr = np.array(self.tlog)
        phases  = np.array(self.regime_hist)
        impacts = np.array(self.impact_hist, dtype=bool)

        ex = est_lc[:,0] - gt_lc[:,0]
        ey = est_lc[:,1] - gt_lc[:,1]
        ate = np.sqrt(ex**2 + ey**2)

        moving = (phases != 'stopped') if len(phases)==len(tarr) else np.ones_like(tarr, dtype=bool)
        mov_i = np.flatnonzero(moving)
        walk_end_i = int(mov_i[-1]) if mov_i.size > 0 else len(tarr)-1

        steady = moving.copy()
        tsettle = None
        if mov_i.size > 0:
            t_first = tarr[mov_i[0]]
            tsettle = t_first + self.warmup_s
            steady &= (tarr >= tsettle)
            if not np.any(steady):
                steady = moving.copy(); tsettle = t_first

        imp_walk    = steady & impacts
        nonimp_walk = steady & ~impacts

        gt_pathlen = 0.
        si = np.flatnonzero(steady)
        if si.size > 1:
            gt_pathlen = float(np.sum(np.linalg.norm(
                np.diff(gt_lc[si], axis=0), axis=1)))

        end_ate = float(ate[walk_end_i])
        drift = end_ate / gt_pathlen if gt_pathlen > 1e-6 else np.nan

        # stats dump
        if self.regime_hist:
            n_st  = self.regime_hist.count('stance')
            n_sw  = self.regime_hist.count('swing')
            n_stp = self.regime_hist.count('stopped')
            tot = len(self.regime_hist)
            print('  Regime: stance=%.1f%% swing=%.1f%% stopped=%.1f%%' %
                  (n_st/tot*100, n_sw/tot*100, n_stp/tot*100))

        if self.vo_regime_hist:
            na = self.vo_regime_hist.count('vo_active')
            ng = self.vo_regime_hist.count('vo_gated')
            nh = self.vo_regime_hist.count('vo_hold')
            nw = self.vo_regime_hist.count('vo_wait')
            nr = self.vo_regime_hist.count('vo_reject')
            tv = len(self.vo_regime_hist)
            print('  VO: active=%.1f%% gated=%.1f%% hold=%.1f%% wait=%.1f%% reject=%.1f%%' %
                  (na/tv*100, ng/tv*100, nh/tv*100, nw/tv*100, nr/tv*100))

        if self.vo_counters['consumed'] > 0:
            nc = self.vo_counters['consumed']
            print('  VO frames: act=%d/%d (%.1f%%) gate=%d/%d (%.1f%%) rej=%d/%d (%.1f%%) '
                  'avg_q=%.3f avg_R=%.4f' %
                  (self.vo_counters['active'], nc, self.vo_counters['active']/nc*100,
                   self.vo_counters['gated'], nc, self.vo_counters['gated']/nc*100,
                   self.vo_counters['reject'], nc, self.vo_counters['reject']/nc*100,
                   self.vo_counters['consumed_quality_sum']/nc,
                   self.vo_counters['applied_var_sum']/nc))

        if self.vo_counters['received'] > 0:
            nr = self.vo_counters['received']
            print('  VO recv: arr_act=%d/%d (%.1f%%) arr_gate=%d/%d (%.1f%%) '
                  'avg_q=%.3f avg_fevar=%.4f' %
                  (self.vo_counters['arrival_active'], nr, self.vo_counters['arrival_active']/nr*100,
                   self.vo_counters['arrival_gated'], nr, self.vo_counters['arrival_gated']/nr*100,
                   self.vo_counters['received_quality_sum']/nr,
                   self.vo_counters['received_var_sum']/nr))

        print('  Frame: initial-pose aligned local')
        print('  Mean ATE: %.4f m' % np.mean(ate))
        print('  Max ATE:  %.4f m' % np.max(ate))
        print('  Final ATE: %.4f m' % ate[-1])
        if np.any(steady):
            print('  Steady walk ATE: %.4f m' % np.mean(ate[steady]))
        if np.any(imp_walk):
            print('  Impact ATE: %.4f m' % np.mean(ate[imp_walk]))
        if np.any(nonimp_walk):
            print('  Non-impact ATE: %.4f m' % np.mean(ate[nonimp_walk]))
        print('  Walk-end ATE: %.4f m' % end_ate)
        if gt_pathlen > 1e-6:
            print('  Drift: %.4f m/m (%.2f%%)' % (drift, drift*100))

        plt.figure(figsize=(14, 13))

        plt.subplot(4,1,1)
        plt.plot(gt_lc[:,0], gt_lc[:,1], 'g-', label='Ground Truth', lw=2)
        plt.plot(est_lc[:,0], est_lc[:,1], 'r--', label=self.filt_label, lw=2)
        plt.title('Trajectory (Aligned Local Frame)')
        plt.xlabel('X (m)');  plt.ylabel('Y (m)')
        plt.legend(); plt.grid(True); plt.axis('equal')

        plt.subplot(4,1,2)
        plt.plot(self.tlog, ex, 'b-', label='X err', lw=1)
        plt.plot(self.tlog, ey, color='orange', label='Y err', lw=1)
        plt.plot(self.tlog, ate, 'r-', label='ATE', lw=1.5, alpha=0.7)
        if tsettle is not None:
            plt.axvline(tsettle, color='gray', ls='--', lw=1, alpha=0.8,
                        label='steady start')
        plt.title('Error Over Time')
        plt.xlabel('Time (s)'); plt.ylabel('Error (m)')
        plt.axhline(0, color='k', lw=0.5)
        plt.legend(); plt.grid(True)

        if self.regime_hist:
            plt.subplot(4,1,3)
            rmap = {'stance': 0, 'swing': 2, 'stopped': -1}
            plt.fill_between(self.tlog, [rmap.get(r,0) for r in self.regime_hist],
                             alpha=0.5, color='green', step='post')
            plt.yticks([-1,0,2], ['Stopped\n(zero-vel)', 'Stance\n(trust)',
                                   'Swing\n(reject)'])
            plt.title('Covariance Regime')
            plt.xlabel('Time (s)'); plt.grid(True, axis='x')

        if self.vo_regime_hist:
            plt.subplot(4,1,4)
            vmap = {'vo_wait':-2, 'vo_hold':-1, 'vo_reject':0,
                    'vo_gated':1, 'vo_active':2}
            plt.fill_between(self.tlog,
                             [vmap.get(r,-2) for r in self.vo_regime_hist],
                             alpha=0.45, color='royalblue', step='post')
            plt.yticks([-2,-1,0,1,2], ['Wait','Hold','Reject','Gated','Active'])
            plt.title('Visual Heading Regime')
            plt.xlabel('Time (s)'); plt.grid(True, axis='x')

        self._write_csvs(tarr, gt_lc, est_lc, ex, ey, ate, phases,
                         impacts, steady, walk_end_i, tsettle,
                         end_ate, drift, imp_walk, nonimp_walk)

        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=150)
        print('  Saved: %s' % self.plot_file)
        print('  Saved: %s' % self.csv_path)
        print('  Saved: %s' % self.csv_summary_path)
        plt.show()


    def destroy_node(self):
        self._make_plots()
        super().destroy_node()


def main():
    rclpy.init()
    node = IEKF_Active()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

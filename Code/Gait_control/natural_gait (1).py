import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation as R_scipy
import numpy as np
import threading
import sys
import csv
import os
import time
from std_msgs.msg import Bool, Float32MultiArray


LEGS = ('fl', 'fr', 'hl', 'hr')
LEFT_LEGS = {'fl', 'hl'}
DIAGONAL_LEGS = {'fl', 'hr'}

TRACE_FIELDNAMES = [
    'time_s', 'gt_x_local_m', 'gt_y_local_m', 'gt_path_length_m',
    'nav_stage', 'override_speed_scale', 'override_turn',
    'clearance_m', 'collision_detected', 'returned_to_lane',
    'tracked_obstacle_count',
]


class SpotLocomotion(Node):
    """Spot trot controller used for Isaac Sim."""

    def __init__(self):
        super().__init__('spot_locomotion')

        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/iekf/odom', self.odom_callback, 10)
        self.gt_sub = self.create_subscription(
            Odometry, '/ground_truth/odom', self.gt_callback, 10)
        self.vo_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/vo/pose', self.vo_callback, 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.walk_enable_sub = self.create_subscription(
            Bool, '/walk_enable', self.walk_enable_callback, 10)
        self.phase_pub = self.create_publisher(Float32MultiArray, '/gait/phase', 10)
        self.estimated_obstacles_sub = self.create_subscription(
            Float32MultiArray, '/estimated_obstacles_world',
            self.estimated_obstacles_callback, 10)
        self.override_sub = self.create_subscription(
            Twist, '/cmd_vel_override', self.override_callback, 10)

        # Heading state, keep IMU as the fallback because VO can
        # drop out for short parts of the run.
        self.initial_yaw = None
        self.yaw_error = 0.0
        self.current_yaw = 0.0
        self.imu_yaw = 0.0
        self.fused_yaw = 0.0
        self.fused_xy = None
        self.imu_heading_ref = None
        self.fused_heading_ref = None
        self.heading_coupling_bias = 0.0
        self.heading_ready = False
        self.fused_heading_ready = False
        self.last_fused_yaw_time = None
        self.walk_start_fused_xy = None
        self.walk_start_fused_yaw = None
        self.estimator_heading_timeout = 0.30
        self.last_vo_quality_time = None
        self.vo_quality_ema = 0.0
        self.vo_quality_timeout = 0.35
        self.min_vo_quality_for_coupling = 0.55
        self.vo_quality_ema_alpha = 0.20
        self.enable_estimator_bias_in_clear = False
        self.heading_source_name = 'hybrid'
        self.heading_coupling_gain = 0.03
        self.heading_coupling_step_limit = 0.0015
        self.heading_coupling_clamp = 0.04
        self.heading_coupling_agree_limit = 0.10
        self.heading_coupling_decay = 0.0008
        self.enable_lateral_correction = False
        self.desired_avoid_heading_offset = 0.0
        self.avoid_heading_feedback_gain = 1.15
        self.avoid_turn_feedforward_gain = 0.45
        self.held_avoid_heading_offset = 0.0
        self.return_to_lane_pending = False
        self.return_to_lane_active = False
        self.return_heading_gain = 0.25
        self.return_heading_clamp = 0.13
        self.return_lane_tolerance_m = 0.08
        self.return_forward_margin_m = 0.35 # It is too short then the robot might hit the obstacle
        self.obstacle_rear_clear_margin_m = 0.75

        self.joint_names = [
            f'{leg}_{joint}'
            for leg in LEGS
            for joint in ('hx', 'hy', 'kn')
        ]

        # Leg geometry from the Spot model used in Isaac Sim.
        self.L2 = 0.320
        self.L3 = 0.370
        self.hip_offset_x = {'fl': 0.29, 'fr': 0.29, 'hl': -0.29, 'hr': -0.29}
        self.hip_offset_y = {'fl': 0.11, 'fr': -0.11, 'hl': 0.11, 'hr': -0.11}

        # Neutral stance
        self.neutral_x = 0.0
        self.neutral_z = 0.50
        self.stand_hx = 0.08

        # Gait numbers. These are tuned for the current sim setup, so do not
        # move them into "nice" constants unless the launch files are updated.
        self.gait_freq = 3.4
        self.step_height = 0.08
        self.duty_cycle = 0.55
        self.max_step_length = 0.50
        self.max_step_lateral = 0.12

        # Weight shift dynamics. Small values here matter more than they look.
        self.stance_inward_pull = 0.03
        self.swing_outward_splay = 0.04
        self.natural_dip_amount = 0.008

        # Heading correction
        self.yaw_gain = 2.0
        self.lateral_gain = 3.0
        self.correction_clamp = 0.12

        # Velocity command and experiment setup.
        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.wz_cmd = 0.0
        self.max_yaw_rate = 0.3
        self.external_cmd_received = False
        self.auto_start_enabled = self.declare_parameter(
            'auto_start_enabled', False).value
        self.autonomous_cruise_vx = 0.40
        self.walk_enabled = False
        self.experiment_stop_distance_m = 0.0
        self.experiment_mode_enabled = False
        self.current_gt_xy = None
        self.current_gt_yaw = None
        self.walk_start_gt_xy = None
        self.walk_start_gt_yaw = None
        self.last_walk_gt_xy = None
        self.walked_gt_distance_m = 0.0
        self.metrics_output_dir = self.declare_parameter(
            'metrics_output_dir', 'results').value
        self.experiment_number = str(self.declare_parameter(
            'experiment_number', '01').value)
        self.obstacle_center_x_local = float(self.declare_parameter(
            'obstacle_center_x_local', 4.0).value)
        self.obstacle_center_y_local = float(self.declare_parameter(
            'obstacle_center_y_local', 0.0).value)
        self.obstacle_radius_m = float(self.declare_parameter(
            'obstacle_radius_m', 0.45).value)
        self.goal_x_local_m = float(self.declare_parameter(
            'goal_x_local_m', 8.0).value)
        self.lane_center_y_local = float(self.declare_parameter(
            'lane_center_y_local', 0.0).value)
        self.lane_return_tolerance_m = float(self.declare_parameter(
            'lane_return_tolerance_m', 0.25).value)
        self.lane_return_forward_margin_m = float(self.declare_parameter(
            'lane_return_forward_margin_m', 0.50).value)
        self.collision_clearance_threshold_m = float(self.declare_parameter(
            'collision_clearance_threshold_m', 0.0).value)
        self.run_start_time = None
        self.current_run_label = None
        self.current_stop_reason = ''
        self.metrics_written_for_run = False
        self.last_metrics_update_time = None
        self.last_avoid_state = False
        self.avoid_time_s = 0.0
        self.stop_time_s = 0.0
        self.avoid_entry_count = 0
        self.first_avoid_time_s = np.nan
        self.time_to_pass_obstacle_s = np.nan
        self.time_to_goal_s = np.nan
        self.min_clearance_m = np.inf
        self.collision_detected = False
        self.returned_to_lane = False
        self.max_abs_lateral_offset_m = 0.0
        self.final_local_xy = np.zeros(2, dtype=np.float64)
        self.gt_trace_rows = []
        self.estimated_obstacles_world = []
        self.last_estimated_obstacles_time = None
        self.estimated_obstacle_timeout_s = 0.75
        self.estimated_obstacle_seen = False
        self.estimated_obstacle_clear_dwell_s = 0.35
        self.estimated_obstacle_clear_start_time = None
        self.max_tracked_obstacles_seen = 0

        # Visual node sends a compact state through /cmd_vel_override:
        #   linear.z == 0.0 -> CLEAR,  0.5 -> AVOID,  1.0 -> STOP.
        # Not elegant, but it kept the experiment wiring simple.
        self.nav_stage = 1
        self.override_speed_scale = 1.0
        self.override_turn = 0.0
        self.last_override_time = None
        self.resume_clear_time = 1.0
        self.avoid_min_forward_speed = 0.18
        self.avoid_min_speed_scale = 0.45

        self.original_yaw = None
        self.heading_restore_gain = 0.30
        self.heading_restore_clamp = 0.12

        self.enable_extra_sway = False
        self.enable_extra_bounce = False
        self.extra_sway_amount = 0.03
        self.extra_bounce_amount = 0.008

        self.ramp_duration = 3.0

        self.mode = 'stand'
        self.walk_start_time = None

        self.cmd_pub = self.create_publisher(JointState, '/spot/joint_commands', 10)
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info(
            'spot gait ready | '
            f'auto={self.auto_start_enabled} | '
            f'cruise={self.autonomous_cruise_vx:.2f} m/s | '
            f'obs=({self.obstacle_center_x_local:.1f}, '
            f'{self.obstacle_center_y_local:.1f}) r={self.obstacle_radius_m:.2f}'
        )

        self.kb_thread = threading.Thread(target=self.keyboard_listener, daemon=True)
        self.kb_thread.start()

    def _now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _publish_joint_positions(self, positions, publish_phase=False):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = positions
        self.cmd_pub.publish(msg)
        if publish_phase:
            self._publish_phase()

    def _zero_velocity_command(self):
        self.vx_cmd = 0.0
        self.vy_cmd = 0.0
        self.wz_cmd = 0.0

    def _reset_avoidance_state(self):
        self.held_avoid_heading_offset = 0.0
        self.return_to_lane_pending = False
        self.return_to_lane_active = False
        self.desired_avoid_heading_offset = 0.0

    def _set_gt_walk_start_from_current(self):
        if self.current_gt_xy is not None:
            self.walk_start_gt_xy = self.current_gt_xy.copy()
            self.last_walk_gt_xy = self.current_gt_xy.copy()
        else:
            self.walk_start_gt_xy = None
            self.last_walk_gt_xy = None
        self.walk_start_gt_yaw = self.current_gt_yaw

    def _set_fused_walk_start_from_current(self):
        if self.fused_xy is None:
            self.walk_start_fused_xy = None
        else:
            self.walk_start_fused_xy = self.fused_xy.copy()
        self.walk_start_fused_yaw = (
            self.fused_yaw if self.fused_heading_ready else None
        )

    def _log_cmd(self):
        self.get_logger().info(
            f'cmd vx={self.vx_cmd:.2f} vy={self.vy_cmd:.2f} '
            f'wz={self.wz_cmd:.2f}'
        )

    def _apply_manual_delta(self, dvx=0.0, dvy=0.0, dwz=0.0):
        self.external_cmd_received = True
        self.walk_enabled = False
        self.vx_cmd = float(np.clip(self.vx_cmd + dvx, -0.3, 0.5))
        self.vy_cmd = float(np.clip(self.vy_cmd + dvy, -0.25, 0.25))
        self.wz_cmd = float(np.clip(
            self.wz_cmd + dwz,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        ))
        self._start_walking()
        self._log_cmd()

    def cmd_vel_callback(self, msg):
        self.external_cmd_received = True
        self.vx_cmd = np.clip(msg.linear.x, -0.3, 0.5)
        self.vy_cmd = np.clip(msg.linear.y, -0.25, 0.25)
        self.wz_cmd = np.clip(msg.angular.z, -self.max_yaw_rate, self.max_yaw_rate)
        commanded_motion = (
            abs(self.vx_cmd) > 0.01 or
            abs(self.vy_cmd) > 0.01 or
            abs(self.wz_cmd) > 0.01
        )
        if self.mode == 'stand' and commanded_motion:
            self._start_walking()

    def _start_walking(self):
        if self.mode != 'walk':
            self.mode = 'walk'
            self.walk_start_time = self._now_s()
            self.run_start_time = self.walk_start_time
            self.walked_gt_distance_m = 0.0
            self._set_gt_walk_start_from_current()
            self._set_fused_walk_start_from_current()
            self._reset_avoidance_state()
            self._reset_obstacle_metrics()
        if self.original_yaw is None:
            self._reset_heading_reference()

    def _stop_walking(self, reason='manual_stop'):
        self.current_stop_reason = reason
        self._finalize_obstacle_metrics()
        self.mode = 'stand'
        self._zero_velocity_command()
        self.original_yaw = None
        self.walk_start_gt_xy = None
        self.walk_start_gt_yaw = None
        self.last_walk_gt_xy = None
        self.walked_gt_distance_m = 0.0
        self.walk_start_fused_xy = None
        self.walk_start_fused_yaw = None
        self._reset_avoidance_state()

    def _reset_heading_reference(self):
        self.imu_heading_ref = self.imu_yaw
        self.fused_heading_ref = self.fused_yaw if self.fused_heading_ready else None
        self.heading_coupling_bias = 0.0
        self.current_yaw = 0.0
        self.initial_yaw = 0.0
        self.original_yaw = 0.0
        self._reset_avoidance_state()

    def _arm_experiment_mode(self, distance_m):
        self.experiment_stop_distance_m = float(distance_m)
        self.experiment_mode_enabled = self.experiment_stop_distance_m > 0.0
        self.walked_gt_distance_m = 0.0
        self._set_gt_walk_start_from_current()

        if self.experiment_mode_enabled:
            self.get_logger().info(
                f'exp mode on: stop at {self.experiment_stop_distance_m:.2f} m '
                '(gt path)'
            )
        else:
            self.get_logger().info('exp mode off')

    def _enable_autonomous_walk(self):
        if not self.heading_ready:
            self.get_logger().warn(
                'auto walk skipped: waiting on imu heading')
            return
        self.external_cmd_received = False
        self.walk_enabled = True
        self.vx_cmd = self.autonomous_cruise_vx
        self.vy_cmd = 0.0
        self.wz_cmd = 0.0
        self._start_walking()
        self.get_logger().info(
            f'auto walk vx={self.vx_cmd:.2f} m/s')

    def walk_enable_callback(self, msg):
        self.walk_enabled = bool(msg.data)

        if self.walk_enabled:
            self._enable_autonomous_walk()
        else:
            self._stop_walking(reason='walk_disabled')
            self.get_logger().info('walk off -> stand')

    def _reset_obstacle_metrics(self):
        run_tag = int(time.time())
        self.current_run_label = (
            f'obstacle_avoidance_exp_{self.experiment_number}_{run_tag}'
        )
        self.metrics_written_for_run = False
        self.last_metrics_update_time = self.walk_start_time
        self.last_avoid_state = False
        self.avoid_time_s = 0.0
        self.stop_time_s = 0.0
        self.avoid_entry_count = 0
        self.first_avoid_time_s = np.nan
        self.time_to_pass_obstacle_s = np.nan
        self.time_to_goal_s = np.nan
        self.min_clearance_m = np.inf
        self.collision_detected = False
        self.returned_to_lane = False
        self.max_abs_lateral_offset_m = 0.0
        self.final_local_xy = np.zeros(2, dtype=np.float64)
        self.gt_trace_rows = []
        self.estimated_obstacle_seen = False
        self.estimated_obstacle_clear_start_time = None
        self.max_tracked_obstacles_seen = 0
        self.current_stop_reason = ''

    def _gt_local_xy(self):
        if self.walk_start_gt_xy is None or self.current_gt_xy is None:
            return None
        yaw0 = 0.0 if self.walk_start_gt_yaw is None else self.walk_start_gt_yaw
        delta = self.current_gt_xy - self.walk_start_gt_xy
        c = np.cos(yaw0)
        s = np.sin(yaw0)
        x_local = c * delta[0] + s * delta[1]
        y_local = -s * delta[0] + c * delta[1]
        return np.array([x_local, y_local], dtype=np.float64)

    def _world_xy_to_gt_local(self, world_xy):
        if self.walk_start_gt_xy is None:
            return None
        yaw0 = 0.0 if self.walk_start_gt_yaw is None else self.walk_start_gt_yaw
        delta = np.asarray(world_xy, dtype=np.float64) - self.walk_start_gt_xy
        c = np.cos(yaw0)
        s = np.sin(yaw0)
        x_local = c * delta[0] + s * delta[1]
        y_local = -s * delta[0] + c * delta[1]
        return np.array([x_local, y_local], dtype=np.float64)

    def _estimated_obstacles_gt_local(self):
        now = self._now_s()
        if (self.last_estimated_obstacles_time is None or
                now - self.last_estimated_obstacles_time > self.estimated_obstacle_timeout_s):
            return None
        if self.walk_start_gt_xy is None:
            return None

        local_obstacles = []
        for obstacle in self.estimated_obstacles_world:
            local_xy = self._world_xy_to_gt_local([obstacle['x'], obstacle['y']])
            if local_xy is None:
                continue
            local_obstacles.append((
                float(local_xy[0]),
                float(local_xy[1]),
                float(obstacle['r']),
            ))
        return local_obstacles

    def _update_obstacle_metrics(self, now):
        if self.mode != 'walk' or self.run_start_time is None:
            self.last_metrics_update_time = now
            return

        # Metrics use GT local frame. That is less "robot realistic" but makes
        # the plots and clearance checks much easier to compare between runs.
        if self.last_metrics_update_time is None:
            self.last_metrics_update_time = now
        dt = max(0.0, now - self.last_metrics_update_time)
        self.last_metrics_update_time = now

        if self.nav_stage == 2:
            self.avoid_time_s += dt
        elif self.nav_stage == 3:
            self.stop_time_s += dt

        is_avoiding = self.nav_stage >= 2
        if is_avoiding and not self.last_avoid_state:
            self.avoid_entry_count += 1
            if np.isnan(self.first_avoid_time_s):
                self.first_avoid_time_s = now - self.run_start_time
        self.last_avoid_state = is_avoiding

        local_xy = self._gt_local_xy()
        if local_xy is None:
            return

        self.final_local_xy = local_xy.copy()
        self.max_abs_lateral_offset_m = max(
            self.max_abs_lateral_offset_m,
            abs(local_xy[1] - self.lane_center_y_local),
        )

        estimated_obstacles_local = self._estimated_obstacles_gt_local()
        using_estimated_obstacles = (
            estimated_obstacles_local is not None and
            (len(estimated_obstacles_local) > 0 or self.estimated_obstacle_seen)
        )
        if using_estimated_obstacles and estimated_obstacles_local:
            self.max_tracked_obstacles_seen = max(
                self.max_tracked_obstacles_seen,
                len(estimated_obstacles_local),
            )
            clearance = min(
                float(np.hypot(local_xy[0] - ox, local_xy[1] - oy) - radius)
                for ox, oy, radius in estimated_obstacles_local
            )
        else:
            clearance = float(
                np.hypot(
                    local_xy[0] - self.obstacle_center_x_local,
                    local_xy[1] - self.obstacle_center_y_local,
                ) - self.obstacle_radius_m
            )
        self.min_clearance_m = min(self.min_clearance_m, clearance)
        if clearance <= self.collision_clearance_threshold_m:
            self.collision_detected = True

        elapsed = now - self.run_start_time
        if np.isnan(self.time_to_pass_obstacle_s):
            if using_estimated_obstacles:
                ahead_obstacles = [
                    (ox, oy, radius)
                    for ox, oy, radius in (estimated_obstacles_local or [])
                    if ox + radius >= local_xy[0]
                ]
                if self.avoid_entry_count > 0 and len(ahead_obstacles) == 0:
                    if self.estimated_obstacle_clear_start_time is None:
                        self.estimated_obstacle_clear_start_time = elapsed
                    elif ((elapsed - self.estimated_obstacle_clear_start_time) >=
                          self.estimated_obstacle_clear_dwell_s):
                        self.time_to_pass_obstacle_s = self.estimated_obstacle_clear_start_time
                else:
                    self.estimated_obstacle_clear_start_time = None
            else:
                passed_x = self.obstacle_center_x_local + self.obstacle_radius_m
                if local_xy[0] >= passed_x:
                    self.time_to_pass_obstacle_s = elapsed

        if np.isnan(self.time_to_goal_s) and self.goal_x_local_m > 0.0:
            if local_xy[0] >= self.goal_x_local_m:
                self.time_to_goal_s = elapsed

        if using_estimated_obstacles:
            if (self.avoid_entry_count > 0 and
                    np.isfinite(self.time_to_pass_obstacle_s) and
                    abs(local_xy[1] - self.lane_center_y_local) <=
                    self.lane_return_tolerance_m):
                self.returned_to_lane = True
        else:
            lane_return_x = (
                self.obstacle_center_x_local +
                self.obstacle_radius_m +
                self.lane_return_forward_margin_m
            )
            if (self.avoid_entry_count > 0 and local_xy[0] >= lane_return_x and
                    abs(local_xy[1] - self.lane_center_y_local) <=
                    self.lane_return_tolerance_m):
                self.returned_to_lane = True

        self.gt_trace_rows.append({
            'time_s': elapsed,
            'gt_x_local_m': float(local_xy[0]),
            'gt_y_local_m': float(local_xy[1]),
            'gt_path_length_m': float(self.walked_gt_distance_m),
            'nav_stage': int(self.nav_stage),
            'override_speed_scale': float(self.override_speed_scale),
            'override_turn': float(self.override_turn),
            'clearance_m': clearance,
            'collision_detected': int(self.collision_detected),
            'returned_to_lane': int(self.returned_to_lane),
            'tracked_obstacle_count': int(
                len(estimated_obstacles_local) if estimated_obstacles_local is not None else 0),
        })

    def _write_obstacle_metrics(self):
        if self.current_run_label is None:
            return

        output_dir = os.path.abspath(self.metrics_output_dir)
        os.makedirs(output_dir, exist_ok=True)

        trace_path = os.path.join(output_dir, f'{self.current_run_label}_trace.csv')
        summary_path = os.path.join(output_dir, f'{self.current_run_label}_summary.csv')
        aggregate_path = os.path.join(output_dir, 'obstacle_avoidance_all_runs.csv')

        with open(trace_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=TRACE_FIELDNAMES)
            writer.writeheader()
            writer.writerows(self.gt_trace_rows)

        duration_s = 0.0
        if self.run_start_time is not None and self.last_metrics_update_time is not None:
            duration_s = max(0.0, self.last_metrics_update_time - self.run_start_time)

        goal_reached = (
            self.goal_x_local_m > 0.0 and
            np.isfinite(self.time_to_goal_s)
        )
        obstacle_passed = np.isfinite(self.time_to_pass_obstacle_s)
        success = (not self.collision_detected) and (
            goal_reached if self.goal_x_local_m > 0.0 else obstacle_passed
        )
        avoid_ratio = (
            self.avoid_time_s / duration_s if duration_s > 1e-6 else 0.0
        )

        summary_row = {
            'run_label': self.current_run_label,
            'experiment_number': self.experiment_number,
            'stop_reason': self.current_stop_reason or 'unknown',
            'success': int(success),
            'collision_detected': int(self.collision_detected),
            'minimum_clearance_m': (
                float(self.min_clearance_m) if np.isfinite(self.min_clearance_m)
                else np.nan
            ),
            'time_to_pass_obstacle_s': float(self.time_to_pass_obstacle_s),
            'time_to_goal_s': float(self.time_to_goal_s),
            'duration_s': float(duration_s),
            'path_length_m': float(self.walked_gt_distance_m),
            'forward_progress_m': float(self.final_local_xy[0]),
            'final_lateral_offset_m': float(
                self.final_local_xy[1] - self.lane_center_y_local),
            'max_abs_lateral_offset_m': float(self.max_abs_lateral_offset_m),
            'returned_to_lane': int(self.returned_to_lane),
            'avoid_time_s': float(self.avoid_time_s),
            'avoid_ratio': float(avoid_ratio),
            'avoid_entry_count': int(self.avoid_entry_count),
            'max_tracked_obstacles_seen': int(self.max_tracked_obstacles_seen),
            'used_estimated_obstacles': int(self.estimated_obstacle_seen),
            'obstacle_center_x_local': float(self.obstacle_center_x_local),
            'obstacle_center_y_local': float(self.obstacle_center_y_local),
            'obstacle_radius_m': float(self.obstacle_radius_m),
            'goal_x_local_m': float(self.goal_x_local_m),
            'lane_center_y_local': float(self.lane_center_y_local),
            'lane_return_tolerance_m': float(self.lane_return_tolerance_m),
        }
        summary_fieldnames = list(summary_row.keys())

        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerow(summary_row)

        aggregate_exists = os.path.exists(aggregate_path)
        with open(aggregate_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            if not aggregate_exists:
                writer.writeheader()
            writer.writerow(summary_row)

        self.get_logger().info(
            f'metrics saved: summary={summary_path} trace={trace_path}'
        )
        self.get_logger().info(
            'run check: '
            f"ok={summary_row['success']} "
            f"hit={summary_row['collision_detected']} "
            f"clear_min={summary_row['minimum_clearance_m']:.3f} m "
            f"path={summary_row['path_length_m']:.3f} m "
            f"goal_t={summary_row['time_to_goal_s']:.3f} s "
            f"lane_back={summary_row['returned_to_lane']}"
        )

    def _finalize_obstacle_metrics(self):
        if self.metrics_written_for_run or self.run_start_time is None:
            return
        now = self._now_s()
        self._update_obstacle_metrics(now)
        self._write_obstacle_metrics()
        self.metrics_written_for_run = True

    def _fused_local_xy(self):
        if self.walk_start_fused_xy is None or self.fused_xy is None:
            return None
        yaw0 = 0.0 if self.walk_start_fused_yaw is None else self.walk_start_fused_yaw
        delta = self.fused_xy - self.walk_start_fused_xy
        c = np.cos(yaw0)
        s = np.sin(yaw0)
        x_local = c * delta[0] + s * delta[1]
        y_local = -s * delta[0] + c * delta[1]
        return np.array([x_local, y_local], dtype=np.float64)

    def _world_xy_to_fused_local(self, world_xy):
        if self.walk_start_fused_xy is None:
            return None
        yaw0 = 0.0 if self.walk_start_fused_yaw is None else self.walk_start_fused_yaw
        delta = np.asarray(world_xy, dtype=np.float64) - self.walk_start_fused_xy
        c = np.cos(yaw0)
        s = np.sin(yaw0)
        x_local = c * delta[0] + s * delta[1]
        y_local = -s * delta[0] + c * delta[1]
        return np.array([x_local, y_local], dtype=np.float64)

    def _estimated_obstacles_local(self):
        now = self._now_s()
        if (self.last_estimated_obstacles_time is None or
                now - self.last_estimated_obstacles_time > self.estimated_obstacle_timeout_s):
            return None
        if self.walk_start_fused_xy is None:
            return None

        local_obstacles = []
        for obstacle in self.estimated_obstacles_world:
            local_xy = self._world_xy_to_fused_local([obstacle['x'], obstacle['y']])
            if local_xy is None:
                continue
            local_obstacles.append((
                float(local_xy[0]),
                float(local_xy[1]),
                float(obstacle['r']),
            ))
        return local_obstacles

    def _estimated_obstacles_ahead_local(self, current_local_x):
        local_obstacles = self._estimated_obstacles_local()
        if local_obstacles is None:
            return None

        ahead = []
        for ox, oy, radius in local_obstacles:
            intrudes_corridor = abs(oy - self.lane_center_y_local) <= (radius + 0.50)
            if not intrudes_corridor:
                continue
            if ox + radius + self.obstacle_rear_clear_margin_m >= current_local_x:
                ahead.append((ox, oy, radius))
        return ahead

    def _wrap_angle(self, angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def _update_control_heading(self):
        now = self._now_s()
        if self.imu_heading_ref is None:
            self.imu_heading_ref = self.imu_yaw

        imu_local_yaw = self._wrap_angle(self.imu_yaw - self.imu_heading_ref)
        vo_quality_fresh = (
            self.last_vo_quality_time is not None and
            (now - self.last_vo_quality_time) < self.vo_quality_timeout
        )
        use_fused = (
            self.fused_heading_ready and
            self.last_fused_yaw_time is not None and
            (now - self.last_fused_yaw_time) < self.estimator_heading_timeout and
            vo_quality_fresh and
            self.vo_quality_ema >= self.min_vo_quality_for_coupling
        )
        allow_estimator_bias = (
            self.enable_estimator_bias_in_clear or
            self.nav_stage != 1 or
            self.return_to_lane_active
        )
        # Note to self: only let the estimator pull heading when VO quality is
        # fresh. Otherwise the robot can slowly steer off the lane after avoid.
        use_fused = use_fused and allow_estimator_bias

        if use_fused:
            if self.fused_heading_ref is None:
                self.fused_heading_ref = self.fused_yaw

            fused_local_yaw = self._wrap_angle(self.fused_yaw - self.fused_heading_ref)
            target_bias = self._wrap_angle(fused_local_yaw - imu_local_yaw)

            if abs(target_bias) <= self.heading_coupling_agree_limit:
                bias_error = self._wrap_angle(target_bias - self.heading_coupling_bias)
                bias_step = np.clip(
                    bias_error * self.heading_coupling_gain,
                    -self.heading_coupling_step_limit,
                    self.heading_coupling_step_limit
                )
                self.heading_coupling_bias = float(np.clip(
                    self.heading_coupling_bias + bias_step,
                    -self.heading_coupling_clamp,
                    self.heading_coupling_clamp
                ))
                self.heading_source_name = 'hybrid'
            else:
                self.heading_coupling_bias *= 0.98
                self.heading_source_name = 'imu_only'
        else:
            if abs(self.heading_coupling_bias) > 1e-6:
                self.heading_coupling_bias -= float(np.clip(
                    self.heading_coupling_bias,
                    -self.heading_coupling_decay,
                    self.heading_coupling_decay
                ))
            if not allow_estimator_bias:
                self.heading_source_name = 'imu_clear'
            elif self.fused_heading_ready and not vo_quality_fresh:
                self.heading_source_name = 'imu_no_vo'
            elif (
                    self.fused_heading_ready and
                    self.vo_quality_ema < self.min_vo_quality_for_coupling):
                self.heading_source_name = 'imu_lowq'
            else:
                self.heading_source_name = 'imu'

        self.current_yaw = self._wrap_angle(imu_local_yaw + self.heading_coupling_bias)

        if self.initial_yaw is None:
            self.initial_yaw = 0.0
        self.heading_ready = True
        self.yaw_error = self._wrap_angle(self.current_yaw - self.initial_yaw)

    def keyboard_listener(self):
        speed_step = 0.4
        lateral_step = 0.08
        yaw_step = 0.1

        # Simple stdin controls for quick sim tests. This is not meant to be a
        # full teleop node, just enough to bump speed and start/stop runs.
        while True:
            try:
                cmd = input().strip().lower()

                if cmd == 'w':
                    self._enable_autonomous_walk()

                elif cmd == 's':
                    self._apply_manual_delta(dvx=-speed_step)

                elif cmd == 'a':
                    self._apply_manual_delta(dvy=lateral_step)

                elif cmd == 'd':
                    self._apply_manual_delta(dvy=-lateral_step)

                elif cmd == 'q':
                    self._apply_manual_delta(dwz=yaw_step)

                elif cmd == 'e':
                    self._apply_manual_delta(dwz=-yaw_step)

                elif cmd == 'x':
                    self.external_cmd_received = False
                    self.walk_enabled = False
                    self._stop_walking(reason='manual_stop')
                    self.get_logger().info('stand')

                elif cmd == '' or cmd == ' ':
                    self.external_cmd_received = True
                    self.walk_enabled = False
                    self._zero_velocity_command()
                    self.get_logger().info('cmd zeroed')

                elif cmd == '+' or cmd == '=':
                    self.gait_freq = min(self.gait_freq + 0.5, 4.0)
                    self.get_logger().info(f'gait freq {self.gait_freq:.1f} Hz')

                elif cmd == '-':
                    self.gait_freq = max(self.gait_freq - 0.5, 1.0)
                    self.get_logger().info(f'gait freq {self.gait_freq:.1f} Hz')

                elif cmd == 'r':
                    if self.heading_ready:
                        self._reset_heading_reference()
                        self.get_logger().info('heading reset')

                elif cmd == 'p':
                    try:
                        meters_str = input(
                            'Experiment stop distance in meters '
                            '(0 to disable): '
                        ).strip()
                    except EOFError:
                        break

                    try:
                        distance_m = float(meters_str)
                    except ValueError:
                        self.get_logger().warn(
                            f"Invalid experiment distance '{meters_str}'")
                        continue

                    if distance_m < 0.0:
                        self.get_logger().warn(
                            'Experiment distance must be non-negative')
                        continue

                    self._arm_experiment_mode(distance_m)

                elif cmd == '0':
                    rclpy.shutdown()
                    sys.exit(0)

            except EOFError:
                break

    def _maybe_auto_start(self):
        if not self.auto_start_enabled:
            return
        if self.external_cmd_received:
            return
        if not self.heading_ready:
            return
        if self.mode == 'stand' and not self.walk_enabled:
            self.vx_cmd = self.autonomous_cruise_vx
            self.vy_cmd = 0.0
            self.wz_cmd = 0.0
            self._start_walking()
            self.get_logger().info(
                f'auto start vx={self.vx_cmd:.2f} m/s')

    def override_callback(self, msg):
        now = self._now_s()
        self.last_override_time = now

        lz = msg.linear.z
        if lz >= 0.9:
            if self.nav_stage != 3:
                self.get_logger().warn('nav 3: stop')
            self.nav_stage = 3
            self.desired_avoid_heading_offset = 0.0

        elif lz >= 0.4:
            if self.nav_stage == 1:
                self.get_logger().info('nav 2: avoid')
            self.return_to_lane_pending = False
            self.return_to_lane_active = False
            self.nav_stage = 2
            self.override_speed_scale = float(np.clip(
                msg.linear.x, self.avoid_min_speed_scale, 1.0))
            self.desired_avoid_heading_offset = float(np.clip(
                msg.angular.x, -0.45, 0.45))
            self.held_avoid_heading_offset = self.desired_avoid_heading_offset
            self.override_turn = float(msg.angular.z)

        else:
            if self.nav_stage == 2:
                self.return_to_lane_pending = True
                self.return_to_lane_active = False
            if self.nav_stage != 1:
                self.get_logger().info('nav 1: clear, heading back')
            self.nav_stage = 1
            self.override_speed_scale = 1.0
            self.override_turn = 0.0
            self.desired_avoid_heading_offset = 0.0

    def imu_callback(self, msg):
        q = msg.orientation
        q_vec = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        if np.linalg.norm(q_vec) < 0.5:
            return

        self.imu_yaw = R_scipy.from_quat(q_vec).as_euler('zyx')[0]
        self._update_control_heading()
        self._maybe_auto_start()

    def odom_callback(self, msg):
        self.fused_xy = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ], dtype=np.float64)
        q = msg.pose.pose.orientation
        q_vec = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        if np.linalg.norm(q_vec) < 0.5:
            return

        self.fused_yaw = R_scipy.from_quat(q_vec).as_euler('zyx')[0]
        self.fused_heading_ready = True
        self.last_fused_yaw_time = self._now_s()
        self._update_control_heading()

    def gt_callback(self, msg):
        self.current_gt_xy = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ], dtype=np.float64)
        q = msg.pose.pose.orientation
        q_vec = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        if np.linalg.norm(q_vec) >= 0.5:
            self.current_gt_yaw = R_scipy.from_quat(q_vec).as_euler('zyx')[0]

        if self.mode != 'walk':
            return

        if self.walk_start_gt_xy is None:
            self.walk_start_gt_xy = self.current_gt_xy.copy()
        if self.last_walk_gt_xy is None:
            self.last_walk_gt_xy = self.current_gt_xy.copy()

        step = float(np.linalg.norm(self.current_gt_xy - self.last_walk_gt_xy))
        if np.isfinite(step):
            self.walked_gt_distance_m += step
        self.last_walk_gt_xy = self.current_gt_xy.copy()
        now = self._now_s()
        self._update_obstacle_metrics(now)

    def _experiment_distance_reached(self):
        if (not self.experiment_mode_enabled or
                self.experiment_stop_distance_m <= 0.0 or
                self.mode != 'walk'):
            return False, 0.0

        if self.walk_start_gt_xy is None or self.current_gt_xy is None:
            return False, 0.0
        distance_value = float(self.walked_gt_distance_m)

        return distance_value >= self.experiment_stop_distance_m, distance_value

    def vo_callback(self, msg):
        quality = float(np.clip(msg.pose.covariance[0], 0.0, 1.0))
        self.vo_quality_ema = (
            (1.0 - self.vo_quality_ema_alpha) * self.vo_quality_ema +
            self.vo_quality_ema_alpha * quality
        )
        self.last_vo_quality_time = self._now_s()

    def estimated_obstacles_callback(self, msg):
        data = list(msg.data)
        obstacles = []
        usable_len = len(data) - (len(data) % 3)
        for idx in range(0, usable_len, 3):
            x_w, y_w, radius = data[idx:idx + 3]
            if not np.all(np.isfinite([x_w, y_w, radius])):
                continue
            obstacles.append({
                'x': float(x_w),
                'y': float(y_w),
                'r': max(0.05, float(radius)),
            })
        self.estimated_obstacles_world = obstacles
        self.last_estimated_obstacles_time = self._now_s()
        if obstacles:
            self.estimated_obstacle_seen = True

    def inverse_kinematics(self, px, pz):
        r_sq = px**2 + pz**2
        r = np.sqrt(r_sq)
        r_max = self.L2 + self.L3 - 0.01
        r_min = abs(self.L2 - self.L3) + 0.01
        r = np.clip(r, r_min, r_max)
        r_sq = r**2

        cos_kn = (r_sq - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_kn = np.clip(cos_kn, -1.0, 1.0)
        kn = -np.arccos(cos_kn)

        sin_kn = np.sin(kn)
        alpha = np.arctan2(px, pz)
        beta = np.arctan2(self.L3 * sin_kn, self.L2 + self.L3 * cos_kn)
        hy = alpha - beta

        return hy, kn

    def compute_foot_velocity(self, leg):
        vx = self.vx_cmd
        vy = self.vy_cmd

        rx = self.hip_offset_x[leg]
        ry = self.hip_offset_y[leg]
        vx_turn = -self.wz_cmd * ry
        vy_turn = self.wz_cmd * rx

        return vx + vx_turn, vy + vy_turn

    def foot_trajectory(self, phase_norm, ramp, leg):
        vx_foot, vy_foot = self.compute_foot_velocity(leg)

        T = 1.0 / self.gait_freq if self.gait_freq > 0 else 1.0

        sl_x = (
            np.clip(
                vx_foot * T * self.duty_cycle,
                -self.max_step_length,
                self.max_step_length,
            ) * ramp
        )
        sl_y = (
            np.clip(
                vy_foot * T * self.duty_cycle,
                -self.max_step_lateral,
                self.max_step_lateral,
            ) * ramp
        )

        sh = self.step_height * ramp

        speed = np.sqrt(vx_foot**2 + vy_foot**2)
        sh *= max(0.5, min(speed / 0.2, 1.5))

        if phase_norm < self.duty_cycle:
            # Stance: keep foot on ground and let body move past it.
            t = phase_norm / self.duty_cycle

            px = self.neutral_x + sl_x / 2 * (2.0 * t - 1.0)
            py_offset = sl_y / 2 * (2.0 * t - 1.0)

            natural_dip = self.natural_dip_amount * ramp * np.sin(np.pi * t)
            pz = self.neutral_z + natural_dip

            if self.enable_extra_bounce:
                pz += self.extra_bounce_amount * ramp * np.sin(np.pi * t)

            is_stance = True

        else:
            # Swing: cycloid-ish path gives softer lift and touchdown.
            t = (phase_norm - self.duty_cycle) / (1.0 - self.duty_cycle)

            cycloid_t = t - (1.0 / (2.0 * np.pi)) * np.sin(2.0 * np.pi * t)

            px = self.neutral_x + sl_x / 2 - sl_x * cycloid_t
            py_offset = sl_y / 2 - sl_y * cycloid_t

            height = np.sin(np.pi * t) ** 2

            if t > 0.85:
                # Last bit of swing retracts slightly; this reduced toe taps
                # in the current sim scene near 0.4 m/s forward speed.
                retract = (t - 0.85) / 0.15
                px -= 0.005 * retract * ramp

            pz = self.neutral_z - sh * height
            is_stance = False

        return px, pz, py_offset, is_stance

    def get_stand_positions(self):
        hy, kn = self.inverse_kinematics(self.neutral_x, self.neutral_z)
        positions = []
        for leg in LEGS:
            hx = self.stand_hx if leg in LEFT_LEGS else -self.stand_hx
            positions.extend([float(hx), float(hy), float(kn)])
        return positions

    def get_walk_positions(self):
        walk_t = self._now_s() - self.walk_start_time

        # Ramp is deliberately slow; sudden trot startup made the feet slip.
        ramp = min(walk_t / self.ramp_duration, 1.0)
        ramp = 0.5 * (1.0 - np.cos(np.pi * ramp))

        phase = (self.gait_freq * walk_t) % 1.0

        positions = []
        for leg in LEGS:
            # Trot timing: FL+HR diagonal, FR+HL diagonal.
            if leg in DIAGONAL_LEGS:
                leg_phase = phase % 1.0
            else:
                leg_phase = (phase + 0.5) % 1.0

            px, pz, py_offset, is_stance = self.foot_trajectory(leg_phase, ramp, leg)
            hy, kn = self.inverse_kinematics(px, pz)

            if leg in LEFT_LEGS:
                base_hx = self.stand_hx
            else:
                base_hx = -self.stand_hx

            if self.enable_lateral_correction:
                lat_corr = np.clip(
                    self.vy_cmd * self.lateral_gain,
                    -self.correction_clamp,
                    self.correction_clamp
                )
            else:
                lat_corr = 0.0

            # Little side shift to keep the stance feet loaded.
            if is_stance:
                weight_shift = -self.stance_inward_pull * ramp
            else:
                t_sw = (leg_phase - self.duty_cycle) / (1.0 - self.duty_cycle)
                weight_shift = self.swing_outward_splay * ramp * np.sin(np.pi * t_sw)

            if leg in LEFT_LEGS:
                hx = base_hx - weight_shift
            else:
                hx = base_hx + weight_shift

            hx += py_offset * 2.0
            hx -= lat_corr

            if self.enable_extra_sway:
                hx += self.extra_sway_amount * ramp * np.sin(2 * np.pi * self.gait_freq * walk_t)

            hx = np.clip(hx, -0.3, 0.3)

            positions.extend([float(hx), float(hy), float(kn)])

        return positions

    def _maybe_stop_for_experiment_limit(self):
        reached_limit, distance_value = self._experiment_distance_reached()
        if not reached_limit:
            return

        self.walk_enabled = False
        self.external_cmd_received = False
        self._stop_walking(reason='experiment_distance')
        self.get_logger().info(f'exp stop: {distance_value:.2f} m gt path')

    def _clear_stale_override(self, now):
        if (self.last_override_time is not None and
                now - self.last_override_time > 1.0):
            self.nav_stage = 1
            self.override_speed_scale = 1.0
            self.override_turn = 0.0
            self.desired_avoid_heading_offset = 0.0
            self.return_to_lane_pending = False

    def _scaled_avoidance_vx(self, vx):
        scaled_vx = vx * self.override_speed_scale
        if abs(vx) > 0.05:
            min_forward_mag = min(self.avoid_min_forward_speed, abs(vx))
            scaled_vx = np.sign(vx) * max(abs(scaled_vx), min_forward_mag)
        return float(scaled_vx)

    def _avoidance_heading_target(self):
        heading_target = self.desired_avoid_heading_offset
        fused_local_xy = self._fused_local_xy()
        if fused_local_xy is None:
            return heading_target

        lane_heading_corr = float(np.clip(
            -(fused_local_xy[1] - self.lane_center_y_local) *
            (0.25 * self.return_heading_gain),
            -(0.25 * self.return_heading_clamp),
            0.25 * self.return_heading_clamp,
        ))
        return heading_target + lane_heading_corr

    def _avoidance_turn_rate(self):
        heading_err = self._wrap_angle(
            self._avoidance_heading_target() - self.current_yaw)
        heading_turn = float(np.clip(
            heading_err * self.avoid_heading_feedback_gain,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        ))
        feedforward_turn = float(np.clip(
            self.override_turn * self.avoid_turn_feedforward_gain,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        ))
        return float(np.clip(
            heading_turn + feedforward_turn,
            -self.max_yaw_rate,
            self.max_yaw_rate,
        ))

    def _publish_avoidance_walk(self):
        saved_vx = self.vx_cmd
        saved_wz = self.wz_cmd

        self.vx_cmd = self._scaled_avoidance_vx(saved_vx)
        self.wz_cmd = self._avoidance_turn_rate()
        try:
            positions = self.get_walk_positions()
        finally:
            self.vx_cmd = saved_vx
            self.wz_cmd = saved_wz
        self._publish_joint_positions(positions, publish_phase=True)

    def _ready_to_return_to_lane(self, fused_local_x):
        ahead_obstacles = self._estimated_obstacles_ahead_local(fused_local_x)
        if ahead_obstacles is not None:
            return len(ahead_obstacles) == 0

        rear_clear_x = (
            self.obstacle_center_x_local +
            self.obstacle_radius_m +
            self.obstacle_rear_clear_margin_m
        )
        return fused_local_x >= rear_clear_x

    def _lane_return_heading_target(self, fused_local_xy):
        ahead_obstacles = self._estimated_obstacles_ahead_local(fused_local_xy[0])
        if ahead_obstacles:
            self.return_to_lane_active = False
            self.return_to_lane_pending = True
            return self.held_avoid_heading_offset

        heading_target = float(np.clip(
            -(fused_local_xy[1] - self.lane_center_y_local) *
            self.return_heading_gain,
            -self.return_heading_clamp,
            self.return_heading_clamp,
        ))
        lane_error = abs(fused_local_xy[1] - self.lane_center_y_local)
        if (lane_error <= self.return_lane_tolerance_m and
                fused_local_xy[0] >= self.return_forward_margin_m):
            self.return_to_lane_active = False
            return self.original_yaw
        return heading_target

    def _clear_heading_target(self):
        heading_target = self.original_yaw
        fused_local_xy = self._fused_local_xy()

        if self.return_to_lane_pending:
            heading_target = self.held_avoid_heading_offset
            if (fused_local_xy is not None and
                    self._ready_to_return_to_lane(fused_local_xy[0])):
                self.return_to_lane_pending = False
                self.return_to_lane_active = True

        if self.return_to_lane_active and fused_local_xy is not None:
            heading_target = self._lane_return_heading_target(fused_local_xy)
        return heading_target

    def _clear_restore_turn(self):
        yaw_err = self._wrap_angle(self._clear_heading_target() - self.current_yaw)
        return float(np.clip(
            yaw_err * self.heading_restore_gain,
            -self.heading_restore_clamp,
            self.heading_restore_clamp,
        ))

    def _publish_clear_walk(self):
        saved_wz = self.wz_cmd
        if self.original_yaw is not None:
            restore_turn = self._clear_restore_turn()
            if abs(self.wz_cmd) < 0.01:
                self.wz_cmd = restore_turn
        try:
            positions = self.get_walk_positions()
        finally:
            self.wz_cmd = saved_wz
        self._publish_joint_positions(positions, publish_phase=True)

    def control_loop(self):
        now = self._now_s()
        self._maybe_auto_start()
        self._maybe_stop_for_experiment_limit()
        self._clear_stale_override(now)

        if self.nav_stage == 3:
            self._publish_joint_positions(self.get_stand_positions())
            return

        if self.nav_stage == 2 and self.mode == 'walk':
            self._publish_avoidance_walk()
            return

        if self.nav_stage == 1 and self.mode == 'walk':
            self._publish_clear_walk()
            return

        self._publish_joint_positions(self.get_stand_positions())

    def _publish_phase(self):
        if self.walk_start_time is None:
            return
        walk_t = self._now_s() - self.walk_start_time
        phase = (self.gait_freq * walk_t) % 1.0
        phases = Float32MultiArray()
        phases.data = [
            float(phase % 1.0),          # fl
            float((phase + 0.5) % 1.0),  # fr
            float((phase + 0.5) % 1.0),  # hl
            float(phase % 1.0),          # hr
            float(self.duty_cycle),
            float(self.gait_freq),
        ]
        self.phase_pub.publish(phases)

    def destroy_node(self):
        self._finalize_obstacle_metrics()
        return super().destroy_node()


def main():
    rclpy.init()
    node = SpotLocomotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

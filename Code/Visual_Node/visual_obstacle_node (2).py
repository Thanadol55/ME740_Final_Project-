import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import Float32MultiArray
from rclpy.qos import qos_profile_sensor_data
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy, Slerp
import time
from collections import deque


class NodeBackedComponent:
    def __init__(self, node):
        object.__setattr__(self, 'node', node)

    def __getattr__(self, name):
        return getattr(self.node, name)

    def __setattr__(self, name, value):
        if name == 'node' or not hasattr(self.node, name):
            object.__setattr__(self, name, value)
        else:
            setattr(self.node, name, value)


class VisualYawEstimator(NodeBackedComponent):
    def derotate_points(self, pts_2d, R_delta_body):
        pts = pts_2d.reshape(-1, 2).astype(np.float64)
        n = len(pts)

        R_delta_cam = self.R_cam_to_body.T @ R_delta_body @ self.R_cam_to_body
        rays = (self.K_inv @ np.vstack([pts.T, np.ones(n)])).T
        rays_rot = (R_delta_cam.T @ rays.T).T
        rays_rot /= rays_rot[:, 2:3]
        return (self.K @ rays_rot.T).T[:, :2].astype(np.float32)

    def compute_relative_rotation(self, pts1, pts2):
        if len(pts1) < self.min_vo_inliers:
            return None, 0

        pts1_f64 = pts1.reshape(-1, 2).astype(np.float64)
        pts2_f64 = pts2.reshape(-1, 2).astype(np.float64)

        E, mask = cv2.findEssentialMat(
            pts1_f64, pts2_f64, self.K, method=cv2.RANSAC,
            prob=0.999, threshold=self.ransac_threshold_px)

        if E is None:
            return None, 0

        _, R_cam, _, mask_pose = cv2.recoverPose(
            E, pts1_f64, pts2_f64, self.K, mask=mask)
        inlier_count = int(np.count_nonzero(mask_pose))

        # note to self: recoverPose yaw is flipped for this update
        R_cam_motion = R_cam.T
        R_rel_full_body = self.R_cam_to_body @ R_cam_motion @ self.R_cam_to_body.T
        yaw_estimate = R_scipy.from_matrix(R_rel_full_body).as_euler('zyx')[0]

        if abs(yaw_estimate) < self.yaw_deadband:
            yaw_estimate = 0.0

        yaw_estimate = np.clip(yaw_estimate, -self.max_rotation_step,
                               self.max_rotation_step)
        return yaw_estimate, inlier_count

    def compute_flow_yaw(self, pts1, pts2):
        """Backup yaw guess from horizontal flow."""
        if len(pts1) < self.min_flow_features:
            return None

        pts1 = pts1.reshape(-1, 2).astype(np.float64)
        pts2 = pts2.reshape(-1, 2).astype(np.float64)

        x1 = (pts1[:, 0] - self.cx) / self.fx
        x2 = (pts2[:, 0] - self.cx) / self.fx
        dx = x2 - x1

        yaw_samples = dx / (1.0 + x1 * x1)

        median = np.median(yaw_samples)
        mad = np.median(np.abs(yaw_samples - median))
        if mad > 0.0:
            keep = np.abs(yaw_samples - median) < (3.0 * 1.4826 * mad + 1e-4)
            yaw_samples = yaw_samples[keep]

        if len(yaw_samples) < self.min_flow_features:
            return None

        yaw_estimate = float(np.median(yaw_samples))
        if abs(yaw_estimate) < self.yaw_deadband:
            yaw_estimate = 0.0

        return float(np.clip(yaw_estimate, -self.max_rotation_step,
                             self.max_rotation_step))

    def compute_quality_score(self, n_inliers, n_rot, yaw_rel, flow_yaw,
                              used_fallback, used_upper):
        inlier_ratio = np.clip(n_inliers / n_rot, 0.0, 1.0)
        confidence_span = self.min_pose_inliers - self.min_publish_inliers
        inlier_count_score = np.clip(
            (n_inliers - self.min_publish_inliers) / confidence_span, 0.0, 1.0)
        support_score = np.clip(
            (n_rot - self.min_rot_features) / self.min_rot_features, 0.0, 1.0)

        if yaw_rel is not None and flow_yaw is not None:
            agreement_span = self.flow_agreement_limit * 6.0
            agreement_score = np.clip(
                1.0 - abs(yaw_rel - flow_yaw) / agreement_span, 0.0, 1.0)
        elif yaw_rel is not None:
            agreement_score = 0.6
        elif flow_yaw is not None:
            agreement_score = 0.35
        else:
            agreement_score = 0.0

        quality = (
            0.40 * inlier_ratio +
            0.25 * inlier_count_score +
            0.20 * support_score +
            0.15 * agreement_score
        )
        if used_fallback:
            quality *= 0.55
        if used_upper:
            quality *= 1.05
        else:
            quality *= 0.95

        return float(np.clip(quality, 0.05, 1.0))

    def quality_to_yaw_variance(self, quality_score):
        quality_gap = 1.0 - quality_score
        variance = (
            self.frontend_yaw_var_floor +
            (quality_gap * quality_gap) *
            (self.frontend_yaw_var_ceiling - self.frontend_yaw_var_floor)
        )
        return float(np.clip(
            variance,
            self.frontend_yaw_var_floor,
            self.frontend_yaw_var_ceiling,
        ))

    def detect_features(self, gray, horizon_y):
        mask_upper = np.zeros_like(gray)
        mask_upper[0:horizon_y, :] = 255
        pts_upper = cv2.goodFeaturesToTrack(
            gray, maxCorners=300, qualityLevel=0.01,
            minDistance=7, blockSize=7, mask=mask_upper)

        mask_lower = np.zeros_like(gray)
        mask_lower[horizon_y:, :] = 255
        pts_lower = cv2.goodFeaturesToTrack(
            gray, maxCorners=100, qualityLevel=0.01,
            minDistance=10, blockSize=7, mask=mask_lower)

        if pts_upper is not None and pts_lower is not None:
            return np.vstack([pts_upper, pts_lower])
        elif pts_upper is not None:
            return pts_upper
        elif pts_lower is not None:
            return pts_lower
        return None


class ReactiveObstacleAvoidance(NodeBackedComponent):
    def ground_point_from_pixel(self, u, v, R_body_world, pos_world):
        if pos_world[2] <= 1e-3:
            return None
        ray_cam = self.K_inv @ np.array([u, v, 1.0], dtype=np.float64)
        ray_body = self.R_cam_to_body @ ray_cam
        ray_world = R_body_world @ ray_body
        if ray_world[2] >= -1e-3:
            return None
        scale = -pos_world[2] / ray_world[2]
        point_world = pos_world + scale * ray_world
        return point_world[:2]

    def extract_obstacle_estimates(
            self, contours, roi_top, roi_shape, near_row, center_left, center_right,
            roi_area, img_w, R_body_world, yaw_world, pos_world):
        estimates = []
        center_band_width = float(center_right - center_left)
        cy = np.cos(yaw_world)
        sy = np.sin(yaw_world)
        for cnt in contours:
            contour_area = float(cv2.contourArea(cnt))
            if contour_area <= 0.0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            bottom = y + h
            if bottom < near_row:
                continue

            area_frac = contour_area / roi_area
            overlap = max(0, min(x + w, center_right) - max(x, center_left))
            overlap_frac = overlap / center_band_width
            touches_side = (x <= 1) or (x + w >= img_w - 1)
            touches_upper = (y <= 1)
            if touches_side and touches_upper and overlap_frac < 0.45:
                continue
            if area_frac < self.obstacle_min_center_area_fraction * 0.35 and overlap_frac < 0.08:
                continue

            v_bottom = float(roi_top + bottom - 1)
            u_center = float(x + 0.5 * w)
            u_left = float(x + 0.2 * w)
            u_right = float(x + 0.8 * w)

            world_center = self.ground_point_from_pixel(
                u_center, v_bottom, R_body_world, pos_world)
            world_left = self.ground_point_from_pixel(
                u_left, v_bottom, R_body_world, pos_world)
            world_right = self.ground_point_from_pixel(
                u_right, v_bottom, R_body_world, pos_world)
            if world_center is None:
                continue

            delta_xy = world_center - pos_world[:2]
            forward_m = cy * delta_xy[0] + sy * delta_xy[1]
            lateral_m = -sy * delta_xy[0] + cy * delta_xy[1]
            if forward_m < self.obstacle_min_forward_range_m:
                continue
            if forward_m > self.obstacle_max_forward_range_m:
                continue

            if world_left is not None and world_right is not None:
                diameter_m = float(np.linalg.norm(world_right - world_left))
                radius_m = 0.5 * diameter_m
            else:
                radius_m = 0.25 + 0.35 * np.sqrt(area_frac)
            radius_m = float(np.clip(
                radius_m,
                self.obstacle_radius_min_m,
                self.obstacle_radius_max_m,
            ))

            estimates.append({
                'x': float(world_center[0]),
                'y': float(world_center[1]),
                'r': radius_m,
                'forward': float(forward_m),
                'lateral': float(lateral_m),
                'stamp': None,
            })

        estimates.sort(key=lambda item: item['forward'])
        return estimates[:self.max_tracked_obstacles]

    def trim_tracked_obstacles(self, stamp):
        self.tracked_obstacles_world = [
            obs for obs in self.tracked_obstacles_world
            if stamp - obs['stamp'] <= self.obstacle_track_ttl
        ]

    def update_tracked_obstacles(self, estimates, stamp):
        self.trim_tracked_obstacles(stamp)
        for estimate in estimates:
            best_idx = None
            best_dist = float('inf')
            for idx, tracked in enumerate(self.tracked_obstacles_world):
                dist = np.hypot(
                    estimate['x'] - tracked['x'],
                    estimate['y'] - tracked['y'],
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None and best_dist <= self.obstacle_track_merge_distance_m:
                tracked = self.tracked_obstacles_world[best_idx]
                tracked['x'] = 0.45 * tracked['x'] + 0.55 * estimate['x']
                tracked['y'] = 0.45 * tracked['y'] + 0.55 * estimate['y']
                tracked['r'] = 0.45 * tracked['r'] + 0.55 * estimate['r']
                tracked['stamp'] = stamp
            else:
                estimate['stamp'] = stamp
                self.tracked_obstacles_world.append(estimate)

        self.tracked_obstacles_world.sort(
            key=lambda obs: np.hypot(
                obs['x'] - self.current_pos[0],
                obs['y'] - self.current_pos[1],
            )
        )
        self.tracked_obstacles_world = self.tracked_obstacles_world[:self.max_tracked_obstacles]

    def tracked_obstacles_local(self, yaw_world, pos_world, stamp):
        self.trim_tracked_obstacles(stamp)
        cy = np.cos(yaw_world)
        sy = np.sin(yaw_world)
        local_obstacles = []
        for obstacle in self.tracked_obstacles_world:
            delta_xy = np.array([
                obstacle['x'] - pos_world[0],
                obstacle['y'] - pos_world[1],
            ], dtype=np.float64)
            forward_m = cy * delta_xy[0] + sy * delta_xy[1]
            lateral_m = -sy * delta_xy[0] + cy * delta_xy[1]
            if forward_m < self.obstacle_min_forward_range_m:
                continue
            if forward_m > self.obstacle_max_forward_range_m:
                continue
            local_obstacles.append({
                'forward': float(forward_m),
                'lateral': float(lateral_m),
                'r': float(obstacle['r']),
            })
        local_obstacles.sort(key=lambda item: item['forward'])
        return local_obstacles

    def filter_nav_decision(self, raw_stage, speed_scale, turn_rate, heading_offset, turn_sign):
        if raw_stage >= 2:
            self.blocked_frame_streak += 1
            self.clear_frame_streak = 0
            self.last_nav_turn = turn_rate
            self.last_nav_speed = speed_scale
            self.last_nav_heading_offset = heading_offset
            self.locked_turn_sign = turn_sign
            if self.blocked_frame_streak >= self.avoid_confirm_frames:
                self.nav_stage_state = 2
        else:
            self.clear_frame_streak += 1
            self.blocked_frame_streak = 0
            if self.clear_frame_streak >= self.clear_confirm_frames:
                self.nav_stage_state = 1
                self.locked_turn_sign = 0.0

        if self.nav_stage_state >= 2:
            if raw_stage >= 2:
                return 2, speed_scale, turn_rate, heading_offset
            return (
                2,
                max(self.last_nav_speed, self.avoid_speed_floor),
                self.last_nav_turn * 0.85,
                self.last_nav_heading_offset * 0.90,
            )
        return 1, 1.0, 0.0, 0.0

    def choose_turn_sign(self, left_score, right_score):
        side_delta = right_score - left_score
        if abs(side_delta) >= self.avoid_turn_decision_deadband:
            return 1.0 if side_delta >= 0.0 else -1.0
        if abs(self.locked_turn_sign) > 0.5:
            return self.locked_turn_sign
        return self.default_avoid_turn_sign

    def compute_obstacle_override(
            self, gray, horizon_y, tracked_pts=None,
            R_body_world=None, yaw_world=None, pos_world=None, image_stamp=0.0):
        img_h, img_w = gray.shape
        roi_top = max(horizon_y, int(img_h * self.obstacle_roi_top_fraction))
        roi_bottom = max(roi_top + 10, int(img_h * (1.0 - self.obstacle_bottom_crop_fraction)))
        roi = gray[roi_top:roi_bottom, :]
        if roi.size == 0:
            return (
                1, 1.0, 0.0, 0.0,
                self.obstacle_scores.copy(), len(self.tracked_obstacles_world)
            )

        blur = cv2.GaussianBlur(roi, (5, 5), 0)
        grad_x = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        edge_mask = (grad_x > self.obstacle_edge_threshold).astype(np.float32)
        dark_mask = (blur < self.obstacle_dark_threshold).astype(np.uint8)
        occupancy_mask = np.logical_or(
            grad_x > self.obstacle_edge_threshold,
            dark_mask > 0,
        ).astype(np.uint8)
        occupancy_mask = cv2.morphologyEx(
            occupancy_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        )
        row_weights = np.linspace(
            0.7, 1.4, edge_mask.shape[0], dtype=np.float32).reshape(-1, 1)
        weighted_edges = edge_mask * row_weights

        center_left = int(img_w * (0.5 - 0.5 * self.obstacle_center_fraction))
        center_right = int(img_w * (0.5 + 0.5 * self.obstacle_center_fraction))
        near_row = int(weighted_edges.shape[0] * self.obstacle_near_fraction)
        bottom_focus_row = int(
            weighted_edges.shape[0] * self.obstacle_bottom_focus_fraction)

        def _mean_region(arr, x0, x1, y0=0):
            if x1 <= x0 or y0 >= arr.shape[0]:
                return 0.0
            region = arr[y0:, x0:x1]
            return float(np.mean(region))

        left_edge = _mean_region(weighted_edges, 0, center_left)
        center_edge = _mean_region(weighted_edges, center_left, center_right)
        right_edge = _mean_region(weighted_edges, center_right, img_w)
        near_center_edge = _mean_region(weighted_edges, center_left, center_right, near_row)

        def _column_occupancy_fraction(arr, x0, x1, y0):
            if x1 <= x0 or y0 >= arr.shape[0]:
                return 0.0
            region = arr[y0:, x0:x1]
            occupied_cols = np.any(region > 0, axis=0)
            return float(np.mean(occupied_cols))

        center_width = _column_occupancy_fraction(
            occupancy_mask, center_left, center_right, bottom_focus_row)
        near_center_width = _column_occupancy_fraction(
            occupancy_mask, center_left, center_right, near_row)
        center_mid = (center_left + center_right) // 2
        lane_left_occ = _column_occupancy_fraction(
            occupancy_mask, center_left, center_mid, near_row)
        lane_right_occ = _column_occupancy_fraction(
            occupancy_mask, center_mid, center_right, near_row)
        center_dark = _mean_region(
            dark_mask.astype(np.float32), center_left, center_right, bottom_focus_row)
        near_center_dark = _mean_region(
            dark_mask.astype(np.float32), center_left, center_right, near_row)
        center_area = 0.0
        feature_left = 0.0
        feature_center = 0.0
        feature_right = 0.0
        roi_area = float(occupancy_mask.shape[0] * occupancy_mask.shape[1])
        contours, _ = cv2.findContours(
            occupancy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_band_width = float(center_right - center_left)
        for cnt in contours:
            contour_area = float(cv2.contourArea(cnt))
            if contour_area <= 0.0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            bottom = y + h
            overlap = max(0, min(x + w, center_right) - max(x, center_left))
            overlap_frac = overlap / center_band_width
            area_frac = contour_area / roi_area
            if bottom < near_row:
                continue
            if (area_frac < self.obstacle_min_center_area_fraction and
                    overlap_frac < self.obstacle_min_center_overlap_fraction):
                continue
            center_area = max(center_area, area_frac)

        obstacle_estimates = self.extract_obstacle_estimates(
            contours, roi_top, occupancy_mask.shape, near_row,
            center_left, center_right, roi_area, img_w,
            self.current_R if R_body_world is None else R_body_world,
            self.current_yaw if yaw_world is None else yaw_world,
            self.current_pos if pos_world is None else pos_world,
        )
        self.update_tracked_obstacles(obstacle_estimates, image_stamp)
        tracked_local_obstacles = self.tracked_obstacles_local(
            self.current_yaw if yaw_world is None else yaw_world,
            self.current_pos if pos_world is None else pos_world,
            image_stamp,
        )

        if tracked_pts is not None and len(tracked_pts) > 0:
            pts = np.asarray(tracked_pts, dtype=np.float32).reshape(-1, 2)
            valid = (
                (pts[:, 1] >= (roi_top + near_row)) & (pts[:, 1] < roi_bottom) &
                (pts[:, 0] >= center_left) & (pts[:, 0] < center_right)
            )
            pts = pts[valid]
            if pts.size > 0:
                pts_y = pts[:, 1] - roi_top
                pts_x = pts[:, 0]
                center_third_left = center_left + (center_right - center_left) / 3.0
                center_third_right = center_right - (center_right - center_left) / 3.0
                y_weights = 0.7 + 0.6 * np.clip(
                    pts_y / float(roi.shape[0]), 0.0, 1.0)
                left_mask = pts_x < center_mid
                center_mask = (
                    (pts_x >= center_third_left) & (pts_x < center_third_right))
                right_mask = pts_x >= center_mid
                feature_norm = max(float(len(pts)), 12.0)
                feature_left = float(np.sum(y_weights[left_mask]) / feature_norm)
                feature_center = float(np.sum(y_weights[center_mask]) / feature_norm)
                feature_right = float(np.sum(y_weights[right_mask]) / feature_norm)

        raw_scores = {
            'left': left_edge, 'center': center_edge, 'right': right_edge,
            'near_center': near_center_edge,
            'lane_left_occ': lane_left_occ, 'lane_right_occ': lane_right_occ,
            'center_width': center_width, 'near_width': near_center_width,
            'center_dark': center_dark, 'near_dark': near_center_dark,
            'center_area': center_area,
            'feature_left': feature_left, 'feature_center': feature_center,
            'feature_right': feature_right,
        }

        alpha = self.obstacle_score_ema_alpha
        for key, value in raw_scores.items():
            self.obstacle_scores[key] = (
                (1.0 - alpha) * self.obstacle_scores[key] + alpha * float(value)
            )

        self.frame_stats['nav_center_sum'] += self.obstacle_scores['center']
        self.frame_stats['nav_near_sum'] += self.obstacle_scores['near_center']
        self.frame_stats['nav_center_width_sum'] += self.obstacle_scores['center_width']
        self.frame_stats['nav_near_width_sum'] += self.obstacle_scores['near_width']
        self.frame_stats['nav_center_dark_sum'] += self.obstacle_scores['center_dark']
        self.frame_stats['nav_near_dark_sum'] += self.obstacle_scores['near_dark']
        self.frame_stats['nav_center_area_sum'] += self.obstacle_scores['center_area']

        center_score = self.obstacle_scores['center']
        near_center_score = self.obstacle_scores['near_center']
        center_width = self.obstacle_scores['center_width']
        near_center_width = self.obstacle_scores['near_width']
        center_dark = self.obstacle_scores['center_dark']
        near_center_dark = self.obstacle_scores['near_dark']
        center_area = self.obstacle_scores['center_area']
        lane_left_occ = self.obstacle_scores['lane_left_occ']
        lane_right_occ = self.obstacle_scores['lane_right_occ']
        feature_left = self.obstacle_scores['feature_left']
        feature_center = self.obstacle_scores['feature_center']
        feature_right = self.obstacle_scores['feature_right']
        left_score = self.obstacle_scores['left']
        right_score = self.obstacle_scores['right']
        tracked_left_occ = 0.0
        tracked_right_occ = 0.0
        tracked_center_blocked = False
        tracked_near_bonus = 0.0
        for obstacle in tracked_local_obstacles:
            corridor_margin = obstacle['r'] + self.tracked_corridor_margin_m
            if abs(obstacle['lateral']) > max(0.9, 2.0 * corridor_margin):
                continue
            distance_gain = float(np.clip(
                (self.tracked_replan_distance_m - obstacle['forward']) /
                self.tracked_replan_distance_m,
                0.0, 1.0,
            ))
            if obstacle['lateral'] >= -obstacle['r']:
                tracked_left_occ = max(tracked_left_occ, distance_gain)
            if obstacle['lateral'] <= obstacle['r']:
                tracked_right_occ = max(tracked_right_occ, distance_gain)
            if abs(obstacle['lateral']) <= corridor_margin:
                tracked_center_blocked = True
                tracked_near_bonus = max(tracked_near_bonus, distance_gain)

        lane_left_occ = max(self.obstacle_scores['lane_left_occ'], tracked_left_occ)
        lane_right_occ = max(self.obstacle_scores['lane_right_occ'], tracked_right_occ)
        nav_left_score = (
            max(0.0, 1.0 - lane_left_occ) - self.feature_side_weight * feature_left)
        nav_right_score = (
            max(0.0, 1.0 - lane_right_occ) - self.feature_side_weight * feature_right)
        side_mean = 0.5 * (left_score + right_score)
        turn_sign = self.choose_turn_sign(nav_left_score, nav_right_score)

        width_blocked = (
            center_width > self.obstacle_center_width_threshold or
            near_center_width > self.obstacle_near_width_threshold
        )
        dark_blocked = (
            center_dark > self.obstacle_center_dark_threshold or
            near_center_dark > self.obstacle_near_dark_threshold
        )
        area_blocked = center_area > self.obstacle_min_center_area_fraction
        feature_center_blocked = (
            feature_center > self.feature_center_trigger and
            (center_width > self.obstacle_center_width_threshold * 0.35 or
             center_area > self.obstacle_min_center_area_fraction * 0.30 or
             center_dark > self.obstacle_center_dark_threshold * 0.45 or
             near_center_score > self.obstacle_near_threshold * 0.75)
        )
        center_blocked = (
            (width_blocked or dark_blocked or area_blocked) and (
                near_center_score > self.obstacle_near_threshold or
                center_score > self.obstacle_blocked_center_threshold or
                (center_score > side_mean * self.obstacle_ratio_threshold and
                 center_score > 0.05) or
                dark_blocked or
                area_blocked
            )
        )
        center_blocked = center_blocked or feature_center_blocked or tracked_center_blocked

        if center_blocked:
            near_excess = max(
                near_center_score - self.obstacle_near_threshold, 0.0)
            center_excess = max(
                center_score - self.obstacle_blocked_center_threshold, 0.0)
            feature_side_excess = max(
                abs(feature_right - feature_left) - self.feature_side_deadband, 0.0)
            tracked_side_excess = max(
                abs(tracked_right_occ - tracked_left_occ) - 0.05, 0.0)
            turn_mag = np.clip(
                self.min_avoid_turn +
                abs(nav_right_score - nav_left_score) * self.avoid_turn_gain +
                feature_side_excess * 0.35 +
                near_excess * 0.8 +
                tracked_side_excess * 0.45 +
                tracked_near_bonus * 0.40,
                self.min_avoid_turn,
                self.max_avoid_turn,
            )
            turn_rate = turn_sign * turn_mag
            heading_offset = turn_sign * float(np.clip(
                self.min_avoid_heading_offset +
                abs(nav_right_score - nav_left_score) * self.avoid_heading_gain +
                feature_side_excess * 0.22 +
                near_excess * 0.55 +
                tracked_side_excess * 0.35 +
                tracked_near_bonus * 0.30,
                self.min_avoid_heading_offset,
                self.max_avoid_heading_offset,
            ))
            speed_scale = float(np.clip(
                0.92 - 1.2 * center_excess - 1.8 * near_excess - 0.35 * tracked_near_bonus,
                self.avoid_speed_floor,
                0.90,
            ))
            stage = 2
        else:
            stage = 1
            speed_scale = 1.0
            turn_rate = 0.0
            heading_offset = 0.0

        stage, speed_scale, turn_rate, heading_offset = self.filter_nav_decision(
            stage, speed_scale, turn_rate, heading_offset, turn_sign)
        return (
            stage, speed_scale, turn_rate, heading_offset,
            self.obstacle_scores.copy(), len(self.tracked_obstacles_world)
        )


class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')
        self.bridge = CvBridge()

        # loose sync since sim topics do not always start together
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, '/iekf/odom', self.odom_callback, 10)
        self.phase_sub = self.create_subscription(
            Float32MultiArray, '/gait/phase', self.phase_callback, 10)

        self.vo_pub = self.create_publisher(PoseWithCovarianceStamped, '/vo/pose', 10)
        self.override_pub = self.create_publisher(Twist, '/cmd_vel_override', 10)
        self.obstacles_pub = self.create_publisher(
            Float32MultiArray, '/estimated_obstacles_world', 10)

        self.MAX_ODOM_STAMP_AGE = 0.10
        self.MAX_ODOM_WALL_AGE = 0.25
        self.MAX_STAMP_COMPARISON_SKEW = 5.0
        self.ODOM_BUFFER_DURATION = 1.0
        self._last_odom_time = None
        self._last_odom_wall_time = None
        self._last_odom_stamp = None
        self._last_odom_age = None
        self._last_odom_callback_age = None
        self._last_sync_mode = 'none'
        self._bad_odom_quat_count = 0
        self.odom_buffer = deque()

        self.current_R = np.eye(3)
        self.current_yaw = 0.0
        self.current_pos = np.zeros(3, dtype=np.float64)
        self.visual_yaw = None
        self.prev_gray = None
        self.prev_pts  = None
        self.prev_R = np.eye(3)

        # Isaac camera numbers from my sim
        self.fx = 1108.5
        self.fy = 1108.5
        self.cx = 640.0
        self.cy = 360.0
        self.K = np.array([[self.fx, 0.0, self.cx],
                           [0.0, self.fy, self.cy],
                           [0.0, 0.0, 1.0]])
        self.K_inv = np.linalg.inv(self.K)

        # camera z maps to robot x
        self.R_cam_to_body = np.array([
            [0,  0,  1],
            [-1, 0,  0],
            [0, -1,  0]
        ], dtype=np.float64)
        self._verify_cam_to_body()

        self.vo_initialized = False

        # gait timing, mainly for impact frames
        self.gait_phase = {'fl': 0.0, 'fr': 0.5, 'hl': 0.5, 'hr': 0.0}
        self.duty_cycle = 0.55
        self.phase_received = False
        self.impact_epsilon = 0.12

        # VO values tuned for this camera/odom setup
        self.min_vo_inliers = 12
        self.min_publish_inliers = 10
        self.min_pose_inliers = 16
        self.min_rot_features = 24
        self.min_flow_features = 32
        self.horizon_fraction = 0.68
        self.max_rotation_step = 0.05
        self.yaw_deadband = 0.025
        self.ransac_threshold_px = 1.5
        self.flow_fallback_limit = 0.02
        self.flow_agreement_limit = 0.01
        self.frontend_yaw_var_floor = 0.015
        self.frontend_yaw_var_ceiling = 0.20

        # obstacle values I have been tuning
        self.enable_reactive_avoidance = self.declare_parameter(
            'enable_reactive_avoidance', True).value
        self.obstacle_roi_top_fraction = 0.48
        self.obstacle_bottom_crop_fraction = 0.04
        self.obstacle_center_fraction = 0.42
        self.obstacle_near_fraction = 0.40
        self.obstacle_bottom_focus_fraction = 0.72
        self.obstacle_edge_threshold = 28.0
        self.obstacle_dark_threshold = 85
        self.obstacle_score_ema_alpha = 0.25
        self.obstacle_blocked_center_threshold = 0.12
        self.obstacle_near_threshold = 0.14
        self.obstacle_ratio_threshold = 1.20
        self.obstacle_center_width_threshold = 0.18
        self.obstacle_near_width_threshold = 0.12
        self.obstacle_center_dark_threshold = 0.18
        self.obstacle_near_dark_threshold = 0.14
        self.obstacle_min_center_area_fraction = 0.015
        self.obstacle_min_center_overlap_fraction = 0.30
        self.feature_side_weight = 0.14
        self.feature_center_trigger = 0.16
        self.feature_side_deadband = 0.06
        self.avoid_turn_gain = 1.00
        self.min_avoid_turn = 0.08
        self.max_avoid_turn = 0.20
        self.avoid_speed_floor = 0.45
        self.min_avoid_heading_offset = 0.10
        self.max_avoid_heading_offset = 0.28
        self.avoid_heading_gain = 0.90
        self.avoid_turn_decision_deadband = 0.03
        self.default_avoid_turn_sign = 1.0
        self.avoid_confirm_frames = 2
        self.clear_confirm_frames = 4
        self.nav_stage_state = 1
        self.blocked_frame_streak = 0
        self.clear_frame_streak = 0
        self.last_nav_turn = 0.0
        self.last_nav_speed = 1.0
        self.last_nav_heading_offset = 0.0
        self.locked_turn_sign = 0.0
        self.obstacle_scores = {
            'left': 0.0, 'center': 0.0, 'right': 0.0,
            'near_center': 0.0,
            'lane_left_occ': 0.0, 'lane_right_occ': 0.0,
            'center_width': 0.0, 'near_width': 0.0,
            'center_dark': 0.0, 'near_dark': 0.0,
            'center_area': 0.0,
            'feature_left': 0.0, 'feature_center': 0.0, 'feature_right': 0.0,
        }
        self.frame_stats = {
            'frames_total': 0, 'odom_fresh': 0,
            'candidate_frames': 0, 'candidate_impact': 0, 'candidate_nonimpact': 0,
            'published': 0, 'published_impact': 0, 'published_nonimpact': 0,
            'low_tracks': 0, 'stale_odom': 0, 'pose_fail': 0,
            'low_inliers': 0, 'flow_fallback': 0,
            'quality_sum': 0.0, 'yaw_var_sum': 0.0,
            'nav_clear': 0, 'nav_avoid': 0, 'nav_stop': 0,
            'nav_center_sum': 0.0, 'nav_near_sum': 0.0,
            'nav_center_width_sum': 0.0, 'nav_near_width_sum': 0.0,
            'nav_center_dark_sum': 0.0, 'nav_near_dark_sum': 0.0,
            'nav_center_area_sum': 0.0,
        }

        # keep obstacles briefly so turns do not flicker
        self.obstacle_track_ttl = 1.2
        self.obstacle_track_merge_distance_m = 0.8
        self.max_tracked_obstacles = 4
        self.obstacle_min_forward_range_m = 0.4
        self.obstacle_max_forward_range_m = 8.0
        self.obstacle_radius_min_m = 0.18
        self.obstacle_radius_max_m = 1.20
        self.tracked_corridor_margin_m = 0.35
        self.tracked_replan_distance_m = 4.0
        self.tracked_obstacles_world = []

        self.redetect_threshold = 80
        self.frontend = VisualYawEstimator(self)
        self.avoidance = ReactiveObstacleAvoidance(self)
        self.get_logger().info(f'avoidance enabled: {self.enable_reactive_avoidance}')

    def _verify_cam_to_body(self):
        cam_fwd   = np.array([0.0, 0.0, 1.0])
        cam_right = np.array([1.0, 0.0, 0.0])
        cam_down  = np.array([0.0, 1.0, 0.0])

        body_fwd   = self.R_cam_to_body @ cam_fwd
        body_right = self.R_cam_to_body @ cam_right
        body_down  = self.R_cam_to_body @ cam_down

        ok = (np.allclose(body_fwd,   [ 1,  0,  0], atol=0.01) and
              np.allclose(body_right,  [ 0, -1,  0], atol=0.01) and
              np.allclose(body_down,   [ 0,  0, -1], atol=0.01))

        if ok:
            self.get_logger().info('camera transform ok')
        else:
            self.get_logger().error(
                f'camera transform bad: fwd={body_fwd}, '
                f'right={body_right}, down={body_down}'
            )

    def odom_callback(self, msg):
        q = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ], dtype=np.float64)
        if (not np.all(np.isfinite(q))) or np.linalg.norm(q) < 0.5:
            self._bad_odom_quat_count += 1
            if self._bad_odom_quat_count % 20 == 1:
                self.get_logger().warn('bad odom quaternion, skipping')
            return

        q = q / np.linalg.norm(q)
        R_now = R_scipy.from_quat(q).as_matrix()
        yaw_now = R_scipy.from_quat(q).as_euler('zyx')[0]
        odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        self.current_R = R_now
        self.current_yaw = yaw_now
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ], dtype=np.float64)
        self._last_odom_time = self.get_clock().now()
        self._last_odom_wall_time = time.monotonic()
        self._last_odom_stamp = odom_stamp
        self.odom_buffer.append({
            'stamp': odom_stamp,
            'R': R_now.copy(),
            'yaw': float(yaw_now),
            'pos': self.current_pos.copy(),
        })
        self._trim_odom_buffer(odom_stamp)

        if not self.vo_initialized:
            self.vo_initialized = True
        if self.visual_yaw is None:
            self.visual_yaw = self.current_yaw

    def _trim_odom_buffer(self, newest_stamp):
        min_stamp = newest_stamp - self.ODOM_BUFFER_DURATION
        while len(self.odom_buffer) > 2 and self.odom_buffer[0]['stamp'] < min_stamp:
            self.odom_buffer.popleft()

    def _lookup_odom_state(self, image_stamp):
        callback_age = None
        if self._last_odom_wall_time is not None:
            callback_age = time.monotonic() - self._last_odom_wall_time
        self._last_odom_callback_age = callback_age

        if not self.odom_buffer:
            self._last_odom_age = None
            self._last_sync_mode = 'none'
            return (
                False, self.current_R.copy(), float(self.current_yaw),
                self.current_pos.copy(),
            )

        comparable_stamp = (
            self._last_odom_stamp is not None and image_stamp > 0.0 and
            abs(image_stamp - self._last_odom_stamp) < self.MAX_STAMP_COMPARISON_SKEW
        )

        odom_list = list(self.odom_buffer)
        if comparable_stamp:
            if image_stamp <= odom_list[0]['stamp']:
                nearest = odom_list[0]
                nearest_age = abs(image_stamp - nearest['stamp'])
                self._last_odom_age = nearest_age
                self._last_sync_mode = 'nearest_first'
                if nearest_age < self.MAX_ODOM_STAMP_AGE:
                    return (
                        True, nearest['R'].copy(), float(nearest['yaw']),
                        nearest['pos'].copy(),
                    )
            else:
                for prev_state, next_state in zip(odom_list[:-1], odom_list[1:]):
                    t0 = prev_state['stamp']
                    t1 = next_state['stamp']
                    if t0 <= image_stamp <= t1:
                        span = t1 - t0
                        self._last_odom_age = min(image_stamp - t0, t1 - image_stamp)
                        if span <= self.MAX_ODOM_STAMP_AGE * 2.0:
                            if span <= 1e-6:
                                self._last_sync_mode = 'nearest_same'
                                return (
                                    True, prev_state['R'].copy(),
                                    float(prev_state['yaw']),
                                    prev_state['pos'].copy(),
                                )
                            rots = R_scipy.from_matrix(np.stack([prev_state['R'], next_state['R']], axis=0))
                            slerp = Slerp([t0, t1], rots)
                            R_interp = slerp([image_stamp]).as_matrix()[0]
                            yaw_interp = R_scipy.from_matrix(R_interp).as_euler('zyx')[0]
                            alpha = float(np.clip((image_stamp - t0) / span, 0.0, 1.0))
                            pos_interp = (
                                (1.0 - alpha) * prev_state['pos'] +
                                alpha * next_state['pos']
                            )
                            self._last_sync_mode = 'interp'
                            return True, R_interp, float(yaw_interp), pos_interp
                        break

                nearest = min(odom_list, key=lambda item: abs(image_stamp - item['stamp']))
                nearest_age = abs(image_stamp - nearest['stamp'])
                self._last_odom_age = nearest_age
                self._last_sync_mode = 'nearest'
                if nearest_age < self.MAX_ODOM_STAMP_AGE:
                    return (
                        True, nearest['R'].copy(), float(nearest['yaw']),
                        nearest['pos'].copy(),
                    )
        else:
            self._last_odom_age = None

        if callback_age is not None and callback_age < self.MAX_ODOM_WALL_AGE:
            self._last_sync_mode = 'wall_fallback'
            return (
                True, self.current_R.copy(), float(self.current_yaw),
                self.current_pos.copy(),
            )

        self._last_sync_mode = 'stale'
        return (
            False, self.current_R.copy(), float(self.current_yaw),
            self.current_pos.copy(),
        )

    def phase_callback(self, msg):
        if len(msg.data) >= 5:
            self.gait_phase['fl'] = msg.data[0]
            self.gait_phase['fr'] = msg.data[1]
            self.gait_phase['hl'] = msg.data[2]
            self.gait_phase['hr'] = msg.data[3]
            self.duty_cycle = msg.data[4]
            self.phase_received = True

    def is_impact_window(self):
        if not self.phase_received:
            return False
        for leg in ('fl', 'fr', 'hl', 'hr'):
            phase = self.gait_phase[leg]
            dist_touchdown = min(phase, 1.0 - phase)
            if dist_touchdown < self.impact_epsilon:
                return True
        return False

    def _wrap_angle(self, angle):
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def derotate_points(self, pts_2d, R_delta_body):
        return self.frontend.derotate_points(pts_2d, R_delta_body)

    def image_callback(self, msg):
        image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        odom_fresh, R_sync, yaw_sync, pos_sync = self._lookup_odom_state(image_stamp)
        self._process_image(msg, odom_fresh, image_stamp, R_sync, yaw_sync, pos_sync)

    def _ground_point_from_pixel(self, u, v, R_body_world, pos_world):
        return self.avoidance.ground_point_from_pixel(
            u, v, R_body_world, pos_world)

    def _extract_obstacle_estimates(
            self, contours, roi_top, roi_shape, near_row, center_left, center_right,
            roi_area, img_w, R_body_world, yaw_world, pos_world):
        return self.avoidance.extract_obstacle_estimates(
            contours, roi_top, roi_shape, near_row, center_left, center_right,
            roi_area, img_w, R_body_world, yaw_world, pos_world)

    def _trim_tracked_obstacles(self, stamp):
        self.avoidance.trim_tracked_obstacles(stamp)

    def _update_tracked_obstacles(self, estimates, stamp):
        self.avoidance.update_tracked_obstacles(estimates, stamp)

    def _publish_tracked_obstacles(self, stamp):
        self._trim_tracked_obstacles(stamp)
        msg = Float32MultiArray()
        data = []
        for obstacle in self.tracked_obstacles_world:
            data.extend([obstacle['x'], obstacle['y'], obstacle['r']])
        msg.data = data
        self.obstacles_pub.publish(msg)

    def _tracked_obstacles_local(self, yaw_world, pos_world, stamp):
        return self.avoidance.tracked_obstacles_local(
            yaw_world, pos_world, stamp)

    def _compute_relative_rotation(self, pts1, pts2):
        return self.frontend.compute_relative_rotation(pts1, pts2)

    def _compute_flow_yaw(self, pts1, pts2):
        return self.frontend.compute_flow_yaw(pts1, pts2)

    def _compute_quality_score(self, n_inliers, n_rot, yaw_rel, flow_yaw,
                               used_fallback, used_upper):
        return self.frontend.compute_quality_score(
            n_inliers, n_rot, yaw_rel, flow_yaw, used_fallback, used_upper)

    def _quality_to_yaw_variance(self, quality_score):
        return self.frontend.quality_to_yaw_variance(quality_score)

    def _publish_visual_heading(self, stamp, quality_score, yaw_variance):
        if self.visual_yaw is None:
            return

        m = PoseWithCovarianceStamped()
        m.header.stamp = stamp
        m.header.frame_id = 'odom'
        m.pose.pose.position.x = 0.0
        m.pose.pose.position.y = 0.0
        m.pose.pose.position.z = 0.0
        q = R_scipy.from_euler('z', self.visual_yaw).as_quat()
        m.pose.pose.orientation.x = float(q[0])
        m.pose.pose.orientation.y = float(q[1])
        m.pose.pose.orientation.z = float(q[2])
        m.pose.pose.orientation.w = float(q[3])
        covariance = [0.0] * 36
        # fusion node expects these two slots
        covariance[0] = float(np.clip(quality_score, 0.0, 1.0))
        covariance[35] = float(yaw_variance)
        m.pose.covariance = covariance
        self.vo_pub.publish(m)

    def _publish_nav_override(self, stage, speed_scale, turn_rate, heading_offset):
        cmd = Twist()
        if stage == 3:
            cmd.linear.x = 0.0
            cmd.angular.z = float(np.clip(turn_rate, -self.max_avoid_turn, self.max_avoid_turn))
            cmd.angular.x = float(np.clip(
                heading_offset,
                -self.max_avoid_heading_offset,
                self.max_avoid_heading_offset,
            ))
            cmd.linear.z = 1.0
            self.frame_stats['nav_stop'] += 1
        elif stage == 2:
            cmd.linear.x = float(np.clip(speed_scale, self.avoid_speed_floor, 1.0))
            cmd.angular.z = float(np.clip(turn_rate, -self.max_avoid_turn, self.max_avoid_turn))
            cmd.angular.x = float(np.clip(
                heading_offset,
                -self.max_avoid_heading_offset,
                self.max_avoid_heading_offset,
            ))
            cmd.linear.z = 0.5
            self.frame_stats['nav_avoid'] += 1
        else:
            cmd.linear.x = 1.0
            cmd.angular.z = 0.0
            cmd.angular.x = 0.0
            cmd.linear.z = 0.0
            self.frame_stats['nav_clear'] += 1
        self.override_pub.publish(cmd)

    def _filter_nav_decision(self, raw_stage, speed_scale, turn_rate, heading_offset, turn_sign):
        return self.avoidance.filter_nav_decision(
            raw_stage, speed_scale, turn_rate, heading_offset, turn_sign)

    def _choose_turn_sign(self, left_score, right_score):
        return self.avoidance.choose_turn_sign(left_score, right_score)

    def _compute_obstacle_override(
            self, gray, horizon_y, tracked_pts=None,
            R_body_world=None, yaw_world=None, pos_world=None, image_stamp=0.0):
        return self.avoidance.compute_obstacle_override(
            gray, horizon_y, tracked_pts,
            R_body_world, yaw_world, pos_world, image_stamp)

    def _process_image(
            self, msg, odom_fresh=True, image_stamp=0.0,
            R_sync=None, yaw_sync=None, pos_sync=None):
        self.frame_stats['frames_total'] += 1
        if odom_fresh:
            self.frame_stats['odom_fresh'] += 1
        in_impact = self.is_impact_window()
        try:
            frame_gray = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='mono8')
        except Exception as e:
            self.get_logger().error(f'Image convert failed: {e}')
            return

        R_now = R_sync.copy() if R_sync is not None else self.current_R.copy()
        current_yaw = float(self.current_yaw if yaw_sync is None else yaw_sync)
        img_h, img_w = frame_gray.shape
        horizon_y = int(img_h * self.horizon_fraction)

        debug_img = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        cv2.line(debug_img, (0, horizon_y), (img_w, horizon_y), (0, 255, 255), 1)

        if self.prev_gray is None:
            self.prev_gray = frame_gray
            self.prev_pts = self.frontend.detect_features(frame_gray, horizon_y)
            self.prev_R = R_now
            cv2.imshow('Visual Odometry Feed', debug_img)
            cv2.waitKey(1)
            return

        vo_published = False
        n_tracked = 0
        n_inliers = 0
        n_rot = 0
        quality_score = 0.0
        yaw_variance = self.frontend_yaw_var_ceiling
        tracked_for_nav = None
        reason = 'waiting'

        if self.prev_pts is not None and len(self.prev_pts) > 0:
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, self.prev_pts, None)

            if curr_pts is None or status is None:
                good_new = np.empty((0, 2), dtype=np.float32)
                good_old = np.empty((0, 2), dtype=np.float32)
            else:
                good_new = curr_pts[status == 1]
                good_old = self.prev_pts[status == 1]
            n_tracked = len(good_new)
            tracked_for_nav = good_new

            if len(good_new) >= self.min_vo_inliers:
                self.frame_stats['candidate_frames'] += 1
                if in_impact:
                    self.frame_stats['candidate_impact'] += 1
                else:
                    self.frame_stats['candidate_nonimpact'] += 1
                upper_mask = (
                    (good_old[:, 1] < horizon_y) &
                    (good_new[:, 1] < horizon_y)
                )
                if int(np.count_nonzero(upper_mask)) >= self.min_rot_features:
                    rot_old = good_old[upper_mask]
                    rot_new = good_new[upper_mask]
                    reason = 'upper'
                    used_upper = True
                else:
                    rot_old = good_old
                    rot_new = good_new
                    reason = 'all'
                    used_upper = False
                n_rot = len(rot_new)

                R_delta_body = self.prev_R.T @ R_now
                good_old_derotated = self.frontend.derotate_points(
                    rot_old.reshape(-1, 1, 2), R_delta_body)

                yaw_rel, n_inliers = self.frontend.compute_relative_rotation(
                    good_old_derotated, rot_new)
                flow_yaw = self.frontend.compute_flow_yaw(good_old_derotated, rot_new)
                used_fallback = False

                if yaw_rel is None and flow_yaw is not None and abs(flow_yaw) <= self.flow_fallback_limit:
                    yaw_rel = flow_yaw
                    n_inliers = n_rot
                    self.frame_stats['flow_fallback'] += 1
                    reason = f'flow_small:{reason}'
                    used_fallback = True
                elif yaw_rel is not None and flow_yaw is not None:
                    if abs(yaw_rel - flow_yaw) <= self.flow_agreement_limit:
                        reason = f'agree:{reason}'
                    else:
                        reason = f'emat_only:{reason}'

                if yaw_rel is not None:
                    quality_score = self.frontend.compute_quality_score(
                        n_inliers, n_rot, yaw_rel, flow_yaw, used_fallback, used_upper)
                    yaw_variance = self.frontend.quality_to_yaw_variance(quality_score)

                if (yaw_rel is not None and n_inliers >= self.min_publish_inliers and
                        self.vo_initialized and odom_fresh):
                    self.visual_yaw = self._wrap_angle(
                        current_yaw + yaw_rel)
                    self._publish_visual_heading(
                        msg.header.stamp, quality_score, yaw_variance)
                    self.frame_stats['published'] += 1
                    self.frame_stats['quality_sum'] += quality_score
                    self.frame_stats['yaw_var_sum'] += yaw_variance
                    if in_impact:
                        self.frame_stats['published_impact'] += 1
                    else:
                        self.frame_stats['published_nonimpact'] += 1
                    vo_published = True
                    reason = f'fused:{reason}'
                elif yaw_rel is None:
                    self.frame_stats['pose_fail'] += 1
                    reason = f'pose_fail:{reason}'
                elif n_inliers < self.min_publish_inliers:
                    self.frame_stats['low_inliers'] += 1
                    reason = f'low_inliers:{n_inliers}'
                elif not self.vo_initialized:
                    reason = 'vo_init_wait'
                elif not odom_fresh:
                    self.frame_stats['stale_odom'] += 1
                    reason = 'stale_odom'

                for pt in good_new:
                    colour = (0, 255, 0) if pt[1] < horizon_y else (0, 128, 0)
                    cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, colour, -1)
            else:
                self.frame_stats['low_tracks'] += 1
                reason = f'low_tracks:{n_tracked}'

            # don't redetect every frame; E-matrix gets worse
            if n_tracked < self.redetect_threshold:
                self.prev_pts = self.frontend.detect_features(frame_gray, horizon_y)
            else:
                self.prev_pts = good_new.reshape(-1, 1, 2)
        else:
            self.prev_pts = self.frontend.detect_features(frame_gray, horizon_y)

        nav_stage = 1
        nav_speed_scale = 1.0
        nav_turn = 0.0
        nav_heading_offset = 0.0
        nav_scores = self.obstacle_scores.copy()
        nav_label = 'off'
        nav_obstacle_count = len(self.tracked_obstacles_world)
        if self.enable_reactive_avoidance:
            nav_stage, nav_speed_scale, nav_turn, nav_heading_offset, nav_scores, nav_obstacle_count = self.avoidance.compute_obstacle_override(
                frame_gray, horizon_y, tracked_for_nav,
                R_now, current_yaw,
                self.current_pos if pos_sync is None else pos_sync,
                image_stamp,
            )
            self._publish_nav_override(
                nav_stage, nav_speed_scale, nav_turn, nav_heading_offset)
            self._publish_tracked_obstacles(image_stamp)
            nav_label = str(nav_stage)

        if not odom_fresh:
            status_str = 'no odom'
        elif vo_published:
            status_str = 'FUSED'
        else:
            status_str = 'skip'

        cv2.putText(
            debug_img,
            f'{status_str}  nav={nav_label}  obs={nav_obstacle_count}  sync={self._last_sync_mode}',
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 200, 255), 2)
        if self.enable_reactive_avoidance:
            cv2.putText(
                debug_img,
                f'center={nav_scores["center"]:.2f}  near={nav_scores["near_center"]:.2f}',
                (20, 64), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 200, 255), 2)

        cv2.imshow('Visual Odometry Feed', debug_img)
        cv2.waitKey(1)

        self.prev_gray = frame_gray
        self.prev_R = R_now

    def _detect_features(self, gray, horizon_y):
        return self.frontend.detect_features(gray, horizon_y)

    def destroy_node(self):
        ft = self.frame_stats['frames_total']
        cf = self.frame_stats['candidate_frames']
        pub = self.frame_stats['published']
        if ft > 0:
            self.get_logger().info(
                f'VO: frames={ft}, odom={self.frame_stats["odom_fresh"]}/{ft}, '
                f'cand={cf}, pub={pub}, '
                f'q={(self.frame_stats["quality_sum"] / pub) if pub > 0 else 0.0:.2f}'
            )
            self.get_logger().info(
                'skips: '
                f'pose_fail={self.frame_stats["pose_fail"]} | '
                f'low_inliers={self.frame_stats["low_inliers"]} | '
                f'low_tracks={self.frame_stats["low_tracks"]} | '
                f'stale_odom={self.frame_stats["stale_odom"]} | '
                f'flow_fallback={self.frame_stats["flow_fallback"]}'
            )
        if cf > 0:
            self.get_logger().info(
                f'impact: cand={self.frame_stats["candidate_impact"]}/{cf}, '
                f'pub={self.frame_stats["published_impact"]}/{pub if pub > 0 else 1}'
            )
        if ft > 0 and self.enable_reactive_avoidance:
            self.get_logger().info(
                f'nav: clear={self.frame_stats["nav_clear"]}, '
                f'avoid={self.frame_stats["nav_avoid"]}, '
                f'stop={self.frame_stats["nav_stop"]}, '
                f'center={self.frame_stats["nav_center_sum"] / ft:.2f}, '
                f'near={self.frame_stats["nav_near_sum"] / ft:.2f}'
            )
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

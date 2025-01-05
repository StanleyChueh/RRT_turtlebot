import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq
from scipy.interpolate import CubicSpline
import tf2_ros
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException
import math
from scipy.interpolate import make_interp_spline
from nav_msgs.msg import OccupancyGrid
import time

class HybridPlannerNode(Node):
    def __init__(self):
        super().__init__('hybrid_planner')

        # ROS2 Subscribers and Publishers
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Publishers for visualization
        self.global_path_pub = self.create_publisher(Marker, 'a_star_path', 10)
        self.dwa_candidates_pub = self.create_publisher(MarkerArray, 'dwa_candidates', 10)

        self.costmap_pub = self.create_publisher(MarkerArray, 'costmap', 10)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)

        self.waypoint_pub = self.create_publisher(Marker, 'waypoint_marker', 10)

        # TF Listener
        self.tf_buffer = Buffer(cache_time=rclpy.time.Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.grid_resolution = 0.2  # Resolution of the grid
        self.goal_threshold = 0.05  # Minimum distance to consider the goal reached (in meters)
        self.robot_radius = 0.12  # Robot radius
        self.map_bounds = (-15, 15, -15, 15)  # Min x, max x, min y, max y
        self.local_obstacles = []
        self.current_goal = None
        self.path = []
        self.followed_path = []  # Store the robot's followed path

        self.lookahead_distance = 0.2  # Distance to look ahead on the path
        self.max_linear_speed = 0.15   # Maximum linear speed
        self.max_angular_speed = 0.8  # Maximum angular speed
        
        self.previous_angle_error = 0.0
        self.replan_cooldown = 3.0  # Minimum time between replanning (in seconds)
        self.last_replan_time = self.get_clock().now()  # Initialize last replan time

        # Parameters for Costmap
        self.costmap_resolution = 0.05  # 5 cm grid cells
        self.costmap_size = 200  # 100 x 100 grid cells (e.g., 5x5 meters if resolution is 0.05)
        self.costmap = np.zeros((self.costmap_size, self.costmap_size), dtype=np.float32)  # Costmap grid
        self.costmap_timer = self.create_timer(0.5, self.visualize_costmap)  # Publish every 0.5 seconds

        self.timer = self.create_timer(0.1, self.control_loop)
        self.static_map = None

    def map_callback(self, msg):
        """
        Update the static map layer from the `/map` topic and verify its completeness.
        """
        try:
            self.static_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            if -1 in self.static_map:  # Check for incomplete regions
                self.get_logger().warn("Static map is incomplete. Relying on dynamic obstacles from /scan.")
                self.map_incomplete = True
            else:
                self.get_logger().info("Static map fully built and updated.")
                self.map_incomplete = False

            # Normalize map values: treat unknown cells (-1) as free space (0), and obstacles (>0) as 1
            self.static_map[self.static_map == -1] = 0
            self.static_map[self.static_map > 0] = 1

        except Exception as e:
            self.get_logger().error(f"Error processing static map: {e}")
            self.map_incomplete = True


    def update_costmap(self, laser_scan, robot_position, robot_yaw):
        """
        Update the costmap dynamically using /scan data and merge with the static map (if available and complete).
        """
        # Reset the costmap
        self.costmap.fill(0)

        # Process /scan data for dynamic obstacles
        angle = laser_scan.angle_min
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)

        for r in laser_scan.ranges:
            if 0.1 < r < laser_scan.range_max:
                # Transform obstacle to the global map frame
                x_robot = r * math.cos(angle)
                y_robot = r * math.sin(angle)
                x_map = x_robot * cos_yaw - y_robot * sin_yaw + robot_position[0]
                y_map = x_robot * sin_yaw + y_robot * cos_yaw + robot_position[1]

                # Map to costmap grid coordinates
                map_x = int((x_map + self.costmap_size * self.costmap_resolution / 2) / self.costmap_resolution)
                map_y = int((y_map + self.costmap_size * self.costmap_resolution / 2) / self.costmap_resolution)

                if 0 <= map_x < self.costmap_size and 0 <= map_y < self.costmap_size:
                    self.costmap[map_x, map_y] = 1.0  # Mark obstacle
            angle += laser_scan.angle_increment

        # Merge with the static map if available and complete
        if self.static_map is not None and not self.map_incomplete:
            self.get_logger().info("Merging static map with /scan data.")
            try:
                static_map_resized = self.resize_static_map()
                self.costmap += static_map_resized
                self.costmap[self.costmap > 1] = 1  # Ensure binary values
            except Exception as e:
                self.get_logger().error(f"Error merging static map: {e}")
        else:
            self.get_logger().warn("Static map is incomplete or unavailable. Using only /scan data.")
        


    def resize_static_map(self):
        # Must return a (100,100) array to match self.costmap
        # For instance, if self.static_map is 106x106, you could crop or downsample:
        # Example: Crop center to 100x100 if you want:
        static_h, static_w = self.static_map.shape
        if static_h == 100 and static_w == 100:
            return self.static_map
        if static_h >= 100 and static_w >= 100:
            # Crop the center 100x100 region
            start_x = (static_w - 100) // 2
            start_y = (static_h - 100) // 2
            return self.static_map[start_y:start_y+100, start_x:start_x+100]


    def lookup_transform_with_retry(self, target_frame, source_frame, timeout=1.0, retries=3):
        for attempt in range(retries):
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time(),  # Use the latest available transform
                    timeout=rclpy.time.Duration(seconds=timeout)
                )
                return transform
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(f"TF Extrapolation Error (Attempt {attempt + 1}/{retries}): {e}")
                time.sleep(0.1)  # Brief delay before retrying
            except tf2_ros.LookupException as e:
                self.get_logger().warn(f"TF Lookup Error (Attempt {attempt + 1}/{retries}): {e}")
            except tf2_ros.ConnectivityException as e:
                self.get_logger().warn(f"TF Connectivity Error (Attempt {attempt + 1}/{retries}): {e}")
                time.sleep(0.1)  # Brief delay before retrying
        
        # Log error if all retries fail
        self.get_logger().error(f"Failed to lookup transform from {source_frame} to {target_frame} after {retries} retries.")
        return None


    def get_current_position(self):
        for attempt in range(5):  # Retry up to 5 times
            try:
                transform = self.tf_buffer.lookup_transform(
                    "map",
                    "base_link",
                    rclpy.time.Time(seconds=0),  # Use latest transform
                    timeout=rclpy.time.Duration(seconds=1.0)
                )
                position = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y
                ])
                return position
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().warn(f"TF Extrapolation Error (attempt {attempt + 1}): {e}")
                time.sleep(0.1)  # Brief delay before retrying
            except tf2_ros.LookupException as e:
                self.get_logger().warn(f"TF Lookup Error (attempt {attempt + 1}): {e}")
        self.get_logger().error("Failed to get current position after retries.")
        return None



    def goal_callback(self, msg):
        self.get_logger().info(f"Received goal: {msg.pose.position}")
        self.current_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.plan_global_path()

    def scan_callback(self, msg):
        """
        Process LaserScan data to overlay dynamic obstacles onto the costmap.
        Use `/map` data as the static layer and `/scan` data for dynamic obstacles.
        """
        self.local_obstacles = []
        self.latest_laserscan = msg  
        if self.static_map is None:
            self.get_logger().warn("Static map not available. Skipping scan overlay.")
            return

        try:
            # Transform obstacles to the map frame
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
                timeout=rclpy.time.Duration(seconds=0.5)
            )
            translation = np.array([transform.transform.translation.x, transform.transform.translation.y])
            yaw = self.get_yaw_from_transform(transform)
            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])

            # Overlay dynamic obstacles
            angle = msg.angle_min
            for r in msg.ranges:
                if 0.1 < r < msg.range_max:
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    obstacle_in_base_link = np.array([x, y])
                    obstacle_in_map = np.dot(rotation_matrix, obstacle_in_base_link) + translation
                    self.local_obstacles.append(obstacle_in_map)

                    grid_x = int((obstacle_in_map[0] + self.costmap_size * self.costmap_resolution / 2) / self.costmap_resolution)
                    grid_y = int((obstacle_in_map[1] + self.costmap_size * self.costmap_resolution / 2) / self.costmap_resolution)

                    if 0 <= grid_x < self.costmap_size and 0 <= grid_y < self.costmap_size:
                        self.costmap[grid_x, grid_y] = 1.0

                angle += msg.angle_increment

        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"TF Lookup Error (scan_callback): {e}")
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"TF Extrapolation Error (scan_callback): {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in scan_callback: {e}")


    def world_to_costmap_coords(self, x, y):
        grid_x = int((x + self.costmap_size * self.costmap_resolution / 2) / self.costmap_resolution)
        grid_y = int((y + self.costmap_size * self.costmap_resolution / 2) / self.costmap_resolution)
        return grid_x, grid_y


    def hybrid_a_star(self, start, goal):
        """
        Modified A* algorithm to avoid obstacles using the costmap dynamically.
        """
        start = tuple(start)
        goal = tuple(goal)

        def heuristic(a, b):
            """
            Calculate heuristic considering Euclidean distance and obstacle penalties.
            """
            euclidean_cost = np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))
            grid_x, grid_y = self.world_to_costmap_coords(a[0], a[1])
            obstacle_penalty = (
                self.costmap[grid_x, grid_y]
                if 0 <= grid_x < self.costmap_size and 0 <= grid_y < self.costmap_size
                else 1.0
            )
            return euclidean_cost + obstacle_penalty * 15.0  # Adjust penalty weight if needed

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            # Goal check
            if np.linalg.norm(np.array(current) - np.array(goal)) < self.grid_resolution:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Return path from start to goal

            # Process neighbors
            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))

                # Add obstacle penalty from the costmap
                grid_x, grid_y = self.world_to_costmap_coords(neighbor[0], neighbor[1])
                if 0 <= grid_x < self.costmap_size and 0 <= grid_y < self.costmap_size:
                    tentative_g_score += self.costmap[grid_x, grid_y] * 15.0  # Adjust penalty scaling

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)

                    # Add neighbor to the open set
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        self.get_logger().warn("Hybrid A* failed to find a path.")
        return []

    
    def estimate_curvature(self, current_position):
        if len(self.path) < 3:
            return 0  # No curvature if there are fewer than 3 waypoints
        p1 = np.array(current_position)
        p2 = np.array(self.path[0])
        p3 = np.array(self.path[1])

        # Calculate the curvature using the circumcircle formula
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)
        if a * b * c == 0:
            return 0  # Avoid division by zero
        curvature = 4 * np.abs(np.cross(p2 - p1, p3 - p1)) / (a * b * c)
        return curvature

    def is_path_valid(self, path):
        for point in path:
            grid_x, grid_y = self.world_to_costmap_coords(point[0], point[1])
            if self.costmap[grid_x, grid_y] >= 0.5:  # Check for obstacles
                return False
        return True


    def is_path_blocked(self, threshold=3):
        if not self.path:
            return False
        blocked_count = 0
        for waypoint in self.path[:5]:  # Check the first few waypoints
            for obstacle in self.local_obstacles:
                distance = np.linalg.norm(np.array(waypoint) - np.array(obstacle))
                if distance < self.robot_radius + 0.1:  # Include safety margin
                    blocked_count += 1
        return blocked_count >= threshold

    
    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(-self.grid_resolution, 0), (self.grid_resolution, 0), (0, -self.grid_resolution), (0, self.grid_resolution)]:
            neighbor = (node[0] + dx, node[1] + dy)
            grid_x, grid_y = self.world_to_costmap_coords(neighbor[0], neighbor[1])

            if 0 <= grid_x < self.costmap_size and 0 <= grid_y < self.costmap_size:
                penalty = self.costmap[grid_x, grid_y]
                if penalty < 0.5:  # Allow only nodes that are not obstacles
                    neighbors.append(neighbor)

        return neighbors

    def smooth_trajectory(self, trajectory):
        if len(trajectory) < 3:
            return trajectory

        x = [point[0] for point in trajectory]
        y = [point[1] for point in trajectory]
        t = np.linspace(0, len(trajectory) - 1, len(trajectory))
        smooth_t = np.linspace(0, len(trajectory) - 1, 100)

        try:
            spl_x = make_interp_spline(t, x, k=3)
            spl_y = make_interp_spline(t, y, k=3)
            smoothed_trajectory = list(zip(spl_x(smooth_t), spl_y(smooth_t)))
            return smoothed_trajectory
        except Exception as e:
            self.get_logger().error(f"Error smoothing trajectory: {e}")
            return trajectory


    def smooth_path(self, path):
        """Smooth the planned path using cubic splines."""
        if len(path) < 4:
            self.get_logger().warn("Not enough points for cubic spline smoothing. Returning the original path.")
            return path  # Return the original path if not enough points

        try:
            x = [point[0] for point in path]
            y = [point[1] for point in path]
            t = np.linspace(0, len(path) - 1, len(path))

            spl_x = make_interp_spline(t, x, k=3)
            spl_y = make_interp_spline(t, y, k=3)

            smooth_t = np.linspace(0, len(path) - 1, 100)
            smoothed_path = list(zip(spl_x(smooth_t), spl_y(smooth_t)))
            return smoothed_path
        except Exception as e:
            self.get_logger().error(f"Error smoothing path: {e}")
            return path  # Return the original path in case of errors

    def plan_global_path(self):
        """
        Plan a global path using both static and dynamic data. Fallback to /scan if static map is incomplete.
        """
        start = self.get_current_position()
        goal = self.current_goal

        if start is None or goal is None:
            self.get_logger().warn("Start or goal position not available.")
            return

        self.get_logger().info(f"Planning path from start: {start} to goal: {goal}")

        expansion_factor = 1.5  # Factor to expand map bounds
        resolution_decrement = 0.05  # Decrement resolution with each attempt

        original_bounds = self.map_bounds
        original_resolution = self.grid_resolution

        attempt = 0
        max_attempts = 10  # Maximum replanning attempts

        while attempt < max_attempts:
            self.get_logger().info(f"Replanning attempt {attempt + 1}...")

            # Always prioritize /scan data
            try:
                current_position = self.get_current_position()
                current_yaw = self.get_current_yaw()

                if current_position is not None and current_yaw is not None and hasattr(self, "latest_laserscan"):
                    self.get_logger().info("Updating costmap using /scan data.")
                    self.update_costmap(self.latest_laserscan, current_position, current_yaw)
                    self.inflate_costmap()

                    # If static map is available, merge it with the dynamic costmap
                    if self.static_map is not None and -1 not in self.static_map:
                        self.get_logger().info("Merging static map with dynamic costmap.")
                        self.costmap += self.static_map
                        self.costmap[self.costmap > 1] = 1  # Ensure binary values
                    else:
                        self.get_logger().warn("Static map incomplete or unavailable. Using only /scan data.")
                else:
                    self.get_logger().warn("Insufficient /scan data for costmap update.")
            except Exception as e:
                self.get_logger().error(f"Error updating costmap during replanning: {e}")

            # Adjust map bounds and resolution for expanded searches
            self.map_bounds = (
                original_bounds[0] * (expansion_factor ** attempt),
                original_bounds[1] * (expansion_factor ** attempt),
                original_bounds[2] * (expansion_factor ** attempt),
                original_bounds[3] * (expansion_factor ** attempt),
            )
            self.grid_resolution = max(0.05, original_resolution - attempt * resolution_decrement)

            # Plan the path using the updated costmap
            path = self.hybrid_a_star(start, goal)

            # Validate the path to ensure it's obstacle-free
            if path and self.is_path_valid(path):
                smoothed_path = self.smooth_path(path)
                self.path = self.segment_path(smoothed_path, segment_distance=0.1)
                self.get_logger().info(f"Global path found with {len(self.path)} waypoints.")
                self.visualize_path(self.path)

                # Reset map bounds and resolution
                self.map_bounds = original_bounds
                self.grid_resolution = original_resolution
                return
            else:
                self.get_logger().warn("Planned path is invalid. Retrying with expanded search bounds...")

            attempt += 1

        # If no valid path is found after max attempts, log a failure
        self.get_logger().warn("Failed to find a valid global path after multiple attempts.")
        self.map_bounds = original_bounds  # Restore original bounds
        self.grid_resolution = original_resolution  # Restore original resolution

    def generate_dwa_trajectories(self, current_position, current_yaw, v_range, w_range, dt, horizon):
        trajectories = []
        for v in np.linspace(v_range[0], v_range[1], 15):  # Increase resolution to 30 steps
            for w in np.linspace(w_range[0], w_range[1], 15):  # Increase angular velocity resolution
                trajectory = self.simulate_trajectory(current_position, current_yaw, v, w, dt, horizon)
                if trajectory:
                    trajectories.append((trajectory, v, w))
                else:
                    self.get_logger().debug(f"Invalid trajectory for v={v}, w={w}")
        self.get_logger().info(f"Generated {len(trajectories)} DWA trajectories.")
        return trajectories


    def evaluate_dwa_trajectory(self, trajectory, waypoint, v, w):
        """
        Evaluate a trajectory’s desirability based on:
        1) Distance to waypoint (the closer, the better).
        2) Collision check (any collision -> invalid).
        3) Heading alignment (smaller heading error -> better).
        """
        if not trajectory or waypoint is None:
            return -float('inf')
        
        total_distance_to_waypoint = 0.0
        min_distance_to_obstacle = float('inf')
        
        # Let's measure distance to the waypoint over the entire trajectory
        for point in trajectory:
            dist_to_wp = np.linalg.norm(np.array(point) - np.array(waypoint))
            total_distance_to_waypoint += dist_to_wp
            
            # Check collision
            obs_dist = min(np.linalg.norm(np.array(point) - obs) 
                        for obs in self.local_obstacles) if self.local_obstacles else float('inf')
            min_distance_to_obstacle = min(min_distance_to_obstacle, obs_dist)
            if obs_dist < (self.robot_radius + 0.05):
                # Hard collision or very close to an obstacle
                return -float('inf')
        
        # Average distance to the waypoint (lower is better)
        avg_distance_to_waypoint = total_distance_to_waypoint / len(trajectory)

        # A rough heading cost: compare direction of final segment to direction
        # from final trajectory point to the waypoint
        final_point = np.array(trajectory[-1])
        final_dir = final_point - np.array(trajectory[0])
        wp_dir = np.array(waypoint) - final_point
        final_heading = np.arctan2(final_dir[1], final_dir[0]) if np.linalg.norm(final_dir) > 1e-3 else 0.0
        waypoint_heading = np.arctan2(wp_dir[1], wp_dir[0]) if np.linalg.norm(wp_dir) > 1e-3 else 0.0
        heading_error = abs(self.normalize_angle(waypoint_heading - final_heading))

        # Construct a cost. Higher is better, so we turn "distance" into a negative factor.
        # Example weighting: penalize average distance to the waypoint, penalize heading error,
        # and penalize being close to obstacles.
        # The smaller avg_distance_to_waypoint is, the more positive we want the cost contribution to be.
        # So we can do something like: cost = big_constant - alpha*(avg_distance) - beta*(heading_error) - gamma*(1/min_obstacle_dist).
        
        cost = 0.0
        # Reward closeness to waypoint
        cost += 10.0 / (avg_distance_to_waypoint + 1e-3) # bigger if avg_distance is small
        
        # Penalize heading error
        cost -= 0.5 * heading_error
        
        # Penalize if min_distance_to_obstacle is small
        # (But not as harsh as total invalidation)
        cost -= 1.0 / (min_distance_to_obstacle + 1e-3)

        cost += 5.0 * v  # or 1.0 * v, tune as needed

        return cost


    def inflate_costmap(self, inflation_radius=0.1):
        inflated_costmap = np.copy(self.costmap)
        radius_in_cells = int(inflation_radius / self.costmap_resolution)

        for x in range(self.costmap_size):
            for y in range(self.costmap_size):
                if self.costmap[x, y] > 0:  # If it's an obstacle
                    for dx in range(-radius_in_cells, radius_in_cells + 1):
                        for dy in range(-radius_in_cells, radius_in_cells + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.costmap_size and 0 <= ny < self.costmap_size:
                                inflated_costmap[nx, ny] = max(inflated_costmap[nx, ny], 1.0)

        self.costmap = inflated_costmap



    def dwa_local_planner(self, current_position, current_yaw):
        self.get_logger().info("dwa_local_planner called.")
        v_range = [0.05, self.max_linear_speed]  # Ensure minimum linear velocity
        w_range = [-self.max_angular_speed, self.max_angular_speed]
        dt = 0.1
        horizon = 2.0

        target_waypoint = self.get_lookahead_point(current_position)
        if target_waypoint is None:
            self.get_logger().warn("No valid waypoint found for DWA.")
            return None

        candidates = self.generate_dwa_trajectories(current_position, current_yaw, v_range, w_range, dt, horizon)
        self.get_logger().info(f"Generated {len(candidates)} DWA candidate trajectories.")

        # Visualize DWA candidates
        self.visualize_dwa_candidates([c[0] for c in candidates if c[0]])  # Extract and visualize only valid trajectories

        best_trajectory = None
        best_score = -float('inf')

        for trajectory, v, w in candidates:
            score = self.evaluate_dwa_trajectory(trajectory, target_waypoint, v, w)
            if score > best_score:
                best_score = score
                best_trajectory = (trajectory, v, w)

        if best_trajectory:
            trajectory, linear_velocity, angular_velocity = best_trajectory
            self.visualize_trajectory(trajectory)
            self.visualize_waypoint(target_waypoint)
            self.get_logger().info(f"Best trajectory selected with score: {best_score}")
            return best_trajectory
        self.get_logger().warn("No valid trajectory found.")
        return None

    def is_collision_free(self, point):
        for obs in self.local_obstacles:
            distance = np.linalg.norm(np.array(point) - np.array(obs))
            if distance < self.robot_radius + 0.15:  # Add safety margin
                self.get_logger().debug(f"Point {point} is too close to obstacle {obs}.")
                return False
        self.get_logger().debug(f"Point {point} is collision-free.")
        return True

    
    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-pi, pi].
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    def motion_primitives(state):
        """Generate possible motions (arcs and straight lines) from a given state."""
        motions = []
        for steering_angle in np.linspace(-np.pi / 4, np.pi / 4, 5):  # Discretize steering angles
            for distance in [0.1, 0.2, 0.3]:  # Discretize distances
                theta = state[2] + steering_angle
                x = state[0] + distance * np.cos(theta)
                y = state[1] + distance * np.sin(theta)
                motions.append((x, y, theta))
        return motions

    def generate_motion_primitives(state):
        """Generate feasible motion primitives based on the robot's kinematics."""
        motions = []
        for steering_angle in np.linspace(-np.pi / 6, np.pi / 6, 5):  # Steering angle range
            for distance in [0.2, 0.4, 0.6]:  # Discretize distances
                theta = state[2] + steering_angle
                x = state[0] + distance * np.cos(theta)
                y = state[1] + distance * np.sin(theta)
                motions.append((x, y, theta))
        return motions
    
    def segment_path(self, path, segment_distance=0.5):
        waypoints = [path[0]]  # Start with the first point
        for point in path:
            if np.linalg.norm(np.array(point) - np.array(waypoints[-1])) >= segment_distance:
                waypoints.append(point)
        waypoints.append(path[-1])  # Ensure the goal is included
        return waypoints

    def control_loop(self):
        # Stop if no path is available
        if not self.path:
            self.get_logger().warn("No path available. Stopping the robot.")
            self.cmd_vel_pub.publish(Twist())  # Publish zero velocity
            return

        # Get the current position and yaw
        current_position = self.get_current_position()
        if current_position is None:
            self.get_logger().warn("Current position not available.")
            return

        yaw = self.get_current_yaw()
        if yaw is None:
            self.get_logger().warn("Yaw not available.")
            return
        
        # Check if the current path is blocked
        if self.is_path_blocked():
            self.get_logger().warn("Path is blocked by obstacles. Triggering replanning.")
            self.plan_global_path()
            return

        # Check if the robot is close enough to the goal
        if self.current_goal is not None:
            distance_to_goal = np.linalg.norm(current_position - self.current_goal)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info("Goal reached. Stopping the robot.")
                self.cmd_vel_pub.publish(Twist())  # Publish zero velocity
                self.current_goal = None  # Clear the goal
                self.path = []            # Clear the path
                return

        # ---------------------------------------------------------
        # Instead of calling DWA, pick the next waypoint and run
        # pure pursuit (or a simpler control approach).
        # ---------------------------------------------------------
        next_waypoint = self.path[0]  # The first waypoint in our path
        v, w = self.pure_pursuit_controller(current_position, yaw, next_waypoint)

        # Publish the computed velocity commands
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"Publishing cmd_vel: linear={v:.2f}, angular={w:.2f}")

        # Visualize the current waypoint if needed
        self.visualize_waypoint(self.path[0])

        # Check if the waypoint is reached
        distance_to_waypoint = np.linalg.norm(current_position - np.array(self.path[0]))
        if distance_to_waypoint < self.lookahead_distance:
            self.get_logger().info(f"Reached waypoint: {self.path[0]}")
            self.path.pop(0)  # Remove the waypoint from the path

        # Stop if no more waypoints are left
        if not self.path:
            self.get_logger().info("All waypoints cleared. Stopping the robot.")
            self.cmd_vel_pub.publish(Twist())  # Publish zero velocity


    def pure_pursuit_controller(self, current_position, current_yaw, waypoint):
        """
        Return (v, w) commands that drive from (current_position, current_yaw)
        toward 'waypoint' using a simple pure-pursuit approach.
        """
        # 1) Compute heading to the waypoint
        dx = waypoint[0] - current_position[0]
        dy = waypoint[1] - current_position[1]
        target_yaw = math.atan2(dy, dx)

        # 2) Heading error
        heading_error = self.normalize_angle(target_yaw - current_yaw)

        # 3) Distance to the waypoint
        dist_to_wp = math.sqrt(dx*dx + dy*dy)

        # 4) Gains and max speeds
        K_lin = 0.7   # linear speed gain
        K_ang = 2.0   # angular speed gain
        
        # 5) Decide linear velocity
        # A typical approach: reduce speed if heading error is large
        # or just proportionally to distance
        v = K_lin * dist_to_wp
        # clamp to max linear speed
        v = min(v, self.max_linear_speed)

        # Optionally slow down if heading error is large
        if abs(heading_error) > 0.7:
            v *= 0.3  # drastically reduce speed when angle is large

        # 6) Decide angular velocity
        w = K_ang * heading_error
        # clamp to max angular speed
        w = max(-self.max_angular_speed, min(self.max_angular_speed, w))

        return (v, w)




    def compute_dynamic_lookahead_distance(self, speed):
        min_lookahead = 0.2  # Minimum lookahead distance
        max_lookahead = 1.0  # Maximum lookahead distance
        return min_lookahead + (max_lookahead - min_lookahead) * (speed / self.max_linear_speed)

    def get_lookahead_point(self, current_position):
        if not self.path:
            self.get_logger().warn("Path is empty.")
            return None

        lookahead_distance = self.compute_dynamic_lookahead_distance(self.max_linear_speed)  # Use dynamic lookahead
        for point in self.path:
            distance = np.linalg.norm(np.array(point) - current_position)
            if distance >= lookahead_distance:
                self.get_logger().info(f"Lookahead point selected: {point}")
                return np.array(point)

        # Return the last waypoint if no point meets the lookahead distance
        self.get_logger().warn("No valid lookahead point found. Using the last waypoint.")
        return np.array(self.path[-1])




    def simulate_trajectory(self, position, yaw, v, w, dt, sim_time):
        trajectory = []
        x, y, theta = position[0], position[1], yaw

        for _ in range(int(sim_time / dt)):
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += w * dt
            point = (x, y)

            # Check for collision at each step
            if not self.is_collision_free(np.array([x, y])):
                self.get_logger().debug(f"Collision detected at ({x:.2f}, {y:.2f}). Trajectory invalid.")
                return []  # Return empty trajectory if collision occurs

            trajectory.append(point)

        self.get_logger().debug(f"Generated trajectory with {len(trajectory)} points.")
        return trajectory



    def evaluate_trajectory(self, trajectory):
        score = 0.0

        for point in trajectory:
            path_dist = min(np.linalg.norm(np.array(point) - np.array(waypoint)) for waypoint in self.path)
            obs_dist = min(np.linalg.norm(np.array(point) - np.array(obs)) for obs in self.local_obstacles) if self.local_obstacles else float('inf')

            if obs_dist < self.robot_radius:
                return -float('inf')  # Penalize collision

            score -= path_dist  # Favor closer trajectories to the path
            score += obs_dist  # Reward safer trajectories away from obstacles

        return score




    def get_yaw_from_transform(self, transform):
        q = transform.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def get_current_yaw(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rclpy.time.Time(),
                timeout=rclpy.time.Duration(seconds=0.5)
            )
            return self.get_yaw_from_transform(transform)
        except LookupException as e:
            self.get_logger().warn(f"TF Error (get_current_yaw): {e}")
            return 0.0
    
    def visualize_path(self, path):
        if not path:
            self.get_logger().warn("Cannot visualize an empty path.")
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "a_star_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for point in path:
            p = Point()
            p.x, p.y, p.z = point[0], point[1], 0.0
            marker.points.append(p)

        self.global_path_pub.publish(marker)
        self.get_logger().info(f"Published path with {len(path)} points.")
                                    
    def visualize_dwa_candidates(self, candidates):
        if not candidates or len(candidates) == 0:
            self.get_logger().warn("No candidates provided to visualize.")
            return

        marker_array = MarkerArray()
        marker_id = 0  # Ensure unique ID for each marker
        current_time = self.get_clock().now().to_msg()

        for trajectory in candidates:
            if not trajectory or len(trajectory) < 2:
                self.get_logger().debug(f"Skipping invalid trajectory: {trajectory}")
                continue

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = current_time
            marker.ns = "dwa_candidates"
            marker.id = marker_id
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.03
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.8

            for point in trajectory:
                p = Point()
                p.x = point[0]
                p.y = point[1]
                p.z = 0.0
                marker.points.append(p)

            marker_array.markers.append(marker)
            marker_id += 1

        # Add logging to confirm publication
        self.get_logger().info(f"Publishing {len(marker_array.markers)} markers to /dwa_candidates.")
        self.dwa_candidates_pub.publish(marker_array)
        self.get_logger().info("Published MarkerArray to /dwa_candidates.")



    def visualize_trajectory(self, trajectory):
        if not trajectory:
            return

        # Create a MarkerArray
        marker_array = MarkerArray()

        # Create a single Marker for the trajectory
        marker = Marker()
        marker.header.frame_id = "map"  # Ensure it matches your RViz setup
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "selected_trajectory"
        marker.id = 0  # Unique ID within the namespace
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0
        marker.color.b = 1.0  # Blue
        marker.color.a = 1.0  # Fully opaque

        # Add points to the marker
        for point in trajectory:
            p = Point()
            p.x, p.y, p.z = point[0], point[1], 0.0  # Flat 2D plane
            marker.points.append(p)

        # Add the Marker to the MarkerArray
        marker_array.markers.append(marker)

        # Publish the MarkerArray
        self.dwa_candidates_pub.publish(marker_array)
        self.get_logger().info("Published selected trajectory as MarkerArray.")



    def visualize_costmap(self):
        marker_array = MarkerArray()
        marker_id = 0

        for x in range(self.costmap_size):
            for y in range(self.costmap_size):
                if self.costmap[x, y] > 0:  # Only visualize obstacles
                    marker = Marker()
                    marker.header.frame_id = "map"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "costmap"
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.scale.x = self.costmap_resolution
                    marker.scale.y = self.costmap_resolution
                    marker.scale.z = 0.05
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.5

                    world_x = (x * self.costmap_resolution) - (self.costmap_size * self.costmap_resolution / 2)
                    world_y = (y * self.costmap_resolution) - (self.costmap_size * self.costmap_resolution / 2)
                    marker.pose.position.x = world_x
                    marker.pose.position.y = world_y
                    marker.pose.position.z = 0.0
                    marker.pose.orientation.w = 1.0

                    marker_array.markers.append(marker)
                    marker_id += 1

        self.costmap_pub.publish(marker_array)
        self.get_logger().info(f"Published costmap with {marker_id} markers.")

    def visualize_waypoint(self, waypoint):
        """
        Visualize the current waypoint the robot is targeting in RViz.
        """
        if waypoint is None:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "waypoint"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Size of the sphere
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 1.0  # Green color for the waypoint
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque
        marker.pose.position.x = waypoint[0]
        marker.pose.position.y = waypoint[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0

        # Publish the marker to the waypoint topic
        self.waypoint_pub.publish(marker)
        self.get_logger().info(f"Visualized waypoint at: {waypoint}.")

    def visualize_selected_trajectory(self, trajectory):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "selected_trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        for point in trajectory:
            p = Point()
            p.x, p.y, p.z = point[0], point[1], 0.0
            marker.points.append(p)
        self.global_path_pub.publish(marker)



def main(args=None):
    rclpy.init(args=args)
    node = HybridPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

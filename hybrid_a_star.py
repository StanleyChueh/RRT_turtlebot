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

class HybridPlannerNode(Node):
    def __init__(self):
        super().__init__('hybrid_planner')

        # ROS2 Subscribers and Publishers
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_marker_pub = self.create_publisher(Marker, 'a_star_path', 10)

        # TF Listener
        self.tf_buffer = Buffer(cache_time=rclpy.time.Duration(seconds=10.0))
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
        self.max_angular_speed = 1.0  # Maximum angular speed
        
        self.previous_angle_error = 0.0
        self.replan_cooldown = 3.0  # Minimum time between replanning (in seconds)
        self.last_replan_time = self.get_clock().now()  # Initialize last replan time

        self.timer = self.create_timer(0.1, self.control_loop)

    def lookup_transform_with_retry(self, target_frame, source_frame, time, timeout, retries=3):
        for attempt in range(retries):
            try:
                return self.tf_buffer.lookup_transform(target_frame, source_frame, time, timeout)
            except (LookupException, ExtrapolationException) as e:
                self.get_logger().warn(f"TF Lookup Error (Attempt {attempt + 1}/{retries}): {e}")
        return None

    def get_current_position(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_footprint",
                rclpy.time.Time(),  # Use the latest available transform
                timeout=rclpy.time.Duration(seconds=0.5)
            )
            position = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            return position
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"TF Extrapolation Error (get_current_position): {e}")
            return None  # Fallback to avoid breaking the control loop

    def goal_callback(self, msg):
        self.get_logger().info(f"Received goal: {msg.pose.position}")
        self.current_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.plan_global_path()

    def scan_callback(self, msg):
        # Initialize local obstacles list
        self.local_obstacles = []
        angle = msg.angle_min  # Start angle of the scan

        try:
            # Use the latest available transform
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_footprint",
                rclpy.time.Time(),
                timeout=rclpy.time.Duration(seconds=0.5)
            )

            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y
            ])
            yaw = self.get_yaw_from_transform(transform)

            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])

            # Process each laser scan range
            for r in msg.ranges:
                if 0.1 < r < msg.range_max:
                    # Calculate obstacle position in the base_link frame
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    obstacle_in_base_link = np.array([x, y])

                    # Transform obstacle position to the map frame
                    obstacle_in_map = np.dot(rotation_matrix, obstacle_in_base_link) + translation
                    self.local_obstacles.append(obstacle_in_map)

                # Increment the angle for the next scan range
                angle += msg.angle_increment

            # Visualize the inflation zone around obstacles
            self.visualize_inflation_zone()

        except tf2_ros.LookupException as e:
            self.get_logger().warn(f"TF Lookup Error (scan_callback): {e}")
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().warn(f"TF Extrapolation Error (scan_callback): {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in scan_callback: {e}")

    def hybrid_a_star(self, start, goal):
        # Convert start and goal to tuples
        start = tuple(start)
        goal = tuple(goal)

        def heuristic(a, b):
            return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if heuristic(current, goal) < self.grid_resolution:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Generate neighbors (ensure neighbors are tuples)
            for neighbor in self.get_neighbors(current):
                neighbor = tuple(neighbor)  # Convert to tuple
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        self.get_logger().warn("A* failed to find a path.")
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

    def is_path_blocked(self):
        """
        Check if any obstacle intersects the planned path.
        """
        if not self.path:
            return False  # No path to block

        for waypoint in self.path[:5]:  # Check the next 5 waypoints or adjust as needed
            for obstacle in self.local_obstacles:
                distance = np.linalg.norm(np.array(waypoint) - np.array(obstacle))
                if distance < self.robot_radius + 0.1:  # Include a safety margin
                    return True
        return False

    
    def get_neighbors(self, node):
        neighbors = []
        for dx, dy in [(-self.grid_resolution, 0), (self.grid_resolution, 0), (0, -self.grid_resolution), (0, self.grid_resolution)]:
            neighbor = (node[0] + dx, node[1] + dy)
            if self.is_collision_free(neighbor):
                neighbors.append(neighbor)
        return neighbors


    def smooth_path(self, path):
        """Smooth the planned path using cubic splines."""
        if len(path) < 3:
            return path  # No need to smooth if the path has fewer than 3 points.

        x = [point[0] for point in path]
        y = [point[1] for point in path]
        t = np.linspace(0, len(path) - 1, len(path))

        spl_x = make_interp_spline(t, x, k=3)
        spl_y = make_interp_spline(t, y, k=3)

        smooth_t = np.linspace(0, len(path) - 1, 100)
        smoothed_path = list(zip(spl_x(smooth_t), spl_y(smooth_t)))
        return smoothed_path


    def plan_global_path(self):
        start = self.get_current_position()
        goal = self.current_goal

        if start is None or goal is None:
            self.get_logger().warn("Start or goal position not available.")
            return

        self.get_logger().info(f"Planning path from start: {start} to goal: {goal}")

        expansion_factor = 1.5  # Factor by which to expand map bounds
        resolution_decrement = 0.05  # Amount to decrease grid resolution in each attempt

        original_bounds = self.map_bounds
        original_resolution = self.grid_resolution

        attempt = 0
        while True:  # Infinite replanning
            self.get_logger().info(f"Replanning attempt {attempt + 1}...")
            self.map_bounds = (
                original_bounds[0] * (expansion_factor ** attempt),
                original_bounds[1] * (expansion_factor ** attempt),
                original_bounds[2] * (expansion_factor ** attempt),
                original_bounds[3] * (expansion_factor ** attempt),
            )
            self.grid_resolution = max(0.05, original_resolution - attempt * resolution_decrement)

            path = self.hybrid_a_star(start, goal)

            if path:
                smoothed_path = self.smooth_path(path)
                self.path = self.segment_path(smoothed_path, segment_distance=0.1)
                self.get_logger().info(f"Global path found with {len(self.path)} waypoints.")
                self.visualize_path(self.path)
                self.map_bounds = original_bounds  # Restore original bounds
                self.grid_resolution = original_resolution  # Restore original resolution
                return

            attempt += 1

            # Optional: Add a termination condition if the robot should stop after some retries
            if attempt >= 10:  # Stop after 10 attempts
                self.get_logger().warn("Failed to find a global path after 10 attempts.")
                self.map_bounds = original_bounds  # Restore original bounds
                self.grid_resolution = original_resolution  # Restore original resolution
                return


    def is_collision_free(self, point):
        for obs in self.local_obstacles:
            distance = np.linalg.norm(np.array(point) - np.array(obs))
            if distance < self.robot_radius + 0.1:  # Add safety margin
                return False  # Point is within collision radius of an obstacle
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
        if not self.path:
            self.get_logger().warn("No path available. Stopping the robot.")
            self.cmd_vel_pub.publish(Twist())  # Stop the robot
            return

        current_position = self.get_current_position()
        if current_position is None:
            self.get_logger().warn("Current position not available.")
            return

        yaw = self.get_current_yaw()
        if yaw is None:
            self.get_logger().warn("Yaw not available.")
            return

        # Check if the robot is close enough to the goal
        if self.current_goal is not None:
            distance_to_goal = np.linalg.norm(current_position - self.current_goal)
            if distance_to_goal < self.goal_threshold:
                self.get_logger().info("Goal reached. Stopping the robot.")
                self.cmd_vel_pub.publish(Twist())  # Stop the robot
                self.current_goal = None  # Clear the goal
                self.path = []  # Clear the path
                return
            
        # Replan if the path is blocked and the cooldown has passed
        if self.is_path_blocked():
            now = self.get_clock().now()
            time_since_last_replan = (now - self.last_replan_time).nanoseconds * 1e-9
            if time_since_last_replan >= self.replan_cooldown:
                self.get_logger().info("Path blocked! Replanning...")
                self.plan_global_path()
                self.last_replan_time = now  # Update the last replan time
            else:
                self.get_logger().info(f"Replanning on cooldown ({time_since_last_replan:.2f}s elapsed).")

        # Find the nearest waypoint and remove it if reached
        while self.path and np.linalg.norm(current_position - np.array(self.path[0])) < self.lookahead_distance:
            self.path.pop(0)

        # If the path becomes empty after removing waypoints, stop the robot
        if not self.path:
            self.get_logger().info("All waypoints cleared. Stopping the robot.")
            self.cmd_vel_pub.publish(Twist())  # Stop the robot
            return

        # Target the next waypoint
        target_point = np.array(self.path[0])
        direction = target_point - current_position
        angle_to_goal = np.arctan2(direction[1], direction[0])
        angle_error = self.normalize_angle(angle_to_goal - yaw)

        # PD Control
        angular_kp = 3.0
        angular_kd = 0.8
        angular_velocity = angular_kp * angle_error + angular_kd * (angle_error - self.previous_angle_error)
        self.previous_angle_error = angle_error

        # Linear speed proportional to distance to the target
        linear_speed = max(0.05, self.max_linear_speed * (1.0 - abs(angle_error) / np.pi))

        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = np.clip(angular_velocity, -self.max_angular_speed, self.max_angular_speed)
        self.cmd_vel_pub.publish(twist)


    def get_lookahead_point(self, current_position):
        for point in self.path:
            if np.linalg.norm(np.array(point) - current_position) >= self.lookahead_distance:
                return np.array(point)
        return self.path[-1]  # Return the last point if no lookahead point is found


    def simulate_trajectory(self, position, yaw, v, w, dt, sim_time):
        trajectory = []
        x, y, theta = position[0], position[1], yaw

        for _ in range(int(sim_time / dt)):
            # Update the robot's state using kinematic equations
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            theta += w * dt
            trajectory.append((x, y))

            # Check for collision
            if not self.is_collision_free(np.array([x, y])):
                return None  # Trajectory is invalid due to collision

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
            p.x = float(point[0])  # Ensure the value is explicitly a float
            p.y = float(point[1])  # Ensure the value is explicitly a float
            p.z = 0.0  # Assume a flat 2D plane
            marker.points.append(p)

        self.path_marker_pub.publish(marker)
        self.get_logger().info(f"Published path with {len(path)} points.")

    def visualize_waypoint(self, waypoint):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_waypoint"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = waypoint[0]
        marker.pose.position.y = waypoint[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 0.2  # Adjust size
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        self.path_marker_pub.publish(marker)

    def visualize_trajectory(self, trajectory):
        if not trajectory:
            return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "robot_trajectory"
        marker.id = 2
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for point in trajectory:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)

    def visualize_followed_path(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "followed_path"
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0  # Green for the followed path
        marker.color.b = 0.0
        marker.color.a = 1.0

        for point in self.followed_path:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)

    def visualize_local_trajectory(self, trajectory):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "local_path"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0  # Blue for the local trajectory
        marker.color.a = 1.0

        for point in trajectory:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = 0.0  # Assume the robot operates on a flat 2D plane
            marker.points.append(p)

        self.local_path_marker_pub.publish(marker)

    def visualize_lookahead_point(self, point):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "lookahead_point"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = 0.0
        marker.scale.x = 0.2  # Size of the sphere
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 0.0
        marker.color.g = 1.0  # Green for the lookahead point
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.lookahead_marker_pub.publish(marker)

    def visualize_curvature_arc(self, robot_position, lookahead_point, curvature_radius):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "curvature_arc"
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Thickness of the arc
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0  # Blue for the curvature arc
        marker.color.a = 1.0

        # Generate arc points
        num_points = 50
        angle_step = np.linspace(0, np.arctan2(lookahead_point[1] - robot_position[1], lookahead_point[0] - robot_position[0]), num_points)
        for angle in angle_step:
            x = robot_position[0] + curvature_radius * np.cos(angle)
            y = robot_position[1] + curvature_radius * np.sin(angle)
            p = Point()
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        self.curvature_marker_pub.publish(marker)

    def visualize_inflation_zone(self):
        """Visualize the inflation zone around obstacles using MarkerArray."""
        marker_array = MarkerArray()

        inflation_radius = self.robot_radius + 0.1  # Robot radius + safety margin

        for i, obstacle in enumerate(self.local_obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "inflation_zone"
            marker.id = i  # Unique ID for each marker
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(obstacle[0])  # X coordinate of obstacle
            marker.pose.position.y = float(obstacle[1])  # Y coordinate of obstacle
            marker.pose.position.z = 0.0  # Assume flat ground
            marker.scale.x = 2 * inflation_radius  # Diameter of the circle
            marker.scale.y = 2 * inflation_radius  # Diameter of the circle
            marker.scale.z = 0.1  # Small height to make it a flat cylinder
            marker.color.r = 1.0  # Red for the inflation zone
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5  # Semi-transparent
            marker_array.markers.append(marker)

        # Clear unused markers in RViz by setting unused markers to DELETE
        for i in range(len(self.local_obstacles), len(marker_array.markers)):
            marker = Marker()
            marker.action = Marker.DELETE
            marker.id = i
            marker_array.markers.append(marker)

            self.inflation_marker_pub.publish(marker_array)




def main(args=None):
    rclpy.init(args=args)
    node = HybridPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

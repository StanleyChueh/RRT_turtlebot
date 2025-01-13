import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, LookupException
from sensor_msgs.msg import LaserScan
import numpy as np
import random

class HybridPlannerNode(Node):
    def __init__(self):
        super().__init__('hybrid_planner')

        # ROS2 Subscribers and Publishers
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, 'rrt_tree', 10)

        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.tree = []  # [(position, parent_index)]
        self.path = []  # RRT-generated path
        self.current_goal = None
        self.local_obstacles = []  # Obstacles from LaserScan
        self.path_ready = False
        self.step_size = 0.3
        self.max_iterations = 20000
        self.search_radius = 0.3
        self.timer = self.create_timer(0.1, self.control_loop)
        self.obstacle_marker_pub = self.create_publisher(MarkerArray, 'obstacle_markers', 10)

        # PID Parameters
        self.kp_linear = 1.0
        self.kp_angular = 1.5
        self.linear_velocity_max = 0.2  # m/s
        self.angular_velocity_max = 1.0  # rad/s


    def get_current_position(self):
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.time.Duration(seconds=0.5))
            position = np.array([transform.transform.translation.x, transform.transform.translation.y])
            self.get_logger().info(f"Current position: {position}")
            return position
        except Exception as e:
            self.get_logger().warn(f"TF Error: {e}")
            return None

    def goal_callback(self, msg):
        self.get_logger().info(f"Received goal: {msg.pose.position}")
        self.current_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.path_ready = False
        self.path = []
        self.plan_global_path()

    def scan_callback(self, msg):
        self.local_obstacles = []
        angle = msg.angle_min

        try:
            # Get the transformation from base_link to map
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.time.Duration(seconds=1.0))
            translation = np.array([transform.transform.translation.x, transform.transform.translation.y])
            yaw = self.get_yaw_from_transform(transform)

            # Create a rotation matrix for the yaw angle
            rotation_matrix = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])

            # Process the scan data
            for r in msg.ranges:
                if 0.1 < r < msg.range_max:  # Ignore invalid readings
                    # Calculate obstacle position in base_link frame
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    obstacle_in_base_link = np.array([x, y])

                    # Transform to the map frame
                    obstacle_in_map = np.dot(rotation_matrix, obstacle_in_base_link) + translation
                    self.local_obstacles.append(obstacle_in_map)
                angle += msg.angle_increment

            # Visualize obstacles in map frame
            self.visualize_collision_obstacles(self.local_obstacles)

        except LookupException as e:
            self.get_logger().warn(f"TF Error: {e}")

    def plan_global_path(self):
        """RRT Global Path Planning"""
        self.tree = []
        start = self.get_current_position()
        if start is None or self.current_goal is None:
            self.get_logger().warn("Unable to retrieve start or goal position.")
            return

        goal = self.current_goal
        self.tree.append((start, None))

        for _ in range(self.max_iterations):
            rand_point = self.sample_random_point()
            nearest, nearest_idx = self.get_nearest_node(rand_point)
            new_point = self.steer(nearest, rand_point)

            if self.is_collision_free(nearest, new_point):
                self.tree.append((new_point, nearest_idx))
                if np.linalg.norm(new_point - goal) < self.step_size:
                    self.path = self.extract_path(len(self.tree) - 1)
                    break

        if self.path:
            self.get_logger().info("Global path found.")
            self.path_ready = True
            self.visualize_tree()
            self.visualize_path()
        else:
            self.get_logger().warn("No path found! Resetting planner.")
            self.tree = []
            self.path_ready = False
            return

    def sample_random_point(self):
        x_min = min(self.current_goal[0], self.tree[0][0][0]) - 1.0
        x_max = max(self.current_goal[0], self.tree[0][0][0]) + 1.0
        y_min = min(self.current_goal[1], self.tree[0][0][1]) - 1.0
        y_max = max(self.current_goal[1], self.tree[0][0][1]) + 1.0
        sampled_point = np.array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])
        self.get_logger().info(f"Sampled point: {sampled_point}")
        return sampled_point

    def is_obstacle_nearby(self):
        for obs in self.local_obstacles:
            if np.linalg.norm(obs) < 0.5:  # Safe distance threshold
                return True
        return False

    def control_loop(self):
        if self.path_ready:
            if not self.is_path_valid():
                self.get_logger().warn("Path invalid! Replanning...")
                self.cmd_vel_pub.publish(Twist())  # Stop the robot
                self.plan_global_path()
            else:
                self.follow_path_with_dwa()
        if self.is_obstacle_nearby():
            self.get_logger().warn("Obstacle too close! Stopping...")
            self.cmd_vel_pub.publish(Twist())  # Stop the robot
            self.plan_global_path()  # Trigger a replan
            return

    def is_path_valid(self):
        for i in range(len(self.path) - 1):
            if not self.is_collision_free(self.path[i], self.path[i + 1]):
                self.get_logger().warn(f"Path invalidated at segment: {self.path[i]} to {self.path[i + 1]}")
                return False
        return True


    def get_nearest_node(self, point):
        distances = [np.linalg.norm(node[0] - point) for node in self.tree]
        nearest_idx = np.argmin(distances)
        return self.tree[nearest_idx][0], nearest_idx

    def steer(self, from_point, to_point):
        direction = to_point - from_point
        norm = np.linalg.norm(direction)
        return from_point + (direction / norm) * self.step_size if norm > self.step_size else to_point

    def is_collision_free(self, from_point, to_point):
        line_vec = to_point - from_point
        norm = np.linalg.norm(line_vec)
        collision_obstacles = []  # To store obstacles causing collision

        for step in np.arange(0, norm, 0.05):  # Check points along the line
            intermediate_point = from_point + (line_vec / norm) * step
            for obs in self.local_obstacles:
                if np.linalg.norm(intermediate_point - obs) < 0.2:  # Robot radius + clearance
                    collision_obstacles.append(obs)  # Store the obstacle
                    return False  # Collision detected

        # Publish collision obstacles for visualization
        self.visualize_collision_obstacles(collision_obstacles)
        return True


    def extract_path(self, goal_idx):
        path = []
        current = goal_idx
        while current is not None:
            path.append(self.tree[current][0])
            self.get_logger().info(f"Path node: {self.tree[current][0]} with parent index: {self.tree[current][1]}")
            current = self.tree[current][1]
        return path[::-1]


    def follow_path_with_pid(self):
        """PID-based local navigation to follow the path."""
        if not self.path or not self.path_ready:
            self.get_logger().warn("Path not ready or empty.")
            return

        current_position = self.get_current_position()
        if current_position is None:
            return

        # Get the next waypoint
        next_waypoint = self.path[0]
        distance = np.linalg.norm(next_waypoint - current_position)

        # Check if we reached the waypoint
        if distance < 0.2:  # Waypoint tolerance
            self.path.pop(0)
            if not self.path:
                self.get_logger().info("Goal reached!")
                self.cmd_vel_pub.publish(Twist())  # Stop the robot
                self.tree = []  # Clear the tree
                self.path = []  # Clear the path
                self.path_ready = False
            return

        # Compute linear and angular errors
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            yaw = self.get_yaw_from_transform(transform)
        except LookupException as e:
            self.get_logger().warn(f"TF Error: {e}")
            return

        angle_to_goal = np.arctan2(next_waypoint[1] - current_position[1], next_waypoint[0] - current_position[0])
        angle_error = self.normalize_angle(angle_to_goal - yaw)

        # PID control for velocities
        twist = Twist()
        twist.linear.x = min(self.kp_linear * distance, self.linear_velocity_max)
        twist.angular.z = np.clip(self.kp_angular * angle_error, -self.angular_velocity_max, self.angular_velocity_max)

        self.cmd_vel_pub.publish(twist)


    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_yaw_from_transform(self, transform):
        q = transform.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        if self.path_ready:
            self.follow_path_with_pid()

        if self.is_obstacle_nearby():
            self.get_logger().warn("Obstacle too close! Stopping...")
            self.cmd_vel_pub.publish(Twist())  # Stop the robot
            self.plan_global_path()  # Trigger a replan


    def create_marker(self, position, marker_id, namespace, color):
        """Create a marker for a tree node."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = 0.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Fully opaque
        return marker

    def create_edge_marker(self, start, end, marker_id, namespace, color):
        """Create a marker for a tree edge."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 1.0  # Fully opaque

        # Add the start and end points of the edge
        start_point = Point()
        start_point.x = float(start[0])
        start_point.y = float(start[1])
        start_point.z = 0.0

        end_point = Point()
        end_point.x = float(end[0])
        end_point.y = float(end[1])
        end_point.z = 0.0

        marker.points.append(start_point)
        marker.points.append(end_point)
        return marker


    def visualize_tree(self):
            marker_array = MarkerArray()

            # Add nodes
            for i, (point, _) in enumerate(self.tree):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "rrt_tree_nodes"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = float(point[0])
                marker.pose.position.y = float(point[1])
                marker.pose.position.z = 0.0
                marker.scale.x = 0.1
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
                marker_array.markers.append(marker)

            # Add edges
            edge_marker = Marker()
            edge_marker.header.frame_id = "map"
            edge_marker.header.stamp = self.get_clock().now().to_msg()
            edge_marker.ns = "rrt_tree_edges"
            edge_marker.id = len(self.tree)
            edge_marker.type = Marker.LINE_LIST
            edge_marker.action = Marker.ADD
            edge_marker.scale.x = 0.05  # Thickness of the lines
            edge_marker.color.r = 0.0
            edge_marker.color.g = 0.0
            edge_marker.color.b = 1.0
            edge_marker.color.a = 1.0

            for point, parent_idx in self.tree:
                if parent_idx is not None:
                    parent_point = self.tree[parent_idx][0]
                    # Add line segment between parent and current point
                    edge_marker.points.append(self.create_point(parent_point))
                    edge_marker.points.append(self.create_point(point))

            marker_array.markers.append(edge_marker)
            self.marker_array_pub.publish(marker_array)

    def create_point(self, point):
        p = Point()
        p.x, p.y, p.z = float(point[0]), float(point[1]), 0.0
        return p
    
    def visualize_path(self):
        path_marker = Marker()
        path_marker.header.frame_id = "map"
        path_marker.header.stamp = self.get_clock().now().to_msg()
        path_marker.ns = "rrt_final_path"
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.1  # Line width
        path_marker.color.r = 1.0
        path_marker.color.g = 0.0
        path_marker.color.b = 0.0
        path_marker.color.a = 1.0

        for waypoint in self.path:
            path_marker.points.append(self.create_point(waypoint))

        self.marker_array_pub.publish(MarkerArray(markers=[path_marker]))

    def visualize_collision_obstacles(self, obstacles):
        marker_array = MarkerArray()

        for i, obs in enumerate(obstacles):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "collision_obstacles"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(obs[0])
            marker.pose.position.y = float(obs[1])
            marker.pose.position.z = 0.0  # Assume a 2D plane
            marker.scale.x = 0.2  # Size of the marker
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.r = 1.0  # Red for obstacles
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0  # Fully opaque

            marker_array.markers.append(marker)

        # Publish the markers
        self.obstacle_marker_pub.publish(marker_array)



def main(args=None):
    rclpy.init(args=args)
    node = HybridPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener, LookupException
from sensor_msgs.msg import LaserScan
import numpy as np
import heapq


class HybridPlannerNode(Node):
    def __init__(self):
        super().__init__('hybrid_planner')

        # ROS2 Subscribers and Publishers
        self.goal_pose_sub = self.create_subscription(PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.path_marker_pub = self.create_publisher(Marker, 'a_star_path', 10)

        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Parameters
        self.grid_resolution = 0.1  # Resolution of the grid
        self.robot_radius = 0.2  # Robot radius
        self.map_bounds = (-5, 5, -5, 5)  # Min x, max x, min y, max y
        self.local_obstacles = []
        self.current_goal = None
        self.path = []

        self.timer = self.create_timer(0.1, self.control_loop)

    def get_current_position(self):
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.time.Duration(seconds=0.5))
            position = np.array([transform.transform.translation.x, transform.transform.translation.y])
            return position
        except Exception as e:
            self.get_logger().warn(f"TF Error: {e}")
            return None

    def goal_callback(self, msg):
        self.get_logger().info(f"Received goal: {msg.pose.position}")
        self.current_goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.plan_global_path()

    def scan_callback(self, msg):
        self.local_obstacles = []
        angle = msg.angle_min

        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time(), timeout=rclpy.time.Duration(seconds=0.5))
            translation = np.array([transform.transform.translation.x, transform.transform.translation.y])
            yaw = self.get_yaw_from_transform(transform)

            rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            for r in msg.ranges:
                if 0.1 < r < msg.range_max:
                    x = r * np.cos(angle)
                    y = r * np.sin(angle)
                    obstacle_in_base_link = np.array([x, y])
                    obstacle_in_map = np.dot(rotation_matrix, obstacle_in_base_link) + translation
                    self.local_obstacles.append(obstacle_in_map)
                angle += msg.angle_increment
        except LookupException as e:
            self.get_logger().warn(f"TF Error: {e}")


    def plan_global_path(self):
        start = self.get_current_position()
        goal = self.current_goal

        if start is None or goal is None:
            self.get_logger().warn("Start or goal position not available.")
            return

        self.path = self.a_star(start, goal)
        if self.path:
            self.get_logger().info("Global path found.")
            self.visualize_path(self.path)
        else:
            self.get_logger().warn("No path found.")

    def a_star(self, start, goal):
        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        def to_grid(point):
            return tuple(((point - np.array([self.map_bounds[0], self.map_bounds[2]])) / self.grid_resolution).astype(int))

        def to_world(grid):
            return np.array(grid) * self.grid_resolution + np.array([self.map_bounds[0], self.map_bounds[2]])

        grid_start = to_grid(start)
        grid_goal = to_grid(goal)

        open_set = []
        heapq.heappush(open_set, (0, grid_start))
        came_from = {}
        g_score = {grid_start: 0}
        f_score = {grid_start: heuristic(grid_start, grid_goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == grid_goal:
                path = []
                while current in came_from:
                    path.append(to_world(current))
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g_score = g_score[current] + 1

                if neighbor in g_score and tentative_g_score >= g_score[neighbor]:
                    continue

                if not self.is_collision_free(to_world(neighbor)):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, grid_goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []
    


    def is_collision_free(self, point):
        for obs in self.local_obstacles:
            if np.linalg.norm(point - obs) < self.robot_radius:
                return False
        return True

    def control_loop(self):
        if not self.path:
            return

        current_position = self.get_current_position()
        if current_position is None or len(current_position) == 0:
            return

        next_waypoint = self.path[0]
        if np.linalg.norm(next_waypoint - current_position) < 0.1:
            self.path.pop(0)
            if not self.path:
                self.cmd_vel_pub.publish(Twist())
                return

        direction = next_waypoint - current_position
        angle_to_goal = np.arctan2(direction[1], direction[0])
        yaw = self.get_current_yaw()
        angle_diff = self.normalize_angle(angle_to_goal - yaw)

        # Store previous angle difference if not already initialized
        if not hasattr(self, 'previous_angle_diff'):
            self.previous_angle_diff = 0.0

        # Derivative term
        angular_rate_of_change = (angle_diff - self.previous_angle_diff) / self.timer.timer_period_ns * 1e-9  # Convert to seconds

        # PD Controller gains
        angular_kp = 3.0  # Proportional gain
        angular_kd = 0.8  # Derivative gain

        # PD control for angular velocity
        angular_velocity = angular_kp * angle_diff + angular_kd * angular_rate_of_change

        # Update previous angle difference
        self.previous_angle_diff = angle_diff

        # Linear velocity proportional to distance
        linear_speed = min(0.2, np.linalg.norm(direction))

        # Create and publish Twist message
        twist = Twist()
        twist.linear.x = linear_speed
        twist.angular.z = np.clip(angular_velocity, -1.0, 1.0)
        self.cmd_vel_pub.publish(twist)


    def get_current_yaw(self):
        """Get the robot's current yaw angle."""
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            return self.get_yaw_from_transform(transform)
        except LookupException:
            return 0.0

    def get_yaw_from_transform(self, transform):
        """Extract yaw angle from a transform."""
        q = transform.transform.rotation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)


    def visualize_path(self, path):
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
            p.x, p.y = point
            p.z = 0.0
            marker.points.append(p)

        self.path_marker_pub.publish(marker)

    def get_current_yaw(self):
        try:
            transform = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            q = transform.transform.rotation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            return np.arctan2(siny_cosp, cosy_cosp)
        except LookupException:
            return 0.0

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi


def main(args=None):
    rclpy.init(args=args)
    node = HybridPlannerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

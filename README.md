# Robot Navigation
## Project Structure
<img width="3620" alt="Welcome to FigJam (3)" src="https://github.com/user-attachments/assets/62b9a3ba-678e-4d33-8a34-2576feaca037" />

## Pre-requirement
1. ROS2 Humble
2. Gazebo
3. Turtlebot
## Usage
### Terminal1 💻
``` 
ros2 launch turtlebot3_gazebo turtlebot3_dqn_stage4.launch.py 
```
### Terminal2 💻
```
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=true
```
### Terminal3 💻
```
Run the code!!(RRT,A*,Hybrid A*)
```
### RRT vs. A* vs. Hybrid A*
#### RRT 
![image](https://github.com/user-attachments/assets/18b75684-7232-40a7-9280-00fcd945fdf3)

https://youtu.be/Bxc5hF18Tw8
#### A* with PD controller
![image](https://github.com/user-attachments/assets/8b3bb5e6-8b2a-46b9-979d-9b53ad6a2828)
https://youtu.be/L-HktZQIf9k?si=_8rFrMYtMGyOo5ih

#### Hybrid A* with PD controller
![image](https://github.com/user-attachments/assets/a37d7aaa-833d-403e-b288-42e22418180c)
https://youtu.be/L-HktZQIf9k?si=_8rFrMYtMGyOo5ih

#### Dynamic Path Planning
![Untitled ‑ Made with FlexClip (39)](https://github.com/user-attachments/assets/e2df8c25-97d0-4f78-95a6-8c3409bea333)
https://youtu.be/WlBBGpIwAvg?si=-KT9otUGZzrRxg3a
## On going
1.Add local planner with costmap for enhancing dynamic replanning ability...

2.Path smoothing enhancement

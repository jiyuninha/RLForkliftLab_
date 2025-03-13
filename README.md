<<<<<<< HEAD
# RLForkliftLab_
=======
# RLForkliftLab

## Description

## Installation

```bash
git clone https://github.com/jiyuninha/RLForkliftLab_.git
cd /home/{$USER}/RLForklift/docker
docker-compose up forklift-lab-base --detach --build --remove-orphans
docker-compose up forklift-lab-ros2 --detach --build --remove-orphans

# Enter the container.
docker exec -it forklift-lab-ros2 bash
```

## Prerequisites

1. **Ubuntu 22.04 & ROS Humble**
2. **Docker**
3. **NVIDIA-Docker toolkit**

## Run

```bash
cd /home/{$USER}/RLForklift/docker

## 1. Enter the container.
docker exec -it forklift-lab-ros2 bash

## 2. Run a Custom Environment in Isaac Lab

## Related Resources
>>>>>>> v1

#!/usr/bin/env bash
set -eo pipefail

source /home/usern/follow-venv/bin/activate
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash

exec /home/usern/follow-venv/bin/python -m deepsort_people_follow.deepsort_people_follow_node "$@"

import os
from subprocess import Popen, DEVNULL


def test_point_robot_example():
    example_folder = os.path.dirname(os.path.abspath(__file__))
    Popen(
        ["python3", example_folder + "/panda_robot.py", "--num_steps", "10"],
        stdout=DEVNULL,
    ).wait()

def test_panda_robot_example():
    example_folder = os.path.dirname(os.path.abspath(__file__))
    Popen(
        ["python3", example_folder + "/panda_robot.py", "--num_steps", "10"],
        stdout=DEVNULL,
    ).wait()

def test_isaac_sanity():
    example_folder = os.path.dirname(os.path.abspath(__file__))
    Popen(
        ["python3", example_folder + "/1080_balls_of_solitude.py", "--num_steps", "10"],
        stdout=DEVNULL,
    ).wait()

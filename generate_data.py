import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(),
        headless=False)
    env.launch()

    task = env.get_task(ReachTarget)

    demos = task.get_demos(2, live_demos=True)
    print(demos[0][0].__dict__.keys())

if __name__ == '__main__':
    main()
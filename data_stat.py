import rlbench.utils
import rlbench.action_modes.action_mode
import rlbench.action_modes.arm_action_modes
import rlbench.action_modes.gripper_action_modes
import rlbench.observation_config
import rlbench.tasks.reach_target
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as R
from PIL import Image
import torchvision.transforms.functional as TF


@hydra.main(version_base=None, config_path='./configs/visualize', config_name='base')
def main(cfg: DictConfig) -> None:
    env: rlbench.Environment = hydra.utils.instantiate(cfg.environment)
    task = env.get_task(hydra.utils.get_class(cfg.task))
    demos = task.get_demos(**cfg.demos)
    
    psum = np.zeros(3)
    psum_sq = np.zeros(3)
    count = 0
    for demo in demos:
        for obs in demo:
            image = TF.to_tensor(Image.open(obs.front_rgb)).numpy()
            psum += np.sum(image, axis=(1, 2))
            psum_sq += np.sum(image ** 2, axis=(1, 2))
            count += image.shape[1] * image.shape[2]
        
    mean = psum / count
    std = np.sqrt(psum_sq / count - mean ** 2)
    print(mean, std)

    env.shutdown()

if __name__ == '__main__':
    main()
import sys
sys.path.append('')

import os
import os.path as osp
import argparse
import numpy as np
import torch
import datetime

from tqdm import tqdm as tqdm

from configs.config import load_config

from nrdf.model.nrdf import NRDF
from nrdf.utils.transforms import axis_angle_to_quaternion


class NRDFProjector:
    def __init__(
            self, 
            device='cuda:0'):
        """

        Args:
            model_dir: Path to  the pretrained NRDF checkpoint
            noisy_pose_path: [optional] Path to the input noisy pose file, npz format
            noisy_pose: [optional] Input noisy pose: numpy [bs, nj*3]
            device: cuda or cpu

        """
        self.device = device
        
        
        # load pretrained NRDF model
        script_dir = os.path.dirname(os.path.abspath(__file__))  # NRDF/nrdf/exp
        project_root = os.path.abspath(os.path.join(script_dir, "../.."))  # Moves up 3 levels to NRDF
        model_dir= os.path.join(project_root,'checkpoints/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1')

        self._load_model(model_dir)



    def predict_dist(
            self, 
            noisy_pose,
        ):
        """
        instead of projecting, we only evaluate how far the input pose is from the learned manifold.
        Args:
            noise_pose: input noisy pose: torch [bs, nj*3], the rotations are in axis-angle format
            
        Returns: dist_pred: torch [bs, 1]

        """
        # parse input
        bs, ndim = noisy_pose.shape # [bs, 63]
        assert ndim == 63, 'Please make sure the input pose is in axis-angle format'
        n_joints = ndim // 3
        
        # move to device
        if type(noisy_pose) == np.ndarray:
            noisy_pose = torch.from_numpy(noisy_pose)
        noisy_pose = noisy_pose.float().to(self.device)

        # convert to quaternion
        noisy_pose_quat = axis_angle_to_quaternion(
            noisy_pose.reshape(-1, n_joints, 3)
        ) # [bs, nj, 4]
        
        # forward
        dist_pred = self.model(noisy_pose_quat, train=False)['dist_pred']

        
        return dist_pred


    
    def _load_model(self, model_dir):
        checkpoint_path = osp.join(model_dir, 'checkpoints', 'checkpoint_epoch_best.tar')
        config_file = osp.join(model_dir, 'config.yaml')
        
        self.model = NRDF(load_config(config_file))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model = self.model.to(self.device)
  




if __name__ == "__main__":

    # load model
    projector = NRDFProjector(
        device="cuda:0"
    )
    # load data
    noisy_pose = np.load('examples/noisy_pose.npz')['noisy_pose_aa']
    
    # compute metric
    dists = projector.predict_dist(noisy_pose) # torch [bs, 1]
    print(dists)




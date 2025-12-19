#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Steerable Diffusion Policy

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

from collections import deque
from collections.abc import Callable
from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from core.env_adapters import BaseEnvAdapter
from utils.logging_utils import SteerLogger
log = SteerLogger("DiffusionPolicySteer")

class DiffusionPolicySteer(DiffusionPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://huggingface.co/papers/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """
    
    name = "diffusion_steer"
    
    def __init__(self, config: DiffusionConfig):
        super().__init__(config)
        self._adapter = None
        self._postprocessor = None

    def post_init(
        self,
        adapter: BaseEnvAdapter,
        postprocessor: Callable,
        sample_batch_size: int = 1,
        action_horizon: int = 8,
    ) -> None:
        self._adapter = adapter
        self._postprocessor = postprocessor
        self._sample_batch_size = sample_batch_size
        self.config.n_action_steps = action_horizon
        # self.._kfd = FKD(
        #         potential_type=PotentialType.MAX,
        #         lmbda=10.0,
        #         num_particles=batch_size,
        #         adaptive_resampling=False,
        #         resample_frequency=1,
        #         resampling_t_start=int(start_step),
        #         resampling_t_end=int(timesteps[-1].item()),
        #         timesteps=timesteps,
        #         reward_fn=_reward_fn,
        #         reward_min_value=-(float('inf')),
        #         device=self.device,
        #     )
    
    @torch.no_grad()
    def select_action(
        self,
        batch: dict[str, Tensor],
        generate_new_chunk: bool = False,
        use_guidance: bool = False,
        keypoints: Optional[np.ndarray] = None,
        guidance_fn: Optional[Callable] = None,
        guide_scale: float = 1.0,
        start_step: Optional[int] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        MCMC_steps: int = 4,
        verbose: bool = False,
    ) -> Tensor:

        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        # NOTE: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if generate_new_chunk:
            if use_guidance:
                action_chunk = self._predict_action_chunk_guided (
                    batch=batch,
                    keypoints=keypoints,
                    guidance_fn=guidance_fn,
                    guide_scale=guide_scale,
                    start_step=start_step,
                    use_diversity=use_diversity,
                    diversity_scale=diversity_scale,
                    verbose=verbose,
                )
            else:
                action_chunk = self.predict_action_chunk(batch)
            # Cache the action chunk for subsequent calls
            self._cached_action_chunk = action_chunk
        else:
            # Return cached action chunk
            action_chunk = self._cached_action_chunk

        return self._postprocessor(action_chunk)

    def _predict_action_chunk_guided(
        self,
        batch: dict[str, Tensor],
        keypoints: Optional[np.ndarray] = None,
        guidance_fn: Optional[Callable] = None,
        guide_scale: float = 1.0,
        start_step: Optional[int] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        verbose: bool = False,
    ) -> Tensor:

        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps
        
        global_cond = self.diffusion._prepare_global_conditioning(batch)
        
        actions = self._guided_conditional_sample(
            batch_size=batch_size,
            global_cond=global_cond,
            keypoints=keypoints,
            guidance_fn=guidance_fn,
            guide_scale=guide_scale,
            start_step=start_step,
            use_diversity=use_diversity,
            diversity_scale=diversity_scale,
            verbose=verbose,
        )
        
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]
    
    def _guided_conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor,
        generator: Optional[torch.Generator] = None,
        keypoints: Optional[np.ndarray] = None,
        guidance_fn: Optional[Callable] = None,
        guide_scale: float = 1.0,
        start_step: Optional[int] = None,
        use_diversity: bool = True,
        diversity_scale: float = 1.0,
        verbose: bool = False,
    ) -> Tensor:
        """
        Core logic of guided conditional sampling.
        
        Sampling process:
        1. t > start_step: Use RBF diversity guidance (make trajectories分散)
        2. t <= start_step: Use keypoint gradient guidance (guide trajectories towards the target)
        """
        device = get_device_from_parameters(self.diffusion)
        dtype = get_dtype_from_parameters(self.diffusion)
        
        
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )
        
        self.diffusion.noise_scheduler.set_timesteps(self.diffusion.num_inference_steps)
        timesteps = self.diffusion.noise_scheduler.timesteps
        
        if start_step is None:
            start_step = int(timesteps[len(timesteps) // 3].item())
        
        use_keypoint_guidance = (
            guidance_fn is not None 
            and keypoints is not None 
        )

        keypoints_tensor = torch.tensor(keypoints, device=device, dtype=dtype)
        
        for i, t in enumerate(timesteps):

            model_output = self.diffusion.unet(
                sample,
                torch.full((batch_size,), t, dtype=torch.long, device=device),
                global_cond=global_cond,
            )
            
            if use_diversity and t > start_step and batch_size > 1:
                div_grad = self._compute_diversity_gradient(sample, verbose=(verbose and i == 0))
                if div_grad is not None:
                    model_output[:, :, :3] += diversity_scale * div_grad[:, :, :3]
            
            elif use_keypoint_guidance and t <= start_step:
                
                kp_grad = self._compute_keypoint_gradient(
                    sample, keypoints_tensor, guidance_fn, verbose=verbose
                )
                if kp_grad is not None:
                    alpha_t = self.diffusion.noise_scheduler.alphas_cumprod[t]
                    scale = guide_scale * (1 - alpha_t).sqrt()
                    model_output[:, :self.config.n_action_steps, :3] -= scale * kp_grad[:, :self.config.n_action_steps, :3]
            
            sample = self.diffusion.noise_scheduler.step(
                model_output, t, sample, generator=generator
            ).prev_sample
        
        return sample
    
    def _sample_to_trajectory_3d(self, sample: Tensor) -> Tensor:
        
        batch_size = sample.shape[0]
        device, dtype = sample.device, sample.dtype
        
        actions = self._postprocessor(sample).to(device, dtype)
        
        if batch_size == 1:
            action_seq = actions.squeeze(0)[:self.config.n_action_steps, :]
            traj = self._adapter.delta_actions_to_ee_trajectory(action_seq).to(device, dtype)
            return traj.unsqueeze(0)
        else:
            trajs = []
            for b in range(batch_size):
                action_seq = actions[b, :self.config.n_action_steps, :]
                traj = self._adapter.delta_actions_to_ee_trajectory(action_seq).to(device, dtype)
                trajs.append(traj)
            return torch.stack(trajs, dim=0)
    
    def _compute_keypoint_gradient(
        self,
        sample: Tensor,
        keypoints_tensor: Tensor,
        guidance_fn: Union[Callable, List[Callable]],
        verbose: bool = False,
    ) -> Optional[Tensor]:
        """
        Compute keypoint gradient.
        """

        if not guidance_fn or self._adapter is None:
            return None
        
        device = sample.device
        dtype = sample.dtype
        
        # Need to enable grad since we're inside a no_grad context from select_action
        with torch.enable_grad():
            sample_grad = sample.detach().requires_grad_(True)
            
            trajectories_3d = self._sample_to_trajectory_3d(sample_grad)
            trajectories_3d = trajectories_3d[:, 1:self.config.n_action_steps, :3] 
            
            total_reward = torch.tensor(0.0, device=device, dtype=dtype)
            
            traj_input = trajectories_3d.squeeze(0) if trajectories_3d.shape[0] == 1 else trajectories_3d
            
            # guidance_fn can be a list of functions or a single function
            if isinstance(guidance_fn, list):
                for fn in guidance_fn:
                    reward = fn(keypoints_tensor, traj_input)
                    if reward is not None:
                        total_reward = total_reward + reward
            else:
                reward = guidance_fn(keypoints_tensor, traj_input)
                if reward is not None:
                    total_reward = total_reward + reward

            if not total_reward.requires_grad:
                return None
            
            gradient = torch.autograd.grad(
                total_reward, sample_grad, create_graph=False, retain_graph=False
            )[0]
            
            grad_norm = torch.norm(gradient)
            if grad_norm > 1e-8:
                gradient = gradient / grad_norm
            
            if verbose:
                log.info(f"Keypoint reward: {total_reward.item():.4f}, grad norm: {grad_norm.item():.6f}")
        
        return gradient
            
    
    def _compute_diversity_gradient(
        self,
        sample: Tensor,
        verbose: bool = False,
    ) -> Optional[Tensor]:
        """
        Compute RBF diversity gradient (make trajectories disperse).
        """
        batch_size = sample.shape[0]
        if batch_size < 2 or self._adapter is None:
            return None
        
        # Need to enable grad since we're inside a no_grad context from select_action
        with torch.enable_grad():
            sample_grad = sample.detach().requires_grad_(True)
            
            trajectories_3d = self._sample_to_trajectory_3d(sample_grad)
            trajectories_pos = trajectories_3d[:, 1:, :3]  # (B, T, 3)
            
            traj_flat = trajectories_pos.reshape(batch_size, -1)  # (B, D)
            
            traj_i = traj_flat.unsqueeze(1)  # (B, 1, D)
            traj_j = traj_flat.unsqueeze(0)  # (1, B, D)
            squared_dist = torch.sum((traj_i - traj_j) ** 2, dim=2)  # (B, B)
            
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=sample.device)
            
            eps = 1e-6
            dist = torch.sqrt(squared_dist + eps)
            inv_dist = (1.0 / (dist + eps)) * mask.float()
            total_potential = torch.sum(inv_dist)
            
            gradient = torch.autograd.grad(
                total_potential, sample_grad, create_graph=False, retain_graph=False
            )[0]
        
            if verbose:
                log.info(f"Diversity gradient norm: {torch.norm(gradient).item():.6f}")
               
        return gradient
        


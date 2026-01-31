from __future__ import annotations

import time
import torch
import warnings
from tensordict import TensorDict
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _PROJECT_ROOT)

from rsl_rl.runners import OnPolicyRunner
from cmg_ws.module.cmg import CMG

class OnPolicyRunnerResidual(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir = None, device = "cuda"):
        super().__init__(env, train_cfg, log_dir, device)
        cmg_policy_path = os.path.join(_PROJECT_ROOT, "cmg_ws/runs/cmg_20260123_194851/cmg_final.pt")
        data_path = os.path.join(_PROJECT_ROOT, "cmg_ws/dataloader/cmg_training_data.pt")
        self.data = torch.load(data_path, weights_only=False)

        stats = self.data["stats"]
        self.model = CMG( # the same as train.py
            motion_dim=stats["motion_dim"],
            command_dim=stats["command_dim"],
            hidden_dim=512,
            num_experts=4,
            num_layers=3,
        )

        self.motion_mean = torch.tensor(stats["motion_mean"], device=device, dtype=torch.float32)
        self.motion_std = torch.tensor(stats["motion_std"], device=device, dtype=torch.float32)
        self.command_min = torch.tensor(stats["command_min"], device=device, dtype=torch.float32)
        self.command_max = torch.tensor(stats["command_max"], device=device, dtype=torch.float32)

        # Load CMG weights, freeze, and set to eval mode
        cmg_checkpoint = torch.load(cmg_policy_path, weights_only=False)
        self.model.load_state_dict(cmg_checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Policy obs layout: [ang_vel(3), gravity(3), cmd(3), joint_pos(29), joint_vel(29), action(29)] = 96
        # With history_length=5, policy obs is 480 dim
        self.single_frame_dim = 96
        self.joint_pos_idx = slice(9, 38)   # 29 dim
        self.joint_vel_idx = slice(38, 67)  # 29 dim
        self.cmg_batch_size = 512  # Process CMG in chunks to avoid OOM

    def _cmg_forward_batched(self, motion_norm, cmd_norm):
        """Run CMG model in mini-batches to avoid OOM from large MoE intermediate tensors."""
        n = motion_norm.shape[0]
        if n <= self.cmg_batch_size:
            return self.model(motion_norm, cmd_norm)

        outputs = []
        for i in range(0, n, self.cmg_batch_size):
            end = min(i + self.cmg_batch_size, n)
            out = self.model(motion_norm[i:end], cmd_norm[i:end])
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations()
        # Handle tuple return from IsaacLab wrapper: (policy_tensor, {"observations": obs_dict})
        if isinstance(obs, tuple):
            obs_tensor, obs_extras = obs
            obs = TensorDict(obs_extras.get("observations", {"policy": obs_tensor}), batch_size=[self.env.num_envs])
        obs = obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # === CMG Forward Pass ===
                    # Get raw joint data directly from robot (not from scaled observations)
                    robot = self.env.unwrapped.scene["robot"]
                    joint_pos = robot.data.joint_pos  # [N, 29] - absolute positions
                    joint_vel = robot.data.joint_vel  # [N, 29] - unscaled velocities

                    # Get velocity command from cmg_input obs group
                    command = obs["cmg_input"]  # [N, 3]

                    # Build motion input and normalize
                    motion_input = torch.cat([joint_pos, joint_vel], dim=-1)  # [N, 58]
                    # Clamp input to training data range (mean ± 3*std) to keep CMG in-distribution
                    motion_input = torch.clamp(motion_input,
                                               self.motion_mean - 3 * self.motion_std,
                                               self.motion_mean + 3 * self.motion_std)
                    motion_norm = (motion_input - self.motion_mean) / self.motion_std
                    cmd_norm = (command - self.command_min) / (self.command_max - self.command_min) * 2 - 1

                    # CMG forward pass (batched to avoid OOM)
                    cmg_out_norm = self._cmg_forward_batched(motion_norm, cmd_norm)
                    qref = cmg_out_norm * self.motion_std + self.motion_mean  # [N, 58] denormalize
                    # Clamp qref to prevent CMG explosion on out-of-distribution inputs
                    qref = torch.clamp(qref, -3.14, 3.14)

                    # Inject CMG output into obs for critic
                    obs["motion"] = qref

                    # === Residual Policy ===
                    # Get residual action from policy (only uses obs["policy"])
                    residual = self.alg.act(obs)  # [N, 29]
                    # Clip residual to prevent instability
                    residual = torch.clamp(residual, -0.8, 0.8)
                    # Zero out arm residuals - let CMG control arms directly
                    # Left arm: 15-21, Right arm: 22-28
                    residual[:, 15:29] = 0.0

                    # Final action = reference joint positions + residual
                    actions = qref[..., :29] + residual  # Only position part of qref


                    # Debug: print every 50 iterations
                    if it % 50 == 0 and _ == 0:
                        tracking_err = (joint_pos - qref[:, :29]).abs().mean()
                        print(f"\n[DEBUG it={it}] qref: [{qref[:, :29].min():.3f}, {qref[:, :29].max():.3f}]")
                        print(f"[DEBUG it={it}] joint_pos: [{joint_pos.min():.3f}, {joint_pos.max():.3f}]")
                        print(f"[DEBUG it={it}] residual: [{residual.min():.3f}, {residual.max():.3f}], mean={residual.abs().mean():.3f}")
                        print(f"[DEBUG it={it}] action: [{actions.min():.3f}, {actions.max():.3f}]")
                        print(f"[DEBUG it={it}] tracking_err: {tracking_err:.4f}")

                    # Inject CMG output into env.extras for reward computation
                    # This must be done BEFORE env.step() so rewards can access qref
                    self.env.unwrapped.extras["cmg_motion"] = qref

                    # Step the environment
                    obs_tensor, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Reconstruct TensorDict from extras (IsaacLab wrapper puts full obs_dict in extras["observations"])
                    if "observations" in extras:
                        obs = TensorDict(extras["observations"], batch_size=[self.env.num_envs])
                    else:
                        obs = TensorDict({"policy": obs_tensor}, batch_size=[self.env.num_envs])
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.alg_cfg["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

            stop = time.time()
            collect_time = stop - start
            start = stop

            # Compute returns
            with torch.inference_mode():
                # Get raw joint data directly from robot (not from scaled observations)
                robot = self.env.unwrapped.scene["robot"]
                joint_pos = robot.data.joint_pos  # [N, 29] - absolute positions
                joint_vel = robot.data.joint_vel  # [N, 29] - unscaled velocities
                command = obs["cmg_input"]
                motion_input = torch.cat([joint_pos, joint_vel], dim=-1) # Concat the obs from robot as CMG input
                # Clamp input to training data range within 3*std to stablise CMG.
                motion_input = torch.clamp(motion_input,
                                           self.motion_mean - 3 * self.motion_std,
                                           self.motion_mean + 3 * self.motion_std)
                motion_norm = (motion_input - self.motion_mean) / self.motion_std # M_t Normalise
                cmd_norm = (command - self.command_min) / (self.command_max - self.command_min) * 2 - 1 # C_t Normalise
                cmg_out_norm = self._cmg_forward_batched(motion_norm, cmd_norm)
                qref = cmg_out_norm * self.motion_std + self.motion_mean # Denormalise using dataset stats.
                qref = torch.clamp(qref, -3.14, 3.14)  # Prevent CMG explosion
                obs["motion"] = qref

                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=self.alg.rnd.weight if self.alg_cfg["rnd_cfg"] else None,
            )

            # Save model
            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training
        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def get_inference_policy(self, device: str | None = None) -> callable:
        """Get a callable inference policy that includes CMG forward pass.

        Returns a wrapper that:
        1. Runs CMG to get reference joint positions
        2. Runs residual policy to get corrections
        3. Returns final action = qref[:29] + residual
        """
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
            self.model.to(device)
            self.motion_mean = self.motion_mean.to(device)
            self.motion_std = self.motion_std.to(device)
            self.command_min = self.command_min.to(device)
            self.command_max = self.command_max.to(device)

        def inference_policy(obs: TensorDict, robot_data=None) -> torch.Tensor:
            """Inference policy that combines CMG and residual.

            Args:
                obs: TensorDict with policy observations
                robot_data: Optional tuple (joint_pos, joint_vel) for raw robot data.
                           If None, will extract from scaled observations (less accurate).
            """
            with torch.inference_mode():
                # Get joint data - prefer raw robot data if provided
                if robot_data is not None:
                    joint_pos, joint_vel = robot_data
                else:
                    # Fallback: extract from policy obs (note: joint_vel may be scaled)
                    policy_obs = obs["policy"]
                    current_frame = policy_obs[..., -self.single_frame_dim:]
                    joint_pos = current_frame[..., self.joint_pos_idx]
                    joint_vel = current_frame[..., self.joint_vel_idx]
                command = obs["cmg_input"]

                # Normalize and run CMG
                motion_input = torch.cat([joint_pos, joint_vel], dim=-1)
                # Clamp input to training data range
                motion_input = torch.clamp(motion_input,
                                           self.motion_mean - 3 * self.motion_std,
                                           self.motion_mean + 3 * self.motion_std)
                motion_norm = (motion_input - self.motion_mean) / self.motion_std
                cmd_norm = (command - self.command_min) / (self.command_max - self.command_min) * 2 - 1
                cmg_out_norm = self._cmg_forward_batched(motion_norm, cmd_norm)
                qref = cmg_out_norm * self.motion_std + self.motion_mean
                qref = torch.clamp(qref, -3.14, 3.14)  # Prevent CMG explosion

                # Inject motion for critic (not used in inference but keeps consistency)
                obs["motion"] = qref

                # Get residual from policy
                residual = self.alg.policy.act_inference(obs)
                # Clip residual to prevent instability
                residual = torch.clamp(residual, -0.8, 0.8)
                # Zero out arm residuals - let CMG control arms directly
                # Left arm: 15-21, Right arm: 22-28
                residual[:, 15:29] = 0.0

                # Final action = reference positions + residual
                actions = qref[..., :29] + residual
                actions[:, 19:22] = 0.0
                actions[:, 26:29] = 0.0 # zero out wrist actions
                return actions

        return inference_policy

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save model including CMG normalization stats."""
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
            # CMG-related info for deployment
            "cmg_state_dict": self.model.state_dict(),
            "cmg_stats": {
                "motion_mean": self.motion_mean.cpu(),
                "motion_std": self.motion_std.cpu(),
                "command_min": self.command_min.cpu(),
                "command_max": self.command_max.cpu(),
                "single_frame_dim": self.single_frame_dim,
                "joint_pos_idx": (self.joint_pos_idx.start, self.joint_pos_idx.stop),
                "joint_vel_idx": (self.joint_vel_idx.start, self.joint_vel_idx.stop),
            },
        }
        # Save RND model if used
        if self.alg_cfg["rnd_cfg"]:
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if self.alg.rnd_optimizer:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)
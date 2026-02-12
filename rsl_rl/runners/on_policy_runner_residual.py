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
from cmg_workspace.module.cmg import CMG

class OnPolicyRunnerResidual(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir = None, device = "cuda"):
        super().__init__(env, train_cfg, log_dir, device)
        cmg_policy_path = os.path.join(_PROJECT_ROOT, "cmg_workspace/runs/cmg_20260211_040530/cmg_ckpt_800.pt")
        data_path = os.path.join(_PROJECT_ROOT, "cmg_workspace/dataloader/cmg_training_data.pt")
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

        # USD → CMG/SDK
        self.joints_usd_to_cmg = torch.tensor([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                                               16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
                                              dtype=torch.long, device=device)

        # CMG/SDK → USD
        self.joints_cmg_to_usd = torch.tensor([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11,
                                               15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
                                              dtype=torch.long, device=device)

        # For autoregressive CMG: store previous output
        self.prev_cmg_output = None

    def cmg2usd(self, motion_cmg):
        pos_cmg = motion_cmg[..., :29]
        vel_cmg = motion_cmg[..., 29:]
        pos_usd = pos_cmg[:, self.joints_cmg_to_usd]
        vel_usd = vel_cmg[:, self.joints_cmg_to_usd]
        return torch.cat([pos_usd, vel_usd], dim=-1)

    def usd2cmg(self, motion_usd):
        pos_usd = motion_usd[..., :29]
        vel_usd = motion_usd[..., 29:]
        pos_cmg = pos_usd[:, self.joints_usd_to_cmg]
        vel_cmg = vel_usd[:, self.joints_usd_to_cmg]
        return torch.cat([pos_cmg, vel_cmg], dim=-1)

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
            # Reset autoregressive state at start of each iteration
            self.prev_cmg_output = None
            # CMG debug accumulators
            _dbg_track_err = []       # per-step mean ||q_robot - q_ref||^2
            _dbg_track_err_gated = [] # same, only high-speed envs
            _dbg_cmg_weight = []      # per-step mean gating weight
            _dbg_residual_abs = []    # per-step mean |residual|
            _dbg_cmd_vx = []          # per-step mean cmd vx
            _dbg_qref_abs = []        # per-step mean |qref_pos|
            # Rollout
            with torch.inference_mode():
                for _step in range(self.cfg["num_steps_per_env"]):
                    # === Hybrid AR/Non-AR CMG Forward Pass ===
                    # Always read fresh robot state
                    robot = self.env.unwrapped.scene["robot"]
                    fresh_usd = torch.cat([robot.data.joint_pos, robot.data.joint_vel], dim=-1)
                    fresh_cmg = self.usd2cmg(fresh_usd)

                    # Get velocity command from cmg_input obs group
                    command = obs["cmg_input"]  # [N, 3]

                    # Hybrid: at high speed use AR (CMG's own output), at low speed use fresh robot state
                    if self.prev_cmg_output is None:
                        cmg_input = fresh_cmg
                    else:
                        ar_gate = torch.clamp((command[:, 0] - 1.1) / 0.2, 0.0, 1.0).unsqueeze(-1)  # [N, 1]
                        cmg_input = ar_gate * self.prev_cmg_output + (1.0 - ar_gate) * fresh_cmg

                    # Clamp and normalize
                    motion_cmg_input = torch.clamp(cmg_input,
                                                   self.motion_mean - 3 * self.motion_std,
                                                   self.motion_mean + 3 * self.motion_std)
                    motion_norm = (motion_cmg_input - self.motion_mean) / self.motion_std  # Normalize in CMG order
                    cmd_norm = (command - self.command_min) / (self.command_max - self.command_min) * 2 - 1

                    # CMG forward pass (batched to avoid OOM)
                    cmg_out_norm = self._cmg_forward_batched(motion_norm, cmd_norm)
                    qref_cmg = cmg_out_norm * self.motion_std + self.motion_mean  # [N, 58] denormalize in CMG order

                    # Store current output for next iteration (autoregressive)
                    self.prev_cmg_output = qref_cmg.clone()

                    # Convert from CMG/SDK order to USD order
                    qref = self.cmg2usd(qref_cmg)
                    
                    # Inject CMG output into obs for critic.
                    obs["motion"] = qref

                    # Residual action from policy
                    residual = self.alg.act(obs)  # [N, 29]
                    # Clip residual (?)
                    # residual = torch.clamp(residual, -0.8, 0.8)

                    # --- CMG Debug: collect per-step diagnostics ---
                    with torch.no_grad():
                        _robot_pos = self.env.unwrapped.scene["robot"].data.joint_pos  # [N, 29]
                        _cmd_vx = command[:, 0]
                        _cmg_w = torch.clamp((_cmd_vx - 1.1) / 0.2, 0.0, 1.0)
                        _err_sq = ((_robot_pos - qref[:, :29]) ** 2).sum(dim=1)  # [N]
                        _dbg_track_err.append(_err_sq.mean().item())
                        _high = _cmg_w > 0
                        if _high.any():
                            _dbg_track_err_gated.append(_err_sq[_high].mean().item())
                        _dbg_cmg_weight.append(_cmg_w.mean().item())
                        _dbg_residual_abs.append(residual.abs().mean().item())
                        _dbg_cmd_vx.append(_cmd_vx.mean().item())
                        _dbg_qref_abs.append(qref[:, :29].abs().mean().item())

                    # Final action = reference joint positions + residual
                    actions = qref[..., :29] + residual  # Only position part of qref (Action order)

                    actions[:, 25] = 0.0  # left wrist pitch
                    actions[:, 26] = 0.0  # right wrist pitch
                    # Inject CMG output into env.extras for reward computation
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
                    if dones.any():
                        robot = self.env.unwrapped.scene["robot"]
                        joint_pos_reset = robot.data.joint_pos[dones]
                        joint_vel_reset = robot.data.joint_vel[dones]
                        motion_usd_reset = torch.cat([joint_pos_reset, joint_vel_reset], dim=-1)
                        motion_cmg_reset = self.usd2cmg(motion_usd_reset)
                        self.prev_cmg_output[dones] = motion_cmg_reset

            # --- CMG Debug: log to tensorboard ---
            if hasattr(self.logger, 'writer') and self.logger.writer is not None:
                _w = self.logger.writer
                n_steps = len(_dbg_track_err)
                _w.add_scalar("CMG_Debug/tracking_error_pos", sum(_dbg_track_err) / n_steps, it)
                if _dbg_track_err_gated:
                    _w.add_scalar("CMG_Debug/tracking_error_gated", sum(_dbg_track_err_gated) / len(_dbg_track_err_gated), it)
                _w.add_scalar("CMG_Debug/cmg_weight_mean", sum(_dbg_cmg_weight) / n_steps, it)
                _w.add_scalar("CMG_Debug/residual_mean_abs", sum(_dbg_residual_abs) / n_steps, it)
                _w.add_scalar("CMG_Debug/cmd_vx_mean", sum(_dbg_cmd_vx) / n_steps, it)
                _w.add_scalar("CMG_Debug/qref_pos_mean_abs", sum(_dbg_qref_abs) / n_steps, it)
                # Error at start, middle, end of rollout — shows if error grows over AR steps
                _w.add_scalar("CMG_Debug/error_step_first", _dbg_track_err[0], it)
                _w.add_scalar("CMG_Debug/error_step_mid", _dbg_track_err[n_steps // 2], it)
                _w.add_scalar("CMG_Debug/error_step_last", _dbg_track_err[-1], it)

            stop = time.time()
            collect_time = stop - start
            start = stop

            # Compute returns
            with torch.inference_mode():
                robot = self.env.unwrapped.scene["robot"]
                joint_pos = robot.data.joint_pos
                joint_vel = robot.data.joint_vel
                command = obs["cmg_input"]
                motion_usd = torch.cat([joint_pos, joint_vel], dim=-1)
                motion_cmg = self.usd2cmg(motion_usd)
                motion_cmg = torch.clamp(motion_cmg,
                                           self.motion_mean - 3 * self.motion_std,
                                           self.motion_mean + 3 * self.motion_std)
                motion_norm = (motion_cmg - self.motion_mean) / self.motion_std
                cmd_norm = (command - self.command_min) / (self.command_max - self.command_min) * 2 - 1
                cmg_out_norm = self._cmg_forward_batched(motion_norm, cmd_norm)
                qref_cmg = cmg_out_norm * self.motion_std + self.motion_mean
                qref_cmg = torch.clamp(qref_cmg, -3.14, 3.14)
                qref = self.cmg2usd(qref_cmg)
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
        1. Runs CMG autoregressively to get reference joint positions
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

        # Autoregressive state (persists across inference calls)
        inference_prev_cmg_output = None

        def inference_policy(obs: TensorDict, robot_data=None, reset=False) -> torch.Tensor:
            """Inference policy with autoregressive CMG + Residual.

            Args:
                obs: TensorDict with policy observations
                robot_data: Optional tuple (joint_pos, joint_vel) for raw robot data.
                           If None, will extract from scaled observations (less accurate).
                reset: If True, reset autoregressive state
            """
            nonlocal inference_prev_cmg_output

            with torch.inference_mode():
                # Reset autoregressive state if requested
                if reset:
                    inference_prev_cmg_output = None

                # Get fresh robot state
                if robot_data is not None:
                    joint_pos, joint_vel = robot_data
                else:
                    policy_obs = obs["policy"]
                    current_frame = policy_obs[..., -self.single_frame_dim:]
                    joint_pos = current_frame[..., self.joint_pos_idx]
                    joint_vel = 20 * current_frame[..., self.joint_vel_idx]

                fresh_usd = torch.cat([joint_pos, joint_vel], dim=-1)
                fresh_cmg = self.usd2cmg(fresh_usd)

                # Get command
                command = obs["cmg_input"]

                # Hybrid: at high speed use AR, at low speed use fresh robot state
                if inference_prev_cmg_output is None:
                    cmg_input = fresh_cmg
                else:
                    ar_gate = torch.clamp((command[:, 0] - 1.1) / 0.2, 0.0, 1.0).unsqueeze(-1)
                    cmg_input = ar_gate * inference_prev_cmg_output + (1.0 - ar_gate) * fresh_cmg

                motion_cmg_input = torch.clamp(cmg_input,
                                               self.motion_mean - 3 * self.motion_std,
                                               self.motion_mean + 3 * self.motion_std)
                motion_norm = (motion_cmg_input - self.motion_mean) / self.motion_std
                cmd_norm = (command - self.command_min) / (self.command_max - self.command_min) * 2 - 1

                # CMG forward pass
                cmg_out_norm = self._cmg_forward_batched(motion_norm, cmd_norm)
                qref_cmg = cmg_out_norm * self.motion_std + self.motion_mean

                # Store for next iteration (autoregressive)
                inference_prev_cmg_output = qref_cmg.clone()

                # Convert to USD order
                qref = self.cmg2usd(qref_cmg)
                obs["motion"] = qref

                # Get residual from policy
                residual = self.alg.policy.act_inference(obs)
                # residual = torch.clamp(residual, -0.8, 0.8)
                actions = qref[..., :29] + residual

                actions[:, 25] = 0.0  # left wrist pitch
                actions[:, 26] = 0.0  # right wrist pitch
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
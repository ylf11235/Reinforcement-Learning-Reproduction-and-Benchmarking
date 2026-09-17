"""LAP replay buffer with history/horizon indexing for MRQ (clean-room).

Design notes (semantics verified against the reference implementation's
observable behavior; no code copied):

- Frames are stored ONCE per transition (the newest frame only); states with
  frame history are reassembled at sample time from index tables.
- ``action_reward_notdone`` rows are ``[reward, not_done, action...]`` where
  the action is one-hot (discrete) or scale-normalized (continuous).
- Sampleability is governed by a mask:
  * a transition becomes sampleable once its forward horizon is complete
    within the episode (streaming unmasks as the episode grows);
  * on episode end, the trailing ``min(ep_len, horizon)`` transitions are
    unmasked for terminated episodes (their horizon crosses the boundary but
    ``not_done`` entries are 0, so multi-step math zeroes out the garbage
    tail) and masked for truncated episodes (the tail is dropped entirely);
  * the last ``history-1`` written slots are pre-masked because their
    reassembled history would span two episodes.
- Priorities are LAP-style: sampling proportional to ``priority**1 * mask``
  via inverse-CDF (cumsum + searchsorted); updated priorities are the raw
  TD-error powers (alpha applied by the agent); no IS-weight correction.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import torch


class ReplayNotReady(Exception):
    """Raised when sampling is requested before any valid transition exists."""


class MRQReplayBuffer:
    def __init__(
        self,
        obs_shape: tuple,
        action_dim: int,
        discrete: bool,
        device: torch.device,
        history: int = 4,
        horizon: int = 5,
        max_size: int = 1_000_000,
        batch_size: int = 256,
        prioritized: bool = True,
        initial_priority: float = 1.0,
        max_action: float = 1.0,
    ):
        self.obs_shape = tuple(obs_shape)
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        self.history = int(history)
        self.horizon = int(horizon)
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)
        self.prioritized = bool(prioritized)
        self.max_priority = float(initial_priority)
        self.action_scale = float(max_action)

        pixel = len(self.obs_shape) == 3
        self.obs_dtype = torch.uint8 if pixel else torch.float32
        self.num_channels = self.obs_shape[0]
        self.state_shape = [self.obs_shape[0] * self.history]
        if pixel:
            self.state_shape += [self.obs_shape[1], self.obs_shape[2]]

        # Frames live on the GPU when they comfortably fit in free VRAM,
        # otherwise on CPU (transfers happen per sampled batch).
        self.storage_device = torch.device("cpu")
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info()
            frame_bytes = 1 if pixel else 4
            obs_bytes = int(np.prod((self.max_size, *self.obs_shape))) * frame_bytes
            ard_bytes = self.max_size * (action_dim + 2) * 4
            if obs_bytes + ard_bytes < free:
                self.storage_device = self.device

        # Ring-buffer write cursor and episode bookkeeping.
        self.ind = 0
        self.size = 0
        self.ep_timesteps = 0
        self.env_terminates = False  # any terminated transition seen so far

        # Index tables: for each slot, the frames composing its state and
        # next_state (length-``history`` index runs).
        self.state_ind = np.zeros((self.max_size, self.history), dtype=np.int32)
        self.next_ind = np.zeros((self.max_size, self.history), dtype=np.int32)
        self.history_queue = deque([0] * self.history, maxlen=self.history)

        # Zero-init: masked slots are never sampled regardless of priority;
        # torch.empty would leak uninitialized memory into the sampling
        # distribution and break seed determinism.
        self.priority = (
            torch.zeros(self.max_size, device=self.device) if self.prioritized else None
        )
        # Mask tensor lives where priorities live (sampling multiplies them).
        mask_device = self.device if self.prioritized else torch.device("cpu")
        self.mask = torch.zeros(self.max_size, device=mask_device)

        self.obs = torch.zeros(
            (self.max_size, *self.obs_shape),
            device=self.storage_device, dtype=self.obs_dtype,
        )
        self.action_reward_notdone = torch.zeros(
            (self.max_size, action_dim + 2), device=self.device, dtype=torch.float32,
        )

        self.sampled_ind = None

    # ------------------------------------------------------------------ add

    def _newest_frame(self, state: np.ndarray) -> torch.Tensor:
        arr = np.asarray(state)
        frame = arr[-self.num_channels:]
        return torch.as_tensor(
            frame.reshape(self.obs_shape).copy(),
            dtype=self.obs_dtype, device=self.storage_device,
        )

    def _encode_action(self, action) -> torch.Tensor:
        if self.discrete:
            one_hot = torch.zeros(self.action_dim, device=self.device)
            one_hot[int(action)] = 1.0
            return one_hot
        return torch.as_tensor(
            np.asarray(action, dtype=np.float32) / self.action_scale,
            device=self.device,
        )

    def add(self, state, action, next_state, reward, terminated, truncated):
        """Store one transition; on episode end also store the final frame."""
        self.obs[self.ind] = self._newest_frame(state)
        row = self.action_reward_notdone[self.ind]
        row[0] = float(reward)
        row[1] = 0.0 if terminated else 1.0
        row[2:] = self._encode_action(action)
        if self.prioritized:
            self.priority[self.ind] = self.max_priority

        self.size = max(self.size, self.ind + 1)
        self.ep_timesteps += 1
        if terminated:
            self.env_terminates = True

        # Pre-mask the slot whose history would be incomplete once the ring
        # wraps onto it.
        self.mask[(self.ind + self.history - 1) % self.max_size] = 0
        # Streaming unmasks transitions whose forward horizon completed.
        if self.ep_timesteps > self.horizon:
            self.mask[(self.ind - self.horizon) % self.max_size] = 1

        # History index tables for this transition.
        next_slot = (self.ind + 1) % self.max_size
        self.state_ind[self.ind] = np.asarray(self.history_queue, dtype=np.int32)
        self.history_queue.append(next_slot)
        self.next_ind[self.ind] = np.asarray(self.history_queue, dtype=np.int32)
        self.ind = next_slot

        if terminated or truncated:
            self._terminal(next_state, truncated)

    def _terminal(self, next_state, truncated: bool):
        # Final frame of the episode.
        self.obs[self.ind] = self._newest_frame(next_state)
        self.mask[(self.ind + self.history - 1) % self.max_size] = 0
        tail = np.arange(min(self.ep_timesteps, self.horizon))
        tail_slots = (self.ind - tail - 1) % self.max_size
        if truncated:
            self.mask[tail_slots] = 0
        else:
            self.mask[tail_slots] = 1
        self.ind = (self.ind + 1) % self.max_size
        self.ep_timesteps = 0
        self.history_queue = deque([self.ind] * self.history, maxlen=self.history)

    # --------------------------------------------------------------- sample

    def can_sample(self, horizon: int) -> bool:
        return bool(self.mask.sum() > 0)

    def _sample_indices(self) -> np.ndarray:
        if not self.can_sample(self.horizon):
            raise ReplayNotReady(
                "no sampleable transitions: complete an episode tail or a "
                f"full horizon first (mask sum = {int(self.mask.sum())})"
            )
        if self.prioritized:
            weights = self.priority * self.mask
            csum = torch.cumsum(weights, dim=0)
            total = csum[-1]
            draws = torch.rand(self.batch_size, device=self.device) * total
            self.sampled_ind = (
                torch.searchsorted(csum, draws).cpu().numpy()
            )
        else:
            valid = torch.nonzero(self.mask).reshape(-1)
            pick = torch.randint(valid.shape[0], (self.batch_size,))
            self.sampled_ind = valid[pick].cpu().numpy()
        return self.sampled_ind

    def sample(self, horizon: int, include_intermediate: bool = False):
        """Sample a batch of transitions (or sub-trajectories).

        Without intermediates: endpoint state/next_state and the FULL reward /
        not_done sequences over ``horizon`` steps (for multi-step returns).
        With intermediates: per-step states/actions (for encoder unrolls).
        """
        start = self._sample_indices()
        idx = (start.reshape(-1, 1) + np.arange(horizon).reshape(1, -1)) % self.max_size
        ard = self.action_reward_notdone[idx]

        if include_intermediate:
            state_idx = np.concatenate(
                [self.state_ind[idx],
                 self.next_ind[idx[:, -1].reshape(-1, 1)]], axis=1)
            frames = self.obs[state_idx].reshape(
                self.batch_size, horizon + 1, *self.state_shape
            ).to(self.device).type(torch.float32)
            state = frames[:, :-1]
            next_state = frames[:, 1:]
            action = ard[:, :, 2:]
        else:
            state_idx = np.concatenate(
                [self.state_ind[idx[:, 0].reshape(-1, 1)],
                 self.next_ind[idx[:, -1].reshape(-1, 1)]], axis=1)
            frames = self.obs[state_idx].reshape(
                self.batch_size, 2, *self.state_shape
            ).to(self.device).type(torch.float32)
            state = frames[:, 0]
            next_state = frames[:, 1]
            action = ard[:, 0, 2:]

        reward = ard[:, :, 0].unsqueeze(-1)
        not_done = ard[:, :, 1].unsqueeze(-1)
        return state, action, next_state, reward, not_done

    # ------------------------------------------------------------ priority

    def update_priority(self, priority: torch.Tensor):
        if not self.prioritized:
            raise RuntimeError("update_priority requires a prioritized buffer")
        idx = torch.as_tensor(
            np.asarray(self.sampled_ind), device=self.priority.device
        ).long()
        vals = priority.reshape(-1).detach().to(
            self.priority.device, torch.float32)
        self.priority[idx] = vals
        self.max_priority = max(float(vals.max()), self.max_priority)

    def reward_scale(self, eps: float = 1e-8) -> float:
        return float(
            self.action_reward_notdone[: self.size, 0].abs().mean().clamp(min=eps)
        )

    # ---------------------------------------------------------- durability

    def save(self, folder: str):
        np.savez_compressed(
            f"{folder}/buffer_data",
            obs=self.obs.cpu().numpy(),
            ard=self.action_reward_notdone.cpu().numpy(),
            state_ind=self.state_ind,
            next_ind=self.next_ind,
            priority=(self.priority.cpu().numpy() if self.prioritized
                      else np.zeros(0)),
            mask=self.mask.cpu().numpy(),
        )
        np.save(
            f"{folder}/buffer_var.npy",
            {
                "ind": self.ind, "size": self.size,
                "env_terminates": self.env_terminates,
                "history_queue": list(self.history_queue),
                "max_priority": self.max_priority,
                "ep_timesteps": self.ep_timesteps,
            },
        )

    def load(self, folder: str):
        data = np.load(f"{folder}/buffer_data.npz")
        self.obs = torch.as_tensor(
            data["obs"], device=self.storage_device, dtype=self.obs_dtype)
        self.action_reward_notdone = torch.as_tensor(
            data["ard"], device=self.device, dtype=torch.float32)
        self.state_ind = data["state_ind"]
        self.next_ind = data["next_ind"]
        if self.prioritized:
            self.priority = torch.as_tensor(
                data["priority"], device=self.device, dtype=torch.float32)
        self.mask = torch.as_tensor(
            data["mask"], device=self.mask.device, dtype=self.mask.dtype)
        var = np.load(f"{folder}/buffer_var.npy", allow_pickle=True).item()
        self.ind = int(var["ind"])
        self.size = int(var["size"])
        self.env_terminates = bool(var["env_terminates"])
        self.history_queue = deque(var["history_queue"], maxlen=self.history)
        self.max_priority = float(var["max_priority"])
        self.ep_timesteps = int(var.get("ep_timesteps", 0))

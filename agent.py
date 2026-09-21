# agent.py
import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

from neural_net import MarioNet


class Mario:
    def __init__(self, state_dim, action_dim, save_dir, save_every=5e5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.save_every = save_every

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.eval_mode = False  # when True, act() is always greedy and never decays exploration_rate

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(100000, device=torch.device("cpu")))
        self.batch_size = 32

        self.gamma = 0.9  # discount factor for future rewards

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

        self.burnin = 1e4       # min. experiences before we start training
        self.learn_every = 3    # steps between calls to update_Q_online
        self.sync_every = 1e4   # steps between target network sync

    def act(self, state):
        """Given a state, choose an action.
        If eval_mode is True: always greedy (argmax), exploration_rate is untouched.
        Otherwise: epsilon-greedy, and exploration_rate decays after every call."""
        if not self.eval_mode and np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        if not self.eval_mode:
            self.exploration_rate *= self.exploration_rate_decay
            self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Store one experience (s, a, r, s', done) into the replay buffer."""
        def first_if_tuple(x):
            return x[0] if isinstance(x, tuple) else x

        state = first_if_tuple(state).__array__()
        next_state = first_if_tuple(next_state).__array__()

        state = torch.tensor(state)
        next_state = torch.tensor(next_state)
        action = torch.tensor([action])
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(
            TensorDict(
                {"state": state, "next_state": next_state, "action": action, "reward": reward, "done": done},
                batch_size=[],
            )
        )

    def recall(self):
        """Sample a random batch of experiences from the replay buffer."""
        batch = self.memory.sample(self.batch_size).to(self.device)
        state, next_state, action, reward, done = (
            batch.get(key) for key in ("state", "next_state", "action", "reward", "done")
        )
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """Q_online(s, a): the current network's estimate for the action actually taken."""
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """r + gamma * Q_target(s', argmax_a' Q_online(s', a')), or just r if the episode ended."""
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        """One gradient step: backprop the loss between estimate and target through Q_online."""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Copy Q_online's weights into Q_target."""
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        """Orchestrates one learning step: sync, sample, estimate, target, update.
        Returns (mean Q value, loss) or (None, None) if no update happened this step."""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)

    def save(self):
        """Save the online network's weights, exploration rate, and step counter to disk."""
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate, curr_step=self.curr_step),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, checkpoint_path):
        """Load network weights, exploration rate, and step counter from a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model"])
        self.exploration_rate = checkpoint["exploration_rate"]
        self.curr_step = checkpoint.get("curr_step",
                                        0)  # .get: retro-compatibile con vecchi checkpoint senza questo campo
        print(
            f"Loaded checkpoint from {checkpoint_path} (exploration_rate={self.exploration_rate:.4f}, curr_step={self.curr_step})")
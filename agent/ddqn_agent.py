import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.dqn_network import DQN
from agent.replay_buffer import ReplayBuffer


class DDQNAgent:
    """
    Double Deep Q-Network (DDQN) agent for 4x4 Tic-Tac-Toe.

    The agent plays as X.

    State size  = 16
    Action size = 16
    """

    def __init__(
        self,
        state_size=16,
        action_size=16,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_frequency=100
    ):

        # --------------------------------------------------
        # Basic parameters
        # --------------------------------------------------

        self.state_size = state_size
        self.action_size = action_size

        # Discount factor
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Training parameters
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # --------------------------------------------------
        # Device
        # --------------------------------------------------

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Using device:", self.device)

        # --------------------------------------------------
        # Online network
        # --------------------------------------------------

        self.policy_net = DQN().to(self.device)

        # --------------------------------------------------
        # Target network
        # --------------------------------------------------

        self.target_net = DQN().to(self.device)

        # Initially both networks are identical
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        # Target network is not trained directly
        self.target_net.eval()

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        # --------------------------------------------------
        # Loss function
        # --------------------------------------------------

        self.loss_function = nn.SmoothL1Loss()

        # --------------------------------------------------
        # Replay buffer
        # --------------------------------------------------

        self.memory = ReplayBuffer(
            capacity=buffer_size
        )

        # Number of learning steps
        self.learn_step = 0

    # ======================================================
    # ACTION SELECTION
    # ======================================================

    
    def select_action(self, state, valid_actions):
        """
        Select an action using DDQN with a tactical safeguard.

        Priority:
        1. Take an immediate winning move.
        2. Block an opponent's immediate winning move.
        3. Otherwise select the legal action with the highest Q-value.
        """
        if not valid_actions:
            return None

        board = np.asarray(state, dtype=np.int8).reshape(-1)

        winning_lines = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
            [0, 5, 10, 15],
            [3, 6, 9, 12],
        ]

        def find_winning_moves(player):
            winning_moves = []

            for action in valid_actions:
                test_board = board.copy()
                test_board[action] = player

                for line in winning_lines:
                    if all(test_board[cell] == player for cell in line):
                        winning_moves.append(action)
                        break

            return winning_moves

        # 1. Always take an immediate win when one exists.
        winning_moves = find_winning_moves(1)

        if winning_moves:
            return winning_moves[0]

        # 2. Block an opponent's immediate win when possible.
        opponent_winning_moves = find_winning_moves(-1)

        if opponent_winning_moves:
            return opponent_winning_moves[0]

        # 3. Otherwise use the original epsilon-greedy DDQN policy.
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        masked_q_values = np.full(self.action_size, -np.inf)

        for action in valid_actions:
            masked_q_values[action] = q_values[action]

        return int(np.argmax(masked_q_values))


    # ======================================================
    # STORE EXPERIENCE
    # ======================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Store an experience in replay memory.
        """

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    # ======================================================
    # TRAIN
    # ======================================================

    def train(self):
        """
        Perform one DDQN learning step.
        """

        # Not enough experiences yet
        if len(self.memory) < self.batch_size:
            return None

        # ----------------------------------------------
        # Sample batch
        # ----------------------------------------------

        (
            states,
            actions,
            rewards,
            next_states,
            dones
        ) = self.memory.sample(self.batch_size)

        # ----------------------------------------------
        # Convert to tensors
        # ----------------------------------------------

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # ==================================================
        # CURRENT Q VALUES
        # ==================================================

        current_q_values = self.policy_net(
            states
        ).gather(
            1,
            actions
        ).squeeze(1)

        # ==================================================
        # DOUBLE DQN TARGET
        # ==================================================

        with torch.no_grad():

            # ------------------------------------------
            # Step 1:
            # Online network calculates Q-values
            # ------------------------------------------

            next_q_policy = self.policy_net(
                next_states
            )

            # ------------------------------------------
            # Mask invalid / occupied cells
            #
            # next_states:
            #   0  -> empty
            #   1  -> X
            #  -1  -> O
            #
            # We only allow actions where state == 0
            # ------------------------------------------

            invalid_actions = next_states != 0

            next_q_policy[invalid_actions] = -float("inf")

            # ------------------------------------------
            # Step 2:
            # Online network selects best VALID action
            # ------------------------------------------

            next_actions = torch.argmax(
                next_q_policy,
                dim=1
            )

            # ------------------------------------------
            # Step 3:
            # Target network evaluates selected action
            # ------------------------------------------

            next_q_target = self.target_net(
                next_states
            )

            next_q_values = next_q_target.gather(
                1,
                next_actions.unsqueeze(1)
            ).squeeze(1)

            # ------------------------------------------
            # Step 4:
            # DDQN target
            # ------------------------------------------

            target_q_values = rewards + (
                self.gamma
                * next_q_values
                * (1 - dones)
            )
            # ------------------------------------------
            # Calculate DDQN target
            # ------------------------------------------

            target_q_values = rewards + (
                self.gamma
                * next_q_values
                * (1 - dones)
            )

        # ==================================================
        # LOSS
        # ==================================================

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # ==================================================
        # BACKPROPAGATION
        # ==================================================

        self.optimizer.zero_grad()

        loss.backward()

        # Prevent extremely large gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        # Increase learning step
        self.learn_step += 1

        # ==================================================
        # UPDATE TARGET NETWORK
        # ==================================================

        if (
            self.learn_step
            % self.target_update_frequency
            == 0
        ):

            self.update_target_network()

        return loss.item()

    # ======================================================
    # TARGET NETWORK UPDATE
    # ======================================================

    def update_target_network(self):
        """
        Copy policy network weights to target network.
        """

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    # ======================================================
    # EPSILON DECAY
    # ======================================================

    def decay_epsilon(self):
        """
        Reduce exploration after each episode.
        """

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )

    # ======================================================
    # SAVE MODEL
    # ======================================================

    def save(self, path):
        """
        Save trained DDQN model.
        """

        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_step": self.learn_step
            },
            path
        )

        print(f"Model saved to: {path}")

    # ======================================================
    # LOAD MODEL
    # ======================================================

    def load(self, path):
        """
        Load a previously trained DDQN model.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon_min
        )

        self.learn_step = checkpoint.get(
            "learn_step",
            0
        )

        print(f"Model loaded from: {path}")
# mcts.py
"""
Implementation of the Monte Carlo Tree Search (MCTS) algorithm.
Uses the neural network to guide the search.
"""

import math
import random
import numpy as np
import torch
import chess
from typing import Dict, Optional, Tuple, List

import config
import utils
from network import PolicyValueNet


class MCTSNode:
    """A node in the Monte Carlo Search Tree."""

    def __init__(
        self, parent: Optional["MCTSNode"], prior_p: float, board_state: chess.Board
    ):
        """
        Initializes a new MCTS node.

        Args:
            parent: Parent node, none for the root node.
            prior_p: The probability of selecting this node's action from the parent. Output from policy head
            board_state: This node's board
        """
        self.parent = parent
        self.children: Dict[chess.Move, "MCTSNode"] = {}
        # current board
        self.board_state = board_state.copy()

        self.n_visits = 0  # this node's visit count
        self.q_value = 0.0  # board evaluation
        self.u_value = 0.0  # UCB
        self.prior_p = prior_p  # prior probability from the policy network

    def expand(self, policy_probs: np.ndarray, legal_moves: List[chess.Move]):
        """
        Creates children for all legal moves.
        The policy probabilities from the network are assigned as priors.

        Args:
            policy_probs: A numpy array of prior probabilities for all possible actions
                          (output from the policy head). Shape: (NUM_ACTIONS,).
            legal_moves: A list of legal chess.Move objects from the current state.
        """
        if self.is_leaf():  # Only expand if not already expanded
            for move in legal_moves:
                if move not in self.children:
                    try:  # Add try-except for move_to_index during expansion
                        move_idx = utils.move_to_index(move)
                        if 0 <= move_idx < len(policy_probs):
                            # get policy from NN
                            prior = policy_probs[move_idx]

                            # give the child the board state after the move
                            child_board = self.board_state.copy()
                            child_board.push(move)
                            self.children[move] = MCTSNode(
                                parent=self, prior_p=prior, board_state=child_board
                            )
                        # else: # Index out of bounds warning (less likely with correct NUM_ACTIONS)
                        #     print(f"Warning: Move index {move_idx} out of bounds for policy_probs during expansion.")
                    except ValueError as e:
                        print(
                            f"Warning: Error getting index for legal move {move.uci()} during expansion: {e}"
                        )

    def select_child(
        self,
    ) -> Tuple[Optional[chess.Move], Optional["MCTSNode"]]:  # Return Optional
        """
        Selects the child node with the highest Upper Confidence Bound for Trees (UCT) score.
        UCT = Q(s, a) + U(s, a)
        where U(s, a) = c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))
        N(s) is the visit count of the parent node.

        Returns:
            A tuple containing the best move (chess.Move) and the corresponding child node (MCTSNode).
            Returns (None, None) if the node is terminal or has no children.
        """
        best_score = -float("inf")
        best_move = None
        best_child = None

        # Ensure parent visit count is available for UCT calculation
        # Use self visits if root
        parent_visits = self.parent.n_visits if self.parent else self.n_visits

        # *** Check if there are children BEFORE calculating UCT ***
        if not self.children:
            return None, None  # Cannot select if no children

        # Handle case where parent_visits might be 0 initially
        # Add small epsilon to parent_visits for sqrt calculation if 0
        sqrt_parent_visits = math.sqrt(
            parent_visits + 1e-6
        )  # Add epsilon for stability

        for move, child in self.children.items():
            uct_score = child.get_uct_score(
                parent_visits, sqrt_parent_visits
            )  # Pass precomputed sqrt
            if uct_score > best_score:
                best_score = uct_score
                best_move = move
                best_child = child

        # It's possible best_child remains None if all children have -inf score (e.g., all priors 0?)
        if best_child is None and self.children:
            print(
                "Warning: No best child found despite having children. Selecting randomly."
            )
            # Fallback: select a random child? Or the first one?
            best_move = random.choice(list(self.children.keys()))
            best_child = self.children[best_move]

        return best_move, best_child

    def get_uct_score(
        self, parent_total_visits: int, sqrt_parent_visits: float
    ) -> float:
        """Calculates the UCT score for the edge leading to this node."""
        if self.n_visits == 0:
            # If a node hasn't been visited, Q is 0. U encourages exploration.
            q_value = 0.0
            # U = c * P * sqrt(N_parent) / (1 + N_child) -> c * P * sqrt(N_parent) when N_child=0
            self.u_value = config.CPUCT * self.prior_p * sqrt_parent_visits
        else:
            q_value = self.q_value
            # Full UCT formula: U = c * P * sqrt(N_parent) / (1 + N_child)
            self.u_value = (
                config.CPUCT * self.prior_p * sqrt_parent_visits / (1 + self.n_visits)
            )

        return q_value + self.u_value

    def update(self, value: float):
        """
        Updates the node's statistics after a simulation backpropagates through it.

        Args:
            value: The outcome of the simulation (-1, 0, or 1) from the perspective
                   of the player whose turn it was *at this node*.
        """
        self.n_visits += 1
        # Q value is the average of simulation outcomes seen so far
        # Q(s, a) = Q(s, a) + (v - Q(s, a)) / N(s, a)
        self.q_value += (value - self.q_value) / self.n_visits

    def update_recursive(self, value: float):
        """
        Recursively updates the node and its ancestors up to the root.
        The value needs to be negated at each step because the perspective changes.

        Args:
            value: The outcome of the simulation from the perspective of the player
                   whose turn it was at the *leaf node* of the simulation.
        """
        # Update this node first
        self.update(value)
        # Then update parent with negated value
        if self.parent:
            self.parent.update_recursive(-value)

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (has no children)."""
        return not self.children

    def is_terminal(self) -> bool:
        """Checks if the node represents a terminal game state."""
        # Use claim_draw=True to catch draws like insufficient material
        return self.board_state.is_game_over(claim_draw=True)


def run_mcts(
    root_board: chess.Board,
    model: PolicyValueNet,
    history: List[chess.Board],  # History BEFORE root_board (max 7 states)
    tracker: utils.RepetitionTracker,
) -> Tuple[Optional[chess.Move], np.ndarray]:  # Return Optional[chess.Move]
    """
    Runs the MCTS algorithm for a given number of simulations to determine the best move.

    Args:
        root_board: The current chess.Board state from which to search.
        model: The trained PolicyValueNet instance.
        history: List of recent board states PRIOR TO root_board (max 7 states).
        tracker: Repetition tracker object.

    Returns:
        A tuple containing:
        - best_move: The selected best move (chess.Move) or None if no legal moves.
        - move_probs: The improved policy (visit counts normalized) after MCTS.
                      Shape: (NUM_ACTIONS,).
    """
    root = MCTSNode(parent=None, prior_p=1.0, board_state=root_board)

    # --- Initial Root Expansion ---
    if not root.is_terminal():
        # *** FIX: History for root encoding needs to include root_board ***
        history_for_root_encoding = (history + [root.board_state])[
            -8:
        ]  # Slice ensures max 8 states

        encoded_state = (
            utils.encode_board(
                root.board_state, history_for_root_encoding, tracker
            )  # Pass correctly sliced history
            .unsqueeze(0)
            .to(config.DEVICE)
        )
        with torch.no_grad():
            policy_logits, value = model(encoded_state)
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()

        # Add Dirichlet noise (ensure noise array size matches num legal moves)
        if config.DIRICHLET_ALPHA > 0:
            legal_moves_list = list(root.board_state.legal_moves)  # Get list once
            legal_mask = utils.get_legal_mask(root.board_state).numpy()
            num_legal_moves = int(np.sum(legal_mask))
            if num_legal_moves > 0:
                noise_values = np.random.dirichlet(
                    [config.DIRICHLET_ALPHA] * num_legal_moves
                )
                noisy_policy = policy_probs.copy()
                noise_idx = 0
                for i in range(len(policy_probs)):
                    if legal_mask[i]:
                        if noise_idx < len(noise_values):  # Boundary check for safety
                            noisy_policy[i] = (
                                1 - config.DIRICHLET_EPSILON
                            ) * policy_probs[
                                i
                            ] + config.DIRICHLET_EPSILON * noise_values[noise_idx]
                            noise_idx += 1
                        else:
                            print(
                                f"Warning: Noise index {noise_idx} out of bounds for noise array size {len(noise_values)}"
                            )
                # Re-normalize after noise
                sum_noisy_policy = np.sum(noisy_policy)
                if sum_noisy_policy > 1e-9:
                    policy_probs = noisy_policy / sum_noisy_policy
                else:
                    # Fallback if sum is zero (unlikely)
                    policy_probs = (
                        torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
                    )  # Use original

        root.expand(
            policy_probs, list(root.board_state.legal_moves)
        )  # Pass legal moves list again

    # --- MCTS Simulation Loop ---
    for _ in range(config.NUM_SIMULATIONS):
        node = root
        search_path = [node]
        # History for simulation starts with history including root
        sim_history = history + [root.board_state]  # Start with history INCLUDING root

        # 1. Selection
        while not node.is_leaf():
            move, next_node = node.select_child()

            if next_node is None:
                # This means node is a leaf or has no selectable children
                # print(f"Warning: Selection returned None node from node {node.board_state.fen()}. Treating as leaf.")
                break  # Stop selection, treat current 'node' as leaf

            # Append state AFTER move to simulation history
            sim_history.append(next_node.board_state.copy())

            node = next_node
            search_path.append(node)

        leaf_node = node

        # 2. Expansion & Evaluation
        value = 0.0  # Default value for backpropagation

        if not leaf_node.is_terminal():
            # *** FIX: Slice history for leaf node encoding ***
            history_for_leaf_encoding = sim_history[
                -8:
            ]  # Get last 8 states for encoding

            encoded_state = (
                utils.encode_board(
                    leaf_node.board_state, history_for_leaf_encoding, tracker
                )  # Pass SLICED history
                .unsqueeze(0)
                .to(config.DEVICE)
            )
            with torch.no_grad():
                policy_logits, network_value = model(encoded_state)
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = (
                network_value.item()
            )  # Value from network (perspective of player to move at leaf)

            # Expand the leaf node
            leaf_node.expand(policy_probs, list(leaf_node.board_state.legal_moves))
        else:
            # If the leaf node IS terminal, get actual game outcome
            # Outcome is from perspective of player who *just moved*
            outcome = utils.get_game_outcome(leaf_node.board_state)
            if outcome is not None:
                value = outcome  # Use this directly for backpropagation
            # else: value remains 0.0

        # 3. Backpropagation
        # Update stats starting from leaf, propagating upwards with negated value
        leaf_node.update_recursive(value)

    # --- After all simulations ---
    # Choose move based on visit counts
    move_visits = []
    legal_moves = list(root.board_state.legal_moves)

    if not legal_moves:
        # print(f"Warning: No legal moves from root node {root.board_state.fen()}. Game likely over.")
        return None, np.zeros(config.NUM_ACTIONS)

    for move in legal_moves:
        if move in root.children:
            move_visits.append((move, root.children[move].n_visits))
        else:
            move_visits.append((move, 0))  # Assign 0 visits if not explored

    total_visits = sum(visits for _, visits in move_visits)
    pi = np.zeros(config.NUM_ACTIONS, dtype=np.float32)

    if total_visits > 0:
        for move, visits in move_visits:
            try:
                move_idx = utils.move_to_index(move)
                if 0 <= move_idx < config.NUM_ACTIONS:
                    pi[move_idx] = visits / total_visits
            except ValueError as e:
                print(
                    f"Error getting index for legal move {move.uci()} during policy calculation: {e}"
                )
    else:  # Fallback if no visits (e.g., root is terminal, low sims)
        num_legal = len(legal_moves)
        if num_legal > 0:
            prob = 1.0 / num_legal
            for move in legal_moves:
                try:
                    move_idx = utils.move_to_index(move)
                    if 0 <= move_idx < config.NUM_ACTIONS:
                        pi[move_idx] = prob
                except ValueError as e:
                    print(
                        f"Error getting index for legal move {move.uci()} during fallback policy calculation: {e}"
                    )

    # Select best move based on visits
    if move_visits:
        # Sort by visits descending, then maybe by prior probability as tie-breaker?
        # For now, just max visits.
        best_move = max(move_visits, key=lambda item: item[1])[0]
    else:
        # Should be covered by legal_moves check, but as a safeguard
        print("Error: No move visits recorded despite having legal moves?")
        return None, pi  # No best move determinable

    return best_move, pi

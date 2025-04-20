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
from network import PolicyValueNet  # Import the network class


class MCTSNode:
    """Represents a node in the Monte Carlo Search Tree."""

    def __init__(
        self, parent: Optional["MCTSNode"], prior_p: float, board_state: chess.Board
    ):
        """
        Initializes a new MCTS node.

        Args:
            parent: The parent node in the tree. None for the root node.
            prior_p: The prior probability of selecting this node's action from the parent,
                     as determined by the neural network's policy head.
            board_state: The chess.Board state represented by this node.
        """
        self.parent = parent
        self.children: Dict[chess.Move, "MCTSNode"] = {}  # Maps move to child node
        self.board_state = board_state.copy()  # Ensure independent board state

        self.n_visits = 0  # N(s, a): Visit count for the edge leading to this node
        self.q_value = 0.0  # Q(s, a): Mean action value (estimated win rate)
        self.u_value = 0.0  # U(s, a): Exploration bonus (Upper Confidence Bound)
        self.prior_p = prior_p  # P(s, a): Prior probability from the policy network

    def expand(self, policy_probs: np.ndarray, legal_moves: List[chess.Move]):
        """
        Expands the node by creating children for all legal moves.
        The policy probabilities from the network are assigned as priors.

        Args:
            policy_probs: A numpy array of prior probabilities for all possible actions
                          (output from the policy head). Shape: (NUM_ACTIONS,).
            legal_moves: A list of legal chess.Move objects from the current state.
        """
        if self.is_leaf():  # Only expand if not already expanded
            for move in legal_moves:
                if move not in self.children:
                    move_idx = utils.move_to_index(move)
                    if 0 <= move_idx < len(policy_probs):
                        prior = policy_probs[move_idx]
                        # Create the board state for the child node
                        child_board = self.board_state.copy()
                        child_board.push(move)
                        self.children[move] = MCTSNode(
                            parent=self, prior_p=prior, board_state=child_board
                        )
                    # else: # Handle cases where move_to_index might be out of bounds (shouldn't happen with correct mapping)
                    #     print(f"Warning: Move index {move_idx} out of bounds for policy_probs.")

    def select_child(self) -> Tuple[chess.Move, "MCTSNode"]:
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
        parent_visits = (
            self.parent.n_visits if self.parent else self.n_visits
        )  # Use self visits if root

        if not self.children or parent_visits == 0:
            # Handle terminal nodes or nodes right after creation (parent_visits might be 0)
            # If root node has 0 visits, U calculation needs adjustment or initial random choice.
            # For simplicity here, if parent_visits is 0, U becomes 0.
            # A better approach might involve virtual losses or first-play urgency.
            parent_visits = (
                1  # Avoid division by zero, effectively making U rely only on prior
            )

        for move, child in self.children.items():
            uct_score = child.get_uct_score(parent_visits)
            if uct_score > best_score:
                best_score = uct_score
                best_move = move
                best_child = child

        return best_move, best_child

    def get_uct_score(self, parent_total_visits: int) -> float:
        """Calculates the UCT score for the edge leading to this node."""
        if self.n_visits == 0:
            # If a node hasn't been visited, its Q value is often initialized
            # heuristically (e.g., 0 or parent's Q). U should be high to encourage exploration.
            # Using prior_p directly in U calculation is common.
            q_value = 0.0  # Or some other initialization
            # Simplified U calculation for unvisited nodes (often just based on prior and parent visits)
            self.u_value = config.CPUCT * self.prior_p * math.sqrt(parent_total_visits)
        else:
            q_value = self.q_value
            # Full UCT formula
            self.u_value = (
                config.CPUCT
                * self.prior_p
                * math.sqrt(parent_total_visits)
                / (1 + self.n_visits)
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
        # Q(s, a) = ( Q(s, a) * (N(s, a) - 1) + v ) / N(s, a)
        self.q_value += (value - self.q_value) / self.n_visits

    def update_recursive(self, value: float):
        """
        Recursively updates the node and its ancestors up to the root.
        The value needs to be negated at each step because the perspective changes.

        Args:
            value: The outcome of the simulation from the perspective of the player
                   whose turn it was at the *leaf node* of the simulation.
        """
        # If it's not the root node, update the parent first
        if self.parent:
            # The value needs to be negated because the parent represents the opponent's turn
            self.parent.update_recursive(-value)

        # Update this node's statistics
        self.update(value)

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (has no children)."""
        return not self.children

    def is_terminal(self) -> bool:
        """Checks if the node represents a terminal game state."""
        return self.board_state.is_game_over()


def run_mcts(
    root_board: chess.Board, model: PolicyValueNet, history: List[chess.Board]
) -> Tuple[chess.Move, np.ndarray]:
    """
    Runs the MCTS algorithm for a given number of simulations to determine the best move.

    Args:
        root_board: The current chess.Board state from which to search.
        model: The trained PolicyValueNet instance.
        history: List of recent board states for network input encoding.

    Returns:
        A tuple containing:
        - best_move: The selected best move (chess.Move).
        - move_probs: The improved policy (visit counts normalized) after MCTS.
                      Shape: (NUM_ACTIONS,).
    """
    root = MCTSNode(parent=None, prior_p=1.0, board_state=root_board)

    # Initial expansion of the root node
    if not root.is_terminal():
        encoded_state = (
            utils.encode_board(root.board_state, history).unsqueeze(0).to(config.DEVICE)
        )
        with torch.no_grad():
            policy_logits, value = model(encoded_state)
        policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
        # Add Dirichlet noise for exploration during self-play training
        if config.DIRICHLET_ALPHA > 0:
            legal_mask = utils.get_legal_mask(
                root.board_state
            ).numpy()  # Get boolean mask
            noise = np.random.dirichlet(
                [config.DIRICHLET_ALPHA] * int(np.sum(legal_mask))
            )
            noisy_policy = policy_probs.copy()
            noise_idx = 0
            for i in range(len(policy_probs)):
                if legal_mask[i]:  # Apply noise only to legal moves
                    noisy_policy[i] = (1 - config.DIRICHLET_EPSILON) * policy_probs[
                        i
                    ] + config.DIRICHLET_EPSILON * noise[noise_idx]
                    noise_idx += 1
            policy_probs = noisy_policy

        root.expand(policy_probs, list(root.board_state.legal_moves))
        # Initial update for the root based on network's value prediction?
        # AlphaZero doesn't explicitly backpropagate this initial value,
        # but it influences the priors. Let's skip explicit root update here.
        # root.update(value.item()) # This might bias the root Q value prematurely

    for _ in range(config.NUM_SIMULATIONS):
        node = root
        search_path = [node]  # Keep track of nodes visited in this simulation

        # 1. Selection: Traverse the tree using UCT scores until a leaf node is reached.
        while not node.is_leaf():
            move, node = node.select_child()
            if node is None:  # Should not happen if selection logic is correct
                print("Warning: Selection returned None node.")
                # Fallback: treat the parent as the leaf for this simulation
                node = search_path[-1]
                break
            search_path.append(node)

        leaf_node = node
        value = 0.0  # Default value

        # 2. Expansion & Evaluation: If the leaf node is not terminal, expand it.
        if not leaf_node.is_terminal():
            # Get network evaluation for the leaf node
            encoded_state = (
                utils.encode_board(leaf_node.board_state, history)
                .unsqueeze(0)
                .to(config.DEVICE)
            )  # TODO: Pass history correctly
            with torch.no_grad():
                policy_logits, network_value = model(encoded_state)
            policy_probs = torch.softmax(policy_logits, dim=1).squeeze(0).cpu().numpy()
            value = (
                network_value.item()
            )  # Value from the network's perspective (current player at leaf)

            # Expand the leaf node
            leaf_node.expand(policy_probs, list(leaf_node.board_state.legal_moves))
        else:
            # If the leaf node is terminal, the value is the actual game outcome
            # Get outcome from the perspective of the player whose turn it *would* be
            outcome = utils.get_game_outcome(leaf_node.board_state)
            if outcome is not None:
                # The value needs to be from the perspective of the player whose turn it was
                # *at the parent* of the terminal node. Since the game ended here,
                # the player who just moved led to this terminal state.
                # If leaf_node.parent.board_state.turn == leaf_node.board_state.turn, something is wrong.
                # The value should be relative to the player to move at the *leaf node*.
                # get_game_outcome returns relative to the player whose turn it is in the *final* state.
                # Let's adjust: if White wins (1.0), value is 1 if White is to move, -1 if Black.
                # If Black wins (-1.0), value is 1 if Black is to move, -1 if White.
                # Draw (0.0) is always 0.
                # Simpler: outcome is from White's perspective (1=W win, -1=B win, 0=Draw)
                # Value for backprop should be from the perspective of player to move at leaf.
                if (
                    leaf_node.board_state.turn == chess.WHITE
                ):  # If it would be white's turn
                    value = outcome
                else:  # If it would be black's turn
                    value = -outcome

        # 3. Backpropagation: Update visit counts and Q-values along the search path.
        # The value must be propagated back from the perspective of the player at each node.
        leaf_node.update_recursive(value)  # Start backpropagation from the leaf

    # After simulations, choose the move based on visit counts (policy target)
    move_visits = []
    legal_moves = list(root.board_state.legal_moves)
    for move in legal_moves:
        if move in root.children:
            move_visits.append((move, root.children[move].n_visits))
        else:
            move_visits.append(
                (move, 0)
            )  # Move might not have been explored if sims are low

    if not move_visits:
        print("Warning: No moves explored by MCTS!")
        # Handle this case: maybe return a random legal move?
        if legal_moves:
            return random.choice(legal_moves), np.zeros(
                config.NUM_ACTIONS
            )  # Return random move, zero probs
        else:
            return None, np.zeros(config.NUM_ACTIONS)  # No legal moves

    # Create the improved policy vector (pi) based on visit counts
    pi = np.zeros(config.NUM_ACTIONS, dtype=np.float32)
    total_visits = sum(visits for _, visits in move_visits)

    if total_visits > 0:
        # Temperature controls exploration vs exploitation in action selection
        # For training, usually temp=1 for first N moves, then decays
        # For evaluation, usually temp is very small (greedy)
        # Here, we just calculate the probabilities based on visits.
        # The temperature application happens in the self-play loop.
        for move, visits in move_visits:
            move_idx = utils.move_to_index(move)
            if 0 <= move_idx < config.NUM_ACTIONS:
                pi[move_idx] = visits / total_visits
    else:
        # If no visits (e.g., only one legal move, or very few sims), assign uniform prob
        # Or handle based on priors? For now, uniform over legal moves.
        num_legal = len(legal_moves)
        if num_legal > 0:
            prob = 1.0 / num_legal
            for move in legal_moves:
                move_idx = utils.move_to_index(move)
                if 0 <= move_idx < config.NUM_ACTIONS:
                    pi[move_idx] = prob

    # Select the best move based on the highest visit count (most robust choice)
    best_move = max(move_visits, key=lambda item: item[1])[0]

    return best_move, pi

# mcts.py
"""
Implementation of the Monte Carlo Tree Search (MCTS) algorithm.
Uses the neural network to guide the search.
"""

# import enum
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
        self.key = board_state._transposition_key()

        self.n_visits = 0  # this node's visit count
        self.q_value = 0.0  # board evaluation
        self.u_value = 0.0  # UCB
        self.prior_p = prior_p  # prior probability from the policy network
        self.encoded_state: Optional[torch.Tensor] = None

    def expand(self, policy_probs: np.ndarray, legal_moves: List[chess.Move]):
        """
        Creates children for all legal moves.
        The policy probabilities from the network are assigned as priors.

        Args:
            policy_probs: A numpy array of prior probabilities for all possible actions
                          (output from the policy head). Shape: (NUM_ACTIONS,).
            legal_moves: A list of legal chess.Move objects from the current state.
        """
        max_children = int(
            config.WIDEN_COEFF * math.sqrt(self.n_visits + 1) or len(legal_moves)
        )
        sorted_moves = sorted(
            legal_moves,
            key=lambda m: policy_probs[utils.move_to_index(m)],
            reverse=True,
        )[:max_children]

        for move in sorted_moves:
            if move not in self.children:
                child_board = self.board_state.copy()
                child_board.push(move)
                prior = policy_probs[utils.move_to_index(move)]
                node = MCTSNode(parent=self, prior_p=prior, board_state=child_board)
                self.children[move] = node

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

        parent_visits = self.parent.n_visits if self.parent else self.n_visits
        # Cannot select if no children
        if not self.children:
            return None, None
        sqrt_parent = math.sqrt(parent_visits + 1e-8)

        for move, child in self.children.items():
            # Q + U
            if child.n_visits > 0:
                q = child.q_value
                u = config.CPUCT * child.prior_p * sqrt_parent / (1 + child.n_visits)
            else:
                q = 0.0
                u = config.CPUCT * child.prior_p * sqrt_parent
            score = q + u
            if score > best_score:
                best_score = score
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

    def update(self, value: float):
        """
        Updates the node's statistics after a simulation backpropagates through it.

        Args:
            value: The outcome of the simulation (-1, 0, or 1) from the perspective
                   of the player whose turn it was *at this node*.
        """
        self.n_visits += 1
        # incremental mean
        self.q_value += (value - self.q_value) / self.n_visits

    def update_recursive(self, value: float):
        """
        Recursively updates the node and its ancestors up to the root.
        The value needs to be negated at each step because the perspective changes.

        Args:
            value: The outcome of the simulation from the perspective of the player
                   whose turn it was at the *leaf node* of the simulation.
        """
        self.update(value)
        # update parent with negated value
        if self.parent:
            self.parent.update_recursive(-value)

    def is_leaf(self) -> bool:
        """Checks if the node is a leaf node (has no children)."""
        return not self.children

    def is_terminal(self) -> bool:
        """Checks if the node represents a terminal game state."""
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
        history_enc = (history + [root.board_state])[-8:]
        root.encoded_state = utils.encode_board(root.board_state, history_enc, tracker)

        with torch.no_grad(), torch.autocast(config.DEVICE):
            logits, value = model(root.encoded_state.unsqueeze(0).to(config.DEVICE))
        policy_probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        root.expand(policy_probs, list(root.board_state.legal_moves))
        value = value.item()

        # Add Dirichlet noise
        if config.DIRICHLET_ALPHA > 0:
            legal = list(root.board_state.legal_moves)
            noise = np.random.dirichlet([config.DIRICHLET_ALPHA] * len(legal))

            for i, move in enumerate(legal):
                idx = utils.move_to_index(move)
                policy_probs[idx] = (1 - config.DIRICHLET_EPSILON) * policy_probs[
                    idx
                ] + config.DIRICHLET_EPSILON * noise[i]

            # Re-normalize
            policy_probs = policy_probs / (policy_probs.sum() + 1e-12)

        root.expand(policy_probs, list(root.board_state.legal_moves))

    # Batched simulations
    pending_nodes = []
    pending_paths = []

    # --- MCTS Simulation Loop ---
    for _ in range(config.NUM_SIMULATIONS):
        node = root
        path = [node]

        max_depth = 1000000
        depth = 0

        # 1. Selection
        while not node.is_leaf():
            move, next_node = node.select_child()
            if next_node is node:
                print("Warning select_child returned the same node")
                break
            if next_node is None:
                break
            node = next_node
            path.append(node)
            depth += 1
            if depth >= max_depth:
                print("Warning max selection depth reached")
                break

        leaf = node

        # 2. Terminal -> Backprop
        if leaf.is_terminal():
            outcome = utils.get_game_outcome(leaf.board_state) or 0.0
            leaf.update_recursive(outcome)
            continue

        # 3. Prepare leaf for batch
        if leaf.encoded_state is None:
            history_enc = (history + [leaf.board_state])[-8:]
            leaf.encoded_state = utils.encode_board(
                leaf.board_state, history_enc, tracker
            )

        pending_paths.append(path)
        pending_nodes.append(leaf)

        # 4. If batch is full run network
        if len(pending_nodes) >= config.MCTS_BATCH_SIZE:
            _evaluate_batch(pending_nodes, pending_paths, model)
            pending_nodes.clear()
            pending_paths.clear()

    if pending_nodes:
        _evaluate_batch(pending_nodes, pending_paths, model)

    # 4. best move selection + build pi
    legal_moves = list(root.board_state.legal_moves)
    visits = [
        (move, root.children.get(move, MCTSNode(None, 0, root.board_state)).n_visits)
        for move in legal_moves
    ]

    total = sum(v for _, v in visits)
    pi = np.zeros(config.NUM_ACTIONS, dtype=np.float32)

    if total > 0:
        for move, count in visits:
            idx = utils.move_to_index(move)
            if 0 <= idx < config.NUM_ACTIONS:
                pi[idx] = count / total
    else:
        for move in legal_moves:
            idx = utils.move_to_index(move)
            if 0 <= idx < config.NUM_ACTIONS:
                pi[idx] = 1.0 / len(legal_moves)
    best_move = max(visits, key=lambda x: x[1])[0]
    return best_move, pi


def _evaluate_batch(nodes, paths, model):
    batch = torch.stack([n.encoded_state for n in nodes], dim=0).to(config.DEVICE)
    with torch.no_grad(), torch.autocast(config.DEVICE):
        logits, values = model(batch)
    policy_batch = torch.softmax(logits, dim=1).cpu().numpy()
    values = values.squeeze(1).cpu().numpy()

    # expand and backprop
    for leaf, path, probs, val in zip(nodes, paths, policy_batch, values):
        legal = list(leaf.board_state.legal_moves)
        leaf.expand(probs, legal)

        leaf.update_recursive(val)

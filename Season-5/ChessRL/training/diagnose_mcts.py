"""Deep MCTS diagnostic: inspect visit distributions, Q-values, priors after search."""

import os
import sys
import math
import numpy as np
import torch

import engine_c as cc

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHESS_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ChineseChess", "backend")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, CHESS_DIR)

from agents.alphazero.network import AlphaZeroNet
from env.action_space import decode_action, NUM_ACTIONS

PIECE_NAMES = {1: "General", 2: "Advisor", 3: "Elephant",
               4: "Horse", 5: "Chariot", 6: "Cannon", 7: "Soldier"}

def piece_str(piece):
    if piece == 0: return "."
    color = "R" if piece > 0 else "B"
    return f"{color}-{PIECE_NAMES.get(abs(piece), '?')}"


class MCTSNode:
    __slots__ = ['parent', 'action', 'prior', 'visit_count', 'value_sum',
                 'children', 'is_expanded']
    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = []
        self.is_expanded = False

    def q_value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def select_child(self, c_puct):
        total_visits = sum(c.visit_count for c in self.children)
        sqrt_total = math.sqrt(total_visits + 1)
        best_score, best_child = -float('inf'), None
        for child in self.children:
            q = child.q_value()
            u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, policy, legal_actions):
        self.is_expanded = True
        total_p = sum(policy[a] for a in legal_actions) + 1e-8
        for a in legal_actions:
            self.children.append(MCTSNode(parent=self, action=a, prior=policy[a] / total_p))

    def backup(self, value):
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value
            node = node.parent

    def is_leaf(self):
        return not self.is_expanded


def run_mcts_diagnostic(network, device, game, num_sims=200, c_puct=1.5,
                         add_noise=True, label=""):
    """Run MCTS and print full diagnostic of the search tree."""

    # Get raw network output first
    obs = cc.board_to_observation(game.board, game.current_turn)
    mask = cc.get_action_mask(game.board, game.current_turn)

    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    mask_t = torch.BoolTensor(mask.astype(bool)).unsqueeze(0).to(device)

    with torch.no_grad():
        log_p, v = network(obs_t, mask_t)
        raw_policy = torch.exp(log_p).squeeze(0).cpu().numpy()
        raw_value = v.item()

    legal_actions = cc.get_legal_action_indices(game.board, game.current_turn)

    print(f"\n{'='*70}")
    print(f"MCTS Diagnostic: {label}")
    print(f"{'='*70}")
    print(f"Position: {'Red' if game.current_turn == cc.RED else 'Black'} to move")
    print(f"Legal moves: {len(legal_actions)}")
    print(f"Raw network value: {raw_value:.4f}")

    # Raw policy distribution
    legal_probs = np.array([raw_policy[a] for a in legal_actions])
    legal_probs /= legal_probs.sum() + 1e-8
    sorted_idx = np.argsort(-legal_probs)

    print(f"\n--- Raw Network Policy (top 10) ---")
    print(f"Entropy: {-np.sum(legal_probs * np.log(legal_probs + 1e-8)):.3f} / {np.log(len(legal_actions)):.3f}")
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        a = legal_actions[idx]
        fr, fc, tr, tc = decode_action(a)
        p = game.board.get(fr, fc)
        print(f"  {i+1:2d}. {piece_str(p):12s} ({fr},{fc})->({tr},{tc})  prior={legal_probs[idx]:.4f}")

    # Now run MCTS step by step
    root = MCTSNode()
    game_states = {id(root): game}

    # Expand root
    root.expand(raw_policy, legal_actions)

    # Add Dirichlet noise
    if add_noise and root.children:
        noise = np.random.dirichlet([0.3] * len(root.children))
        for j, child in enumerate(root.children):
            child.prior = 0.75 * child.prior + 0.25 * noise[j]

    # Track value predictions at leaves
    leaf_values = []

    # Run simulations ONE AT A TIME (no virtual loss) for clarity
    for sim in range(num_sims):
        node = root
        current_game = game

        # Selection
        depth = 0
        while not node.is_leaf():
            node = node.select_child(c_puct)
            if id(node) in game_states:
                current_game = game_states[id(node)]
            else:
                current_game = current_game.simulate_action(node.action)
                game_states[id(node)] = current_game
            depth += 1

        # Terminal?
        if current_game.status != cc.STATUS_PLAYING:
            if current_game.status == cc.STATUS_RED_WIN:
                tv = 1.0 if current_game.current_turn == cc.RED else -1.0
            elif current_game.status == cc.STATUS_BLACK_WIN:
                tv = 1.0 if current_game.current_turn == cc.BLACK else -1.0
            else:
                tv = 0.0
            node.backup(tv)
            leaf_values.append(tv)
            continue

        # Expand leaf
        leaf_obs = cc.board_to_observation(current_game.board, current_game.current_turn)
        leaf_mask = cc.get_action_mask(current_game.board, current_game.current_turn)

        obs_t = torch.FloatTensor(leaf_obs).unsqueeze(0).to(device)
        mask_t = torch.BoolTensor(leaf_mask.astype(bool)).unsqueeze(0).to(device)

        with torch.no_grad():
            lp, lv = network(obs_t, mask_t)
            leaf_policy = torch.exp(lp).squeeze(0).cpu().numpy()
            leaf_value = lv.item()

        leaf_legal = cc.get_legal_action_indices(current_game.board, current_game.current_turn)
        if leaf_legal:
            node.expand(leaf_policy, leaf_legal)

        node.backup(leaf_value)
        leaf_values.append(leaf_value)

    # Print results
    print(f"\n--- After {num_sims} MCTS Simulations ---")
    print(f"Leaf values: mean={np.mean(leaf_values):.4f}, std={np.std(leaf_values):.4f}, "
          f"min={np.min(leaf_values):.4f}, max={np.max(leaf_values):.4f}")

    # Visit distribution
    children_data = []
    for child in root.children:
        fr, fc, tr, tc = decode_action(child.action)
        p = game.board.get(fr, fc)
        children_data.append({
            'action': child.action,
            'visits': child.visit_count,
            'q': child.q_value(),
            'prior': child.prior,
            'piece': piece_str(p),
            'move': f"({fr},{fc})->({tr},{tc})",
        })

    children_data.sort(key=lambda x: -x['visits'])
    total_visits = sum(c['visits'] for c in children_data)

    print(f"\nTotal root visits: {total_visits}")
    visit_counts = np.array([c['visits'] for c in children_data], dtype=float)
    visit_probs = visit_counts / (visit_counts.sum() + 1e-8)
    visit_entropy = -np.sum(visit_probs * np.log(visit_probs + 1e-8))
    max_entropy = np.log(len(children_data))
    print(f"Visit entropy: {visit_entropy:.3f} / {max_entropy:.3f} ({visit_entropy/max_entropy*100:.1f}%)")

    print(f"\n{'Rank':>4} {'Piece':>12} {'Move':>14} {'Visits':>7} {'Pct':>6} {'Q-value':>8} {'Prior':>7}")
    print("-" * 65)
    for i, c in enumerate(children_data[:20]):
        pct = c['visits'] / total_visits * 100 if total_visits > 0 else 0
        print(f"{i+1:4d} {c['piece']:>12} {c['move']:>14} {c['visits']:7d} {pct:5.1f}% {c['q']:+8.4f} {c['prior']:7.4f}")

    # Q-value statistics
    q_values = [c['q'] for c in children_data if c['visits'] > 0]
    if q_values:
        print(f"\nQ-value stats: mean={np.mean(q_values):.4f}, std={np.std(q_values):.4f}, "
              f"range=[{np.min(q_values):.4f}, {np.max(q_values):.4f}]")
        print(f"Q spread (max-min): {np.max(q_values) - np.min(q_values):.4f}")

    # How many children got 0 visits?
    zero_visits = sum(1 for c in children_data if c['visits'] == 0)
    print(f"Children with 0 visits: {zero_visits}/{len(children_data)}")

    return children_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate4")

    checkpoints = {
        "Pretrained": os.path.join(ckpt_dir, "az_pretrained.pt"),
        "Best (iter 25)": os.path.join(ckpt_dir, "az_best.pt"),
    }

    # Also check no-pretrain checkpoint
    nopretrain_ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "candidate4_v3_nopretrain", "az_checkpoint.pt")
    if os.path.exists(nopretrain_ckpt):
        checkpoints["No-pretrain (latest)"] = nopretrain_ckpt

    for label, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: not found")
            continue

        net = AlphaZeroNet(num_blocks=5, channels=64)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
        elif 'model_state' in ckpt:
            net.load_state_dict(ckpt['model_state'])
        else:
            net.load_state_dict(ckpt)
        net.to(device)
        net.eval()

        print(f"\n\n{'#'*70}")
        print(f"# {label}: {path}")
        print(f"{'#'*70}")

        # Starting position
        game = cc.Game()
        np.random.seed(42)
        run_mcts_diagnostic(net, device, game, num_sims=200, label=f"{label} — Opening Position")

        # Also test after a few moves (mid-game position)
        # Play 10 random moves to get a mid-game position
        np.random.seed(123)
        mid_game = cc.Game()
        for _ in range(10):
            if mid_game.status != cc.STATUS_PLAYING:
                break
            legal = cc.get_legal_action_indices(mid_game.board, mid_game.current_turn)
            a = int(np.random.choice(legal))
            fr, fc, tr, tc = decode_action(a)
            mid_game.make_move(fr, fc, tr, tc)

        np.random.seed(42)
        run_mcts_diagnostic(net, device, mid_game, num_sims=200, label=f"{label} — After 10 random moves")


if __name__ == "__main__":
    main()

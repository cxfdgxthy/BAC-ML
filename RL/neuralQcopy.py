#!/usr/bin/env python3
"""
Q-learning Othello lab: STARTER CODE. Fill in TODO 1-6; everything else is provided.
Pure Python 3, no dependencies.

Two agents, same Q-learning update, different Q representation:
  tabular : Q stored in a dict keyed by canonical (symmetry-reduced) state.
            Converges on 4x4, learns something on 6x6, hopeless on 8x8.
  linear  : Q(s,a) = w . phi(s,a), phi = features of the board after move a.
            Works on 8x8.

Zero-sum self-play formulation (negamax Q):
  Q(s,a) is always from the perspective of the player to move in s.
  target = final result                if the game ended  (+1 win, 0 draw, -1 loss)
         = -gamma * max_a' Q(s',a')    if the opponent moves next
         = +gamma * max_a' Q(s',a')    if the opponent had to pass (same player again)
  Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))

Usage:
  python othello_q.py train --agent tabular --size 4 --episodes 50000
  python othello_q.py train --agent linear  --size 8 --episodes 20000
  python othello_q.py eval  --agent linear  --size 8 --load q_linear_8.json
  python othello_q.py play  --agent linear  --size 8 --load q_linear_8.json --human x
"""
import argparse
import json
import pickle
import random
import time
import numpy as np
import torch
import torch.nn as nn
from sympy import false

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
X, O = 1, -1  # x = black, moves first
SYM = {1: 'x', -1: 'o', 0: '.'}


# ---------------------------------------------------------------- game engine
class Othello:
    def __init__(self, n=8):
        assert n % 2 == 0 and n >= 4
        self.n, self.nn = n, n * n
        self.rays = []
        self.nbrs = []
        for i in range(self.nn):
            r, c = divmod(i, n)
            rays, nb = [], []
            for dr, dc in DIRS:
                ray, rr, cc = [], r + dr, c + dc
                while 0 <= rr < n and 0 <= cc < n:
                    ray.append(rr * n + cc)
                    rr += dr
                    cc += dc
                if ray:
                    nb.append(ray[0])
                if len(ray) >= 2:
                    rays.append(ray)
            self.rays.append(rays)
            self.nbrs.append(nb)
        self.syms = self._symmetries()  # new[j] = old[perm[j]]
        self.sym_inv = [self._inverse(p) for p in self.syms]  # old index -> new index
        # squares grouped by D4 orbit (corner, X-square, C-square, edge, ...)
        seen, self.classes = set(), []
        for i in range(self.nn):
            if i not in seen:
                orbit = sorted({p[i] for p in self.syms})
                seen.update(orbit)
                self.classes.append(orbit)

    def _symmetries(self):
        n = self.n

        def idx(r, c): return r * n + c

        maps = [lambda r, c: (r, c), lambda r, c: (c, n - 1 - r),
                lambda r, c: (n - 1 - r, n - 1 - c), lambda r, c: (n - 1 - c, r),
                lambda r, c: (r, n - 1 - c), lambda r, c: (n - 1 - r, c),
                lambda r, c: (c, r), lambda r, c: (n - 1 - c, n - 1 - r)]
        return [[idx(*f(*divmod(j, n))) for j in range(self.nn)] for f in maps]

    @staticmethod
    def _inverse(perm):
        inv = [0] * len(perm)
        for j, i in enumerate(perm):
            inv[i] = j
        return inv

    def initial(self):
        n, h = self.n, self.n // 2
        b = [0] * self.nn
        b[(h - 1) * n + h - 1] = b[h * n + h] = O
        b[(h - 1) * n + h] = b[h * n + h - 1] = X
        return tuple(b)

    def _flips(self, b, p, i):
        out = []
        for ray in self.rays[i]:
            run = []
            for j in ray:
                v = b[j]
                if v == -p:
                    run.append(j)
                elif v == p:
                    if run:
                        out.extend(run)
                    break
                else:
                    break
        return out

    def _is_legal(self, b, p, i):
        for ray in self.rays[i]:
            if b[ray[0]] != -p:
                continue
            for j in ray[1:]:
                v = b[j]
                if v == p:
                    return True
                if v == 0:
                    break
        return False

    def legal(self, b, p):
        return [i for i in range(self.nn) if b[i] == 0 and self._is_legal(b, p, i)]

    def play(self, b, p, i):
        nb = list(b)
        nb[i] = p
        for j in self._flips(b, p, i):
            nb[j] = p
        return tuple(nb)

    def step(self, b, p, a):
        """Returns (board', player_to_move or None if game over, legal moves)."""
        b2 = self.play(b, p, a)
        m = self.legal(b2, -p)
        if m:
            return b2, -p, m
        m = self.legal(b2, p)
        if m:
            return b2, p, m  # opponent passes
        return b2, None, []

    def show(self, b, moves=()):
        n = self.n
        cols = 'abcdefgh'[:n]
        rows = ['  ' + ' '.join(cols)]
        for r in range(n):
            cells = []
            for c in range(n):
                i = r * n + c
                cells.append('*' if i in moves else SYM[b[i]])
            rows.append(f'{r + 1} ' + ' '.join(cells))
        rows.append(f'x={b.count(X)} o={b.count(O)}')
        return '\n'.join(rows)


def sign(v):
    return (v > 0) - (v < 0)


# ---------------------------------------------------------------- agents
class TabularQ:
    """Q[canonical_state][canonical_action]. Canonical = mover-relative board,
    lexicographically smallest over the 8 board symmetries."""

    def __init__(self, game, alpha=0.2):
        self.g, self.alpha, self.Q = game, alpha, {}

    def _canon(self, b, p):
        """Returns (canonical key, list of symmetries that map b to it).
        More than one symmetry means the position is self-symmetric, and moves that
        are images of each other must share one Q entry."""
        v = tuple(x * p for x in b)
        best, ks = None, []
        for k, perm in enumerate(self.g.syms):
            t = tuple(v[j] for j in perm)
            if best is None or t < best:
                best, ks = t, [k]
            elif t == best:
                ks.append(k)
        return best, ks

    def evaluate(self, b, p, moves):
        """Returns list of (Q value, handle used by update)."""
        key, ks = self._canon(b, p)
        row = self.Q.get(key, {})
        invs = [self.g.sym_inv[k] for k in ks]
        out = []
        for a in moves:
            ca = min(inv[a] for inv in invs)
            out.append((row.get(ca, 0.0), (key, ca)))
        return out

    def update(self, handle, q, target):
        """TODO 4: tabular Q-learning update.
        handle = (key, ca): canonical state key and canonical action index.
        q      = current Q(s,a) (what evaluate returned for this action).
        target = TD target from q_target.
        Store the new value in self.Q[key][ca]. self.Q[key] may not exist yet.
        Learning rate is self.alpha."""
        key, ca = handle

        if key not in self.Q:
            self.Q[key] = {}
        self.Q[key][ca] = q + self.alpha * (target - q)

        return

        raise NotImplementedError('TODO 4: TabularQ.update')

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'n': self.g.n, 'Q': self.Q}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)['Q']

    def stats(self):
        return f'states={len(self.Q)}'


class LinearQ:
    """Q(s,a) = w_phase . phi(after-board from mover's view).
    Features: mean disc ownership per symmetry class (corners, X, C, edges...),
    disc diff, frontier diff, opponent mobility, bias. Separate weights per game phase."""

    PHASES = 3

    def __init__(self, game, alpha=0.01):
        self.g, self.alpha = game, alpha
        self.nf = len(game.classes) + 4
        self.w = [[0.0] * self.nf for _ in range(self.PHASES)]

    def features(self, after, p):
        g = self.g
        v = [x * p for x in after]
        f = [sum(v[i] for i in cls) / len(cls) for cls in g.classes]
        occupied = g.nn - v.count(0)
        f.append(sum(v) / occupied)
        fr = 0
        for i in range(g.nn):
            if v[i] and any(v[j] == 0 for j in g.nbrs[i]):
                fr += v[i]
        f.append(-fr / 10.0)
        f.append(-len(g.legal(after, -p)) / 10.0)
        f.append(1.0)
        empties = g.nn - occupied
        phase = min(self.PHASES - 1, (g.nn - empties) * self.PHASES // g.nn)
        return phase, f

    def evaluate(self, b, p, moves):
        out = []
        for a in moves:
            phase, f = self.features(self.g.play(b, p, a), p)
            out.append((self.q_value(phase, f), (phase, f)))
        return out

    def q_value(self, phase, f):
        """TODO 5: linear Q value.
        phase: int in [0, PHASES). f: list of floats (feature vector phi(s,a)).
        Return Q(s,a) = w[phase] . f   (self.w is a list of weight lists)."""

        weights = self.w[phase]

        q = float(np.dot(weights, f))
        return q

        raise NotImplementedError('TODO 5: LinearQ.q_value')

    def update(self, handle, q, target):
        """TODO 6: semi-gradient Q-learning update for a linear Q.
        handle = (phase, f). Only the weights for this phase change.
        Hint: what is dQ/dw[phase][i]?"""

        phase, f = handle

        delta = self.alpha * (target - q) * np.asarray(f)

        self.w[phase] = (np.asarray(self.w[phase]) + delta).tolist()

        return

        raise NotImplementedError('TODO 6: LinearQ.update')

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'n': self.g.n, 'w': self.w}, f, indent=1)

    def load(self, path):
        with open(path) as f:
            self.w = json.load(f)['w']

    def stats(self):
        return 'w_mid=' + ' '.join(f'{x:+.2f}' for x in self.w[1])


import numpy as np
import torch
import torch.nn as nn


class NeuralQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
        )

    @staticmethod
    def generateMask(board):
        dirs = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
        """
        board: (8, 8) array from the perspective of the player to move:
               +1 = own disc, -1 = opponent disc, 0 = empty.
        returns: (8, 8) bool array, True where the move is legal.
                 Square (r, c) maps to index 8*r + c.
        """
        mask = np.zeros(64, dtype=bool)
        for r in range(8):
            for c in range(8):
                if board[r, c] != 0:
                    continue
                for dr, dc in dirs:
                    rr, cc = r + dr, c + dc
                    seen_opp = False
                    while 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == -1:
                        seen_opp = True
                        rr += dr
                        cc += dc
                    if seen_opp and 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == 1:
                        mask[8 * r + c] = True
                        break
        return mask.reshape(8, 8)

    @staticmethod
    def applyMove(board, a):
        """Play square a for the player to move (+1); returns the new board (same perspective)."""
        r, c = divmod(int(a), 8)
        new = board.copy()
        new[r, c] = 1
        for dr, dc in [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]:
            rr, cc, run = r + dr, c + dc, []
            while 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == -1:
                run.append((rr, cc))
                rr += dr
                cc += dc
            if run and 0 <= rr < 8 and 0 <= cc < 8 and board[rr, cc] == 1:
                for fr, fc in run:
                    new[fr, fc] = 1
        return new

    def _get_device(self):
        return next(self.parameters()).device

    def _board_to_tensor_and_mask(self, board, device):
        if isinstance(board, np.ndarray):
            mask_np = self.generateMask(board)
            self_tiles = (board == 1).astype(np.float32)
            opp_tiles = (board == -1).astype(np.float32)
            ones = np.ones_like(board, dtype=np.float32)
            x = (
                torch.from_numpy(
                    np.stack([mask_np.astype(np.float32), self_tiles, opp_tiles, ones])
                )
                .unsqueeze(0)
                .to(device)
            )
            mask_t = torch.from_numpy(mask_np.reshape(1, 64)).to(device)
            return x, mask_t

        if torch.is_tensor(board):
            if board.ndim == 3:
                board = board.unsqueeze(0)
            x = board.to(device=device, dtype=torch.float32)
            mask_t = x[:, 0, :, :].reshape(x.size(0), 64) > 0.5
            return x, mask_t

        raise TypeError(f"Unsupported board type: {type(board)}")

    def forward(self, subjectiveBoard, mask=None):
        device = self._get_device()
        x, inferred_mask = self._board_to_tensor_and_mask(subjectiveBoard, device)
        y = self.net(x)

        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_t = torch.from_numpy(mask.reshape(y.shape[0], 64)).to(
                    device=device, dtype=torch.bool
                )
            else:
                mask_t = mask.to(device=device, dtype=torch.bool).reshape(y.shape[0], 64)
            y = y.masked_fill(~mask_t, -1e9)
        elif isinstance(subjectiveBoard, np.ndarray):
            y = y.masked_fill(~inferred_mask, -1e9)

        return y

    @staticmethod
    def _get_epsilon(game_idx, total_games, eps_start, eps_end):
        if total_games <= 1:
            return eps_end
        progress = min(max(game_idx / (total_games - 1), 0.0), 1.0)
        return eps_start + (eps_end - eps_start) * progress

    def _choose_opponent_move(self, board, legal, mode):
        if mode == "greedy":
            scores = np.array([(self.applyMove(board, a) == 1).sum() for a in legal])
            return int(np.random.choice(legal[scores == scores.max()]))
        if mode == "random":
            return int(np.random.choice(legal))
        raise ValueError(f"Unknown opponent mode: {mode}")

    def trainQ(
            self,
            n_games=25000,
            alpha=1e-4,
            gamma=0.99,
            eps_start=1.0,
            eps_end=0.05,
            opponent_weights=(0.5, 0.25, 0.25),  # (self, greedy, random)
    ):
        device = self._get_device()
        opt = torch.optim.Adam(self.parameters(), lr=alpha)

        modes = ["self", "greedy", "random"]
        weights = np.asarray(opponent_weights, dtype=np.float64)
        weights = weights / weights.sum()

        for game in range(n_games):
            eps = self._get_epsilon(game, n_games, eps_start, eps_end)
            mode = str(np.random.choice(modes, p=weights))
            net_turn = bool(game % 2 == 0)

            board = np.zeros((8, 8), dtype=int)
            board[3, 3] = board[4, 4] = -1
            board[3, 4] = board[4, 3] = 1
            history = []

            while True:
                legal = np.flatnonzero(self.generateMask(board))

                if mode != "self" and not net_turn:
                    a = self._choose_opponent_move(board, legal, mode)
                elif np.random.rand() < eps:
                    a = int(np.random.choice(legal))
                else:
                    with torch.no_grad():
                        a = int(self(board).argmax())

                new = self.applyMove(board, a)
                if self.generateMask(-new).any():
                    nxt, sign, done, r = -new, -1.0, False, 0.0
                    net_turn = not net_turn
                elif self.generateMask(new).any():
                    nxt, sign, done, r = new, 1.0, False, 0.0
                else:
                    nxt, sign, done = -new, -1.0, True
                    r = float(np.sign((new == 1).sum() - (new == -1).sum()))

                history.append((board, a, r, nxt, sign, done))
                if done:
                    break
                board = nxt

            loss = torch.zeros((), device=device, dtype=torch.float32)
            for b, a, r, nxt, sign, done in history:
                q = self(b)[0, a]
                with torch.no_grad():
                    y = r if done else r + sign * gamma * self(nxt).max().item()
                    y_t = torch.tensor(y, device=device, dtype=torch.float32)
                loss = loss + 0.5 * (q - y_t) ** 2

            loss = loss / len(history)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if (game + 1) % 100 == 0:
                print(
                    f"{game + 1}/{n_games} games | mode={mode:6s} | "
                    f"eps={eps:.3f} | loss={loss.item():.6f}"
                )

    def testVsGreedy(self):
        net_turn = bool(np.random.choice([True, False]))
        board = np.zeros((8, 8), dtype=int)
        board[3, 3] = board[4, 4] = -1
        board[3, 4] = board[4, 3] = 1

        while True:
            legal = np.flatnonzero(self.generateMask(board))

            if not net_turn:
                scores = np.array([(self.applyMove(board, a) == 1).sum() for a in legal])
                a = int(np.random.choice(legal[scores == scores.max()]))
            else:
                with torch.no_grad():
                    a = int(self(board).argmax())

            new = self.applyMove(board, a)
            if self.generateMask(-new).any():
                board = -new
                net_turn = not net_turn
            elif self.generateMask(new).any():
                board = new
            else:
                r = int(np.sign((new == 1).sum() - (new == -1).sum()))
                return r if net_turn else -r

    def testCycle(self, cycles):
        wins = 0
        losses = 0
        ties = 0

        for _ in range(cycles):
            result = self.testVsGreedy()
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                ties += 1

        print(f"{wins} wins, {losses} losses, {ties} ties. wr = {wins / cycles * 100:.2f}%")


if __name__ == "__main__":

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Built CUDA version:", torch.version.cuda)
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = NeuralQ().to(device)
    model.trainQ()
    model.testCycle(1000)


def terminal_reward(board, p):
    """TODO 1: final result from player p's point of view: +1.0 win, 0.0 draw, -1.0 loss.
    board is a tuple of +1 (x) / -1 (o) / 0 (empty). p is +1 or -1. Return a float."""

    score = p * sum(board)

    if score > 0:
        return 1.0
    elif score < 0:
        return -1.0
    else:
        return 0.0

    raise NotImplementedError('TODO 1: terminal_reward')


def q_target(reward, next_qs, same_player, gamma):
    """TODO 2: TD target for Q(s,a), from the point of view of the player who chose a.
    reward:      immediate reward to that player (the final result if s' is terminal, else 0).
    next_qs:     list of Q(s',a') values, from the point of view of whoever moves in s',
                 or None if s' is terminal.
    same_player: True if the player to move in s' is the one who chose a (opponent passed).
    gamma:       discount factor.
    Remember: Q values are always from the mover's perspective, and the game is zero-sum."""

    if next_qs is None:
        return reward

    best = max(next_qs)

    if same_player:
        return reward + gamma * best
    else:
        return reward - gamma * best

    raise NotImplementedError('TODO 2: q_target')


def epsilon_greedy(evals, eps):
    """TODO 3: epsilon-greedy action selection.
    evals: list of (q_value, handle) pairs, one per legal move. Return an INDEX into evals.
    With probability eps pick uniformly at random; otherwise pick a highest-Q index,
    breaking ties uniformly at random (ties are common early on when everything is 0)."""

    z = random.uniform(0, 1)

    if z < eps:
        return random.choice(range(len(evals)))

    qs = [eval[0] for eval in evals]
    best = max(qs)

    indices = [i for i, _ in enumerate(evals) if qs[i] == best]

    return random.choice(indices)

    raise NotImplementedError('TODO 3: epsilon_greedy')


# ---------------------------------------------------------------- training
def train(game, agent, episodes, eps_start, eps_end, gamma=1.0, eval_every=0, log=True):
    t0 = time.time()
    for ep in range(1, episodes + 1):
        eps = eps_start + (eps_end - eps_start) * ep / episodes
        b, p = game.initial(), X
        moves = game.legal(b, p)
        evals = agent.evaluate(b, p, moves)
        while True:
            k = epsilon_greedy(evals, eps)
            q, handle = evals[k]
            b2, p2, moves2 = game.step(b, p, moves[k])
            if p2 is None:
                evals2 = None
                target = q_target(terminal_reward(b2, p), None, False, gamma)
            else:
                evals2 = agent.evaluate(b2, p2, moves2)
                target = q_target(0.0, [v for v, _ in evals2], p2 == p, gamma)
            agent.update(handle, q, target)
            if p2 is None:
                break
            # Q values for s' were computed before the update; for tabular
            # s' != s so they are still exact, for linear they are slightly stale.
            b, p, moves, evals = b2, p2, moves2, evals2
        if log and eval_every and ep % eval_every == 0:
            r = evaluate(game, agent, random_player, 200)
            gr = evaluate(game, agent, greedy_player, 200)
            print(f'ep {ep:>7}  eps {eps:.3f}  vs random W/D/L {r}  vs greedy {gr}  '
                  f'{agent.stats()}  {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- opponents and evaluation
def random_player(game, b, p, moves):
    return random.choice(moves)


def greedy_player(game, b, p, moves):
    """Maximize discs flipped this move."""
    return max(moves, key=lambda a: (len(game._flips(b, p, a)), random.random()))


def agent_player(agent):
    def f(game, b, p, moves):
        evals = agent.evaluate(b, p, moves)
        return moves[epsilon_greedy(evals, 0.0)]

    return f


def play_game(game, px, po):
    b, p = game.initial(), X
    moves = game.legal(b, p)
    while p is not None:
        a = (px if p == X else po)(game, b, p, moves)
        b, p, moves = game.step(b, p, a)
    return sign(sum(b))


def evaluate(game, agent, opponent, games):
    me = agent_player(agent)
    w = d = l = 0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(game, me, opponent)
        else:
            r = -play_game(game, opponent, me)
        w += r > 0
        d += r == 0
        l += r < 0
    return f'{w}/{d}/{l}'


def human_player(game, b, p, moves):
    print(game.show(b, moves))
    while True:
        s = input(f'{SYM[p]} to move (e.g. c4): ').strip().lower()
        try:
            i = (int(s[1:]) - 1) * game.n + 'abcdefgh'.index(s[0])
            if i in moves:
                return i
        except (ValueError, IndexError):
            pass
        print('illegal')


# ---------------------------------------------------------------- string-board interface
def best_move_from_string(agent, board_str, token):
    """board_str: n*n chars of '.xo' (row-major, like the lab format). token: 'x' or 'o'.
    Returns the index of the agent's greedy move, or -1 if none."""
    b = tuple({'x': X, 'o': O}.get(ch, 0) for ch in board_str.lower())
    p = X if token.lower() == 'x' else O
    moves = agent.g.legal(b, p)
    return agent_player(agent)(agent.g, b, p, moves) if moves else -1


class Strategy:
    """Drop-in for the lab server: board is a 64-char string, player is 'x'/'o'."""
    logging = True
    WEIGHTS = 'q_linear_8.json'

    def __init__(self):
        self.agent = LinearQ(Othello(8))
        self.agent.load(self.WEIGHTS)

    def best_strategy(self, board, player, best_move, running):
        if running.value:
            best_move.value = best_move_from_string(self.agent, board, player)


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'eval', 'play'])
    ap.add_argument('--agent', choices=['tabular', 'linear'], default='linear')
    ap.add_argument('--size', type=int, default=8)
    ap.add_argument('--episodes', type=int, default=20000)
    ap.add_argument('--alpha', type=float)
    ap.add_argument('--eps-start', type=float, default=0.3)
    ap.add_argument('--eps-end', type=float, default=0.02)
    ap.add_argument('--eval-every', type=int, default=0)
    ap.add_argument('--games', type=int, default=500)
    ap.add_argument('--load')
    ap.add_argument('--save')
    ap.add_argument('--human', choices=['x', 'o'], default='x')
    ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    if a.seed is not None:
        random.seed(a.seed)

    game = Othello(a.size)
    if a.agent == 'tabular':
        agent = TabularQ(game, a.alpha or 0.2)
        ext = 'pkl'
    else:
        agent = LinearQ(game, a.alpha or 0.01)
        ext = 'json'
    if a.load:
        agent.load(a.load)

    if a.cmd == 'train':
        ee = a.eval_every or max(1, a.episodes // 10)
        train(game, agent, a.episodes, a.eps_start, a.eps_end, eval_every=ee)
        path = a.save or f'q_{a.agent}_{a.size}.{ext}'
        agent.save(path)
        print('saved', path)
    elif a.cmd == 'eval':
        print('vs random W/D/L', evaluate(game, agent, random_player, a.games))
        print('vs greedy W/D/L', evaluate(game, agent, greedy_player, a.games))
    else:
        me = agent_player(agent)
        hp = human_player
        r = play_game(game, hp, me) if a.human == 'x' else play_game(game, me, hp)
        print({1: 'x wins', -1: 'o wins', 0: 'draw'}[r])


if __name__ == '__main__':
    main()

"""

Questions:

1:
Zero sum -> opponents gain is your loss
opponents gain: gamma* max across a( Q(s',a'))
therefore your reward is r (you get this instantly) - gamma* max across a( Q(s',a')) (what your oppnent gets)

2:
4x4 stores 4491 wr = 174/200 , 6x6 stores  712754 wr = 84/200
tabular loses in 6x6 but not 4x4 because it can memorize a significant portion of the states
 (and likely discover winning ones from epsilon greedy explorations). It cannot do so in 6x6. 
 In 4x4 you have 3 states per square with 16 squares -> 3^16 arrangements. For 6x6, you get 3^36 which is orders of magnitude larger, meaning 
 the chance you discover the winning paths is far less likely. 

3:

WR agaisnt random (random exploration, greedy exploration): 196/200 , 157/200

Greedy takes a decent path each time, random explores more paths (off policy) and thus is more likely to find a better path given enough time.


4:

I havent heard of othello or the strategy to play it until now; for this so I'm trusting Claude:

+1.64 +0.05 +0.33 +0.15 -0.33 -0.13 +0.04 +0.05 +0.07 -0.04 +0.27 +0.17 +0.08 +0.04 
corner                    X     
Apparently, it is as expected that the corner squares are highlyl positive because they cannot be flipped.
The X squares are the worst (-0.33) because they help the opponent to take the corner .
The rest I do not understand why they're important but Claude says everything lines up with what is already known by human players. 


5.   

 Imagine a surface  above a space with each parameter as a dimmension. Gradient descent finds a valley and tries to get as close to the bottom of it as possible.
 Higher lr gets near the bottom of the well it finds faster but doesnt get to the bottom compeltely because the steps are too large, it just goes around it .
  Lower lr gets to the bottom slower but the smaller step sizes means it plateaus at a lower loss. Given that both were roughly equal at  30k episodes, we'd need more time to observe this.  


6. 
It made no difference (this is probably an error but neither me nor claude can tell where). It probably matters because disks are flippable 
(they may actually be strategically bad  beyond the next move). In contrast, optionality is never a negative (whenever you add an option, worse case scenario, you can just  ignore it)

note: I've checked the code, made sure to delete the weights from previous training, and asked claude so I reall dk whats going on here.  
the only difference is a small improvement in episode 3000 when you remove it. 

 the weights at each stage for opponent optionality were 0.028,0.032, and 0.105 ,so perhaps its not the most significant thing.

"""


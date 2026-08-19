"""Build the checked-in research notebooks.

Keeping notebook sources here makes large JSON artifacts reproducible and keeps
the mathematical narrative reviewable in ordinary diffs. Run this file from the
repository root after editing a cell below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--check",
    action="store_true",
    help="Fail if a checked-in notebook differs from the deterministic build.",
)
CHECK = parser.parse_args().check


def _source(text: str) -> list[str]:
    normalized = dedent(text).strip("\n") + "\n"
    return normalized.splitlines(keepends=True)


def md(text: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


def write_notebook(name: str, cells: list[dict[str, Any]]) -> None:
    for index, cell in enumerate(cells):
        identity = f"{name}:{index}:{''.join(cell['source'])}".encode()
        cell["id"] = hashlib.sha1(identity, usedforsecurity=False).hexdigest()[:8]
    payload = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (rl-lab)",
                "language": "python",
                "name": "rl-lab",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    rendered = json.dumps(payload, indent=1) + "\n"
    path = NOTEBOOKS / name
    if CHECK:
        if not path.exists() or path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{path} is stale; run scripts/build_notebooks.py")
    else:
        NOTEBOOKS.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")


primer = [
    md(r"""
    # Reinforcement learning from Bellman operators to entropy regularization

    This notebook is an executable mathematical refresher, not an API survey.  We
    begin with exact finite-state operators, introduce sampling one approximation
    at a time, and end with readable PyTorch implementations of DQN, REINFORCE,
    advantage actor--critic, DDPG, and SAC.

    **Execution modes.** `QUICK=True` gives a CPU-friendly structural smoke run.
    Set it to `False` for plots with lower Monte Carlo error. Deep-control results
    are intentionally diagnostics, not benchmark claims.
    """),
    code("""
    from __future__ import annotations

    from collections import deque
    from dataclasses import dataclass
    import math
    import random

    import gymnasium as gym
    import matplotlib.pyplot as plt
    import numpy as np

    SEED = 7
    QUICK = True
    rng = np.random.default_rng(SEED)
    np.set_printoptions(precision=3, suppress=True)
    available_styles = set(plt.style.available)
    plot_style = next(
        (
            style
            for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot")
            if style in available_styles
        ),
        "default",
    )
    plt.style.use(plot_style)
    """),
    md(r"""
    ## 1. Finite Markov decision processes

    A discounted MDP is a tuple

    $$\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma),$$

    with transition kernel $P(s'\mid s,a)$, conditional expected reward
    $R(s,a,s')=\mathbb E[R_{t+1}\mid S_t=s,A_t=a,S_{t+1}=s']$, and
    $\gamma\in[0,1)$.  A (stationary Markov) policy $\pi(a\mid s)$ induces

    $$G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1},\qquad
      V^\pi(s)=\mathbb E_\pi[G_t\mid S_t=s],\qquad
      Q^\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a].$$

    The Bellman expectation operator is

    $$({\cal T}^{\pi}V)(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)
       [R(s,a,s')+\gamma V(s')],$$

    while $({\cal T}^*V)(s)=\max_a\sum_{s'}P(s'\mid s,a)
    [R(s,a,s')+\gamma V(s')]$. Equivalently, the action-value equations are

    $$Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)\left[R(s,a,s')+
      \gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\right],$$
    $$Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\left[R(s,a,s')+
      \gamma\max_{a'}Q^*(s',a')\right].$$

    Both value operators are $\gamma$-contractions in the sup norm
    for discounted finite MDPs. Episodic problems instead obtain finiteness from
    absorption; continuing problems need discounting or an average-reward
    formulation.
    """),
    code("""
    # Four states, two actions; state 3 is absorbing. The risky branch via state 2
    # can pay more but sometimes falls back to state 1.
    nS, nA, gamma = 4, 2, 0.92
    P = np.zeros((nS, nA, nS))
    R = np.zeros_like(P)
    P[0, 0, 1] = 1.0
    P[0, 1, 2], P[0, 1, 1] = 0.75, 0.25
    P[1, 0, 3], P[1, 1, 1] = 1.0, 1.0
    P[2, 0, 3], P[2, 1, 0] = 1.0, 1.0
    P[3, :, 3] = 1.0
    R[1, 0, 3] = 1.0
    R[2, 0, 3] = 2.2
    R[1, 1, 1] = -0.08
    terminal = np.array([False, False, False, True])
    assert np.allclose(P.sum(axis=-1), 1.0)
    """),
    md(r"""
    ## 2. Dynamic programming: exact expectations

    Policy evaluation repeatedly applies ${\cal T}^{\pi}$. Policy iteration
    alternates exact/approximate evaluation with greedy improvement. Value
    iteration applies ${\cal T}^*$ directly. The terminal mask below enforces
    zero continuation after absorption, making the episodic convention explicit.
    """),
    code("""
    def q_from_v(P, R, V, gamma, terminal):
        continuation = gamma * V * (~terminal)
        return np.sum(P * (R + continuation[None, None, :]), axis=2)


    def policy_evaluation(P, R, policy, gamma, terminal, tol=1e-12):
        V = np.zeros(P.shape[0])
        residuals = []
        while True:
            Q = q_from_v(P, R, V, gamma, terminal)
            updated = np.sum(policy * Q, axis=1)
            updated[terminal] = 0.0
            residuals.append(np.max(np.abs(updated - V)))
            V = updated
            if residuals[-1] < tol:
                return V, np.asarray(residuals)


    def policy_iteration(P, R, gamma, terminal):
        policy = np.full(P.shape[:2], 1 / P.shape[1])
        changes = []
        while True:
            V, _ = policy_evaluation(P, R, policy, gamma, terminal)
            greedy = np.argmax(q_from_v(P, R, V, gamma, terminal), axis=1)
            improved = np.eye(P.shape[1])[greedy]
            improved[terminal] = 1 / P.shape[1]
            changes.append(np.mean(np.argmax(policy, axis=1) != greedy))
            if np.array_equal(np.argmax(policy, axis=1)[~terminal], greedy[~terminal]):
                return V, policy, np.asarray(changes)
            policy = improved


    def value_iteration(P, R, gamma, terminal, tol=1e-12):
        V = np.zeros(P.shape[0])
        residuals = []
        while True:
            Q = q_from_v(P, R, V, gamma, terminal)
            updated = Q.max(axis=1)
            updated[terminal] = 0.0
            residuals.append(np.max(np.abs(updated - V)))
            V = updated
            if residuals[-1] < tol:
                Q = q_from_v(P, R, V, gamma, terminal)
                return V, Q, np.argmax(Q, axis=1), np.asarray(residuals)


    uniform = np.full((nS, nA), 0.5)
    V_uniform, eval_residuals = policy_evaluation(P, R, uniform, gamma, terminal)
    V_pi, pi, policy_changes = policy_iteration(P, R, gamma, terminal)
    V_star, Q_star, greedy, vi_residuals = value_iteration(P, R, gamma, terminal)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].bar(np.arange(nS) - .18, V_uniform, .36, label=r"$V^{uniform}$")
    ax[0].bar(np.arange(nS) + .18, V_star, .36, label=r"$V^*$")
    ax[0].set(xlabel="state", ylabel="value", title=f"Greedy actions: {greedy}")
    ax[0].legend()
    ax[1].semilogy(eval_residuals, label="policy evaluation")
    ax[1].semilogy(vi_residuals, label="value iteration")
    ax[1].set(xlabel="sweep", ylabel=r"$\\|V_{k+1}-V_k\\|_\\infty$",
              title="Bellman residual")
    ax[1].legend()
    plt.tight_layout()
    """),
    md(r"""
    ## 3. Monte Carlo: replace expectations by complete returns

    First-visit prediction averages $G_t$ only at the first occurrence of a state
    in each episode. Exploring-starts control (used here for clarity) estimates
    $Q$ and greedifies after every return:

    $$Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\frac{1}{N(S_t,A_t)}
      [G_t-Q(S_t,A_t)].$$

    Ordinary on-policy $\epsilon$-soft control is usually easier to deploy than
    exploring starts. Off-policy Monte Carlo requires importance ratios, whose
    heavy tails can dominate finite-sample behavior.
    """),
    code("""
    def sample_transition(s, a, rng):
        next_s = rng.choice(nS, p=P[s, a])
        return next_s, R[s, a, next_s], bool(terminal[next_s])


    def generate_episode(policy, rng, start=(0, None), max_steps=100):
        s, forced_action = start
        episode = []
        for t in range(max_steps):
            a = forced_action if t == 0 and forced_action is not None else rng.choice(nA, p=policy[s])
            next_s, reward, done = sample_transition(s, a, rng)
            episode.append((s, a, reward))
            s = next_s
            if done:
                break
        return episode


    def first_visit_mc_prediction(policy, episodes, rng):
        values, counts = np.zeros(nS), np.zeros(nS)
        history = []
        for _ in range(episodes):
            trajectory = generate_episode(policy, rng)
            G, returns = 0.0, []
            for s, _, reward in reversed(trajectory):
                G = reward + gamma * G
                returns.append((s, G))
            visited = set()
            for s, G in reversed(returns):
                if s not in visited:
                    visited.add(s)
                    counts[s] += 1
                    values[s] += (G - values[s]) / counts[s]
            history.append(values.copy())
        return values, np.asarray(history)


    def mc_control(episodes, rng, epsilon=0.8):
        Q, counts = np.zeros((nS, nA)), np.zeros((nS, nA))
        errors = []
        for episode_index in range(episodes):
            s0, a0 = int(rng.integers(0, nS - 1)), int(rng.integers(nA))
            epsilon_t = epsilon / np.sqrt(episode_index + 1)
            greedy_actions = Q.argmax(1)
            policy = np.full((nS, nA), epsilon_t / nA)
            policy[np.arange(nS), greedy_actions] += 1 - epsilon_t
            trajectory = generate_episode(policy, rng, start=(s0, a0))
            returns, G = np.zeros(len(trajectory)), 0.0
            for t in reversed(range(len(trajectory))):
                _, _, reward = trajectory[t]
                G = reward + gamma * G
                returns[t] = G
            visited = set()
            for (s, a, _), G in zip(trajectory, returns, strict=True):
                if (s, a) not in visited:
                    visited.add((s, a))
                    counts[s, a] += 1
                    Q[s, a] += (G - Q[s, a]) / counts[s, a]
            errors.append(np.max(np.abs(Q[~terminal] - Q_star[~terminal])))
        return Q, np.asarray(errors)


    episodes = 1_500 if QUICK else 20_000
    V_mc, V_mc_path = first_visit_mc_prediction(uniform, episodes, rng)
    Q_mc, Q_mc_error = mc_control(episodes, rng)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(np.abs(V_mc_path[:, 0] - V_uniform[0]))
    ax[0].set(xlabel="episode", ylabel="absolute error", title="First-visit MC at state 0")
    ax[1].plot(Q_mc_error)
    ax[1].set(xlabel="episode", ylabel=r"$\\|Q-Q^*\\|_\\infty$", title="MC control")
    plt.tight_layout()
    """),
    md(r"""
    ## 4. Temporal differences: bootstrap before the episode ends

    TD(0) uses

    $$\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t),\qquad
      V(S_t)\leftarrow V(S_t)+\alpha\delta_t.$$

    The $n$-step target interpolates between one-step TD and Monte Carlo:
    $G_{t:t+n}=\sum_{k=0}^{n-1}\gamma^kR_{t+k+1}+\gamma^nV(S_{t+n})$.
    For action values, the only difference among several canonical methods is the
    bootstrap target:

    | method | target after $R_{t+1}$ |
    |---|---|
    | SARSA | $\gamma Q(S_{t+1},A_{t+1})$ |
    | Expected SARSA | $\gamma\sum_a\pi(a\mid S_{t+1})Q(S_{t+1},a)$ |
    | Q-learning | $\gamma\max_a Q(S_{t+1},a)$ |
    | Double Q | evaluate one estimator's argmax with the other estimator |

    SARSA is on-policy and prices exploratory actions into its values. Q-learning
    is off-policy. Double Q-learning attacks the maximization bias caused by
    selecting and evaluating an action with the same noisy estimator.
    """),
    code("""
    def epsilon_probs(q_row, epsilon):
        probs = np.full(len(q_row), epsilon / len(q_row))
        probs[np.flatnonzero(q_row == q_row.max())] += (1 - epsilon) / np.sum(q_row == q_row.max())
        return probs


    def td_prediction(policy, episodes, alpha, n_step=1):
        V = np.zeros(nS)
        errors = []
        local_rng = np.random.default_rng(SEED + n_step)
        for _ in range(episodes):
            trajectory = generate_episode(policy, local_rng)
            states = [x[0] for x in trajectory]
            rewards = [x[2] for x in trajectory]
            for t, s in enumerate(states):
                horizon = min(t + n_step, len(states))
                target = sum(gamma ** k * rewards[t + k] for k in range(horizon - t))
                if horizon < len(states):
                    target += gamma ** n_step * V[states[horizon]]
                V[s] += alpha * (target - V[s])
            errors.append(abs(V[0] - V_uniform[0]))
        return V, np.asarray(errors)


    def control(algorithm, episodes, alpha=.12, epsilon=.12):
        Q = np.zeros((nS, nA))
        algorithm_seed = {"sarsa": 11, "expected_sarsa": 12, "q_learning": 13}[algorithm]
        local_rng = np.random.default_rng(SEED + algorithm_seed)
        errors, td_errors = [], []
        for _ in range(episodes):
            s = 0
            a = local_rng.choice(nA, p=epsilon_probs(Q[s], epsilon))
            for _ in range(100):
                next_s, reward, done = sample_transition(s, a, local_rng)
                next_probs = epsilon_probs(Q[next_s], epsilon)
                next_a = local_rng.choice(nA, p=next_probs)
                if done:
                    target = reward
                elif algorithm == "sarsa":
                    target = reward + gamma * Q[next_s, next_a]
                elif algorithm == "expected_sarsa":
                    target = reward + gamma * np.dot(next_probs, Q[next_s])
                else:
                    target = reward + gamma * Q[next_s].max()
                delta = target - Q[s, a]
                Q[s, a] += alpha * delta
                td_errors.append(delta)
                s, a = next_s, next_a
                if done:
                    break
            errors.append(np.max(np.abs(Q[~terminal] - Q_star[~terminal])))
        return Q, np.asarray(errors), np.asarray(td_errors)


    def double_q_learning(episodes, alpha=.12, epsilon=.12):
        QA, QB = np.zeros((nS, nA)), np.zeros((nS, nA))
        local_rng = np.random.default_rng(SEED + 91)
        errors, td_errors = [], []
        for _ in range(episodes):
            s = 0
            for _ in range(100):
                a = local_rng.choice(nA, p=epsilon_probs(QA[s] + QB[s], epsilon))
                next_s, reward, done = sample_transition(s, a, local_rng)
                first = bool(local_rng.integers(2))
                update, evaluate = (QA, QB) if first else (QB, QA)
                target = reward if done else reward + gamma * evaluate[next_s, np.argmax(update[next_s])]
                delta = target - update[s, a]
                update[s, a] += alpha * delta
                td_errors.append(delta)
                s = next_s
                if done:
                    break
            Q = (QA + QB) / 2
            errors.append(np.max(np.abs(Q[~terminal] - Q_star[~terminal])))
        return (QA + QB) / 2, np.asarray(errors), np.asarray(td_errors)


    td_episodes = 1_000 if QUICK else 10_000
    _, td0_error = td_prediction(uniform, td_episodes, alpha=.08, n_step=1)
    _, td4_error = td_prediction(uniform, td_episodes, alpha=.08, n_step=4)
    results = {name: control(name, td_episodes) for name in ("sarsa", "expected_sarsa", "q_learning")}
    results["double_q"] = double_q_learning(td_episodes)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    ax[0].plot(td0_error, alpha=.8, label="TD(0)")
    ax[0].plot(td4_error, alpha=.8, label="4-step TD")
    ax[0].set(title="Prediction", xlabel="episode", ylabel=r"$|V(0)-V^\\pi(0)|$")
    ax[0].legend()
    for name, (_, error, _) in results.items():
        ax[1].plot(error, alpha=.8, label=name)
    ax[1].set(title="Control", xlabel="episode", ylabel=r"$\\|Q-Q^*\\|_\\infty$")
    ax[1].legend(fontsize=8)
    for name, (_, _, deltas) in results.items():
        ax[2].hist(deltas[-500:], bins=30, alpha=.35, density=True, label=name)
    ax[2].set(title="Late-training TD errors", xlabel=r"$\\delta_t$", ylabel="density")
    ax[2].legend(fontsize=8)
    plt.tight_layout()
    """),
    md(r"""
    ### Failure modes and neighboring algorithms

    Constant step sizes preserve adaptation but leave a noise floor; Robbins--Monro
    schedules converge only under adequate visitation and stationary assumptions.
    Bootstrapping, off-policy sampling, and function approximation form the
    *deadly triad*. Eligibility traces ($\operatorname{TD}(\lambda)$, SARSA($\lambda$))
    mix all $n$-step targets. Expected SARSA removes action-sampling variance;
    Q-learning removes behavior-policy bias but can amplify max bias. Inspect TD
    error distributions and visits—not only mean return.
    """),
    md(r"""
    ## 5. Function approximation

    Replace a table with $Q_\theta(s,a)$. Semi-gradient TD minimizes a moving
    target, e.g.

    $$L(\theta)=\mathbb E[(R+\gamma\max_{a'}Q_{\bar\theta}(S',a')
      -Q_\theta(S,A))^2].$$

    A target network $\bar\theta$ slows target drift; replay reduces serial
    correlation and reuses data. Neither makes the objective stationary. The code
    below is a compact DQN with explicit replay, target synchronization, gradient
    clipping, and seeded environments. Install the repository's `notebooks` extra
    from a terminal rather than installing PyTorch inside a notebook cell; the
    declared platform markers keep PyTorch and NumPy binary-compatible.
    """),
    code("""
    import torch
    from torch import nn
    from torch.distributions import Categorical, Normal
    import torch.nn.functional as F

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    class Replay:
        def __init__(self, capacity=100_000):
            self.data = deque(maxlen=capacity)

        def add(self, *transition):
            self.data.append(transition)

        def sample(self, batch_size, rng):
            idx = rng.choice(len(self.data), batch_size, replace=False)
            columns = list(zip(*(self.data[i] for i in idx)))
            return [np.asarray(column) for column in columns]

        def __len__(self):
            return len(self.data)


    def mlp(sizes, final=nn.Identity):
        layers = []
        for left, right in zip(sizes[:-2], sizes[1:-1]):
            layers += [nn.Linear(left, right), nn.ReLU()]
        layers += [nn.Linear(sizes[-2], sizes[-1]), final()]
        return nn.Sequential(*layers)


    def train_dqn(episodes=40):
        torch.manual_seed(SEED)
        env = gym.make("CartPole-v1")
        env.action_space.seed(SEED)
        online = mlp([4, 64, 64, 2]).to(device)
        target = mlp([4, 64, 64, 2]).to(device)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=7e-4)
        replay, local_rng = Replay(), np.random.default_rng(SEED)
        returns, losses, steps = [], [], 0
        for episode in range(episodes):
            state, _ = env.reset(seed=SEED + episode)
            total = 0.0
            for _ in range(500):
                epsilon = 0.03 + 0.97 * math.exp(-steps / 1_500)
                if local_rng.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = int(online(torch.as_tensor(state, dtype=torch.float32, device=device)).argmax())
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # TimeLimit truncation resets the rollout but is not an absorbing
                # MDP transition, so only true termination masks the target.
                replay.add(state, action, reward, next_state, terminated)
                state, total, steps = next_state, total + reward, steps + 1
                if len(replay) >= 64:
                    s, a, r, ns, d = replay.sample(64, local_rng)
                    st = torch.as_tensor(s, dtype=torch.float32, device=device)
                    at = torch.as_tensor(a, dtype=torch.long, device=device)
                    rt = torch.as_tensor(r, dtype=torch.float32, device=device)
                    nst = torch.as_tensor(ns, dtype=torch.float32, device=device)
                    dt = torch.as_tensor(d, dtype=torch.float32, device=device)
                    prediction = online(st).gather(1, at[:, None]).squeeze(1)
                    with torch.no_grad():
                        y = rt + .99 * (1 - dt) * target(nst).max(1).values
                    loss = F.smooth_l1_loss(prediction, y)
                    optimizer.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    optimizer.step(); losses.append(float(loss))
                if steps % 250 == 0:
                    target.load_state_dict(online.state_dict())
                if done:
                    break
            returns.append(total)
        env.close()
        return np.asarray(returns), np.asarray(losses)


    dqn_returns, dqn_losses = train_dqn(30 if QUICK else 250)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.3))
    ax[0].plot(dqn_returns); ax[0].set(title="DQN: episodic return", xlabel="episode")
    ax[1].plot(dqn_losses); ax[1].set(title="DQN: Huber loss", xlabel="gradient step")
    plt.tight_layout()
    """),
    md(r"""
    ## 6. Policy gradients and actor--critic

    The policy-gradient theorem gives

    $$\nabla_\theta J(\theta)=\mathbb E_{d^{\pi_\theta},\pi_\theta}
      [\nabla_\theta\log\pi_\theta(A\mid S)Q^{\pi_\theta}(S,A)].$$

    REINFORCE substitutes a Monte Carlo return $G_t$ and a baseline $b(S_t)$.
    Actor--critic substitutes a bootstrapped critic and uses the TD residual as a
    one-step advantage estimate. The implementation below is deliberately a
    single-environment advantage actor--critic, rather than claiming the
    synchronous parallel actors usually denoted A2C.

    The corresponding sample objectives are

    $$\Delta\theta_{REINFORCE}\propto
      \sum_t\nabla_\theta\log\pi_\theta(A_t\mid S_t)
      [G_t-b_\phi(S_t)],$$
    $$\delta_t=R_{t+1}+\gamma(1-D_{t+1})V_\phi(S_{t+1})-V_\phi(S_t),$$
    $$L_V(\phi)=\tfrac12\delta_t^2,\qquad
      L_\pi(\theta)=-\log\pi_\theta(A_t\mid S_t)\,\operatorname{stopgrad}(\delta_t)
      -\beta{\cal H}(\pi_\theta(\cdot\mid S_t)).$$
    """),
    code("""
    class DiscreteActorCritic(nn.Module):
        def __init__(self, obs_dim, n_actions):
            super().__init__()
            self.body = mlp([obs_dim, 64, 64, 64])
            self.policy = nn.Linear(64, n_actions)
            self.value = nn.Linear(64, 1)

        def forward(self, x):
            features = self.body(x)
            return self.policy(features), self.value(features).squeeze(-1)


    def train_policy_gradient(method="reinforce", episodes=40):
        torch.manual_seed(SEED + (0 if method == "reinforce" else 1))
        env = gym.make("CartPole-v1")
        env.action_space.seed(SEED)
        model = DiscreteActorCritic(4, 2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)
        episode_returns = []
        diagnostics = {"actor_loss": [], "critic_loss": [], "entropy": [], "gradient_norm": []}
        for episode in range(episodes):
            state, _ = env.reset(seed=SEED + 10_000 + episode)
            logps, values, rewards, entropies, terminations = [], [], [], [], []
            for _ in range(500):
                x = torch.as_tensor(state, dtype=torch.float32, device=device)
                logits, value = model(x)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                state, reward, terminated, truncated, _ = env.step(int(action))
                logps.append(distribution.log_prob(action)); values.append(value)
                rewards.append(reward); entropies.append(distribution.entropy())
                terminations.append(float(terminated))
                if terminated or truncated:
                    break
            raw_returns, G = [], 0.0
            for reward in reversed(rewards):
                G = reward + .99 * G; raw_returns.append(G)
            raw_returns = torch.as_tensor(raw_returns[::-1], dtype=torch.float32, device=device)
            logps, values = torch.stack(logps), torch.stack(values)
            if method == "reinforce":
                # A learned baseline lowers variance; detach keeps this unbiased.
                advantages = raw_returns - values.detach()
                critic_target = raw_returns
            else:
                with torch.no_grad():
                    final_state = torch.as_tensor(state, dtype=torch.float32, device=device)
                    _, final_value = model(final_state)
                    bootstrap = torch.zeros((), device=device) if terminated else final_value
                next_values = torch.cat([values[1:].detach(), bootstrap.reshape(1)])
                rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)
                terminated_t = torch.as_tensor(terminations, dtype=torch.float32, device=device)
                critic_target = rewards_t + .99 * (1 - terminated_t) * next_values
                advantages = critic_target - values.detach()
            normalized_advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-6
            )
            actor_loss = -(logps * normalized_advantages).mean()
            critic_loss = .5 * F.mse_loss(values, critic_target.detach())
            entropy_bonus = torch.stack(entropies).mean()
            loss = actor_loss + critic_loss - .01 * entropy_bonus
            optimizer.zero_grad(); loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            episode_returns.append(sum(rewards))
            diagnostics["actor_loss"].append(float(actor_loss))
            diagnostics["critic_loss"].append(float(critic_loss))
            diagnostics["entropy"].append(float(entropy_bonus))
            diagnostics["gradient_norm"].append(float(gradient_norm))
        env.close()
        return np.asarray(episode_returns), {
            key: np.asarray(values) for key, values in diagnostics.items()
        }


    pg_runs = {
        name: train_policy_gradient(name, 30 if QUICK else 250)
        for name in ("reinforce", "advantage_actor_critic")
    }
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.3))
    for name, (returns, diagnostics) in pg_runs.items():
        ax[0].plot(returns, label=name)
        ax[1].plot(diagnostics["actor_loss"], label=name)
        ax[2].plot(diagnostics["critic_loss"], label=f"{name}: critic")
        ax[2].plot(diagnostics["entropy"], linestyle="--", label=f"{name}: entropy")
    ax[0].set(title="Policy-gradient returns", xlabel="episode")
    ax[1].set(title="Actor loss", xlabel="episode")
    ax[2].set(title="Critic loss / policy entropy", xlabel="episode")
    for axis in ax: axis.legend()
    plt.tight_layout()
    """),
    md(r"""
    ## 7. Continuous control: DDPG and SAC

    DDPG uses a deterministic actor $\mu_\theta$ and critic $Q_\phi$:

    $$\nabla_\theta J\approx\mathbb E[\nabla_a Q_\phi(s,a)|_{a=\mu_\theta(s)}
      \nabla_\theta\mu_\theta(s)],$$
    $$y=r+\gamma Q_{\bar\phi}(s',\mu_{\bar\theta}(s')).$$

    Exploration is external action noise, so its scale and temporal structure are
    consequential. SAC instead optimizes return plus entropy,

    $$J(\pi)=\mathbb E\sum_t\gamma^t[R_{t+1}+\alpha
      {\cal H}(\pi(\cdot\mid S_t))],$$

    with a squashed Gaussian actor and clipped double critics. SAC's stochastic
    policy makes exploration part of the objective. Its sampled soft target and
    actor objective are

    $$y=r+\gamma(1-d)\left[\min_{i=1,2}Q_{\bar\phi_i}(s',a')
      -\alpha\log\pi_\theta(a'\mid s')\right],\quad a'\sim\pi_\theta,$$
    $$J_\pi(\theta)=\mathbb E_{s\sim{\cal D},a\sim\pi_\theta}
      [\alpha\log\pi_\theta(a\mid s)-\min_iQ_{\phi_i}(s,a)].$$

    These compact implementations share replay but leave every target visible.
    """),
    code("""
    class DeterministicActor(nn.Module):
        def __init__(self, obs_dim, act_dim, limit):
            super().__init__(); self.net = mlp([obs_dim, 128, 128, act_dim], nn.Tanh); self.limit = limit
        def forward(self, state): return self.limit * self.net(state)


    class ContinuousQ(nn.Module):
        def __init__(self, obs_dim, act_dim):
            super().__init__(); self.net = mlp([obs_dim + act_dim, 128, 128, 1])
        def forward(self, state, action): return self.net(torch.cat([state, action], -1)).squeeze(-1)


    def soft_update(target, source, tau):
        with torch.no_grad():
            for target_p, source_p in zip(target.parameters(), source.parameters()):
                target_p.mul_(1 - tau).add_(source_p, alpha=tau)


    def train_ddpg(total_steps=2_000):
        torch.manual_seed(SEED)
        env = gym.make("Pendulum-v1")
        env.action_space.seed(SEED)
        obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        limit = float(env.action_space.high[0])
        actor, critic = DeterministicActor(obs_dim, act_dim, limit).to(device), ContinuousQ(obs_dim, act_dim).to(device)
        actor_t, critic_t = DeterministicActor(obs_dim, act_dim, limit).to(device), ContinuousQ(obs_dim, act_dim).to(device)
        actor_t.load_state_dict(actor.state_dict()); critic_t.load_state_dict(critic.state_dict())
        actor_opt = torch.optim.Adam(actor.parameters(), 1e-3); critic_opt = torch.optim.Adam(critic.parameters(), 1e-3)
        replay, local_rng = Replay(), np.random.default_rng(SEED + 200)
        state, _ = env.reset(seed=SEED); ep_return, returns, q_losses, actor_losses = 0.0, [], [], []
        for step in range(total_steps):
            if step < 300:
                action = env.action_space.sample()
            else:
                with torch.no_grad(): action = actor(torch.as_tensor(state, dtype=torch.float32, device=device)).cpu().numpy()
                action = np.clip(action + local_rng.normal(0, .18 * limit, act_dim), -limit, limit)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.add(state, action, reward, next_state, terminated); ep_return += reward; state = next_state
            if len(replay) >= 128:
                s, a, r, ns, d = replay.sample(128, local_rng)
                st = torch.as_tensor(s, dtype=torch.float32, device=device); at = torch.as_tensor(a, dtype=torch.float32, device=device)
                rt = torch.as_tensor(r, dtype=torch.float32, device=device); nst = torch.as_tensor(ns, dtype=torch.float32, device=device)
                dt = torch.as_tensor(d, dtype=torch.float32, device=device)
                with torch.no_grad(): y = rt + .99 * (1 - dt) * critic_t(nst, actor_t(nst))
                q_loss = F.mse_loss(critic(st, at), y)
                critic_opt.zero_grad(); q_loss.backward(); critic_opt.step()
                actor_loss = -critic(st, actor(st)).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                soft_update(actor_t, actor, .005); soft_update(critic_t, critic, .005)
                q_losses.append(float(q_loss))
                actor_losses.append(float(actor_loss))
            if done:
                returns.append(ep_return); state, _ = env.reset(); ep_return = 0.0
        env.close(); return np.asarray(returns), np.asarray(q_losses), np.asarray(actor_losses)
    """),
    code("""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    class SquashedGaussianActor(nn.Module):
        def __init__(self, obs_dim, act_dim, limit):
            super().__init__(); self.body = mlp([obs_dim, 128, 128, 128]); self.mean = nn.Linear(128, act_dim)
            self.log_std = nn.Linear(128, act_dim); self.limit = limit

        def sample(self, state):
            h = self.body(state); mean = self.mean(h)
            log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX); std = log_std.exp()
            raw = Normal(mean, std).rsample(); squashed = torch.tanh(raw)
            action = self.limit * squashed
            logp = Normal(mean, std).log_prob(raw).sum(-1)
            logp -= torch.log(self.limit * (1 - squashed.pow(2)) + 1e-6).sum(-1)
            return action, logp


    def train_sac(total_steps=2_000, entropy_coefficient=.2):
        torch.manual_seed(SEED)
        env = gym.make("Pendulum-v1")
        env.action_space.seed(SEED)
        obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        limit = float(env.action_space.high[0])
        actor = SquashedGaussianActor(obs_dim, act_dim, limit).to(device)
        q1, q2 = ContinuousQ(obs_dim, act_dim).to(device), ContinuousQ(obs_dim, act_dim).to(device)
        q1_t, q2_t = ContinuousQ(obs_dim, act_dim).to(device), ContinuousQ(obs_dim, act_dim).to(device)
        q1_t.load_state_dict(q1.state_dict()); q2_t.load_state_dict(q2.state_dict())
        actor_opt = torch.optim.Adam(actor.parameters(), 3e-4)
        q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 3e-4)
        replay, local_rng = Replay(), np.random.default_rng(SEED + 300)
        state, _ = env.reset(seed=SEED); ep_return, returns, q_losses, actor_losses, entropies = 0.0, [], [], [], []
        for step in range(total_steps):
            if step < 300: action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _ = actor.sample(torch.as_tensor(state, dtype=torch.float32, device=device))
                    action = action.cpu().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.add(state, action, reward, next_state, terminated); ep_return += reward; state = next_state
            if len(replay) >= 128:
                s, a, r, ns, d = replay.sample(128, local_rng)
                st = torch.as_tensor(s, dtype=torch.float32, device=device); at = torch.as_tensor(a, dtype=torch.float32, device=device)
                rt = torch.as_tensor(r, dtype=torch.float32, device=device); nst = torch.as_tensor(ns, dtype=torch.float32, device=device)
                dt = torch.as_tensor(d, dtype=torch.float32, device=device)
                with torch.no_grad():
                    next_a, next_logp = actor.sample(nst)
                    soft_value = torch.minimum(q1_t(nst, next_a), q2_t(nst, next_a)) - entropy_coefficient * next_logp
                    y = rt + .99 * (1 - dt) * soft_value
                q_loss = F.mse_loss(q1(st, at), y) + F.mse_loss(q2(st, at), y)
                q_opt.zero_grad(); q_loss.backward(); q_opt.step()
                sampled_a, logp = actor.sample(st)
                actor_loss = (entropy_coefficient * logp - torch.minimum(q1(st, sampled_a), q2(st, sampled_a))).mean()
                actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()
                soft_update(q1_t, q1, .005); soft_update(q2_t, q2, .005)
                q_losses.append(float(q_loss))
                actor_losses.append(float(actor_loss))
                entropies.append(float(-logp.mean()))
            if done:
                returns.append(ep_return); state, _ = env.reset(); ep_return = 0.0
        env.close()
        return (
            np.asarray(returns),
            np.asarray(q_losses),
            np.asarray(actor_losses),
            np.asarray(entropies),
        )


    continuous_steps = 1_200 if QUICK else 40_000
    ddpg_returns, ddpg_losses, ddpg_actor_losses = train_ddpg(continuous_steps)
    sac_returns, sac_losses, sac_actor_losses, sac_entropies = train_sac(continuous_steps)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.3))
    ax[0].plot(ddpg_returns, label="DDPG"); ax[0].plot(sac_returns, label="SAC")
    ax[0].set(title="Pendulum return", xlabel="episode"); ax[0].legend()
    ax[1].plot(ddpg_losses, alpha=.7, label="DDPG critic")
    ax[1].plot(sac_losses, alpha=.7, label="SAC twin critics")
    ax[1].set(title="Critic loss", xlabel="gradient step", yscale="log"); ax[1].legend()
    ax[2].plot(ddpg_actor_losses, alpha=.7, label="DDPG actor")
    ax[2].plot(sac_actor_losses, alpha=.7, label="SAC actor")
    ax[2].plot(sac_entropies, alpha=.6, label="SAC entropy")
    ax[2].set(title="Actor objectives / entropy", xlabel="gradient step"); ax[2].legend()
    plt.tight_layout()
    """),
    md(r"""
    ## 8. One view of the algorithm family

    Every method above decides (i) which distribution supplies data, (ii) which
    operator defines a target, and (iii) how that target is projected into a
    representable function class.

    - DP knows $P,R$ and applies the operator exactly.
    - Monte Carlo samples an unbiased full return with potentially high variance.
    - TD samples a reward/transition and bootstraps, trading variance for bias and
      target coupling.
    - DQN stabilizes approximate off-policy optimality updates with replay and a
      delayed target.
    - REINFORCE differentiates the trajectory distribution; actor--critic replaces
      its return with a learned control variate/advantage.
    - DDPG differentiates through a deterministic critic; SAC regularizes the
      control problem with entropy and retains a stochastic actor.

    | algorithm | failure modes worth diagnosing |
    |---|---|
    | policy/value iteration | model error, state explosion, improper undiscounted policies |
    | Monte Carlo | long-horizon variance, importance-weight tails, insufficient exploring starts |
    | SARSA / Expected SARSA | persistent behavior-policy bias, sensitivity to step-size and exploration decay |
    | Q-learning / Double Q | max bias, rare-action undercoverage, deadly-triad divergence after approximation |
    | DQN | replay support mismatch, target drift, exploding Q scale, correlated evaluation seeds |
    | REINFORCE | extreme gradient variance, weak baseline, premature entropy collapse |
    | advantage actor--critic | critic bias leaking into the actor, bootstrapping at truncations, loss-scale imbalance |
    | DDPG | brittle external action noise, critic extrapolation, actor saturation |
    | SAC | entropy-temperature mismatch, log-probability correction errors, critic underestimation |

    **Diagnostics to retain.** Learning curves alone conceal support mismatch,
    extrapolation, and unstable targets. Record visitation, policy entropy, target
    and TD-error distributions, gradient norms, Q scale, Bellman residuals, and
    seed-level outcome distributions. The remaining notebooks make those objects
    first-class experimental data.
    """),
]


maze_lab = [
    md(r"""
    # The stochastic maze as an experimental laboratory

    The maze is useful because its latent stochastic process can be made exactly
    finite without becoming trivial. We will isolate action noise, spatial
    heterogeneity, several wall processes, hazards, partial observation, and
    temporal nonstationarity. Whenever the assumptions permit, we construct the
    exact kernel and solve the corresponding MDP.

    The notebook uses the convention north/east/south/west = 0/1/2/3. A wall is
    an *edge* between adjacent cells; a blocked cell is removed from navigation.
    """),
    code("""
    from __future__ import annotations

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from rllab.environments import (
        EventWall,
        ExactModelUnavailable,
        Hazard,
        IndependentWall,
        MarkovWall,
        MazeAction,
        MovingHazard,
        NonstationarityConfig,
        ScheduledWall,
        StochasticMazeEnv,
    )
    from rllab.theory import value_iteration
    from rllab.visualization import (
        animate_topology,
        plot_maze,
        plot_policy,
        plot_transition_noise,
    )

    SEED = 17
    available_styles = set(plt.style.available)
    plot_style = next(
        (
            style
            for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot")
            if style in available_styles
        ),
        "default",
    )
    plt.style.use(plot_style)
    """),
    md(r"""
    ## 1. Deterministic baseline

    With reliability one, no dynamic walls, deterministic rewards, and full state
    observation, the environment is a conventional finite MDP. `reset(seed=...)`
    seeds every stochastic component; `info` still reports the latent wall,
    hazard, and regime state so an experiment recorder can retain it.
    """),
    code("""
    deterministic = StochasticMazeEnv(
        shape=(5, 7),
        start=(4, 0),
        goals={(0, 6): 8.0},
        blocked_cells={(1, 1), (1, 2), (3, 4), (3, 5)},
        static_walls={((2, 2), (2, 3)), ((1, 5), (2, 5))},
        action_reliability=1.0,
        step_reward=-0.04,
        max_episode_steps=150,
    )
    observation, reset_info = deterministic.reset(seed=SEED)
    print("initial observation:", observation)
    print("initial coordinate:", deterministic.index_to_state[int(observation)])
    print("diagnostic fields:", sorted(reset_info))
    plot_maze(deterministic, title="Deterministic topology")
    plt.show()
    """),
    md(r"""
    ## 2. Stochastic action channel

    If action $a$ is intended, the realized action is sampled from a configurable
    channel. For the common relative-slip parameterization,

    $$P(A^{real}=a\mid s,a)=p(s,a),$$

    and the remaining mass is assigned to left, right, stay, or backward motion.
    Collision with a boundary, cell, or active edge wall maps any attempted move
    back to the current state. The exact kernel therefore combines action-channel
    randomness with topology-induced aggregation.
    """),
    code("""
    slippery = StochasticMazeEnv(
        shape=(4, 5),
        start=(3, 0),
        goals={(0, 4): 5.0},
        action_reliability=0.70,
        slip_weights={"left": 0.35, "right": 0.35, "stay": 0.20, "backward": 0.10},
        step_reward=-0.03,
        max_episode_steps=100,
    )
    P, R = slippery.transition_reward_kernels()
    start = slippery.state_to_index[(3, 0)]
    north = int(MazeAction.NORTH)
    support = np.flatnonzero(P[start, north] > 0)
    labels = [str(slippery.index_to_state[int(state)]) for state in support]
    probabilities = P[start, north, support]

    fig, ax = plt.subplots(figsize=(7, 3.3))
    ax.bar(labels, probabilities)
    ax.set(
        title="Exact next-state distribution for intended north",
        xlabel="next coordinate",
        ylabel="probability",
        ylim=(0, 1),
    )
    for x, probability in enumerate(probabilities):
        ax.text(x, probability + 0.025, f"{probability:.2f}", ha="center")
    plt.show()
    assert np.allclose(P.sum(axis=2), 1.0)
    """),
    md(r"""
    ## 3. Spatially heterogeneous transition noise

    Reliability may be scalar, state-dependent, or state--action-dependent. A
    scalar is the baseline; sparse maps override it. This lets a region represent
    ice, wind, intermittent control, or an uncertain actuator without changing
    the agent interface.
    """),
    code("""
    reliability_map = {
        (row, column): 0.52 + 0.06 * row
        for row in range(1, 4)
        for column in range(2, 5)
    }
    heterogeneous = StochasticMazeEnv(
        shape=(5, 7),
        start=(4, 0),
        goals={(0, 6): 8.0},
        action_reliability=0.96,
        state_reliability=reliability_map,
        slip_weights={"left": 0.45, "right": 0.45, "stay": 0.10},
        step_reward=-0.04,
    )
    plot_transition_noise(heterogeneous, title="Reliability $p(s)$")
    plt.show()
    """),
    md(r"""
    ## 4. Walls are stochastic processes

    Four dynamic mechanisms answer different questions:

    - independent walls resample a Bernoulli presence state;
    - Markov walls retain temporal correlation through $p_{01}$ and $p_{11}$;
    - scheduled walls impose known interventions;
    - event walls change after the agent reaches a trigger set.

    Static edges may coexist with all four. The environment updates walls at a
    documented point in its step transition and reports structural events.
    """),
    code("""
    edge_independent = ((1, 1), (1, 2))
    edge_markov = ((2, 2), (2, 3))
    edge_scheduled = ((3, 1), (3, 2))
    edge_event = ((3, 3), (3, 4))
    dynamic = StochasticMazeEnv(
        shape=(5, 6),
        start=(4, 0),
        goals={(0, 5): 5.0},
        action_reliability=0.85,
        independent_walls=[IndependentWall(edge_independent, presence_probability=0.25)],
        markov_walls=[MarkovWall(edge_markov, p01=0.12, p11=0.88, initial_probability=0.5)],
        scheduled_walls=[ScheduledWall(edge_scheduled, changes={10: True, 25: False})],
        event_walls=[
            EventWall(
                edge_event,
                trigger_states={(4, 0)},
                present_after_trigger=True,
                initial_present=False,
                once=True,
            )
        ],
        max_episode_steps=80,
    )

    observation, _ = dynamic.reset(seed=SEED)
    wall_history = [tuple(dynamic.current_walls)]
    scalar_info = []
    for _ in range(35):
        action = dynamic.action_space.sample()
        observation, reward, terminated, truncated, info = dynamic.step(action)
        wall_history.append(tuple(dynamic.current_walls))
        scalar_info.append({key: value for key, value in info.items() if np.isscalar(value)})
        if terminated or truncated:
            observation, _ = dynamic.reset()

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7))
    for time, ax in zip((0, 10, 20, 30), axes, strict=True):
        plot_maze(dynamic, walls=wall_history[time], ax=ax, title=f"walls at t={time}")
    plt.tight_layout()
    plt.show()

    # In Jupyter, display this object to animate every recorded realization.
    topology_animation = animate_topology(dynamic, wall_history, interval=180)[2]
    """),
    md(r"""
    ### Markovization by state augmentation

    A maze position alone is not Markov when a persistent wall state is latent
    from the state. For $m$ binary Markov walls, the exact state is
    $(s,w_1,\ldots,w_m)$ and has up to $|\mathcal S|2^m$ elements. The environment
    builds this augmented MDP only when requested and tractable; it never calls a
    marginal one-step kernel an exact solution to the persistent process.
    """),
    code("""
    one_markov_wall = StochasticMazeEnv(
        shape=(3, 4),
        start=(2, 0),
        goals={(0, 3): 4.0},
        action_reliability=0.9,
        markov_walls=[
            MarkovWall(((1, 1), (1, 2)), p01=0.08, p11=0.94, initial_probability=0.4)
        ],
        step_reward=-0.02,
    )
    augmented_mdp = one_markov_wall.exact_mdp(augment_walls=True)
    augmented_solution = value_iteration(augmented_mdp, gamma=0.98)
    print(
        f"physical cells={np.prod(one_markov_wall.shape)}, "
        f"augmented states={augmented_mdp.n_states}, "
        f"Bellman sweeps={augmented_solution.iterations}"
    )
    """),
    md(r"""
    ## 5. Hazards and partial observation

    Hazards can be terminal, penalizing, probabilistically active, or moving.
    Observation modes are separate from latent dynamics: `state` returns an exact
    integer index, `full` exposes a structured diagnostic observation, `local`
    returns a finite neighborhood with noisy wall readings, and `noisy_state`
    corrupts the reported position. Tabular exact-control comparisons in this
    repository use the fully observed state mode unless the latent state is
    explicitly augmented.
    """),
    code("""
    hazard_kwargs = dict(
        shape=(5, 7),
        start=(4, 0),
        goals={(0, 6): 7.0},
        hazards=[
            Hazard(
                (1, 4),
                penalty=-5.0,
                terminal=True,
                activation_probability=0.8,
                terminal_probability=1.0,
            )
        ],
        moving_hazards=[
            MovingHazard((3, 3), penalty=-2.0, terminal=False, movement_probability=0.7)
        ],
        action_reliability=0.9,
        max_episode_steps=100,
    )
    local_env = StochasticMazeEnv(
        **hazard_kwargs,
        observation_mode="local",
        observation_radius=1,
        wall_observation_noise=0.10,
    )
    noisy_env = StochasticMazeEnv(
        **hazard_kwargs,
        observation_mode="noisy_state",
        state_observation_noise=0.15,
    )
    local_observation, local_info = local_env.reset(seed=SEED)
    noisy_observation, noisy_info = noisy_env.reset(seed=SEED)
    print("local observation space:", local_env.observation_space)
    print("local observation:", local_observation)
    print("noisy-state observation / latent state:", noisy_observation, noisy_info["state_index"])
    plot_maze(local_env, title="Hazards remain part of the latent process")
    plt.show()
    """),
    md(r"""
    ## 6. Nonstationarity

    The transition and reward kernels may drift gradually, jump at an abrupt
    change point, switch periodically, or switch at geometrically distributed
    times. The regime is recorded in `info`. Such a run has no single stationary
    $Q^*$; evaluation must use a time-indexed oracle, instantaneous frozen model,
    or a tracking metric, and label that choice.
    """),
    code("""
    regimes = {
        "gradual": NonstationarityConfig(
            mode="gradual",
            reliability_multipliers=(1.0, 0.65),
            reward_multipliers=(1.0, 1.0),
            horizon=120,
        ),
        "abrupt": NonstationarityConfig(
            mode="abrupt",
            reliability_multipliers=(1.0, 0.65),
            reward_multipliers=(1.0, 1.0),
            change_step=50,
        ),
        "periodic": NonstationarityConfig(
            mode="periodic",
            reliability_multipliers=(1.0, 0.65),
            reward_multipliers=(1.0, 1.0),
            period=25,
        ),
        "random": NonstationarityConfig(
            mode="random",
            reliability_multipliers=(1.0, 0.65),
            reward_multipliers=(1.0, 1.0),
            switch_probability=0.02,
        ),
    }
    regime_rows = []
    for name, regime in regimes.items():
        env = StochasticMazeEnv(
            shape=(4, 6),
            start=(3, 0),
            goals={(0, 5): 5.0},
            action_reliability=0.92,
            nonstationarity=regime,
            max_episode_steps=140,
        )
        env.reset(seed=SEED)
        for time in range(120):
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            scalars = {key: value for key, value in info.items() if np.isscalar(value)}
            regime_rows.append({"mechanism": name, "time": time, **scalars})
            if terminated or truncated:
                env.reset()
    regime_frame = pd.DataFrame(regime_rows)
    candidate = next(
        (column for column in regime_frame if "reliability" in column and "multiplier" in column),
        None,
    )
    if candidate is not None:
        for name, sample in regime_frame.groupby("mechanism"):
            plt.plot(sample["time"], sample[candidate], label=name)
        plt.ylabel(candidate.replace("_", " "))
        plt.xlabel("environment step")
        plt.legend()
        plt.title("Recorded transition regime")
        plt.show()
    else:
        display(regime_frame.head())
    """),
    md(r"""
    ## 7. Exact policies under increasing action noise

    Consider two routes. The center corridor is short, but lateral slips enter
    terminal hazards. The lower route is longer and shielded by edge walls. At
    high reliability, the direct route dominates; as control deteriorates, the
    value of physical separation from the hazards can exceed the step cost.
    """),
    code("""
    corridor_hazards = [
        Hazard((row, column), penalty=-8.0, terminal=True)
        for row in (1, 3)
        for column in range(2, 7)
    ]
    protected_edges = {
        ((3, column), (4, column))
        for column in range(1, 8)
    }


    def risk_safe_maze(reliability):
        return StochasticMazeEnv(
            shape=(5, 9),
            start=(2, 0),
            goals={(2, 8): 8.0},
            hazards=corridor_hazards,
            static_walls=protected_edges,
            action_reliability=reliability,
            slip_weights={"left": 0.5, "right": 0.5},
            step_reward=-0.06,
            max_episode_steps=250,
        )


    reliabilities = (1.0, 0.90, 0.75, 0.60)
    solved = []
    fig, axes = plt.subplots(1, len(reliabilities), figsize=(18, 4.0))
    for reliability, ax in zip(reliabilities, axes, strict=True):
        env = risk_safe_maze(reliability)
        solution = value_iteration(env.exact_mdp(), gamma=0.98)
        solved.append((env, solution))
        plot_policy(
            solution.policy,
            env,
            values=solution.values,
            ax=ax,
            title=f"p={reliability:.2f}",
        )
    plt.tight_layout()
    plt.show()
    """),
    code("""
    reliability_grid = np.linspace(0.55, 1.0, 24)
    direct_advantage = []
    selected_action = []
    for reliability in reliability_grid:
        env = risk_safe_maze(float(reliability))
        solution = value_iteration(env.exact_mdp(), gamma=0.98)
        start = env.state_to_index[(2, 0)]
        direct_advantage.append(
            solution.q_values[start, int(MazeAction.EAST)]
            - solution.q_values[start, int(MazeAction.SOUTH)]
        )
        selected_action.append(solution.policy[start])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhline(0, color="0.2", linewidth=1)
    ax.plot(reliability_grid, direct_advantage, marker="o", markersize=3)
    ax.set(
        xlabel="intended-action reliability",
        ylabel=r"$Q^*(s_0, east)-Q^*(s_0, south)$",
        title="Exact preference: short risky route versus longer protected route",
    )
    plt.show()
    """),
    md(r"""
    ## 8. Exactness boundaries

    `exact_mdp()` is valid for a stationary, fully observed configuration. Markov
    wall memory can be made exact with explicit augmentation, at exponential cost.
    Scheduled/event walls, moving hazards, parameter drift, episode-level random
    parameters, and corrupted observations generally require additional time,
    event, hazard, parameter, or belief state. The environment raises
    `ExactModelUnavailable` when the requested finite model would silently omit
    such state. This distinction is essential: simulation access is not exact
    model access.
    """),
]


q_experiments = [
    md(r"""
    # How does environmental stochasticity affect Q-learning convergence?

    We study movement reliability

    $$p\in\{1.0,0.95,0.90,0.80,0.70,0.60\}$$

    with exact $Q^*$ available separately for every stationary environment. The
    primary target is not merely return: we retain $\|Q_t-Q^*\|_\infty$, L2
    error, tie-aware policy disagreement, visitation, success, and online TD
    summaries. All uncertainty summaries treat the trial—not an episode—as the
    independent experimental unit, with the seed retained as a pairing label.

    This notebook uses Protocol v2 throughout. Exploratory training episodes and
    update-free held-out evaluations are different tables. Scientific
    conditions have stable `condition_id` values; seeded replicates have stable
    `trial_id` values. Raw transitions are deterministically sampled while
    episode and state-action summaries still observe every transition.

    `QUICK=True` is a structural run. Set it to `False` for the specified
    100-seed, 5,000-episode experiment, or run the versioned YAML configuration
    from the command line.
    """),
    code("""
    from __future__ import annotations

    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from rllab.agents import LinearDecaySchedule, QLearningAgent
    from rllab.environments import Hazard, MazeAction, StochasticMazeEnv
    from rllab.evaluation import compare_to_optimal, episodes_to_threshold
    from rllab.experiments import Experiment, ExperimentConfig, RunStore, estimate_run
    from rllab.metrics import td_error_summary
    from rllab.theory import value_iteration
    from rllab.visualization import (
        plot_final_distribution,
        plot_learning_curves,
        plot_maze,
        plot_policy,
        plot_state_heatmap,
        plot_sweep_response,
        plot_td_error_heatmap,
        plot_transition_noise,
    )

    SMOKE = os.environ.get("RL_LAB_NOTEBOOK_SMOKE") == "1"
    QUICK = True
    SEED = 29
    RELIABILITIES = (1.0, 0.70) if SMOKE else (1.0, 0.95, 0.90, 0.80, 0.70, 0.60)
    N_SEEDS = 1 if SMOKE else (3 if QUICK else 100)
    EPISODES = 8 if SMOKE else (180 if QUICK else 5_000)
    EVALUATION_INTERVAL = 4 if SMOKE else (30 if QUICK else 250)
    EVALUATION_EPISODES = 1 if SMOKE else (3 if QUICK else 10)
    STEP_SAMPLE_FRACTION = 0.25 if SMOKE else 0.05
    RESULTS_DIR = Path(os.environ.get("RL_LAB_NOTEBOOK_RESULTS", "results"))
    reliability_column = "sweep_environment_action_reliability"
    available_styles = set(plt.style.available)
    plot_style = next(
        (
            style
            for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot")
            if style in available_styles
        ),
        "default",
    )
    plt.style.use(plot_style)
    """),
    md(r"""
    ## 1. Registered experimental design

    The sweep expands a Cartesian product of environment parameters, algorithms,
    and seeds. `scenario_id` identifies environment semantics, `condition_id`
    identifies a seed-independent scientific condition, and `trial_id` identifies
    one seeded replicate. Changing worker count, output location, or diagnostic
    retention does not silently create a new scientific condition.

    A root seed spawns independent environment, agent, and held-out evaluation
    streams. The evaluation policy is cloned and receives no updates. We evaluate
    both the matched training environment and a deterministic probe, always on
    paired seeds at each checkpoint.

    The step-size remains constant here. That is intentional: in a stochastic MDP
    it yields a visible asymptotic noise floor and preserves capacity to track
    later nonstationarity. A decaying step size addresses a different estimand.
    """),
    code("""
    base_environment = {
        "shape": (6, 8),
        "start": (5, 0),
        "goals": {(0, 7): 8.0},
        "blocked_cells": [(1, 1), (1, 2), (2, 5), (3, 2), (4, 5)],
        "static_walls": [
            ((4, 1), (4, 2)),
            ((2, 3), (2, 4)),
            ((1, 6), (2, 6)),
        ],
        "slip_weights": {"left": 0.45, "right": 0.45, "stay": 0.10},
        "step_reward": -0.04,
        "max_episode_steps": 180,
    }
    config = ExperimentConfig.from_mapping(
        {
            "config_schema_version": 2,
            "experiment": {
                "name": "q-learning-reliability",
                "episodes": EPISODES,
                "seeds": list(range(N_SEEDS)),
                "environment": {
                    "name": "stationary-maze",
                    "kind": "stochastic_maze",
                    "parameters": base_environment,
                },
                "agents": [
                    {
                        "name": "q_learning",
                        "kind": "q_learning",
                        "parameters": {
                            "learning_rate": 0.14,
                            "gamma": 0.98,
                            "epsilon": {
                                "kind": "linear",
                                "start": 0.30,
                                "end": 0.03,
                                "duration": max(1_000, EPISODES * 20),
                            },
                        },
                    }
                ],
                "sweep": {"environment.action_reliability": list(RELIABILITIES)},
                "snapshot_interval": 2 if SMOKE else (10 if QUICK else 25),
                "exact_reference": True,
                "tags": {"question": "stochasticity-and-q-convergence"},
            },
            "policy_evaluation": {
                "enabled": True,
                "interval_episodes": EVALUATION_INTERVAL,
                "episodes_per_checkpoint": EVALUATION_EPISODES,
                "include_initial": True,
                "include_final": True,
                "scenarios": [
                    {"name": "matched_training_environment"},
                    {
                        "name": "deterministic_probe",
                        "environment_overrides": {"action_reliability": 1.0},
                    },
                ],
            },
            "execution": {
                "parallel_workers": 1 if QUICK else 8,
                "failure_policy": "fail_fast",
            },
            "artifacts": {
                "output_dir": str(RESULTS_DIR),
                "table_format": "auto",
                "flush_rows": 250 if SMOKE else 10_000,
                "save_q_snapshots": True,
                "step_retention": {
                    "mode": "sample",
                    "fraction": STEP_SAMPLE_FRACTION,
                    "keep_terminal": True,
                    "keep_events": True,
                },
            },
        }
    )

    run_estimate = estimate_run(config)
    design = pd.DataFrame(
        {
            "scenario_id": trial.scenario_id,
            "condition_id": trial.condition_id,
            "trial_id": trial.trial_id,
            "seed": trial.seed,
            reliability_column: trial.sweep_values["environment.action_reliability"],
        }
        for trial in config.trials()
    )
    assert design["trial_id"].is_unique
    assert design.groupby("condition_id")[reliability_column].nunique().eq(1).all()
    display(pd.Series(run_estimate.as_dict(), name="preflight estimate"))
    display(design.head())

    result = Experiment(config).run(persist=True, progress=True)
    training_episodes = result.training_episodes
    held_out_evaluations = result.evaluations
    print(result.experiment_id)
    print(result.run_directory)
    print(
        "training/evaluation/snapshot rows:",
        len(training_episodes),
        len(held_out_evaluations),
        len(result.snapshots),
    )
    """),
    md(r"""
    ## 2. The artifact store is part of the protocol

    A persisted result is a lazy handle over a versioned run store, not one giant
    in-memory frame. Each trial writes bounded table parts into a private attempt
    directory; only an atomic commit makes those parts visible. The manifest then
    selects exactly one successful attempt per `trial_id` and records checksums,
    row counts, source provenance, and the resolved configuration.

    Step retention changes storage cost, not the learning process or its online
    summaries. Here terminal/event steps are always kept and the remaining steps
    are selected by a deterministic hash sample. Temporal analyses such as TD
    autocorrelation require a dedicated run with `mode: all`; the sampled main
    sweep is appropriate for inspecting individual transitions, not adjacency.
    """),
    code("""
    assert result.run_directory is not None
    store = RunStore.open(result.run_directory)
    commits = store.committed_attempts()
    retention_audit = pd.DataFrame(
        {
            "trial_id": commit.trial_id,
            "observed_steps": commit.metadata["observed_steps"],
            "retained_steps": commit.metadata["retained_steps"],
            "retained_fraction": (
                commit.metadata["retained_steps"] / commit.metadata["observed_steps"]
            ),
            "step_parts": sum(
                artifact.table == "steps" for artifact in commit.artifacts
            ),
        }
        for commit in commits
    )
    assert store.manifest.artifact_schema_version == 2
    assert store.manifest.status == "complete"
    display(retention_audit.head())

    lazy_preview = next(
        result.iter_table(
            "training_episodes",
            columns=("condition_id", "trial_id", "seed", "episode", "episode_return"),
            batch_size=5,
            verify=True,
        )
    )
    display(lazy_preview)
    """),
    md(r"""
    ## 3. Training return and held-out return answer different questions

    `training_episodes` contains exploratory behavior and update diagnostics.
    `evaluations` contains frozen-policy rollouts on seeds disjoint from training;
    repeated evaluation episodes are reduced within each trial/checkpoint before
    trials are bootstrapped. In quick mode three seeds only exercise the pipeline
    and do not support a scientific confidence interval.
    """),
    code("""
    display_columns = [
        "condition_id",
        "trial_id",
        reliability_column,
        "seed",
        "episode",
        "episode_return",
        "success",
        "episode_length",
        "td_error_variance",
    ]
    display(training_episodes[display_columns].head())

    evaluation_curves = (
        held_out_evaluations.groupby(
            [
                "condition_id",
                "trial_id",
                "seed",
                reliability_column,
                "evaluation_scenario",
                "checkpoint_episode",
            ],
            as_index=False,
        )
        .agg(episode_return=("episode_return", "mean"), success=("success", "mean"))
    )
    matched_evaluation_curves = evaluation_curves.loc[
        evaluation_curves["evaluation_scenario"].eq("matched_training_environment")
    ].copy()
    deterministic_probe_summary = (
        evaluation_curves.loc[
            evaluation_curves["evaluation_scenario"].eq("deterministic_probe")
        ]
        .groupby([reliability_column, "checkpoint_episode"], as_index=False)
        .agg(mean_return=("episode_return", "mean"), n_trials=("trial_id", "nunique"))
    )
    display(deterministic_probe_summary.tail())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    plot_learning_curves(
        training_episodes,
        metric="episode_return",
        group=reliability_column,
        smooth=min(15, EPISODES),
        individual=not QUICK,
        n_resamples=300 if QUICK else 2_000,
        ax=axes[0],
    )
    plot_learning_curves(
        matched_evaluation_curves,
        metric="episode_return",
        x="checkpoint_episode",
        group=reliability_column,
        n_resamples=300 if QUICK else 2_000,
        ax=axes[1],
    )
    axes[0].set_title("Exploratory training return")
    axes[1].set_title("Update-free held-out return")
    plt.tight_layout()
    plt.show()

    matched_final = matched_evaluation_curves.copy()
    matched_final = matched_final.loc[
        matched_final["checkpoint_episode"]
        .eq(matched_final.groupby("trial_id")["checkpoint_episode"].transform("max"))
    ].rename(columns={"checkpoint_episode": "episode"})
    plot_final_distribution(
        matched_final,
        metric="episode_return",
        group=reliability_column,
        last_episodes=1,
    )
    plt.title("Trial-level final held-out return")
    plt.show()
    """),
    md(r"""
    ## 4. Convergence against a different exact $Q^*$ at each reliability

    Comparing all learners to the deterministic optimum would confound learning
    error with a change in the estimand. The runner freezes each trial's stationary
    model, solves it by value iteration, and records norms and policy diagnostics
    at Q snapshots.
    """),
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    plot_learning_curves(
        result.snapshots.query("episode >= 0"),
        metric="q_error_inf",
        group=reliability_column,
        log_y=True,
        n_resamples=300 if QUICK else 2_000,
        ax=axes[0],
    )
    plot_learning_curves(
        result.snapshots.query("episode >= 0"),
        metric="policy_disagreement",
        group=reliability_column,
        n_resamples=300 if QUICK else 2_000,
        ax=axes[1],
    )
    axes[0].set_title(r"$\\|Q_t-Q^*\\|_\\infty$")
    axes[1].set_title("Tie-aware policy disagreement")
    plt.tight_layout()
    plt.show()

    threshold = episodes_to_threshold(
        result.snapshots.query("episode >= 0"),
        metric="policy_disagreement",
        threshold=0.10,
        sustain=2,
        groups=("trial_id", reliability_column),
    )
    display(
        threshold.groupby(reliability_column, as_index=False)
        .agg(
            median_episodes=("episodes_to_threshold", "median"),
            fraction_reached=("reached", "mean"),
        )
        .sort_values(reliability_column, ascending=False)
    )
    """),
    code("""
    plot_sweep_response(
        result.snapshots.query("episode >= 0"),
        parameter="environment.action_reliability",
        metric="q_error_inf",
        last_episodes=1,
    )
    plt.title("Reliability versus final sup-norm error")
    plt.show()

    plot_sweep_response(
        training_episodes,
        parameter="environment.action_reliability",
        metric="td_error_variance",
        last_episodes=min(100, EPISODES // 3),
    )
    plt.title("Reliability versus late TD-error variance")
    plt.show()
    """),
    md(r"""
    ## 5. TD-error distribution and location under bounded retention

    For Q-learning,

    $$\delta_t=R_{t+1}+\gamma(1-D_{t+1})\max_a Q_t(S_{t+1},a)
      -Q_t(S_t,A_t).$$

    Conditional TD variance mixes transition/reward aleatoric noise, changing
    value estimates, and under-explored targets. It is therefore a diagnostic,
    not automatically an epistemic uncertainty estimator. Online episode and
    state-action tables contain moments computed from every update. The raw step
    table is smaller: `retention_reason == "sample"` selects a value-independent
    hash sample, while terminal/event rows are kept for forensic inspection.

    Marginal quantiles can be estimated from the hash-sampled rows. Sign changes,
    rolling windows, and autocorrelation cannot: sampling destroys adjacency.
    Those require a narrower Protocol-v2 run whose retention mode is `all`.
    """),
    code("""
    lowest_reliability = min(RELIABILITIES)
    retained_steps = result.steps
    sampled_steps = retained_steps.loc[
        retained_steps[reliability_column].eq(lowest_reliability)
        & retained_steps["retention_reason"].eq("sample")
    ]
    if sampled_steps.empty:
        print("No hash-sampled rows in this tiny run; online summaries remain available.")
    else:
        sampled_summary = td_error_summary(sampled_steps, autocorrelation_lags=())
        display(
            sampled_summary[
                [
                    "state",
                    "action",
                    "n_trials",
                    "count",
                    "mean_td_error",
                    "variance_td_error",
                    "q05_td_error",
                    "q95_td_error",
                ]
            ]
            .sort_values("variance_td_error", ascending=False)
            .head(12)
        )

    final_state_actions = (
        result.state_actions.loc[
            result.state_actions[reliability_column].eq(lowest_reliability)
        ]
        .sort_values("episode")
        .groupby(["trial_id", "state", "action"], as_index=False)
        .tail(1)
        .copy()
    )
    final_state_actions["absolute_td_total"] = (
        final_state_actions["mean_absolute_td_error"] * final_state_actions["td_count"]
    )
    trial_state_td = (
        final_state_actions.groupby(["trial_id", "state"], as_index=False)
        .agg(absolute_td_total=("absolute_td_total", "sum"), td_count=("td_count", "sum"))
    )
    trial_state_td["mean_absolute_td_error"] = (
        trial_state_td["absolute_td_total"] / trial_state_td["td_count"]
    )
    online_td_by_state = (
        trial_state_td.groupby("state", as_index=False)["mean_absolute_td_error"].mean()
    )

    analysis_env = StochasticMazeEnv(
        **base_environment,
        action_reliability=lowest_reliability,
    )
    plot_td_error_heatmap(online_td_by_state, analysis_env, statistic="mean_absolute")
    plt.title(f"All-update mean absolute TD error at p={lowest_reliability}")
    plt.show()
    """),
    md(r"""
    ## 6. Spatial heterogeneity: unreliable shortcut, protected detour

    We now hold the overall topology fixed and assign poor reliability only to a
    short central corridor. The lower route is longer but nearly deterministic
    and protected from lateral slips by edge walls. This separates the global
    difficulty effect above from a local risk--distance tradeoff.
    """),
    code("""
    corridor_hazards = [
        Hazard((row, column), penalty=-8.0, terminal=True)
        for row in (1, 3)
        for column in range(2, 7)
    ]
    protected_edges = [
        ((3, column), (4, column))
        for column in range(1, 8)
    ]
    corridor_states = {(2, column): 0.60 for column in range(1, 8)}
    heterogeneous_parameters = {
        "shape": (5, 9),
        "start": (2, 0),
        "goals": {(2, 8): 8.0},
        "hazards": corridor_hazards,
        "static_walls": protected_edges,
        "action_reliability": 0.97,
        "state_reliability": corridor_states,
        "slip_weights": {"left": 0.5, "right": 0.5},
        "step_reward": -0.06,
        "max_episode_steps": 250,
    }
    hetero_env = StochasticMazeEnv(**heterogeneous_parameters)
    exact_hetero = value_iteration(hetero_env.exact_mdp(), gamma=0.98)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_transition_noise(hetero_env, ax=axes[0], title="Spatial reliability")
    plot_policy(
        exact_hetero.policy,
        hetero_env,
        values=exact_hetero.values,
        ax=axes[1],
        title="Exact optimal policy",
    )
    plt.tight_layout()
    plt.show()
    """),
    code("""
    hetero_config = ExperimentConfig.from_mapping(
        {
            "config_schema_version": 2,
            "experiment": {
                "name": "q-learning-heterogeneous",
                "episodes": 6 if SMOKE else (300 if QUICK else 5_000),
                "seeds": list(range(1 if SMOKE else (4 if QUICK else 100))),
                "environment": {
                    "name": "risky-shortcut-safe-detour",
                    "kind": "stochastic_maze",
                    "parameters": heterogeneous_parameters,
                },
                "agents": [
                    {
                        "name": "q_learning",
                        "kind": "q_learning",
                        "parameters": {
                            "learning_rate": 0.12,
                            "gamma": 0.98,
                            "epsilon": {
                                "kind": "linear",
                                "start": 0.35,
                                "end": 0.03,
                                "duration": 30_000,
                            },
                        },
                    }
                ],
                "snapshot_interval": 2 if SMOKE else (10 if QUICK else 25),
                "exact_reference": True,
            },
            "policy_evaluation": {
                "enabled": True,
                "interval_episodes": 3 if SMOKE else (50 if QUICK else 250),
                "episodes_per_checkpoint": 1 if SMOKE else (3 if QUICK else 10),
                "include_initial": True,
                "include_final": True,
            },
            "execution": {"parallel_workers": 1 if QUICK else 8},
            "artifacts": {
                "output_dir": str(RESULTS_DIR),
                "table_format": "auto",
                "flush_rows": 250 if SMOKE else 10_000,
                "save_q_snapshots": True,
                "step_retention": {
                    "mode": "none",
                    "keep_terminal": True,
                    "keep_events": True,
                },
            },
        }
    )
    heterogeneous_result = Experiment(hetero_config).run(persist=True)
    plot_learning_curves(
        heterogeneous_result.snapshots.query("episode >= 0"),
        metric="policy_disagreement",
        group="agent",
        n_resamples=300 if QUICK else 2_000,
    )
    plt.title("Learning the local risk--distance tradeoff")
    plt.show()

    final_heterogeneous_state_actions = (
        heterogeneous_result.state_actions.sort_values("episode")
        .groupby(["trial_id", "state", "action"], as_index=False)
        .tail(1)
    )
    trial_state_visits = (
        final_heterogeneous_state_actions.groupby(["trial_id", "state"], as_index=False)[
            "visit_count"
        ].sum()
    )
    state_visits = (
        trial_state_visits.groupby("state", as_index=False)["visit_count"]
        .mean()
        .set_index("state")["visit_count"]
    )
    plot_state_heatmap(
        state_visits.to_dict(),
        hetero_env,
        colorbar_label="mean visits per seed",
        title="Where Q-learning samples",
    )
    plt.show()
    """),
    md(r"""
    ### Inspect one learned policy without hiding the interaction loop

    The registered runner is the reproducible path for a sweep. For close
    inspection it is also useful to keep one learner in memory. This loop uses the
    same agent/environment contract and demonstrates the exact quantity logged as
    a TD residual.
    """),
    code("""
    single_env = StochasticMazeEnv(**heterogeneous_parameters)
    single_agent = QLearningAgent(
        single_env.n_states,
        single_env.n_actions,
        learning_rate=0.12,
        gamma=0.98,
        epsilon=LinearDecaySchedule(0.35, 0.03, 20_000),
        seed=SEED,
    )
    q_errors, policy_errors, td_errors = [], [], []
    for episode in range(30 if SMOKE else (500 if QUICK else 5_000)):
        observation, _ = single_env.reset(seed=SEED if episode == 0 else None)
        state = int(observation)
        terminated = truncated = False
        while not (terminated or truncated):
            action = single_agent.act(state)
            observation, reward, terminated, truncated, _ = single_env.step(action)
            next_state = int(observation)
            update = single_agent.update(state, action, reward, next_state, terminated)
            td_errors.append(update.td_error)
            state = next_state
        if episode % 10 == 0:
            diagnostics = compare_to_optimal(single_agent.q_values, exact_hetero.q_values)
            q_errors.append(diagnostics["q_error_inf"])
            policy_errors.append(diagnostics["policy_disagreement"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_policy(
        single_agent.greedy_policy,
        single_env,
        values=single_agent.values,
        ax=axes[0],
        title="Learned greedy policy",
    )
    axes[1].plot(np.arange(len(q_errors)) * 10, q_errors, label=r"$\\|Q-Q^*\\|_\\infty$")
    axes[1].plot(np.arange(len(policy_errors)) * 10, policy_errors, label="policy disagreement")
    axes[1].set(xlabel="episode", title="Ground-truth diagnostics")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    """),
    md(r"""
    ## 7. Interpretation and next experiments

    Environmental stochasticity changes at least three things simultaneously:
    the optimal value/policy, the conditional variance of TD targets, and the
    state-action occupancy induced by exploration. Consequently, slower
    sup-norm convergence cannot be attributed to "noise" without checking
    coverage and local action gaps.

    Follow-ups enabled by the stored schema include variance-adaptive
    $\alpha_t(s,a)$, forgetting under regime changes, change-point tests on TD
    residuals in a targeted full-retention run, model-based uncertainty from
    transition counts, and hitting-time rather than discounted-return objectives.
    Each should be compared on the same trial-level distributions and, where
    meaningful, the same exact model and paired held-out evaluation seeds.
    """),
]


policies_under_risk_drift_memory = [
    md(r"""
    # Policies under risk, drift, and memory

    Risk changes which backup is attractive. Drift changes the backup's target.
    Hidden memory determines whether that target can be represented at all.

    This notebook turns the three follow-ups from notebook 02 into one connected
    investigation. Act I compares SARSA, Expected SARSA, Q-learning, and Double
    Q-learning on a risky shortcut. Act II aligns their TD errors around a
    repeated within-episode reliability shock. Act III holds average wall
    frequency fixed while increasing wall persistence, then compares a
    position-only representation with one that includes the wall bit.

    This is a controlled comparison of learning mechanics, not a universally
    tuned leaderboard. Every method receives the same transition budget,
    exploration schedule, and root seeds. Held-out greedy evaluation is the
    primary performance view; exploratory training return answers a different
    question.

    QUICK mode runs every act with a small seed panel. Set QUICK to False only
    for a deliberate research run. Each act can also reopen an existing
    Protocol-v2 directory, so rerunning plots never requires retraining.
    """),
    code("""
    from __future__ import annotations

    from dataclasses import replace
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from rllab.environments import MarkovWall, NonstationarityConfig
    from rllab.evaluation import (
        episodes_to_threshold,
        evaluation_checkpoint_summary,
    )
    from rllab.experiments import (
        AgentSpec,
        ArtifactSpec,
        EnvironmentSpec,
        EvaluationScenario,
        ExecutionSpec,
        Experiment,
        ExperimentConfig,
        ExperimentResult,
        PolicyEvaluationSpec,
        RunStore,
        StepRetentionSpec,
        estimate_run,
        make_environment,
    )
    from rllab.metrics import paired_seed_contrast
    from rllab.theory import value_iteration
    from rllab.visualization import (
        plot_final_distribution,
        plot_learning_curves,
        plot_maze,
        plot_paired_contrasts,
        plot_policy,
        plot_transition_noise,
    )

    SMOKE = os.environ.get("RL_LAB_NOTEBOOK_SMOKE") == "1"
    QUICK = True
    SHOW_PROGRESS = False  # avoids stale Jupyter widget models; the CLI has live progress
    AGENT_ORDER = ("q_learning", "sarsa", "expected_sarsa", "double_q_learning")
    AGENT_LABELS = {
        "q_learning": "Q-learning",
        "sarsa": "SARSA",
        "expected_sarsa": "Expected SARSA",
        "double_q_learning": "Double Q-learning",
    }
    EXISTING_RUNS: dict[str, str | Path | None] = {
        "risk": None,
        "drift": None,
        "memory": None,
    }

    def find_repo_root(start: Path) -> Path:
        for candidate in (start.resolve(), *start.resolve().parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "src" / "rllab").exists():
                return candidate
        raise FileNotFoundError("Run this notebook from inside the rl-lab repository.")

    REPO_ROOT = find_repo_root(Path.cwd())
    configured_results = os.environ.get("RL_LAB_NOTEBOOK_RESULTS")
    RESULTS_DIR = Path(configured_results) if configured_results else REPO_ROOT / "results"
    N_RESAMPLES = 100 if SMOKE else (400 if QUICK else 2_000)
    POLICY_SEED = 0

    available_styles = set(plt.style.available)
    plot_style = next(
        (
            style
            for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot")
            if style in available_styles
        ),
        "default",
    )
    plt.style.use(plot_style)

    def add_method(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        if "agent" in result:
            result["method"] = result["agent"].map(AGENT_LABELS).fillna(result["agent"])
        return result

    def last_checkpoint(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[
            frame["checkpoint_episode"].eq(
                frame.groupby("trial_id")["checkpoint_episode"].transform("max")
            )
        ].copy()

    def run_or_reopen(label: str, config: ExperimentConfig) -> ExperimentResult:
        display(pd.Series(estimate_run(config).as_dict(), name=f"{label} preflight"))
        existing = EXISTING_RUNS[label]
        if existing is not None:
            path = Path(existing)
            path = path if path.is_absolute() else REPO_ROOT / path
            store = RunStore.open(path)
            if store.manifest.experiment_name != config.name:
                raise ValueError(
                    f"{label} run is {store.manifest.experiment_name!r}, expected {config.name!r}"
                )
            print(f"Reopened {label}: {store.run_directory}")
            return ExperimentResult(
                experiment_id=store.manifest.run_id,
                run_directory=store.run_directory,
                metadata={"reopened": True},
            )
        print(f"Starting {label} run with {len(config.trials())} matched trials...")
        result = Experiment(config).run(persist=True, progress=SHOW_PROGRESS)
        print(f"Completed {label}: {result.run_directory}")
        return result

    def final_q_tables(result: ExperimentResult) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        final_rows = (
            result.snapshots.query("episode >= 0")
            .sort_values(["trial_id", "episode"])
            .groupby("trial_id", as_index=False)
            .tail(1)
            .copy()
        )
        tables = {}
        for row in final_rows.itertuples(index=False):
            snapshots = result.q_snapshots(
                row.trial_id,
                keys=(row.snapshot_key,),
            )
            tables[row.trial_id] = snapshots[row.snapshot_key]
        return final_rows, tables

    def modal_policy(q_tables: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        policies = np.stack([np.argmax(table, axis=1) for table in q_tables])
        modes = np.array(
            [
                np.bincount(policies[:, state], minlength=q_tables[0].shape[1]).argmax()
                for state in range(policies.shape[1])
            ],
            dtype=int,
        )
        consensus = np.mean(policies == modes[None, :], axis=0)
        return modes, consensus
    """),
    md(r"""
    ## The four backups and the hypotheses

    The algorithms differ at one deliberately narrow point: the bootstrap in
    the one-step target.

    | Method | Bootstrap at the next state |
    |---|---|
    | SARSA | the sampled exploratory action |
    | Expected SARSA | the expectation under the exploratory policy |
    | Q-learning | the largest current action value |
    | Double Q-learning | one table selects and the other evaluates |

    We keep alpha, gamma, epsilon, seeds, and episode budgets fixed. Equal
    hyperparameters are useful for isolating mechanisms, but they do not imply
    that every method is individually tuned to its best setting.
    """),
    code("""
    hypotheses = pd.DataFrame(
        [
            {
                "act": "risk",
                "prediction": "SARSA-style targets price exploratory accidents into the learned route.",
                "primary evidence": "paired held-out return and learned policy",
            },
            {
                "act": "risk",
                "prediction": "Expected SARSA removes next-action sampling noise from SARSA.",
                "primary evidence": "late TD-error variance",
            },
            {
                "act": "drift",
                "prediction": "every fixed-step-size learner shows a TD shock when reliability falls.",
                "primary evidence": "event-aligned absolute TD error",
            },
            {
                "act": "memory",
                "prediction": "backup choice cannot recover a wall bit omitted from the observation.",
                "primary evidence": "position-only versus wall-aware held-out return",
            },
        ]
    )
    display(hypotheses)
    """),
    md(r"""
    ## Act I — Risk: shortcut or detour?

    The first act reuses the versioned heterogeneous-route experiment. The
    central route is short but locally unreliable; the detour is longer. This
    stationary, fully observed environment has an exact finite MDP, so return,
    value error, and tie-aware policy error are all legitimate diagnostics.

    Training return includes epsilon-greedy actions and updates. Held-out return
    freezes a clone of the learned policy and takes greedy actions on a seed
    panel disjoint from training. We treat the training seed—not an evaluation
    episode—as the independent unit.
    """),
    code("""
    full_risk_config = ExperimentConfig.from_yaml(REPO_ROOT / "configs" / "heterogeneous_routes.yaml")
    RISK_EPISODES = 5 if SMOKE else (450 if QUICK else full_risk_config.episodes)
    RISK_SEEDS = (0,) if SMOKE else (tuple(range(5)) if QUICK else full_risk_config.seeds)
    risk_agents = tuple(
        replace(
            agent,
            parameters={
                **agent.parameters,
                "epsilon": {
                    "kind": "linear",
                    "start": 0.30,
                    "end": 0.03,
                    "duration": max(40, RISK_EPISODES * 20),
                },
            },
        )
        for agent in full_risk_config.agents
    )
    risk_config = replace(
        full_risk_config,
        episodes=RISK_EPISODES,
        seeds=RISK_SEEDS,
        agents=risk_agents,
        snapshot_interval=2 if SMOKE else (25 if QUICK else full_risk_config.snapshot_interval),
        policy_evaluation=replace(
            full_risk_config.policy_evaluation,
            interval_episodes=2 if SMOKE else (75 if QUICK else 250),
            episodes_per_checkpoint=1 if SMOKE else (5 if QUICK else 10),
        ),
        execution=ExecutionSpec(parallel_workers=1 if QUICK or SMOKE else 4),
        artifacts=replace(
            full_risk_config.artifacts,
            output_dir=RESULTS_DIR,
            flush_rows=200 if SMOKE else 10_000,
        ),
    )
    risk_result = run_or_reopen("risk", risk_config)
    risk_training = add_method(risk_result.training_episodes)
    risk_evaluations = add_method(evaluation_checkpoint_summary(risk_result.evaluations))
    assert set(risk_training["agent"]) == set(AGENT_ORDER)
    """),
    code("""
    risk_env = make_environment(risk_config.environments[0])
    risk_exact = value_iteration(risk_env.exact_mdp(), gamma=0.98)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    plot_maze(risk_env, ax=axes[0], title="Risky shortcut and detour")
    plot_transition_noise(risk_env, ax=axes[1], title="Intended-action reliability")
    plot_policy(
        risk_exact.policy,
        risk_env,
        values=risk_exact.values,
        ax=axes[2],
        title="Exact optimal policy",
    )
    plt.tight_layout()
    plt.show()
    """),
    md(r"""
    ### Learning behavior versus deployed behavior

    A method can look worse during training simply because its exploratory
    actions are being scored. Conversely, a smooth training curve can conceal a
    brittle greedy policy. The two panels below must be read together.
    """),
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_learning_curves(
        risk_training,
        metric="episode_return",
        group="method",
        smooth=min(25, RISK_EPISODES),
        n_resamples=N_RESAMPLES,
        ax=axes[0],
    )
    plot_learning_curves(
        risk_evaluations,
        metric="episode_return",
        x="checkpoint_episode",
        group="method",
        n_resamples=N_RESAMPLES,
        ax=axes[1],
    )
    axes[0].set_title("Exploratory training return")
    axes[1].set_title("Frozen greedy held-out return")
    plt.tight_layout()
    plt.show()

    risk_final = last_checkpoint(risk_evaluations)
    risk_final_plot = risk_final.rename(columns={"checkpoint_episode": "episode"})
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_final_distribution(
        risk_final_plot,
        metric="episode_return",
        group="method",
        last_episodes=1,
        ax=axes[0],
    )
    plot_final_distribution(
        risk_final_plot,
        metric="success",
        group="method",
        last_episodes=1,
        ax=axes[1],
    )
    axes[0].set_title("Final held-out return by training seed")
    axes[1].set_title("Final held-out success by training seed")
    plt.tight_layout()
    plt.show()
    """),
    code("""
    risk_contrasts = []
    for comparison in AGENT_ORDER:
        if comparison == "q_learning":
            continue
        contrast = paired_seed_contrast(
            risk_final,
            metric="episode_return",
            factor="agent",
            baseline="q_learning",
            comparison=comparison,
            pair_by=("seed",),
            strata=("evaluation_scenario",),
            n_resamples=N_RESAMPLES,
            random_seed=17,
        )
        risk_contrasts.append(contrast.summary)
    risk_contrast_summary = pd.concat(risk_contrasts, ignore_index=True)
    risk_contrast_summary["label"] = risk_contrast_summary["comparison"].map(AGENT_LABELS)
    display(
        risk_contrast_summary[
            ["label", "n_pairs", "mean_difference", "ci_low", "ci_high", "win_rate"]
        ]
    )
    plot_paired_contrasts(
        risk_contrast_summary,
        label="label",
        difference_label="Held-out return difference versus Q-learning",
        title="Matched-seed final contrasts",
    )
    plt.show()
    """),
    md(r"""
    ### Convergence and policy stability

    Exact error is informative here, with one caveat: while epsilon remains
    nonzero, SARSA and Expected SARSA evaluate an exploratory behavior policy.
    Their larger distance from greedy Q-star can therefore be an estimand
    difference rather than a broken update.
    """),
    code("""
    risk_snapshots = add_method(risk_result.snapshots.query("episode >= 0"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    plot_learning_curves(
        risk_snapshots,
        metric="q_error_inf",
        group="method",
        log_y=True,
        n_resamples=N_RESAMPLES,
        ax=axes[0],
    )
    plot_learning_curves(
        risk_snapshots,
        metric="policy_disagreement",
        group="method",
        n_resamples=N_RESAMPLES,
        ax=axes[1],
    )
    plot_final_distribution(
        risk_training,
        metric="td_error_variance",
        group="method",
        last_episodes=max(1, min(75, RISK_EPISODES // 3)),
        ax=axes[2],
    )
    axes[0].set_title("Sup-norm Q error")
    axes[1].set_title("Tie-aware policy disagreement")
    axes[2].set_title("Late within-episode TD variance")
    plt.tight_layout()
    plt.show()

    risk_threshold = episodes_to_threshold(
        risk_snapshots,
        metric="policy_disagreement",
        threshold=0.10,
        sustain=2,
        groups=("trial_id", "agent"),
    )
    display(
        risk_threshold.groupby("agent", as_index=False)
        .agg(
            median_episodes=("episodes_to_threshold", "median"),
            fraction_reached=("reached", "mean"),
        )
        .assign(method=lambda frame: frame["agent"].map(AGENT_LABELS))
        [["method", "median_episodes", "fraction_reached"]]
    )
    """),
    code("""
    risk_final_rows, risk_q_tables = final_q_tables(risk_result)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    plot_policy(risk_exact.policy, risk_env, ax=axes[0, 0], title="Exact optimal policy")
    for ax, agent in zip(axes.flat[1:], AGENT_ORDER, strict=False):
        trial_ids = risk_final_rows.loc[risk_final_rows["agent"].eq(agent), "trial_id"]
        policy, consensus = modal_policy([risk_q_tables[trial_id] for trial_id in trial_ids])
        plot_policy(
            policy,
            risk_env,
            values=consensus,
            ax=ax,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            colorbar_label="fraction of seeds choosing modal action",
            title=AGENT_LABELS[agent],
        )
    axes.flat[-1].set_visible(False)
    plt.show()
    """),
    md(r"""
    **Act I reading rule.** Prefer the paired held-out contrast and the raw seed
    distribution to a visual ranking of smoothed lines. The policy atlas is a
    modal summary across seeds; its blue background reveals where that summary
    is stable and where a single representative arrow would be misleading.
    """),
    md(r"""
    ## Act II — Drift: the target moves

    The environment now drops action reliability after a fixed number of
    decisions. In the current maze API this clock resets at every episode, so
    this is a repeated, event-aligned shock—not one surprise halfway through
    training. That repetition is useful: every episode provides a before/after
    trace, while training seeds remain the independent replicates.

    No stationary Q-star exists for this act. Exact comparison is disabled and
    every transition is retained because adjacency is essential. We align on
    the logged action reliability actually used for each transition; the
    reported next regime changes one row earlier than the affected dynamics.
    """),
    code("""
    DRIFT_CHANGE_STEP = 8
    DRIFT_EPISODES = 5 if SMOKE else (250 if QUICK else 1_500)
    DRIFT_SEEDS = (0,) if SMOKE else (tuple(range(4)) if QUICK else tuple(range(20)))
    drift_agents = tuple(
        AgentSpec(
            name=name,
            kind=name,
            parameters={
                "learning_rate": 0.12,
                "gamma": 0.98,
                "epsilon": {
                    "kind": "linear",
                    "start": 0.30,
                    "end": 0.04,
                    "duration": max(40, DRIFT_EPISODES * 25),
                },
            },
        )
        for name in AGENT_ORDER
    )
    drift_environment = EnvironmentSpec(
        name="repeated_reliability_shock",
        kind="stochastic_maze",
        parameters={
            "shape": (3, 15),
            "start": (1, 0),
            "goals": {(1, 14): 5.0},
            "blocked_cells": [(0, 5), (2, 9)],
            "action_reliability": 0.98,
            "slip_weights": {"left": 0.45, "right": 0.45, "stay": 0.10},
            "step_reward": -0.03,
            "max_episode_steps": 70,
            "nonstationarity": NonstationarityConfig(
                mode="abrupt",
                reliability_multipliers=(1.0, 0.55),
                reward_multipliers=(1.0, 1.0),
                change_step=DRIFT_CHANGE_STEP,
            ),
        },
    )
    drift_config = ExperimentConfig(
        name="policies_under_repeated_drift",
        episodes=DRIFT_EPISODES,
        seeds=DRIFT_SEEDS,
        environments=(drift_environment,),
        agents=drift_agents,
        snapshot_interval=2 if SMOKE else (25 if QUICK else 50),
        exact_reference=False,
        policy_evaluation=PolicyEvaluationSpec(
            enabled=True,
            interval_episodes=2 if SMOKE else (50 if QUICK else 150),
            episodes_per_checkpoint=1 if SMOKE else (4 if QUICK else 10),
            include_initial=True,
            include_final=True,
            scenarios=(EvaluationScenario(name="repeated_shock"),),
        ),
        execution=ExecutionSpec(parallel_workers=1 if QUICK or SMOKE else 4),
        artifacts=ArtifactSpec(
            output_dir=RESULTS_DIR,
            table_format="auto",
            flush_rows=200 if SMOKE else 10_000,
            save_q_snapshots=True,
            step_retention=StepRetentionSpec(mode="all"),
        ),
        tags={"question": "event_aligned_td_response_to_repeated_drift"},
    )
    drift_result = run_or_reopen("drift", drift_config)
    drift_training = add_method(drift_result.training_episodes)
    drift_evaluations = add_method(evaluation_checkpoint_summary(drift_result.evaluations))
    drift_steps = add_method(drift_result.steps)
    assert not drift_result.snapshots["exact_evaluation_available"].any()
    """),
    code("""
    late_drift_steps = drift_steps.loc[
        drift_steps["episode"].ge(max(0, DRIFT_EPISODES // 2))
    ].copy()
    late_drift_steps["absolute_td_error"] = late_drift_steps["td_error"].abs()
    per_trial_step = (
        late_drift_steps.groupby(
            ["trial_id", "agent", "method", "seed", "step"],
            as_index=False,
        )
        .agg(
            mean_absolute_td_error=("absolute_td_error", "mean"),
            mean_reward=("reward", "mean"),
            action_reliability=("env_action_reliability", "mean"),
        )
    )
    drift_profile = (
        per_trial_step.groupby(["method", "step"], as_index=False)
        .agg(
            mean_absolute_td_error=("mean_absolute_td_error", "mean"),
            sem_absolute_td_error=("mean_absolute_td_error", "sem"),
            mean_reward=("mean_reward", "mean"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for method, sample in drift_profile.groupby("method", sort=False):
        axes[0].plot(sample["step"], sample["mean_absolute_td_error"], label=method)
        axes[0].fill_between(
            sample["step"],
            sample["mean_absolute_td_error"] - sample["sem_absolute_td_error"].fillna(0),
            sample["mean_absolute_td_error"] + sample["sem_absolute_td_error"].fillna(0),
            alpha=0.15,
        )
        axes[1].plot(sample["step"], sample["mean_reward"], label=method)
    for ax in axes:
        ax.axvline(DRIFT_CHANGE_STEP, color="black", linestyle="--", label="reliability drop")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        ax.set_xlabel("step within episode")
    axes[0].set(ylabel="mean absolute TD error", title="Event-aligned TD shock")
    axes[1].set(ylabel="mean reward", title="Reward around the same shock")
    plt.tight_layout()
    plt.show()

    late_drift_steps["window"] = np.select(
        [
            late_drift_steps["step"].lt(DRIFT_CHANGE_STEP),
            late_drift_steps["step"].lt(DRIFT_CHANGE_STEP + 5),
        ],
        ["before", "shock"],
        default="later",
    )
    drift_windows = (
        late_drift_steps.groupby(
            ["trial_id", "agent", "method", "seed", "window"],
            as_index=False,
        )
        .agg(
            mean_absolute_td_error=("absolute_td_error", "mean"),
            mean_reward=("reward", "mean"),
        )
    )
    display(
        drift_windows.groupby(["method", "window"], as_index=False)
        .agg(
            mean_absolute_td_error=("mean_absolute_td_error", "mean"),
            mean_reward=("mean_reward", "mean"),
            n_trials=("trial_id", "nunique"),
        )
    )

    plot_learning_curves(
        drift_evaluations,
        metric="episode_return",
        x="checkpoint_episode",
        group="method",
        n_resamples=N_RESAMPLES,
    )
    plt.title("Greedy performance in the repeated-shock environment")
    plt.show()
    """),
    md(r"""
    **Act II reading rule.** A TD spike demonstrates surprise under the current
    estimates; it is not by itself a change-point detector or proof of
    adaptation. Recovery claims need the later window and held-out return.
    Because the clock restarts, these results do not describe one irreversible
    lifetime regime switch.
    """),
    md(r"""
    ## Act III — Memory: when position is not a state

    One wall follows a two-state Markov chain. We vary persistence while holding
    its stationary probability of being present at one half:

    $$
    \pi_{\mathrm{present}}=\frac{p_{01}}{p_{01}+1-p_{11}},
    \qquad \rho=p_{11}-p_{01}.
    $$

    Position-only Q-learning and wall-aware Q-learning receive the same seeds
    and budgets. The latter observes the position plus the current wall bit and
    therefore indexes the exact augmented MDP. This is a representation control,
    not a new backup-rule tournament. If the wall bit changes the best action,
    no position-only Q table can express both conditional policies.
    """),
    code("""
    MEMORY_EPISODES = 5 if SMOKE else (450 if QUICK else 2_500)
    MEMORY_SEEDS = (0,) if SMOKE else (tuple(range(5)) if QUICK else tuple(range(20)))
    WALL_EDGE = ((1, 2), (1, 3))
    PERSISTENCE = (
        (0.0, 0.50, 0.50),
        (0.6, 0.20, 0.80),
        (0.9, 0.05, 0.95),
    )
    memory_environments = []
    memory_metadata = {}
    for rho, p01, p11 in PERSISTENCE:
        parameters = {
            "shape": (3, 5),
            "start": (1, 0),
            "goals": {(1, 4): 4.0},
            "markov_walls": [
                MarkovWall(
                    edge=WALL_EDGE,
                    p01=p01,
                    p11=p11,
                    initial_probability=0.5,
                )
            ],
            "action_reliability": 0.98,
            "slip_weights": {"left": 0.45, "right": 0.45, "stay": 0.10},
            "step_reward": -0.03,
            "max_episode_steps": 60,
        }
        suffix = str(rho).replace(".", "_")
        for representation, kind in (
            ("position only", "stochastic_maze"),
            ("position + wall", "stochastic_maze_wall_state"),
        ):
            name = f"{representation.replace(' ', '_').replace('+', 'plus')}_rho_{suffix}"
            memory_environments.append(
                EnvironmentSpec(name=name, kind=kind, parameters=parameters)
            )
            memory_metadata[name] = (representation, rho)

    memory_agent = AgentSpec(
        name="q_learning",
        kind="q_learning",
        parameters={
            "learning_rate": 0.12,
            "gamma": 0.98,
            "epsilon": {
                "kind": "linear",
                "start": 0.30,
                "end": 0.03,
                "duration": max(40, MEMORY_EPISODES * 18),
            },
        },
    )
    memory_config = ExperimentConfig(
        name="markov_wall_memory",
        episodes=MEMORY_EPISODES,
        seeds=MEMORY_SEEDS,
        environments=tuple(memory_environments),
        agents=(memory_agent,),
        snapshot_interval=2 if SMOKE else (25 if QUICK else 50),
        exact_reference=True,
        policy_evaluation=PolicyEvaluationSpec(
            enabled=True,
            interval_episodes=2 if SMOKE else (75 if QUICK else 250),
            episodes_per_checkpoint=1 if SMOKE else (5 if QUICK else 10),
            include_initial=True,
            include_final=True,
            scenarios=(EvaluationScenario(name="matched_markov_wall"),),
        ),
        execution=ExecutionSpec(parallel_workers=1 if QUICK or SMOKE else 4),
        artifacts=ArtifactSpec(
            output_dir=RESULTS_DIR,
            table_format="auto",
            flush_rows=200 if SMOKE else 10_000,
            save_q_snapshots=True,
            step_retention=StepRetentionSpec(
                mode="all" if SMOKE else "sample",
                fraction=1.0 if SMOKE else (0.20 if QUICK else 0.05),
                keep_terminal=True,
                keep_events=True,
            ),
        ),
        tags={"question": "state_representation_under_markov_wall_memory"},
    )
    memory_result = run_or_reopen("memory", memory_config)

    def add_memory_factors(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        result["representation"] = result["environment"].map(
            lambda name: memory_metadata[name][0]
        )
        result["rho"] = result["environment"].map(lambda name: memory_metadata[name][1])
        return result

    memory_evaluations = add_memory_factors(
        evaluation_checkpoint_summary(memory_result.evaluations)
    )
    memory_final = last_checkpoint(memory_evaluations)
    memory_snapshots = add_memory_factors(memory_result.snapshots.query("episode >= 0"))
    exact_availability = (
        memory_snapshots.groupby(["representation", "rho"], as_index=False)[
            "exact_evaluation_available"
        ].all()
    )
    display(exact_availability)
    assert not exact_availability.query("representation == 'position only'")[
        "exact_evaluation_available"
    ].any()
    assert exact_availability.query("representation == 'position + wall'")[
        "exact_evaluation_available"
    ].all()
    """),
    code("""
    fig, axes = plt.subplots(1, len(PERSISTENCE), figsize=(16, 4.4), sharey=True)
    for ax, (rho, _, _) in zip(axes, PERSISTENCE, strict=True):
        sample = memory_final.loc[memory_final["rho"].eq(rho)].rename(
            columns={"checkpoint_episode": "episode"}
        )
        plot_final_distribution(
            sample,
            metric="episode_return",
            group="representation",
            last_episodes=1,
            ax=ax,
        )
        ax.set_title(rf"$\\rho={rho:g}$")
    fig.suptitle("Held-out return as wall memory increases")
    plt.tight_layout()
    plt.show()

    memory_contrast = paired_seed_contrast(
        memory_final,
        metric="episode_return",
        factor="representation",
        baseline="position only",
        comparison="position + wall",
        pair_by=("seed",),
        strata=("rho", "evaluation_scenario"),
        n_resamples=N_RESAMPLES,
        random_seed=23,
    )
    memory_contrast_summary = memory_contrast.summary.copy()
    memory_contrast_summary["label"] = memory_contrast_summary["rho"].map(
        lambda rho: rf"wall-aware minus position-only, $\\rho={rho:g}$"
    )
    display(
        memory_contrast_summary[
            ["rho", "n_pairs", "mean_difference", "ci_low", "ci_high", "win_rate"]
        ]
    )
    plot_paired_contrasts(
        memory_contrast_summary,
        label="label",
        difference_label="Held-out return difference",
        title="Value of observing the wall bit",
    )
    plt.show()

    plot_learning_curves(
        memory_snapshots.loc[memory_snapshots["representation"].eq("position + wall")],
        metric="q_error_inf",
        group="rho",
        log_y=True,
        n_resamples=N_RESAMPLES,
    )
    plt.title("Wall-aware Q-learning versus its augmented Q-star")
    plt.show()
    """),
    code("""
    target_rho = max(rho for rho, _, _ in PERSISTENCE)
    target_specs = [
        spec
        for spec in memory_config.environments
        if memory_metadata[spec.name][1] == target_rho
    ]
    position_spec = next(
        spec for spec in target_specs if memory_metadata[spec.name][0] == "position only"
    )
    wall_spec = next(
        spec for spec in target_specs if memory_metadata[spec.name][0] == "position + wall"
    )
    position_env = make_environment(position_spec)
    wall_env = make_environment(wall_spec)
    augmented_exact = value_iteration(wall_env.exact_mdp(), gamma=0.98)
    base_env = wall_env.unwrapped
    open_indices = np.arange(base_env.n_states) * 2
    closed_indices = open_indices + 1

    memory_final_rows, memory_q_tables = final_q_tables(memory_result)
    position_row = memory_final_rows.loc[
        memory_final_rows["environment"].eq(position_spec.name)
        & memory_final_rows["seed"].eq(POLICY_SEED)
    ].iloc[0]
    wall_row = memory_final_rows.loc[
        memory_final_rows["environment"].eq(wall_spec.name)
        & memory_final_rows["seed"].eq(POLICY_SEED)
    ].iloc[0]
    position_q = memory_q_tables[position_row["trial_id"]]
    wall_q = memory_q_tables[wall_row["trial_id"]]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), constrained_layout=True)
    plot_policy(
        augmented_exact.policy[open_indices],
        base_env,
        values=augmented_exact.values[open_indices],
        ax=axes[0, 0],
        title="Exact: wall absent",
    )
    plot_policy(
        augmented_exact.policy[closed_indices],
        base_env,
        values=augmented_exact.values[closed_indices],
        ax=axes[0, 1],
        title="Exact: wall present",
    )
    plot_policy(
        np.argmax(position_q, axis=1),
        position_env,
        values=np.max(position_q, axis=1),
        ax=axes[0, 2],
        title=f"Position only, seed {POLICY_SEED}",
    )
    plot_policy(
        np.argmax(wall_q[open_indices], axis=1),
        base_env,
        values=np.max(wall_q[open_indices], axis=1),
        ax=axes[1, 0],
        title="Wall-aware learned: absent",
    )
    plot_policy(
        np.argmax(wall_q[closed_indices], axis=1),
        base_env,
        values=np.max(wall_q[closed_indices], axis=1),
        ax=axes[1, 1],
        title="Wall-aware learned: present",
    )
    conflict = augmented_exact.policy[open_indices] != augmented_exact.policy[closed_indices]
    plot_policy(
        augmented_exact.policy[open_indices],
        base_env,
        values=conflict.astype(float),
        ax=axes[1, 2],
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        colorbar_label="optimal action depends on wall",
        title="Where memory changes the action",
    )
    plt.show()
    """),
    code("""
    memory_steps = add_memory_factors(memory_result.steps)
    decision_state = position_env.state_to_index[(1, 2)]
    wall_conditioned_td = memory_steps.loc[
        memory_steps["representation"].eq("position only")
        & memory_steps["rho"].eq(target_rho)
        & memory_steps["state"].eq(decision_state)
    ].copy()
    if wall_conditioned_td.empty:
        print("No retained decision-state rows in this small run.")
    else:
        wall_conditioned_td["wall"] = wall_conditioned_td["env_decision_wall_mask"].map(
            {0: "absent", 1: "present"}
        )
        per_trial_wall = (
            wall_conditioned_td.groupby(["trial_id", "seed", "wall"], as_index=False)
            .agg(
                mean_absolute_td_error=("absolute_td_error", "mean"),
                visits=("td_error", "size"),
            )
        )
        display(per_trial_wall)
        samples = [
            per_trial_wall.loc[
                per_trial_wall["wall"].eq(label), "mean_absolute_td_error"
            ].to_numpy()
            for label in ("absent", "present")
        ]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax.boxplot(samples, tick_labels=["wall absent", "wall present"], showfliers=False)
        ax.set(
            ylabel="per-seed mean absolute TD error",
            title="Position-only ambiguity at the blocked edge",
        )
        ax.grid(axis="y", alpha=0.22)
        plt.show()
    """),
    md(r"""
    ## Synthesis

    Better backups help when the problem is estimation. They cannot manufacture
    stationarity or restore information omitted from the state.

    - Act I is the clean algorithm comparison: stationary, fully observed, exact,
      paired, and evaluated without updates.
    - Act II measures response to a repeated target shift. It supports claims
      about event-aligned surprise and recovery, not a one-time lifetime change.
    - Act III changes temporal memory without changing average wall frequency.
      The gap between position-only and wall-aware learning is the value of
      state information, not evidence that one backup rule is universally best.

    QUICK mode validates these mechanics. For stable confidence intervals, use
    the full risk YAML from the command line and set QUICK to False for the two
    targeted notebook runs. Reopen the resulting directories through
    EXISTING_RUNS whenever you want to iterate on the analysis.
    """),
]


_shortcut_or_shelter_raw_analysis = [
    md(r"""
    # Shortcut or shelter?

    ## Finding the policy boundary in a noisy maze

    A short corridor reaches the goal quickly, but lateral execution errors can
    enter hazards above or below it. A longer southern route faces the same
    actuator noise; a wall and the grid boundary convert its lateral errors into
    delays rather than danger.

    We ask three deliberately separate questions:

    1. At what intended-action reliability does the **exact** optimal policy
       switch routes?
    2. How do Q-learning, SARSA, and Expected SARSA approach that boundary?
    3. Does continuing exploration after training change which route should be
       valued?

    Two consequence laws use exactly the same topology. A recoverable hazard
    charges a penalty and lets the episode continue; a lethal hazard charges a
    large penalty and ends the episode. The lethal case is not softened to make
    a prettier plot: its exact transition really is compressed close to perfect
    control.

    This notebook is a controlled study of backup targets, not an algorithm
    leaderboard. `QUICK=True` is a mechanics check. Publication claims require
    the versioned full configurations, their complete seed panels, and the saved
    Protocol-v2 manifests.
    """),
    code("""
    from __future__ import annotations

    from dataclasses import replace
    import os
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from rllab.environments import RiskyCorridorEnv
    from rllab.experiments import (
        Experiment,
        ExperimentConfig,
        ExperimentResult,
        ExecutionSpec,
        RunStore,
        estimate_run,
    )
    from rllab.metrics import bootstrap_confidence_interval
    from rllab.theory import (
        epsilon_soft_value_iteration,
        policy_evaluation as exact_policy_evaluation,
        value_iteration,
    )
    from rllab.visualization import plot_maze, plot_policy, plot_transition_noise

    SMOKE = os.environ.get("RL_LAB_NOTEBOOK_SMOKE") == "1"
    QUICK = True
    SHOW_PROGRESS = False  # reliable in every Jupyter frontend; the CLI has live progress
    GAMMA = 0.98
    PERSISTENT_EPSILON = 0.10
    INITIAL_Q = 8.0
    RECOVERABLE_PENALTY = -0.50  # about eight ordinary movement costs
    LETHAL_PENALTY = -8.0
    METHOD_ORDER = ("q_learning", "sarsa", "expected_sarsa")
    METHOD_LABELS = {
        "q_learning": "Q-learning",
        "sarsa": "SARSA",
        "expected_sarsa": "Expected SARSA",
    }
    METHOD_COLORS = {
        "q_learning": "#0072B2",
        "sarsa": "#D55E00",
        "expected_sarsa": "#009E73",
    }
    DISAGREEMENT_POINTS = {"recoverable": 0.825, "lethal": 1.0}
    EXISTING_RUNS: dict[str, str | Path | None] = {
        "recoverable": None,
        "lethal": None,
        "annealed": None,
    }

    def find_repo_root(start: Path) -> Path:
        for candidate in (start.resolve(), *start.resolve().parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "src" / "rllab").exists():
                return candidate
        raise FileNotFoundError("Run this notebook from inside the rl-lab repository.")

    REPO_ROOT = find_repo_root(Path.cwd())
    configured_results = os.environ.get("RL_LAB_NOTEBOOK_RESULTS")
    RESULTS_DIR = Path(configured_results) if configured_results else REPO_ROOT / "results"
    N_RESAMPLES = 50 if SMOKE else (500 if QUICK else 2_000)
    REPRESENTATIVE_SEED = 0
    RUN_PROFILE = "SMOKE" if SMOKE else ("QUICK" if QUICK else "FULL")
    EMPIRICAL_TITLE = "" if RUN_PROFILE == "FULL" else f" — {RUN_PROFILE}: NOT FOR INFERENCE"
    if RUN_PROFILE != "FULL":
        print(
            f"*** {RUN_PROFILE} PROFILE: validates mechanics only; "
            "do not publish or interpret learned boundaries. ***"
        )

    available_styles = set(plt.style.available)
    plot_style = next(
        (
            style
            for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot")
            if style in available_styles
        ),
        "default",
    )
    plt.style.use(plot_style)
    np.set_printoptions(precision=4, suppress=True)
    """),
    md(r"""
    ## 1. Research contract

    The fork is the start state. An intended EAST action expresses the corridor
    policy; SOUTH expresses the shelter policy. NORTH, WEST, and exact ties stay
    visible as `other` rather than being silently forced into either route.

    The primary theoretical estimand is

    $$
    \Delta^*(p)=Q^*(s_0,\mathrm{EAST})-Q^*(s_0,\mathrm{SOUTH}),
    $$

    and its zero crossing $p^*$. The primary learned estimand is the fraction of
    independent training seeds whose final greedy action selects EAST. Training
    seeds—not episodes and not repeated held-out rollouts—are the independent
    units.

    Equal interaction budgets matter here. Shelter episodes are longer, so an
    equal episode count would give shelter-taking methods more updates.

    Every learner also starts with `Q(s,a)=8` for every state and action. This
    route-neutral optimism is a declared coverage device, not a guess that one
    route is better. Without it, an early random preference can make the long
    alternative route exponentially hard to rediscover under epsilon-greedy
    control. The exact oracles are unchanged; the initialization only affects
    how efficiently the learners gather evidence about both routes.
    """),
    code("""
    research_contract = pd.DataFrame(
        [
            {
                "claim": "In the declared high-reliability domain, the greedy optimum has one algorithm-independent EAST/SOUTH boundary.",
                "evidence": "exact start-action gap and numerical zero crossing",
            },
            {
                "claim": "Persistent exploration changes the continuation policy being valued.",
                "evidence": "greedy versus epsilon-soft exact boundary",
            },
            {
                "claim": "Q-learning and on-policy backups can approach different persistent-epsilon boundaries.",
                "evidence": "seed-level learned corridor probability",
            },
            {
                "claim": "Expected SARSA should mainly reduce target variance relative to SARSA.",
                "evidence": "boundary agreement plus a controlled exact backup-variance decomposition",
            },
        ]
    )
    display(research_contract)
    """),
    md(r"""
    ## 2. Two kinds of noise, two exact objectives

    Reliability $p<1$ belongs to the environment: an intended action can be
    executed incorrectly even after learning ends. Epsilon belongs to the
    behavior policy: with probability $\epsilon$ the agent deliberately samples
    an action from the uniform distribution.

    Q-learning bootstraps with $\max_a Q(s',a)$ and therefore targets greedy
    control. SARSA samples the next exploratory action. Expected SARSA replaces
    that sample by its conditional expectation. With persistent epsilon, the
    exact on-policy control operator is

    $$
    V_\epsilon(s)=(1-\epsilon)\max_a Q(s,a)
      +\epsilon\frac{1}{|\mathcal A|}\sum_a Q(s,a).
    $$

    Turning exploration off after training and continuing to explore online are
    therefore different deployment questions. We evaluate both without allowing
    either evaluation clone to update.
    """),
    code("""
    def corridor_environment(
        hazard_mode: str,
        reliability: float,
    ) -> RiskyCorridorEnv:
        return RiskyCorridorEnv(
            corridor_reliability=float(reliability),
            hazard_mode=hazard_mode,
            recoverable_hazard_penalty=RECOVERABLE_PENALTY,
            lethal_hazard_penalty=LETHAL_PENALTY,
        )

    diagram_env = corridor_environment("lethal", 0.90)
    assert diagram_env.fork_state == diagram_env.start_state
    assert diagram_env.shelter_states.isdisjoint(diagram_env.hazard_positions)
    assert all(edge in diagram_env.static_walls for edge in diagram_env.shelter_walls)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_maze(diagram_env, ax=axes[0], title="One topology, two consequence laws")
    axes[0].plot(
        [0, 8], [1, 1], color="#D55E00", linewidth=3, alpha=0.72, label="exposed corridor"
    )
    axes[0].plot(
        [0, 0, 8, 8], [1, 3, 3, 1],
        color="#0072B2", linewidth=3, alpha=0.72, label="protected shelter",
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    plot_transition_noise(
        diagram_env,
        ax=axes[1],
        title="Same actuator reliability on both routes",
    )
    plt.tight_layout()
    plt.show()
    """),
    md(r"""
    A recoverable encounter costs `-0.50`, a little more than eight ordinary
    movement costs, and the episode continues. A lethal encounter costs `-8`
    and terminates, forfeiting the chance to collect the `+8` goal. This is a
    qualitative change in the process, not merely a cosmetic change of scale.
    """),
    code("""
    def oracle_solution(hazard_mode: str, reliability: float, epsilon: float):
        env = corridor_environment(hazard_mode, reliability)
        model = env.exact_mdp()
        if epsilon == 0.0:
            solution = value_iteration(model, gamma=GAMMA)
            greedy_policy = solution.policy
        else:
            solution = epsilon_soft_value_iteration(
                model,
                epsilon=epsilon,
                gamma=GAMMA,
            )
            greedy_policy = solution.greedy_policy
        if not solution.converged:
            raise RuntimeError("Exact value iteration did not converge")
        return env, solution, greedy_policy

    def exact_gap(hazard_mode: str, reliability: float, epsilon: float) -> float:
        env, solution, _ = oracle_solution(hazard_mode, reliability, epsilon)
        start = env.state_to_index[env.fork_state]
        return float(
            solution.q_values[start, int(env.corridor_action)]
            - solution.q_values[start, int(env.shelter_action)]
        )

    def crossing(x: np.ndarray, y: np.ndarray) -> float:
        indices = np.flatnonzero((y[:-1] <= 0.0) & (y[1:] > 0.0))
        if not len(indices):
            return float("nan")
        index = int(indices[0])
        weight = -y[index] / (y[index + 1] - y[index])
        return float(x[index] + weight * (x[index + 1] - x[index]))

    oracle_points = 41 if SMOKE else (301 if QUICK else 1_001)
    oracle_grid = np.linspace(0.55, 1.0, oracle_points)
    oracle_rows = []
    for hazard_mode in ("recoverable", "lethal"):
        for epsilon in (0.0, 0.01, PERSISTENT_EPSILON):
            for reliability in oracle_grid:
                oracle_rows.append(
                    {
                        "hazard_mode": hazard_mode,
                        "epsilon": epsilon,
                        "reliability": float(reliability),
                        "start_action_gap": exact_gap(
                            hazard_mode,
                            float(reliability),
                            epsilon,
                        ),
                    }
                )
    oracle_frame = pd.DataFrame(oracle_rows)
    threshold_rows = []
    for (hazard_mode, epsilon), sample in oracle_frame.groupby(
        ["hazard_mode", "epsilon"], sort=False
    ):
        threshold_rows.append(
            {
                "hazard_mode": hazard_mode,
                "epsilon": epsilon,
                "threshold": crossing(
                    sample["reliability"].to_numpy(),
                    sample["start_action_gap"].to_numpy(),
                ),
                "gap_at_perfect_control": float(sample.iloc[-1]["start_action_gap"]),
            }
        )
    exact_thresholds = pd.DataFrame(threshold_rows)
    display(exact_thresholds)
    """),
    code("""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    epsilon_styles = {0.0: "-", 0.01: "--", PERSISTENT_EPSILON: ":"}
    for ax, hazard_mode in zip(axes, ("recoverable", "lethal"), strict=True):
        sample = oracle_frame.loc[oracle_frame["hazard_mode"].eq(hazard_mode)]
        if hazard_mode == "lethal":
            sample = sample.loc[sample["reliability"].ge(0.965)]
        for epsilon, line in sample.groupby("epsilon", sort=True):
            label = "greedy oracle" if epsilon == 0 else f"epsilon-soft, ε={epsilon:g}"
            ax.plot(
                line["reliability"],
                line["start_action_gap"],
                linestyle=epsilon_styles[float(epsilon)],
                linewidth=2.2,
                label=label,
            )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set(
            xlabel="intended-action reliability p",
            ylabel="exact EAST - SOUTH action value",
            title=f"{hazard_mode.title()} hazards",
        )
        ax.legend(frameon=False)
        ax.grid(alpha=0.22)
    plt.tight_layout()
    plt.show()
    """),
    md(r"""
    ## 3. Matched, interaction-budgeted learning experiment

    Exact roots determined where the grids needed resolution; small pilot runs
    then calibrated coverage, alpha, and the 100,000-interaction budget. The
    resulting YAML files are the pre-specified full-run design—not a claim that
    this pilot-informed study was preregistered. The recoverable grid resolves
    both theoretical transitions. The lethal grid magnifies the narrow greedy
    transition near perfect control. Every method receives the same interaction
    budget, root-seed panel, alpha, gamma, and epsilon, with isolated random
    streams.

    The two held-out scenarios share environment seeds:

    - `frozen_greedy`: exploration is disabled;
    - `continuing_behavior`: the learned exploration schedule is sampled but no
      updates are performed.
    """),
    code("""
    FULL_CONFIG_PATHS = {
        "recoverable": REPO_ROOT / "configs" / "shortcut_or_shelter_recoverable.yaml",
        "lethal": REPO_ROOT / "configs" / "shortcut_or_shelter_lethal.yaml",
    }
    SMOKE_GRIDS = {
        "recoverable": (0.79, 0.83),
        "lethal": (0.985, 1.0),
    }
    QUICK_GRIDS = {
        "recoverable": (0.70, 0.79, 0.805, 0.825, 0.84, 0.85, 0.86, 0.90, 1.0),
        "lethal": (0.970, 0.982, 0.987, 0.989, 0.991, 0.995, 1.0),
    }

    def configured_main_experiment(hazard_mode: str) -> ExperimentConfig:
        full = ExperimentConfig.from_yaml(FULL_CONFIG_PATHS[hazard_mode])
        if not (SMOKE or QUICK):
            return full
        grid = SMOKE_GRIDS[hazard_mode] if SMOKE else QUICK_GRIDS[hazard_mode]
        seeds = (0,) if SMOKE else tuple(range(4))
        interaction_steps = 80 if SMOKE else 2_500
        return replace(
            full,
            seeds=seeds,
            sweep={"environment.corridor_reliability": tuple(grid)},
            total_interaction_steps=interaction_steps,
            snapshot_interval=1_000_000,
            snapshot_step_interval=20 if SMOKE else 625,
            policy_evaluation=replace(
                full.policy_evaluation,
                interval_episodes=1_000_000,
                episodes_per_checkpoint=1 if SMOKE else 4,
                include_initial=False,
                include_final=True,
            ),
            execution=ExecutionSpec(parallel_workers=1),
            artifacts=replace(
                full.artifacts,
                output_dir=RESULTS_DIR,
                flush_rows=200 if SMOKE else 5_000,
            ),
        )

    main_configs = {
        hazard_mode: configured_main_experiment(hazard_mode)
        for hazard_mode in ("recoverable", "lethal")
    }
    assert all(
        float(agent.parameters["initial_q"]) == INITIAL_Q
        and float(agent.parameters["epsilon"]) == PERSISTENT_EPSILON
        and float(agent.parameters["learning_rate"]) == 0.05
        for config in main_configs.values()
        for agent in config.agents
    )
    preflight = pd.DataFrame(
        [
            {"hazard_mode": hazard_mode, **estimate_run(config).as_dict()}
            for hazard_mode, config in main_configs.items()
        ]
    )
    display(preflight)
    """),
    code("""
    def run_or_reopen(label: str, config: ExperimentConfig) -> ExperimentResult:
        existing = EXISTING_RUNS[label]
        if existing is not None:
            path = Path(existing)
            path = path if path.is_absolute() else REPO_ROOT / path
            store = RunStore.open(path)
            if store.manifest.experiment_name != config.name:
                raise ValueError(
                    f"{label} run is {store.manifest.experiment_name!r}, expected {config.name!r}"
                )
            print(f"Reopened {label}: {store.run_directory}")
            return ExperimentResult(
                experiment_id=store.manifest.run_id,
                run_directory=store.run_directory,
                metadata={"reopened": True},
            )
        print(f"Starting {label}: {len(config.trials())} matched trials")
        result = Experiment(config).run(persist=True, progress=SHOW_PROGRESS)
        print(f"Completed {label}: {result.run_directory}")
        return result

    main_results = {
        hazard_mode: run_or_reopen(hazard_mode, config)
        for hazard_mode, config in main_configs.items()
    }
    """),
    code("""
    def trial_design(config: ExperimentConfig) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "trial_id": trial.trial_id,
                    "seed": trial.seed,
                    "agent": trial.agent.name,
                    "hazard_mode": trial.environment.parameters["hazard_mode"],
                    "reliability": float(
                        trial.environment.parameters.get(
                            "corridor_reliability",
                            trial.environment.parameters.get("action_reliability", 1.0),
                        )
                    ),
                }
                for trial in config.trials()
            ]
        )

    def concatenate_tables(frames: list[pd.DataFrame]) -> pd.DataFrame:
        # Parquet schemas can contain diagnostics that are entirely missing in
        # one hazard condition. Drop only per-frame all-missing columns before
        # concatenation; this avoids dtype guessing warnings while preserving a
        # column wherever at least one condition actually observed it.
        return pd.concat(
            [frame.dropna(axis=1, how="all") for frame in frames],
            ignore_index=True,
            sort=False,
        )

    def stream_training_integrity(
        result: ExperimentResult,
    ) -> tuple[pd.DataFrame, int]:
        '''Reduce episode rows to one final observed-step count per trial.'''
        final_steps: dict[str, int] = {}
        row_count = 0
        for batch in result.iter_table(
            "training_episodes",
            columns=("trial_id", "observed_step_count"),
            batch_size=50_000,
        ):
            row_count += len(batch)
            if batch.empty:
                continue
            if batch[["trial_id", "observed_step_count"]].isna().any().any():
                raise ValueError("Training integrity columns contain missing values")
            batch_final = batch.groupby("trial_id")["observed_step_count"].max()
            for trial_id, observed_steps in batch_final.items():
                key = str(trial_id)
                final_steps[key] = max(final_steps.get(key, 0), int(observed_steps))
        summary = pd.DataFrame(
            sorted(final_steps.items()),
            columns=["trial_id", "observed_step_count"],
        )
        return summary, row_count

    main_design = concatenate_tables(
        [trial_design(config) for config in main_configs.values()],
    )
    training_integrity_parts = {
        hazard_mode: stream_training_integrity(result)
        for hazard_mode, result in main_results.items()
    }
    main_training = concatenate_tables(
        [summary for summary, _ in training_integrity_parts.values()],
    )
    training_episode_row_count = sum(
        row_count for _, row_count in training_integrity_parts.values()
    )
    main_evaluations = concatenate_tables(
        [result.evaluations for result in main_results.values()],
    )
    assert set(main_design["agent"]) == set(METHOD_ORDER)
    assert main_design["trial_id"].is_unique
    assert main_training["trial_id"].is_unique
    assert set(main_evaluations["evaluation_policy_mode"]) == {"greedy", "behavior"}
    paired_panels = main_evaluations.pivot_table(
        index=["trial_id", "evaluation_episode"],
        columns="evaluation_policy_mode",
        values="evaluation_seed",
        aggfunc="first",
    ).dropna()
    assert (paired_panels["greedy"] == paired_panels["behavior"]).all()
    final_steps = main_training.set_index("trial_id")["observed_step_count"]
    expected_steps = {
        trial.trial_id: trial.total_interaction_steps
        for config in main_configs.values()
        for trial in config.trials()
    }
    if not (SMOKE or QUICK):
        assert set(expected_steps.values()) == {100_000}, (
            "The publication design requires exactly 100,000 interactions per trial"
        )
    assert set(final_steps.index) == set(expected_steps), (
        "Training rows do not cover the exact configured trial panel"
    )
    step_mismatches = {
        trial_id: (int(final_steps[trial_id]), int(budget))
        for trial_id, budget in expected_steps.items()
        if int(final_steps[trial_id]) != int(budget)
    }
    assert not step_mismatches, f"Interaction-budget mismatches: {step_mismatches}"
    print(
        "trials / streamed training episode rows / held-out episodes:",
        len(main_design), training_episode_row_count, len(main_evaluations),
    )
    """),
    code("""
    def classify_action(q_row: np.ndarray, env: RiskyCorridorEnv) -> tuple[str, int]:
        maximizers = np.flatnonzero(np.isclose(q_row, np.max(q_row), rtol=1e-10, atol=1e-12))
        if len(maximizers) != 1:
            return "tie", -1
        action = int(maximizers[0])
        if action == int(env.corridor_action):
            return "corridor", action
        if action == int(env.shelter_action):
            return "shelter", action
        return "other", action

    def final_policy_rows(
        result: ExperimentResult,
        config: ExperimentConfig,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        design = trial_design(config)
        final = (
            result.snapshots.query("episode >= 0")
            .sort_values(["trial_id", "global_step", "episode"])
            .groupby("trial_id", as_index=False)
            .tail(1)
            .merge(design, on=["trial_id", "seed", "agent"], validate="one_to_one")
        )
        rows = []
        tables: dict[str, np.ndarray] = {}
        for row in final.itertuples(index=False):
            table = result.q_snapshots(row.trial_id, keys=(row.snapshot_key,))[row.snapshot_key]
            tables[row.trial_id] = table
            env = corridor_environment(row.hazard_mode, row.reliability)
            start = env.state_to_index[env.fork_state]
            choice, action = classify_action(table[start], env)
            model = env.exact_mdp()
            learned_policy = np.argmax(table, axis=1)
            learned_values = exact_policy_evaluation(
                model,
                learned_policy,
                gamma=GAMMA,
                method="direct",
            ).values
            optimum = value_iteration(model, gamma=GAMMA)
            learned_gap = float(
                table[start, int(env.corridor_action)]
                - table[start, int(env.shelter_action)]
            )
            target_epsilon = 0.0 if row.agent == "q_learning" else PERSISTENT_EPSILON
            target_gap = exact_gap(row.hazard_mode, row.reliability, target_epsilon)
            rows.append(
                {
                    "trial_id": row.trial_id,
                    "seed": row.seed,
                    "agent": row.agent,
                    "hazard_mode": row.hazard_mode,
                    "reliability": row.reliability,
                    "global_step": row.global_step,
                    "choice": choice,
                    "greedy_action": action,
                    "corridor_selected": float(choice == "corridor"),
                    "start_action_gap": learned_gap,
                    "target_oracle_gap": target_gap,
                    "absolute_gap_error": abs(learned_gap - target_gap),
                    "exact_deployment_regret": max(
                        0.0,
                        float(optimum.values[start] - learned_values[start]),
                    ),
                }
            )
        return pd.DataFrame(rows), tables

    final_frames = []
    final_q_tables: dict[str, np.ndarray] = {}
    for hazard_mode in ("recoverable", "lethal"):
        frame, tables = final_policy_rows(
            main_results[hazard_mode],
            main_configs[hazard_mode],
        )
        final_frames.append(frame)
        final_q_tables.update(tables)
    final_choices = pd.concat(final_frames, ignore_index=True)
    display(
        final_choices.groupby(["hazard_mode", "agent", "choice"], as_index=False)
        .size()
        .rename(columns={"size": "training_seeds"})
    )
    unresolved = final_choices["choice"].isin(["other", "tie"])
    unresolved_rate = float(unresolved.mean())
    gap_calibration = (
        final_choices.groupby(["hazard_mode", "agent"], as_index=False)
        .agg(
            mean_absolute_gap_error=("absolute_gap_error", "mean"),
            median_absolute_gap_error=("absolute_gap_error", "median"),
        )
    )
    display(gap_calibration)
    if unresolved.any():
        message = (
            f"{unresolved.sum()} of {len(final_choices)} final policies "
            f"({unresolved_rate:.1%}) have a tied or off-route fork action."
        )
        if not (SMOKE or QUICK):
            raise RuntimeError(
                message + " The publication run is under-resolved; do not interpret its boundary."
            )
        print("UNDERTRAINING WARNING:", message, "QUICK/SMOKE output is mechanics-only.")

    endpoint_rows = []
    endpoint_expectations = (
        ("recoverable", "min", METHOD_ORDER, 0.0),
        ("recoverable", "max", METHOD_ORDER, 1.0),
        ("lethal", "min", METHOD_ORDER, 0.0),
        ("lethal", "max", ("q_learning",), 1.0),
        ("lethal", "max", ("sarsa", "expected_sarsa"), 0.0),
    )
    for hazard_mode, endpoint, agents, expected in endpoint_expectations:
        mode_rows = final_choices.loc[final_choices["hazard_mode"].eq(hazard_mode)]
        reliability = (
            float(mode_rows["reliability"].min())
            if endpoint == "min"
            else float(mode_rows["reliability"].max())
        )
        for agent in agents:
            observed = float(
                mode_rows.loc[
                    mode_rows["agent"].eq(agent)
                    & mode_rows["reliability"].eq(reliability),
                    "corridor_selected",
                ].mean()
            )
            endpoint_rows.append(
                {
                    "hazard_mode": hazard_mode,
                    "reliability": reliability,
                    "agent": agent,
                    "expected_corridor_fraction": expected,
                    "observed_corridor_fraction": observed,
                    "passes": observed >= 0.8 if expected == 1.0 else observed <= 0.2,
                }
            )
    endpoint_calibration = pd.DataFrame(endpoint_rows)
    display(endpoint_calibration)
    if RUN_PROFILE == "FULL" and not endpoint_calibration["passes"].all():
        raise RuntimeError(
            "The full run failed its pre-specified endpoint calibration; "
            "do not interpret the interior route boundary."
        )
    """),
    code("""
    def binary_seed_summary(
        frame: pd.DataFrame,
        *,
        value: str,
        groups: tuple[str, ...],
    ) -> pd.DataFrame:
        z = 1.959963984540054
        rows = []
        for keys, sample in frame.groupby(list(groups), dropna=False, sort=True):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            values = sample[value].to_numpy(dtype=float)
            if not np.isin(values, (0.0, 1.0)).all():
                raise ValueError(f"{value} must be binary for a Wilson interval")
            n = len(values)
            proportion = float(values.mean())
            denominator = 1.0 + z**2 / n
            center = (proportion + z**2 / (2.0 * n)) / denominator
            radius = (
                z
                * np.sqrt(proportion * (1.0 - proportion) / n + z**2 / (4.0 * n**2))
                / denominator
            )
            rows.append(
                {
                    **dict(zip(groups, key_tuple, strict=True)),
                    "mean": proportion,
                    "ci_low": max(0.0, center - radius),
                    "ci_high": min(1.0, center + radius),
                    "n_seeds": int(sample["seed"].nunique()),
                }
            )
        return pd.DataFrame(rows)

    def continuous_seed_summary(
        frame: pd.DataFrame,
        *,
        value: str,
        groups: tuple[str, ...],
        seed: int,
    ) -> pd.DataFrame:
        rows = []
        for keys, sample in frame.groupby(list(groups), dropna=False, sort=True):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            values = sample[value].to_numpy(dtype=float)
            low, high = bootstrap_confidence_interval(
                values,
                n_resamples=N_RESAMPLES,
                seed=seed,
            )
            rows.append(
                {
                    **dict(zip(groups, key_tuple, strict=True)),
                    "mean": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "n_seeds": int(sample["seed"].nunique()),
                }
            )
        return pd.DataFrame(rows)

    corridor_summary = binary_seed_summary(
        final_choices,
        value="corridor_selected",
        groups=("hazard_mode", "agent", "reliability"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.suptitle("Learned route selection" + EMPIRICAL_TITLE, fontweight="bold")
    for ax, hazard_mode in zip(axes, ("recoverable", "lethal"), strict=True):
        sample = corridor_summary.loc[corridor_summary["hazard_mode"].eq(hazard_mode)]
        for agent in METHOD_ORDER:
            line = sample.loc[sample["agent"].eq(agent)].sort_values("reliability")
            ax.plot(
                line["reliability"], line["mean"], marker="o",
                color=METHOD_COLORS[agent], label=METHOD_LABELS[agent],
            )
            ax.fill_between(
                line["reliability"], line["ci_low"], line["ci_high"],
                color=METHOD_COLORS[agent], alpha=0.14,
            )
        oracle = exact_thresholds.loc[
            exact_thresholds["hazard_mode"].eq(hazard_mode)
            & exact_thresholds["epsilon"].eq(0.0),
            "threshold",
        ].iloc[0]
        ax.axvline(oracle, color="black", linestyle="--", label="greedy exact boundary")
        soft = exact_thresholds.loc[
            exact_thresholds["hazard_mode"].eq(hazard_mode)
            & exact_thresholds["epsilon"].eq(PERSISTENT_EPSILON),
            "threshold",
        ].iloc[0]
        if np.isfinite(soft):
            ax.axvline(soft, color="0.35", linestyle=":", label="epsilon-soft boundary")
        else:
            ax.text(
                0.02, 0.04, "epsilon-soft oracle never selects corridor",
                transform=ax.transAxes, fontsize=9,
            )
        ax.set(
            title=f"{hazard_mode.title()} hazards",
            xlabel="intended-action reliability p",
            ylabel="fraction of training seeds selecting corridor",
            ylim=(-0.04, 1.04),
        )
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()
    """),
    md(r"""
    The exact oracle changes action discretely. The empirical transition is
    softened because independently trained finite-sample Q tables disagree near
    a small action gap. A smooth-looking seed proportion is uncertainty across
    learned policies, not a stochastic mixture chosen by one greedy policy.

    Protocol-v2's generic `q_error` fields always use the greedy exact MDP. We do
    not use those fields to score SARSA or Expected SARSA against an objective
    they did not target. `target_oracle_gap` above is greedy for Q-learning and
    epsilon-soft for the two on-policy methods; `exact_deployment_regret` asks a
    separate question about the final greedy policy after exploration is turned
    off.
    """),
    code("""
    def checkpoint_policy_rows(
        result: ExperimentResult,
        config: ExperimentConfig,
        targets: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    ) -> pd.DataFrame:
        design = trial_design(config)
        snapshots = result.snapshots.query("episode >= 0").merge(
            design,
            on=["trial_id", "seed", "agent"],
            validate="many_to_one",
        )
        rows = []
        budget = int(config.total_interaction_steps or 0)
        if budget <= 0:
            raise ValueError("checkpoint boundary analysis requires an interaction budget")
        for trial_id, sample in snapshots.groupby("trial_id", sort=False):
            sample = sample.sort_values("global_step")
            selected = []
            for target in targets:
                eligible = sample.loc[sample["global_step"].le(target * budget)]
                selected.append(eligible.iloc[-1] if len(eligible) else sample.iloc[0])
            keys = tuple(dict.fromkeys(str(row["snapshot_key"]) for row in selected))
            tables = result.q_snapshots(trial_id, keys=keys)
            for target, row in zip(targets, selected, strict=True):
                env = corridor_environment(row["hazard_mode"], float(row["reliability"]))
                start = env.state_to_index[env.fork_state]
                table = tables[str(row["snapshot_key"])]
                choice, _ = classify_action(table[start], env)
                rows.append(
                    {
                        "trial_id": trial_id,
                        "seed": int(row["seed"]),
                        "agent": row["agent"],
                        "hazard_mode": row["hazard_mode"],
                        "reliability": float(row["reliability"]),
                        "progress": target,
                        "target_global_step": int(target * budget),
                        "observed_global_step": int(row["global_step"]),
                        "step_lag": int(target * budget) - int(row["global_step"]),
                        "corridor_selected": float(choice == "corridor"),
                    }
                )
        return pd.DataFrame(rows)

    checkpoint_choices = pd.concat(
        [
            checkpoint_policy_rows(main_results[mode], main_configs[mode])
            for mode in ("recoverable", "lethal")
        ],
        ignore_index=True,
    )
    assert checkpoint_choices["step_lag"].eq(0).all(), (
        "Boundary checkpoints must be captured at exact interaction counts; "
        "check snapshot_step_interval."
    )
    stability_wide = checkpoint_choices.loc[
        checkpoint_choices["progress"].isin([0.75, 1.0])
    ].pivot(
        index=["trial_id", "seed", "agent", "hazard_mode", "reliability"],
        columns="progress",
        values="corridor_selected",
    )
    stability_wide["changed_last_quarter"] = stability_wide[0.75] != stability_wide[1.0]
    stability_summary = (
        stability_wide.reset_index()
        .groupby(["hazard_mode", "agent", "reliability"], as_index=False)
        .agg(
            changed_fraction=("changed_last_quarter", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )
    display(stability_summary)

    endpoint_stability = []
    for hazard_mode in ("recoverable", "lethal"):
        sample = stability_summary.loc[stability_summary["hazard_mode"].eq(hazard_mode)]
        for reliability in (float(sample["reliability"].min()), float(sample["reliability"].max())):
            endpoint_stability.append(sample.loc[sample["reliability"].eq(reliability)])
    endpoint_stability = pd.concat(endpoint_stability, ignore_index=True)
    if RUN_PROFILE == "FULL" and endpoint_stability["changed_fraction"].gt(0.2).any():
        raise RuntimeError(
            "More than 20% of seeds changed route at a calibration endpoint in "
            "the final quarter; the finite-budget boundary is not stable enough to report."
        )
    checkpoint_summary = binary_seed_summary(
        checkpoint_choices,
        value="corridor_selected",
        groups=("hazard_mode", "agent", "progress", "reliability"),
    )

    def empirical_half_boundary(sample: pd.DataFrame) -> tuple[float, int]:
        ordered = sample.sort_values("reliability")
        x = ordered["reliability"].to_numpy(dtype=float)
        y = ordered["mean"].to_numpy(dtype=float)
        violations = int(np.count_nonzero(np.diff(y) < -0.05))
        indices = np.flatnonzero(y >= 0.5)
        if not len(indices):
            return float("nan"), violations
        index = int(indices[0])
        if index == 0:
            return float(x[0]), violations
        if y[index] == y[index - 1]:
            return float(x[index]), violations
        weight = (0.5 - y[index - 1]) / (y[index] - y[index - 1])
        return float(x[index - 1] + weight * (x[index] - x[index - 1])), violations

    def seed_block_boundary_bootstrap(
        frame: pd.DataFrame,
        *,
        n_resamples: int,
        seed: int,
    ) -> pd.DataFrame:
        seeds = np.sort(frame["seed"].unique())
        rng = np.random.default_rng(seed)
        rows = []
        for bootstrap_id in range(n_resamples):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            blocks = []
            for block_id, sampled_seed in enumerate(sampled_seeds):
                block = frame.loc[frame["seed"].eq(sampled_seed)].copy()
                block["bootstrap_block"] = block_id
                blocks.append(block)
            sampled = pd.concat(blocks, ignore_index=True)
            curve = (
                sampled.groupby(["hazard_mode", "agent", "reliability"], as_index=False)[
                    "corridor_selected"
                ]
                .mean()
                .rename(columns={"corridor_selected": "mean"})
            )
            for (hazard_mode, agent), panel in curve.groupby(
                ["hazard_mode", "agent"], sort=False
            ):
                boundary, violations = empirical_half_boundary(panel)
                rows.append(
                    {
                        "bootstrap_id": bootstrap_id,
                        "hazard_mode": hazard_mode,
                        "agent": agent,
                        "half_boundary": boundary,
                        "right_censored": not np.isfinite(boundary),
                        "monotonicity_violations": violations,
                    }
                )
        return pd.DataFrame(rows)

    learned_boundary_rows = []
    for keys, sample in checkpoint_summary.groupby(
        ["hazard_mode", "agent", "progress"], sort=False
    ):
        boundary, violations = empirical_half_boundary(sample)
        learned_boundary_rows.append(
            {
                "hazard_mode": keys[0],
                "agent": keys[1],
                "progress": keys[2],
                "half_boundary": boundary,
                "monotonicity_violations": violations,
            }
        )
    learned_boundaries = pd.DataFrame(learned_boundary_rows)

    boundary_bootstrap = seed_block_boundary_bootstrap(
        final_choices,
        n_resamples=N_RESAMPLES,
        seed=41,
    )
    final_boundary_rows = []
    for (hazard_mode, agent), sample in boundary_bootstrap.groupby(
        ["hazard_mode", "agent"], sort=False
    ):
        finite = sample.loc[np.isfinite(sample["half_boundary"]), "half_boundary"]
        final_boundary_rows.append(
            {
                "hazard_mode": hazard_mode,
                "agent": agent,
                "median_half_boundary": float(finite.median()) if len(finite) else np.nan,
                "ci_low": float(finite.quantile(0.025)) if len(finite) else np.nan,
                "ci_high": float(finite.quantile(0.975)) if len(finite) else np.nan,
                "right_censored_fraction": float(sample["right_censored"].mean()),
            }
        )
    final_boundary_summary = pd.DataFrame(final_boundary_rows)
    observed_final_boundaries = learned_boundaries.loc[
        learned_boundaries["progress"].eq(1.0),
        ["hazard_mode", "agent", "half_boundary", "monotonicity_violations"],
    ].rename(columns={"half_boundary": "observed_half_boundary"})
    final_boundary_summary = observed_final_boundaries.merge(
        final_boundary_summary,
        on=["hazard_mode", "agent"],
        validate="one_to_one",
    )

    boundary_wide = boundary_bootstrap.pivot(
        index=["bootstrap_id", "hazard_mode"],
        columns="agent",
        values="half_boundary",
    )
    contrast_rows = []
    for hazard_mode, sample in boundary_wide.groupby(level="hazard_mode"):
        for comparison in ("sarsa", "expected_sarsa"):
            differences = (sample[comparison] - sample["q_learning"]).dropna()
            contrast_rows.append(
                {
                    "hazard_mode": hazard_mode,
                    "comparison": comparison,
                    "median_boundary_shift_vs_q": float(differences.median())
                    if len(differences)
                    else np.nan,
                    "ci_low": float(differences.quantile(0.025))
                    if len(differences)
                    else np.nan,
                    "ci_high": float(differences.quantile(0.975))
                    if len(differences)
                    else np.nan,
                    "complete_bootstrap_pairs": len(differences),
                }
            )
    boundary_contrasts = pd.DataFrame(contrast_rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle("Boundary over training" + EMPIRICAL_TITLE, fontweight="bold")
    for ax, hazard_mode in zip(axes, ("recoverable", "lethal"), strict=True):
        sample = learned_boundaries.loc[learned_boundaries["hazard_mode"].eq(hazard_mode)]
        for agent in METHOD_ORDER:
            line = sample.loc[sample["agent"].eq(agent)].sort_values("progress")
            ax.plot(
                100 * line["progress"], line["half_boundary"], marker="o",
                color=METHOD_COLORS[agent], label=METHOD_LABELS[agent],
            )
        greedy_boundary = exact_thresholds.loc[
            exact_thresholds["hazard_mode"].eq(hazard_mode)
            & exact_thresholds["epsilon"].eq(0.0),
            "threshold",
        ].iloc[0]
        ax.axhline(greedy_boundary, color="black", linestyle="--", label="greedy oracle")
        soft_boundary = exact_thresholds.loc[
            exact_thresholds["hazard_mode"].eq(hazard_mode)
            & exact_thresholds["epsilon"].eq(PERSISTENT_EPSILON),
            "threshold",
        ].iloc[0]
        if np.isfinite(soft_boundary):
            ax.axhline(soft_boundary, color="0.35", linestyle=":", label="epsilon-soft oracle")
        ax.set(
            title=f"{hazard_mode.title()} learned boundary",
            xlabel="interaction budget completed (%)",
            ylabel="reliability at 50% corridor selection",
        )
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.show()
    display(learned_boundaries)
    display(final_boundary_summary)
    display(boundary_contrasts)
    """),
    md(r"""
    We do not force these finite-seed curves to be monotone. A reported
    `monotonicity_violations` count warns when a single crossover would be a poor
    summary. Missing boundaries are right-censored: fewer than half of the seed
    panel selected the corridor anywhere on the pre-specified full-run grid.
    Pointwise binary ribbons use Wilson score intervals. Boundary intervals and
    method contrasts resample each root seed as one matched block across every
    reliability and algorithm; episodes never become fake replicates.
    """),
    code("""
    def evaluation_trial_summary(frame: pd.DataFrame) -> pd.DataFrame:
        grouped = [
            "trial_id", "agent", "seed", "env_hazard_mode", "env_action_reliability",
            "evaluation_policy_mode",
        ]
        result = (
            frame.groupby(grouped, as_index=False)
            .agg(
                episode_return=("episode_return", "mean"),
                success=("success", "mean"),
                failure=("failure", "mean"),
                total_episode_steps=("episode_length", "sum"),
                hazard_penalty_steps=("env_hazard_penalty_steps", "sum"),
                realized_corridor=(
                    "env_realized_route",
                    lambda values: float(np.mean(np.asarray(values) == "corridor")),
                ),
            )
            .rename(
                columns={
                    "env_hazard_mode": "hazard_mode",
                    "env_action_reliability": "reliability",
                }
            )
        )
        result["penalty_steps_per_1000"] = (
            1_000.0 * result["hazard_penalty_steps"] / result["total_episode_steps"]
        )
        return result

    evaluation_trials = evaluation_trial_summary(main_evaluations)
    regret_summary = continuous_seed_summary(
        final_choices,
        value="exact_deployment_regret",
        groups=("hazard_mode", "agent", "reliability"),
        seed=53,
    )
    return_summary = continuous_seed_summary(
        evaluation_trials,
        value="episode_return",
        groups=("hazard_mode", "agent", "reliability", "evaluation_policy_mode"),
        seed=59,
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    fig.suptitle("Consequences" + EMPIRICAL_TITLE, fontweight="bold")
    for row_index, hazard_mode in enumerate(("recoverable", "lethal")):
        for agent in METHOD_ORDER:
            line = regret_summary.loc[
                regret_summary["hazard_mode"].eq(hazard_mode)
                & regret_summary["agent"].eq(agent)
            ]
            axes[row_index, 0].plot(
                line["reliability"], line["mean"], marker="o",
                color=METHOD_COLORS[agent], label=METHOD_LABELS[agent],
            )
            axes[row_index, 0].fill_between(
                line["reliability"], line["ci_low"], line["ci_high"],
                color=METHOD_COLORS[agent], alpha=0.12,
            )

        for agent in METHOD_ORDER:
            for policy_mode, linestyle in (("greedy", "-"), ("behavior", "--")):
                line = return_summary.loc[
                    return_summary["hazard_mode"].eq(hazard_mode)
                    & return_summary["agent"].eq(agent)
                    & return_summary["evaluation_policy_mode"].eq(policy_mode)
                ]
                axes[row_index, 1].plot(
                    line["reliability"], line["mean"],
                    color=METHOD_COLORS[agent], linestyle=linestyle,
                    marker="o" if policy_mode == "greedy" else None,
                    label=f"{METHOD_LABELS[agent]} — {policy_mode}",
                )
                axes[row_index, 1].fill_between(
                    line["reliability"], line["ci_low"], line["ci_high"],
                    color=METHOD_COLORS[agent], alpha=0.08,
                )
        axes[row_index, 0].set(
            title=f"{hazard_mode.title()}: exact regret of deployed greedy policy",
            xlabel="reliability p", ylabel="start-state regret",
        )
        axes[row_index, 1].set(
            title=f"{hazard_mode.title()}: held-out return",
            xlabel="reliability p", ylabel="episode return",
        )
        for ax in axes[row_index]:
            ax.grid(alpha=0.22)
            ax.legend(frameon=False, fontsize=8)
    plt.show()

    exposure_summary = continuous_seed_summary(
        evaluation_trials,
        value="penalty_steps_per_1000",
        groups=("hazard_mode", "agent", "reliability", "evaluation_policy_mode"),
        seed=61,
    )
    failure_summary = continuous_seed_summary(
        evaluation_trials,
        value="failure",
        groups=("hazard_mode", "agent", "reliability", "evaluation_policy_mode"),
        seed=67,
    ).rename(
        columns={
            "mean": "failure_mean",
            "ci_low": "failure_ci_low",
            "ci_high": "failure_ci_high",
        }
    )
    exposure_table = exposure_summary.merge(
        failure_summary[
            [
                "hazard_mode", "agent", "reliability", "evaluation_policy_mode",
                "failure_mean", "failure_ci_low", "failure_ci_high",
            ]
        ],
        on=["hazard_mode", "agent", "reliability", "evaluation_policy_mode"],
        validate="one_to_one",
    )
    exposure_table["method"] = exposure_table["agent"].map(METHOD_LABELS)
    key_exposure = pd.concat(
        [
            exposure_table.loc[
                exposure_table["hazard_mode"].eq(hazard_mode)
                & exposure_table["reliability"].eq(reliability)
            ]
            for hazard_mode, reliability in DISAGREEMENT_POINTS.items()
        ],
        ignore_index=True,
    )
    display(
        key_exposure[
            [
                "hazard_mode", "reliability", "method", "evaluation_policy_mode",
                "mean", "ci_low", "ci_high",
                "failure_mean", "failure_ci_low", "failure_ci_high",
            ]
        ].rename(columns={"mean": "penalty_steps_per_1000"})
    )
    """),
    md(r"""
    ### The variance claim gets a controlled probe

    Pooling empirical TD errors across a learned trajectory would confound the
    backup rule with different state visits, actions, and policies. Instead we
    hold one state-action pair, one exact epsilon-soft Q table, and one transition
    kernel fixed. We then enumerate the one-step target distribution exactly.
    SARSA samples the next action; Expected SARSA integrates it out. Their target
    means must agree, and the variance difference is precisely the removable
    next-action sampling component.
    """),
    code("""
    def exact_backup_moments(
        model,
        q_values: np.ndarray,
        policy: np.ndarray,
        *,
        state: int,
        action: int,
        integrate_next_action: bool,
    ) -> tuple[float, float]:
        probabilities = []
        targets = []
        for next_state, transition_probability in enumerate(model.P[state, action]):
            if transition_probability == 0.0:
                continue
            reward = float(model.R[state, action, next_state])
            if model.terminal[next_state]:
                probabilities.append(float(transition_probability))
                targets.append(reward)
            elif integrate_next_action:
                probabilities.append(float(transition_probability))
                targets.append(
                    reward
                    + GAMMA * float(np.dot(policy[next_state], q_values[next_state]))
                )
            else:
                for next_action, action_probability in enumerate(policy[next_state]):
                    if action_probability == 0.0:
                        continue
                    probabilities.append(float(transition_probability * action_probability))
                    targets.append(reward + GAMMA * float(q_values[next_state, next_action]))
        weights = np.asarray(probabilities, dtype=float)
        values = np.asarray(targets, dtype=float)
        weights /= weights.sum()
        mean = float(np.dot(weights, values))
        variance = float(np.dot(weights, np.square(values - mean)))
        return mean, variance

    backup_variance_rows = []
    for hazard_mode, reliability in DISAGREEMENT_POINTS.items():
        env, solution, _ = oracle_solution(
            hazard_mode,
            reliability,
            PERSISTENT_EPSILON,
        )
        model = env.exact_mdp()
        state = env.state_to_index[env.fork_state]
        action = int(env.corridor_action)
        expected_mean, expected_variance = exact_backup_moments(
            model,
            solution.q_values,
            solution.policy,
            state=state,
            action=action,
            integrate_next_action=True,
        )
        sampled_mean, sampled_variance = exact_backup_moments(
            model,
            solution.q_values,
            solution.policy,
            state=state,
            action=action,
            integrate_next_action=False,
        )
        np.testing.assert_allclose(sampled_mean, expected_mean, atol=1e-12)
        backup_variance_rows.append(
            {
                "hazard_mode": hazard_mode,
                "reliability": reliability,
                "target_mean": expected_mean,
                "expected_sarsa_variance": expected_variance,
                "sarsa_variance": sampled_variance,
                "next_action_sampling_component": sampled_variance - expected_variance,
            }
        )
    backup_variance = pd.DataFrame(backup_variance_rows)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    for ax, row in zip(axes, backup_variance.itertuples(index=False), strict=True):
        ax.bar(
            ["Expected SARSA", "SARSA"],
            [row.expected_sarsa_variance, row.sarsa_variance],
            color=[METHOD_COLORS["expected_sarsa"], METHOD_COLORS["sarsa"]],
        )
        ax.set(
            title=f"{row.hazard_mode.title()} | p={row.reliability:g}",
            ylabel="exact one-step target variance",
        )
        ax.grid(axis="y", alpha=0.22)
    plt.show()
    display(backup_variance)
    """),
    code("""
    def modal_policy(tables: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        policies = np.stack([np.argmax(table, axis=1) for table in tables])
        modes = np.array(
            [
                np.bincount(policies[:, state], minlength=tables[0].shape[1]).argmax()
                for state in range(policies.shape[1])
            ],
            dtype=int,
        )
        return modes, np.mean(policies == modes[None, :], axis=0)

    fig, axes = plt.subplots(2, 5, figsize=(20, 7.5), constrained_layout=True)
    fig.suptitle("Exact and learned policies" + EMPIRICAL_TITLE, fontweight="bold")
    for row_index, hazard_mode in enumerate(("recoverable", "lethal")):
        available = np.sort(final_choices.loc[
            final_choices["hazard_mode"].eq(hazard_mode), "reliability"
        ].unique())
        reliability = float(available[np.argmin(np.abs(available - DISAGREEMENT_POINTS[hazard_mode]))])
        env, greedy, greedy_policy = oracle_solution(hazard_mode, reliability, 0.0)
        _, soft, soft_policy = oracle_solution(
            hazard_mode,
            reliability,
            PERSISTENT_EPSILON,
        )
        plot_policy(
            greedy_policy, env, ax=axes[row_index, 0],
            title=f"Greedy oracle\\np={reliability:g}",
        )
        plot_policy(
            soft_policy, env, ax=axes[row_index, 1],
            title=f"ε-soft oracle\\nε={PERSISTENT_EPSILON:g}",
        )
        for column, agent in enumerate(METHOD_ORDER, start=2):
            trial_ids = final_choices.loc[
                final_choices["hazard_mode"].eq(hazard_mode)
                & final_choices["reliability"].eq(reliability)
                & final_choices["agent"].eq(agent),
                "trial_id",
            ]
            policy, consensus = modal_policy([final_q_tables[trial_id] for trial_id in trial_ids])
            plot_policy(
                policy,
                env,
                values=consensus,
                vmin=0.0,
                vmax=1.0,
                cmap="Blues",
                colorbar_label="fraction choosing modal action",
                ax=axes[row_index, column],
                title=METHOD_LABELS[agent],
            )
        axes[row_index, 0].set_ylabel(hazard_mode.title())
    plt.show()
    """),
    code("""
    def greedy_rollout(
        table: np.ndarray,
        *,
        hazard_mode: str,
        reliability: float,
        seed: int,
    ) -> dict[str, object]:
        env = corridor_environment(hazard_mode, reliability)
        observation, _ = env.reset(seed=seed)
        trajectory = [env.agent_position]
        total_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(np.argmax(table[int(observation)]))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            trajectory.append(env.agent_position)
        summary = env.episode_summary()
        env.close()
        return {
            "trajectory": trajectory,
            "episode_return": total_return,
            **summary,
        }

    rollout_seed = 20_260_818
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("Fixed-seed rollouts" + EMPIRICAL_TITLE, fontweight="bold")
    representative_rollouts = []
    for row_index, hazard_mode in enumerate(("recoverable", "lethal")):
        available = np.sort(final_choices.loc[
            final_choices["hazard_mode"].eq(hazard_mode), "reliability"
        ].unique())
        reliability = float(available[np.argmin(np.abs(available - DISAGREEMENT_POINTS[hazard_mode]))])
        for column, agent in enumerate(METHOD_ORDER):
            candidates = final_choices.loc[
                final_choices["hazard_mode"].eq(hazard_mode)
                & final_choices["reliability"].eq(reliability)
                & final_choices["agent"].eq(agent)
                & final_choices["seed"].eq(REPRESENTATIVE_SEED)
            ]
            selected = candidates.iloc[0]
            rollout = greedy_rollout(
                final_q_tables[selected["trial_id"]],
                hazard_mode=hazard_mode,
                reliability=reliability,
                seed=rollout_seed,
            )
            representative_rollouts.append(
                {
                    "hazard_mode": hazard_mode,
                    "reliability": reliability,
                    "agent": agent,
                    **{key: value for key, value in rollout.items() if key != "trajectory"},
                }
            )
            env = corridor_environment(hazard_mode, reliability)
            plot_maze(
                env,
                trajectory=rollout["trajectory"],
                ax=axes[row_index, column],
                title=(
                    f"{METHOD_LABELS[agent]} | R={rollout['episode_return']:.2f} | "
                    f"{rollout['realized_route']}"
                ),
            )
            env.close()
        axes[row_index, 0].set_ylabel(f"{hazard_mode.title()} | p={reliability:g}")
    plt.show()
    display(pd.DataFrame(representative_rollouts))
    """),
    md(r"""
    These are not best-looking trajectories. They use training seed zero and one
    fixed, declared rollout seed. The same environment seed does not force equal
    transitions after policies choose different actions, but it prevents visual
    examples from being selected post hoc by outcome.
    """),
    md(r"""
    ## 4. Annealing exploration is a falsification check

    Persistent exploration intentionally makes the on-policy target different
    from greedy control. We therefore run a smaller sensitivity experiment at
    disagreement points while epsilon decays toward zero. If the story above is
    correct, SARSA, Expected SARSA, and Q-learning should move toward the same
    greedy route given enough interactions. A failure to do so is evidence about
    optimization, coverage, or budget—not evidence that the environment has
    three different optimal policies.
    """),
    code("""
    full_annealed_config = ExperimentConfig.from_yaml(
        REPO_ROOT / "configs" / "shortcut_or_shelter_annealed.yaml"
    )
    annealed_config = full_annealed_config
    if SMOKE or QUICK:
        annealed_config = replace(
            full_annealed_config,
            seeds=(0,) if SMOKE else tuple(range(4)),
            total_interaction_steps=100 if SMOKE else 4_000,
            snapshot_interval=1_000_000,
            snapshot_step_interval=20 if SMOKE else 250,
            policy_evaluation=replace(
                full_annealed_config.policy_evaluation,
                interval_episodes=1_000_000,
                episodes_per_checkpoint=1 if SMOKE else 4,
                include_initial=False,
                include_final=True,
            ),
            execution=ExecutionSpec(parallel_workers=1),
            artifacts=replace(
                full_annealed_config.artifacts,
                output_dir=RESULTS_DIR,
                flush_rows=200 if SMOKE else 5_000,
            ),
        )
    display(pd.Series(estimate_run(annealed_config).as_dict(), name="annealed preflight"))
    annealed_result = run_or_reopen("annealed", annealed_config)
    annealed_choices, annealed_q_tables = final_policy_rows(
        annealed_result,
        annealed_config,
    )
    annealed_summary = binary_seed_summary(
        annealed_choices,
        value="corridor_selected",
        groups=("hazard_mode", "agent", "reliability"),
    )
    annealed_summary["method"] = annealed_summary["agent"].map(METHOD_LABELS)
    display(
        annealed_summary[
            ["hazard_mode", "reliability", "method", "mean", "ci_low", "ci_high", "n_seeds"]
        ]
    )
    """),
    md(r"""
    ## 5. What the evidence can and cannot say

    **Supported interpretations**

    - The route boundary belongs to a declared environment, reward law,
      discount, and policy class—not to an algorithm's personality.
    - Lethal risk compounds across repeated exposed decisions, compressing the
      greedy transition near perfect execution.
    - Persistent exploration can rationally move the epsilon-soft boundary or
      remove the corridor region entirely.
    - SARSA and Expected SARSA should share the same expected persistent-epsilon
      target; their finite-sample variability need not match.

    **Not supported**

    - “SARSA is universally safer” or “Q-learning is reckless.”
    - Treating held-out episodes as independent replicates.
    - Calling a smoothed crossover ground truth when the exact action gap is
      available.
    - Generalizing beyond this reward scale, topology, horizon, and exploration
      protocol without a sensitivity experiment.
    - Extending the one EAST/SOUTH boundary below the plotted reliability range;
      at sufficiently poor control, NORTH or WEST can become exact-optimal.

    The next notebook will add a visible CLEAR/STORM regime. Once regime is part
    of state, stable switching is an ordinary augmented MDP with a policy
    conditional on both position and weather. Hiding a persistent regime creates
    a POMDP; a position-only Q table is then an information-limited baseline, not
    the “real optimum.”
    """),
    code("""
    provenance = pd.DataFrame(
        [
            {
                "study": label,
                "experiment": result.metadata.get("experiment_name", config.name),
                "run_directory": str(result.run_directory),
                "training_budget": config.total_interaction_steps,
                "training_seeds": len(config.seeds),
                "trials": len(config.trials()),
            }
            for label, result, config in (
                ("recoverable", main_results["recoverable"], main_configs["recoverable"]),
                ("lethal", main_results["lethal"], main_configs["lethal"]),
                ("annealed", annealed_result, annealed_config),
            )
        ]
    )
    display(provenance)
    print("Canonical configs:")
    for path in (*FULL_CONFIG_PATHS.values(), REPO_ROOT / "configs" / "shortcut_or_shelter_annealed.yaml"):
        print(" -", path.relative_to(REPO_ROOT))
    """),
]


shortcut_or_shelter = [
    md(r"""
    # Shortcut or shelter?

    ## What three TD-control rules learn at a noisy route boundary

    A short corridor reaches the goal quickly, but an execution error can enter
    hazards above or below it. A longer southern route is protected by a wall.
    We solve the route choice exactly, then compare it with 1,992 completed
    training runs of Q-learning, SARSA, and Expected SARSA.

    This is now a **results notebook**. It does not retrain the study and it does
    not rescan the raw multi-gigabyte run store. `Run All` loads a small,
    versioned evidence package produced once from the immutable Protocol-v2
    artifacts. The raw-analysis program and run identifiers are listed at the
    end.
    """),
    code("""
    from __future__ import annotations

    import json
    from pathlib import Path

    from IPython.display import SVG, Markdown, display
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from rllab.environments import RiskyCorridorEnv
    from rllab.visualization import plot_maze

    def find_repo_root(start: Path) -> Path:
        for candidate in (start.resolve(), *start.resolve().parents):
            if (candidate / "pyproject.toml").exists() and (candidate / "src" / "rllab").exists():
                return candidate
        raise FileNotFoundError("Run this notebook from inside the rl-lab repository.")

    REPO_ROOT = find_repo_root(Path.cwd())
    REPORT_DIR = REPO_ROOT / "reports" / "shortcut_or_shelter"
    REQUIRED = (
        "analysis_manifest.json",
        "exact_thresholds.csv",
        "final_choices.csv",
        "corridor_summary.csv",
        "final_boundary_summary.csv",
        "boundary_contrasts.csv",
        "stability_summary.csv",
        "endpoint_calibration.csv",
        "annealed_summary.csv",
        "backup_variance.csv",
        "route_selection.svg",
        "mechanism_checks.svg",
    )
    missing = [name for name in REQUIRED if not (REPORT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "The compact evidence package is incomplete: "
            + ", ".join(missing)
            + ". Rebuild it with the command in the final notebook section."
        )

    def load_csv(name: str) -> pd.DataFrame:
        return pd.read_csv(REPORT_DIR / name)

    analysis_manifest = json.loads((REPORT_DIR / "analysis_manifest.json").read_text())
    exact_thresholds = load_csv("exact_thresholds.csv")
    final_choices = load_csv("final_choices.csv")
    corridor_summary = load_csv("corridor_summary.csv")
    final_boundary_summary = load_csv("final_boundary_summary.csv")
    boundary_contrasts = load_csv("boundary_contrasts.csv")
    stability_summary = load_csv("stability_summary.csv")
    endpoint_calibration = load_csv("endpoint_calibration.csv")
    annealed_summary = load_csv("annealed_summary.csv")
    backup_variance = load_csv("backup_variance.csv")

    METHOD_LABELS = {
        "q_learning": "Q-learning",
        "sarsa": "SARSA",
        "expected_sarsa": "Expected SARSA",
    }
    print(
        f"Loaded {len(final_choices):,} primary policies and "
        f"{int(annealed_summary['n_seeds'].sum()):,} summarized annealing seed-observations "
        f"from {REPORT_DIR.relative_to(REPO_ROOT)}"
    )
    """),
    md(r"""
    ## 1. The world and the estimand

    The fork is the start state. EAST expresses the corridor policy; SOUTH
    expresses the shelter policy. Intended actions execute correctly with
    probability $p$ and otherwise slip left or right with equal probability.
    The same actuator noise applies on both routes; the wall changes the
    consequences of a slip.

    The exact start-state action gap is

    $$
    \Delta^*(p)=Q^*_p(s_0,\mathrm{EAST})-Q^*_p(s_0,\mathrm{SOUTH}).
    $$

    Its zero crossing is the model boundary. The learned estimand is different:
    at each $p$, what fraction of 20 independent training seeds ends with EAST
    as the unique greedy action? Ties, NORTH, and WEST remain visible as
    `other`.
    """),
    code("""
    diagram_env = RiskyCorridorEnv(
        corridor_reliability=0.90,
        hazard_mode="lethal",
        recoverable_hazard_penalty=-0.50,
        lethal_hazard_penalty=-8.0,
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    plot_maze(diagram_env, ax=ax, title="The exposed corridor and protected southern route")
    ax.plot([0, 8], [1, 1], color="#D55E00", linewidth=3, alpha=0.72, label="corridor")
    ax.plot([0, 0, 8, 8], [1, 3, 3, 1], color="#0072B2", linewidth=3, alpha=0.72, label="shelter")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, frameon=False)
    plt.tight_layout()
    plt.show()

    threshold_view = exact_thresholds.copy()
    threshold_view["objective"] = np.where(
        threshold_view["epsilon"].eq(0.0), "greedy", "epsilon-soft"
    )
    display(
        threshold_view.loc[
            threshold_view["epsilon"].isin([0.0, 0.10]),
            ["hazard_mode", "objective", "epsilon", "threshold", "gap_at_perfect_control"],
        ]
    )
    """),
    md(r"""
    The recoverable penalty leaves a broad transition: the greedy model changes
    route near $p=.806$, while the persistent-$\epsilon=.10$ continuation
    policy changes near $p=.848$. With lethal hazards, repeated exposed moves
    compress the greedy boundary near $p=.989$. The epsilon-soft model never
    enters the corridor even at perfect actuation, because actuation noise and
    deliberate exploration are different sources of risk.
    """),
    md(r"""
    ## 2. Where the trained policies changed route

    Every point below is a fraction of independent trained policies, not a
    collection of episodes. The lines connect the pre-specified reliability
    grid only as visual guides. Ribbons are 95% Wilson intervals. Vertical lines
    are model calculations, not fits to the learned data.
    """),
    code("""
    display(SVG(filename=str(REPORT_DIR / "route_selection.svg")))

    boundary_table = final_boundary_summary.copy()
    boundary_table["method"] = boundary_table["agent"].map(METHOD_LABELS)
    boundary_table["target"] = np.where(
        boundary_table["agent"].eq("q_learning"), "greedy", "epsilon-soft"
    )
    boundary_table = boundary_table.merge(
        stability_summary.groupby(["hazard_mode", "agent"], as_index=False).agg(
            max_final_quarter_change=("changed_fraction", "max")
        ),
        on=["hazard_mode", "agent"],
        how="left",
        validate="one_to_one",
    )
    display(
        boundary_table[
            [
                "hazard_mode", "method", "target", "observed_half_boundary",
                "ci_low", "ci_high", "right_censored_fraction",
                "monotonicity_violations", "max_final_quarter_change",
            ]
        ]
    )
    display(boundary_contrasts)
    """),
    code("""
    unresolved = final_choices.loc[final_choices["choice"].isin(["other", "tie"])]
    choice_counts = (
        final_choices.groupby(["hazard_mode", "agent", "choice"], as_index=False)
        .size()
        .rename(columns={"size": "seeds"})
    )
    display(choice_counts)
    if len(unresolved):
        display(
            Markdown(
                f"**Declared exception.** {len(unresolved)} of {len(final_choices):,} final "
                "policies selected an off-route action or tie. It remains a third outcome "
                "and is not recoded as shelter."
            )
        )
        display(unresolved)
    """),
    md(r"""
    ### Analysis amendment — 19 August 2026

    The first executable validation check stopped if *any* final policy selected
    NORTH, WEST, or a tie. It stopped on one WEST action among 1,920 primary
    policies: lethal SARSA at $p=.990$, seed 10. WEST exceeded SOUTH by only
    $0.00550$ in the learned Q table.

    That all-or-nothing check conflicted with the declared three-outcome
    estimand above. Before estimating the empirical boundaries, it was revised
    to retain and report `other` and to fail only when unresolved actions exceed
    20% within a condition. Here the maximum is 5% in one condition and 0%
    elsewhere. The data, configurations, and training runs were not changed.
    """),
    md(r"""
    ## 3. Does the difference survive when exploration disappears?

    Q-learning values a greedy continuation. With persistent exploration,
    SARSA and Expected SARSA value the epsilon-soft continuation actually being
    followed. The smaller annealing study sends epsilon to zero, so the three
    methods should move toward the same greedy choice at two points where the
    persistent objectives disagree.

    The right side of the figure isolates a second claim. Holding the state,
    transition kernel, Q table, and epsilon-soft policy fixed, SARSA and Expected
    SARSA have the same target mean. Expected SARSA integrates over the next
    action and removes only that sampling component of target variance.
    """),
    code("""
    display(SVG(filename=str(REPORT_DIR / "mechanism_checks.svg")))
    display(annealed_summary)
    display(backup_variance)
    """),
    md(r"""
    ## 4. Checks and limits

    All primary trials received exactly 100,000 interactions. The recoverable
    panel contains 1,080 trials, the lethal panel 840, and the annealing check
    72. Endpoint route choices passed their pre-specified 80/20 calibration,
    and no endpoint seed changed route between 75,000 and 100,000 interactions.
    Held-out greedy and continuing-behavior rollouts use paired environment
    seeds, but repeated rollouts are measurements of one trained policy—not new
    independent training replicates.

    This study does not show that SARSA is universally safer or that Q-learning
    is reckless. It shows how a continuation objective, a consequence law, and
    a finite training budget meet at one exactly solvable route boundary.
    """),
    code("""
    display(endpoint_calibration)
    validation = {
        key: analysis_manifest.get(key)
        for key in (
            "generated_at", "analysis_version", "run_ids", "primary_trial_count",
            "sensitivity_trial_count", "quality_gates",
        )
        if key in analysis_manifest
    }
    display(pd.Series(validation, name="analysis record"))
    """),
    md(r"""
    ## Rebuild the compact evidence package

    Normal notebook use ends here. To audit or change the analysis, run the
    standalone program below from the repository root. It reads the three
    immutable run directories once and rewrites `reports/shortcut_or_shelter/`.

    ```bash
    python experiments/stochastic_maze/analyze_shortcut_or_shelter.py \
      --recoverable-run results/shortcut_or_shelter_recoverable-dea8b3bb98-20260818T165653.284759Z \
      --lethal-run results/shortcut_or_shelter_lethal-c224eb9e19-20260818T184221.884416Z \
      --annealed-run results/shortcut_or_shelter_annealed-7a6ceda8a8-20260818T200356.410162Z
    ```

    The checked report manifest records those run identifiers, the analysis
    source, the validation decisions, and every compact output used here and in
    the accompanying article.
    """),
]


lunar_lander_sac = [
    md(r"""
    # Continuous Lunar Lander with Soft Actor-Critic

    `LunarLander-v3` is discrete by default. Here we deliberately request
    `continuous=True`, giving SAC a two-dimensional bounded action: main-engine
    throttle and lateral-engine throttle. This is the right version for a
    squashed-Gaussian actor; DQN would be the natural baseline for the default
    four-action version.

    The notebook has two execution profiles. `QUICK=True` checks the complete
    training/evaluation/animation workflow on CPU but is **not a learning claim**.
    Set `QUICK=False` for the preregistered 500k-step profile. Automated tests set
    `RL_LAB_NOTEBOOK_SMOKE=1` for an even smaller structural run.
    """),
    code("""
    from __future__ import annotations

    import copy
    from dataclasses import asdict, dataclass
    import os
    from pathlib import Path
    import random
    import time

    import gymnasium as gym
    from gymnasium.spaces import Box
    from IPython.display import HTML, display
    from matplotlib import animation
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.distributions import Normal

    SEED = 23
    QUICK = True
    SMOKE = os.environ.get("RL_LAB_NOTEBOOK_SMOKE") == "1"
    DEVICE = torch.device("cpu")  # Small MLPs are reproducible and fast on CPU.

    available_styles = set(plt.style.available)
    plot_style = next(
        (
            style
            for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot")
            if style in available_styles
        ),
        "default",
    )
    plt.style.use(plot_style)


    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


    @dataclass(frozen=True)
    class SACConfig:
        env_id: str = "LunarLander-v3"
        seed: int = SEED
        total_steps: int = 320 if SMOKE else (8_000 if QUICK else 500_000)
        learning_starts: int = 64 if SMOKE else (1_000 if QUICK else 10_000)
        replay_capacity: int = 2_000 if SMOKE else (100_000 if QUICK else 1_000_000)
        batch_size: int = 32 if SMOKE else 256
        hidden_sizes: tuple[int, ...] = (
            (32, 32) if SMOKE else ((128, 128) if QUICK else (400, 300))
        )
        actor_lr: float = 3e-4
        critic_lr: float = 3e-4
        alpha_lr: float = 3e-4
        gamma: float = 0.99
        tau: float = 0.01
        initial_alpha: float = 0.2
        max_grad_norm: float = 10.0
        eval_interval: int = 160 if SMOKE else (2_000 if QUICK else 10_000)
        eval_episodes: int = 1 if SMOKE else (3 if QUICK else 10)
        diagnostic_interval: int = 8 if SMOKE else (25 if QUICK else 100)


    config = SACConfig()
    seed_everything(config.seed)

    probe = gym.make(config.env_id, continuous=True, enable_wind=False)
    try:
        observation, _ = probe.reset(seed=config.seed)
        probe.action_space.seed(config.seed)
        assert isinstance(probe.action_space, Box)
        assert observation.shape == (8,) and probe.action_space.shape == (2,)
        OBS_DIM = int(np.prod(probe.observation_space.shape))
        ACT_DIM = int(np.prod(probe.action_space.shape))
        ACTION_LOW = probe.action_space.low.astype(np.float32).copy()
        ACTION_HIGH = probe.action_space.high.astype(np.float32).copy()
    finally:
        probe.close()

    print(f"device={DEVICE}; observation={OBS_DIM}; action={ACT_DIM}")
    display(pd.Series(asdict(config), name="value").to_frame())
    """),
    md(r"""
    ## 1. Environment contract and SAC objectives

    The state is $(x,y,v_x,v_y,\theta,\dot\theta,c_L,c_R)$. For continuous
    actions $a=(a_{main},a_{lateral})\in[-1,1]^2$, the main engine is off below
    zero and each lateral engine has a dead zone $|a_{lateral}|<0.5$. We therefore
    retain engine-activation diagnostics rather than treating the action vector
    as an opaque control.

    SAC learns two critics and uses the smaller target:

    $$y=r+\gamma(1-z)\left[\min_i Q_{\bar\phi_i}(s',a')
      -\alpha\log\pi_\theta(a'\mid s')\right],\qquad
      a'\sim\pi_\theta(\cdot\mid s'),$$

    $$J_{Q_i}=\mathbb E[(Q_{\phi_i}(s,a)-y)^2],\qquad
      J_\pi=\mathbb E[\alpha\log\pi_\theta(a\mid s)-\min_iQ_{\phi_i}(s,a)].$$

    Here $z$ means **true MDP termination**. A Gymnasium `TimeLimit` truncation
    resets interaction but does not erase the bootstrap term. Temperature is
    learned with target entropy $-\dim(\mathcal A)=-2$.

    References: [Gymnasium Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/),
    [SAC algorithms and applications](https://arxiv.org/abs/1812.05905), and the
    readable [CleanRL SAC implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py).
    """),
    code("""
    class ReplayBuffer:
        def __init__(self, observation_dim, action_dim, capacity, seed):
            self.observations = np.empty((capacity, observation_dim), dtype=np.float32)
            self.actions = np.empty((capacity, action_dim), dtype=np.float32)
            self.rewards = np.empty((capacity, 1), dtype=np.float32)
            self.next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
            self.terminated = np.empty((capacity, 1), dtype=np.float32)
            self.capacity = capacity
            self.position = 0
            self.size = 0
            self.rng = np.random.default_rng(seed)

        def add(self, observation, action, reward, next_observation, terminated):
            index = self.position
            self.observations[index] = observation
            self.actions[index] = action
            self.rewards[index, 0] = reward
            self.next_observations[index] = next_observation
            self.terminated[index, 0] = terminated
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

        def sample(self, batch_size, device):
            if self.size < batch_size:
                raise ValueError("Not enough transitions for one batch")
            indices = self.rng.integers(0, self.size, size=batch_size)
            arrays = (
                self.observations[indices],
                self.actions[indices],
                self.rewards[indices],
                self.next_observations[indices],
                self.terminated[indices],
            )
            return tuple(torch.as_tensor(array, device=device) for array in arrays)

        def __len__(self):
            return self.size


    replay_probe = ReplayBuffer(OBS_DIM, ACT_DIM, config.batch_size, config.seed)
    for _ in range(config.batch_size):
        replay_probe.add(
            np.zeros(OBS_DIM, dtype=np.float32),
            np.zeros(ACT_DIM, dtype=np.float32),
            0.0,
            np.ones(OBS_DIM, dtype=np.float32),
            False,
        )
    probe_batch = replay_probe.sample(config.batch_size, DEVICE)
    assert probe_batch[0].shape == (config.batch_size, OBS_DIM)
    assert probe_batch[1].shape == (config.batch_size, ACT_DIM)
    del replay_probe, probe_batch
    """),
    md(r"""
    ## 2. Bounded stochastic actor and twin critics

    Let $u=\mu_\theta(s)+\sigma_\theta(s)\epsilon$ and transform it to arbitrary
    Box bounds with $a=c+d\odot\tanh u$. The action density must include the
    affine and tanh Jacobians. The implementation uses the stable identity

    $$\log(1-\tanh^2u)=2(\log2-u-\operatorname{softplus}(-2u)).$$
    """),
    code("""
    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


    def build_mlp(input_dim, hidden_sizes, output_dim):
        layers = []
        previous = input_dim
        for width in hidden_sizes:
            layer = nn.Linear(previous, width)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.ReLU()))
            previous = width
        output = nn.Linear(previous, output_dim)
        nn.init.uniform_(output.weight, -3e-3, 3e-3)
        nn.init.zeros_(output.bias)
        layers.append(output)
        return nn.Sequential(*layers)


    class SquashedGaussianActor(nn.Module):
        def __init__(self, observation_dim, action_low, action_high, hidden_sizes):
            super().__init__()
            self.body = build_mlp(observation_dim, hidden_sizes, hidden_sizes[-1])[:-1]
            self.mean = nn.Linear(hidden_sizes[-1], len(action_low))
            self.log_std = nn.Linear(hidden_sizes[-1], len(action_low))
            nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
            nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
            nn.init.zeros_(self.mean.bias)
            nn.init.zeros_(self.log_std.bias)
            action_low = torch.as_tensor(action_low, dtype=torch.float32)
            action_high = torch.as_tensor(action_high, dtype=torch.float32)
            self.register_buffer("action_scale", (action_high - action_low) / 2)
            self.register_buffer("action_bias", (action_high + action_low) / 2)

        def distribution_parameters(self, observation):
            hidden = self.body(observation)
            mean = self.mean(hidden)
            raw_log_std = torch.tanh(self.log_std(hidden))
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (raw_log_std + 1)
            return mean, log_std

        def sample(self, observation):
            mean, log_std = self.distribution_parameters(observation)
            normal = Normal(mean, log_std.exp())
            raw_action = normal.rsample()
            squashed = torch.tanh(raw_action)
            action = self.action_bias + self.action_scale * squashed
            log_tanh_jacobian = 2 * (
                np.log(2.0) - raw_action - F.softplus(-2 * raw_action)
            )
            log_probability = (
                normal.log_prob(raw_action)
                - log_tanh_jacobian
                - torch.log(self.action_scale)
            ).sum(dim=-1, keepdim=True)
            return action, log_probability, log_std

        def deterministic(self, observation):
            mean, _ = self.distribution_parameters(observation)
            return self.action_bias + self.action_scale * torch.tanh(mean)


    class Critic(nn.Module):
        def __init__(self, observation_dim, action_dim, hidden_sizes):
            super().__init__()
            self.network = build_mlp(observation_dim + action_dim, hidden_sizes, 1)

        def forward(self, observation, action):
            return self.network(torch.cat((observation, action), dim=-1))


    actor_probe = SquashedGaussianActor(
        OBS_DIM, ACTION_LOW, ACTION_HIGH, config.hidden_sizes
    ).to(DEVICE)
    probe_states = torch.zeros((16, OBS_DIM), device=DEVICE)
    probe_actions, probe_log_probabilities, _ = actor_probe.sample(probe_states)
    assert torch.isfinite(probe_log_probabilities).all()
    assert torch.all(probe_actions >= torch.as_tensor(ACTION_LOW, device=DEVICE) - 1e-6)
    assert torch.all(probe_actions <= torch.as_tensor(ACTION_HIGH, device=DEVICE) + 1e-6)
    del actor_probe, probe_states, probe_actions, probe_log_probabilities
    """),
    code("""
    def soft_update(target, source, tau):
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                target.parameters(), source.parameters(), strict=True
            ):
                target_parameter.mul_(1 - tau).add_(source_parameter, alpha=tau)


    class SACAgent:
        def __init__(self, config, device):
            self.config = config
            self.device = device
            self.actor = SquashedGaussianActor(
                OBS_DIM, ACTION_LOW, ACTION_HIGH, config.hidden_sizes
            ).to(device)
            self.q1 = Critic(OBS_DIM, ACT_DIM, config.hidden_sizes).to(device)
            self.q2 = Critic(OBS_DIM, ACT_DIM, config.hidden_sizes).to(device)
            self.q1_target = copy.deepcopy(self.q1).requires_grad_(False)
            self.q2_target = copy.deepcopy(self.q2).requires_grad_(False)
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(), lr=config.actor_lr
            )
            self.critic_parameters = list(self.q1.parameters()) + list(self.q2.parameters())
            self.critic_optimizer = torch.optim.Adam(
                self.critic_parameters, lr=config.critic_lr
            )
            self.log_alpha = torch.tensor(
                np.log(config.initial_alpha),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha], lr=config.alpha_lr
            )
            self.target_entropy = -float(ACT_DIM)

        @property
        def alpha(self):
            return self.log_alpha.exp()

        def act(self, observation, deterministic=False):
            state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                if deterministic:
                    action = self.actor.deterministic(state)
                else:
                    action, _, _ = self.actor.sample(state)
            return action.cpu().numpy().astype(np.float32)

        def update(self, batch):
            observations, actions, rewards, next_observations, terminated = batch
            with torch.no_grad():
                next_actions, next_log_probabilities, _ = self.actor.sample(next_observations)
                next_soft_values = torch.minimum(
                    self.q1_target(next_observations, next_actions),
                    self.q2_target(next_observations, next_actions),
                ) - self.alpha.detach() * next_log_probabilities
                targets = rewards + self.config.gamma * (1 - terminated) * next_soft_values

            q1_values = self.q1(observations, actions)
            q2_values = self.q2(observations, actions)
            q1_loss = F.mse_loss(q1_values, targets)
            q2_loss = F.mse_loss(q2_values, targets)
            critic_loss = q1_loss + q2_loss
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_parameters, self.config.max_grad_norm
            )
            self.critic_optimizer.step()

            for parameter in self.critic_parameters:
                parameter.requires_grad_(False)
            sampled_actions, log_probabilities, log_std = self.actor.sample(observations)
            sampled_q = torch.minimum(
                self.q1(observations, sampled_actions),
                self.q2(observations, sampled_actions),
            )
            actor_loss = (
                self.alpha.detach() * log_probabilities - sampled_q
            ).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_grad_norm
            )
            self.actor_optimizer.step()
            for parameter in self.critic_parameters:
                parameter.requires_grad_(True)

            alpha_loss = -(
                self.alpha * (log_probabilities.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

            soft_update(self.q1_target, self.q1, self.config.tau)
            soft_update(self.q2_target, self.q2, self.config.tau)

            absolute_td_error = 0.5 * (
                (q1_values.detach() - targets).abs()
                + (q2_values.detach() - targets).abs()
            )
            return {
                "q1_loss": float(q1_loss.detach()),
                "q2_loss": float(q2_loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "alpha_loss": float(alpha_loss.detach()),
                "alpha": float(self.alpha.detach()),
                "entropy": float(-log_probabilities.detach().mean()),
                "mean_log_std": float(log_std.detach().mean()),
                "mean_q": float(0.5 * (q1_values.detach() + q2_values.detach()).mean()),
                "mean_target": float(targets.mean()),
                "q_disagreement": float((q1_values.detach() - q2_values.detach()).abs().mean()),
                "mean_absolute_td_error": float(absolute_td_error.mean()),
                "actor_gradient_norm": float(actor_gradient_norm),
                "critic_gradient_norm": float(critic_gradient_norm),
            }


    # A truncation must retain the bootstrap; a true termination must not.
    reward_probe = torch.tensor([[1.0]])
    bootstrap_probe = torch.tensor([[7.0]])
    terminated_targets = reward_probe + config.gamma * (1 - torch.tensor([[1.0]])) * bootstrap_probe
    truncated_targets = reward_probe + config.gamma * (1 - torch.tensor([[0.0]])) * bootstrap_probe
    assert float(terminated_targets) == 1.0
    assert float(truncated_targets) > 1.0
    """),
    md(r"""
    ## 3. Training and paired-seed evaluation

    Training uses exploratory actions, while evaluation uses the deterministic
    squashed mean on the same held-out seeds at every checkpoint. This common
    random-number design makes changes over training easier to interpret.
    Evaluation does not choose checkpoints; a separate final seed panel should be
    used for publication claims.

    The full profile follows the scale of RL Zoo's current LunarLanderContinuous
    SAC baseline (500k steps, 10k random warm-up, batch 256, automatic entropy),
    while keeping a constant conservative learning rate. See the
    [registered comparison configuration](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml).
    """),
    code("""
    def make_lander(render_mode=None):
        return gym.make(
            config.env_id,
            continuous=True,
            enable_wind=False,
            render_mode=render_mode,
        )


    def evaluate_policy(agent, seeds):
        rows = []
        environment = make_lander()
        try:
            for seed in seeds:
                observation, _ = environment.reset(seed=int(seed))
                episode_return = 0.0
                episode_length = 0
                main_engine_steps = 0
                lateral_engine_steps = 0
                saturated_components = 0
                terminated = truncated = False
                while not (terminated or truncated):
                    action = agent.act(observation, deterministic=True)
                    observation, reward, terminated, truncated, _ = environment.step(action)
                    episode_return += float(reward)
                    episode_length += 1
                    main_engine_steps += int(action[0] > 0)
                    lateral_engine_steps += int(abs(action[1]) > 0.5)
                    saturated_components += int(np.sum(np.abs(action) > 0.95))
                rows.append(
                    {
                        "seed": int(seed),
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "solved": episode_return >= 200,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "main_engine_rate": main_engine_steps / episode_length,
                        "lateral_engine_rate": lateral_engine_steps / episode_length,
                        "action_saturation_rate": saturated_components / (ACT_DIM * episode_length),
                    }
                )
        finally:
            environment.close()
        return pd.DataFrame(rows)


    @dataclass
    class TrainingResult:
        agent: SACAgent
        episodes: pd.DataFrame
        updates: pd.DataFrame
        evaluations: pd.DataFrame
        elapsed_seconds: float


    EVALUATION_SEEDS = tuple(
        range(10_000, 10_000 + config.eval_episodes)
    )


    def train_sac(config):
        seed_everything(config.seed)
        environment = make_lander()
        environment.action_space.seed(config.seed)
        agent = SACAgent(config, DEVICE)
        replay = ReplayBuffer(
            OBS_DIM,
            ACT_DIM,
            config.replay_capacity,
            seed=config.seed + 1,
        )
        episode_rows, update_rows, evaluation_frames = [], [], []
        episode_index = 0
        episode_return = 0.0
        episode_length = 0
        episode_main_steps = 0
        episode_lateral_steps = 0
        observation, _ = environment.reset(seed=config.seed)
        start_time = time.perf_counter()

        initial_evaluation = evaluate_policy(agent, EVALUATION_SEEDS)
        initial_evaluation.insert(0, "step", 0)
        evaluation_frames.append(initial_evaluation)

        try:
            for step in range(1, config.total_steps + 1):
                if step <= config.learning_starts:
                    action = environment.action_space.sample()
                else:
                    action = agent.act(observation)

                next_observation, reward, terminated, truncated, _ = environment.step(action)
                replay.add(observation, action, reward, next_observation, terminated)
                episode_return += float(reward)
                episode_length += 1
                episode_main_steps += int(action[0] > 0)
                episode_lateral_steps += int(abs(action[1]) > 0.5)
                observation = next_observation

                if step > config.learning_starts and len(replay) >= config.batch_size:
                    diagnostics = agent.update(replay.sample(config.batch_size, DEVICE))
                    if step % config.diagnostic_interval == 0:
                        update_rows.append({"step": step, **diagnostics})

                if terminated or truncated:
                    episode_index += 1
                    episode_rows.append(
                        {
                            "step": step,
                            "episode": episode_index,
                            "episode_return": episode_return,
                            "episode_length": episode_length,
                            "solved": episode_return >= 200,
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "main_engine_rate": episode_main_steps / episode_length,
                            "lateral_engine_rate": episode_lateral_steps / episode_length,
                        }
                    )
                    observation, _ = environment.reset()
                    episode_return = 0.0
                    episode_length = 0
                    episode_main_steps = 0
                    episode_lateral_steps = 0

                if step % config.eval_interval == 0 or step == config.total_steps:
                    evaluation = evaluate_policy(agent, EVALUATION_SEEDS)
                    evaluation.insert(0, "step", step)
                    evaluation_frames.append(evaluation)
                    mean_return = evaluation["episode_return"].mean()
                    print(
                        f"step={step:>7,}  episodes={episode_index:>4}  "
                        f"eval_return={mean_return:>8.1f}  "
                        f"alpha={float(agent.alpha.detach()):.3f}"
                    )
        finally:
            environment.close()

        return TrainingResult(
            agent=agent,
            episodes=pd.DataFrame(episode_rows),
            updates=pd.DataFrame(update_rows),
            evaluations=pd.concat(evaluation_frames, ignore_index=True),
            elapsed_seconds=time.perf_counter() - start_time,
        )
    """),
    code("""
    result = train_sac(config)
    print(
        f"finished {config.total_steps:,} steps and {len(result.episodes)} complete episodes "
        f"in {result.elapsed_seconds:.1f}s"
    )
    """),
    md(r"""
    ## 4. Diagnostic dashboard

    A return curve alone cannot distinguish weak exploration from critic failure.
    We retain deterministic evaluation distributions, TD scale, twin-critic
    disagreement, entropy temperature, policy spread, and Lunar-specific engine
    activation. The official environment calls an episode solved at return 200;
    the line is a task convention, not a confidence interval.
    """),
    code("""
    episodes = result.episodes.copy()
    updates = result.updates.copy()
    evaluations = result.evaluations.copy()
    evaluation_summary = (
        evaluations.groupby("step", as_index=False)
        .agg(
            mean_return=("episode_return", "mean"),
            return_std=("episode_return", "std"),
            solved_fraction=("solved", "mean"),
            mean_length=("episode_length", "mean"),
        )
        .fillna({"return_std": 0.0})
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    if not episodes.empty:
        axes[0, 0].plot(episodes["step"], episodes["episode_return"], alpha=0.35)
        axes[0, 0].plot(
            episodes["step"],
            episodes["episode_return"].rolling(20, min_periods=1).mean(),
            linewidth=2,
            label="20-episode mean",
        )
    axes[0, 0].axhline(200, color="tab:green", linestyle="--", label="solved threshold")
    axes[0, 0].set(title="Exploratory training return", xlabel="environment step")
    axes[0, 0].legend()

    x = evaluation_summary["step"].to_numpy()
    mean_return = evaluation_summary["mean_return"].to_numpy()
    return_std = evaluation_summary["return_std"].to_numpy()
    axes[0, 1].plot(x, mean_return, marker="o", label="deterministic mean")
    axes[0, 1].fill_between(x, mean_return - return_std, mean_return + return_std, alpha=0.2)
    axes[0, 1].axhline(200, color="tab:green", linestyle="--")
    axes[0, 1].set(title="Paired-seed evaluation", xlabel="environment step")
    solved_axis = axes[0, 1].twinx()
    solved_axis.plot(x, evaluation_summary["solved_fraction"], color="tab:green", alpha=0.6)
    solved_axis.set(ylabel="solved fraction", ylim=(-0.03, 1.03))

    if not updates.empty:
        axes[0, 2].plot(updates["step"], updates["q1_loss"], label="Q1")
        axes[0, 2].plot(updates["step"], updates["q2_loss"], label="Q2", alpha=0.75)
        axes[0, 2].set_yscale("log")
        axes[0, 2].legend()
        axes[1, 0].plot(
            updates["step"], updates["mean_absolute_td_error"], label="|TD error|"
        )
        axes[1, 0].plot(
            updates["step"], updates["q_disagreement"], label="|Q1-Q2|", alpha=0.8
        )
        axes[1, 0].set_yscale("log")
        axes[1, 0].legend()
        axes[1, 1].plot(updates["step"], updates["actor_loss"], label="actor loss")
        alpha_axis = axes[1, 1].twinx()
        alpha_axis.plot(updates["step"], updates["alpha"], color="tab:red", label="alpha")
        alpha_axis.set_ylabel("temperature alpha", color="tab:red")
        axes[1, 2].plot(updates["step"], updates["entropy"], label="policy entropy")
        axes[1, 2].plot(updates["step"], updates["mean_log_std"], label="mean log std")
        axes[1, 2].legend()
    axes[0, 2].set(title="Twin critic losses", xlabel="environment step")
    axes[1, 0].set(title="Critic uncertainty proxies", xlabel="environment step")
    axes[1, 1].set(title="Actor objective / temperature", xlabel="environment step")
    axes[1, 2].set(title="Exploration diagnostics", xlabel="environment step")
    plt.tight_layout()
    if SMOKE:
        plt.close(fig)
    else:
        plt.show()

    action_summary = (
        evaluations.groupby("step", as_index=False)
        .agg(
            main_engine_rate=("main_engine_rate", "mean"),
            lateral_engine_rate=("lateral_engine_rate", "mean"),
            action_saturation_rate=("action_saturation_rate", "mean"),
        )
    )
    engine_figure, engine_axes = plt.subplots(1, 2, figsize=(13, 4))
    if not episodes.empty:
        engine_axes[0].plot(
            episodes["step"],
            episodes["main_engine_rate"].rolling(20, min_periods=1).mean(),
            label="main engine",
        )
        engine_axes[0].plot(
            episodes["step"],
            episodes["lateral_engine_rate"].rolling(20, min_periods=1).mean(),
            label="lateral engines",
        )
    engine_axes[0].set(
        xlabel="environment step",
        ylabel="activation rate",
        ylim=(-0.03, 1.03),
        title="Exploratory engine use",
    )
    engine_axes[0].legend()
    engine_axes[1].plot(
        action_summary["step"], action_summary["main_engine_rate"], marker="o", label="main"
    )
    engine_axes[1].plot(
        action_summary["step"], action_summary["lateral_engine_rate"], marker="o", label="lateral"
    )
    engine_axes[1].plot(
        action_summary["step"],
        action_summary["action_saturation_rate"],
        marker="o",
        label="saturated components",
    )
    engine_axes[1].set(
        xlabel="environment step",
        ylabel="rate",
        ylim=(-0.03, 1.03),
        title="Deterministic evaluation actions",
    )
    engine_axes[1].legend()
    plt.tight_layout()
    if SMOKE:
        plt.close(engine_figure)
    else:
        plt.show()

    final_step = evaluations["step"].max()
    final_evaluation = evaluations[evaluations["step"] == final_step]
    final_summary = pd.Series(
        {
            "training_step": int(final_step),
            "mean_return": final_evaluation["episode_return"].mean(),
            "return_std": final_evaluation["episode_return"].std(ddof=0),
            "median_return": final_evaluation["episode_return"].median(),
            "solved_fraction": final_evaluation["solved"].mean(),
            "mean_episode_length": final_evaluation["episode_length"].mean(),
            "main_engine_rate": final_evaluation["main_engine_rate"].mean(),
            "lateral_engine_rate": final_evaluation["lateral_engine_rate"].mean(),
            "action_saturation_rate": final_evaluation["action_saturation_rate"].mean(),
        },
        name="final deterministic evaluation",
    )
    display(final_summary.to_frame())
    """),
    md(r"""
    ## 5. Pick a showcase without contaminating the benchmark

    The animation seeds are disjoint from the paired evaluation seeds. If at
    least one fixed showcase seed is solved, we animate the median solved rollout;
    otherwise we show the best available attempt and label it honestly. Only the
    seed-aggregated panel above is performance evidence.
    """),
    code("""
    @dataclass
    class Rollout:
        seed: int
        total_return: float
        terminated: bool
        truncated: bool
        observations: np.ndarray
        actions: np.ndarray
        rewards: np.ndarray
        frames: list[np.ndarray]
        frame_steps: np.ndarray


    def record_rollout(agent, seed, capture_stride=2):
        environment = make_lander(render_mode="rgb_array")
        observations, actions, rewards, frames, frame_steps = [], [], [], [], []
        try:
            observation, _ = environment.reset(seed=int(seed))
            observations.append(observation.copy())
            frames.append(environment.render()[::2, ::2].copy())
            frame_steps.append(0)
            terminated = truncated = False
            step = 0
            while not (terminated or truncated):
                action = agent.act(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = environment.step(action)
                step += 1
                actions.append(action.copy())
                rewards.append(float(reward))
                observations.append(observation.copy())
                if step % capture_stride == 0 or terminated or truncated:
                    frames.append(environment.render()[::2, ::2].copy())
                    frame_steps.append(step)
        finally:
            environment.close()
        return Rollout(
            seed=int(seed),
            total_return=float(np.sum(rewards)),
            terminated=bool(terminated),
            truncated=bool(truncated),
            observations=np.asarray(observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            frames=frames,
            frame_steps=np.asarray(frame_steps, dtype=int),
        )


    showcase_count = 1 if SMOKE else (4 if QUICK else 12)
    SHOWCASE_SEEDS = tuple(range(20_000, 20_000 + showcase_count))
    showcase_scores = evaluate_policy(result.agent, SHOWCASE_SEEDS)
    solved_scores = showcase_scores[showcase_scores["solved"]]
    if not solved_scores.empty:
        solved_median = solved_scores["episode_return"].median()
        selected_row = solved_scores.iloc[
            (solved_scores["episode_return"] - solved_median).abs().argmin()
        ]
        selection_reason = "median solved showcase"
    else:
        selected_row = showcase_scores.loc[showcase_scores["episode_return"].idxmax()]
        selection_reason = "best available attempt; no showcase seed solved"

    showcase = record_rollout(result.agent, int(selected_row["seed"]))
    showcase_scores = showcase_scores.assign(
        selected=showcase_scores["seed"].eq(showcase.seed)
    )
    print(f"Selected seed {showcase.seed}: {selection_reason}")
    display(showcase_scores)
    """),
    md(r"""
    ## 6. Landing replay with flight telemetry

    The animation is generated from a fresh deterministic evaluation episode.
    Its heads-up display reports position, velocity, attitude, engine commands,
    and accumulated reward. Playback is downsampled to at most 300 frames so the
    notebook remains responsive.
    """),
    code("""
    def make_landing_animation(rollout, fps=30, max_frames=300):
        if len(rollout.frames) > max_frames:
            frame_indices = np.linspace(
                0, len(rollout.frames) - 1, max_frames, dtype=int
            )
        else:
            frame_indices = np.arange(len(rollout.frames))
        cumulative_rewards = np.cumsum(rollout.rewards)

        fig, axis = plt.subplots(figsize=(8, 5.4), dpi=80)
        fig.patch.set_facecolor("#070b14")
        axis.set_facecolor("#070b14")
        image = axis.imshow(rollout.frames[int(frame_indices[0])])
        axis.set_axis_off()
        status = "SOLVED" if rollout.total_return >= 200 else "ATTEMPT"
        title = axis.set_title(
            f"LunarLander SAC — {status} — seed {rollout.seed}",
            color="white",
            fontsize=14,
            pad=10,
        )
        hud = axis.text(
            0.018,
            0.975,
            "",
            transform=axis.transAxes,
            va="top",
            ha="left",
            color="#e8f1ff",
            family="monospace",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.55", "facecolor": "#08101f", "alpha": 0.82, "edgecolor": "#4ea5ff"},
        )

        def update(animation_index):
            stored_index = int(frame_indices[animation_index])
            step = int(rollout.frame_steps[stored_index])
            state = rollout.observations[min(step, len(rollout.observations) - 1)]
            if step:
                action = rollout.actions[min(step - 1, len(rollout.actions) - 1)]
                accumulated_reward = cumulative_rewards[min(step - 1, len(cumulative_rewards) - 1)]
            else:
                action = np.zeros(ACT_DIM)
                accumulated_reward = 0.0
            image.set_data(rollout.frames[stored_index])
            hud.set_text(
                f"step      {step:4d}\\n"
                f"return   {accumulated_reward:7.1f} / {rollout.total_return:7.1f}\\n"
                f"x, y     {state[0]:+6.3f}  {state[1]:+6.3f}\\n"
                f"vx, vy   {state[2]:+6.3f}  {state[3]:+6.3f}\\n"
                f"angle    {state[4]:+6.3f}\\n"
                f"main     {action[0]:+6.3f}\\n"
                f"lateral  {action[1]:+6.3f}"
            )
            return image, hud, title

        movie = animation.FuncAnimation(
            fig,
            update,
            frames=len(frame_indices),
            interval=1_000 / fps,
            blit=True,
            repeat=False,
        )
        return fig, movie


    if SMOKE:
        smoke_figure = plt.figure(figsize=(6, 4))
        plt.imshow(showcase.frames[-1])
        plt.axis("off")
        plt.title("Animation HTML skipped only in automated smoke mode")
        plt.close(smoke_figure)
        landing_movie = None
    else:
        plt.rcParams["animation.embed_limit"] = 120
        landing_figure, landing_movie = make_landing_animation(showcase)
        landing_html = HTML(landing_movie.to_jshtml(fps=30, default_mode="once"))
        plt.close(landing_figure)
        display(landing_html)

    # Optional portable artifact. Toggle after training if you want a GIF on disk.
    SAVE_GIF = False
    if SAVE_GIF and landing_movie is not None:
        output_directory = Path("results/lunar_lander_sac")
        output_directory.mkdir(parents=True, exist_ok=True)
        gif_path = output_directory / f"landing_seed_{showcase.seed}.gif"
        landing_movie.save(gif_path, writer=animation.PillowWriter(fps=30))
        print(f"saved {gif_path}")
    """),
    code("""
    states = showcase.observations
    actions = showcase.actions
    state_steps = np.arange(len(states))
    action_steps = np.arange(1, len(actions) + 1)
    cumulative_reward = np.cumsum(showcase.rewards)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes[0, 0].plot(states[:, 0], states[:, 1], color="tab:blue")
    axes[0, 0].scatter(states[0, 0], states[0, 1], marker="o", label="start")
    axes[0, 0].scatter(states[-1, 0], states[-1, 1], marker="X", s=80, label="finish")
    axes[0, 0].scatter(0, 0, marker="*", s=110, color="tab:green", label="pad center")
    axes[0, 0].set(xlabel="x", ylabel="y", title="Flight path in observation coordinates")
    axes[0, 0].legend()

    axes[0, 1].plot(state_steps, states[:, 2], label="horizontal velocity")
    axes[0, 1].plot(state_steps, states[:, 3], label="vertical velocity")
    axes[0, 1].plot(state_steps, states[:, 4], label="angle", alpha=0.8)
    axes[0, 1].axhline(0, color="black", linewidth=0.8)
    axes[0, 1].set(xlabel="step", title="Approach stability")
    axes[0, 1].legend()

    axes[1, 0].plot(action_steps, actions[:, 0], label="main")
    axes[1, 0].plot(action_steps, actions[:, 1], label="lateral")
    axes[1, 0].axhline(0, color="tab:blue", linestyle="--", alpha=0.5)
    axes[1, 0].axhline(0.5, color="tab:orange", linestyle=":", alpha=0.5)
    axes[1, 0].axhline(-0.5, color="tab:orange", linestyle=":", alpha=0.5)
    axes[1, 0].set(xlabel="step", ylabel="action", title="Engine commands and dead zones")
    axes[1, 0].legend()

    axes[1, 1].plot(action_steps, cumulative_reward, color="tab:green")
    axes[1, 1].set(xlabel="step", ylabel="cumulative reward", title="Reward accumulation")
    plt.tight_layout()
    if SMOKE:
        plt.close(fig)
    else:
        plt.show()
    """),
    md(r"""
    ## 7. What to trust, save, and try next

    - `QUICK=True` validates mechanics. It is expected to crash often.
    - For a real baseline, use `QUICK=False`, at least three independent training
      seeds, and 50+ untouched final evaluation seeds. Report the distribution,
      not the best animation.
    - Wind is disabled for the baseline. Turn it on only as a separately labeled
      robustness experiment.
    - SAC can still fail through critic extrapolation, temperature collapse,
      overconfident Q targets, or action saturation; the dashboard is designed to
      make those failures visible.

    The trained actor remains in `result.agent`. Set `SAVE_GIF=True` to export the
    replay, or save `result.agent.actor.state_dict()` together with `asdict(config)`
    when you want a checkpoint.
    """),
]


NOTEBOOK_SPECS = {
    "00_rl_primer.ipynb": primer,
    "01_stochastic_maze.ipynb": maze_lab,
    "02_q_learning_experiments.ipynb": q_experiments,
    "03_lunar_lander_sac.ipynb": lunar_lander_sac,
    "04_policies_under_risk_drift_and_memory.ipynb": policies_under_risk_drift_memory,
    "05_shortcut_or_shelter.ipynb": shortcut_or_shelter,
}

if CHECK:
    expected_notebooks = set(NOTEBOOK_SPECS)
    actual_notebooks = {path.name for path in NOTEBOOKS.glob("*.ipynb")}
    if actual_notebooks != expected_notebooks:
        missing = sorted(expected_notebooks - actual_notebooks)
        unexpected = sorted(actual_notebooks - expected_notebooks)
        raise SystemExit(f"Notebook inventory mismatch; missing={missing}, unexpected={unexpected}")

for notebook_name, notebook_cells in NOTEBOOK_SPECS.items():
    write_notebook(notebook_name, notebook_cells)

verb = "Checked" if CHECK else "Wrote"
for notebook_name in NOTEBOOK_SPECS:
    print(f"{verb} {NOTEBOOKS / notebook_name}")

"""Persistent synthetic environment, path-dependent plastic agent and trace measurement.

All three provenance routes drive the same plastic pathway: one delta-rule update of
the slow state h = (W, U) per step, with identical gain. Routes differ only in where
the (input, target) pair comes from, so source cannot be read off the presence or
absence of a trace. The label y_k lives in the simulator; the log record, the query and
the trace computation never see it.
"""
import numpy as np

ROUTES = ("own", "imported", "inferred")
ROLES = ("train", "val", "test", "m2")


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class Measurement:
    """Preregistered M and Phi (config.measurement)."""

    def __init__(self, mcfg, d_state):
        rng = np.random.default_rng(mcfg["R_seed"])
        self.R = rng.standard_normal((mcfg["d_z"], d_state)) / np.sqrt(mcfg["d_z"])
        self.sigma = mcfg["sigma_z"]

    def M(self, dh, eta, rng):
        """z_t = R vec(h_t - h_{t-1}) / eta + eta_t, for all steps of one episode."""
        z = dh @ self.R.T / eta
        return z + self.sigma * rng.standard_normal(z.shape)

    @staticmethod
    def Phi(z_ep):
        """s_k = (mean, sd) of z over the steps of the episode."""
        return np.concatenate([z_ep.mean(0), z_ep.std(0)])


class Agent:
    """Slow state h = (W, U): fast synapse W consolidating into a slow variable U."""

    def __init__(self, pcfg, d_out, d_in):
        self.W = np.zeros((d_out, d_in))
        self.U = np.zeros((d_out, d_in))
        self.eta, self.k_wu, self.k_uw = pcfg["eta"], pcfg["k_wu"], pcfg["k_uw"]
        self.rho, self.sxi = pcfg["rho"], pcfg["synaptic_noise"]
        self.n = d_out * d_in

    def state(self):
        return np.concatenate([self.W.ravel(), self.U.ravel()])

    def set_state(self, h):
        self.W = h[: self.n].reshape(self.W.shape).copy()
        self.U = h[self.n:].reshape(self.U.shape).copy()

    def run_episode(self, ep, gamma, rng):
        """Apply the learning rule for every step of the episode; return h_t - h_{t-1}."""
        xs = ep["xs"]
        dh = np.empty((len(xs), 2 * self.n))
        for t, x in enumerate(xs):
            if ep["route"] == 2:   # rehearsal: target regenerated from slow memory, partly inferred
                target = (1.0 - gamma) * (self.U @ x) + gamma * ep["o"]
            else:
                target = ep["ts"][t]
            e = target - self.W @ x
            dW = self.eta * np.outer(e, x) - self.k_wu * (self.W - self.U)
            dW += self.sxi * rng.standard_normal(dW.shape)
            dU = self.k_uw * (self.W - self.U) - self.rho * self.U
            self.W += dW
            self.U += dU
            dh[t, : self.n] = dW.ravel()
            dh[t, self.n:] = dU.ravel()
        return dh


class World:
    """Persistent environment: kinds, world map, scripted other agent, shared code."""

    def __init__(self, ecfg, rng):
        self.c = ecfg
        d_in, d_out = ecfg["d_in"], ecfg["d_out"]
        self.mu = _unit(rng.standard_normal((ecfg["n_kinds"], d_in)))
        self.A = rng.standard_normal((d_out, d_in)) * ecfg["world_gain"] / np.sqrt(d_in)
        err = ecfg["imported"]["sender_model_error"]
        self.A_sender = self.A + rng.standard_normal((d_out, d_in)) * err / np.sqrt(d_in)
        Q, _ = np.linalg.qr(rng.standard_normal((d_in, ecfg["imported"]["comm_dim"])))
        self.P_comm = Q @ Q.T
        self.memory = []   # cues of previously acquired items (internal content for rehearsal)

    def episode(self, route, kind, rng):
        """Episode specification. The record/query are drawn independently of the route."""
        E, T = self.c, self.c["steps_per_episode"]
        d_in, d_out = E["d_in"], E["d_out"]
        c = _unit(self.mu[kind] + E["cue_spread"] * rng.standard_normal(d_in) / np.sqrt(d_in))
        o = np.tanh(self.A @ c)
        ep = {"route": route, "kind": kind, "o": o}
        if route == 0:      # own: the agent acts and observes, step after step
            p = E["own"]
            ep["xs"] = c + p["perceptual_noise"] * rng.standard_normal((T, d_in)) / np.sqrt(d_in)
            ep["ts"] = o + p["outcome_noise"] * rng.standard_normal((T, d_out)) / np.sqrt(d_out)
        elif route == 1:    # imported: the other agent describes the contingency
            p = E["imported"]
            x = _unit(self.P_comm @ c)
            ep["xs"] = np.repeat(x[None], T, 0)
            ep["ts"] = np.tanh(self.A_sender @ c) + p["outcome_noise"] * rng.standard_normal((T, d_out)) / np.sqrt(d_out)
        else:               # inferred: recombination of the item with earlier retained content
            p = E["inferred"]
            cj = self.memory[rng.integers(len(self.memory))] if self.memory else self.mu[kind]
            x = _unit((1.0 - p["blend"]) * c + p["blend"] * cj)
            ep["xs"] = np.repeat(x[None], T, 0)
        self.memory.append(c)
        if len(self.memory) > E["inferred"]["memory_buffer"]:
            self.memory.pop(0)
        # log record and retained proposition: route-independent random descriptors
        r = rng.standard_normal(E["d_record"])
        ep["record"] = r
        ep["query"] = r + E["query_noise"] * rng.standard_normal(E["d_record"])
        return ep


def window_layout(cfg):
    """Fixed timing of a window: log timestamps (episode starts), trace timestamps, boundaries."""
    E = cfg["environment"]
    K, T = E["episodes_per_window"], E["steps_per_episode"]
    t_tr = np.arange(K * T)
    return {"t_log": np.arange(K) * T, "t_tr": t_tr, "bnd": (t_tr % T == 0).astype(np.float32)}


def role_sequence(cfg, rng):
    D = cfg["data"]
    n_train = D["n0_windows"] * max(D["budgets"])
    n_m2 = D["m2_multiplier"] * D["n0_windows"] * D["confirmatory_budget"]
    roles = np.repeat(np.arange(4), [n_train, D["n_val_windows"], D["n_test_windows"], n_m2])
    return rng.permutation(roles)   # roles interleaved along one continuous life


def m1_check(cfg, agent, world, rng):
    """M1: final-state distance under permuted episode order vs test-retest (same order)."""
    C, g = cfg["checks"], cfg["environment"]["inferred"]["infer_weight"]
    h0 = agent.state()
    mem0 = list(world.memory)
    E = cfg["environment"]
    block = []
    for _ in range(C["m1_block_episodes"]):
        block.append(world.episode(rng.choice(3, p=E["route_probs"]), rng.integers(E["n_kinds"]), rng))
    world.memory = mem0

    def replay(order, noise_seed):
        agent.set_state(h0)
        r = np.random.default_rng(noise_seed)
        for i in order:
            agent.run_episode(block[i], g, r)
        return agent.state()

    ident = np.arange(len(block))
    retest, permuted = [], []
    for rep in range(C["m1_repeats"]):
        a = replay(ident, 10_000 + 2 * rep)
        b = replay(ident, 10_001 + 2 * rep)
        p = replay(rng.permutation(len(block)), 20_000 + rep)
        retest.append(np.linalg.norm(a - b))
        permuted.append(np.linalg.norm(p - a))
    scale = np.linalg.norm(a)
    agent.set_state(h0)
    retest, permuted = np.array(retest) / scale, np.array(permuted) / scale
    delta_c = float(np.quantile(retest, C["m1_quantile"]))
    return {"retest_rel": retest.tolist(), "permuted_rel": permuted.tolist(),
            "delta_c": delta_c, "median_permuted": float(np.median(permuted)),
            "ratio": float(np.median(permuted) / delta_c),
            "pass": bool(np.median(permuted) > delta_c)}


def simulate_life(cfg, seed, with_m1=True):
    """One agent life in one persistent environment instance.

    Returns per-window arrays for all roles plus the M1 result. Window roles
    (train / val / test / m2-decoder) are interleaved along the same life.
    """
    E, P = cfg["environment"], cfg["plasticity"]
    K, T, nk = E["episodes_per_window"], E["steps_per_episode"], E["n_kinds"]
    rng_env = np.random.default_rng([seed, 1])
    rng_xi = np.random.default_rng([seed, 2])
    rng_eta = np.random.default_rng([seed, 3])
    world = World(E, rng_env)
    agent = Agent(P, E["d_out"], E["d_in"])
    meas = Measurement(cfg["measurement"], 2 * E["d_out"] * E["d_in"])
    g = E["inferred"]["infer_weight"]

    for _ in range(E["burn_in_episodes"]):
        ep = world.episode(rng_env.choice(3, p=E["route_probs"]), rng_env.integers(nk), rng_env)
        agent.run_episode(ep, g, rng_xi)
    m1 = m1_check(cfg, agent, world, np.random.default_rng([seed, 4])) if with_m1 else None

    roles = role_sequence(cfg, np.random.default_rng([seed, 5]))
    nw, dz = len(roles), cfg["measurement"]["d_z"]
    out = {
        "role": roles.astype(np.int8),
        "y": np.empty((nw, K), np.int8),
        "kind": np.empty((nw, K), np.int8),
        "rec": np.empty((nw, K, E["d_record"]), np.float32),
        "q": np.empty((nw, K, E["d_record"]), np.float32),
        "z": np.empty((nw, K * T, dz), np.float32),
        "s": np.empty((nw, K, 2 * dz), np.float32),
    }
    for w in range(nw):
        for k in range(K):
            route = rng_env.choice(3, p=E["route_probs"])
            kind = rng_env.integers(nk)
            ep = world.episode(route, kind, rng_env)
            z = meas.M(agent.run_episode(ep, g, rng_xi), P["eta"], rng_eta)
            out["y"][w, k], out["kind"][w, k] = route, kind
            out["rec"][w, k], out["q"][w, k] = ep["record"], ep["query"]
            out["z"][w, k * T:(k + 1) * T] = z
            out["s"][w, k] = meas.Phi(z)
    return out, m1


def records_with_kind(rec, kind, n_kinds):
    """Log record / query features = descriptor plus one-hot content kind."""
    return np.concatenate([rec, np.eye(n_kinds, dtype=np.float32)[kind]], -1)

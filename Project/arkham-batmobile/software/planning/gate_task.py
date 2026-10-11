"""
The M4 task: drive through a narrow gate and stop at a pose beyond it.

Chosen because it is a task where being WRONG ABOUT SLIP COSTS YOU. A gate with
~120 mm of clearance either side is tight enough that a 20 mm prediction error
at the moment of passing is the difference between through and clipped, and the
approach usually needs diagonal strafe - the mode that loses most to slip
(81% delivered vs 92% forward, see experiments/P0-sim-baseline).

An open-field reach would not discriminate: every predictor gets there, slowly
or quickly, and the reactive controller wins on simplicity.
"""

import numpy as np

ROBOT_HALF_LENGTH = 0.10
ROBOT_HALF_WIDTH = 0.065
POST_RADIUS = 0.022
CLEARANCE = ROBOT_HALF_WIDTH + POST_RADIUS      # 87 mm: centre-to-post contact
BARRIER_MARGIN = 0.012                          # must be < min gate half-width - CLEARANCE


class GateTask:
    """One randomly placed gate, a start pose before it, a goal pose beyond it."""

    def __init__(self, rng: np.random.Generator):
        # Gate centre, orientation, and half-width. Half-width 0.125-0.165 m
        # leaves 38-78 mm of clear air each side of the chassis. The first
        # version used 0.17-0.22 m (83-133 mm) and every condition including
        # the model-free reactive controller scored 100%, so it measured
        # nothing. 0.125-0.165 m still saturated every MPPI condition at
        # 100%, so this is 0.105-0.140 m: 18-53 mm of clear air per side.
        self.gate_xy = np.array([rng.uniform(-0.15, 0.25), rng.uniform(-0.30, 0.30)])
        self.gate_angle = rng.uniform(-0.5, 0.5)
        self.half_width = rng.uniform(0.105, 0.140)

        normal = np.array([np.cos(self.gate_angle), np.sin(self.gate_angle)])
        along = np.array([-normal[1], normal[0]])
        self.post_l = self.gate_xy + along * self.half_width
        self.post_r = self.gate_xy - along * self.half_width

        approach = rng.uniform(0.55, 0.75)
        lateral = rng.uniform(-0.22, 0.22)
        self.start = np.array(
            [
                *(self.gate_xy - normal * approach + along * lateral),
                self.gate_angle + rng.uniform(-0.6, 0.6),
            ]
        )
        self.goal = np.array([*(self.gate_xy + normal * 0.33), self.gate_angle])
        self.normal = normal
        self.along = along

    # ── Scoring ─────────────────────────────────────────────────────────────

    def post_distances(self, poses: np.ndarray) -> np.ndarray:
        """Min distance from each pose's centre to either post. Shape in = (..., 3)."""
        xy = poses[..., :2]
        dl = np.linalg.norm(xy - self.post_l, axis=-1)
        dr = np.linalg.norm(xy - self.post_r, axis=-1)
        return np.minimum(dl, dr)

    def collided(self, poses: np.ndarray) -> np.ndarray:
        """
        Circle approximation of the chassis against the posts.

        Conservative: uses the half-WIDTH, so a robot passing broadside is
        scored more leniently than reality. Good enough to rank conditions,
        and documented rather than silently assumed.
        """
        return self.post_distances(poses) < CLEARANCE

    def reached(self, pose: np.ndarray, tol: float = 0.09) -> bool:
        return bool(np.linalg.norm(pose[:2] - self.goal[:2]) < tol)

    def past_gate(self, pose: np.ndarray) -> bool:
        return bool((pose[:2] - self.gate_xy) @ self.normal > 0.0)

    # ── MPPI cost ───────────────────────────────────────────────────────────

    def cost(self, poses: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        poses (K, H, 3) -> cost (K,).

        Terminal distance to goal, a soft barrier around each post, and a small
        heading term. No collision hard-constraint: MPPI needs a gradient to
        follow, and an infinite wall gives it none.
        """
        terminal = np.linalg.norm(poses[:, -1, :2] - self.goal[:2], axis=1)
        progress = np.mean(np.linalg.norm(poses[..., :2] - self.goal[:2], axis=2), axis=1)

        # Barrier margin must be SMALLER than the narrowest gate half-width,
        # or the two posts' zones overlap across the whole opening: every
        # trajectory including the optimal one takes a constant penalty, the
        # gradients from the two sides cancel, and the planner gets no steering
        # signal at all.
        #
        # Measured the hard way. At margin 0.05 the barrier activated below
        # 137 mm while gate half-widths are 105-140 mm, so a perfect centre-line
        # pass was already penalised. Success rate then tracked nothing - a 2.3x
        # more accurate model scored the same - because the cost function, not
        # the dynamics model, was the binding constraint.
        d = self.post_distances(poses)
        barrier = np.sum(np.clip(CLEARANCE + BARRIER_MARGIN - d, 0.0, None) ** 2, axis=1)

        heading = np.abs((poses[:, -1, 2] - self.goal[2] + np.pi) % (2 * np.pi) - np.pi)

        return 12.0 * terminal + 1.5 * progress + 400.0 * barrier + 0.6 * heading

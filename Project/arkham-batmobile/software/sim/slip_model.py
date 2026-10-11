"""
Slip model — the deliberate imperfection in the Phase 0 simulator.

WHY THIS FILE EXISTS
--------------------
If the simulator realised commanded velocities exactly, mecanum forward
kinematics would be a perfect predictor, M2 would trivially "succeed" in sim,
and the whole thing would collapse the moment it met a real floor. So the sim
is built wrong on purpose, in the same way real hardware is wrong.

THE ONE RULE
------------
The residual must be a *structured function of state and action* - speed,
direction, surface, pack voltage, wheel angle - and NOT zero-mean noise. A
learned model can only beat the analytic baseline if there is signal to find.
Noise would make the honest result "no model can beat FK here", which is a fine
scientific outcome but a useless training environment.

WHERE SLIP IS INJECTED
----------------------
Between commanded wheel speed and achieved wheel speed, which is where it
happens on real hardware. The chain is:

    body velocity command
        -> IK            (kinematics.mecanum, exact)
        -> saturation    (motors have a top speed)
        -> mode efficiency, speed-dependent   <- the dominant effect
        -> pack droop    (current draw sags the battery)
        -> per-wheel gain asymmetry           <- also produces scrub
        -> roller ripple (periodic in wheel angle)
        -> small noise
        -> achieved wheel speeds  ==  what the encoders report
        -> FK            -> achieved body velocity, handed to MuJoCo

MODE EFFICIENCY
---------------
Section 5 of notes/just-in-time/mecanum-kinematics.md shows J has orthogonal
columns, so wheel-space decomposes cleanly into three independent modes:
forward, strafe, rotate. That makes "strafing loses more to slip than driving"
expressible as a single per-mode gain - which is both the physically dominant
real-world effect and the most natural thing to write here.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinematics.mecanum import MecanumKinematics  # noqa: E402


@dataclass
class SurfaceParams:
    """
    One floor surface. The real robot gets these fitted from data in M1; here
    they are the ground truth the learned model is trying to rediscover.
    """

    name: str

    # Per-mode efficiency at zero speed: (forward, strafe, rotate).
    # Strafe is worst because lateral motion loads the rollers in their
    # compliant direction. This asymmetry is the main thing M2 should learn.
    eta0: tuple[float, float, float] = (0.980, 0.905, 0.955)

    # Extra fractional loss per unit speed, same mode order. Slip grows with
    # speed, so error-vs-horizon curves steepen on fast episodes.
    eta_speed: tuple[float, float, float] = (0.055, 0.170, 0.040)

    # Fixed per-wheel gain error (FL, FR, RL, RR): gearbox spread, tyre wear,
    # one PID tuned differently. Asymmetric on purpose - it biases the body
    # velocity AND shows up as non-zero scrub, exactly like a real base.
    wheel_gain: tuple[float, float, float, float] = (0.985, 1.010, 0.995, 0.972)

    # Roller contact migrates as each roller rotates into and out of contact,
    # giving mecanum drive a characteristic small periodic ripple.
    ripple_amp: float = 0.022      # fraction of wheel speed
    ripple_rollers: int = 9        # rollers per wheel -> ripple frequency

    # Pack sag: achieved speed drops with total commanded current.
    droop_per_amp: float = 0.0032
    amp_per_rad_s: float = 0.26

    noise_std: float = 0.012       # rad/s, small relative to the structure above


SURFACES: dict[str, SurfaceParams] = {
    # Hard flooring: the primary arena. Mild, well-behaved slip.
    "hard": SurfaceParams(name="hard"),
    # Thin low-pile mat: the M6 domain-shift condition. Worse everywhere, and
    # disproportionately worse on strafe - rollers bind in soft pile.
    "mat": SurfaceParams(
        name="mat",
        eta0=(0.945, 0.790, 0.900),
        eta_speed=(0.115, 0.330, 0.090),
        wheel_gain=(0.975, 1.018, 0.988, 0.958),
        ripple_amp=0.035,
        droop_per_amp=0.0041,
        noise_std=0.019,
    ),
}


@dataclass
class SlipState:
    wheel_angle: np.ndarray = field(default_factory=lambda: np.zeros(4))
    pack_soc: float = 1.0          # state of charge, 1.0 -> 0.0 over a session
    last_wheel: np.ndarray = field(default_factory=lambda: np.zeros(4))


class SlipModel:
    """
    Maps a commanded body velocity to the body velocity the wheels actually
    deliver, plus the wheel speeds the encoders would report.

    Deterministic given a seed, so an episode is exactly reproducible from its
    manifest.
    """

    WHEEL_SPEED_MAX = 14.0         # rad/s, ~134 rpm: a 0.5 m/s base with r=40 mm
    PACK_NOMINAL_V = 12.6          # 3S LiPo, fully charged
    PACK_EMPTY_V = 10.5            # session stops here

    def __init__(
        self,
        surface: str = "hard",
        kinematics: MecanumKinematics | None = None,
        seed: int = 0,
    ):
        if surface not in SURFACES:
            raise ValueError(f"unknown surface {surface!r}, expected one of {list(SURFACES)}")
        self.params = SURFACES[surface]
        self.kin = kinematics or MecanumKinematics()
        self.rng = np.random.default_rng(seed)
        self.state = SlipState()

    def reset(self, pack_soc: float = 1.0) -> None:
        self.state = SlipState(pack_soc=float(pack_soc))

    @property
    def pack_voltage(self) -> float:
        """Open-circuit pack voltage from state of charge."""
        p = self.params
        del p
        return self.PACK_EMPTY_V + (self.PACK_NOMINAL_V - self.PACK_EMPTY_V) * self.state.pack_soc

    def step(self, body_vel_cmd: np.ndarray, dt: float) -> dict:
        """
        body_vel_cmd  (3,)  commanded (vx, vy, omega) in the body frame
        dt            float control period

        Returns a dict with:
            wheel_cmd       (4,) what the controller asked each wheel for
            wheel_meas      (4,) what the encoders report (the achieved speed)
            body_vel        (3,) achieved body velocity -> goes to MuJoCo
            batt_v          float pack voltage under load
            saturated       bool any wheel at its speed limit
        """
        p = self.params
        cmd = np.asarray(body_vel_cmd, dtype=float)

        # ── Motor-side effects: these change what the WHEELS do, so the encoders
        #    see them. Everything in this block is observable in wheel_meas.

        # 1. Exact inverse kinematics, then motor speed limits.
        #
        # Saturation scales ALL FOUR wheels by one factor rather than clipping
        # each independently. Independent clipping breaks the mirrored roller
        # symmetry, so a straight strafe becomes a curve and the discarded
        # motion lands in the scrub direction. (Observed: clipping drove mean
        # |scrub| to 1.0 rad/s, twenty times its proper value.)
        # The relative tolerance matters: callers clamp actions to exactly
        # WHEEL_SPEED_MAX, so an exact-equality test flips on floating-point
        # noise and reports ~43% of ticks as saturated when none are.
        wheel_cmd = self.kin.ik(cmd)
        peak = float(np.max(np.abs(wheel_cmd)))
        saturated = bool(peak > self.WHEEL_SPEED_MAX * (1.0 + 1e-9))
        wheel = wheel_cmd * (self.WHEEL_SPEED_MAX / peak) if saturated else wheel_cmd.copy()

        # 2. Pack droop. Total current scales with total wheel effort; a sagging
        #    pack delivers less speed for the same duty cycle. batt_v is logged
        #    as a model INPUT, not metadata - without it this is unobservable.
        current = p.amp_per_rad_s * float(np.sum(np.abs(wheel)))
        droop = 1.0 - p.droop_per_amp * current * (2.0 - self.state.pack_soc)
        wheel = wheel * max(droop, 0.5)

        # 3. Per-wheel gain asymmetry: gearbox spread, tyre wear, one PID tuned
        #    differently. Biases the body velocity and injects scrub.
        wheel = wheel * np.array(p.wheel_gain)

        # 4. Roller ripple: periodic in wheel angle, so it is predictable from
        #    integrated encoder counts and invisible to a memoryless model.
        self.state.wheel_angle = self.state.wheel_angle + wheel * dt
        ripple = 1.0 + p.ripple_amp * np.sin(p.ripple_rollers * self.state.wheel_angle)
        wheel = wheel * ripple

        # 5. Encoder noise, deliberately small next to the structure above.
        wheel_meas = wheel + self.rng.normal(0.0, p.noise_std, size=4)

        # ── Contact-side effects: these happen BETWEEN the wheel and the floor,
        #    so the encoders never see them. This is the crucial asymmetry - the
        #    wheels turn, the robot does not move as far, and wheel odometry
        #    therefore OVER-estimates motion. That drift is what makes the
        #    overhead fiducial the only honest pose label (and why the real rig
        #    needs one).

        # 6. Per-mode contact efficiency. Decompose what FK believes the body is
        #    doing into the three orthogonal modes, scale each by its own
        #    speed-dependent efficiency, recompose. Strafe loses most because
        #    lateral motion loads the rollers in their compliant direction.
        v_wheel = self.kin.fk(wheel_meas)                   # what FK thinks
        speed = np.abs(v_wheel)
        eta = np.array(p.eta0) * (1.0 - np.array(p.eta_speed) * speed)
        eta = np.clip(eta, 0.30, 1.0)
        body_vel = v_wheel * eta                            # what actually happens

        # 7. Drain the pack in proportion to the work done.
        self.state.pack_soc = float(
            np.clip(self.state.pack_soc - 2.2e-6 * current * dt / 0.05, 0.0, 1.0)
        )
        self.state.last_wheel = wheel_meas

        return {
            "wheel_cmd": wheel_cmd,
            "wheel_meas": wheel_meas,
            "body_vel": body_vel,
            "odom_vel": v_wheel,            # FK(encoders): what naive odometry believes
            "slip_fraction": 1.0 - eta,     # ground truth, for diagnostics only
            "batt_v": self.pack_voltage - 0.055 * current,
            "saturated": saturated,
        }


def characterise(surface: str = "hard", n: int = 4000, seed: int = 0) -> dict:
    """
    Quick ground-truth summary of how wrong the analytic model is on a surface.

    This is the sim equivalent of the M1 calibration sweep, and it predicts what
    the real M1 report should look like in shape (not in magnitude).
    """
    kin = MecanumKinematics()
    slip = SlipModel(surface=surface, kinematics=kin, seed=seed)
    rng = np.random.default_rng(seed + 1)

    rows = []
    for mode, template in (
        ("forward", np.array([1.0, 0.0, 0.0])),
        ("strafe", np.array([0.0, 1.0, 0.0])),
        ("rotate", np.array([0.0, 0.0, 1.0])),
        ("diagonal", np.array([0.7, 0.7, 0.0])),
    ):
        errs, scrubs, odom_errs = [], [], []
        for _ in range(n // 4):
            scale = rng.uniform(0.1, 1.0)
            cmd = template * scale * (0.5 if mode != "rotate" else 2.0)
            out = slip.step(cmd, 0.05)
            # Fraction of the commanded motion actually delivered
            denom = np.linalg.norm(cmd)
            errs.append(np.linalg.norm(out["body_vel"] - cmd) / max(denom, 1e-9))
            odom_errs.append(np.linalg.norm(out["odom_vel"] - out["body_vel"]) / max(denom, 1e-9))
            scrubs.append(abs(kin.scrub(out["wheel_meas"])))
        rows.append(
            {
                "mode": mode,
                "mean_rel_error": float(np.mean(errs)),
                "p95_rel_error": float(np.percentile(errs, 95)),
                "mean_scrub_rad_s": float(np.mean(scrubs)),
                "mean_odom_overread": float(np.mean(odom_errs)),
            }
        )
    return {"surface": surface, "modes": rows}


if __name__ == "__main__":
    for surf in SURFACES:
        result = characterise(surf)
        print(f"\nsurface: {result['surface']}")
        print(f"  {'mode':10s} {'mean rel err':>13s} {'p95 rel err':>12s} {'odom overread':>14s} {'mean scrub':>12s}")
        for row in result["modes"]:
            print(
                f"  {row['mode']:10s} {row['mean_rel_error']:12.1%} "
                f"{row['p95_rel_error']:11.1%} {row['mean_odom_overread']:13.1%} {row['mean_scrub_rad_s']:11.3f}"
            )
    print(
        "\nStrafe should be the worst mode on both surfaces, and worse on mat than hard.\n"
        "That ordering is the structure M2 has to find."
    )

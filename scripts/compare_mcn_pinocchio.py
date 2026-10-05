#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare both MCN backends against Pinocchio.

This is an offline model check. It does not publish torque and does not need
Gazebo controllers.

It reports two comparisons when the C++ extension is available:
    1. MCN_cpp    vs Pinocchio
    2. MCN_python vs Pinocchio

Run:
    cd ~/MCN_ComputedTorque/scripts
    source /opt/ros/noetic/setup.bash
    source ~/z1_ws/devel/setup.bash
    python3 compare_mcn_pinocchio.py --samples 50 --verbose
"""

import argparse
import os

import numpy as np
import pinocchio as pin
import rospkg

from MCN_CT import JOINT_LOWER, JOINT_UPPER, MCNsim, SCARA_initialize

try:
    import mcn_dynamics
except ImportError:
    mcn_dynamics = None

np.set_printoptions(precision=6, suppress=True)


def parse_six_floats(text):
    values = [float(x) for x in text.replace(",", " ").split()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("expected exactly 6 numbers")
    return np.array(values, dtype=float)


def load_pinocchio_model():
    z1_controller_path = rospkg.RosPack().get_path("z1_controller")
    urdf_path = os.path.join(z1_controller_path, "config", "z1.urdf")

    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"Cannot find URDF: {urdf_path}")

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    if model.nq != 6 or model.nv != 6:
        raise RuntimeError(f"Expected nq=nv=6, got nq={model.nq}, nv={model.nv}")

    print("[Pinocchio] URDF:", urdf_path)
    print("[Pinocchio] nq =", model.nq, "nv =", model.nv)

    return model, data


def pin_terms(model, data, q, dq, qdd):
    M = pin.crba(model, data, q)
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)

    h = pin.nonLinearEffects(model, data, q, dq)
    h = np.asarray(h, dtype=float).reshape(6)

    G = pin.computeGeneralizedGravity(model, data, q)
    G = np.asarray(G, dtype=float).reshape(6)

    Cqd = h - G
    tau = M @ qdd + h

    return M, G, Cqd, h, tau


def mcn_terms(backend_func, params, q, dq, qdd):
    M, C, N = backend_func(
        params["xi"],
        params["calMp"],
        params["lM"],
        params["beta"],
        params["mg"],
        q,
        dq,
    )

    G = N
    Cqd = C @ dq
    h = Cqd + G
    tau = M @ qdd + h

    return M, G, Cqd, h, tau


def max_abs(x):
    return float(np.max(np.abs(x)))


def compare_backend(name, backend_func, model, data, params, states):
    accum = {
        "M": [],
        "G": [],
        "Cqd": [],
        "h": [],
        "tau": [],
    }
    worst = None

    for i, (q, dq, qdd) in enumerate(states):
        M_pin, G_pin, Cqd_pin, h_pin, tau_pin = pin_terms(model, data, q, dq, qdd)
        M_mcn, G_mcn, Cqd_mcn, h_mcn, tau_mcn = mcn_terms(
            backend_func,
            params,
            q,
            dq,
            qdd,
        )

        result = {
            "M": max_abs(M_mcn - M_pin),
            "G": max_abs(G_mcn - G_pin),
            "Cqd": max_abs(Cqd_mcn - Cqd_pin),
            "h": max_abs(h_mcn - h_pin),
            "tau": max_abs(tau_mcn - tau_pin),
            "G_pin": G_pin,
            "G_mcn": G_mcn,
            "Cqd_pin": Cqd_pin,
            "Cqd_mcn": Cqd_mcn,
            "h_pin": h_pin,
            "h_mcn": h_mcn,
            "tau_pin": tau_pin,
            "tau_mcn": tau_mcn,
        }

        for key in accum:
            accum[key].append(result[key])

        if worst is None or result["tau"] > worst["result"]["tau"]:
            worst = {
                "index": i,
                "q": q,
                "dq": dq,
                "qdd": qdd,
                "result": result,
            }

    print(f"\n=== {name} vs Pinocchio ===")
    print("Max absolute error summary:")
    for key in ("M", "G", "Cqd", "h", "tau"):
        values = np.asarray(accum[key], dtype=float)
        print(
            f"  {key:4s}: max={values.max():.6e}, "
            f"mean={values.mean():.6e}, median={np.median(values):.6e}"
        )

    print("Worst tau case:")
    print("  sample =", worst["index"])
    print("  q   =", worst["q"])
    print("  dq  =", worst["dq"])
    print("  qdd =", worst["qdd"])
    print("  errors:")
    for key in ("M", "G", "Cqd", "h", "tau"):
        print(f"    {key:4s} = {worst['result'][key]:.6e}")

    return worst


def print_verbose_worst(name, worst):
    result = worst["result"]
    print(f"\n{name} worst-case vectors:")
    for term in ("G", "Cqd", "h", "tau"):
        print(f"  {term}_pin =", result[f"{term}_pin"])
        print(f"  {term}_mcn =", result[f"{term}_mcn"])
        print(f"  {term}_err =", result[f"{term}_mcn"] - result[f"{term}_pin"])


def build_states(args):
    if args.q is not None:
        q = args.q
        dq = args.dq if args.dq is not None else np.zeros(6)
        qdd = args.qdd if args.qdd is not None else np.zeros(6)
        return [(q, dq, qdd)]

    rng = np.random.default_rng(args.seed)
    q_low = JOINT_LOWER + 0.05
    q_high = JOINT_UPPER - 0.05

    states = []
    for _ in range(args.samples):
        q = rng.uniform(q_low, q_high)
        dq = rng.uniform(-1.0, 1.0, size=6)
        qdd = rng.uniform(-2.0, 2.0, size=6)
        states.append((q, dq, qdd))

    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50, help="number of random states")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--q", type=parse_six_floats, default=None, help="single q to test")
    parser.add_argument("--dq", type=parse_six_floats, default=None, help="single dq to test")
    parser.add_argument("--qdd", type=parse_six_floats, default=None, help="single qdd to test")
    parser.add_argument("--verbose", action="store_true", help="print worst-case vectors")
    args = parser.parse_args()

    model, data = load_pinocchio_model()
    params = SCARA_initialize()
    states = build_states(args)

    print("[MCN] beta =", params["beta"])
    print("[MCN] Python backend = MCN_CT.MCNsim")
    print("[MCN] C++ backend    =", "mcn_dynamics.MCNsim" if mcn_dynamics is not None else "not available")
    print("[MCN] Compare convention:")
    print("  M_mcn        vs crba(q)")
    print("  G_mcn        vs computeGeneralizedGravity(q)")
    print("  C_mcn @ dq   vs nonLinearEffects(q,dq) - G_pin")
    print("  h_mcn        vs nonLinearEffects(q,dq)")
    print("  tau_mcn      vs M_pin @ qdd + h_pin")

    worst_cases = []

    if mcn_dynamics is not None:
        worst_cases.append(
            (
                "MCN_cpp",
                compare_backend(
                    "MCN_cpp",
                    mcn_dynamics.MCNsim,
                    model,
                    data,
                    params,
                    states,
                ),
            )
        )
    else:
        print("\n[WARNING] C++ backend not available; skipping MCN_cpp comparison.")

    worst_cases.append(
        (
            "MCN_python",
            compare_backend(
                "MCN_python",
                MCNsim,
                model,
                data,
                params,
                states,
            ),
        )
    )

    if args.verbose:
        for name, worst in worst_cases:
            print_verbose_worst(name, worst)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real Z1 MCN computed-torque controller through UnitreeArm LowCmd.

This is the real-robot counterpart of MCN_CT_lowcmd_gazebo.py:
    - Does not publish to Gazebo effort_controller topics.
    - Connects to the real z1_controller with UnitreeArm.
    - Sends computed torque through LowCmd tau_d.
    - LowCmd Kp = 0 and Kd = 0, matching pure torque control.
    - Reuses the MCN model and reference helpers from MCN_CT.py.

Run only after z1_controller is running and the arm is ready for LowCmd.

Typical real-robot startup:
    cd ~/z1_controller/build
    ./z1_ctrl

Then run this script in another terminal.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rospkg

from MCN_CT import (
    SCARA_initialize,
    clamp_q,
    compute_lagrange_terms_mcn,
    reference_trajectory_go_and_return,
)

z1_sdk_path = rospkg.RosPack().get_path("z1_sdk")
sys.path.append(os.path.join(z1_sdk_path, "lib"))

import unitree_arm_interface

np.set_printoptions(precision=4, suppress=True)


def parse_six_floats(text):
    values = [float(x) for x in text.replace(",", " ").split()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError("expected exactly 6 numbers")
    return np.array(values, dtype=float)


def compute_metrics(t_log, q_ref_log, q_log, tau_log, move_start_2):
    err = q_ref_log - q_log
    abs_err = np.abs(err)

    metrics = {
        "rms_error_all": float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error_all": float(np.max(abs_err)),
        "final_max_abs_error": float(np.max(abs_err[-1])),
        "rms_error_by_joint": np.sqrt(np.mean(err ** 2, axis=0)).tolist(),
        "max_abs_error_by_joint": np.max(abs_err, axis=0).tolist(),
    }

    hold_goal_mask = (t_log >= move_start_2 - 2.0) & (t_log <= move_start_2 - 0.2)
    final_hold_mask = t_log >= (t_log[-1] - 1.5)

    for name, mask in (("goal_hold", hold_goal_mask), ("final_hold", final_hold_mask)):
        if np.count_nonzero(mask) < 5:
            continue
        err_hold = err[mask]
        q_hold = q_log[mask]
        tau_hold = tau_log[mask]
        tau_diff = np.diff(tau_hold, axis=0)

        metrics[f"{name}_mean_abs_error_all"] = float(np.mean(np.abs(err_hold)))
        metrics[f"{name}_error_std_all"] = float(np.mean(np.std(err_hold, axis=0)))
        metrics[f"{name}_q_vibration_all"] = float(np.mean(np.std(q_hold, axis=0)))
        metrics[f"{name}_tau_diff_rms_all"] = float(np.sqrt(np.mean(tau_diff ** 2))) if tau_diff.size else 0.0
        metrics[f"{name}_mean_abs_error_by_joint"] = np.mean(np.abs(err_hold), axis=0).tolist()
        metrics[f"{name}_q_vibration_by_joint"] = np.std(q_hold, axis=0).tolist()

    tau_diff_all = np.diff(tau_log, axis=0)
    metrics["tau_diff_rms_all"] = float(np.sqrt(np.mean(tau_diff_all ** 2))) if tau_diff_all.size else 0.0

    return metrics


class Z1RealLowCmdTorqueInterface:
    def __init__(self, torque_scale=1.0):
        self.torque_scale = float(torque_scale)

        print("[Z1 Real] Connecting to z1_controller through LowCmd interface.")
        if not hasattr(unitree_arm_interface, "ArmInterface"):
            names = ", ".join(name for name in dir(unitree_arm_interface) if not name.startswith("_"))
            raise AttributeError(
                "unitree_arm_interface does not expose ArmInterface. "
                f"Available names: {names}"
            )

        self.z1 = unitree_arm_interface.ArmInterface(hasGripper=True)
        self.enter_lowcmd_arm_interface()
        time.sleep(0.8)

        print("[Z1 Real] backend = ArmInterface, matching example_lowcmd.py")

        self.q = np.zeros(6)
        self.dq = np.zeros(6)
        self.dq_source = None
        self.last_t = time.perf_counter()
        self.update_feedback(force=True)

        print("[Z1 Real] Connected.")
        print("[Z1 Real] q0 =", self.q)

    def enter_lowcmd_arm_interface(self):
        if hasattr(self.z1, "setFsmLowcmd"):
            self.z1.setFsmLowcmd()
            return

        if hasattr(unitree_arm_interface, "ArmFSMState") and hasattr(self.z1, "setFsm"):
            for state_name in ("LOWCMD", "LowCmd", "lowcmd"):
                if hasattr(unitree_arm_interface.ArmFSMState, state_name):
                    self.z1.setFsm(getattr(unitree_arm_interface.ArmFSMState, state_name))
                    break

        if hasattr(self.z1, "setControlGain"):
            self.z1.setControlGain(np.zeros(6), np.zeros(6))
        elif hasattr(self.z1, "setCtrlGain"):
            self.z1.setCtrlGain(np.zeros(6), np.zeros(6))

    def get_dt(self):
        if hasattr(self.z1, "dt"):
            return float(self.z1.dt)
        if hasattr(self.z1, "_ctrlComp") and hasattr(self.z1._ctrlComp, "dt"):
            return float(self.z1._ctrlComp.dt)
        return 0.002

    def read_raw_q(self):
        if hasattr(self.z1, "lowstate") and hasattr(self.z1.lowstate, "getQ"):
            try:
                q = np.asarray(self.z1.lowstate.getQ(), dtype=float).reshape(-1)
                if q.size >= 6:
                    return q[:6].copy()
            except Exception:
                pass
        for name in ("q", "theta", "q_d"):
            if hasattr(self.z1, name):
                try:
                    q = np.asarray(getattr(self.z1, name), dtype=float).reshape(-1)
                    if q.size >= 6:
                        return q[:6].copy()
                except Exception:
                    pass
        return self.q.copy()

    def read_raw_dq(self):
        if hasattr(self.z1, "lowstate"):
            for getter in ("getQd", "getDQ", "getDq", "getQdot"):
                if hasattr(self.z1.lowstate, getter):
                    try:
                        dq = np.asarray(getattr(self.z1.lowstate, getter)(), dtype=float).reshape(-1)
                        if dq.size >= 6:
                            self.dq_source = f"lowstate.{getter}()"
                            return dq[:6].copy()
                    except Exception:
                        pass
        for name in ("dq", "qd", "qdot", "dtheta"):
            if hasattr(self.z1, name):
                try:
                    dq = np.asarray(getattr(self.z1, name), dtype=float).reshape(-1)
                    if dq.size >= 6:
                        self.dq_source = f"arm.{name}"
                        return dq[:6].copy()
                except Exception:
                    pass
        self.dq_source = None
        return None

    def update_feedback(self, force=False):
        now = time.perf_counter()
        dt = now - self.last_t
        if (not force) and dt <= 1e-6:
            return

        q_new = self.read_raw_q()
        dq_direct = self.read_raw_dq()
        if dq_direct is not None:
            dq_new = dq_direct
        elif dt > 1e-6:
            dq_new = (q_new - self.q) / dt
        else:
            dq_new = self.dq.copy()

        self.q = q_new.copy()
        self.dq = dq_new.copy()
        self.last_t = now

    def send_tau(self, tau):
        tau = np.asarray(tau, dtype=float).reshape(6)

        q_d = self.q.copy()
        dq_d = np.zeros(6)
        tau_d = self.torque_scale * tau

        if not hasattr(self.z1, "setArmCmd"):
            names = ", ".join(name for name in dir(self.z1) if not name.startswith("_"))
            raise AttributeError(
                "ArmInterface backend does not expose setArmCmd(q, dq, tau). "
                f"Available ArmInterface names: {names}"
            )

        self.z1.setArmCmd(q_d, dq_d, tau_d)
        self.z1.sendRecv()
        self.update_feedback(force=True)

    def zero_torque(self, duration=0.2):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            self.send_tau(np.zeros(6))
            time.sleep(self.get_dt())

    def passive(self):
        try:
            self.send_tau(np.zeros(6))
            if hasattr(self.z1, "loopOff"):
                self.z1.loopOff()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=18.0, help="total control time")
    parser.add_argument("--dt", type=float, default=0.005, help="control period")
    parser.add_argument("--move-duration", type=float, default=5.5, help="reference movement duration")
    parser.add_argument("--hold-duration", type=float, default=3.0, help="hold time at q_goal before returning")
    parser.add_argument("--final-hold", type=float, default=3.0, help="hold time at q0 after returning")
    parser.add_argument("--torque-scale", type=float, default=1.0, help="scale applied to tau_d")
    parser.add_argument("--target-scale", type=float, default=1.0, help="scale from q0 toward nominal q_goal")
    parser.add_argument("--wn", type=parse_six_floats, default=None, help="6 natural frequencies")
    parser.add_argument("--eta", type=parse_six_floats, default=None, help="6 damping ratios")
    parser.add_argument("--beta", type=parse_six_floats, default=None, help="6 viscous friction compensation gains: tau += beta * dq_ref")
    parser.add_argument("--friction", type=parse_six_floats, default=None, help="6 Coulomb friction compensation gains: tau += friction * tanh(dq_ref / friction_eps)")
    parser.add_argument("--friction-eps", type=float, default=0.03, help="velocity smoothing value for Coulomb friction compensation")
    parser.add_argument("--tau-limit", type=parse_six_floats, default=None, help="6 torque clamp values")
    parser.add_argument("--save-dir", type=str, default="mcn_ct_lowcmd_real_logs", help="directory for npz/csv/json logs")
    parser.add_argument("--tag", type=str, default="run", help="log filename tag")
    parser.add_argument("--no-plot", action="store_true", help="skip matplotlib plots")
    parser.add_argument("--check-dq", action="store_true", help="only print q/dq feedback; do not run trajectory control")
    parser.add_argument("--check-dq-duration", type=float, default=5.0, help="duration for --check-dq")
    args = parser.parse_args()

    print("===================================================")
    print("Z1 REAL MCN Computed Torque via LowCmd tau_d only")
    print("LowCmd Kp = 0, Kd = 0.")
    print("No Gazebo effort_controller topics are used.")
    print("Requires z1_controller running first, usually: cd ~/z1_controller/build && ./z1_ctrl")
    print("===================================================")

    params = SCARA_initialize()

    # Keep the same computed-torque gains as MCN_CT_lowcmd_gazebo.py.
    preset_wn = np.array([22.0, 46.0, 55.0, 150.0, 140.0, 540.0], dtype=float)
    preset_eta = np.array([1.0, 1.35, 0.9, 1.05, 1.4, 1.25], dtype=float)
    preset_beta = np.array([0.5, 0.8, 0.8, 0.2, 0.5, 0.4], dtype=float)
    preset_friction = np.array([0.4, 0.8, 0.5, 0.2, 0.4, 0.3], dtype=float)
    preset_tau_limit = np.array([45, 60, 45, 50, 50, 50], dtype=float)

    robot = Z1RealLowCmdTorqueInterface(
        torque_scale=args.torque_scale,
    )

    if args.check_dq:
        print("[DQ Check] Reading feedback only. No trajectory command will be run.")
        print("[DQ Check] Move joints gently by hand only if the arm state allows it safely.")
        print("[DQ Check] dq source =", robot.dq_source)
        t0_check = time.perf_counter()
        last_print = 0.0
        samples = []
        try:
            while time.perf_counter() - t0_check < args.check_dq_duration:
                robot.update_feedback(force=True)
                t_check = time.perf_counter() - t0_check
                samples.append(robot.dq.copy())
                if t_check - last_print >= 0.2:
                    print(
                        f"[DQ Check] t={t_check:6.3f} "
                        f"source={robot.dq_source} "
                        f"q={robot.q} dq={robot.dq}"
                    )
                    last_print = t_check
                time.sleep(robot.get_dt())
        finally:
            robot.zero_torque(duration=0.1)
            robot.passive()

        samples = np.asarray(samples)
        if samples.size > 0:
            print("[DQ Check] dq min =", samples.min(axis=0))
            print("[DQ Check] dq max =", samples.max(axis=0))
            print("[DQ Check] dq std =", samples.std(axis=0))
        return

    q0 = clamp_q(robot.q)
    nominal_goal = clamp_q(np.array([0.60, 0.60, -0.60, -0.35, 0.80, 0.80], dtype=float))
    q_goal = clamp_q(q0 + args.target_scale * (nominal_goal - q0))

    wn = args.wn if args.wn is not None else preset_wn
    eta = args.eta if args.eta is not None else preset_eta
    Kp = np.diag(wn ** 2)
    Kd = np.diag(2.0 * eta * wn)

    beta = args.beta if args.beta is not None else preset_beta
    friction = args.friction if args.friction is not None else preset_friction
    friction_eps = max(float(args.friction_eps), 1e-6)
    tau_limit = args.tau_limit if args.tau_limit is not None else preset_tau_limit

    print("[MCN] Model initialized.")
    print("[MCN] xi shape    =", params["xi"].shape)
    print("[MCN] calMp shape =", params["calMp"].shape)
    print("[Reference] q0           =", q0)
    print("[Reference] nominal_goal =", nominal_goal)
    print("[Reference] target_scale =", args.target_scale)
    print("[Reference] q_goal       =", q_goal)
    print("[Outer gains]")
    print("  wn =", wn)
    print("  eta =", eta)
    print("  beta =", beta)
    print("  friction =", friction)
    print("  friction_eps =", friction_eps)
    print("  Kp diag =", np.diag(Kp))
    print("  Kd diag =", np.diag(Kd))
    print("[Torque]")
    print("  tau_limit =", tau_limit)
    print("  torque_scale =", args.torque_scale)

    move_start_1 = 1.0
    move_duration_1 = args.move_duration
    hold_duration = args.hold_duration
    move_duration_2 = args.move_duration
    move_start_2 = move_start_1 + move_duration_1 + hold_duration
    total_time_needed = move_start_2 + move_duration_2 + args.final_hold
    if args.T < total_time_needed:
        print("[Time] Automatically set T from", args.T, "to", total_time_needed)
        args.T = total_time_needed

    t_log = []
    q_ref_log = []
    q_log = []
    dq_ref_log = []
    dq_log = []
    tau_log = []
    tau_raw_log = []
    h_log = []
    n_log = []
    mcn_dt_log = []
    loop_dt_log = []

    print("[Control] Start real LowCmd MCN computed torque control.")
    t_start = time.perf_counter()
    next_time = t_start
    interrupted = False

    try:
        while True:
            loop_start = time.perf_counter()
            t = time.perf_counter() - t_start
            if t > args.T:
                break

            q = clamp_q(robot.q)
            dq = robot.dq.copy()
            q_ref, dq_ref, ddq_ref = reference_trajectory_go_and_return(
                t,
                q0,
                q_goal,
                move_start_1,
                move_duration_1,
                hold_duration,
                move_duration_2,
            )

            e = q_ref - q
            de = dq_ref - dq
            qdd_cmd = ddq_ref + Kd @ de + Kp @ e

            mcn_t0 = time.perf_counter()
            M, h, _, N = compute_lagrange_terms_mcn(params, q, dq)
            mcn_dt_log.append(time.perf_counter() - mcn_t0)

            tau_raw = M @ qdd_cmd + h
            tau_raw = tau_raw + beta * dq_ref
            tau_raw = tau_raw + friction * np.tanh(dq_ref / friction_eps)
            tau = np.clip(tau_raw, -tau_limit, tau_limit)

            robot.send_tau(tau)

            t_log.append(t)
            q_ref_log.append(q_ref.copy())
            q_log.append(q.copy())
            dq_ref_log.append(dq_ref.copy())
            dq_log.append(dq.copy())
            tau_log.append(tau.copy())
            tau_raw_log.append(tau_raw.copy())
            h_log.append(h.copy())
            n_log.append(N.copy())

            next_time += args.dt
            remain = next_time - time.perf_counter()
            if remain > 0:
                time.sleep(remain)
            loop_dt_log.append(time.perf_counter() - loop_start)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[Control] Interrupted by user.")
    finally:
        print("[Control] Sending zero torque, then Passive.")
        robot.zero_torque(duration=0.3)
        robot.passive()

    t_log = np.asarray(t_log)
    q_ref_log = np.asarray(q_ref_log)
    q_log = np.asarray(q_log)
    dq_ref_log = np.asarray(dq_ref_log)
    dq_log = np.asarray(dq_log)
    tau_log = np.asarray(tau_log)
    tau_raw_log = np.asarray(tau_raw_log)
    h_log = np.asarray(h_log)
    n_log = np.asarray(n_log)
    mcn_dt_log = np.asarray(mcn_dt_log)
    loop_dt_log = np.asarray(loop_dt_log)

    print("[Control] Stopped before completing trajectory." if interrupted else "[Control] Finished go-and-return trajectory.")

    if loop_dt_log.size > 0:
        overrun = loop_dt_log > args.dt * 1.05
        print("\n[Timing]")
        print(f"  requested dt       = {args.dt:.6f} s ({1.0 / args.dt:.1f} Hz)")
        print(f"  actual loop dt avg = {loop_dt_log.mean():.6f} s ({1.0 / loop_dt_log.mean():.1f} Hz)")
        print(f"  actual loop dt max = {loop_dt_log.max():.6f} s")
        print(f"  MCN calc dt avg    = {mcn_dt_log.mean():.6f} s")
        print(f"  MCN calc dt max    = {mcn_dt_log.max():.6f} s")
        print(f"  overrun fraction   = {100.0 * overrun.mean():.2f}%")

    if t_log.size == 0:
        return

    final_error = q_ref_log[-1] - q_log[-1]
    print("\nFinal tracking error:")
    for i in range(6):
        print(f"Joint {i + 1}: error = {final_error[i]: .5f} rad")
    print("Max abs error =", np.max(np.abs(final_error)))

    metrics = compute_metrics(
        t_log=t_log,
        q_ref_log=q_ref_log,
        q_log=q_log,
        tau_log=tau_log,
        move_start_2=move_start_2,
    )

    print("\n[Metrics]")
    print(f"  rms_error_all          = {metrics['rms_error_all']:.6f}")
    print(f"  max_abs_error_all      = {metrics['max_abs_error_all']:.6f}")
    print(f"  final_max_abs_error    = {metrics['final_max_abs_error']:.6f}")
    if "goal_hold_q_vibration_all" in metrics:
        print(f"  goal_hold_q_vibration = {metrics['goal_hold_q_vibration_all']:.6f}")
    if "final_hold_q_vibration_all" in metrics:
        print(f"  final_hold_q_vibration = {metrics['final_hold_q_vibration_all']:.6f}")
    print(f"  tau_diff_rms_all       = {metrics['tau_diff_rms_all']:.6f}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        tag = args.tag

        np.savez(
            save_dir / f"{tag}_logs.npz",
            t=t_log,
            reference_q=q_ref_log,
            tracking_q=q_log,
            tau=tau_log,
            reference_dq=dq_ref_log,
            tracking_dq=dq_log,
            q_ref=q_ref_log,
            q=q_log,
            dq=dq_log,
            tau_raw=tau_raw_log,
            h=h_log,
            N=n_log,
        )

        csv_header = (
            ["t"]
            + [f"reference_q_{i + 1}" for i in range(6)]
            + [f"tracking_q_{i + 1}" for i in range(6)]
            + [f"tau_{i + 1}" for i in range(6)]
            + [f"reference_dq_{i + 1}" for i in range(6)]
            + [f"tracking_dq_{i + 1}" for i in range(6)]
        )
        csv_data = np.column_stack(
            [
                t_log,
                q_ref_log,
                q_log,
                tau_log,
                dq_ref_log,
                dq_log,
            ]
        )
        np.savetxt(
            save_dir / f"{tag}_tracking_export.csv",
            csv_data,
            delimiter=",",
            header=",".join(csv_header),
            comments="",
        )

        summary = {
            "tag": tag,
            "wn": wn.tolist(),
            "eta": eta.tolist(),
            "beta": beta.tolist(),
            "friction": friction.tolist(),
            "friction_eps": friction_eps,
            "tau_limit": tau_limit.tolist(),
            "target_scale": args.target_scale,
            "metrics": metrics,
        }
        with (save_dir / f"{tag}_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print("[Log] saved to", save_dir)

    if args.no_plot:
        return

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        axes[i].plot(t_log, q_ref_log[:, i], "r--", linewidth=2, label=r"Reference $q_d$")
        axes[i].plot(t_log, q_log[:, i], "b", linewidth=1.5, label=r"Measured $q$")
        axes[i].set_ylabel(f"J{i + 1} rad")
        axes[i].grid(True)
        axes[i].legend(loc="best")
        axes[i].set_title(f"Joint {i + 1} Real LowCmd MCN CT Tracking")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Z1 Real LowCmd MCN CT: q0 -> q_goal -> q0")
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
    for i in range(6):
        axes2[i].plot(t_log, tau_log[:, i], linewidth=1.5)
        axes2[i].set_ylabel(f"tau{i + 1}")
        axes2[i].grid(True)

    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Real LowCmd MCN Computed Torque Command")
    plt.tight_layout()
    plt.show()

    fig3, axes3 = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
    for i in range(6):
        axes3[i].plot(t_log, tau_raw_log[:, i], "r--", linewidth=1.0, label="raw tau")
        axes3[i].plot(t_log, tau_log[:, i], "b", linewidth=1.2, label="sent tau")
        axes3[i].set_ylabel(f"tau{i + 1}")
        axes3[i].grid(True)
        axes3[i].legend(loc="best")

    axes3[-1].set_xlabel("Time (s)")
    fig3.suptitle("Raw Tau vs Sent Real LowCmd Tau")
    plt.tight_layout()
    plt.show()

    fig4, axes4 = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
    for i in range(6):
        axes4[i].plot(t_log, dq_ref_log[:, i], "r--", linewidth=1.2, label=r"Reference $\dot q_d$")
        axes4[i].plot(t_log, dq_log[:, i], "b", linewidth=1.2, label=r"Measured $\dot q$")
        axes4[i].set_ylabel(f"dq{i + 1}")
        axes4[i].grid(True)
        axes4[i].legend(loc="best")

    axes4[-1].set_xlabel("Time (s)")
    fig4.suptitle("Measured Joint Velocity Feedback dq")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

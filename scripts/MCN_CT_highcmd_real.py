#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real Z1 HighCmd joint trajectory tracker.

This is the HighCmd counterpart of MCN_CT_lowcmd_real.py:
    - Uses ArmInterface HighCmd API.
    - Sends per-cycle joint velocity commands through jointCtrlCmd().
    - Does not send LowCmd torque; no MCN torque is computed.
    - Reuses the same go-and-return reference helper from MCN_CT.py.

Run only after z1_controller is running and the arm is ready.

Typical real-robot startup:
    cd ~/z1_controller/build
    ./z1_ctrl

Then run this script in another terminal.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rospkg

from MCN_CT import (
    clamp_q,
    reference_trajectory_go_and_return,
)

z1_sdk_path = rospkg.RosPack().get_path("z1_sdk")
script_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib"))
sys.path.insert(0, os.path.join(z1_sdk_path, "lib"))
sys.path.insert(0, script_lib_path)

import unitree_arm_interface

np.set_printoptions(precision=4, suppress=True)


def read_state_vector(arm, names):
    if hasattr(arm, "lowstate"):
        getter_groups = {
            "q": ("getQ", "getQRaw", "getTheta"),
            "theta": ("getQ", "getQRaw", "getTheta"),
            "q_d": ("getQ", "getQRaw", "getTheta"),
            "dq": ("getQd", "getDQ", "getDq", "getQdot"),
            "dtheta": ("getQd", "getDQ", "getDq", "getQdot"),
            "qdot": ("getQd", "getDQ", "getDq", "getQdot"),
            "qd": ("getQd", "getDQ", "getDq", "getQdot"),
            "dq_d": ("getQd", "getDQ", "getDq", "getQdot"),
        }
        for name in names:
            for getter in getter_groups.get(name, ()):
                if hasattr(arm.lowstate, getter):
                    try:
                        value = np.asarray(getattr(arm.lowstate, getter)(), dtype=float).reshape(-1)
                        if value.size >= 6:
                            return value[:6].copy()
                    except Exception:
                        pass

    if hasattr(arm, "armState"):
        for name in names:
            if hasattr(arm.armState, name):
                try:
                    value = np.asarray(getattr(arm.armState, name), dtype=float).reshape(-1)
                    if value.size >= 6:
                        return value[:6].copy()
                except Exception:
                    pass

    for name in names:
        if hasattr(arm, name):
            try:
                value = np.asarray(getattr(arm, name), dtype=float).reshape(-1)
                if value.size >= 6:
                    return value[:6].copy()
            except Exception:
                pass

    return None


def compute_metrics(t_log, q_ref_log, q_log, move_start_2):
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

        metrics[f"{name}_mean_abs_error_all"] = float(np.mean(np.abs(err_hold)))
        metrics[f"{name}_error_std_all"] = float(np.mean(np.std(err_hold, axis=0)))
        metrics[f"{name}_q_vibration_all"] = float(np.mean(np.std(q_hold, axis=0)))
        metrics[f"{name}_mean_abs_error_by_joint"] = np.mean(np.abs(err_hold), axis=0).tolist()
        metrics[f"{name}_q_vibration_by_joint"] = np.std(q_hold, axis=0).tolist()

    return metrics


class Z1RealHighCmdInterface:
    def __init__(self):
        print("[Z1 Real] Connecting through ArmInterface HighCmd.")

        if not hasattr(unitree_arm_interface, "ArmInterface"):
            names = ", ".join(name for name in dir(unitree_arm_interface) if not name.startswith("_"))
            raise AttributeError(
                "unitree_arm_interface does not expose ArmInterface. "
                f"Available names: {names}"
            )
        if not hasattr(unitree_arm_interface, "ArmFSMState"):
            names = ", ".join(name for name in dir(unitree_arm_interface) if not name.startswith("_"))
            raise AttributeError(
                "unitree_arm_interface does not expose ArmFSMState. "
                f"Available names: {names}"
            )

        arm_state = unitree_arm_interface.ArmFSMState
        if not hasattr(arm_state, "JOINTCTRL"):
            names = ", ".join(name for name in dir(arm_state) if not name.startswith("_"))
            raise AttributeError(
                "ArmFSMState does not expose JOINTCTRL. "
                f"Available states: {names}"
            )

        self.z1 = unitree_arm_interface.ArmInterface(hasGripper=True)
        missing_methods = [
            name
            for name in ("loopOn", "startTrack", "jointCtrlCmd", "loopOff")
            if not hasattr(self.z1, name)
        ]
        if missing_methods:
            names = ", ".join(name for name in dir(self.z1) if not name.startswith("_"))
            raise AttributeError(
                "ArmInterface backend is missing methods required for HighCmd joint control: "
                f"{missing_methods}. Available ArmInterface names: {names}"
            )
        self.joint_speed_limit = 1.0
        self.tracking_kp = np.full(6, 2.0, dtype=float)
        self.q = np.zeros(6)
        self.dq = np.zeros(6)
        self.last_q = None
        self.last_t = time.perf_counter()
        self.dq_source = None

        self.z1.loopOn()
        self.z1.startTrack(arm_state.JOINTCTRL)
        time.sleep(0.3)
        self.update_feedback(force=True)

        print("[Z1 Real] backend = ArmInterface HighCmd JOINTCTRL")
        print("[Z1 Real] Connected.")
        print("[Z1 Real] q0 =", self.q)

    def get_dt(self):
        if hasattr(self.z1, "dt"):
            return float(self.z1.dt)
        if hasattr(self.z1, "_ctrlComp") and hasattr(self.z1._ctrlComp, "dt"):
            return float(self.z1._ctrlComp.dt)
        return 0.002

    def read_q(self):
        q = read_state_vector(self.z1, ("q", "theta", "q_d"))
        if q is not None:
            return q
        return self.q.copy()

    def read_dq(self):
        dq = read_state_vector(self.z1, ("dq", "dtheta", "qdot", "qd", "dq_d"))
        if dq is not None:
            self.dq_source = "direct"
            return dq
        self.dq_source = "finite_difference"
        return None

    def update_feedback(self, force=False):
        now = time.perf_counter()
        dt = now - self.last_t
        if (not force) and dt <= 1e-6:
            return

        q_now = self.read_q()
        dq_direct = self.read_dq()
        if dq_direct is not None:
            dq_now = dq_direct
        elif self.last_q is not None and dt > 1e-6:
            dq_now = (q_now - self.last_q) / dt
        else:
            dq_now = self.dq.copy()

        self.q = q_now.copy()
        self.dq = dq_now.copy()
        self.last_q = q_now.copy()
        self.last_t = now

    def poll_feedback(self):
        self.update_feedback(force=True)

    def send_joint_speed(self, dq_cmd):
        dq_cmd = np.asarray(dq_cmd, dtype=float).reshape(6)
        dq_cmd = np.clip(dq_cmd, -self.joint_speed_limit, self.joint_speed_limit)
        max_abs_speed = float(np.max(np.abs(dq_cmd)))

        if max_abs_speed < 1e-8:
            direction = np.zeros(6)
            speed = self.joint_speed_limit
        else:
            direction = dq_cmd / max_abs_speed
            speed = max_abs_speed

        command = np.concatenate([direction, [0.0]])
        self.z1.jointCtrlCmd(command.tolist(), speed)

    def send_highcmd(self, q_ref, dq_ref):
        q_ref = np.asarray(q_ref, dtype=float).reshape(6)
        dq_ref = np.asarray(dq_ref, dtype=float).reshape(6)

        self.update_feedback(force=True)
        q_error = q_ref - self.q
        dq_cmd = dq_ref + self.tracking_kp * q_error

        self.send_joint_speed(dq_cmd)
        self.update_feedback(force=True)

    def hold_current_then_passive(self, duration=0.3):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < duration:
            self.send_joint_speed(np.zeros(6))
            self.update_feedback(force=True)
            time.sleep(self.get_dt())

        self.passive()

    def passive(self):
        try:
            self.send_joint_speed(np.zeros(6))
        except Exception:
            pass
        try:
            self.z1.loopOff()
        except Exception:
            pass


def save_q_response_csv(save_dir, tag, t_log, q_log):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_header = ["t"] + [f"q_{i + 1}" for i in range(6)]
    csv_data = np.column_stack([t_log, q_log])
    csv_path = save_dir / f"{tag}_q_response.csv"
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header=",".join(csv_header),
        comments="",
        fmt="%.9f",
    )

    print("[CSV] q response saved to:", csv_path)


def plot_logs(t_log, q_ref_log, q_log, dq_ref_log, dq_log):
    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for i in range(6):
        axes[i].plot(t_log, q_ref_log[:, i], "r--", linewidth=2, label=r"Reference $q_d$")
        axes[i].plot(t_log, q_log[:, i], "b", linewidth=1.5, label=r"Measured $q$")
        axes[i].set_ylabel(f"J{i + 1} rad")
        axes[i].grid(True)
        axes[i].legend(loc="best")
        axes[i].set_title(f"Joint {i + 1} Real ArmInterface HighCmd Tracking")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Z1 Real ArmInterface HighCmd: q0 -> q_goal -> q0")
    plt.tight_layout()
    plt.show()

    fig2, axes2 = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
    for i in range(6):
        axes2[i].plot(t_log, dq_ref_log[:, i], "r--", linewidth=1.2, label=r"Reference $\dot q_d$")
        axes2[i].plot(t_log, dq_log[:, i], "b", linewidth=1.2, label=r"Measured $\dot q$")
        axes2[i].set_ylabel(f"dq{i + 1}")
        axes2[i].grid(True)
        axes2[i].legend(loc="best")

    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Z1 Real ArmInterface HighCmd Joint Velocity Feedback")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=float, default=18.0, help="total control time")
    parser.add_argument("--dt", type=float, default=0.005, help="control period")
    parser.add_argument("--move-duration", type=float, default=5.5, help="reference movement duration")
    parser.add_argument("--hold-duration", type=float, default=3.0, help="hold time at q_goal before returning")
    parser.add_argument("--final-hold", type=float, default=3.0, help="hold time at q0 after returning")
    parser.add_argument("--target-scale", type=float, default=1.0, help="scale from q0 toward nominal q_goal")
    parser.add_argument("--save-dir", type=str, default="mcn_ct_highcmd_real_logs", help="directory for q response CSV")
    parser.add_argument("--tag", type=str, default="run", help="log filename tag")
    parser.add_argument("--no-plot", action="store_true", help="skip matplotlib plots")
    parser.add_argument("--check-dq", action="store_true", help="only print q/dq feedback; do not run trajectory control")
    parser.add_argument("--check-dq-duration", type=float, default=5.0, help="duration for --check-dq")
    args = parser.parse_args()

    print("===================================================")
    print("Z1 REAL HighCmd ArmInterface jointCtrlCmd tracking")
    print("Reference trajectory is the same q0 -> q_goal -> q0 plan as MCN_CT_lowcmd_real.py.")
    print("Interface path: ArmInterface loopOn -> startTrack(JOINTCTRL) -> jointCtrlCmd.")
    print("No LowCmd torque control. No UnitreeArm armCmd is used.")
    print("Requires z1_controller running first, usually: cd ~/z1_controller/build && ./z1_ctrl")
    print("===================================================")

    robot = Z1RealHighCmdInterface()

    if args.check_dq:
        print("[DQ Check] Reading feedback only. No trajectory command will be run.")
        t0_check = time.perf_counter()
        last_print = 0.0
        samples = []
        try:
            while time.perf_counter() - t0_check < args.check_dq_duration:
                robot.poll_feedback()
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

    print("[Reference] q0           =", q0)
    print("[Reference] nominal_goal =", nominal_goal)
    print("[Reference] target_scale =", args.target_scale)
    print("[Reference] q_goal       =", q_goal)

    move_start_1 = 1.0
    move_duration_1 = args.move_duration
    hold_duration = args.hold_duration
    move_duration_2 = args.move_duration
    move_start_2 = move_start_1 + move_duration_1 + hold_duration
    total_time_needed = move_start_2 + move_duration_2 + args.final_hold
    if args.T < total_time_needed:
        print("[Time] Automatically set T from", args.T, "to", total_time_needed)
        args.T = total_time_needed

    print("[Motion Plan]")
    print("  0.0 ~", move_start_1, ": hold q0")
    print(" ", move_start_1, "~", move_start_1 + move_duration_1, ": q0 -> q_goal")
    print(" ", move_start_1 + move_duration_1, "~", move_start_2, ": hold q_goal")
    print(" ", move_start_2, "~", move_start_2 + move_duration_2, ": q_goal -> q0")
    print(" ", move_start_2 + move_duration_2, "~", args.T, ": hold q0")

    t_log = []
    q_ref_log = []
    q_log = []
    dq_ref_log = []
    dq_log = []
    loop_dt_log = []

    print("[Control] Start real ArmInterface HighCmd reference tracking.")
    t_start = time.perf_counter()
    next_time = t_start
    interrupted = False

    try:
        while True:
            loop_start = time.perf_counter()
            t = time.perf_counter() - t_start
            if t > args.T:
                break

            q_ref, dq_ref, _ = reference_trajectory_go_and_return(
                t=t,
                q0=q0,
                q_goal=q_goal,
                move_start_1=move_start_1,
                move_duration_1=move_duration_1,
                hold_duration=hold_duration,
                move_duration_2=move_duration_2,
            )

            robot.send_highcmd(q_ref, dq_ref)

            t_log.append(t)
            q_ref_log.append(q_ref.copy())
            q_log.append(clamp_q(robot.q).copy())
            dq_ref_log.append(dq_ref.copy())
            dq_log.append(robot.dq.copy())

            next_time += args.dt
            remain = next_time - time.perf_counter()
            if remain > 0:
                time.sleep(remain)
            loop_dt_log.append(time.perf_counter() - loop_start)

    except KeyboardInterrupt:
        interrupted = True
        print("\n[Control] Interrupted by user.")
    finally:
        print("[Control] Send zero joint speed briefly, then stop ArmInterface loop.")
        robot.hold_current_then_passive(duration=0.3)

    t_log = np.asarray(t_log)
    q_ref_log = np.asarray(q_ref_log)
    q_log = np.asarray(q_log)
    dq_ref_log = np.asarray(dq_ref_log)
    dq_log = np.asarray(dq_log)
    loop_dt_log = np.asarray(loop_dt_log)

    print("[Control] Stopped before completing trajectory." if interrupted else "[Control] Finished go-and-return trajectory.")

    if loop_dt_log.size > 0:
        overrun = loop_dt_log > args.dt * 1.05
        print("\n[Timing]")
        print(f"  requested dt       = {args.dt:.6f} s ({1.0 / args.dt:.1f} Hz)")
        print(f"  actual loop dt avg = {loop_dt_log.mean():.6f} s ({1.0 / loop_dt_log.mean():.1f} Hz)")
        print(f"  actual loop dt max = {loop_dt_log.max():.6f} s")
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
        move_start_2=move_start_2,
    )

    print("\n[Metrics]")
    print(f"  rms_error_all       = {metrics['rms_error_all']:.6f}")
    print(f"  max_abs_error_all   = {metrics['max_abs_error_all']:.6f}")
    print(f"  final_max_abs_error = {metrics['final_max_abs_error']:.6f}")
    if "goal_hold_q_vibration_all" in metrics:
        print(f"  goal_hold_q_vibration  = {metrics['goal_hold_q_vibration_all']:.6f}")
    if "final_hold_q_vibration_all" in metrics:
        print(f"  final_hold_q_vibration = {metrics['final_hold_q_vibration_all']:.6f}")

    if args.save_dir is not None:
        save_q_response_csv(
            save_dir=args.save_dir,
            tag=args.tag,
            t_log=t_log,
            q_log=q_log,
        )

    if not args.no_plot:
        plot_logs(t_log, q_ref_log, q_log, dq_ref_log, dq_log)


if __name__ == "__main__":
    main()

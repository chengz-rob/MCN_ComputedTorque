#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
z1_pure_ct_effort_topic_MCN.py

Pure computed torque control for Unitree Z1 Gazebo using effort_controller topics.

This version:
    1. Does NOT use UnitreeArm.
    2. Does NOT use LowCmd.
    3. Does NOT send q_ref / dq_ref / Kp / Kd to motor.
    4. Reads q, dq from /z1_gazebo/joint_states.
    5. Computes dynamics using MCNsim:
           M(q), C(q,dq), N(q,dq)
       instead of directly calling Pinocchio.
    6. Computes:
           tau = M(q) @ qdd_cmd + C(q,dq) @ dq + N(q,dq)
    7. Publishes tau only to effort_controller topics.
    8. Supports q0 -> q_goal -> q0 motion.
"""

import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import rospy

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

try:
    import mcn_dynamics as mcn_dynamics_cpp
except ImportError:
    mcn_dynamics_cpp = None

np.set_printoptions(precision=4, suppress=True)


# ============================================================
# 1. Joint limits from Z1 URDF
# ============================================================

JOINT_LOWER = np.array([
    -2.6179938779914944,
     0.0,
    -2.8797932657906435,
    -1.5184364492350666,
    -1.3439035240356338,
    -2.792526803190927,
], dtype=float)

JOINT_UPPER = np.array([
     2.6179938779914944,
     2.9670597283903604,
     0.0,
     1.5184364492350666,
     1.3439035240356338,
     2.792526803190927,
], dtype=float)


def clamp_q(q, margin=0.02):
    q = np.asarray(q, dtype=float).reshape(6)
    return np.minimum(np.maximum(q, JOINT_LOWER + margin), JOINT_UPPER - margin)


# ============================================================
# 2. MCN modeling functions
#    Converted from your z1_functions_sensor_gazebo.py
# ============================================================

def hatm_num(omega):
    omega = np.asarray(omega, dtype=float).reshape(3)
    h_omega = np.zeros((3, 3), dtype=float)

    h_omega[0, 1] = -omega[2]
    h_omega[0, 2] =  omega[1]
    h_omega[1, 2] = -omega[0]
    h_omega[1, 0] =  omega[2]
    h_omega[2, 0] = -omega[1]
    h_omega[2, 1] =  omega[0]

    return h_omega


def rodrigues(homega, theta):
    homega = np.asarray(homega, dtype=float).reshape(3, 3)
    return np.eye(3) + homega * np.sin(theta) + homega @ homega * (1.0 - np.cos(theta))


def adjmg(g):
    g = np.asarray(g, dtype=float).reshape(4, 4)

    R = g[0:3, 0:3]
    p = g[0:3, 3]

    upper = np.hstack((R, hatm_num(p) @ R))
    lower = np.hstack((np.zeros((3, 3)), R))

    return np.vstack((upper, lower))


def adjmginv(g):
    g = np.asarray(g, dtype=float).reshape(4, 4)

    R = g[0:3, 0:3].T
    p = g[0:3, 3]

    upper = np.hstack((R, -R @ hatm_num(p)))
    lower = np.hstack((np.zeros((3, 3)), R))

    return np.vstack((upper, lower))


def gtwist_num(xi, theta):
    xi = np.asarray(xi, dtype=float).reshape(6)

    v = xi[0:3]
    omega = xi[3:6]

    nomega = np.linalg.norm(omega)

    if np.isclose(nomega, 0.0):
        if not np.isclose(np.linalg.norm(v), 1.0):
            raise ValueError(
                "Error in gtwist_num: pure translation requires norm(v)=1."
            )

        mat1 = np.eye(4)
        mat1[0:3, 3] = v * theta
        return mat1

    if not np.isclose(nomega, 1.0):
        raise ValueError("Error in gtwist_num: norm(omega) is not 1.")

    homega = hatm_num(omega)
    R = rodrigues(homega, theta)

    dum1 = (np.eye(3) - R) @ (homega @ v) + np.outer(omega, omega) @ v * theta

    mat1 = np.eye(4)
    mat1[0:3, 0:3] = R
    mat1[0:3, 3] = dum1

    return mat1


def lie_num(xi1, xi2):
    xi1 = np.asarray(xi1, dtype=float).reshape(6)
    xi2 = np.asarray(xi2, dtype=float).reshape(6)

    hxi1 = np.zeros((4, 4), dtype=float)
    hxi2 = np.zeros((4, 4), dtype=float)

    hxi1[0:3, 0:3] = hatm_num(xi1[3:6])
    hxi1[0:3, 3] = xi1[0:3]

    hxi2[0:3, 0:3] = hatm_num(xi2[3:6])
    hxi2[0:3, 3] = xi2[0:3]

    twistout = hxi1 @ hxi2 - hxi2 @ hxi1

    twistcout = np.array([
        twistout[0, 3],
        twistout[1, 3],
        twistout[2, 3],
        twistout[2, 1],
        twistout[0, 2],
        twistout[1, 0],
    ], dtype=float)

    return twistcout


def extract_com_from_spatial_inertia(calMp_i, m_i):
    calMp_i = np.asarray(calMp_i, dtype=float).reshape(6, 6)

    hmc = calMp_i[3:6, 0:3] / m_i

    p_com_home = np.array([
        hmc[2, 1],
        hmc[0, 2],
        hmc[1, 0],
    ], dtype=float)

    return p_com_home


def calc_gravity_by_potential(xi, calMp, g_prefix, mg):
    xi = np.asarray(xi, dtype=float)
    calMp = np.asarray(calMp, dtype=float)
    g_prefix = np.asarray(g_prefix, dtype=float)
    mg = np.asarray(mg, dtype=float).reshape(-1)

    n_dof = xi.shape[1]
    N_gravity = np.zeros(n_dof, dtype=float)

    ez = np.array([0.0, 0.0, 1.0])

    # 1. Recover each link COM position in the home pose.
    p_com_home = np.zeros((3, n_dof), dtype=float)

    for ll in range(n_dof):
        m_ll = np.trace(calMp[0:3, 0:3, ll]) / 3.0
        p_com_home[:, ll] = extract_com_from_spatial_inertia(
            calMp[:, :, ll],
            m_ll,
        )

    # 2. Current joint-axis directions and points.
    omega_cur = np.zeros((3, n_dof), dtype=float)
    q_cur = np.zeros((3, n_dof), dtype=float)

    for jj in range(n_dof):
        wj = xi[3:6, jj].copy()
        vj = xi[0:3, jj].copy()

        if jj == 0:
            g_prev = np.eye(4)
        else:
            g_prev = g_prefix[:, :, jj - 1]

        R_prev = g_prev[0:3, 0:3]
        p_prev = g_prev[0:3, 3]

        norm_wj = np.linalg.norm(wj)

        if np.isclose(norm_wj, 0.0):
            raise ValueError("Gravity helper assumes revolute joints.")

        wj = wj / norm_wj

        # xi = [-w x q; w]
        q_home = np.cross(wj, vj)

        omega_cur[:, jj] = R_prev @ wj
        q_cur[:, jj] = R_prev @ q_home + p_prev

    # 3. Gravity generalized force.
    for ll in range(n_dof):
        g_ll = g_prefix[:, :, ll]

        R_ll = g_ll[0:3, 0:3]
        p_ll = g_ll[0:3, 3]

        p_com_ll = R_ll @ p_com_home[:, ll] + p_ll

        for jj in range(ll + 1):
            Jv_col = np.cross(omega_cur[:, jj], p_com_ll - q_cur[:, jj])
            N_gravity[jj] += mg[ll] * (ez @ Jv_col)

    return N_gravity


def build_mcn_kinematics(xi, theta):
    xi = np.asarray(xi, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    n_dof = xi.shape[1]

    gg = np.zeros((4, 4, n_dof), dtype=float)

    for ii in range(n_dof):
        gg[:, :, ii] = gtwist_num(xi[:, ii], theta[ii])

    g_prefix = np.zeros((4, 4, n_dof), dtype=float)
    g_now = np.eye(4)

    for ii in range(n_dof):
        g_now = g_now @ gg[:, :, ii]
        g_prefix[:, :, ii] = g_now

    Aij = np.zeros((n_dof, n_dof, 6, 6), dtype=float)

    for jj in range(n_dof):
        for ii in range(n_dof):
            ggjl = np.eye(4)

            if ii > jj:
                for kk in range(jj + 1, ii + 1):
                    ggjl = ggjl @ gg[:, :, kk]

                Aij[ii, jj] = adjmginv(ggjl)

            elif ii == jj:
                Aij[ii, jj] = np.eye(6)

            else:
                Aij[ii, jj] = np.zeros((6, 6))

    return gg, g_prefix, Aij


def build_twist_exponentials(xi, theta):
    xi = np.asarray(xi, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    n_dof = xi.shape[1]
    gg = np.zeros((4, 4, n_dof), dtype=float)
    g_prefix = np.zeros((4, 4, n_dof), dtype=float)
    g_now = np.eye(4)

    for ii in range(n_dof):
        gg[:, :, ii] = gtwist_num(xi[:, ii], theta[ii])
        g_now = g_now @ gg[:, :, ii]
        g_prefix[:, :, ii] = g_now

    return gg, g_prefix


def rigid_inverse(g):
    g = np.asarray(g, dtype=float).reshape(4, 4)
    R = g[0:3, 0:3]
    p = g[0:3, 3]

    g_inv = np.eye(4)
    g_inv[0:3, 0:3] = R.T
    g_inv[0:3, 3] = -R.T @ p

    return g_inv


def adjmginv_apply(g, xi_vec):
    g = np.asarray(g, dtype=float).reshape(4, 4)
    xi_vec = np.asarray(xi_vec, dtype=float).reshape(6)

    R_t = g[0:3, 0:3].T
    p = g[0:3, 3]
    v = xi_vec[0:3]
    w = xi_vec[3:6]

    out = np.zeros(6, dtype=float)
    out[0:3] = R_t @ (v - np.cross(p, w))
    out[3:6] = R_t @ w

    return out


def compute_mass_matrix_mcn(xi, calMp, lM0, theta):
    xi = np.asarray(xi, dtype=float)
    calMp = np.asarray(calMp, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    n_dof = xi.shape[1]
    _, g_prefix = build_twist_exponentials(xi, theta)

    body_twist = np.zeros((n_dof, n_dof, 6), dtype=float)
    inertia_twist = np.zeros((n_dof, n_dof, 6), dtype=float)

    for ll in range(n_dof):
        for jj in range(ll + 1):
            if ll == jj:
                body_twist[ll, jj] = xi[:, jj]
            else:
                g_rel = rigid_inverse(g_prefix[:, :, jj]) @ g_prefix[:, :, ll]
                body_twist[ll, jj] = adjmginv_apply(g_rel, xi[:, jj])

            inertia_twist[ll, jj] = calMp[:, :, ll] @ body_twist[ll, jj]

    M = np.zeros((n_dof, n_dof), dtype=float)

    for ii in range(n_dof):
        for jj in range(n_dof):
            for ll in range(int(lM0[ii, jj]), n_dof):
                M[ii, jj] += body_twist[ll, ii] @ inertia_twist[ll, jj]

    return 0.5 * (M + M.T)


def compute_coriolis_matrix_by_mass_diff(
    xi,
    calMp,
    lM0,
    theta,
    dtheta,
    M_base,
    eps=1e-7,
):
    """
    Fast Coriolis helper.

    The previous MCNsim computed dM/dq analytically through nested Python
    loops. That is accurate but too slow for a 200 Hz control loop. This uses
    forward differences of M(q), reusing the M(q) already needed by the
    computed-torque law. This is less exact than central differences but cuts
    the number of mass-matrix evaluations in half.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    dtheta = np.asarray(dtheta, dtype=float).reshape(-1)
    n_dof = theta.size

    dM = np.zeros((n_dof, n_dof, n_dof), dtype=float)

    for kk in range(n_dof):
        theta_plus = theta.copy()
        theta_plus[kk] += eps

        M_plus = compute_mass_matrix_mcn(xi, calMp, lM0, theta_plus)
        dM[:, :, kk] = (M_plus - M_base) / eps

    C = np.zeros((n_dof, n_dof), dtype=float)

    for ii in range(n_dof):
        for jj in range(n_dof):
            for kk in range(n_dof):
                C[ii, jj] += 0.5 * (
                    dM[ii, jj, kk]
                    + dM[ii, kk, jj]
                    - dM[kk, jj, ii]
                ) * dtheta[kk]

    return C


def MCNsim(xi, calMp, lM, beta, mg, theta, dtheta):
    """
    Return:
        M(q)
        C(q,dq)
        N(q,dq)

    Dynamics form:
        M(q) ddq + C(q,dq) dq + N(q,dq) = tau
    """
    xi = np.asarray(xi, dtype=float)
    calMp = np.asarray(calMp, dtype=float)
    lM = np.asarray(lM, dtype=int)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    mg = np.asarray(mg, dtype=float).reshape(-1)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    dtheta = np.asarray(dtheta, dtype=float).reshape(-1)

    n_dof = xi.shape[1]

    # Allow either 0-based or 1-based lM.
    lM0 = lM.copy()

    if lM0.min() >= 1:
        lM0 = lM0 - 1

    M = compute_mass_matrix_mcn(xi, calMp, lM0, theta)
    C = compute_coriolis_matrix_by_mass_diff(
        xi,
        calMp,
        lM0,
        theta,
        dtheta,
        M,
    )

    # N(q,dq) = viscosity + gravity.
    _, g_prefix, _ = build_mcn_kinematics(xi, theta)
    N_viscosity = beta * dtheta
    N_gravity = calc_gravity_by_potential(xi, calMp, g_prefix, mg)

    N = N_viscosity + N_gravity

    return M, C, N


def SCARA_initialize():
    """
    Z1 MCN model initialization.
    Function name is kept from the original MATLAB/Python conversion.
    """
    # Z1 screw axes.
    omega1 = np.array([0.0, 0.0, 1.0])
    omega2 = np.array([0.0, 1.0, 0.0])
    omega3 = np.array([0.0, 1.0, 0.0])
    omega4 = np.array([0.0, 1.0, 0.0])
    omega5 = np.array([0.0, 0.0, 1.0])
    omega6 = np.array([1.0, 0.0, 0.0])

    q1 = np.array([ 0.0000, 0.0000, 0.0585])
    q2 = np.array([ 0.0000, 0.0000, 0.1035])
    q3 = np.array([-0.3500, 0.0000, 0.1035])
    q4 = np.array([-0.1320, 0.0000, 0.1605])
    q5 = np.array([-0.0620, 0.0000, 0.1605])
    q6 = np.array([-0.0128, 0.0000, 0.1605])

    xi1 = np.concatenate((-hatm_num(omega1) @ q1, omega1))
    xi2 = np.concatenate((-hatm_num(omega2) @ q2, omega2))
    xi3 = np.concatenate((-hatm_num(omega3) @ q3, omega3))
    xi4 = np.concatenate((-hatm_num(omega4) @ q4, omega4))
    xi5 = np.concatenate((-hatm_num(omega5) @ q5, omega5))
    xi6 = np.concatenate((-hatm_num(omega6) @ q6, omega6))

    xi = np.column_stack((xi1, xi2, xi3, xi4, xi5, xi6))

    # Link masses.
    m1 = 0.67332551
    m2 = 1.19132258
    m3 = 0.83940874
    m4 = 0.56404563
    m5 = 0.38938492
    m6 = 0.28875807

    # Link inertia matrices.
    I1 = np.array([
        [ 0.00128328, -0.00000006, -0.00000040],
        [-0.00000006,  0.00071931,  0.00000050],
        [-0.00000040,  0.00000050,  0.00083936],
    ])

    I2 = np.array([
        [ 0.00102138,  0.00062358,  0.00000513],
        [ 0.00062358,  0.02429457, -0.00000210],
        [ 0.00000513, -0.00000210,  0.02466114],
    ])

    I3 = np.array([
        [ 0.00108061, -0.00008669, -0.00208102],
        [-0.00008669,  0.00954238, -0.00001332],
        [-0.00208102, -0.00001332,  0.00886621],
    ])

    I4 = np.array([
        [ 0.00031576,  0.00008130,  0.00004091],
        [ 0.00008130,  0.00092996, -0.00000596],
        [ 0.00004091, -0.00000596,  0.00097912],
    ])

    I5 = np.array([
        [ 0.00017605,  0.00000040,  0.00005689],
        [ 0.00000040,  0.00055896, -0.00000013],
        [ 0.00005689, -0.00000013,  0.00053860],
    ])

    I6 = np.array([
        [ 0.00018328,  0.00000122,  0.00000054],
        [ 0.00000122,  0.00014750,  0.00000008],
        [ 0.00000054,  0.00000008,  0.00014680],
    ])

    calM1 = np.block([[m1 * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), I1]])
    calM2 = np.block([[m2 * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), I2]])
    calM3 = np.block([[m3 * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), I3]])
    calM4 = np.block([[m4 * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), I4]])
    calM5 = np.block([[m5 * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), I5]])
    calM6 = np.block([[m6 * np.eye(3), np.zeros((3, 3))], [np.zeros((3, 3)), I6]])

    def make_gsl(p):
        g = np.eye(4)
        g[0:3, 3] = np.asarray(p, dtype=float)
        return g

    # Link COM transforms in base/home frame.
    gsl10 = make_gsl([ 0.00000247, -0.00025198, 0.08167169])
    gsl20 = make_gsl([-0.11012601,  0.00240029, 0.10508266])
    gsl30 = make_gsl([-0.24390792, -0.00541815, 0.13826383])
    gsl40 = make_gsl([-0.08833319,  0.00364738, 0.15879808])
    gsl50 = make_gsl([-0.03078467,  0.00000000, 0.16696316])
    gsl60 = make_gsl([ 0.01135690, -0.00017355, 0.15906124])

    Adgsl10m1 = np.linalg.inv(adjmg(gsl10))
    Adgsl20m1 = np.linalg.inv(adjmg(gsl20))
    Adgsl30m1 = np.linalg.inv(adjmg(gsl30))
    Adgsl40m1 = np.linalg.inv(adjmg(gsl40))
    Adgsl50m1 = np.linalg.inv(adjmg(gsl50))
    Adgsl60m1 = np.linalg.inv(adjmg(gsl60))

    calMp = np.zeros((6, 6, 6), dtype=float)

    calMp[:, :, 0] = Adgsl10m1.T @ calM1 @ Adgsl10m1
    calMp[:, :, 1] = Adgsl20m1.T @ calM2 @ Adgsl20m1
    calMp[:, :, 2] = Adgsl30m1.T @ calM3 @ Adgsl30m1
    calMp[:, :, 3] = Adgsl40m1.T @ calM4 @ Adgsl40m1
    calMp[:, :, 4] = Adgsl50m1.T @ calM5 @ Adgsl50m1
    calMp[:, :, 5] = Adgsl60m1.T @ calM6 @ Adgsl60m1

    # Viscous damping. Keep this zero to match Pinocchio's
    # nonLinearEffects(), which does not include joint viscous friction.
    beta = np.zeros(6, dtype=float)

    # Gravity.
    g_const = 9.81
    mg = g_const * np.array([m1, m2, m3, m4, m5, m6], dtype=float)

    n_dof = 6

    # Python version uses 0-based lM.
    lM = np.zeros((n_dof, n_dof), dtype=int)

    for ii in range(n_dof):
        for jj in range(n_dof):
            lM[ii, jj] = max(ii, jj)

    return {
        "xi": xi,
        "calMp": calMp,
        "lM": lM,
        "beta": beta,
        "mg": mg,
        "n_dof": n_dof,
    }


# ============================================================
# 3. Read q, dq from /z1_gazebo/joint_states
# ============================================================

class Z1JointStateReader:
    def __init__(self):
        self.joint_names = [
            "joint1", "joint2", "joint3",
            "joint4", "joint5", "joint6",
        ]

        self.q = np.zeros(6)
        self.dq = np.zeros(6)
        self.effort = np.zeros(6)
        self.received = False

        self.sub = rospy.Subscriber(
            "/z1_gazebo/joint_states",
            JointState,
            self.callback,
            queue_size=1,
        )

    def callback(self, msg):
        name_to_index = {name: i for i, name in enumerate(msg.name)}

        q = np.zeros(6)
        dq = np.zeros(6)
        effort = np.zeros(6)

        for i, name in enumerate(self.joint_names):
            if name not in name_to_index:
                rospy.logwarn_throttle(
                    1.0,
                    f"{name} not found in /z1_gazebo/joint_states"
                )
                return

            idx = name_to_index[name]

            q[i] = msg.position[idx]

            if len(msg.velocity) > idx:
                dq[i] = msg.velocity[idx]

            if len(msg.effort) > idx:
                effort[i] = msg.effort[idx]

        self.q = q
        self.dq = dq
        self.effort = effort
        self.received = True

    def wait_until_ready(self, timeout=5.0):
        t0 = time.time()
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            if self.received:
                return True

            if time.time() - t0 > timeout:
                return False

            rate.sleep()

        return False


# ============================================================
# 4. Publish tau to effort_controller topics
# ============================================================

class Z1EffortTopicSender:
    def __init__(self):
        topic_names = [
            "/z1_gazebo/joint1_effort_controller/command",
            "/z1_gazebo/joint2_effort_controller/command",
            "/z1_gazebo/joint3_effort_controller/command",
            "/z1_gazebo/joint4_effort_controller/command",
            "/z1_gazebo/joint5_effort_controller/command",
            "/z1_gazebo/joint6_effort_controller/command",
        ]

        self.pubs = [
            rospy.Publisher(topic, Float64, queue_size=1)
            for topic in topic_names
        ]

        rospy.sleep(0.5)

    def send_tau(self, tau):
        tau = np.asarray(tau, dtype=float).reshape(6)

        for i in range(6):
            self.pubs[i].publish(Float64(float(tau[i])))

    def stop(self):
        self.send_tau(np.zeros(6))


# ============================================================
# 5. MCN dynamics wrapper
# ============================================================

def compute_lagrange_terms_mcn(params, q, dq, backend=None):
    """
    Replace Pinocchio dynamics with MCN dynamics.

    MCNsim returns:
        M(q), C(q,dq), N(q,dq)

    The computed torque code needs:
        tau = M @ qdd_cmd + h

    Therefore:
        h = C @ dq + N
    """
    q = np.asarray(q, dtype=float).reshape(6)
    dq = np.asarray(dq, dtype=float).reshape(6)

    if backend == "cpp":
        if mcn_dynamics_cpp is None:
            raise RuntimeError(
                "C++ backend requested, but the mcn_dynamics extension is unavailable."
            )
        M, C, N = mcn_dynamics_cpp.MCNsim(
            params["xi"],
            params["calMp"],
            params["lM"],
            params["beta"],
            params["mg"],
            q,
            dq,
        )
    elif backend == "python":
        M, C, N = MCNsim(
            params["xi"],
            params["calMp"],
            params["lM"],
            params["beta"],
            params["mg"],
            q,
            dq,
        )
    else:
        raise ValueError(f"Unsupported MCN backend: {backend}")

    h = C @ dq + N

    return M, h, C, N


# ============================================================
# 6. Reference trajectory
# ============================================================

def smooth_step(s):
    """
    Quintic smooth step:
        y(0)=0, y(1)=1
        yd(0)=yd(1)=0
        ydd(0)=ydd(1)=0
    """
    s = np.clip(s, 0.0, 1.0)

    y = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
    yd = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
    ydd = 60.0 * s - 180.0 * s**2 + 120.0 * s**3

    return y, yd, ydd


def reference_trajectory(t, q_start, q_goal, move_start=1.0, move_duration=4.0):
    q_start = np.asarray(q_start, dtype=float).reshape(6)
    q_goal = clamp_q(q_goal)

    if t <= move_start:
        return q_start.copy(), np.zeros(6), np.zeros(6)

    if t >= move_start + move_duration:
        return q_goal.copy(), np.zeros(6), np.zeros(6)

    s = (t - move_start) / move_duration
    y, yd, ydd = smooth_step(s)

    delta = q_goal - q_start

    q_ref = q_start + y * delta
    dq_ref = yd * delta / move_duration
    ddq_ref = ydd * delta / (move_duration ** 2)

    q_ref = clamp_q(q_ref)

    return q_ref, dq_ref, ddq_ref


def reference_trajectory_go_and_return(
    t,
    q0,
    q_goal,
    move_start_1,
    move_duration_1,
    hold_duration,
    move_duration_2,
):
    """
    Full reference:
        1. hold q0
        2. q0 -> q_goal
        3. hold q_goal
        4. q_goal -> q0
        5. hold q0
    """
    move_start_2 = move_start_1 + move_duration_1 + hold_duration

    if t < move_start_2:
        q_ref, dq_ref, ddq_ref = reference_trajectory(
            t=t,
            q_start=q0,
            q_goal=q_goal,
            move_start=move_start_1,
            move_duration=move_duration_1,
        )
    else:
        q_ref, dq_ref, ddq_ref = reference_trajectory(
            t=t,
            q_start=q_goal,
            q_goal=q0,
            move_start=move_start_2,
            move_duration=move_duration_2,
        )

    return q_ref, dq_ref, ddq_ref


def export_tracking_tau_csv(path, t_log, q_ref_log, q_log, tau_log):
    path = os.path.abspath(os.path.expanduser(path))
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    data = np.column_stack((t_log, q_ref_log, q_log, tau_log))
    header = ",".join(
        ["time"]
        + [f"q_ref{i + 1}" for i in range(6)]
        + [f"q{i + 1}" for i in range(6)]
        + [f"tau{i + 1}" for i in range(6)]
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%.9f")
    return path


# ============================================================
# 7. Main pure computed torque loop
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--T", type=float, default=18.0, help="total control time")
    parser.add_argument("--dt", type=float, default=0.005, help="control period")
    parser.add_argument("--move-duration", type=float, default=5.5, help="reference movement duration")
    parser.add_argument("--hold-duration", type=float, default=3.0, help="hold time at q_goal before returning")
    parser.add_argument("--final-hold", type=float, default=3.0, help="hold time at q0 after returning")
    parser.add_argument("--tau-scale", type=float, default=1.0, help="scale applied to computed torque")
    parser.add_argument(
        "--backend",
        choices=("python", "cpp"),
        required=True,
        help="force the Python or C++/pybind11 MCN dynamics backend",
    )
    parser.add_argument(
        "--tracking-tau-csv",
        default=None,
        help="CSV output path for q tracking/reference and tau data",
    )

    args = parser.parse_args()

    if args.backend == "cpp" and mcn_dynamics_cpp is None:
        parser.error(
            "--backend cpp requested, but mcn_dynamics could not be imported. "
            "Build it first with: python3 setup_mcn_cpp.py build_ext --inplace"
        )

    if args.tracking_tau_csv is None:
        args.tracking_tau_csv = os.path.join(
            os.path.dirname(__file__),
            f"MCN_CT_{args.backend}_tracking_tau.csv",
        )

    rospy.init_node("z1_pure_ct_effort_topic_mcn")

    print("===================================================")
    print("Z1 PURE Computed Torque via effort_controller topic")
    print("Dynamics model: MCNsim, not Pinocchio")
    print("No UnitreeArm.")
    print("No LowCmd.")
    print("Only tau is published.")
    print("Return-to-initial-pose is enabled.")
    print("===================================================")

    # Make sure Gazebo physics is running.
    try:
        rospy.wait_for_service("/gazebo/unpause_physics", timeout=2.0)

        unpause = rospy.ServiceProxy(
            "/gazebo/unpause_physics",
            __import__("std_srvs.srv").srv.Empty,
        )

        unpause()
        print("[Gazebo] Physics unpaused.")

    except Exception:
        print("[Gazebo] Could not call unpause_physics. Continue anyway.")

    # ========================================================
    # MCN model initialization
    # ========================================================
    params = SCARA_initialize()

    print("[MCN] Model initialized.")
    print("[MCN] xi shape    =", params["xi"].shape)
    print("[MCN] calMp shape =", params["calMp"].shape)
    print("[MCN] lM shape    =", params["lM"].shape)
    print(
        "[MCN] backend     =",
        "C++ mcn_dynamics" if args.backend == "cpp" else "Python MCNsim",
    )

    state_reader = Z1JointStateReader()
    effort_sender = Z1EffortTopicSender()

    print("[ROS] Waiting for /z1_gazebo/joint_states ...")

    if not state_reader.wait_until_ready(timeout=5.0):
        raise RuntimeError("Did not receive /z1_gazebo/joint_states")

    q0 = clamp_q(state_reader.q)

    # Target pose.
    q_goal = np.array([0.60, 0.60, -0.60, -0.35, 0.80, 0.80], dtype=float)
    q_goal = clamp_q(q_goal)

    print("[Reference] q0     =", q0)
    print("[Reference] q_goal =", q_goal)

    # These Kp/Kd are ONLY inside Python computed torque.
    # They are NOT sent to the motor. Keep them consistent with
    # CT_GPT_torque.py.
    wn = np.array([
        22.0,
        46.0,
        55.0,
        150.0,
        140.0,
        540.0,
    ], dtype=float)

    eta = np.array([
        1.0,
        1.35,
        0.9,
        1.05,
        1.4,
        1.25,
    ], dtype=float)

    Kp = np.diag(wn ** 2)
    Kd = np.diag(2.0 * eta * wn)

    # Conservative torque limits.
    tau_limit = np.array([45, 60, 45, 50, 50, 50], dtype=float)

    # ========================================================
    # Time plan for go-and-return motion
    # ========================================================
    move_start_1 = 1.0
    move_duration_1 = args.move_duration
    hold_duration = args.hold_duration
    move_duration_2 = args.move_duration

    move_start_2 = move_start_1 + move_duration_1 + hold_duration
    total_time_needed = move_start_2 + move_duration_2 + args.final_hold

    if args.T < total_time_needed:
        print("[Time] Given --T is too short for go-and-return motion.")
        print("[Time] Automatically set T from", args.T, "to", total_time_needed)
        args.T = total_time_needed

    print("[Motion Plan]")
    print("  0.0 ~", move_start_1, ": hold q0")
    print(" ", move_start_1, "~", move_start_1 + move_duration_1, ": q0 -> q_goal")
    print(" ", move_start_1 + move_duration_1, "~", move_start_2, ": hold q_goal")
    print(" ", move_start_2, "~", move_start_2 + move_duration_2, ": q_goal -> q0")
    print(" ", move_start_2 + move_duration_2, "~", args.T, ": hold q0")

    # Logs.
    t_log = []
    q_ref_log = []
    q_log = []
    dq_log = []
    tau_log = []
    tau_raw_log = []
    h_log = []
    n_log = []
    loop_dt_log = []
    calc_dt_log = []

    rate = rospy.Rate(1.0 / args.dt)

    t0 = rospy.Time.now().to_sec()

    print("[Control] Start pure computed torque control with MCN model.")

    try:
        while not rospy.is_shutdown():
            loop_start = time.perf_counter()
            t = rospy.Time.now().to_sec() - t0

            if t > args.T:
                break

            q = state_reader.q.copy()
            dq = state_reader.dq.copy()



            # Reference: q0 -> q_goal -> q0.
            q_ref, dq_ref, ddq_ref = reference_trajectory_go_and_return(
                t=t,
                q0=q0,
                q_goal=q_goal,
                move_start_1=move_start_1,
                move_duration_1=move_duration_1,
                hold_duration=hold_duration,
                move_duration_2=move_duration_2,
            )

            e = q_ref - q
            de = dq_ref - dq

            qdd_cmd = ddq_ref + Kd @ de + Kp @ e

            calc_start = time.perf_counter()

            # ====================================================
            # MCN dynamics instead of Pinocchio:
            #   M, C, N = MCNsim(...)
            #   h = C @ dq + N
            # ====================================================
            M, h, C, N = compute_lagrange_terms_mcn(
                params,
                q,
                dq,
                backend=args.backend,
            )
            calc_dt_log.append(time.perf_counter() - calc_start)

            # Computed torque core.
            tau_raw = M @ qdd_cmd + h
            tau_raw = args.tau_scale * tau_raw

            tau=np.clip(tau_raw, -tau_limit, tau_limit)

            effort_sender.send_tau(tau)

            t_log.append(t)
            q_ref_log.append(q_ref.copy())
            q_log.append(q.copy())
            dq_log.append(dq.copy())
            tau_log.append(tau.copy())
            tau_raw_log.append(tau_raw.copy())
            h_log.append(h.copy())
            n_log.append(N.copy())

            rate.sleep()
            loop_dt_log.append(time.perf_counter() - loop_start)

    except KeyboardInterrupt:
        print("\n[Control] Interrupted.")

    finally:
        print("[Control] Finished go-and-return motion.")
        print("[Control] Sending zero torque after returning to q0.")

        for _ in range(20):
            effort_sender.stop()
            rospy.sleep(0.01)

    # ========================================================
    # Convert logs
    # ========================================================
    t_log = np.asarray(t_log)
    q_ref_log = np.asarray(q_ref_log)
    q_log = np.asarray(q_log)
    dq_log = np.asarray(dq_log)
    tau_log = np.asarray(tau_log)
    tau_raw_log = np.asarray(tau_raw_log)
    h_log = np.asarray(h_log)
    n_log = np.asarray(n_log)
    loop_dt_log = np.asarray(loop_dt_log)
    calc_dt_log = np.asarray(calc_dt_log)

    if t_log.size > 0:
        csv_path = export_tracking_tau_csv(args.tracking_tau_csv, t_log, q_ref_log, q_log, tau_log)
        print(f"\n[CSV] q tracking/reference and tau data saved to: {csv_path}")
    else:
        print("\n[CSV] No data was collected; CSV was not written.")

    if loop_dt_log.size > 0:
        overrun = loop_dt_log > args.dt * 1.05
        print("\n[Timing]")
        print(f"  requested dt       = {args.dt:.6f} s ({1.0 / args.dt:.1f} Hz)")
        print(f"  actual loop dt avg = {loop_dt_log.mean():.6f} s ({1.0 / loop_dt_log.mean():.1f} Hz)")
        print(f"  actual loop dt max = {loop_dt_log.max():.6f} s")
        print(f"  MCN calc dt avg    = {calc_dt_log.mean():.6f} s")
        print(f"  MCN calc dt max    = {calc_dt_log.max():.6f} s")
        print(f"  overrun fraction   = {100.0 * overrun.mean():.2f}%")

        if np.any(overrun):
            print("[Timing WARNING] MCN loop is slower than requested dt; high gains may oscillate.")

    # ========================================================
    # Plots
    # ========================================================
    if t_log.size > 0:
        final_error = q_ref_log[-1] - q_log[-1]

        print("\nFinal tracking error:")
        for i in range(6):
            print(f"Joint {i + 1}: error = {final_error[i]: .5f} rad")

        print("Max abs error =", np.max(np.abs(final_error)))

        fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)

        for i in range(6):
            axes[i].plot(
                t_log,
                q_ref_log[:, i],
                "r--",
                linewidth=2,
                label=r"Reference $q_d$",
            )

            axes[i].plot(
                t_log,
                q_log[:, i],
                "b",
                linewidth=1.5,
                label=r"Measured $q$",
            )

            axes[i].set_ylabel(f"J{i + 1} rad")
            axes[i].grid(True)
            axes[i].legend(loc="best")
            axes[i].set_title(f"Joint {i + 1} Pure CT Effort Topic Tracking")

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Z1 Pure CT with MCN Model: q0 -> q_goal -> q0")
        plt.tight_layout()
        plt.show()

        fig2, axes2 = plt.subplots(6, 1, figsize=(10, 10), sharex=True)

        for i in range(6):
            axes2[i].plot(t_log, tau_log[:, i], linewidth=1.5)
            axes2[i].set_ylabel(f"tau{i + 1}")
            axes2[i].grid(True)

        axes2[-1].set_xlabel("Time (s)")
        fig2.suptitle("Pure Computed Torque Command from MCN Model")
        plt.tight_layout()
        plt.show()

        fig3, axes3 = plt.subplots(6, 1, figsize=(10, 10), sharex=True)

        for i in range(6):
            axes3[i].plot(
                t_log,
                tau_raw_log[:, i],
                "r--",
                linewidth=1.0,
                label="raw tau",
            )

            axes3[i].plot(
                t_log,
                tau_log[:, i],
                "b",
                linewidth=1.2,
                label="clipped tau",
            )

            axes3[i].set_ylabel(f"tau{i + 1}")
            axes3[i].grid(True)
            axes3[i].legend(loc="best")

        axes3[-1].set_xlabel("Time (s)")
        fig3.suptitle("Raw Tau vs Clipped Tau from MCN Model")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

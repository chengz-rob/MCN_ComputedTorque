#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using Vec3 = std::array<double, 3>;
using Vec6 = std::array<double, 6>;
using Mat3 = std::array<double, 9>;
using Mat4 = std::array<double, 16>;
using Mat6 = std::array<double, 36>;
using AijStore = std::array<double, 6 * 6 * 6 * 6>;

static inline double &m3(Mat3 &m, int r, int c) { return m[3 * r + c]; }
static inline double m3c(const Mat3 &m, int r, int c) { return m[3 * r + c]; }
static inline double &m4(Mat4 &m, int r, int c) { return m[4 * r + c]; }
static inline double m4c(const Mat4 &m, int r, int c) { return m[4 * r + c]; }
static inline double &m6(Mat6 &m, int r, int c) { return m[6 * r + c]; }
static inline double m6c(const Mat6 &m, int r, int c) { return m[6 * r + c]; }
static inline double &aij(AijStore &a, int link, int joint, int r, int c) {
    return a[((link * 6 + joint) * 6 + r) * 6 + c];
}
static inline double aijc(const AijStore &a, int link, int joint, int r, int c) {
    return a[((link * 6 + joint) * 6 + r) * 6 + c];
}

static Mat3 eye3() {
    Mat3 out{};
    m3(out, 0, 0) = 1.0;
    m3(out, 1, 1) = 1.0;
    m3(out, 2, 2) = 1.0;
    return out;
}

static Mat4 eye4() {
    Mat4 out{};
    for (int i = 0; i < 4; ++i) {
        m4(out, i, i) = 1.0;
    }
    return out;
}

static Mat6 eye6() {
    Mat6 out{};
    for (int i = 0; i < 6; ++i) {
        m6(out, i, i) = 1.0;
    }
    return out;
}

static Mat3 hat3(const Vec3 &w) {
    Mat3 h{};
    m3(h, 0, 1) = -w[2];
    m3(h, 0, 2) = w[1];
    m3(h, 1, 0) = w[2];
    m3(h, 1, 2) = -w[0];
    m3(h, 2, 0) = -w[1];
    m3(h, 2, 1) = w[0];
    return h;
}

static Vec3 cross3(const Vec3 &a, const Vec3 &b) {
    return Vec3{
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

static Mat3 mat3_mul(const Mat3 &a, const Mat3 &b) {
    Mat3 out{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += m3c(a, r, k) * m3c(b, k, c);
            }
            m3(out, r, c) = s;
        }
    }
    return out;
}

static Mat4 mat4_mul(const Mat4 &a, const Mat4 &b) {
    Mat4 out{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) {
                s += m4c(a, r, k) * m4c(b, k, c);
            }
            m4(out, r, c) = s;
        }
    }
    return out;
}

static Vec3 mat3_vec(const Mat3 &a, const Vec3 &v) {
    Vec3 out{};
    for (int r = 0; r < 3; ++r) {
        out[r] = m3c(a, r, 0) * v[0] + m3c(a, r, 1) * v[1] + m3c(a, r, 2) * v[2];
    }
    return out;
}

static Vec6 mat6_vec(const Mat6 &a, const Vec6 &v) {
    Vec6 out{};
    for (int r = 0; r < 6; ++r) {
        double s = 0.0;
        for (int c = 0; c < 6; ++c) {
            s += m6c(a, r, c) * v[c];
        }
        out[r] = s;
    }
    return out;
}

static Vec6 aij_vec(const AijStore &a, int link, int joint, const Vec6 &v) {
    Vec6 out{};
    for (int r = 0; r < 6; ++r) {
        double s = 0.0;
        for (int c = 0; c < 6; ++c) {
            s += aijc(a, link, joint, r, c) * v[c];
        }
        out[r] = s;
    }
    return out;
}

static Vec6 calmp_vec(const py::detail::unchecked_reference<double, 3> &calMp, int link, const Vec6 &v) {
    Vec6 out{};
    for (int r = 0; r < 6; ++r) {
        double s = 0.0;
        for (int c = 0; c < 6; ++c) {
            s += calMp(r, c, link) * v[c];
        }
        out[r] = s;
    }
    return out;
}

static double dot6(const Vec6 &a, const Vec6 &b) {
    double s = 0.0;
    for (int i = 0; i < 6; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

static Vec6 xi_col(const py::detail::unchecked_reference<double, 2> &xi, int col) {
    Vec6 out{};
    for (int i = 0; i < 6; ++i) {
        out[i] = xi(i, col);
    }
    return out;
}

static Mat4 gtwist(const Vec6 &xi, double theta) {
    Vec3 v{xi[0], xi[1], xi[2]};
    Vec3 w{xi[3], xi[4], xi[5]};
    double nw = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);

    Mat4 out = eye4();

    if (nw < 1e-12) {
        out[3] = v[0] * theta;
        out[7] = v[1] * theta;
        out[11] = v[2] * theta;
        return out;
    }

    Mat3 hw = hat3(w);
    Mat3 hw2 = mat3_mul(hw, hw);
    Mat3 R = eye3();
    double st = std::sin(theta);
    double ct = std::cos(theta);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m3(R, r, c) += m3c(hw, r, c) * st + m3c(hw2, r, c) * (1.0 - ct);
            m4(out, r, c) = m3c(R, r, c);
        }
    }

    Vec3 hwv = mat3_vec(hw, v);
    Vec3 omega_dot_v{
        w[0] * (w[0] * v[0] + w[1] * v[1] + w[2] * v[2]) * theta,
        w[1] * (w[0] * v[0] + w[1] * v[1] + w[2] * v[2]) * theta,
        w[2] * (w[0] * v[0] + w[1] * v[1] + w[2] * v[2]) * theta,
    };
    Vec3 Rhwv = mat3_vec(R, hwv);

    for (int i = 0; i < 3; ++i) {
        m4(out, i, 3) = hwv[i] - Rhwv[i] + omega_dot_v[i];
    }

    return out;
}

static Mat6 adjmginv(const Mat4 &g) {
    Mat6 out{};
    Vec3 p{m4c(g, 0, 3), m4c(g, 1, 3), m4c(g, 2, 3)};
    Mat3 hp = hat3(p);

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double rt = m4c(g, c, r);
            m6(out, r, c) = rt;
            m6(out, r + 3, c + 3) = rt;
        }
    }

    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += m4c(g, k, r) * m3c(hp, k, c);
            }
            m6(out, r, c + 3) = -s;
        }
    }

    return out;
}

static Vec6 lie(const Vec6 &x1, const Vec6 &x2) {
    Vec3 v1{x1[0], x1[1], x1[2]};
    Vec3 w1{x1[3], x1[4], x1[5]};
    Vec3 v2{x2[0], x2[1], x2[2]};
    Vec3 w2{x2[3], x2[4], x2[5]};

    Vec3 w1xv2 = cross3(w1, v2);
    Vec3 w2xv1 = cross3(w2, v1);
    Vec3 w1xw2 = cross3(w1, w2);

    return Vec6{
        w1xv2[0] - w2xv1[0],
        w1xv2[1] - w2xv1[1],
        w1xv2[2] - w2xv1[2],
        w1xw2[0],
        w1xw2[1],
        w1xw2[2],
    };
}

static Vec3 extract_com(const py::detail::unchecked_reference<double, 3> &calMp, int link, double mass) {
    double hmc21 = calMp(5, 1, link) / mass;
    double hmc02 = calMp(3, 2, link) / mass;
    double hmc10 = calMp(4, 0, link) / mass;
    return Vec3{hmc21, hmc02, hmc10};
}

static py::tuple MCNsim_cpp(
    py::array_t<double, py::array::c_style | py::array::forcecast> xi_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> calMp_arr,
    py::array_t<int, py::array::c_style | py::array::forcecast> lM_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> beta_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> mg_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> theta_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> dtheta_arr
) {
    if (xi_arr.ndim() != 2 || xi_arr.shape(0) != 6 || xi_arr.shape(1) != 6) {
        throw std::runtime_error("xi must have shape (6, 6)");
    }
    if (calMp_arr.ndim() != 3 || calMp_arr.shape(0) != 6 || calMp_arr.shape(1) != 6 || calMp_arr.shape(2) != 6) {
        throw std::runtime_error("calMp must have shape (6, 6, 6)");
    }

    auto xi = xi_arr.unchecked<2>();
    auto calMp = calMp_arr.unchecked<3>();
    auto lM = lM_arr.unchecked<2>();
    auto beta = beta_arr.unchecked<1>();
    auto mg = mg_arr.unchecked<1>();
    auto theta = theta_arr.unchecked<1>();
    auto dtheta = dtheta_arr.unchecked<1>();

    std::array<Mat4, 6> gg;
    std::array<Mat4, 6> g_prefix;
    Mat4 g_now = eye4();

    for (int i = 0; i < 6; ++i) {
        gg[i] = gtwist(xi_col(xi, i), theta(i));
        g_now = mat4_mul(g_now, gg[i]);
        g_prefix[i] = g_now;
    }

    AijStore Aij{};

    for (int jj = 0; jj < 6; ++jj) {
        for (int ii = 0; ii < 6; ++ii) {
            Mat6 value{};

            if (ii > jj) {
                Mat4 ggjl = eye4();
                for (int kk = jj + 1; kk <= ii; ++kk) {
                    ggjl = mat4_mul(ggjl, gg[kk]);
                }
                value = adjmginv(ggjl);
            } else if (ii == jj) {
                value = eye6();
            }

            for (int r = 0; r < 6; ++r) {
                for (int c = 0; c < 6; ++c) {
                    aij(Aij, ii, jj, r, c) = m6c(value, r, c);
                }
            }
        }
    }

    py::array_t<double> M_arr({6, 6});
    py::array_t<double> C_arr({6, 6});
    py::array_t<double> N_arr({6});
    auto M = M_arr.mutable_unchecked<2>();
    auto C = C_arr.mutable_unchecked<2>();
    auto N = N_arr.mutable_unchecked<1>();

    for (int i = 0; i < 6; ++i) {
        N(i) = 0.0;
        for (int j = 0; j < 6; ++j) {
            M(i, j) = 0.0;
            C(i, j) = 0.0;
        }
    }

    int lM0[6][6];
    bool one_based = true;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (lM(i, j) < 1) {
                one_based = false;
            }
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            lM0[i][j] = lM(i, j) - (one_based ? 1 : 0);
        }
    }

    for (int ii = 0; ii < 6; ++ii) {
        Vec6 xi_i = xi_col(xi, ii);
        for (int jj = 0; jj < 6; ++jj) {
            Vec6 xi_j = xi_col(xi, jj);
            double sum = 0.0;
            for (int ll = lM0[ii][jj]; ll < 6; ++ll) {
                Vec6 ai = aij_vec(Aij, ll, ii, xi_i);
                Vec6 aj = aij_vec(Aij, ll, jj, xi_j);
                Vec6 mp_aj = calmp_vec(calMp, ll, aj);
                sum += dot6(ai, mp_aj);
            }
            M(ii, jj) = sum;
        }
    }

    double dM[6][6][6] = {};

    for (int ii = 0; ii < 6; ++ii) {
        Vec6 xi_i = xi_col(xi, ii);
        for (int jj = 0; jj < 6; ++jj) {
            Vec6 xi_j = xi_col(xi, jj);
            for (int kk = 0; kk < 6; ++kk) {
                Vec6 xi_k = xi_col(xi, kk);
                double sum = 0.0;

                for (int ll = lM0[ii][jj]; ll < 6; ++ll) {
                    Vec6 aki = aij_vec(Aij, kk, ii, xi_i);
                    Vec6 lie1 = lie(aki, xi_k);
                    Vec6 alk = aij_vec(Aij, ll, kk, lie1);

                    Vec6 alj = aij_vec(Aij, ll, jj, xi_j);
                    Vec6 mp_alj = calmp_vec(calMp, ll, alj);
                    double term1 = dot6(alk, mp_alj);

                    Vec6 ali = aij_vec(Aij, ll, ii, xi_i);
                    Vec6 akj = aij_vec(Aij, kk, jj, xi_j);
                    Vec6 lie2 = lie(akj, xi_k);
                    Vec6 alk2 = aij_vec(Aij, ll, kk, lie2);
                    Vec6 mp_alk2 = calmp_vec(calMp, ll, alk2);
                    double term2 = dot6(ali, mp_alk2);

                    sum += term1 + term2;
                }

                dM[ii][jj][kk] = sum;
            }
        }
    }

    for (int ii = 0; ii < 6; ++ii) {
        for (int jj = 0; jj < 6; ++jj) {
            double sum = 0.0;
            for (int kk = 0; kk < 6; ++kk) {
                sum += 0.5 * (
                    dM[ii][jj][kk] + dM[ii][kk][jj] - dM[kk][jj][ii]
                ) * dtheta(kk);
            }
            C(ii, jj) = sum;
        }
    }

    Vec3 ez{0.0, 0.0, 1.0};
    std::array<Vec3, 6> p_com_home;

    for (int ll = 0; ll < 6; ++ll) {
        double mass = (calMp(0, 0, ll) + calMp(1, 1, ll) + calMp(2, 2, ll)) / 3.0;
        p_com_home[ll] = extract_com(calMp, ll, mass);
    }

    std::array<Vec3, 6> omega_cur;
    std::array<Vec3, 6> q_cur;

    for (int jj = 0; jj < 6; ++jj) {
        Vec6 xj = xi_col(xi, jj);
        Vec3 w{xj[3], xj[4], xj[5]};
        Vec3 v{xj[0], xj[1], xj[2]};
        double nw = std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        for (int i = 0; i < 3; ++i) {
            w[i] /= nw;
        }

        Vec3 q_home = cross3(w, v);
        Mat4 g_prev = jj == 0 ? eye4() : g_prefix[jj - 1];

        for (int r = 0; r < 3; ++r) {
            omega_cur[jj][r] = m4c(g_prev, r, 0) * w[0] + m4c(g_prev, r, 1) * w[1] + m4c(g_prev, r, 2) * w[2];
            q_cur[jj][r] = (
                m4c(g_prev, r, 0) * q_home[0]
                + m4c(g_prev, r, 1) * q_home[1]
                + m4c(g_prev, r, 2) * q_home[2]
                + m4c(g_prev, r, 3)
            );
        }
    }

    for (int ll = 0; ll < 6; ++ll) {
        Mat4 g_ll = g_prefix[ll];
        Vec3 p_com_ll{};

        for (int r = 0; r < 3; ++r) {
            p_com_ll[r] = (
                m4c(g_ll, r, 0) * p_com_home[ll][0]
                + m4c(g_ll, r, 1) * p_com_home[ll][1]
                + m4c(g_ll, r, 2) * p_com_home[ll][2]
                + m4c(g_ll, r, 3)
            );
        }

        for (int jj = 0; jj <= ll; ++jj) {
            Vec3 diff{
                p_com_ll[0] - q_cur[jj][0],
                p_com_ll[1] - q_cur[jj][1],
                p_com_ll[2] - q_cur[jj][2],
            };
            Vec3 jv = cross3(omega_cur[jj], diff);
            N(jj) += mg(ll) * (ez[0] * jv[0] + ez[1] * jv[1] + ez[2] * jv[2]);
        }
    }

    for (int i = 0; i < 6; ++i) {
        N(i) += beta(i) * dtheta(i);
    }

    return py::make_tuple(M_arr, C_arr, N_arr);
}

PYBIND11_MODULE(mcn_dynamics, m) {
    m.doc() = "C++ MCN dynamics for Unitree Z1";
    m.def("MCNsim", &MCNsim_cpp, "Compute M, C, N for the Z1 MCN model");
}

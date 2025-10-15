###############################################
# IMPORTS AND SETUP
# Import necessary libraries and configure settings
###############################################
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

np.set_printoptions(suppress=True)
# plt.rcParams["text.usetex"] = True


###############################################
# HELPER (OPTIONAL) FUNCTIONS TO AID IN DEBUGGING
# These are not graded functions, but may help you debug your code.
###############################################


def plot_orbit_2d(X):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_box_aspect([1, 1, 1])
    return fig


def plot_orbit_3d(X):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_box_aspect([1, 1, 1])
    return fig


def integrate_trajectory(X0, t_span, t_eval):
    # 1D spring mass damper
    def spring_equations_of_motion(t, X):
        x = X[0]
        v = X[1]
        k = 1.0  # spring constant
        m = 1.0  # mass
        c = 0.2
        dxdt = v
        dvdt = -(k / m) * x - (c / m) * v
        return np.array([dxdt, dvdt])

    sol = sp.integrate.solve_ivp(
        spring_equations_of_motion,
        t_span,
        X0,
        t_eval=t_eval,
        rtol=_EPS,
        atol=_EPS,
    )
    return sol


###############################################
# INTERNAL USE CONSTANTS AND FUNCTIONS
###############################################

# CONSTANTS
# Gravitational parameter for Earth in km^3/s^2
_MU = 3.986004418e5
# Machine Epsilon (Threshold Error)
_EPS = 1e-12


# FUNCTIONS
def _wrap_0_2pi(angle):
    a = np.fmod(angle, np.pi * 2)
    return a + np.pi * 2 if a < 0.0 else a


def _angle_around_h(u, v, h_hat):
    u_n = np.linalg.norm(u)
    v_n = np.linalg.norm(v)
    if u_n < _EPS or v_n < _EPS:
        return 0.0
    cu = np.dot(u, v) / (u_n * v_n)
    cu = np.clip(cu, -1.0, 1.0)
    su = np.dot(h_hat, np.cross(u, v)) / (u_n * v_n)
    return _wrap_0_2pi(np.arctan2(su, cu))


def _two_body_ode(t, X):
    rx, ry, rz, vx, vy, vz = X
    r3 = (rx * rx + ry * ry + rz * rz) ** 1.5
    ax = -_MU * rx / r3
    ay = -_MU * ry / r3
    az = -_MU * rz / r3
    return np.array([vx, vy, vz, ax, ay, az], dtype=float)


def _integrate_one_period(X0, num=2000, rtol=_EPS, atol=_EPS):
    oe0 = state_to_orbital_elements(X0)
    T = orbital_period(oe0)  # raises error if e>=1 or a invalid

    t_eval = np.linspace(0.0, T, num)
    sol = sp.integrate.solve_ivp(
        fun=_two_body_ode,
        t_span=(0.0, T),
        y0=np.asarray(X0, dtype=float),
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        # faster alg, optimized for smooth dynamics
        # use RK45 for more complex problems
        method="DOP853",
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")
    # (N, 6) array
    Y = sol.y.T
    return sol.t, Y


def _equal_lims_for_xyz(ax, x, y, z):
    rmax = np.max(np.abs(np.concatenate([x, y, z])))
    if rmax == 0:
        rmax = 1.0
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)
    # 3D axes also have zlim; call on 3D axes only
    if hasattr(ax, "set_zlim"):
        ax.set_zlim(-rmax, rmax)


###############################################
# REQUIRED FUNCTIONS FOR AUTOGRADER
# Keep the function signatures the same!!
###############################################


# REQUIRED
# Problem 3a:
def state_to_orbital_elements(X):
    r = np.array(X[:3], dtype=float)
    v = np.array(X[3:], dtype=float)

    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)

    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    k_hat = np.array([0.0, 0.0, 1.0])

    # Node vector (pointing toward ascending node)
    n = np.cross(k_hat, h)
    nnorm = np.linalg.norm(n)

    # Eccentricity vector
    e_vec = (np.cross(v, h) / _MU) - (r / rnorm)
    e = np.linalg.norm(e_vec)

    # Semi-major axis (vis-viva)
    energy = vnorm**2 / 2.0 - _MU / rnorm
    if abs(e - 1.0) < _EPS:  # Parabolic (no finite a)
        a = np.inf
    else:
        a = -_MU / (2.0 * energy)

    # Inclination
    i = np.arccos(np.clip(h[2] / max(hnorm, _EPS), -1.0, 1.0))

    # Unit angular-momentum for signed angles
    h_hat = h / max(hnorm, _EPS)

    # RAAN (Omega)
    if nnorm < _EPS:
        Omega = 0.0  # equatorial: undefined; set to 0 by convention
    else:
        Omega = _wrap_0_2pi(np.arctan2(n[1], n[0]))

    # Argument of perigee (omega) and true anomaly (f)
    if e < _EPS:
        # Circular: argument of perigee undefined.
        # Use argument of latitude u = angle from node to r.
        if nnorm < _EPS:
            # Circular + equatorial: everything undefined; use x-axis reference
            omega = 0.0
            f = _wrap_0_2pi(np.arctan2(r[1], r[0]))
        else:
            omega = 0.0
            f = _angle_around_h(n, r, h_hat)  # u
    else:
        # Non-circular: compute ω and f with signed angles in orbital plane
        if nnorm < _EPS:
            # Equatorial, non-circular: use x-axis as reference for ω
            ref = np.array([1.0, 0.0, 0.0])
            omega = _angle_around_h(ref, e_vec, h_hat)
        else:
            omega = _angle_around_h(n, e_vec, h_hat)

        f = _angle_around_h(e_vec, r, h_hat)

    return np.array([a, e, i, omega, Omega, f])


# REQUIRED
# Problem 3b
def orbital_elements_to_state(oe):
    a, e, i, omega, Omega, f = [float(x) for x in oe]

    if e >= 1.0 - _EPS and not np.isinf(a):
        raise ValueError("Must provide an elliptical orbit (e < 1).")

    # Semi-latus rectum
    p = a * (1.0 - e**2)

    # Perifocal coordinates (PQW frame)
    r_pf = (p / (1.0 + e * np.cos(f))) * np.array([np.cos(f), np.sin(f), 0.0])
    v_pf = np.sqrt(_MU / p) * np.array([-np.sin(f), e + np.cos(f), 0.0])

    # Rotation from PQW -> ECI (3-1-3 sequence)
    R_pqw_eci = sp.spatial.transform.Rotation.from_euler("ZXZ", [-omega, -i, -Omega])
    r_eci = r_pf @ R_pqw_eci.as_matrix()
    v_eci = v_pf @ R_pqw_eci.as_matrix()

    X_N = np.concatenate([r_eci, v_eci])

    return X_N


# REQUIRED
# Problem 3c
def orbital_period(oe):
    a = float(oe[0])
    e = float(oe[1])
    if e >= 1.0 - _EPS:
        raise ValueError(
            "Orbital period is defined only for elliptical orbits (e < 1)."
        )
    if a <= 0.0 or not np.isfinite(a):
        raise ValueError(
            "Semi-major axis must be positive and finite for an elliptical orbit."
        )
    period = np.pi * 2 * np.sqrt(a**3 / _MU)

    return period


# REQUIRED
# Problem 4a
def plot_orbits(X_N):
    figs = []
    t, Y = _integrate_one_period(X_N)
    x, y, z = Y[:, 0], Y[:, 1], Y[:, 2]

    # Plot 2D XY
    fig2d, ax2d = plt.subplots()
    ax2d.plot(x, y, lw=1.5)
    ax2d.plot([x[0]], [y[0]], marker="o", color="r", ms=5)  # start
    ax2d.plot([0], [0], marker="x", color="k", ms=6)  # focus/body at origin
    ax2d.set_xlabel("x [km]")
    ax2d.set_ylabel("y [km]")
    ax2d.set_aspect("equal", adjustable="box")
    _equal_lims_for_xyz(ax2d, x, y, z)
    ax2d.set_title("Orbit (XY projection)")
    figs.append(fig2d)

    # Plot 3D XYZ
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot3D(x, y, z, lw=1.2)
    ax3d.scatter([x[0]], [y[0]], [z[0]], marker="o", color="r", s=15)  # start
    ax3d.scatter([0], [0], [0], marker="x", color="k", s=30)  # central body
    ax3d.set_xlabel("x [km]")
    ax3d.set_ylabel("y [km]")
    ax3d.set_zlabel("z [km]")
    _equal_lims_for_xyz(ax3d, x, y, z)
    ax3d.set_title("Orbit (3D)")
    figs.append(fig3d)

    return figs


# REQUIRED
# Problem 4b
def plot_oe_v_time(X_N):
    figs = []
    t, Y = _integrate_one_period(X_N)

    # Compute orbital elements over time
    N = Y.shape[0]
    a = np.empty(N)
    e = np.empty(N)
    inc = np.empty(N)
    argp = np.empty(N)
    raan = np.empty(N)
    tru = np.empty(N)

    # Convert state vector to orbital elements for each timestep
    for k in range(N):
        a[k], e[k], inc[k], argp[k], raan[k], tru[k] = state_to_orbital_elements(Y[k])

    # Set up plots
    fig, axs = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    axs = axs.ravel()
    for ax in axs:
        ax.set_xlabel("time [s]")
        ax.grid(True, ls=":", lw=0.5)

    # Plot semi major axis (a)
    axs[0].plot(t, a)
    axs[0].set_title("Semi-major axis a [km]")
    axs[0].set_ylim(0, 7500)

    # Plot eccentricity (e)
    axs[1].plot(t, e)
    axs[1].set_title("Eccentricity e [-]")
    axs[1].set_ylim(0, 1)

    # Plot inclination (i)
    axs[2].plot(t, inc)
    axs[2].set_title("Inclination i [rad]")
    axs[2].set_ylim(0, 1)

    # Plot argument of perigee (ω)
    axs[3].plot(t, argp)
    axs[3].set_title("Argument of perigee $\\omega$ [rad]")
    axs[3].set_ylim(0, np.pi / 2)

    # Plot right ascension of the ascending node (Ω)
    axs[4].plot(t, raan)
    axs[4].set_title("RAAN $\\Omega$ [rad]")
    axs[4].set_ylim(0, 1)

    # Plot true anomaly (f)
    axs[5].plot(t, tru)
    axs[5].set_title("True anomaly f [rad]")
    # axs[5].set_ylim(0, np.pi*2)

    figs.append(fig)
    return figs


# REQUIRED
# Problem 4c
def discussion():
    return "Due to being a pure two-body point-mass model (with no external forces nor third-body effects), the orbital plots depict a closed elliptical path, with perfectly repeated motion. A closed elliptical path should depict constant orbital elements a, e, i, ω, Ω — which we see in their respective orbital element plots. Such motion should also have steady procession in f wrt. time: beginning at f=f₀, wrapping at f=2π, and continuing its procession until f=f₀ again — which we also see in its respective orbital element plot. Thus, all depicted behavior in the plots makes sense, as they line up with the expected behavior for a pure two-body system."


###############################################
# Main Script to test / debug your code
# This will NOT be run by the autograder!
# The individual functions above will be called and tested
###############################################


def main():
    #############
    # provided
    #############
    r_N_truth = np.array([594.193479, -5881.90168, -4579.29092])
    v_N_truth = np.array([5.97301650, 2.50988687, -2.44880269])
    X_N_truth = np.concatenate((r_N_truth, v_N_truth))
    oe_truth = np.array([6798.1366, 0.1, np.pi / 4, np.pi / 3, np.pi / 4, np.pi])

    #############
    # solutions
    #############
    oe_fake = state_to_orbital_elements(X_N_truth)
    print(oe_fake)

    X_N_fake = orbital_elements_to_state(oe_truth)
    print(X_N_fake)

    plot_orbits(X_N_truth)
    plot_oe_v_time(X_N_truth)
    plt.show()


if __name__ == "__main__":
    main()

import time
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from RL.env import FishGoalEnv, make_torus_mesh, make_sphere_mesh
from GMR.gmr import GMRGMM
from controllers.spacemouse import SpaceMouse3D

from GMR.utils import select_demos, refresh_wireframes
from MPC.LMPC_solver_obs import LinearMPCController
import spatialmath as sm

def compute_terminal_nodes(goal_W, tol=1e-12):
    G = goal_W.shape[0]
    terminal = np.zeros(G, dtype=bool)

    for i in range(G):
        row = goal_W[i]

        # indices with outgoing weight
        outgoing = np.where(row > tol)[0]

        if len(outgoing) == 0:
            # no outgoing edges
            terminal[i] = True

        elif len(outgoing) == 1 and outgoing[0] == i:
            # only self-loop
            terminal[i] = True

    return terminal

def merge_meshes(meshes):
    """
    meshes: list of (verts, faces)
      verts: (Nv,3) float32/float64
      faces: (Nf,3) int32/int64 indices into verts
    returns: (V,F) merged
    """
    V_all = []
    F_all = []
    v_off = 0

    for V, F in meshes:
        V = np.asarray(V, dtype=np.float32)
        F = np.asarray(F, dtype=np.int32)

        V_all.append(V)
        F_all.append(F + v_off)

        v_off += V.shape[0]

    Vm = np.vstack(V_all).astype(np.float32)
    Fm = np.vstack(F_all).astype(np.int32)
    return Vm, Fm

def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

def compute_alpha(
    u_h: np.ndarray,
    v_ref: np.ndarray,
    Sigma: np.ndarray | float,
    *,
    alpha_max: float = 1.0,
    # alignment gate
    c0: float = 0.7,       # cosine threshold: must be "somewhat along"
    k_a: float = 10.0,     # sharpness
    u_deadzone: float = 1e-3,
    # confidence gate (scalar uncertainty s)
    s0: float = 10.0,      # uncertainty threshold (tune to your Sigma scale)
    k_s: float = 10.0,
) -> float:
    """
    Returns alpha in [0, alpha_max]. Increases when:
      - user input aligns with reference direction, AND
      - Sigma is small (confident).
    Drops when either worsens.
    """

    u_h = np.asarray(u_h, dtype=float).reshape(-1)
    v_ref = np.asarray(v_ref, dtype=float).reshape(-1)

    # --- deadzone: if user not pushing, don't assist
    if np.linalg.norm(u_h) < u_deadzone or np.linalg.norm(v_ref) < 1e-12:
        alpha_star = 0.0
    else:
        # alignment cosine in [-1, 1]
        c = float(u_h @ v_ref) / (float(np.linalg.norm(u_h) * np.linalg.norm(v_ref)) + 1e-12)
        c = max(-1.0, min(1.0, c))

        g_align = _sigmoid(k_a * (c - c0))

        # scalar uncertainty s
        if np.isscalar(Sigma):
            s = float(Sigma)
        else:
            Sigma = np.asarray(Sigma, dtype=float)
            # robust choice: sqrt(trace(Sigma))
            s = float(np.sqrt(np.trace(Sigma)))

        g_sigma = _sigmoid(k_s * (s0 - s))  # smaller s -> closer to 1

        alpha_star = alpha_max * g_align

    # clamp
    alpha = max(0.0, min(alpha_max, alpha_star))
    return alpha


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # ----------------------------
    # 1) Load policy + generate fish trajectories
    # ----------------------------
    theta_path = "save/best_policy.pkl"
    action = pickle.load(open(theta_path, "rb"))["best_theta"]

    boid_count = 600
    max_steps = 500
    dt = 1
    t1 = make_torus_mesh(R=3.0, r=1.0, segR=24, segr=16, center=(30.0, 20.0, 20.0))
    t2 = make_torus_mesh(R=3.0, r=1.0, segR=24, segr=16, center=(30.0, 30.0, 20.0))
    t3 = make_torus_mesh(R=3.0, r=1.0, segR=24, segr=16, center=(30.0, 10.0, 20.0))
    s1 = make_sphere_mesh(R=3.0, seg_theta=16, seg_phi=16, center=(15.0, 20.0, 20.0))
    obs_list = [t1, t2, t3, s1]

    verts, faces = merge_meshes(obs_list)

    goals = np.array([
        [25.0, 20.0, 20.0],  # 0 - initial
        [40.0, 20.0, 20.0],  # 1
        [40.0, 30.0, 20.0],  # 2
        [40.0, 10.0, 20.0],  # 3
    ], dtype=np.float32)
    goal_W = np.array([
        [0.0, 1.0, 1.0, 1.0],  # from 0 → {1,2,3}
        [0.0, 1.0, 0.0, 0.0],  # from 1 → 0
        [0.0, 0.0, 1.0, 0.0],  # from 2 → 0
        [0.0, 0.0, 0.0, 1.0],  # from 3 → 0
    ], dtype=np.float32)

    ends_goals = compute_terminal_nodes(goal_W)
    x_start = np.array([6.0, 20.0, 20.0], dtype=np.float32)
    env = FishGoalEnv(boid_count=boid_count, max_steps=max_steps, dt=dt, 
                      doAnimation = False, returnTrajectory = True, verts=verts, faces=faces, goals=goals, goal_W=goal_W, start=x_start)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(action)
    boid_pos = np.array(info["trajectory_boid_pos"])  # (T, N, 3)

    # ----------------------------
    # 2) GMR parameters
    # ----------------------------
    n_demos = 15
    time_stride = 1
    n_components = 6
    cov_type = "full"

    # history settings
    history_len = 8         
    update_period = 0.05      
    update_iters = 10         # EM iters per update
    move_eps = 1e-3           # don't update if you didn't move

    # ----------------------------
    # 3) SpaceMouse
    # ----------------------------
    spm = SpaceMouse3D(trans_scale=10.0, deadzone=0.0, lowpass=0.0, rate_hz=200)
    spm.start()
    x = x_start.astype(float).copy()
    history = [x.copy()] 
    path = [x.copy()]     # full path for plotting

    # ----------------------------
    # 4) Warm-start GMR with initial history
    # ----------------------------
    history_points = np.array([x_start]*history_len)
    pos_demos = select_demos(boid_pos, history_points, time_stride=time_stride, space_stride=1, n_demos=15)

    gmr = GMRGMM(n_components=n_components, seed=0, cov_type=cov_type)
    gmr.fit(pos_demos)
    mu_y, Sigma_y, gamma, loglik = gmr.regress(T=max_steps, pos_dim=3)

    # ----------------------------
    # 5) Plot setup
    # ----------------------------
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # obstacles
    tris = [verts[faces[f]].tolist() for f in range(faces.shape[0])]

    mesh_poly = Poly3DCollection(
        tris,
        alpha=0.25,
        linewidths=0.5,
        edgecolor="k",
        facecolor=(0.6, 0.6, 0.6, 0.25),
    )
    ax.add_collection3d(mesh_poly)

    ax.scatter(x_start[0], x_start[1], x_start[2], s=80, label="Start")
    for goal in goals[ends_goals]:
        ax.scatter(goal[0], goal[1], goal[2], s=80, label="Goal")

    # demos + model
    demo_lines = [ax.plot([], [], [], "k--", lw=1.0, alpha=0.7)[0] for _ in range(n_demos)]
    # mu_line = ax.plot(mu_y[:, 0], mu_y[:, 1], mu_y[:, 2], "k", lw=2.0)[0]

    # live cursor + history path (avoid orange)
    cursor_scatter = ax.scatter(x[0], x[1], x[2], s=70)      # default color
    path_line = ax.plot([x[0]], [x[1]], [x[2]], lw=2.0)[0]
    hist_scatter = ax.scatter([x[0]], [x[1]], [x[2]], s=12, alpha=0.8)

    ax.legend()

    # keep bounds reasonable
    c0 = (x_start + goals[1]) / 2
    span = np.linalg.norm(goals[1] - x_start) / 2 + 2.0
    ax.set_xlim(c0[0] - span, c0[0] + span)
    ax.set_ylim(c0[1] - span, c0[1] + span)
    ax.set_zlim(c0[2] - span, c0[2] + span)
    ax.view_init(elev=20, azim=-180, roll=0)

    def set_demo_lines(demos):
        for i, ln in enumerate(demo_lines):
            if i < len(demos):
                d = demos[i]
                ln.set_data(d[:, 0], d[:, 1])
                ln.set_3d_properties(d[:, 2])
                ln.set_visible(True)
            else:
                ln.set_visible(False)

    # set_demo_lines(pos_demos)

    # ----------------------------
    # 6) Live loop
    # ----------------------------
    last_update_t = time.time()
    last_t = time.time()
    last_x_for_update = x.copy()
    alpha = 0.0
    dt = 20

    speed_limit = 10
    lmpc_solver = LinearMPCController(horizon=10, dt=dt, gamma = 0.05,
                                    u_min=np.array([-speed_limit, -speed_limit, -speed_limit, -speed_limit, -speed_limit, -speed_limit]),
                                    u_max=np.array([ speed_limit,  speed_limit,  speed_limit,  speed_limit,  speed_limit,  speed_limit]))

    def interpolate_traj(p0, p1, n):
        p0 = np.asarray(p0)
        p1 = np.asarray(p1)

        t = np.linspace(0.0, 1.0, n)[:, None]   # shape (n,1)
        traj = p0 + t * (p1 - p0)

        return traj

    try:
        plt.ion()
        while plt.fignum_exists(fig.number):
            now = time.time()
            trans, rot, buttons = spm.read() 
            v = np.array(trans, dtype=float)

            # Compute autonomy
            d = np.linalg.norm(mu_y - x[None, :], axis=1)
            tidx = int(np.argmin(d))
            if tidx < max_steps -lmpc_solver.horizon:
                v_ref = (mu_y[tidx + 1] - mu_y[tidx]) / dt
                traj = interpolate_traj(mu_y[tidx], mu_y[tidx + lmpc_solver.horizon], lmpc_solver.horizon)
            else:
                v_ref = np.zeros(3)
                traj = None
            if np.linalg.norm(v_ref) > 1e-3:
                v_ref *= np.linalg.norm(v) / np.linalg.norm(v_ref)
            sigma = Sigma_y[tidx]
            alpha = compute_alpha(v, v_ref, sigma)

            T_i = sm.SE3.Trans(x)
            T_des_human = sm.SE3.Trans(x + v * dt)
            T_des_GMR = sm.SE3.Trans(x + v_ref * dt)

            Uopt, Xopt, poses = lmpc_solver.solve(T_i, T_des_human, T_des_GMR, 1 - alpha, obstacles=obs_list, traj=traj, margin=0.5)
            
            if Uopt is None:
                vopt = np.zeros(3)
            else:
                vopt = Uopt[0:3]
            if np.linalg.norm(vopt) > 1e-3:
                vopt *= np.linalg.norm(v) / np.linalg.norm(vopt)

            x = x + vopt * dt

            if np.linalg.norm(x - history[-1]) > 1e-6:
                history.append(x.copy())
                if len(history) > history_len:
                    history = history[-history_len:]
                path.append(x.copy())

            cursor_scatter._offsets3d = ([x[0]], [x[1]], [x[2]])

            P = np.array(path)
            path_line.set_data(P[:, 0], P[:, 1])
            path_line.set_3d_properties(P[:, 2])

            H = np.array(history)
            hist_scatter._offsets3d = (H[:, 0], H[:, 1], H[:, 2])

            # throttled GMR update
            if (now - last_update_t) >= update_period and np.linalg.norm(x - last_x_for_update) > move_eps:
                last_update_t = now
                last_x_for_update = x.copy()
                
                pos_demos = select_demos(boid_pos, history, time_stride=time_stride, n_demos=15)

                # update + regress
                gmrUpdate = copy.deepcopy(gmr)
                gmrUpdate.update(pos_demos, n_iter=update_iters)
                mu_y, Sigma_y, gamma, loglik = gmrUpdate.regress(T=max_steps, pos_dim=3)

            plt.pause(0.001)

            if np.linalg.norm(buttons) > 0.5: # Restart by pressing a button
                x = x_start.astype(float).copy()
                history = [x.copy()] 
                path = [x.copy()] 

    finally:
        try:
            spm.stop()
        except Exception:
            pass
        plt.ioff()
        plt.show()
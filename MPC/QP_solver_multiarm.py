import numpy as np
import scipy.sparse as sp
from qpsolvers import solve_qp


class MultiArmQPController:
    """
    Multi-arm extension of QPController (joint-space decision variables).
    Keeps the same 3-call architecture:
        update_robot_state(...)
        add_local_tangent_plane_constraints(...)
        solve(...)

    Decision variable:
        qdot_all = [qdot_0; qdot_1; ...] in R^(sum n_i)

    Objective (per arm i):
        || W (J_i qdot_i - xdot_i) ||^2  + alpha * ||N_i qdot_i||^2
    (NO manipulability term)

    Constraints:
      - per-arm joint position/velocity bounds (lb/ub)
      - per-arm floor constraint (like QPController.add_floor_constraint)
      - obstacle avoidance:
          (A) tangent-plane constraints reusing your existing idea (fast, linear)
      - inter-arm CBF constraints (pairwise):
          h = ||p_i - p_j||^2 - d_safe^2
          dh + gamma*h >= 0  with pdot = Jp qdot
    """

    def __init__(self, robots, dt=0.05):
        self.dt = float(dt)

        self.robots = list(robots)
        self.M = len(self.robots)
        assert self.M >= 1

        self.n_dofs = [r.n for r in self.robots]
        self.offsets = np.cumsum([0] + self.n_dofs[:-1])
        self.n_tot = int(sum(self.n_dofs))

        # per-arm state caches
        self.qs = [r.q.copy() for r in self.robots]
        self.qds = [r.qd.copy() for r in self.robots]

        # limits (copied from your QPController defaults)
        self.qlims = [
            np.vstack([r.qlim[0, :] + 0.01, r.qlim[1, :] - 0.01]) for r in self.robots
        ]
        self.qdlims = [r.qdlim.copy() for r in self.robots]

        # QP matrices
        self.H = np.eye(self.n_tot)
        self.g = np.zeros(self.n_tot)

        self.A = np.zeros((0, self.n_tot))
        self.b = np.zeros((0,))

        self.eqA = np.zeros((0, self.n_tot))
        self.eqb = np.zeros((0,))

        self.lb = -np.inf * np.ones(self.n_tot)
        self.ub = +np.inf * np.ones(self.n_tot)

        self.solution = None  # warm start (like your single-arm QP) :contentReference[oaicite:5]{index=5}

        # optional: store slack from last CBF solve if you add it later
        self._last_debug = {}

    # -----------------------
    # API: same architecture
    # -----------------------
    def update_robot_state(self, robots_or_list):
        """
        Accept either:
          - a list of robot models (length M)
          - or a single robot model when M==1

        In your sim, you'd pass [panda0, panda1, ...].
        """
        if self.M == 1 and not isinstance(robots_or_list, (list, tuple)):
            robots = [robots_or_list]
        else:
            robots = list(robots_or_list)

        assert len(robots) == self.M

        self.robots = robots
        for i, r in enumerate(robots):
            self.qs[i] = r.q.copy()
            self.qds[i] = r.qd.copy()

    def solve(self, xdots, alpha=0.02, W=None, damping=1e-6, z_floor=0.0, floor_margin=0.02, eps=1e-6):
        """
        Keep beta argument for compatibility with your current call site,
        but it's ignored (manipulability removed). :contentReference[oaicite:6]{index=6}

        xdots: (M,6) or flat (6,) if M==1
        """
        if self.M == 1 and np.asarray(xdots).shape == (6,):
            xdots = np.asarray(xdots, float).reshape(1, 6)
        else:
            xdots = np.asarray(xdots, float).reshape(self.M, 6)

        if W is None:
            W = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        W = np.asarray(W, float).reshape(6, 6)

        if np.isscalar(eps):
            eps = np.ones(self.M) * float(eps)

        # build objective blocks + bounds
        H_blocks = []
        g_all = []

        lb_all = []
        ub_all = []

        for i in range(self.M):
            Hi, gi = self._build_single_arm_objective(i, xdots[i], alpha=alpha, W=W, damping=damping, eps=eps[i])
            H_blocks.append(Hi)
            g_all.append(gi)

            lbi, ubi = self._joint_velocity_bounds(i, self.qlims[i])
            lb_all.append(lbi)
            ub_all.append(ubi)

        self.H = self._blockdiag_dense(H_blocks)
        self.g = np.concatenate(g_all, axis=0)

        self.lb = np.concatenate(lb_all, axis=0)
        self.ub = np.concatenate(ub_all, axis=0)

        # floor constraints per arm (same logic as QPController.add_floor_constraint) :contentReference[oaicite:7]{index=7}
        for i in range(self.M):
            self._add_floor_constraint_arm(i, z_floor=z_floor, margin=floor_margin)

        # solve QP
        x = solve_qp(
            sp.csc_matrix(self.H),
            self.g,
            G=sp.csc_matrix(self.A),
            h=self.b,
            A=sp.csc_matrix(self.eqA),
            b=self.eqb,
            lb=self.lb,
            ub=self.ub,
            solver="osqp",
            initvals=self.solution,
        )
        self.solution = x
        self.reset_constraints()

        # return per-arm splits (handy)
        if x is None:
            return None

        return self.split_solution(x)

    def add_local_tangent_plane_constraints(
        self,
        obstacles,
        margin=0.05,
        # tangent-plane thinning params (copied style from your optimized version) :contentReference[oaicite:8]{index=8}
        gap_threshold=0.03,
        max_constraints_per_obstacle=40,
        qdot_max_scalar=1.0,
        # inter-arm CBF params
        add_interarm_cbf=True,
        d_safe=0.25,
        gamma_pair=2.0,
        # which points to use for inter-arm CBF
        interarm_use_ee_only=True,
    ):
        """
        We do:
          (1) per-arm tangent-plane vs obstacles
          (2) optional inter-arm CBF constraints
        """
        obstacles = list(obstacles) if obstacles is not None else []

        # (1) Tangent planes per arm (fast, linear)
        for i in range(self.M):
            self._add_tangent_planes_for_arm(
                i,
                obstacles,
                margin=margin,
                gap_threshold=gap_threshold,
                max_constraints_per_obstacle=max_constraints_per_obstacle,
                qdot_max_scalar=qdot_max_scalar,
            )

        # (2) Inter-arm CBF constraints (pairwise)
        if add_interarm_cbf and self.M >= 2 and gamma_pair > 0.0:
            if interarm_use_ee_only:
                self._add_interarm_cbf_ee_only(d_safe=d_safe, gamma=gamma_pair)
            else:
                # optional extension: use a small set of tool-frame points
                self._add_interarm_cbf_tool_points(d_safe=d_safe, gamma=gamma_pair)

    # -----------------------
    # internals: objective
    # -----------------------
    def _build_single_arm_objective(self, arm_idx, xdot, alpha, W, damping, eps):
        """
        Cost:
            || W (J qdot - xdot) ||^2 + alpha * ||N qdot||^2
        Secondary task:
            keep elbow at 0 for joint 2 (index 1)
        """
        r = self.robots[arm_idx]
        q = self.qs[arm_idx]

        n = r.n
        I = np.eye(n)

        # Jacobian in end-effector frame (like your code uses jacobe) :contentReference[oaicite:11]{index=11}
        J = np.asarray(r.jacobe(q), dtype=float)  # (6,n)

        # weighted damped pinv (same structure as your helper) :contentReference[oaicite:12]{index=12}
        WTW = (W.T @ W)
        JT_WTW_J = J.T @ WTW @ J
        inv = np.linalg.inv(JT_WTW_J + damping * np.eye(n))
        Jpinv = inv @ J.T @ WTW

        N = I - Jpinv @ J

        # secondary desired joint velocities (elbow toward 0)
        qdot_des = np.zeros(n)
        qdot_des[1] = -q[1] / self.dt

        # Build quadratic form:
        # g = -2 xdot^T WTW J  -2 alpha qdot_des^T (N^T N)
        # H = 2 (J^T WTW J + alpha N^T N)
        g = (-2.0 * (xdot.T @ WTW @ J) - 2.0 * alpha * (qdot_des.T @ (N.T @ N))).reshape(-1)
        H = 2.0 * (J.T @ WTW @ J + alpha * (N.T @ N)) + eps * np.eye(n)

        return H, g

    def _joint_velocity_bounds(self, arm_idx, qlim):
        """
        Same logic as QPController.update_joints_limits: convert position limits to velocity bounds,
        then clip with qdlim. :contentReference[oaicite:13]{index=13}
        """
        q = self.qs[arm_idx]
        qdlim = self.qdlims[arm_idx][: self.n_dofs[arm_idx]]

        lb = 0.1 * (qlim[0, :] - q) / self.dt
        ub = 0.1 * (qlim[1, :] - q) / self.dt
        lb = np.maximum(lb, -qdlim)
        ub = np.minimum(ub, +qdlim)
        return lb, ub

    # -----------------------
    # internals: constraints
    # -----------------------
    def add_constraint_global(self, row, b):
        row = np.asarray(row, float).reshape(1, self.n_tot)
        self.A = np.vstack((self.A, row))
        self.b = np.hstack((self.b, float(b)))

    def reset_constraints(self):
        self.A = np.zeros((0, self.n_tot))
        self.b = np.zeros((0,))
        self.eqA = np.zeros((0, self.n_tot))
        self.eqb = np.zeros((0,))

    def _add_floor_constraint_arm(self, arm_idx, z_floor=0.0, margin=0.02):
        r = self.robots[arm_idx]
        q = self.qs[arm_idx]
        z = r.fkine(q).t[2]
        J = np.asarray(r.jacobe(q), dtype=float)  # (6,n)
        Jz = J[2, :].reshape(1, -1)

        required = (z_floor + margin - z) / float(self.dt)
        # In your single-arm code: adds G=Jz and h=[-required] :contentReference[oaicite:15]{index=15}
        # Here we keep the same.
        row_local = Jz.reshape(-1)
        row_global = self._embed_row(arm_idx, row_local)
        self.add_constraint_global(row_global, -required)

    def _add_tangent_planes_for_arm(
        self,
        arm_idx,
        obstacles,
        margin,
        gap_threshold,
        max_constraints_per_obstacle,
        qdot_max_scalar,
    ):
        if obstacles is None or len(obstacles) == 0:
            return

        rbt = self.robots[arm_idx]
        q = self.qs[arm_idx]
        n = self.n_dofs[arm_idx]

        # tool contact grid (same idea as your optimized function) :contentReference[oaicite:17]{index=17}
        palm_y_min, palm_y_max = -0.12, 0.12
        palm_z_min, palm_z_max = -0.08, -0.04
        palm_res = 12  # smaller than 20 to keep it light across many arms

        fingertip_z = 0.0
        fingertip_res = 12

        ys = np.linspace(palm_y_min, palm_y_max, palm_res)
        zs = np.linspace(palm_z_min, palm_z_max, palm_res)
        palm = np.array([[0.0, y, z] for y in ys for z in zs])

        yf = np.linspace(-0.01, 0.01, fingertip_res)
        zf = np.linspace(-0.015, 0.015, fingertip_res)
        tip = np.array([[0.0, y, fingertip_z + z] for y in yf for z in zf])

        franka_local = np.vstack((palm, tip))  # (N,3)
        Np = franka_local.shape[0]

        # FK + Jacobian once :contentReference[oaicite:18]{index=18}
        T_ee = rbt.fkine(q)
        R_ee, p_ee = T_ee.R, T_ee.t
        J0 = np.asarray(rbt.jacobe(q), dtype=float)
        Jv = J0[0:3, :]
        Jw = J0[3:6, :]

        # world positions
        p_world = p_ee + (R_ee @ franka_local.T).T  # (Np,3)
        r_off = p_world - p_ee                       # (Np,3)

        # batch skew
        S = np.zeros((Np, 3, 3))
        S[:, 0, 1] = -r_off[:, 2]
        S[:, 0, 2] =  r_off[:, 1]
        S[:, 1, 0] =  r_off[:, 2]
        S[:, 1, 2] = -r_off[:, 0]
        S[:, 2, 0] = -r_off[:, 1]
        S[:, 2, 1] =  r_off[:, 0]

        # Jp = Jv - skew(r)*Jw :contentReference[oaicite:19]{index=19}
        Jp = Jv[None, :, :] - np.einsum("nij,jk->nik", S, Jw)  # (Np,3,n)

        for obs in obstacles:
            c = np.asarray(obs["center"], float).reshape(3)
            Rinfl = float(obs["radius"]) + abs(margin)

            vec = c[None, :] - p_world           # (Np,3)
            dist = np.linalg.norm(vec, axis=1)   # (Np,)
            gap = dist - Rinfl

            near_mask = gap < gap_threshold
            if not np.any(near_mask):
                continue

            idx = np.where(near_mask)[0]
            # keep most "dangerous" points
            if idx.size > max_constraints_per_obstacle:
                order = np.argsort(gap[idx])
                idx = idx[order[:max_constraints_per_obstacle]]

            for k in idx:
                v = c - p_world[k]
                d = np.linalg.norm(v)
                if d < 1e-9:
                    s = np.array([1.0, 0.0, 0.0])
                else:
                    s = v / d

                o_prime = c - Rinfl * s

                A_row_local = (s @ Jp[k])  # (n,)
                b_scalar = np.linalg.norm(o_prime - p_world[k]) / (self.dt * 2.0)  # same scaling as your code :contentReference[oaicite:20]{index=20}

                A_row_global = self._embed_row(arm_idx, A_row_local)
                self.add_constraint_global(A_row_global, b_scalar)

    def _add_interarm_cbf_ee_only(self, d_safe=0.25, gamma=2.0):
        """
        Inter-arm CBF using only EE positions (cheap).
        Constraint:
          h = ||p_i - p_j||^2 - d_safe^2
          dh = 2(p_i-p_j)^T (Jpi qdoti - Jpj qdotj)
          dh + gamma*h >= 0
        -> linear inequality in stacked qdot.
        """
        d_safe = float(d_safe)
        gamma = float(gamma)

        # precompute ee positions and Jp (translation Jacobian at ee) per arm
        ps = []
        Jps = []
        for i in range(self.M):
            r = self.robots[i]
            q = self.qs[i]
            T = r.fkine(q)
            ps.append(np.asarray(T.t, float).reshape(3))
            J0 = np.asarray(r.jacobe(q), float)
            Jps.append(J0[0:3, :])  # translation part (3,n)

        for i in range(self.M):
            for j in range(i + 1, self.M):
                d = ps[i] - ps[j]
                h_val = float(d @ d - d_safe * d_safe)

                # if far enough, skip (optional speed)
                if h_val > (0.5 * d_safe * d_safe):
                    continue

                # build global row:
                # -2 d^T Jpi * qdoti  + 2 d^T Jpj * qdotj <= gamma * h
                ai = (-2.0 * d.reshape(1, 3)) @ Jps[i]  # (1,ni)
                aj = (+2.0 * d.reshape(1, 3)) @ Jps[j]  # (1,nj)

                row = np.zeros(self.n_tot)
                oi = self.offsets[i]
                oj = self.offsets[j]
                row[oi : oi + self.n_dofs[i]] = ai.reshape(-1)
                row[oj : oj + self.n_dofs[j]] = aj.reshape(-1)

                self.add_constraint_global(row, gamma * h_val)

    def _add_interarm_cbf_tool_points(self, d_safe=0.25, gamma=2.0):
        """
        Optional extension: use 2-3 tool-frame points per arm.
        Keep it small to avoid too many constraints.
        """
        tool_pts = [
            np.array([0.0, 0.0,  0.0]),
            np.array([0.0, 0.2, 0.2]), 
            np.array([0.0, 0.2, -0.2]),
            np.array([0.0, -0.2, 0.2]), 
            np.array([0.0, -0.2, -0.2]),
        ]

        # Precompute world positions + Jp for each arm and each point
        Pw = []
        Jp = []
        for i in range(self.M):
            r = self.robots[i]
            q = self.qs[i]
            T = r.fkine(q)
            R_ee, p_ee = T.R, T.t
            J0 = np.asarray(r.jacobe(q), float)
            Jv = J0[0:3, :]
            Jw = J0[3:6, :]

            Pi = []
            Ji = []
            for pl in tool_pts:
                p_world = p_ee + R_ee @ pl
                r_off = (p_world - p_ee).reshape(3)
                S = np.array([[0, -r_off[2], r_off[1]],
                              [r_off[2], 0, -r_off[0]],
                              [-r_off[1], r_off[0], 0]], float)
                Jp_pt = Jv - S @ Jw
                Pi.append(p_world)
                Ji.append(Jp_pt)

            Pw.append(Pi)
            Jp.append(Ji)

        # Pairwise constraints (only nearest point pairs)
        for i in range(self.M):
            for j in range(i + 1, self.M):
                # find closest point pair
                best = None
                for a in range(len(tool_pts)):
                    for b in range(len(tool_pts)):
                        dvec = Pw[i][a] - Pw[j][b]
                        dist2 = float(dvec @ dvec)
                        if best is None or dist2 < best[0]:
                            best = (dist2, a, b, dvec)

                dist2, a, b, dvec = best
                h_val = dist2 - float(d_safe * d_safe)
                if h_val > (0.5 * d_safe * d_safe):
                    continue

                ai = (-2.0 * dvec.reshape(1, 3)) @ Jp[i][a]
                aj = (+2.0 * dvec.reshape(1, 3)) @ Jp[j][b]

                row = np.zeros(self.n_tot)
                oi = self.offsets[i]
                oj = self.offsets[j]
                row[oi : oi + self.n_dofs[i]] = ai.reshape(-1)
                row[oj : oj + self.n_dofs[j]] = aj.reshape(-1)

                self.add_constraint_global(row, gamma * float(h_val))

    # -----------------------
    # utils
    # -----------------------
    def _embed_row(self, arm_idx, row_local):
        row = np.zeros(self.n_tot)
        o = self.offsets[arm_idx]
        n = self.n_dofs[arm_idx]
        row[o : o + n] = np.asarray(row_local, float).reshape(n)
        return row

    @staticmethod
    def _blockdiag_dense(blocks):
        # blocks: list of (ni,ni)
        n = sum(b.shape[0] for b in blocks)
        out = np.zeros((n, n))
        k = 0
        for b in blocks:
            m = b.shape[0]
            out[k : k + m, k : k + m] = b
            k += m
        return out

    def split_solution(self, x):
        """Return list of qdot_i arrays."""
        outs = []
        for i in range(self.M):
            o = self.offsets[i]
            outs.append(np.asarray(x[o : o + self.n_dofs[i]]).copy())
        return outs

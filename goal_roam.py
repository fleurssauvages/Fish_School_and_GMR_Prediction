from __future__ import annotations

import math
import numpy as np
import os
os.environ.setdefault("NUMBA_DISABLE_COVERAGE", "1")
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons


# -----------------------------
# Helper math (Numba-friendly)
# -----------------------------
@njit(cache=True)
def _norm3(v0, v1, v2):
    return math.sqrt(v0 * v0 + v1 * v1 + v2 * v2)


@njit(cache=True)
def _clamp_len(v0, v1, v2, max_len):
    n = _norm3(v0, v1, v2)
    if n <= 1e-12:
        return 0.0, 0.0, 0.0
    if n > max_len:
        s = max_len / n
        return v0 * s, v1 * s, v2 * s
    return v0, v1, v2


@njit(cache=True)
def _set_len(v0, v1, v2, new_len):
    n = _norm3(v0, v1, v2)
    if n <= 1e-12:
        return 0.0, 0.0, 0.0
    s = new_len / n
    return v0 * s, v1 * s, v2 * s


@njit(cache=True)
def _lcg_rand01(seed_arr):
    # linear congruential generator, matching JS constants:
    # m = 2^32, a = 1664525, c = 1
    m = 4294967296.0
    a = 1664525.0
    c = 1.0
    seed = seed_arr[0]
    seed = (a * seed + c) % m
    seed_arr[0] = seed
    return seed / m


@njit(cache=True)
def _cubic_interpolate(v0, v1, v2, v3, x):
    # Paul Breeuwsma coefficients (same as JS)
    x2 = x * x
    a0 = -0.5 * v0 + 1.5 * v1 - 1.5 * v2 + 0.5 * v3
    a1 = v0 - 2.5 * v1 + 2.0 * v2 - 0.5 * v3
    a2 = -0.5 * v0 + 0.5 * v2
    a3 = v1
    return a0 * x * x2 + a1 * x2 + a2 * x + a3


@njit(cache=True)
def _noise(time, cum_wavlen, rv0, rv1, rv2, rv3, seed_arr):
    wavelen = 0.3
    if time >= cum_wavlen:
        # advance one wavelen segment
        cum_wavlen = cum_wavlen + wavelen
        rv0, rv1, rv2 = rv1, rv2, rv3
        rv3 = _lcg_rand01(seed_arr)

    frac = (time % wavelen) / wavelen
    value = _cubic_interpolate(rv0, rv1, rv2, rv3, frac)
    return (value * 2.0 - 1.0), cum_wavlen, rv0, rv1, rv2, rv3


# -----------------------------
# Parameter packing
# -----------------------------
# p indices
P_BOUND = 0
P_PLAY_SPEED = 1

P_BOID_COUNT = 2
P_PRED_COUNT = 3

P_RULE_SCALAR = 4
P_RULE_SCALAR_P = 5

P_MAX_SPEED = 6
P_MAX_SPEED_P = 7

P_SEP_R = 8
P_ALI_R = 9
P_COH_R = 10
P_PRED_AVOID_R = 11
P_OBS_AVOID_R = 12

P_SEP_S = 13
P_ALI_S = 14
P_COH_S = 15
P_BND_S = 16
P_RAND_S = 17
P_PRED_AVOID_S = 18
P_OBS_AVOID_S = 19
P_ATTACK_S = 20

P_RANDOM_WAVELEN_SCALAR = 21

P_ENABLE_SEP = 22
P_ENABLE_ALI = 23
P_ENABLE_COH = 24
P_ENABLE_BND = 25
P_ENABLE_RAND = 26
P_ENABLE_PRED_AVOID = 27
P_ENABLE_OBS_AVOID = 28

P_OBS_CX = 29
P_OBS_CY = 30
P_OBS_CZ = 31
P_OBS_RADIUS = 32

P_ENABLE_GOAL = 33
P_GOAL_GAIN = 34
P_GOAL_X = 35
P_GOAL_Y = 36
P_GOAL_Z = 37

P_OBS2_CX = 38
P_OBS2_CY = 39
P_OBS2_CZ = 40
P_OBS3_CX = 41
P_OBS3_CY = 42
P_OBS3_CZ = 43

P_START_X = 44
P_START_Y = 45
P_START_Z = 46
P_START_SPREAD = 47
P_MAX_STEPS = 48

P_OBS_GAIN = 49

def make_default_params(boid_count=400, pred_count=8):
    # These are "good-looking" defaults close-ish to the original demo.
    p = np.zeros(50, dtype=np.float64)
    p[P_BOUND] = 40.0
    p[P_PLAY_SPEED] = 1.0

    p[P_BOID_COUNT] = float(boid_count)
    p[P_PRED_COUNT] = float(pred_count)

    p[P_RULE_SCALAR] = 1.0
    p[P_RULE_SCALAR_P] = 1.0

    p[P_MAX_SPEED] = 0.18
    p[P_MAX_SPEED_P] = 0.28

    p[P_SEP_R] = 1.6
    p[P_ALI_R] = 4.0
    p[P_COH_R] = 5.5
    p[P_PRED_AVOID_R] = 6.0
    p[P_OBS_AVOID_R] = 4.0

    p[P_SEP_S] = 1.0
    p[P_ALI_S] = 1.0
    p[P_COH_S] = 1.0
    p[P_BND_S] = 1.0
    p[P_RAND_S] = 1.0
    p[P_PRED_AVOID_S] = 1.0
    p[P_OBS_AVOID_S] = 1.0
    p[P_ATTACK_S] = 1.0

    p[P_RANDOM_WAVELEN_SCALAR] = 1.0

    p[P_ENABLE_SEP] = 1.0
    p[P_ENABLE_ALI] = 1.0
    p[P_ENABLE_COH] = 1.0
    p[P_ENABLE_BND] = 1.0
    p[P_ENABLE_RAND] = 1.0
    p[P_ENABLE_PRED_AVOID] = 1.0
    p[P_ENABLE_OBS_AVOID] = 1.0

    # Spherical obstacle (simplified vs mesh field)
    p[P_OBS_CX] = 20.0
    p[P_OBS_CY] = 20.0
    p[P_OBS_CZ] = 20.0
    p[P_OBS_RADIUS] = 3.5
    p[P_OBS_GAIN] = 1.0


    # Optional goal (fish attracted to a point)
    p[P_ENABLE_GOAL] = 1.0
    p[P_GOAL_GAIN] = 0.0  # slider 0..2
    # A sensible default goal point (can be changed in UI later if desired)
    p[P_GOAL_X] = 34.0  # far from start
    p[P_GOAL_Y] = 20.0
    p[P_GOAL_Z] = 20.0

    # Additional spherical obstacles (share the same radius slider)
    p[P_OBS2_CX] = 12.0
    p[P_OBS2_CY] = 28.0
    p[P_OBS2_CZ] = 22.0

    p[P_OBS3_CX] = 28.0
    p[P_OBS3_CY] = 14.0
    p[P_OBS3_CZ] = 26.0


    # Episode start + max steps
    p[P_START_X] = 6.0  # start
    p[P_START_Y] = 20.0
    p[P_START_Z] = 20.0
    p[P_START_SPREAD] = 2.5  # initial cloud radius
    p[P_MAX_STEPS] = 2000.0


    return p


# -----------------------------
# Core rules (Numba)
# -----------------------------
@njit(cache=True)
def _bounds_steer(px, py, pz, bound_size):
    min_b = 0.0
    max_b = bound_size
    sx = 0.0
    sy = 0.0
    sz = 0.0

    if px < min_b:
        sx = min_b - px
    elif px > max_b:
        sx = max_b - px

    if py < min_b:
        sy = (min_b - py) * 2.0
    elif py > max_b:
        sy = max_b - py

    if pz < min_b:
        sz = min_b - pz
    elif pz > max_b:
        sz = max_b - pz

    sy = sy * 2.0  # matches JS
    return sx, sy, sz


@njit(cache=True)
def _obstacle_avoid(px, py, pz, cx, cy, cz, sphere_r, avoid_r, gain):
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    d = _norm3(dx, dy, dz)

    influence = sphere_r + avoid_r
    if d <= 1e-9 or d >= influence:
        return 0.0, 0.0, 0.0

    # normalized proximity in [0,1]
    x = 1.0 - d / influence

    alpha = 4.0  # steepness (2–6 good range)
    mag = gain * (math.exp(alpha * x) - 1.0)

    ux = dx / d
    uy = dy / d
    uz = dz / d
    return ux * mag, uy * mag, uz * mag


@njit(cache=True)
def _reynolds(i, pos, vel, count, sep_r, ali_r, coh_r):
    # "else" branch in JS (not commonReynolds): weighted by (1 - dist/radius)
    px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]

    sep0 = sep1 = sep2 = 0.0
    ali0 = ali1 = ali2 = 0.0
    coh0 = coh1 = coh2 = 0.0

    for j in range(count):
        if j == i:
            continue
        dx = px - pos[j, 0]
        dy = py - pos[j, 1]
        dz = pz - pos[j, 2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d <= 1e-12:
            continue

        if d < sep_r:
            mag = 1.0 - d / sep_r
            # set length to mag => multiply unit vector
            sep0 += (dx / d) * mag
            sep1 += (dy / d) * mag
            sep2 += (dz / d) * mag

        if d < ali_r:
            mag = 1.0 - d / ali_r
            vx, vy, vz = vel[j, 0], vel[j, 1], vel[j, 2]
            vn = _norm3(vx, vy, vz)
            if vn > 1e-12:
                ali0 += (vx / vn) * mag
                ali1 += (vy / vn) * mag
                ali2 += (vz / vn) * mag

        if d < coh_r:
            # cohesion: towards neighbor (pos_j - pos_i)
            mag = 1.0 - d / coh_r
            coh0 += (-dx / d) * mag
            coh1 += (-dy / d) * mag
            coh2 += (-dz / d) * mag

    # clamp each to length <= 1 (JS clampLength(0,1))
    sep0, sep1, sep2 = _clamp_len(sep0, sep1, sep2, 1.0)
    ali0, ali1, ali2 = _clamp_len(ali0, ali1, ali2, 1.0)
    coh0, coh1, coh2 = _clamp_len(coh0, coh1, coh2, 1.0)
    return sep0, sep1, sep2, ali0, ali1, ali2, coh0, coh1, coh2


@njit(cache=True)
def _predator_avoid(i, pos, predators_pos, pred_count, avoid_r):
    px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]
    sx = sy = sz = 0.0

    for j in range(pred_count):
        dx = px - predators_pos[j, 0]
        dy = py - predators_pos[j, 1]
        dz = pz - predators_pos[j, 2]
        d = _norm3(dx, dy, dz)
        if d < avoid_r and d > 1e-12:
            mag = 1.0 - d / avoid_r
            sx += (dx / d) * mag
            sy += (dy / d) * mag
            sz += (dz / d) * mag

    sx, sy, sz = _clamp_len(sx, sy, sz, 1.0)
    return sx, sy, sz


@njit(cache=True)
def _velocity_attack_step(pred_i, predators_pos, predators_vel, pred_own_time,
                          pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
                          boid_pos, boid_count):
    """
    JS velocityAttack(): modifies predator state (rest cycles + choose closest prey),
    returns a steering vector to be ADDED to predator velocity (not acceleration).
    """
    rest_time = 0.2
    attack_time = 1.4

    t = pred_own_time[pred_i]

    if pred_rest[pred_i] == 1:
        if (t - pred_rest_start[pred_i]) > rest_time:
            pred_rest[pred_i] = 0
            pred_attack_start[pred_i] = t

            # choose closest prey
            px, py, pz = predators_pos[pred_i, 0], predators_pos[pred_i, 1], predators_pos[pred_i, 2]
            closest = 0
            closest_dist = 1e18
            for k in range(boid_count):
                dx = px - boid_pos[k, 0]
                dy = py - boid_pos[k, 1]
                dz = pz - boid_pos[k, 2]
                d = _norm3(dx, dy, dz)
                if d < closest_dist:
                    closest_dist = d
                    closest = k
            pred_prey_idx[pred_i] = closest

        return 0.0, 0.0, 0.0

    prey = pred_prey_idx[pred_i]
    if prey < 0 or prey >= boid_count:
        # safety fallback
        pred_rest[pred_i] = 1
        pred_rest_start[pred_i] = t
        pred_prey_idx[pred_i] = -1
        return 0.0, 0.0, 0.0

    dx = boid_pos[prey, 0] - predators_pos[pred_i, 0]
    dy = boid_pos[prey, 1] - predators_pos[pred_i, 1]
    dz = boid_pos[prey, 2] - predators_pos[pred_i, 2]
    dist = _norm3(dx, dy, dz)

    speed_up_time = t - pred_attack_start[pred_i]
    if speed_up_time > attack_time or dist < 1.0:
        pred_rest[pred_i] = 1
        pred_rest_start[pred_i] = t
        pred_prey_idx[pred_i] = -1
        return 0.0, 0.0, 0.0

    # diff.normalize()
    if dist > 1e-12:
        ux, uy, uz = dx / dist, dy / dist, dz / dist
    else:
        ux, uy, uz = 0.0, 0.0, 0.0

    # diff.sub(predator.velocity)  (steer velocity direction)
    vx, vy, vz = predators_vel[pred_i, 0], predators_vel[pred_i, 1], predators_vel[pred_i, 2]
    ux -= vx
    uy -= vy
    uz -= vz

    # diff.multiplyScalar(Math.pow(speedUpTime, 8))
    s = speed_up_time ** 8
    ux *= s
    uy *= s
    uz *= s

    # diff.clampLength(0, 0.01)
    ux, uy, uz = _clamp_len(ux, uy, uz, 0.01)
    return ux, uy, uz


@njit(cache=True)
def step_sim(boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
             pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
             pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
             seed_arr, dt, p):
    """
    One simulation step (roughly equivalent to moveBoids(delta) in JS).
    All arrays are updated in-place.
    """
    bound_size = p[P_BOUND]
    play_speed = p[P_PLAY_SPEED]

    boid_count = int(p[P_BOID_COUNT])
    pred_count = int(p[P_PRED_COUNT])

    rule_scalar = p[P_RULE_SCALAR]
    rule_scalar_p = p[P_RULE_SCALAR_P]
    max_speed = p[P_MAX_SPEED]
    max_speed_p = p[P_MAX_SPEED_P]

    sep_r = p[P_SEP_R]
    ali_r = p[P_ALI_R]
    coh_r = p[P_COH_R]
    pred_avoid_r = p[P_PRED_AVOID_R]
    obs_avoid_r = p[P_OBS_AVOID_R]
    obs_gain = p[P_OBS_GAIN]

    sep_s = p[P_SEP_S]
    ali_s = p[P_ALI_S]
    coh_s = p[P_COH_S]
    bnd_s = p[P_BND_S]
    rand_s = p[P_RAND_S]
    pred_avoid_s = p[P_PRED_AVOID_S]
    obs_avoid_s = p[P_OBS_AVOID_S]
    attack_s = p[P_ATTACK_S]

    rand_wavelen_scalar = p[P_RANDOM_WAVELEN_SCALAR]

    en_sep = p[P_ENABLE_SEP] > 0.5
    en_ali = p[P_ENABLE_ALI] > 0.5
    en_coh = p[P_ENABLE_COH] > 0.5
    en_bnd = p[P_ENABLE_BND] > 0.5
    en_rand = p[P_ENABLE_RAND] > 0.5
    en_pred_avoid = p[P_ENABLE_PRED_AVOID] > 0.5
    en_obs_avoid = p[P_ENABLE_OBS_AVOID] > 0.5

    en_goal = p[P_ENABLE_GOAL] > 0.5
    goal_gain = p[P_GOAL_GAIN]
    goal_x = p[P_GOAL_X]
    goal_y = p[P_GOAL_Y]
    goal_z = p[P_GOAL_Z]


    obs_cx = p[P_OBS_CX]
    obs_cy = p[P_OBS_CY]
    obs_cz = p[P_OBS_CZ]
    obs2_cx = p[P_OBS2_CX]
    obs2_cy = p[P_OBS2_CY]
    obs2_cz = p[P_OBS2_CZ]
    obs3_cx = p[P_OBS3_CX]
    obs3_cy = p[P_OBS3_CY]
    obs3_cz = p[P_OBS3_CZ]
    obs_radius = p[P_OBS_RADIUS]

    if play_speed == 0.0 or (max_speed == 0.0 and max_speed_p == 0.0):
        return

    play_delta = play_speed * dt * 100.0
    if play_delta > 1000.0:
        play_delta = 30.0

    # --- boids ---
    for i in range(boid_count):
        boid_time[i] += play_delta * 0.0002

        # acceleration rules
        ax = ay = az = 0.0

        # Reynolds (sep, ali, coh)
        sep0, sep1, sep2, ali0, ali1, ali2, coh0, coh1, coh2 = _reynolds(
            i, boid_pos, boid_vel, boid_count, sep_r, ali_r, coh_r
        )
        if en_sep:
            ax += sep0 * sep_s
            ay += sep1 * sep_s
            az += sep2 * sep_s
        if en_ali:
            ax += ali0 * ali_s
            ay += ali1 * ali_s
            az += ali2 * ali_s
        if en_coh:
            ax += coh0 * coh_s
            ay += coh1 * coh_s
            az += coh2 * coh_s

        # bounds
        if en_bnd:
            sx, sy, sz = _bounds_steer(boid_pos[i, 0], boid_pos[i, 1], boid_pos[i, 2], bound_size)
            ax += sx * bnd_s
            ay += sy * bnd_s
            az += sz * bnd_s

        # random (noise)
        if en_rand:
            t = boid_time[i] * rand_wavelen_scalar

            # axis 0=x,1=y,2=z ; noise_vals shape (N,3,4)
            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 0, 0], boid_noise_vals[i, 0, 1], boid_noise_vals[i, 0, 2], boid_noise_vals[i, 0, 3]
            nx, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.0, boid_noise_cum[i, 0], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 0] = cwl
            boid_noise_vals[i, 0, 0], boid_noise_vals[i, 0, 1], boid_noise_vals[i, 0, 2], boid_noise_vals[i, 0, 3] = rv0, rv1, rv2, rv3

            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 1, 0], boid_noise_vals[i, 1, 1], boid_noise_vals[i, 1, 2], boid_noise_vals[i, 1, 3]
            ny, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.1, boid_noise_cum[i, 1], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 1] = cwl
            boid_noise_vals[i, 1, 0], boid_noise_vals[i, 1, 1], boid_noise_vals[i, 1, 2], boid_noise_vals[i, 1, 3] = rv0, rv1, rv2, rv3

            rv0, rv1, rv2, rv3 = boid_noise_vals[i, 2, 0], boid_noise_vals[i, 2, 1], boid_noise_vals[i, 2, 2], boid_noise_vals[i, 2, 3]
            nz, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.2, boid_noise_cum[i, 2], rv0, rv1, rv2, rv3, seed_arr)
            boid_noise_cum[i, 2] = cwl
            boid_noise_vals[i, 2, 0], boid_noise_vals[i, 2, 1], boid_noise_vals[i, 2, 2], boid_noise_vals[i, 2, 3] = rv0, rv1, rv2, rv3

            # y reduced like JS (*0.2)
            ax += nx * rand_s
            ay += (ny * 0.2) * rand_s
            az += nz * rand_s

        # predator avoidance
        if en_pred_avoid and pred_count > 0:
            sx, sy, sz = _predator_avoid(i, boid_pos, pred_pos, pred_count, pred_avoid_r)
            ax += sx * pred_avoid_s
            ay += sy * pred_avoid_s
            az += sz * pred_avoid_s

        # obstacle avoidance (spheres)
        if en_obs_avoid:
            sx, sy, sz = _obstacle_avoid(boid_pos[i, 0], boid_pos[i, 1], boid_pos[i, 2],
                                         obs_cx, obs_cy, obs_cz, obs_radius, obs_avoid_r, obs_gain)
            ax += sx * obs_avoid_s
            ay += sy * obs_avoid_s
            az += sz * obs_avoid_s

            sx, sy, sz = _obstacle_avoid(boid_pos[i, 0], boid_pos[i, 1], boid_pos[i, 2],
                                         obs2_cx, obs2_cy, obs2_cz, obs_radius, obs_avoid_r, obs_gain)
            ax += sx * obs_avoid_s
            ay += sy * obs_avoid_s
            az += sz * obs_avoid_s

            sx, sy, sz = _obstacle_avoid(boid_pos[i, 0], boid_pos[i, 1], boid_pos[i, 2],
                                         obs3_cx, obs3_cy, obs3_cz, obs_radius, obs_avoid_r, obs_gain)
            ax += sx * obs_avoid_s
            ay += sy * obs_avoid_s
            az += sz * obs_avoid_s


        # goal attraction (acc proportional to gain * distance-to-goal)
        if en_goal and goal_gain > 0.0:
            dx = goal_x - boid_pos[i, 0]
            dy = goal_y - boid_pos[i, 1]
            dz = goal_z - boid_pos[i, 2]
            d = math.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12

            dmax = 8.0  # cap distance influence
            eff = d if d < dmax else dmax
            pwr = 3.0
            mag = goal_gain * (d ** pwr)
            mag = min(mag, goal_gain * 8.0)  # cap

            ax += (dx / d) * mag
            ay += (dy / d) * mag
            az += (dz / d) * mag

        # scale acceleration like JS
        ax *= play_delta * rule_scalar * 0.005
        ay *= play_delta * rule_scalar * 0.005
        az *= play_delta * rule_scalar * 0.005
        ay *= 0.8

        # integrate velocity + clamp
        boid_vel[i, 0] += ax
        boid_vel[i, 1] += ay
        boid_vel[i, 2] += az
        boid_vel[i, 0], boid_vel[i, 1], boid_vel[i, 2] = _clamp_len(
            boid_vel[i, 0], boid_vel[i, 1], boid_vel[i, 2], max_speed
        )

        # integrate position (pos += vel * play_delta)
        boid_pos[i, 0] += boid_vel[i, 0] * play_delta
        boid_pos[i, 1] += boid_vel[i, 1] * play_delta
        boid_pos[i, 2] += boid_vel[i, 2] * play_delta

    # --- predators ---
    for i in range(pred_count):
        pred_time[i] += play_delta * 0.0002

        # acceleration rules for predators: sep + bounds + random + obstacle
        ax = ay = az = 0.0

        sep0, sep1, sep2, _, _, _, _, _, _ = _reynolds(
            i, pred_pos, pred_vel, pred_count, sep_r, ali_r, coh_r
        )
        ax += sep0 * sep_s
        ay += sep1 * sep_s
        az += sep2 * sep_s

        sx, sy, sz = _bounds_steer(pred_pos[i, 0], pred_pos[i, 1], pred_pos[i, 2], bound_size)
        ax += sx * (bnd_s / 1.5)
        ay += sy * (bnd_s / 1.5)
        az += sz * (bnd_s / 1.5)

        # random /2
        t = pred_time[i] * rand_wavelen_scalar
        rv0, rv1, rv2, rv3 = pred_noise_vals[i, 0, 0], pred_noise_vals[i, 0, 1], pred_noise_vals[i, 0, 2], pred_noise_vals[i, 0, 3]
        nx, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.0, pred_noise_cum[i, 0], rv0, rv1, rv2, rv3, seed_arr)
        pred_noise_cum[i, 0] = cwl
        pred_noise_vals[i, 0, 0], pred_noise_vals[i, 0, 1], pred_noise_vals[i, 0, 2], pred_noise_vals[i, 0, 3] = rv0, rv1, rv2, rv3

        rv0, rv1, rv2, rv3 = pred_noise_vals[i, 1, 0], pred_noise_vals[i, 1, 1], pred_noise_vals[i, 1, 2], pred_noise_vals[i, 1, 3]
        ny, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.1, pred_noise_cum[i, 1], rv0, rv1, rv2, rv3, seed_arr)
        pred_noise_cum[i, 1] = cwl
        pred_noise_vals[i, 1, 0], pred_noise_vals[i, 1, 1], pred_noise_vals[i, 1, 2], pred_noise_vals[i, 1, 3] = rv0, rv1, rv2, rv3

        rv0, rv1, rv2, rv3 = pred_noise_vals[i, 2, 0], pred_noise_vals[i, 2, 1], pred_noise_vals[i, 2, 2], pred_noise_vals[i, 2, 3]
        nz, cwl, rv0, rv1, rv2, rv3 = _noise(t + 0.2, pred_noise_cum[i, 2], rv0, rv1, rv2, rv3, seed_arr)
        pred_noise_cum[i, 2] = cwl
        pred_noise_vals[i, 2, 0], pred_noise_vals[i, 2, 1], pred_noise_vals[i, 2, 2], pred_noise_vals[i, 2, 3] = rv0, rv1, rv2, rv3

        ax += nx * (rand_s / 2.0)
        ay += (ny * 0.2) * (rand_s / 2.0)
        az += nz * (rand_s / 2.0)

        sx, sy, sz = _obstacle_avoid(pred_pos[i, 0], pred_pos[i, 1], pred_pos[i, 2],
                                     obs_cx, obs_cy, obs_cz, obs_radius, obs_avoid_r, obs_gain)
        ax += sx * (obs_avoid_s * 4.0)
        ay += sy * (obs_avoid_s * 4.0)
        az += sz * (obs_avoid_s * 4.0)

        sx, sy, sz = _obstacle_avoid(pred_pos[i, 0], pred_pos[i, 1], pred_pos[i, 2],
                                     obs2_cx, obs2_cy, obs2_cz, obs_radius, obs_avoid_r, obs_gain)
        ax += sx * (obs_avoid_s * 4.0)
        ay += sy * (obs_avoid_s * 4.0)
        az += sz * (obs_avoid_s * 4.0)

        sx, sy, sz = _obstacle_avoid(pred_pos[i, 0], pred_pos[i, 1], pred_pos[i, 2],
                                     obs3_cx, obs3_cy, obs3_cz, obs_radius, obs_avoid_r, obs_gain)
        ax += sx * (obs_avoid_s * 4.0)
        ay += sy * (obs_avoid_s * 4.0)
        az += sz * (obs_avoid_s * 4.0)

        ax *= play_delta * rule_scalar_p * 0.005
        ay *= play_delta * rule_scalar_p * 0.005
        az *= play_delta * rule_scalar_p * 0.005
        ay *= 0.8

        pred_vel[i, 0] += ax
        pred_vel[i, 1] += ay
        pred_vel[i, 2] += az

        # velocity-attack (added to velocity, scaled, like JS)
        vx, vy, vz = _velocity_attack_step(
            i, pred_pos, pred_vel, pred_time,
            pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
            boid_pos, boid_count
        )
        vx *= play_delta * (attack_s * 2.0)
        vy *= play_delta * (attack_s * 2.0)
        vz *= play_delta * (attack_s * 2.0)
        pred_vel[i, 0] += vx
        pred_vel[i, 1] += vy
        pred_vel[i, 2] += vz

        pred_vel[i, 0], pred_vel[i, 1], pred_vel[i, 2] = _clamp_len(
            pred_vel[i, 0], pred_vel[i, 1], pred_vel[i, 2], max_speed_p
        )

        pred_pos[i, 0] += pred_vel[i, 0] * play_delta
        pred_pos[i, 1] += pred_vel[i, 1] * play_delta
        pred_pos[i, 2] += pred_vel[i, 2] * play_delta


# -----------------------------
# Python side: init + UI
# -----------------------------
def _init_agents(total_boids, total_preds, bound_size, start_xyz, start_spread, seed=0.1):
    sx, sy, sz = float(start_xyz[0]), float(start_xyz[1]), float(start_xyz[2])

    # positions start near center; predators start at origin like JS
    boid_pos = np.empty((total_boids, 3), dtype=np.float64)
    boid_vel = np.zeros((total_boids, 3), dtype=np.float64)
    boid_time = np.zeros((total_boids,), dtype=np.float64)

    pred_pos = np.zeros((total_preds, 3), dtype=np.float64)
    if total_preds > 0:
        rngp = np.random.default_rng(int(seed * 1e6 + 12345) % (2**32 - 1))
        jitter = rngp.normal(0.0, 1.0, size=(total_preds, 3))
        jitter_norm = np.linalg.norm(jitter, axis=1) + 1e-12
        jitter = jitter / jitter_norm[:, None]
        rj = (rngp.random(total_preds) ** (1.0/3.0)) * (float(start_spread) * 1.5)
        jitter = jitter * rj[:, None]
        pred_pos[:, 0] = sx + jitter[:, 0] - 2.0
        pred_pos[:, 1] = sy + jitter[:, 1]
        pred_pos[:, 2] = sz + jitter[:, 2]
    pred_vel = np.zeros((total_preds, 3), dtype=np.float64)
    pred_time = np.zeros((total_preds,), dtype=np.float64)

    # start as a small random cloud around start
    rng = np.random.default_rng(int(seed * 1e6) % (2**32 - 1))
    offsets = rng.normal(0.0, 1.0, size=(total_boids, 3))
    # normalize to within a ball
    norms = np.linalg.norm(offsets, axis=1) + 1e-12
    offsets = offsets / norms[:, None]
    radii = rng.random(total_boids) ** (1.0/3.0)
    offsets = offsets * (radii[:, None] * float(start_spread))
    boid_pos[:, 0] = sx + offsets[:, 0]
    boid_pos[:, 1] = sy + offsets[:, 1]
    boid_pos[:, 2] = sz + offsets[:, 2]

    # noise state: cum wavelen + 4 random values per axis
    boid_noise_cum = np.zeros((total_boids, 3), dtype=np.float64)
    boid_noise_vals = np.empty((total_boids, 3, 4), dtype=np.float64)

    pred_noise_cum = np.zeros((total_preds, 3), dtype=np.float64)
    pred_noise_vals = np.empty((total_preds, 3, 4), dtype=np.float64)

    seed_arr = np.array([math.floor(seed * 4294967296.0)], dtype=np.float64)

    # initialize noise random values using the same RNG
    for i in range(total_boids):
        for a in range(3):
            boid_noise_vals[i, a, 0] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 1] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 2] = _lcg_rand01(seed_arr)
            boid_noise_vals[i, a, 3] = _lcg_rand01(seed_arr)

    for i in range(total_preds):
        for a in range(3):
            pred_noise_vals[i, a, 0] = _lcg_rand01(seed_arr)
            pred_noise_vals[i, a, 1] = _lcg_rand01(seed_arr)
            pred_noise_vals[i, a, 2] = _lcg_rand01(seed_arr)
            pred_noise_vals[i, a, 3] = _lcg_rand01(seed_arr)

    # predator state
    pred_rest = np.ones((total_preds,), dtype=np.int64)
    pred_rest_start = np.zeros((total_preds,), dtype=np.float64)
    pred_attack_start = np.zeros((total_preds,), dtype=np.float64)
    pred_prey_idx = -np.ones((total_preds,), dtype=np.int64)

    # IMPORTANT: reset seed_arr to the initial seed for runtime to match JS-ish behavior
    seed_arr[0] = math.floor(seed * 4294967296.0)

    return (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
            pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
            pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
            seed_arr)


def run():
    total_boids = 800
    total_preds = 15
    p = make_default_params(boid_count=400, pred_count=8)

    (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
     pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
     pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
     seed_arr) = _init_agents(total_boids, total_preds, p[P_BOUND], np.array([p[P_START_X], p[P_START_Y], p[P_START_Z]]), p[P_START_SPREAD])

    # precompile numba on first call (small dt)
    step_sim(boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
             pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
             pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
             seed_arr, 1/60, p)

    # --- Matplotlib UI ---
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.98)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    def _set_bounds():
        b = p[P_BOUND]
        ax.set_xlim(0, b)
        ax.set_ylim(0, b)
        ax.set_zlim(0, b)

    _set_bounds()

    boid_scatter = ax.scatter(boid_pos[:int(p[P_BOID_COUNT]), 0],
                              boid_pos[:int(p[P_BOID_COUNT]), 1],
                              boid_pos[:int(p[P_BOID_COUNT]), 2],
                              s=6, depthshade=False)

    pred_scatter = ax.scatter(pred_pos[:int(p[P_PRED_COUNT]), 0],
                              pred_pos[:int(p[P_PRED_COUNT]), 1],
                              pred_pos[:int(p[P_PRED_COUNT]), 2],
                              s=20, marker="^", depthshade=False)

    goal_scatter = ax.scatter([p[P_GOAL_X]], [p[P_GOAL_Y]], [p[P_GOAL_Z]],
                              s=80, marker="*", depthshade=False)

    # obstacle sphere wireframe(s)
    sphere_lines = []

    def _draw_spheres():
        nonlocal sphere_lines
        # remove old
        for ln in sphere_lines:
            try:
                ln.remove()
            except Exception:
                pass
        sphere_lines = []

        r = p[P_OBS_RADIUS]
        centers = (
            (p[P_OBS_CX], p[P_OBS_CY], p[P_OBS_CZ]),
            (p[P_OBS2_CX], p[P_OBS2_CY], p[P_OBS2_CZ]),
            (p[P_OBS3_CX], p[P_OBS3_CY], p[P_OBS3_CZ]),
        )
        u = np.linspace(0, 2*np.pi, 24)
        v = np.linspace(0, np.pi, 12)

        for (cx, cy, cz) in centers:
            x = cx + r * np.outer(np.cos(u), np.sin(v))
            y = cy + r * np.outer(np.sin(u), np.sin(v))
            z = cz + r * np.outer(np.ones_like(u), np.cos(v))
            for k in range(0, x.shape[0], 4):
                sphere_lines.append(ax.plot(x[k, :], y[k, :], z[k, :], linewidth=0.8)[0])
            for k in range(0, x.shape[1], 3):
                sphere_lines.append(ax.plot(x[:, k], y[:, k], z[:, k], linewidth=0.8)[0])

    _draw_spheres()

    # episode step counter
    step_count = {'n': 0}

    # --- sliders ---
    axcolor = "lightgoldenrodyellow"
    sx = 0.10
    w = 0.22
    h = 0.03
    y0 = 0.17
    dy = 0.04

    ax_sep_r = plt.axes([sx, y0, w, h], facecolor=axcolor)
    ax_ali_r = plt.axes([sx, y0 - dy, w, h], facecolor=axcolor)
    ax_coh_r = plt.axes([sx, y0 - 2*dy, w, h], facecolor=axcolor)

    s_sep_r = Slider(ax_sep_r, "sep R", 0.3, 6.0, valinit=p[P_SEP_R])
    s_ali_r = Slider(ax_ali_r, "ali R", 0.3, 10.0, valinit=p[P_ALI_R])
    s_coh_r = Slider(ax_coh_r, "coh R", 0.3, 12.0, valinit=p[P_COH_R])

    ax_max = plt.axes([sx + 0.30, y0, w, h], facecolor=axcolor)
    ax_play = plt.axes([sx + 0.30, y0 - dy, w, h], facecolor=axcolor)
    ax_obs_r = plt.axes([sx + 0.30, y0 - 2*dy, w, h], facecolor=axcolor)

    s_max = Slider(ax_max, "max v", 0.02, 0.6, valinit=p[P_MAX_SPEED])
    s_play = Slider(ax_play, "play", 0.0, 3.0, valinit=p[P_PLAY_SPEED])
    s_obs_r = Slider(ax_obs_r, "obs R", 0.0, 12.0, valinit=p[P_OBS_RADIUS])

    ax_obs_avoid = plt.axes([sx + 0.60, y0, w, h], facecolor=axcolor)
    ax_pred = plt.axes([sx + 0.60, y0 - dy, w, h], facecolor=axcolor)
    ax_boids = plt.axes([sx + 0.60, y0 - 2*dy, w, h], facecolor=axcolor)

    s_obs_avoid = Slider(ax_obs_avoid, "obs avoid", 0.0, 12.0, valinit=p[P_OBS_AVOID_R])
    s_pred = Slider(ax_pred, "pred #", 0, total_preds, valinit=int(p[P_PRED_COUNT]), valstep=1)
    s_boids = Slider(ax_boids, "boids #", 10, total_boids, valinit=int(p[P_BOID_COUNT]), valstep=1)

    ax_goal = plt.axes([sx + 0.60, y0 - 3*dy, w, h], facecolor=axcolor)
    s_goal = Slider(ax_goal, "goal k", 0.0, 0.3, valinit=p[P_GOAL_GAIN])

    ax_steps = plt.axes([sx + 0.30, y0 - 3*dy, w, h], facecolor=axcolor)
    s_steps = Slider(ax_steps, "steps", 100, 20000, valinit=int(p[P_MAX_STEPS]), valstep=50)
    ax_spread = plt.axes([sx, y0 - 3*dy, w, h], facecolor=axcolor)
    s_spread = Slider(ax_spread, "start spr", 0.2, 10.0, valinit=p[P_START_SPREAD])
    
    ax_obs_gain = plt.axes([sx + 0.30, y0 - 4*dy, w, h], facecolor=axcolor)
    s_obstacles = Slider(ax_obs_gain, "obs gain", 0.0, 10.0, valinit=p[P_OBS_GAIN])

    # toggles
    ax_checks = plt.axes([0.02, 0.25, 0.07, 0.20], facecolor=axcolor)
    labels = ["sep", "ali", "coh", "bnd", "rnd", "pred", "obs", "goal"]
    actives = [p[P_ENABLE_SEP] > 0.5, p[P_ENABLE_ALI] > 0.5, p[P_ENABLE_COH] > 0.5,
               p[P_ENABLE_BND] > 0.5, p[P_ENABLE_RAND] > 0.5, p[P_ENABLE_PRED_AVOID] > 0.5,
               p[P_ENABLE_OBS_AVOID] > 0.5, p[P_ENABLE_GOAL] > 0.5]
    checks = CheckButtons(ax_checks, labels, actives)

    def _apply_slider_vals(_=None):
        p[P_SEP_R] = s_sep_r.val
        p[P_ALI_R] = s_ali_r.val
        p[P_COH_R] = s_coh_r.val
        p[P_MAX_SPEED] = s_max.val
        p[P_PLAY_SPEED] = s_play.val
        p[P_OBS_RADIUS] = s_obs_r.val
        p[P_OBS_AVOID_R] = s_obs_avoid.val
        p[P_PRED_COUNT] = float(s_pred.val)
        p[P_BOID_COUNT] = float(s_boids.val)
        p[P_GOAL_GAIN] = s_goal.val
        p[P_MAX_STEPS] = float(s_steps.val)
        p[P_START_SPREAD] = s_spread.val
        p[P_OBS_GAIN] = s_obstacles.val

        # keep predators max speed a bit higher
        p[P_MAX_SPEED_P] = max(0.03, min(1.0, p[P_MAX_SPEED] * 1.5))

        # when bounds change (not slider-exposed), we'd redraw
        _draw_spheres()

    for s in (s_sep_r, s_ali_r, s_coh_r, s_max, s_play, s_obs_r, s_obs_avoid, s_pred, s_boids, s_goal, s_steps, s_spread):
        s.on_changed(_apply_slider_vals)

    def _on_check(label):
        mapping = {
            "sep": P_ENABLE_SEP,
            "ali": P_ENABLE_ALI,
            "coh": P_ENABLE_COH,
            "bnd": P_ENABLE_BND,
            "rnd": P_ENABLE_RAND,
            "pred": P_ENABLE_PRED_AVOID,
            "obs": P_ENABLE_OBS_AVOID,
            "goal": P_ENABLE_GOAL,
        }
        idx = mapping[label]
        p[idx] = 0.0 if p[idx] > 0.5 else 1.0

    checks.on_clicked(_on_check)

    ax_restart = plt.axes([0.02, 0.82, 0.10, 0.06])
    btn_restart = Button(ax_restart, 'Restart')

    def _restart(_event=None):
        nonlocal boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals
        nonlocal pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals
        nonlocal pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx, seed_arr
        # reset episode counter
        step_count['n'] = 0
        # re-init agents at start
        (boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
         pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
         pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
         seed_arr) = _init_agents(total_boids, total_preds, p[P_BOUND],
                                 np.array([p[P_START_X], p[P_START_Y], p[P_START_Z]]),
                                 p[P_START_SPREAD])
        # force visual update
        nb = int(p[P_BOID_COUNT]); npred = int(p[P_PRED_COUNT])
        boid_scatter._offsets3d = (boid_pos[:nb, 0], boid_pos[:nb, 1], boid_pos[:nb, 2])
        pred_scatter._offsets3d = (pred_pos[:npred, 0], pred_pos[:npred, 1], pred_pos[:npred, 2])
        goal_scatter._offsets3d = ([p[P_GOAL_X]], [p[P_GOAL_Y]], [p[P_GOAL_Z]])
        fig.canvas.draw_idle()

    btn_restart.on_clicked(_restart)

    # simple camera nicety
    ax.view_init(elev=22, azim=35)

    status_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)

    dt = 1.0 / 60.0

    def _update(_frame):
        max_steps = int(p[P_MAX_STEPS])
        if step_count['n'] < max_steps:
            step_sim(boid_pos, boid_vel, boid_time, boid_noise_cum, boid_noise_vals,
                     pred_pos, pred_vel, pred_time, pred_noise_cum, pred_noise_vals,
                     pred_rest, pred_rest_start, pred_attack_start, pred_prey_idx,
                     seed_arr, dt, p)
            step_count['n'] += 1

        nb = int(p[P_BOID_COUNT])
        npred = int(p[P_PRED_COUNT])

        # Update scatters (matplotlib 3D needs _offsets3d)
        boid_scatter._offsets3d = (boid_pos[:nb, 0], boid_pos[:nb, 1], boid_pos[:nb, 2])
        pred_scatter._offsets3d = (pred_pos[:npred, 0], pred_pos[:npred, 1], pred_pos[:npred, 2])
        goal_scatter._offsets3d = ([p[P_GOAL_X]], [p[P_GOAL_Y]], [p[P_GOAL_Z]])
        status_text.set_text(f"step {step_count['n']} / {int(p[P_MAX_STEPS])}")

        return (boid_scatter, pred_scatter, goal_scatter, *sphere_lines)

    ani = FuncAnimation(fig, _update, interval=16, blit=False)
    plt.show()


if __name__ == "__main__":
    run()
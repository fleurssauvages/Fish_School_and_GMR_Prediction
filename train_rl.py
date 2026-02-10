import numpy as np
from RL.env import FishGoalEnv, make_sphere_mesh
from RL.multiagent_power_rl import MultiAgentPowerRL
import pickle
import numpy as np

def main():
    # --- Simulation Parameters ---
    boid_count = 500
    max_steps = 300
    dt = 1.0

    # --- RL Parameters ---
    n_agents = 24
    iters = 10
    rollouts_per_iter = 4
    eval_episodes = 2
    seed0 = 0

    # Weights: goal, time penalty, eaten penalty, entropy (penalty application is already negative in env)
    w_goal, w_time, w_diversity = 10.0, 0.5, 2.5

    # Initial policy
    """
    Parameters order:
    0: separation scalar
    1: alignment scalar
    2: cohesion scalar
    3: boundary scalar
    4: randomness scalar
    5: obstacle avoidance scalar
    6: goal attraction gain
    """
    theta0 = np.array([1.0, #0: separation scalar
                       1.0, #1: alignment scalar
                       1.0, #2: cohesion scalar
                       1.0, #3: boundary scalar
                       1.0, #4: randomness scalar
                       1.0, #5: obstacle avoidance scalar
                       0.3, #6: goal attraction gain
                       ], dtype=np.float32)

    # Exploration std: scalar or per-dim vector
    exploration_std = 5 * np.full((7,), [0.5, 0.5, 0.5, 0.5, 2.0, 0.1, 0.1], dtype=np.float32)


    verts, faces = make_sphere_mesh(
            R=3.0,
            seg_theta=24,
            seg_phi=24,
            center=(20.0, 20.0, 20.0)
        )
    goals = np.array([
        [34.0, 20.0, 20.0],  # 0 - initial
        [40.0, 20.0, 20.0],  # 1
        [40.0, 30.0, 20.0],  # 2
        [40.0, 10.0, 20.0],  # 3
    ], dtype=np.float32)
    goal_W = np.array([
        [0.0, 2.0, 1.0, 1.0],  # from 0 → {1,2,3}
        [0.0, 1.0, 0.0, 0.0],  # from 1 → 0
        [0.0, 0.0, 1.0, 0.0],  # from 2 → 0
        [0.0, 0.0, 0.0, 1.0],  # from 3 → 0
    ], dtype=np.float32)

    env = FishGoalEnv(boid_count=boid_count, max_steps=max_steps, dt=dt,
                      verts=verts, faces=faces, goals=goals, goal_W=goal_W,
                      w_goal=w_goal, w_time=w_time, w_div=w_diversity)

    ma = MultiAgentPowerRL(
        init_params=theta0,
        exploration_std=exploration_std,
        n_agents=n_agents,
        reuse_top_n=2,
        diversity_strength=0.08,
    )

    best_R = -1e18
    best_theta = theta0.copy()

    for it in range(iters):
        # Clear each agent's rollout history for this iteration
        ma.reset_histories()

        # For each agent, collect rollouts and add to that agent's PoWER memory
        for a_idx, agent in enumerate(ma.agents):
            for k in range(rollouts_per_iter):
                theta_k = agent.sample_policy().astype(np.float32)
                theta_k = np.clip(theta_k, 0.0, 20.0).astype(np.float32)

                # Evaluate on multiple seeds to reduce noise
                seeds = [seed0 + 100000 * it + 1000 * a_idx + 10 * k + r for r in range(eval_episodes)]
                R_theta = 0.0
                for seed in seeds:
                    env.reset(seed=seed)
                    _, R, _, _, _ = env.step(theta_k)
                    R_theta += R
                R = R_theta / eval_episodes

                agent.add_rollout(theta_k, R)

                if R > best_R:
                    best_R = R
                    best_theta = theta_k.copy()

        # Update each agent via PoWER
        ma.update_agents()

        # Apply diversity / repulsion (after update)
        ma.apply_diversity_pressure(exploration_std=exploration_std, iteration=it, decay=0.99)

        # Optional: anneal exploration std over time
        new_std = np.maximum(0.05, exploration_std * (0.985 ** it))
        ma.update_exploration(new_std)

        # Logging
        bidx, bagent = ma.best_agent()
        print(
            f"[it {it+1:04d}] global_best={best_R: .4f}  "
            f"best_agent={bidx}"
        )

    print("==== DONE ====")
    print("Best return:", best_R)
    print("Best theta:", best_theta)

    pickle.dump({
        'best_theta': best_theta,
    }, open('save/best_policy.pkl', 'wb'))

if __name__ == "__main__":
    main()
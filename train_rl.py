
import numpy as np
from RL.env import FishGoalEnv
from RL.multiagent_power_rl import MultiAgentPowerRL
import pickle
import numpy as np

def eval_theta(env: FishGoalEnv, theta: np.ndarray, seeds):
    """Average return over multiple episodes for variance reduction."""
    Rs = np.zeros((len(seeds), 4), dtype=np.float32)
    for i, s in enumerate(seeds):
        env.reset(seed=int(s))
        _, R, _, _, _ = env.step(theta.astype(np.float32))
        Rs[i] = R
    return Rs

def eval_reward(Rs, weights):
    R = np.mean(Rs, axis=0)
    return np.dot(R, weights)

def main():
    boid_count = 200
    pred_count = 4
    max_steps = 300

    env = FishGoalEnv(
        boid_count=boid_count,
        pred_count=pred_count,
        max_steps=max_steps,
    )
    n_agents = 12
    iters = 5
    rollouts_per_iter = 8
    eval_episodes = 3
    seed0 = 0

    """
    Parameters order:
    0: separation scalar
    1: alignment scalar
    2: cohesion scalar
    3: boundary scalar
    4: randomness scalar
    5: predator avoidance scalar
    6: obstacle avoidance scalar
    7: goal attraction gain
    8: obstacle avoidance scalar
    """
    theta0 = np.array([1.0, #0: separation scalar
                       1.0, #1: alignment scalar
                       1.0, #2: cohesion scalar
                       1.0, #3: boundary scalar
                       1.0, #4: randomness scalar
                       10.0, #5: predator avoidance scalar
                       1.0, #6: obstacle avoidance scalar
                       0.3, #7: goal attraction gain
                       ], dtype=np.float32)

    # Weights: goal, eaten penalty, time penalty, entropy
    w_goal, w_eaten, w_obs, w_speed = 1.0, -0.5, -0.3, 0.2
    W = np.array([w_goal, w_eaten, w_obs, w_speed], dtype=np.float32)

    # Exploration std: scalar or per-dim vector
    exploration_std = np.full((8,), [0.5, 0.5, 0.5, 0.5, 0.5, 2.0, 0.1, 0.1], dtype=np.float32)

    ma = MultiAgentPowerRL(
        init_params=theta0,
        exploration_std=exploration_std,
        n_agents=n_agents,
        reuse_top_n=6,
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

                # Evaluate on multiple seeds to reduce noise
                seeds = [seed0 + 100000 * it + 1000 * a_idx + 10 * k + r for r in range(eval_episodes)]
                Rs = eval_theta(env, theta_k, seeds)
                R = eval_reward(Rs, W)

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
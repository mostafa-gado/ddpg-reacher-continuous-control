"""Train the shared DDPG agent on the 20-arm Reacher environment from the terminal.

Examples
--------
    # start (or resume) training, headless, with the default settings
    python train.py

    # 300 more episodes with less exploration noise
    python train.py --episodes 300 --sigma 0.1

    # start from plain weight files saved from an older run
    python train.py --no-resume --init-actor checkpoint_actor_multi.pth --init-critic checkpoint_critic_multi.pth

    # keep training after the terminal / VS Code closes, and follow the log
    nohup python train.py > /dev/null 2>&1 &
    tail -f train.log

Stopping with Ctrl+C (or `kill <pid>`) saves a checkpoint first, so no work is lost.
"""

import argparse
import logging
import os
import signal
import time
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')                  # draw plots to a file, no window needed
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from ddpg_multi_agent import MultiAgent


def parse_args():
    """Read the command-line options."""
    p = argparse.ArgumentParser(description='Train DDPG on the 20-arm Reacher environment.')
    p.add_argument('--env', default='Reacher_Linux_20/Reacher.x86_64', help='path to the Unity environment')
    p.add_argument('--episodes', type=int, default=500, help='how many MORE episodes to run')
    p.add_argument('--max-t', type=int, default=1000, help='max steps per episode')
    p.add_argument('--sigma', type=float, default=0.2, help='exploration noise size')
    p.add_argument('--seed', type=int, default=2, help='random seed')
    p.add_argument('--checkpoint', default='checkpoint_multi.pth', help='full checkpoint file (save + resume)')
    p.add_argument('--no-resume', action='store_true', help='ignore an existing checkpoint and start fresh')
    p.add_argument('--init-actor', help='plain actor weights to start from (optional)')
    p.add_argument('--init-critic', help='plain critic weights to start from (optional)')
    p.add_argument('--save-every', type=int, default=10, help='save a checkpoint every N episodes')
    p.add_argument('--log', default='train.log', help='log file')
    p.add_argument('--plot', default='scores.png', help='where to save the score plot')
    p.add_argument('--graphics', action='store_true', help='show the Unity window (slower)')
    p.add_argument('--worker-id', type=int, default=0, help='change if another Unity env is already running')
    return p.parse_args()


def setup_logging(log_file):
    """Log to the terminal AND to a file, so progress can be checked with `tail -f`."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,                    # unityagents sets up logging on import; replace its setup
    )
    return logging.getLogger('train')


def save_plot(scores, path):
    """Save the score of every episode plus the 100-episode average to an image."""
    if not scores:
        return
    rolling = [np.mean(scores[max(0, i - 99):i + 1]) for i in range(len(scores))]
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(scores) + 1), scores, alpha=0.4, label='Episode score')
    ax.plot(np.arange(1, len(scores) + 1), rolling, label='Average of last 100')
    ax.axhline(30, color='gray', linestyle='--', label='Target (30)')
    ax.set_xlabel('Episode #')
    ax.set_ylabel('Score (mean over 20 arms)')
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def stop_on_sigterm(signum, frame):
    """Turn `kill <pid>` into a KeyboardInterrupt, so it saves a checkpoint like Ctrl+C."""
    raise KeyboardInterrupt


def main():
    args = parse_args()
    log = setup_logging(args.log)
    signal.signal(signal.SIGTERM, stop_on_sigterm)

    # ---------------------------- environment ---------------------------- #
    env = UnityEnvironment(file_name=args.env, no_graphics=not args.graphics, worker_id=args.worker_id)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)                    # 20 arms
    action_size = brain.vector_action_space_size         # 4 torques per arm
    state_size = env_info.vector_observations.shape[1]   # 33 values per arm
    log.info('Agents: %d, state size: %d, action size: %d', num_agents, state_size, action_size)

    # ------------------------------- agent ------------------------------- #
    agent = MultiAgent(state_size=state_size, action_size=action_size, num_agents=num_agents,
                       random_seed=args.seed, noise_sigma=args.sigma)
    log.info('Noise sigma: %.2f', args.sigma)

    scores = []                                          # mean score of every episode
    if not args.no_resume and os.path.exists(args.checkpoint):
        # resume: networks, optimizers and score history
        checkpoint = agent.load(args.checkpoint)
        scores = checkpoint['scores']
        log.info('Resumed from %s after %d episodes (last 100 avg: %.2f)',
                 args.checkpoint, len(scores), np.mean(scores[-100:]))
    elif args.init_actor and args.init_critic:
        # start from plain weights (optimizers and score history start fresh)
        agent.load_weights(args.init_actor, args.init_critic)
        log.info('Loaded weights from %s and %s', args.init_actor, args.init_critic)
    else:
        log.info('Starting fresh')

    # the rolling window continues from the resumed history
    scores_deque = deque(scores[-100:], maxlen=100)
    start_episode = len(scores) + 1

    # ------------------------------ training ----------------------------- #
    try:
        for i_episode in range(start_episode, start_episode + args.episodes):
            t0 = time.time()
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations          # shape (20, 33): one row per arm
            agent.reset()                                  # reset the exploration noise
            episode_scores = np.zeros(num_agents)          # one running score per arm
            for t in range(args.max_t):
                actions = agent.act(states)                # shape (20, 4): one action per arm
                env_info = env.step(actions)[brain_name]   # send all actions to the environment
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                agent.step(states, actions, rewards, next_states, dones)  # store all 20, learn sometimes
                states = next_states
                episode_scores += rewards
                if np.any(dones):                          # all arms finish together
                    break

            # the project's score for an episode = the average over the 20 arms
            score = float(np.mean(episode_scores))
            scores.append(score)
            scores_deque.append(score)
            log.info('Episode %d\tScore: %.2f\tAverage (last 100): %.2f\tTime: %.1fs',
                     i_episode, score, np.mean(scores_deque), time.time() - t0)

            # save regularly, so a crash loses at most --save-every episodes
            if i_episode % args.save_every == 0:
                agent.save(args.checkpoint, scores=scores)
                save_plot(scores, args.plot)

            # solved = average of +30 over the last 100 episodes
            if len(scores_deque) == 100 and np.mean(scores_deque) >= 30.0:
                log.info('Environment solved at episode %d! Average of episodes %d-%d: %.2f',
                         i_episode, i_episode - 99, i_episode, np.mean(scores_deque))
                # plain actor/critic weights, for the project submission
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_multi.pth')
                torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_multi.pth')
                break
    except KeyboardInterrupt:
        log.info('Interrupted after %d episodes', len(scores))
    finally:
        # always save and clean up, however training ended
        agent.save(args.checkpoint, scores=scores)
        save_plot(scores, args.plot)
        log.info('Saved %s and %s', args.checkpoint, args.plot)
        env.close()


if __name__ == '__main__':
    main()

"""Test the trained agent on the 20-arm Reacher environment.

Loads the solved actor weights and runs the arms WITHOUT exploration noise,
so the scores show what the learned policy can really do.

Examples
--------
    # watch the trained arms for 1 episode
    python test.py --graphics --episodes 1

    # check the score over 100 episodes, headless (fast)
    python test.py --episodes 100
"""

import argparse

import numpy as np
import torch
from unityagents import UnityEnvironment

from model import Actor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Read the command-line options."""
    p = argparse.ArgumentParser(description='Test a trained DDPG agent on the 20-arm Reacher environment.')
    p.add_argument('--env', default='Reacher_Linux_20/Reacher.x86_64', help='path to the Unity environment')
    p.add_argument('--actor', default='checkpoint_actor_multi.pth', help='trained actor weights')
    p.add_argument('--episodes', type=int, default=5, help='how many test episodes to run')
    p.add_argument('--graphics', action='store_true', help='show the Unity window, at normal speed')
    p.add_argument('--worker-id', type=int, default=0, help='change if another Unity env is already running')
    return p.parse_args()


def main():
    args = parse_args()

    # ---------------------------- environment ---------------------------- #
    env = UnityEnvironment(file_name=args.env, no_graphics=not args.graphics, worker_id=args.worker_id)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)                    # 20 arms
    action_size = brain.vector_action_space_size         # 4 torques per arm
    state_size = env_info.vector_observations.shape[1]   # 33 values per arm

    # ------------------------------- actor ------------------------------- #
    # Only the actor is needed to act; the critic is only used during training.
    actor = Actor(state_size, action_size, seed=0).to(device)
    actor.load_state_dict(torch.load(args.actor, map_location=device))
    actor.eval()                                         # inference mode
    print('Loaded {}'.format(args.actor))

    # ------------------------------- test -------------------------------- #
    # train_mode=False runs the simulation at normal speed, so it can be watched
    train_mode = not args.graphics
    scores = []
    try:
        for i_episode in range(1, args.episodes + 1):
            env_info = env.reset(train_mode=train_mode)[brain_name]
            states = env_info.vector_observations          # shape (20, 33): one row per arm
            episode_scores = np.zeros(num_agents)          # one running score per arm
            while True:
                with torch.no_grad():
                    # no noise: the actor's own action, for every arm at once
                    actions = actor(torch.from_numpy(states).float().to(device)).cpu().numpy()
                env_info = env.step(actions)[brain_name]
                states = env_info.vector_observations
                episode_scores += env_info.rewards
                if np.any(env_info.local_done):            # all arms finish together
                    break
            # same score as in training: the average over the 20 arms
            score = np.mean(episode_scores)
            scores.append(score)
            print('Episode {}\tScore: {:.2f}\t(min arm {:.2f}, max arm {:.2f})'.format(
                i_episode, score, episode_scores.min(), episode_scores.max()))
    except KeyboardInterrupt:
        print('\nStopped')
    finally:
        env.close()

    if scores:
        print('\nAverage score over {} episodes: {:.2f}  (target: 30)'.format(len(scores), np.mean(scores)))


if __name__ == '__main__':
    main()

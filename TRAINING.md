# Training the Reacher Agent from the Terminal

`train.py` trains the shared DDPG agent ([ddpg_multi_agent.py](ddpg_multi_agent.py)) on the 20-arm Reacher environment.
It resumes from its checkpoint automatically and saves whenever it stops.

All commands below are run from this folder:

```bash
cd "deep-reinforcement-learning/14-p2_continuous-control"
```

## Quick start

```bash
python train.py                  # train (resumes from checkpoint_multi.pth if it exists)
tail -f train.log                # follow progress from another terminal
```

- Stop with **Ctrl+C**. It saves a checkpoint before exiting.
- Run it again to continue where it stopped.

## Options

| Option | Default | What it does |
| --- | --- | --- |
| `--episodes N` | `500` | How many **more** episodes to run (on top of resumed ones) |
| `--max-t N` | `1000` | Max steps per episode |
| `--sigma X` | `0.2` | Exploration noise size. Smaller = less random actions |
| `--seed N` | `2` | Random seed |
| `--checkpoint FILE` | `checkpoint_multi.pth` | Full checkpoint to resume from and save to |
| `--no-resume` | off | Ignore the checkpoint and start fresh |
| `--init-actor FILE` | none | Plain actor weights to start from |
| `--init-critic FILE` | none | Plain critic weights to start from |
| `--save-every N` | `10` | Save a checkpoint every N episodes |
| `--log FILE` | `train.log` | Log file |
| `--plot FILE` | `scores.png` | Score plot image |
| `--graphics` | off | Show the Unity window (slower) |
| `--env PATH` | `Reacher_Linux_20/Reacher.x86_64` | Path to the Unity environment |
| `--worker-id N` | `0` | Change if another Unity environment is already running |

Run `python train.py --help` to see the same list.

## Common recipes

**Resume training** (default):
```bash
python train.py --episodes 300
```

**Start from scratch** (keeps the old checkpoint safe under a new name):
```bash
mv checkpoint_multi.pth checkpoint_multi_old.pth
python train.py
```
`--no-resume` also starts fresh, but it **overwrites** `checkpoint_multi.pth` on the first save.

**Start from plain weight files** (e.g. the trained actor and critic):
```bash
python train.py --no-resume \
    --init-actor checkpoint_actor_multi.pth \
    --init-critic checkpoint_critic_multi.pth
```
Optimizers and score history start fresh. Expect a short dip in score while the replay buffer refills.

**Try a different noise level:**
```bash
python train.py --sigma 0.05
```

**Run an experiment without touching the main checkpoint:**
```bash
python train.py --checkpoint exp_sigma005.pth --log exp_sigma005.log --plot exp_sigma005.png --sigma 0.05
```

**Watch the arms while training:**
```bash
python train.py --graphics
```

## Running in the background

Keeps training after you close the terminal or VS Code:

```bash
nohup python train.py > /dev/null 2>&1 &
```

| Task | Command |
| --- | --- |
| Follow progress | `tail -f train.log` |
| See the last 20 episodes | `tail -n 20 train.log` |
| Check it's still running | `pgrep -fa train.py` |
| Stop it safely (saves first) | `pkill -f train.py` |

`pkill` sends a normal stop signal, which saves a checkpoint like Ctrl+C.
Avoid `kill -9`: it stops immediately and skips the save, so you lose up to `--save-every` episodes.

## Output files

| File | What it is |
| --- | --- |
| `checkpoint_multi.pth` | Full checkpoint: actor, critic, target networks, optimizers and score history. Used to resume |
| `checkpoint_actor_multi.pth` | Plain actor weights, written when the task is solved. Use for the submission |
| `checkpoint_critic_multi.pth` | Plain critic weights, written when the task is solved |
| `train.log` | One line per episode: score, 100-episode average and time |
| `scores.png` | Plot of episode scores, the 100-episode average and the target (30) |

The replay buffer is **not** saved. It refills within a few episodes after a resume.

## Testing the trained agent

`test.py` loads `checkpoint_actor_multi.pth` and runs the arms with **no exploration noise**.

```bash
python test.py --graphics --episodes 1   # watch the trained arms
python test.py --episodes 100            # check the average score, headless (fast)
```

| Option | Default | What it does |
| --- | --- | --- |
| `--episodes N` | `5` | How many test episodes to run |
| `--actor FILE` | `checkpoint_actor_multi.pth` | Trained actor weights |
| `--graphics` | off | Show the Unity window at normal speed |
| `--env PATH` | `Reacher_Linux_20/Reacher.x86_64` | Path to the Unity environment |
| `--worker-id N` | `0` | Change if another Unity environment is already running |

It prints each episode's score (mean over 20 arms, plus the weakest and strongest arm) and the overall average.

## Reading the log

```
21:21:51  Episode 63	Score: 12.40	Average (last 100): 9.85	Time: 19.7s
```

- **Score**: this episode's score, averaged over the 20 arms.
- **Average (last 100)**: the number that must reach **30** to solve the task.
- **Time**: seconds for this episode. About 20 s is normal for 1000 steps.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Unity fails to start: port already in use | Another environment is still open. Use `--worker-id 1` (or any other number), or close the other one |
| Two training runs at the same time | Stop one, or give each its own `--checkpoint`. Otherwise they overwrite each other |
| Score drops right after a resume | Normal: the replay buffer starts empty. Judge progress after 30–50 episodes |
| Episodes much slower than ~20 s | Check `--graphics` is off and nothing else is using the GPU |
| `unityagents` not found | Activate the course's Python environment first |

import tempfile
import mlflow
import shutil
from pathlib import Path
from goalagent import repo_root
import pandas as pd
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from goalagent.env.rg_env import RgEnv
from regelum.simulator import CasADi
from goalagent.env.pendulum import InvertedPendulum, PendulumStabilizingPolicy
from goalagent.env.running_objective import PendulumRunningObjective
import torch
import numpy as np


def save_source_code():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        code_folder = repo_root / "goalagent"
        # Iterate through all files in the directory tree
        for file_path in code_folder.rglob("*.py"):
            # Check if the file is in the excluded folder
            if "regelum_data" not in str(file_path):
                # Calculate the relative path from the code_folder root
                relative_path = file_path.relative_to(code_folder)
                # Create the same directory structure in the temp dir
                (temp_path / relative_path.parent).mkdir(parents=True, exist_ok=True)
                # Copy the file to the temp dir
                shutil.copy(file_path, temp_path / relative_path)

        mlflow.log_artifacts(temp_path, artifact_path="source_code/goalagent")
    mlflow.log_artifacts(repo_root / "presets", artifact_path="source_code/presets")
    # mlflow.log_artifacts(
    #     repo_root / "checkpoints", artifact_path="source_code/checkpoints"
    # )


def save_episodic_data(
    info: dict,
    global_step: int,
    episodic_observations: int,
    episodic_actions: int,
    observations_names: list[str],
    actions_names: list[str],
):
    plt.clf()
    plt.cla()
    plt.close()
    mlflow.log_metric("episodic_return", info["episode"]["r"], step=global_step)
    mlflow.log_metric("charts/episodic_length", info["episode"]["l"], global_step)
    with tempfile.TemporaryDirectory() as tmpdir:
        pd_episodic_observations = pd.DataFrame(
            data=np.vstack(episodic_observations), columns=observations_names
        )
        pd_episodic_observations.to_hdf(
            f"{tmpdir}/observations_{global_step:08}.h5", key="data"
        )
        mlflow.log_artifact(
            f"{tmpdir}/observations_{global_step:08}.h5",
            "raw/observations",
        )
        pd_episodic_actions = pd.DataFrame(
            data=np.vstack(episodic_actions), columns=actions_names
        )
        pd_episodic_actions.to_hdf(f"{tmpdir}/actions_{global_step:08}.h5", key="data")
        mlflow.log_artifact(f"{tmpdir}/actions_{global_step:08}.h5", "raw/actions")
        pd_episodic_observations_actions = pd.concat(
            [pd_episodic_observations, pd_episodic_actions],
            axis=1,
        )
        axes = pd_episodic_observations_actions.plot(
            subplots=True,
            legend=True,
            title="Observations and actions",
            kind="line",
            grid=True,
        )
        fig = axes[0].get_figure()
        fig_name = f"observations_actions_{global_step:08}.svg"
        fig.savefig(f"{tmpdir}/{fig_name}")
        mlflow.log_artifact(f"{tmpdir}/{fig_name}", "observations_actions_plots")


def make_env(env):
    def thunk():
        wrapped_env = gym.wrappers.RecordEpisodeStatistics(env)
        return wrapped_env

    return thunk


def evaluate_pendulum_policy(agent, global_step):
    env = RgEnv(
        simulator=CasADi(
            system=InvertedPendulum(),
            state_init=np.array([[np.pi, 1.0]]),
            time_final=10,
            max_step=0.01,
        ),
        running_objective=PendulumRunningObjective(),
    )
    agent.eval()
    with torch.no_grad():
        obs, info = env.reset()
        total_reward = 0
        for i in range(1000):
            # action, logprob, entropy, value = agent.get_action_and_value(
            #     torch.tensor(obs.reshape(1, -1), device="cuda:0")
            # )
            actions, _, _ = agent.get_action(
                torch.Tensor(obs.reshape(1, -1)).to("cuda:0")
            )
            # if i % 4 == 0:
            #     agent.reset_noise()
            obs, reward, done, info, _ = env.step(actions.cpu().numpy())
            total_reward += reward

    agent.train()
    mlflow.log_metric(
        "eval/total_reward_pendulum_from_bottom", total_reward, step=global_step
    )
    return total_reward

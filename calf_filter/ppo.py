# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

import regelum as rg

import goalagent.agent
from goalagent import repo_root, checkpoints_root
import mlflow
from goalagent.calf_filter.calf_filter import CalfFilter, compute_decay_penalty
from goalagent.utils import (
    save_source_code,
    save_episodic_data,
    evaluate_pendulum_policy,
)
import matplotlib.pyplot as plt

config_path = repo_root / "presets"
config_name = os.path.basename(__file__)[: -len(".py")]


@rg.main(
    config_path=config_path,
    config_name=config_name,
)
def launch(cfg):
    ############################
    # Instantiation of classes #
    ############################

    args = ~cfg
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    save_source_code()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = args.agent.to(device)
    if hasattr(agent, "origin"):
        agent.origin = agent.origin.to(device)

    if args.is_load_checkpoint:
        assert (
            args.env_spec.checkpoint_path is not None
        ), "checkpoint path must be specified if is_load_checkpoint is True"
        agent.actor_features.load_state_dict(
            torch.load(
                checkpoints_root
                / os.path.basename(__file__)[: -len(".py")]
                / args.env_spec.checkpoint_path
                / "model_features.pth"
            )
        )
        agent.actor_action.load_state_dict(
            torch.load(
                checkpoints_root
                / os.path.basename(__file__)[: -len(".py")]
                / args.env_spec.checkpoint_path
                / "model_action.pth"
            )
        )
    p_instantiated = torch.nn.Parameter(
        torch.tensor(args.p), requires_grad=args.is_p_learnable
    )
    agent.p = p_instantiated
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    use_sde = isinstance(agent, goalagent.agent.AgentSDE)
    if args.is_calf_filter_on:
        calf_filter = CalfFilter(
            agent=agent,
            nominal_policy=args.env_spec.nominal_policy,
            nominal_std=args.nominal_std,
            safe_decay_parameter=args.safe_decay_parameter,
            is_dynamic_decay_rate=args.is_dynamic_decay_rate,
            p=p_instantiated,
            transform_logprobs=args.transform_logprobs,
        )
    envs = agent.envs
    ##############################
    # Data buffer initialization #
    ##############################
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    if args.is_calf_filter_on:
        obs_last_good = torch.zeros_like(obs).to(device)
        calf_fires = torch.zeros((args.num_steps, args.num_envs)).to(device)
        initial_decays = torch.zeros((args.num_steps, args.num_envs)).to(device)

    #######################
    # Main training loop  #
    #######################
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=[int(np.random.get_state()[1][0])])
    episodic_observations = [next_obs]
    episodic_actions = []
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        if use_sde:
            agent.reset_noise()
        iteration_returns = []

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if use_sde and step % args.sde_sample_freq == 0:
                agent.reset_noise()
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, entropy, value = agent.get_action_and_value(next_obs)
                if args.is_calf_filter_on:
                    if calf_filter.vf_last_good is None:
                        vf_last_good = value.flatten()
                        ob_last_good = next_obs
                    else:
                        vf_last_good = calf_filter.vf_last_good
                        ob_last_good = calf_filter.observation_last_good

                    calf_fire, action, logprob, entropy, value = (
                        calf_filter.accept_or_transform(
                            next_obs,
                            action,
                            logprob,
                            entropy,
                            value,
                        )
                    )
                    if calf_fire.any():
                        initial_decays[step] = calf_filter.vf_last_good - vf_last_good

                        assert torch.allclose(
                            initial_decays[step],
                            agent.get_value(obs[step]) - agent.get_value(ob_last_good),
                        )

                    obs_last_good[step] = calf_filter.observation_last_good
                    calf_fires[step] = calf_fire.flatten()

            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            episodic_actions.append(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)

            if not next_done.any():
                episodic_observations.append(next_obs)

            if args.is_calf_filter_on:
                calf_filter.reset(next_done, iteration, global_step)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        save_episodic_data(
                            info,
                            global_step,
                            episodic_observations,
                            episodic_actions,
                            args.env_spec.observations_names,
                            args.env_spec.actions_names,
                        )
                        episodic_observations = [next_obs.cpu().numpy()]
                        episodic_actions.clear()
                        iteration_returns.append(info["episode"]["r"])

        ##################################
        # Episodes in iteration are done #
        ##################################

        mlflow.log_metric(
            "mean_iteration_return", np.mean(iteration_returns), step=global_step
        )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        if args.is_calf_filter_on:
            b_calf_fires = calf_fires.reshape(-1)
            b_initial_decays = initial_decays.reshape(-1)
            b_obs_last_good = obs_last_good.reshape(
                (-1,) + envs.single_observation_space.shape
            )

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        if args.is_calf_filter_on and calf_filter.assert_decays:
            decays_sum = torch.where(
                calf_fires == 1.0,
                agent.get_value(b_obs) - agent.get_value(b_obs_last_good),
                torch.zeros_like(initial_decays),
            ).sum()
            initial_decays_sum = initial_decays.sum()
            assert torch.allclose(decays_sum, initial_decays_sum)
            print(f"decays_sum={decays_sum}, initial_decays_sum={initial_decays_sum}")

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                if args.is_calf_filter_on:
                    _, _, newlogprob, entropy, newvalue = (
                        calf_filter.accept_or_transform(
                            b_obs[mb_inds],
                            b_actions[mb_inds],
                            newlogprob,
                            entropy,
                            newvalue,
                            b_calf_fires[mb_inds].reshape(-1, 1),
                        )
                    )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)

                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                if agent.pos_def_penalty_coeff is not None:
                    v_loss = (
                        v_loss
                        + agent.pos_def_penalty_coeff
                        * torch.where(newvalue < 0.0, 0.0, newvalue).mean()
                    )

                if args.is_calf_filter_on and calf_filter.force_critic_decay:
                    decays = torch.where(
                        b_calf_fires[mb_inds] == 1.0,
                        agent.get_value(b_obs[mb_inds]).reshape(-1)
                        - agent.get_value(b_obs_last_good[mb_inds]).reshape(-1),
                        torch.zeros_like(b_initial_decays[mb_inds]),
                    )
                    decay_penalty = args.decay_penalty_coef * compute_decay_penalty(
                        decays, b_initial_decays[mb_inds], b_calf_fires[mb_inds]
                    )
                    v_loss = v_loss + decay_penalty

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes

        mlflow.log_metric(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        mlflow.log_metric("losses/value_loss", v_loss.item(), global_step)
        mlflow.log_metric("losses/policy_loss", pg_loss.item(), global_step)
        mlflow.log_metric("losses/entropy", entropy_loss.item(), global_step)
        mlflow.log_metric("losses/old_approx_kl", old_approx_kl.item(), global_step)
        mlflow.log_metric("losses/approx_kl", approx_kl.item(), global_step)
        mlflow.log_metric("losses/clipfrac", np.mean(clipfracs), global_step)
        mlflow.log_metric("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        mlflow.log_metric(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )
        # evaluate_pendulum_policy(agent, global_step=global_step)
        plt.clf()
        plt.cla()
        plt.close()
    envs.close()


if __name__ == "__main__":
    launch()

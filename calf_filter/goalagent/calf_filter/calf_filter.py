from typing import Callable
import matplotlib.pyplot as plt
import torch
import numpy as np
import tempfile
import mlflow


def compute_decay_penalty(decays, calf_fires, initial_decays=None):
    decay_penalty = torch.tensor(0.0).to(decays.device)
    for t in range(len(decays)):
        if calf_fires[t] == 1:
            if initial_decays is not None:
                decay_penalty += torch.min(
                    torch.hstack((decays[t], initial_decays[t].detach()))
                )
            else:
                decay_penalty += decays[t]
    return -decay_penalty


class ActionFilter:
    def __init__(
        self,
        agent,
        nominal_policy: Callable[[torch.Tensor], torch.Tensor],
        safe_decay_parameter: float,
        is_dynamic_decay_rate: bool = False,
        lb_parameter: float = 0.01,
        ub_parameter: float = 80.0,
        assert_decays: bool = False,
        transform_logprobs: bool = False,
        force_critic_decay: bool = False,
    ):
        self.agent = agent
        self.nominal_policy = nominal_policy
        self.safe_decay_parameter = safe_decay_parameter
        self.is_dynamic_decay_rate = is_dynamic_decay_rate
        self.lb_parameter = lb_parameter
        self.ub_parameter = ub_parameter
        self.vf_last_good = None
        self.observation_last_good = None
        self.assert_decays = assert_decays
        self.transform_logprobs = transform_logprobs
        self.force_critic_decay = force_critic_decay

        self.vf_history = []
        self.constraint_history = []

    def reset(self, is_done, current_iteration, global_step):

        if is_done.all():
            if len(self.vf_history) > 0 and len(self.vf_history) == len(
                self.constraint_history
            ):
                plt.clf()
                plt.cla()
                plt.close()
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].plot(np.hstack(self.vf_history))
                axes[0].set_title("Value Function")
                axes[0].set_xlabel("Iteration")
                axes[0].grid()
                axes[1].plot(np.hstack(self.constraint_history))
                axes[1].set_title("Constraint Satisfaction")
                axes[1].set_xlabel("Step")
                fig.tight_layout()
                with tempfile.TemporaryDirectory() as tmpdir:
                    fig.savefig(
                        f"{tmpdir}/effective_lyapunov_function_{global_step:08}.svg"
                    )
                    mlflow.log_artifact(
                        f"{tmpdir}/effective_lyapunov_function_{global_step:08}.svg",
                        "effective_lyapunov_function",
                    )
            self.vf_last_good = None
            self.observation_last_good = None
            self.vf_history = []
            self.constraint_history = []

    def accept_or_transform(
        self,
        observations,
        actions,
        logprobs,
        entropies,
        values,
        calf_fires=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        return (
            torch.ones_like(values),
            actions,
            logprobs,
            entropies,
            values,
        )


class CalfFilter(ActionFilter):
    def __init__(
        self, *args, p: torch.Tensor = 0.8, nominal_std: float = 0.001, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.p = p
        self.nominal_std = nominal_std

    def transform_logprob(self, nominal_actions, logprobs, calf_fires):
        nominal_dist = torch.distributions.Normal(
            nominal_actions,
            scale=torch.full_like(nominal_actions, self.nominal_std),
        )
        nominal_logprobs = nominal_dist.log_prob(nominal_actions).sum(1)

        p = torch.ones_like(logprobs) * torch.nn.functional.sigmoid(self.p)
        one_minus_p = torch.ones_like(p) - p
        vstacked_logprobs = torch.vstack(
            (logprobs + torch.log(one_minus_p), nominal_logprobs + torch.log(p))
        )
        return torch.where(
            calf_fires.reshape(-1) == 1.0,
            logprobs,
            torch.logsumexp(vstacked_logprobs, 0),
        )

    def sample_action_from_mixed_distribution(
        self, actions, nominal_actions, calf_fires
    ):
        bernoulli_params = (
            torch.ones_like(calf_fires)
            * torch.nn.functional.sigmoid(self.p)
            * (1 - calf_fires)
        )
        is_nominal = torch.distributions.Bernoulli(bernoulli_params).sample()
        return is_nominal * nominal_actions + (1 - is_nominal) * actions

    def accept_or_transform(
        self,
        observations,
        actions,
        logprobs,
        entropies,
        values,
        calf_fires=None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        nominal_actions = torch.vstack(
            [
                self.nominal_policy(observation.reshape(1, -1))
                for observation in observations
            ]
        )
        nominal_actions_noise = (
            nominal_actions + torch.randn_like(nominal_actions) * self.nominal_std
        )
        if calf_fires is None:
            # inference
            result_calf_fires = self.is_action_safe(observations, values)
            result_actions = self.sample_action_from_mixed_distribution(
                actions, nominal_actions_noise, result_calf_fires
            )
        else:
            # training
            result_calf_fires = calf_fires
            result_actions = actions

        if self.transform_logprobs:
            logprobs = self.transform_logprob(
                nominal_actions_noise, logprobs, result_calf_fires
            )
        return (
            result_calf_fires,
            result_actions,
            logprobs,
            entropies,
            values,
        )

    def get_decay_threshold(self, state):
        if self.is_dynamic_decay_rate:
            return torch.linalg.norm(state, ord=2) * self.safe_decay_parameter
        else:
            return self.safe_decay_parameter

    def is_action_safe(self, observation, value):
        if self.vf_last_good is None or self.observation_last_good is None:
            self.vf_last_good = value
            self.observation_last_good = observation
            self.vf_history.append(self.vf_last_good.detach().cpu().numpy().flatten())
            self.constraint_history.append(np.array(0.0))
            return torch.zeros_like(value)

        observation_transferred = (
            observation - self.agent.origin
            if self.agent.origin is not None
            else observation
        )
        safe_decay_thr = self.get_decay_threshold(observation_transferred)
        decay = value - self.vf_last_good
        norm = torch.linalg.norm(observation_transferred, ord=2)

        decay_condition = decay > safe_decay_thr
        low_kappa = norm * self.lb_parameter
        up_kappa = self.ub_parameter * norm
        kappa_condition = low_kappa < -value < up_kappa

        are_constraints_satisfied = (
            decay_condition and kappa_condition
        ) * torch.ones_like(value)

        if are_constraints_satisfied.all():
            self.vf_last_good = value
            self.observation_last_good = observation

        self.vf_history.append(self.vf_last_good.detach().cpu().numpy().flatten())
        self.constraint_history.append(
            are_constraints_satisfied.detach().cpu().numpy().flatten()
        )
        return are_constraints_satisfied

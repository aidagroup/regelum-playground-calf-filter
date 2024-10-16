# CALF Filter

## Overview

The CALF filter is an implementation of a reinforcement learning algorithm that combines safe control with learning-based approaches. This sub-project is part of a larger reinforcement learning framework and focuses on implementing the CALF algorithm within the regelum-playground repository.

## Project Structure

The CALF filter sub-project is organized as follows:

1. Main implementation files:
   - `calf_filter/goalagent/calf_filter/calf_filter.py`: Contains the core implementation of the CALF filter algorithm.
   - `src/scenario/calf.py`: Defines the CALF scenario for running the algorithm.
   - `src/scenario/calf_agent/calfv.py`: Implements the CALF agent with value function.
   - `src/scenario/calf_agent/calfq.py`: Implements the CALF agent with Q-function.

2. Configuration files:
   - `presets/scenario/calf.yaml`: Configuration for the CALF scenario.
   - `presets/agent_calf/*.yaml`: Agent-specific configurations for different environments.

3. Environment and utility files:
   - `calf_filter/goalagent/env/`: Contains environment-specific implementations.
   - `calf_filter/goalagent/utils/`: Utility functions for the project.

4. Main execution script:
   - `calf_filter/ppo.py`: The entry point for running experiments.

## How It Works

The CALF filter algorithm combines a nominal (safe) policy with a learned policy to ensure safe exploration and performance improvement. Here's a high-level overview of how it works:

1. Initialization:
   - A nominal policy is defined for safe control.
   - A critic is initialized.
   - A replay buffer is set up to store experiences.

2. Action Selection:
   - The agent computes both the nominal action and the learned action.
   - A relaxation probability determines whether to use the nominal or learned action.
   - The relaxation probability is adjusted over time to gradually shift from safe to learned actions.

3. Learning:
   - The critic is updated using temporal difference learning.
   - The actor (policy) is optimized to maximize the critic's output.
   - Constraints are applied to ensure safety and stability.

4. Safety Mechanisms:
   - The CALF filter checks if the selected action satisfies safety constraints.
   - If constraints are violated, the nominal action is used instead.

## Underlying Algorithm

The CALF algorithm is based on the following key principles:

1. Safe Exploration: It uses a nominal policy to ensure safe actions during initial exploration (or uses learned policy in case of PPO for instance).
2. Constrained Learning: The critic and actor are updated with safety constraints to maintain stability.
3. Adaptive Relaxation: The algorithm gradually shifts from the nominal policy to the learned policy to some degree as training progresses.
4. Value Function Constraints: It enforces constraints on the value function to ensure safety and stability properties.

The algorithm combines concepts from Lyapunov-based control, actor-critic methods, and constrained optimization to achieve safe and efficient learning.

## How to Launch

To run a CALF filter experiment, use the following steps:

1. Use python 3.10.

2. Ensure all dependencies are installed (run [`pip install -r requirements.txt`](../requirements.txt) in the repo root).

3. `cd calf_filter`

4. Run the experiment using the `ppo.py` script with the appropriate configuration:

    ```
    python ppo.py is_calf_filter_on=True env=pendulum_determ_reset agent=default 
    ```


# Copyright (c) 2023-2024 La Javaness (https://lajavaness.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN 2023 LJN Agent, a repository for the winning agent of L2RPN 2023 competition. It is a submodule contribution to the L2RPN Baselines repository.

import os

import numpy as np
from grid2op.gym_compat import BoxGymObsSpace, GymEnv

from .gym_assets.action_space import GlobalTopoActionSpace
from .LJNAgent import LJNAgent, LJNAgentTopoNN
from .utils import NN_ACT_SPACE_DIR


def make_agent_challenge(env, this_directory_path):
    agent = LJNAgent(env.action_space, env)
    return agent


def make_agent_topoNN(env, this_directory_path):

    gym_env = GymEnv(env)
    gym_env.observation_space.close()
    gym_env.observation_space = BoxGymObsSpace(env.observation_space, ["rho"])
    gym_env.action_space.close()
    act_space = np.load(
        os.path.join(NN_ACT_SPACE_DIR, "action_12_unsafe_nn.npz"), allow_pickle=True
    )["g2op_id_actions"]
    gym_env.action_space = GlobalTopoActionSpace(act_space, env.action_space)

    agent = LJNAgentTopoNN(
        env.action_space,
        env,
        gym_env,
        topk=20,
        model_path=os.path.join(this_directory_path, "models", "RL_training_PPO.zip"),
    )

    return agent

# LJN Agent - A modular agent for combined continuous control and topology management on large grid

[![Static Badge](https://img.shields.io/badge/La_Javaness-blue?style=flat&labelColor=FFFFFF&color=0000FF)](https://www.lajavaness.com/)
![Static Badge](https://img.shields.io/badge/license-MPL%202.0-black)

This agent is a refactored version of the winning agent of the L2RPN 2023 IDF AI challenge co-organized by RTE and Paris Region. It comes as an enhanced combination of pre-existing baselines directly inspired by the curriculum agent and the optimCVXPY agent with additional features. Its development was guided towards the challenge's specific requirements. Especially, it combines continuous control and bus reconfiguration decision making with additional heuristics to operate the grid safely at low cost (i.e. maximizing an operational score) while maximizing the use of renewable energy available at all time. This refactor tries to use, as much as possible, pre-existing components from the grid2op package, as well as l2rpn_baselines. 

This agent is highly competitive on large complex environment while remaining computationally efficient (max. decision time < 3 seconds).

Additionally, we included training scripts for imitation and reinforcement learning for substation reconfiguration.

For more insight on our method and associated results, you can check out [our Medium blog post](https://lajavaness.medium.com/how-we-built-the-winning-real-time-autonomous-agent-for-power-grid-management-in-the-l2rpn-41ab3cfaddbd).

**Content**
1. [Authors](#authors)
2. [Setup](#setup)
3. [How to Run the Agent](#how-to-run-the-agent)
   - [Training on custom environment](#training-on-a-custom-environment)
   - [Evaluate](#evaluate)
   - [Parameters](#parameters)
4. [Acknowledgments](#acknowledgments)
5. [License](#license)

## Authors

- Jules SINTES, Data-Scientist at La Javaness - [jules\@lajavaness.com](mailto:jules@lajavaness.com?subject=Test) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jules-sintes/)
- Van Tuan DANG, Phd, Data-Scientist at La Javaness - [van.tuan\@lajavaness.com](mailto:van.tuan@lajavaness.com?subject=Test) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/van-tuan-dang-7b0b4444/)


## Setup

The dependecies can be installed simply using `pip install -r requirements.txt`
This agent has been developed for the particular environment of the L2RPN Ile-de-France challenge 2023, hence the provided action set is based on the environment with id `l2rpn_idf_2023` and actions have been cherry-picked for this particular setup aiming at optimizing the associated scoring system.

## How to collect data using Curriculum Agent

To build custom reduced action space for the particular environment of the challenge, we are leveraging the great work done at the Fraunhofer Institute with their implementation of the [Curiculum Agent](https://github.com/FraunhoferIEE/curriculumagent/tree/master), directly inspired by [BinBinChen's solution](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution/tree/master) from 2020's challenge.

Teachers scripts from the curriculum agent repository have been running on for nearly 5 days on a 128 CPUs instance to get a representative distribution of most used actions and then we kept actions spaces separated for different cases :

- Teacher 1 and 2 collect actions for general unsafe states either following a line disconnection (1) or in case of an overflow (2).
- Teacher N-1 collects actions to help keep the grid safe in case of line disconnection.

We also built 2 additional action spaces for safe and intermediate state by varying rho threshold from teacher n-1. These action space are provided with this repository but actually not used in the final implementation for the challenge.

Finally, we get 4 action spaces :
1. action_space_12_unsafe corresponding to teacher 1 and teacher 2
2. action_space_n1_unsafe corresponding to teacher n-1 in case unsafe : `rho_threshold_0 = 0.99, rho_threshold_1 = 1.99`
3. action_space_n1_interm corresponding to teacher n-1 with intermediate rho thresholds : `rho_threshold_0 = 0.9, rho_threshold_1 = 0.99`
4. action_space_n1_safe corresponding to teacher n-1 with a lower rho thresholds for safe state : `rho_threshold_0 = 0.6, rho_threshold_1 = 0.9`

## How to run the agent

### Quickstart

To use this agent as it is as a baseline, we recommend using `make_agent_challenge` or `make_agent_topoNN` to directly instantiate agent with default parameters.

```python
import grid2op
from lightsim2grid.lightSimBackend import LightSimBackend

from LJNAgent import make_agent_topoNN
from LJNAgent.modules.rewards import MaxRhoReward

# Instantiate environment
# Our agent is using greedy search mechanism based on a reward function. Hence one should specify a relevent reward function and pass it as an arg for the env.
env = grid2op.make('l2rpn_idf_2023',backend = LightSimBackend(), reward_class = MaxRhoReward)

# Instantiate the agent

# Heuristic version of our agent used during the challenge 
agent_heuristic = make_agent_challenge(env,"this/directory/path")

# Enhanced version with neural network based policy for topological decision for unsafe state
agent_topo_nn = make_agent_topo_nn(env,"this/directory/path")
```

If you want to modify hyperparameters of the agent or use a custom NN policy, you can use these functions as templates for correctly instantiating the agent.

### Modular architecture - Build your own agent

This agent comes with a modular architecture providing abstraction building block for various heuristics inspired by pre-existing baselines and implemented in our agent for the L2RPN Challenge. Especially, we implemented a `BaseModule` object used as a wrapper to combine different decision making mechanism in a single Grid2Op agent. 

Currently provided modules are :
| Module  | Type | Description |
|---|---|---|
| BaseModule | Core module | Basic wrapper inheriting from BaseAgent class. The module has the same logic as the agent in Grid2Op but it adds a `get_act` methods that can combine a base action with module's decision making mechanism. |
| GreedyModule | Core module | Greedy search module, enhanced module wrapper base on GreedyAgent in grid2op. |
| RecoPowerlineModule | GreedyModule | Reconnect a disconnected powerline based on greedy mechanism (highest specified reward). |
| RecoPowerlinePerAreaModule | GreedyModule | Reconnect at most 1 powerline per-area based on greedy mechanism. |
| RecoverInitTopoModule | GreedyModule | Find an action suitable to get back to initial topology (every elements connected to bus 1) |
| TopoSearchModule | GreedyModule | Find the best topological action (highest reward) based on a provided list of actions. |
| ZoneBasedTopoSearchModule | TopoSearchModule | Find the best topological reconfiguration for each sub-area and combines actions. |
| TopoNNTopKModule | GreedyModule | Use a neural-network based actor-critic policy to predict the top-k actions (logits) and find the best action among top-k prediction. |
| OptimModule | BaseModule | Convex optimization for redispatch/curtailment/storage. Enhanced implementation of the OptimCVXPY from l2rpn_baselines. |

Using those module wrapper, one can create a Grid2Op Agent that use high-level rules orchestrate hierarchical complex heuristic decision making.

For example :
```python
class MyModularAgent(BaseAgent):
    def __init__(
        self,
        action_space: ActionSpace,
        env,
        rho_danger: float = 0.99,
        rho_safe: float = 0.9,
    ):
        BaseAgent.__init__(self, action_space=action_space)
        # Environment
        self.env = env
        self.rho_danger = rho_danger
        self.rho_safe = rho_safe
        # Sub-modules
        # Heuristic
        self.reconnect = RecoPowerline(self.action_space)
        self.recover_topo = RecoverInitTopoModule(self.action_space)
        # Continuous control
        self.optim = OptimModule(env, self.action_space)

    def act(
        self, observation: BaseObservation, reward: float, done: bool = False
    ) -> BaseAction:
        start = time.time()

        # Init action with "do nothing"
        act = self.action_space()

        # Try to perform reconnection if necessary
        act = self.reconnect.get_act(observation, act, reward)

        if observation.rho.max() > self.rho_danger:
                act = self.optim.get_act(observation, act, reward)
        elif _obs.rho.max() < self.rho_safe:
                # Try to find a recovery action when the grid is safe
                act = self.recover_topo.get_act(
                observation, act, reward, rho_threshold=0.8
            )
        return act
```

### Train Neural-network ActorCritic policies

The training folder contains scripts to use pairs of observation and actions with behaviour cloning algorithms, a simple imitation learning technique. This supervised training is followed by a Proximal Policy Optimization (RL method) to further train the policies. 

The resulting neural network based policies can be used in place of the search algorithms on reduced action spaces. This highly reduces the computing time for topological actions, hence making the agent even more computationnally efficient.
However, one should note that in order to be trained properly, it requires a large samples.

We provide 2 training scripts :
1. Imitation supervised learning : based on a dataset of observation / action the model learns to predict the best action for a given observation.
2. Further RL training : As the policy used during supervised learning is an actor critic policy, we are leveraging this architecture to perform further training using PPO algorithm. We observe that this training enhance the performance of our agent.

Baselines models are provided for both training steps. And can be used directly with the `make_agent_topoNN` function.

_Note : We expect to improve the imitation learning process in the coming months_

### Evaluate

Given the provided action space, the evaluation should be run on environment `l2rpn_idf_2023` but one can try with its own action space set on a different environment.

**Command line** 
```bash
python l2rpn_baselines/LJNAgent/evaluate.py --nb_episode=2 --nb_process=1 --verbose=True
```

**Python code**
```python
from LJNAgent import LJNAgent, evaluate
from lightsim2grid.lightSimBackend import LightSimBackend # Recommended for faster simulation
from grid2op import make

env = make("l2rpn_idf_2023", backend = LightSimBackend())
evaluate(env,
        logs_path=None,
        nb_episode=10,
        nb_process=1,
        max_steps=-1,
        verbose=True,
        save_gif=False)

```

### Training on a custom environment

The training process to cherry pick the discrete bus reconfiguration mainly relies on exhaustive simulation and greedy search on the complete action space. It mainly used the work done for the [curriculum agent](https://github.com/FraunhoferIEE/curriculumagent/tree/master) with custom implementations for the specific needs within the frame of the L2RPN IDF 2023 challenge. This process is computionally intensive and it is recommended to use large CPU resources when trying to generate action space on large environment. 

The agent requires 4 different action spaces described as follow :
| Name | Description | Size for l2rpn_idf_2023 |
|---|---|---|
| action_12_unsafe | The grid is in unsafe state with no line disconnection | 421 |
| action_N1_interm | The grid is in an untermediate state and there is a line disconnection | 136 |
| action_N1_unsafe | The grid is in a unsafe state and there is a line disconnection | 909 |
| action_N1_safe | The grid is in a safe state and there is a line disconnection | 50 |

_NOTE : The size of the provided actions spaces are given as an indicator to build a custom action space for a different environment. The complete size of the action space of bus reconfiguration is _70k_ unitary actions for this environment. The actions are encoded as vectors, as implemented in the grid2op package._

### Parameters

#### High Level heuristics

| Parameter                      | Value                      | Description                                   |
| ------------------------------ | -------------------------- | --------------------------------------------- |
| rho_danger                     | 0.99                       | Line capacity danger threshold               |
| rho_safe                       | 0.9                        | Line capacity safe state threshold           |
| areas                          | True                       | Flag for considering areas                   |
| sim_range_time_step            | 1                          | Number of time-step to simulate when checking action choice|

#### Continuous opitmization

| Parameter                      | Value                      | Description                                   |
| ------------------------------ | -------------------------- | --------------------------------------------- |
| margin_th_limit                | 0.93                       | Margin thermal limit                         |
| penalty_curtailment_unsafe     | 15                         | Penalty for curtailment in unsafe conditions |
| penalty_redispatching_unsafe   | 0.005                      | Penalty for redispatching in unsafe conditions|
| penalty_storage_unsafe         | 0.0075                     | Penalty for storage in unsafe conditions     |
| penalty_curtailment_safe       | 0.0                        | Penalty for curtailment in safe conditions   |
| penalty_redispatching_safe     | 0.0                        | Penalty for redispatching in safe conditions |
| penalty_storage_safe           | 0.0                        | Penalty for storage in safe conditions       |
| weight_redisp_target           | 1.0                        | Weight for redispatching target               |
| weight_storage_target          | 1.0                        | Weight for storage target                     |
| weight_curtail_target          | 1.0                        | Weight for curtailment target                 |
| margin_rounding                | 0.01                       | Margin rounding                              |
| margin_sparse                  | 5e-3                       | Sparse margin                                |
| max_iter                       | 100000                     | Maximum number of iterations of the solver   |


#### Bus reconfiguration

There is a tradeoff between computationnal efficiency and size of action space: Larger - _high-quality_ - action space should increase agent's performance while the decision time might significantly decrease. However, we found out that small cherry-picked action spaces are well suited to handle most of the situations.

## Reference

This baseline agent is mainly inspired by the work done on :
- [The OptimCVXPY agent from this package](https://github.com/rte-france/l2rpn-baselines/tree/master/l2rpn_baselines/OptimCVXPY)
- [Binbinchen's solution to the NEURIPS 2020 challenge](https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution)
- [The curriculum agent by Fraunhofer Institute](https://github.com/FraunhoferIEE/curriculumagent)

## License

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE](LICENSE) file for details.

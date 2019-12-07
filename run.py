from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.agents import random_agent
from pysc2.env import sc2_env
from pysc2.tests import utils
from pysc2.lib import actions as sc2_actions
from pysc2.lib import features
from pysc2.maps import lib
from absl import flags
from absl.testing import absltest
from pysc2.env import environment
import time
import sys
import os
from pysc2.env import run_loop
from pysc2.lib import run_parallel
from future.builtins import range  # pylint: disable=redefined-builtin

steps1 = 1000000


# New map created
class MiniGame(lib.Map):
    directory = "mini_games"
    players = 2
    score_index = 0
    step_mul = 8
    game_steps_per_episode = steps1 * step_mul // 2


mini_games = ["DefeatRoaches"]  # 120s

for name in mini_games:
    globals()[name] = type(name, (MiniGame,), dict(filename=name))

FLAGS = flags.FLAGS
FLAGS(sys.argv)

agent_format1 = sc2_env.AgentInterfaceFormat(
    feature_dimensions=sc2_env.Dimensions(
        screen=(32, 32), minimap=(32, 32)))

# agent_format2 = sc2_env.AgentInterfaceFormat(
#     feature_dimensions=sc2_env.Dimensions(
#         screen=(32, 32), minimap=(32, 32)))

env = sc2_env.SC2Env(
    map_name=mini_games[0],
    players=[sc2_env.Agent(sc2_env.Race.terran),
             sc2_env.Agent(sc2_env.Race.zerg)],
    step_mul=MiniGame.step_mul,
    visualize=False,
    agent_interface_format=agent_format1)

agents = [random_agent.RandomAgent() for _ in range(MiniGame.players)]

# run_loop.run_loop(agents, env, max_frames=1000, max_episodes=10000)
obs_specs = env.observation_spec()
actions_specs = env.action_spec()

for agent, obs_spec, act_spec in zip(agents, obs_specs, actions_specs):
    agent.setup(obs_spec, act_spec)

total_episodes = 0
score1 = []
score2 = []

while not total_episodes == 100:
    total_episodes += 1
    print(total_episodes)

    time_steps = env.reset()
    for a in agents:
        a.reset()

    for t in range(2000):
        actions = [agent.step(time_step) for agent, time_step in zip(agents, time_steps)]

        if time_steps[0].last():
            break

        time_steps = env.step(actions=actions)
    score1.append(time_steps[0][3]["score_cumulative"][0])
    score2.append(time_steps[1][3]["score_cumulative"][0])
    print(time_steps[0][3]["score_cumulative"][0], time_steps[1][3]["score_cumulative"][0])

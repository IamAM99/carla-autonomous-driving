from model import DQNAgent
from environment import CarlaEnv

env = CarlaEnv()
try:
    agent = DQNAgent(env)
    agent.train()
finally:
    env.clear()
    print("Cleared")
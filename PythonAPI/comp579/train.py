import time

from model import DQNAgent
from environment import CarlaEnv


if __name__=="__main__":
    env = CarlaEnv()
    agent = DQNAgent(env)

    try:
        agent.train()

    finally:
        agent.kill_thread()

        try:
            env.clear(final=True)
            print("Cleared everything")
        except RuntimeError:
            print("Failed to clear the environment. Trying again in 2 seconds.")
            time.sleep(2.0)
            pass
        
    
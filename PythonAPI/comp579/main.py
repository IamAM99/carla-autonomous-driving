import time
from environment import CarlaEnv

def main():
    env = CarlaEnv()
    env.reset()

    try:
        
        _, reward, _, _ = env.step(0)
        print(reward)

        time.sleep(2)

        _, reward, _, _ = env.step(1)
        print(reward)
        time.sleep(1)

        _, reward, _, _ = env.step(0)
        print(reward)
        time.sleep(1)

        _, reward, _, _ = env.step(2)
        print(reward)
        time.sleep(1)

        _, reward, _, _ = env.step(0)
        print(reward)
        time.sleep(2)

    finally:
        env.clear()

if __name__=="__main__":
    main()

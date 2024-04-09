import time
from environment import CarlaEnv

def main():
    env = CarlaEnv()
    env.reset()

    try:
        
        states, reward, _, _ = env.step(0)
        print(f"R = {reward:.2f}")
        print(states["waypoints"])
        time.sleep(2)

        states, reward, _, _ = env.step(1)
        print(f"R = {reward:.2f}")
        print(states["waypoints"])
        time.sleep(1)

        states, reward, _, _ = env.step(0)
        print(f"R = {reward:.2f}")
        print(states["waypoints"])
        time.sleep(1)

        states, reward, _, _ = env.step(2)
        print(f"R = {reward:.2f}")
        print(states["waypoints"])
        time.sleep(1)

        states, reward, _, _ = env.step(0)
        print(f"R = {reward:.2f}")
        print(states["waypoints"])
        time.sleep(2)

    finally:
        env.clear()
        print("Cleared successfully")

if __name__=="__main__":
    main()

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from stable_baselines3 import PPO
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
import os
import warnings

warnings.filterwarnings("ignore")

# Setup a directory to store monitor logs
log_dir = "./ppo_learning_logs/"
os.makedirs(log_dir, exist_ok=True)

# 🚀 Custom environment loader with monitor
def make_env():
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    env = TimeLimit(env, max_episode_steps=100)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return env

# Wrap the environment
env = DummyVecEnv([make_env])
env = VecTransposeImage(env)

# ✅ Train PPO agent
print("🚀 Training Started...")
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
print("✅ Training Completed.")

# 📈 Plot learning curve (reward vs episode)
print("\n📊 Plotting Learning Curve...")
df = pd.read_csv(os.path.join(log_dir, "monitor.csv"), skiprows=1)
plt.plot(df["r"])
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Learning Curve")
plt.grid(True)
plt.show()

# 🎞 Final animation of trained agent
print("\n🎥 Rendering Trained Agent...")
eval_env = make_env()
obs, _ = eval_env.reset()
fig = plt.figure()
ims = []
episode_reward = 0

for step in range(50):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = eval_env.step(action)
    episode_reward += reward
    im = plt.imshow(eval_env.render(), animated=True)
    ims.append([im])
    if terminated or truncated:
        break

print(f"🎯 Final Episode Reward: {episode_reward}")
ani = animation.ArtistAnimation(fig, ims, interval=200, repeat=False)
plt.show()
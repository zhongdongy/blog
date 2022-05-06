### Introduction

At Hugging Face, we are contributing to the ecosystem for Deep Reinforcement Learning researchers and enthusiasts. And thanks to your insightful feedback, we developed a new integration of **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** to the Hugging Face Hub.

**[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** is one of the most popular PyTorch Deep Reinforcement Learning library that makes it easy to train and test your agents in a variety of environments (Gym, Atari, MuJoco, Procgen...).

With this new version, you can with one line of code, `package_to_hub()` , evaluate, record a replay, generate a model card of your agent and push it to the hub. And load powerful models from the community.

[Model Card]

In this article, we’re going to show how you can do it.

### Installation

To use stable-baselines3 with Hugging Face Hub, you just need to install this library:

```
pip install huggingface_sb3
```

### **Sharing a model to the Hub**

In just a minute, you can get your model evaluated and pushed in the Hub with a nice replay video and a insightful model card.

[Picture model card]

With `package_to_hub()`**we'll save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub**. It currently **works for Gym and Atari environments** . If you use another environment, you should use `push_to_hub()`instead.

First, you need to be logged in to Hugging Face to upload a model:

- If you're using Colab/Jupyter Notebooks:

```
from huggingface_hub import notebook_login
notebook_login()

```

- Else:

```
huggingface-cli login
```

Then, in this example, we train a PPO agent to play LunarLander-v2, and by using `package_to_hub()` **we'll save, evaluate, generate a model card and record a replay video of the agent before pushing the repo**`ThomasSimonini/ppo-LunarLander-v2` **to the Hub.** 
 

```
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from huggingface_sb3 import load_from_hub, package_to_hub

# Create the environment
env_id = 'LunarLander-v2'
env = DummyVecEnv([lambda: gym.make(env_id)])

# Create the evaluation env
eval_env = DummyVecEnv([lambda: gym.make(env_id)])

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=int(5000))

# This method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
package_to_hub(model=model,
               model_name="ppo-LunarLander-v2",
               model_architecture="PPO",
               env_id=env_id,
               eval_env = eval_env,
               repo_id="ThomasSimonini/ppo-LunarLander-v2",
               commit_message="Push the model")
```

### **Finding Models**

We’re currently uploading new models every week. On top of this, you can find **[all stable-baselines-3 models from the community here](https://huggingface.co/models?other=stable-baselines3).**

When you found the model you need, you just have to copy the repository id:

[Picture repo id]

### **Download a model from the Hub**

With this update, you **can now very easily load a saved model from Hub to Stable-baselines3.**

In order to do that you just need **to copy the repo-id that contains your saved model and the name of the saved model zip file in the repo.**

For instance`sb3/demo-hf-CartPole-v1`:

```
import gym

from huggingface_sb3 import load_from_hub
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Retrieve the model from the hub## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})## filename = name of the model zip file from the repository
checkpoint = load_from_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
)

# If you're using Python 3.7, this agent was trained with 3.8 to avoid Pickle errors:
custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
model= PPO.load(checkpoint, custom_objects=custom_objects)

# Evaluate the agent and watch it
eval_env = gym.make("CartPole-v1")
mean_reward, std_reward = evaluate_policy(
    model, eval_env, render=True, n_eval_episodes=5, deterministic=True, warn=False
)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
```

Try it out and share your models with the community!

### ****What's next?****

In the coming weeks and months, we plan on supporting other tools from the ecosystem:

- Integrating **[RL-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo)**
- Uploading **[RL-trained-agents models](https://github.com/DLR-RM/rl-trained-agents)** into the Hub: a big collection of pre-trained Reinforcement Learning agents using stable-baselines3
- Integrating other Deep Reinforcement Learning libraries (RLlib, MLAgents)
- Implementing Convolutional Decision Transformers For Atari
- And more to come 🥳

The best way to keep in touch is to **[join our discord server](https://discord.gg/YRAq8fMnUG)** to exchange with us and with the community.

### Conclusion

We're excited to see what you're working on with Stable-baselines3 and try your models in the Hub 😍.

And we would love to hear your feedback 💖. 📧 Feel free to **[reach us](mailto:thomas.simonini@huggingface.co)**.

Finally, we would like to thank the SB3 team and in particular **[Antonin Raffin](https://araffin.github.io/)** for their precious help for the integration of the library 🤗.

### **Would you like to integrate your library to the Hub?**

This integration is possible thanks to the **`[huggingface_hub](https://github.com/huggingface/huggingface_hub)`** library which has all our widgets and the API for all our supported libraries. If you would like to integrate your library to the Hub, we have a **[guide](https://huggingface.co/docs/hub/adding-a-library)** for you!

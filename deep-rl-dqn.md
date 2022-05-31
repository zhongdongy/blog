---
title: "Deep Q-Learning with Space Invaders"
thumbnail: /blog/assets/78_deep_rl_dqn/thumbnail.gif
---

<html>
<head>
<style>
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
</style>
<h1>Deep Q-Learning with Space Invaders</h1>
<h2>Unit 3, of the the <a href="https://github.com/huggingface/deep-rl-class">Deep Reinforcement Learning Class with Hugging Face 🤗</a></h2>

<div class="author-card">
    <a href="/ThomasSimonini">
        <img class="avatar avatar-user" src="https://aeiljuispo.cloudimg.io/v7/https://s3.amazonaws.com/moonup/production/uploads/1632748593235-60cae820b1c79a3e4b436664.jpeg?w=200&h=200&f=face" title="Gravatar">
        <div class="bfc">
            <code>ThomasSimonini</code>
            <span class="fullname">Thomas Simonini</span>
        </div>
  </a>
</div>

</head>

<body>

*This article is part of the Deep Reinforcement Learning Class. A free course from beginner to expert. Check the syllabus [here.](https://github.com/huggingface/deep-rl-class)*
---

[In the second unit, part 2](https://huggingface.co/blog/deep-rl-q-part2), **we learned about Q-learning** and we had great results with FrozenLake-v1 and Taxi-v3.

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/envs.gif" alt="Environments"/>
</figure>
  
However, Q-Learning can become easily ineffective in environments with a large State Space, since we'll not be able to learn about every single state, action value pair and hold all of them in a q-table.
  
So today, we're going to study about Deep Q-Learning, a deep reinforcement learning algorithm that instead of using a Q-table, use a Neural Network that takes a state and approximates Q-values for each action based on that state.

And we'll train our first Deep Q-Learning agent to play Space Invaders and other Atari environments using [RL-Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), the training framework for Reinforcement Learning using Stable Baselines3.
A version of Stable-baselines-3 that provides  scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

TODO: ADD ATARI ENVIRONMENTS GIF

So let’s get started! 🚀
  
- [From Q-Learning to Deep Q-Learning](#)
- [The Deep Q-Network](#)
- [How to stabilize the training](#)
- [The Deep Q-Learning algorithm](#)

  
## **From Q-Learning to Deep Q-Learning**
### Why we can't rely on Q-Learning for large State Space environments?
  
Last time, we learned that **Q-Learning is the algorithm we use to train our Q-Function**, an **action-value function** that determines the value of being at a particular state and taking a specific action at that state.

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-function.jpg" alt="Q-function"/>
  <figcaption>Given a state and action, our Q Function outputs a state-action value (also called Q-value)</figcaption>
</figure>

The **Q comes from "the Quality" of that action at that state.**

Internally, our Q-function has **a Q-table, a table where each cell corresponds to a state-action value pair value.** Think of this Q-table as **the memory or cheat sheet of our Q-function.**

The problem is that Q-Learning is a *tabular method*. Aka, a problem in which the state and actions spaces **are small enough for approximate value functions to be represented as arrays and tables**. And this is **not scalable**.

Since imagine what we're going to do today, we're going to train an agent to learn to play Space Invaders using the frames as input. Therefore, the state space is gigantic (billions of different frames possible).

Q-Learning was working well with small state space environments like:
- FrozenLake, we had 14 states.
- Taxi-v3 we had 500 states.
  
But when we use 


But we might ask is tabular method scale?

- In Frozen lake, we have 14 states.
- In Taxi-v3, we have 500 states.

But, in discrete environment:

- Tetris is 10^60 states
- Atari game: 10^16992 states

  
  
  
Imagine what we’re going to do today. We’ll create an agent that learns to play Doom. Doom is a big environment with a gigantic state space (millions of different states). Creating and updating a Q-table for that environment would not be efficient at all.

The best idea in this case is to create a [neural network](http://neuralnetworksanddeeplearning.com/) that will approximate, given a state, the different Q-values for each action.
  
  
  
If we take this maze example:

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Maze-1.jpg" alt="Maze example"/>
</figure>

The Q-Table is initialized. That's why all values are = 0. This table **contains, for each state, the four state-action values.**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Maze-2.jpg" alt="Maze example"/>
</figure>

Here we see that the **state-action value of the initial state and going up is 0:**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Maze-3.jpg" alt="Maze example"/>
</figure>

Therefore, Q-function contains a Q-table **that has the value of each-state action pair.** And given a state and action, **our Q-Function will search inside its Q-table to output the value.**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-function-2.jpg" alt="Q-function"/>
  <figcaption>Given a state and action pair, our Q-function will search inside its Q-table to output the state-action pair value (the Q value).</figcaption>
</figure>

If we recap, *Q-Learning* **is the RL algorithm that:**

- Trains *Q-Function* (an **action-value function**) which internally is a *Q-table* **that contains all the state-action pair values.**
- Given a state and action, our Q-Function **will search into its Q-table the corresponding value.**
- When the training is done, **we have an optimal Q-function, which means we have optimal Q-Table.**
- And if we **have an optimal Q-function**, we **have an optimal policy** since we **know for each state what is the best action to take.**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/link-value-policy.jpg" alt="Link value policy"/>
</figure>

But, in the beginning, **our Q-Table is useless since it gives arbitrary values for each state-action pair** (most of the time, we initialize the Q-Table to 0 values). But, as we'll **explore the environment and update our Q-Table, it will give us better and better approximations.**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-learning-1.jpg" alt="Q-learning"/>
  <figcaption>We see here that with the training, our Q-Table is better since, thanks to it, we can know the value of each state-action pair.</figcaption>
</figure>

So now that we understand what Q-Learning, Q-Function, and Q-Table are, **let's dive deeper into the Q-Learning algorithm**.

### **The Q-Learning algorithm**

This is the Q-Learning pseudocode; let's study each part and **see how it works with a simple example before implementing it.** Don't be intimidated by it, it's simpler than it looks! We'll go over each step.

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-learning-2.jpg" alt="Q-learning"/>
</figure>

**Step 1: We initialize the Q-Table**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-learning-3.jpg" alt="Q-learning"/>
</figure>

We need to initialize the Q-Table for each state-action pair. **Most of the time, we initialize with values of 0.**

**Step 2: Choose action using Epsilon Greedy Strategy**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-learning-4.jpg" alt="Q-learning"/>
</figure>

Epsilon Greedy Strategy is a policy that handles the exploration/exploitation trade-off.

The idea is that we define epsilon ɛ = 1.0:

- *With probability 1 — ɛ* : we do **exploitation** (aka our agent selects the action with the highest state-action pair value).
- With probability ɛ: **we do exploration** (trying random action).

At the beginning of the training, **the probability of doing exploration will be huge since ɛ is very high, so most of the time, we'll explore.** But as the training goes on, and consequently our **Q-Table gets better and better in its estimations, we progressively reduce the epsilon value** since we will need less and less exploration and more exploitation.

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-learning-5.jpg" alt="Q-learning"/>
</figure>

**Step 3: Perform action At, gets reward Rt+1 and next state St+1**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-learning-6.jpg" alt="Q-learning"/>
</figure>

**Step 4: Update Q(St, At)**

Remember that in TD Learning, we update our policy or value function (depending on the RL method we choose) **after one step of the interaction.**

To produce our TD target, **we used the immediate reward Rt+1 plus the discounted value of the next state best state-action pair** (we call that bootstrap).

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-learning-7.jpg" alt="Q-learning"/>
</figure>

Therefore, our Q(St, At) **update formula goes like this:**

  <figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Q-learning-8.jpg" alt="Q-learning"/>
</figure>

It means that to update our Q(St,At):

- We need St, At, Rt+1, St+1.
- To update our Q-value at a given state-action pair, we use the TD target.

How do we form the TD target? 
1. We obtain the reward after taking the action Rt+1.
2. To get the **best next-state-action pair value**, we use a greedy policy to select the next best action. Note that this is not an epsilon greedy policy, this will always take the action with the highest state-action value.

Then when the update of this Q-value is done. We start in a new_state and select our action **using our epsilon-greedy policy again.**

**It's why we say that this is an off-policy algorithm.**

### **Off-policy vs On-policy**

The difference is subtle:

- *Off-policy*: using **a different policy for acting and updating.**

For instance, with Q-Learning, the Epsilon greedy policy (acting policy), is different from the greedy policy that is **used to select the best next-state action value to update our Q-value (updating policy).**


<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/off-on-1.jpg" alt="Off-on policy"/>
  <figcaption>Acting Policy</figcaption>
</figure>

Is different from the policy we use during the training part:


<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/off-on-2.jpg" alt="Off-on policy"/>
  <figcaption>Updating policy</figcaption>
</figure>

- *On-policy:* using the **same policy for acting and updating.**

For instance, with Sarsa, another value-based algorithm, **the Epsilon-Greedy Policy selects the next_state-action pair, not a greedy policy.**


<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/off-on-3.jpg" alt="Off-on policy"/>
    <figcaption>Sarsa</figcaption>
</figure>

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/off-on-4.jpg" alt="Off-on policy"/>
</figure>

## **A Q-Learning example**

To better understand Q-Learning, let's take a simple example:

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Maze-Example-2.jpg" alt="Maze-Example"/>
</figure>

- You're a mouse in this tiny maze. You always **start at the same starting point.**
- The goal is **to eat the big pile of cheese at the bottom right-hand corner** and avoid the poison. After all, who doesn't like cheese?
- The episode ends if we eat the poison, **eat the big pile of cheese or if we spent more than five steps.**
- The learning rate is 0.1
- The gamma (discount rate) is 0.99

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/q-ex-1.jpg" alt="Maze-Example"/>
</figure>
The reward function goes like this:

- **+0:** Going to a state with no cheese in it.
- **+1:** Going to a state with a small cheese in it.
- **+10:** Going to the state with the big pile of cheese.
- **-10:** Going to the state with the poison and thus die.
- **+0** If we spend more than five steps.

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/q-ex-2.jpg" alt="Maze-Example"/>
</figure>
To train our agent to have an optimal policy (so a policy that goes right, right, down), **we will use the Q-Learning algorithm.**

**Step 1: We initialize the Q-Table**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Example-1.jpg" alt="Maze-Example"/>
</figure>

So, for now, **our Q-Table is useless**; we need **to train our Q-function using the Q-Learning algorithm.**

Let's do it for 2 training timesteps:

Training timestep 1:

**Step 2: Choose action using Epsilon Greedy Strategy**

Because epsilon is big = 1.0, I take a random action, in this case, I go right.

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/q-ex-3.jpg" alt="Maze-Example"/>
</figure>

**Step 3: Perform action At, gets Rt+1 and St+1**

By going right, I've got a small cheese, so Rt+1 = 1, and I'm in a new state.


<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/q-ex-4.jpg" alt="Maze-Example"/>
</figure>

**Step 4: Update Q(St, At)**

We can now update Q(St, At) using our formula.

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/q-ex-5.jpg" alt="Maze-Example"/>
</figure>
<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/Example-4.jpg" alt="Maze-Example"/>
</figure>

Training timestep 2:

**Step 2: Choose action using Epsilon Greedy Strategy**

**I take a random action again, since epsilon is big 0.99** (since we decay it a little bit because as the training progress, we want less and less exploration).

I took action down. **Not a good action since it leads me to the poison.**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/q-ex-6.jpg" alt="Maze-Example"/>
</figure>

**Step 3: Perform action At, gets Rt+1 and St+1**

Because I go to the poison state, **I get Rt+1 = -10, and I die.**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/q-ex-7.jpg" alt="Maze-Example"/>
</figure>

**Step 4: Update Q(St, At)**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/q-ex-8.jpg" alt="Maze-Example"/>
  </figure>
  
Because we're dead, we start a new episode. But what we see here is that **with two explorations steps, my agent became smarter.**

As we continue exploring and exploiting the environment and updating Q-values using TD target, **Q-Table will give us better and better approximations. And thus, at the end of the training, we'll get an optimal Q-Function.**

---
Now that we **studied the theory of Q-Learning**, let's **implement it from scratch**. A Q-Learning agent that we will train in two environments:

1. *Frozen-Lake-v1* ❄️ (non-slippery version): where our agent will need to **go from the starting state (S) to the goal state (G)** by walking only on frozen tiles (F) and avoiding holes (H).
2. *An autonomous taxi* 🚕 will need **to learn to navigate** a city to **transport its passengers from point A to point B.**

<figure class="image table text-center m-0 w-full">
  <img src="assets/73_deep_rl_q_part2/envs.gif" alt="Environments"/>
</figure>

Start the tutorial here 👉 https://colab.research.google.com/github/huggingface/deep-rl-class/blob/main/unit2/unit2.ipynb

The leaderboard 👉 https://huggingface.co/spaces/chrisjay/Deep-Reinforcement-Learning-Leaderboard

---
Congrats on finishing this chapter! There was a lot of information. And congrats on finishing the tutorials. You’ve just implemented your first RL agent from scratch and shared it on the Hub 🥳.
  
Implementing from scratch when you study a new architecture **is important to understand how it works.**

That’s **normal if you still feel confused** with all these elements. **This was the same for me and for all people who studied RL.**

Take time to really grasp the material before continuing. It’s essential to master these elements and having a solid foundations before entering the **fun part.**
Don't hesitate to modify the implementation, try ways to improve it and change environments, **the best way to learn is to try things on your own!** 

We published additional readings in the syllabus if you want to go deeper 👉 https://github.com/huggingface/deep-rl-class/blob/main/unit2/README.md

In the next unit, we’re going to learn about Deep-Q-Learning.

And don't forget to share with your friends who want to learn 🤗 !

### Keep learning, stay awesome,
</body>
</html>
  

import yaml
from init import init

# Change Config file location if needed
config = yaml.safe_load(open("./train/cfgs/config.yml"))


agents,env = init(config)

# Create two DQN agents
agent1 = agents[0]
agent2 = agents[1]

#Some hyperparameters for the training loop
batch_size = config["batch_size"]
episodes = config["episodes"]
num_steps = config["num_steps"]

#Aggregating rewards to report during loop
agg_reward1 = 0
agg_reward2 = 0

for e in range(episodes):
    # Reset the environment
    state1, state2 = env.reset()

    for time in range(num_steps):  # Limiting the episode length
        # Agents take actions
        action1 = agent1.act(state1)
        action2 = agent2.act(state2)
        next_state1, next_state2, reward1, reward2, done, _ = env.step(action1, action2)
        
        # Remember the previous state, action, reward, and next state
        agent1.remember(state1, action1, reward1, next_state1, done)
        agent2.remember(state2, action2, reward2, next_state2, done)

        # Update the current state
        state1 = next_state1
        state2 = next_state2

        if done or time == num_steps - 1:
            agg_reward1 += reward1
            agg_reward2 += reward2
            print("episode: {}/{}, score1: {}, score2: {}, e: {:.2}, P1's aggregated rewards: {}, P2's aggregated rewards: {}"
                  .format(e, episodes, reward1, reward2, agent1.epsilon, agg_reward1, agg_reward2))
            break

    # Replay to train the agents
    if len(agent1.memory) > batch_size:
        agent1.replay(batch_size, e)
    if len(agent2.memory) > batch_size:
        agent2.replay(batch_size, e)
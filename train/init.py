from Env.env import createEnv  
from agent.agent import createAgents
def init(cfg):
    env = createEnv(cfg['env'])
    agents = createAgents(cfg['agent'])
    return agents,env
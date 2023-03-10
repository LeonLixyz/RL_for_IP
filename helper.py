import numpy as np

def evolution_strategies(rewards, sigma):
    for r in rewards:
        r = r + sigma * np.random.normal(0, 1)
    return rewards

def discounted_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_sum = 0
    for i in reversed(range(0,len(r))):
        discounted_r[i] = running_sum * gamma + r[i]
        running_sum = discounted_r[i]
    return list(discounted_r)

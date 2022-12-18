from gymenv_v2 import timelimit_wrapper, GurobiOriginalEnv
import numpy as np
import argparse
import os
import torch

cwd = os.getcwd()
PATH = cwd + '/Policy/hard_model'
Policy = torch.load(PATH)


def make_gurobi_env(load_dir, idx, timelimit):
    print('loading training instances, dir {} idx {}'.format(load_dir, idx))
    A = np.load('{}/A_{}.npy'.format(load_dir, idx))
    b = np.load('{}/b_{}.npy'.format(load_dir, idx))
    c = np.load('{}/c_{}.npy'.format(load_dir, idx))
    env = timelimit_wrapper(GurobiOriginalEnv(A, b, c, solution=None, reward_type='obj'), timelimit)
    return env


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--instance-idx', type=int, default=0)
    parser.add_argument('--instance-name', type=str, default='randomip_n60_m60')
    parser.add_argument('--timelimit', type=int, default=100)
    args = parser.parse_args()

    instance_idx = args.instance_idx
    instance_name = args.instance_name
    timelimit = args.timelimit

    # create an environment
    env = make_gurobi_env('instances/{}'.format(instance_name), instance_idx, timelimit)

    # gym loop
    s = env.reset()
    d = False
    t = 00
    repisode = 0
    while not d:
        A, b, c0, cuts_a, cuts_b = s

        # find attention score
        a_b = np.concatenate((A, np.expand_dims(b, -1)), 1)
        d_e = np.concatenate((cuts_a, np.expand_dims(cuts_b, -1)), 1)
        total = np.concatenate((a_b, d_e), 0)

        '''
        three way to preprocess data
        1. normalize
        2. standardize
        3. standardize across column
        '''
        # total = (total - np.mean(total)) / np.std(total)
        total / np.linalg.norm(total)
        # total = (total - np.mean(total, axis=0)) / np.std(total, axis=0)

        constraint = total[:len(a_b)]
        candidate = total[len(a_b):]

        attention_score = Policy.compute_attention(constraint, candidate)
        prob = Policy.compute_prob(attention_score)

        a = np.array([np.argmax(prob)])

        s, r, d, _ = env.step(a)

        print('step', t, 'reward', r, 'action space size', s[-1].size)
        t += 1
        repisode += r

    print('total episode reward: ', repisode)

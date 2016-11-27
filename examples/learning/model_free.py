import numpy as np
from learning.action_selection import EGreedySelection
from learning.action_selection import argmax_random


class Problem:

    def __init__(self, nstates, nactions):
        self.nstates = nstates
        self.nactions = nactions

    def sample_initial_state(self):
        return 0

    def actions(self, s):
        return [0]

    def state_reward(self, s, a):
        return (0, 0)

    def is_final(self, s):
        return True


def qlearning(problem, nepisodes, alpha=0.1, gamma=0.9, epsilon=0.1):
    aselection = EGreedySelection(epsilon)

    nstates, nactions = problem.nstates, problem.nactions

    q = np.zeros((nstates, nactions))
    q.fill(float('-inf'))

    for s in range(nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for _ in range(nepisodes):
        s = problem.sample_initial_state()

        cum_reward = 0
        while not problem.is_final(s):
            a = aselection(q[s], problem.actions(s))

            next_s, r = problem.state_reward(s, a)
            #print(r)
            cum_reward += r
            q[s, a] = q[s, a] + alpha * (r + gamma * max(q[next_s]) - q[s, a])

            s = next_s
        #print(cum_reward)
    pi = q.argmax(axis=1)
    v = q.max(axis=1)
    
    return q,pi, v




def q_lambda(problem, nepisodes, alpha=0.1, gamma=0.9, epsilon=0.1,
             _lambda=0.9):
    # Accumulating traces
    aselection = EGreedySelection(epsilon)

    nstates, nactions = problem.nstates, problem.nactions

    q = np.zeros((nstates, nactions))
    q.fill(float('-inf'))
    et = np.zeros((nstates, nactions))

    for s in range(nstates):
        actions = problem.actions(s)
        for a in actions:
            q[s, a] = 0

    for _ in range(nepisodes):
        s = problem.sample_initial_state()
        a = aselection(q[s], problem.actions(s))

        et.fill(0)
        while not problem.is_final(s):
            next_s, r = problem.state_reward(s, a)

            next_a = aselection(q[next_s], problem.actions(next_s))

            a_star_next = argmax_random(q[next_s])
            if np.allclose(q[next_s, next_a], q[next_s, a_star_next]):
                a_star_next = next_a

            et[s, a] += 1
            q = q + alpha * (r + gamma * q[next_s, a_star_next] - q[s, a]) * et
            if next_a == a_star_next:
                et = et * gamma * _lambda
            else:
                et.fill(0)

            s = next_s
            a = next_a

    pi = q.argmax(axis=1)
    v = q.max(axis=1)

    return pi, v

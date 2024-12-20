import os, sys
import pickle
import random
import numpy as np
from tqdm import tqdm

randn = np.random.randn
rand  = np.random.rand

win_rows = [[0,1,2],[3,4,5],[6,7,8]]
win_cols = [[0,3,6],[1,4,7],[2,5,8]]
win_diags= [[0,4,8],[2,4,6]]
win_comb = win_rows + win_cols + win_diags

def check_state(state):
    assert len(state) == 9,  f"State vector must have 9 elements, found {len(state)}."
    assert all([s in "OX " for s in state]), f"State vector must contain only X,O, or empty spaces."

    status  = ""
    for ch in "XO":
        for inds in win_comb:
            vals = "".join([state[i] for i in inds])
            if vals == "".join([ch]*3):
                status += ch
                break    

    n_empty = sum([s == " " for s in state])
    if status == "" and n_empty == 0:
        status = "T"
    return status

def render(state):
    check_state(state)
    for i in range(3):
        line = " " + "|".join(state[i*3:i*3+3]) + " "
        print(line)
        if i != 2:
            print("".join(["-"]*len(line)))

def possible_actions(state):
    n_X = sum([s == "X" for s in state])
    n_O = sum([s == "O" for s in state])
    assert n_X <= n_O,     "There shouldn't be more Os than Xs"
    assert n_O - n_X <= 1, "There should be at most one more O than X."

    ch = "OX"[n_O - n_X]
    return [(ch, i) for i,s in enumerate(state) if s == " "]

def take_action(state, action):
    ch, pos = action
    state = "".join([ch if i==pos else s for i,s in enumerate(state)])
    return state, check_state(state)

def random_policy(state):
    actions = possible_actions(state)
    return random.choice(actions)

def greedy_policy(state, q, epsilon = 0):
    actions = possible_actions(state)
    q_vals  = [q[state, a] + randn() * 1e-8 for a in actions]
    return actions[np.argmax(q_vals)] if rand() >= epsilon else random.choice(actions)

test_win_states = False
if test_win_states:
    states = []
    for comb in win_comb:
        state = "".join([" " if i not in comb else "O" for i in range(9)])
        states.append(state)
    
    for state in states:
        render(state)
        win = check_state(state)
        print(win)
        assert win == "O", "Win condition undetected."
        state_X = state.replace("O", "X") 
    
        render(state_X)
        win = check_state(state_X)
        print(win)
        assert win == "X", "Win condition undetected."

state = "".join([
    "O", "X", " ",
    "O", "O", "X",
    " ", " ", " ",
    ])

#render(state)
#print(possible_actions(state))
#action = random_policy(state)
#print("Doing ", action)
#state, status = take_action(state, action)
#render(state)
#print("Status ", status)

# Play a game to conclusion.
def play(policy,
         initial_state = "         ",
         verbose = True,
         ):
         
    state = initial_state
    game_over = False
    turn = 1
    states = [state]
    actions = []
    results = []
    while not game_over:
        action = policy(state)
        actions.append(action)
        next_state, status = take_action(state, action)
        results.append(status)
        verbose and print(f"{turn=}")
        verbose and render(next_state)
        if status:
            if status in "XO":
                verbose and print(f"{status} wins.")
            elif status == "T":
                verbose and print(f"Tie game.")
            else:
                raise ValueError(f"Unexpected {status=}.")
            game_over = True
        state = next_state
        states.append(state)
        turn+=1
    return states, actions, results
#random.seed(0)
#states, actions, results = play(random_policy, verbose =False)
#print(actions)
#print(results)
#render(states[-1])
# How to improve the policy?
# Evaluate q values using sarsa...
# Then greedy action selection?
#
# SARSA
# First, the relevant bellman equation
# q(a_t, s_t) = E_r, s' (r + gamma * E_q p(a|s') q(a, s'))

# Then, the sampled version
# q(a_t, s_t) = r_t + gamma * q(a_{t+1}, s_{t+1}).

# SARSA then does this with a smoothing
# q(a_t, s_t)_new = (1 - alpha) q(a_t, s_t)_old + alpha * (r_t + gamma * q(a_{t+1}, s_{t+1}))

class Q:
    def __init__(self, initial_value = 0):
        self.q = {}
        self.initial_value = 0
        pass

    def __getitem__(self, key):
        state, action = key
        if state not in self.q:
            self.q[state]={action:self.initial_value}
        
        if action not in self.q[state]:
            self.q[state][action] = self.initial_value

        return self.q[state][action]

    def __setitem__(self, key, value):
        state, action = key
        if state not in self.q:
            self.q[state] = {action:value}
        else:
            self.q[state][action] = value

def Q_size(q):
    n_states = len(q.q)
    n_actions = sum([len(a) for s,a in q.q.items()])
    return n_states, n_actions

def sarsa(q, alpha, states, actions, results, verbose=False):
    result = results[-1]
    alpha = 0.2
    for imove in range(len(actions)-1,-1,-1):
        state = states[imove]
        action = actions[imove]
        result = results[imove]
        verbose and print("Processing ", state, action, result)
        terminal = len(actions) - imove <= 2
        r = 0
        if terminal:
            if result == action[0]:
                r = 1
            elif result!="T":
                r = -1            
            q[state, action] += alpha * (r - q[state, action])
        else:
            next_state = states[imove+2]
            next_action= actions[imove+2]
            q[state, action] += alpha * (r + q[next_state, next_action] - q[state, action])
            
        verbose and print(f"q[{state}, {action}] = ", f"{q[state,action]:.3f}")

    return q

def show_q(q, state):
    actions = possible_actions(state)
    q_vals = [q[state,a] for a in actions]
    q_norm = np.max(np.abs(q_vals)) + 1e-6
    for i, s in enumerate(state):
        if s == " ":
            ch, pos = actions[0][0], i
            action  = (ch, pos)
            q_val   = q[state, action]/q_norm
            print(f" {ch}:{q_val:+.3f} ", end="")
        else:
            print("    " + s + "    ",end="")        
        if (i % 3 == 2):
            print("")
            i<=6 and print("-"*32)
        else:
            print("|", end="")
        

def learn_q(policy, q = Q(), num_games = 1000, verbose = False, name = "noname", seed = 0, alpha=0.2):
    random.seed(seed)
    print(f"Learning {name} policy.")

    for i in range(num_games):
        states, actions, results = play(policy, verbose = False)
        q = sarsa(q, alpha, states, actions, results, verbose=False)
        if (i % (num_games // 100)) == 0:
            print(".", end = "")
            sys.stdout.flush()

    output_file = f"q_{name}.p"
    with open(output_file, "wb") as f:
        pickle.dump(q, f)

    print(f"Wrote q of size {Q_size(q)} to {output_file}.")
    return q
        
train = False
if train:
    q_random = learn_q(random_policy, num_games = 1000000, name = "random")
else:
    with open("q_random.p", "rb") as f:
        q_random = pickle.load(f)
        print(f"Loaded q_random of size {Q_size(q_random)}.")


greedify = lambda state: greedy_policy(state, q_random, 0.01)
#q_1 = learn_q(greedify, q = q_random, num_games = 1000000, name = "greedy1")
with open("q_greedy1.p", "rb") as f:
    q_1 = pickle.load(f)
q = q_random
#print(results[-1])
#render(states[-3])
#show_q(q, states[-3])

while True:
    ans = input("State? ").replace(",","")
    if ans.startswith("q"):
        print("Bye!")
        break
    elif ans.startswith("r"):
        states, actions, results = play(random_policy, verbose =False)
        state = random.choice(states[:-1])
    else:
        state = ans
    check_state(state)
    render(state)
    show_q(q,  state)


#initial_state = "X O" + "  O" + "XOX"
initial_state = " OO" + "O X" + " X "
# states, actions, results = play(random_policy, verbose = False, initial_state = initial_state)
# for s,a,r in zip(states,actions,results):
#     render(s)
#     print(f"action: {a} -> {r}")
# render(states[-1])

q0 = Q()
for i in range(100000):
    states, actions, results = play(random_policy, verbose = False,
                                    initial_state = initial_state)
    q0 = sarsa(q0, 0.2, states, actions, results, verbose=False)
render(initial_state)
qs = sorted(list(q0.q.keys()), key=lambda s: -sum([ss== " " for ss in s]))
for s in qs:
    show_q(q0, s)
    print("*"*100)

import numpy as np

def create_hmm_model():
    return {
        "states": {},
        "symbols": set(),
        "transitions": {},  # (from, to): prob
        "emissions": {},    # (state, symbol): prob
    }

def add_state(model, state_name, initial_prob):
    model["states"][state_name] = initial_prob

def add_symbol(model, symbol):
    model["symbols"].add(symbol)

def set_transition_probabilities(model, from_state, to_state, probability):
    model["transitions"][(from_state, to_state)] = probability

def set_emission_probabilities(model, state, symbol, probability):
    model["emissions"][(state, symbol)] = probability
    model["symbols"].add(symbol)

def build_matrices(model):
    states = list(model["states"].keys())
    symbols = list(model["symbols"])
    N = len(states)
    M = len(symbols)
    pi = np.array([model["states"][s] for s in states])
    A = np.zeros((N, N))
    B = np.zeros((N, M))
    for i, s_from in enumerate(states):
        for j, s_to in enumerate(states):
            A[i, j] = model["transitions"].get((s_from, s_to), 0.0)
    for i, s in enumerate(states):
        for k, sym in enumerate(symbols):
            B[i, k] = model["emissions"].get((s, sym), 0.0)
    return states, symbols, pi, A, B

def forward_algorithm(obs_seq, states, start_prob, trans_prob, emit_prob, symbols):
    N = len(states)
    T = len(obs_seq)
    alpha = np.zeros((T, N))
    
    for i in range(N):
        alpha[0, i] = start_prob[i] * emit_prob[i, symbols.index(obs_seq[0])]
    
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = sum(alpha[t-1, i] * trans_prob[i, j] for i in range(N)) * emit_prob[j, symbols.index(obs_seq[t])]
    
    prob = np.sum(alpha[T-1, :])
    return prob, alpha

def backward_algorithm(obs_seq, states, start_prob, trans_prob, emit_prob, symbols):
    N = len(states)
    T = len(obs_seq)
    beta = np.zeros((T, N))
    
    beta[T-1, :] = 1
    
    for t in reversed(range(T-1)):
        for i in range(N):
            beta[t, i] = sum(trans_prob[i, j] * emit_prob[j, symbols.index(obs_seq[t+1])] * beta[t+1, j] for j in range(N))
    
    prob = sum(start_prob[i] * emit_prob[i, symbols.index(obs_seq[0])] * beta[0, i] for i in range(N))
    return prob, beta

def parse_hmm_from_gui(states, symbols, pi, A, B):
    states = states
    symbols = symbols
    start_prob = np.array(pi)
    trans_prob = np.array(A)
    emit_prob = np.array(B)
    return states, symbols, start_prob, trans_prob, emit_prob


def perform_inference(model, observations):
    prob_forward, _ = forward_algorithm(observations, model['states'], model['start_prob'], model['trans_prob'], model['emit_prob'], model['symbols'])
    prob_backward, _ = backward_algorithm(observations, model['states'], model['start_prob'], model['trans_prob'], model['emit_prob'], model['symbols'])
    return prob_forward, prob_backward
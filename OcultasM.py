import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def create_hmm_model():
    return {
        "states": {},       # nombre: prob_inicial
        "symbols": set(),   # conjunto de símbolos observables
        "transitions": {},  # (from, to): prob
        "emissions": {},    # (state, symbol): prob
    }

def add_state(model, state_name, initial_prob):
    model["states"][state_name] = float(initial_prob)

def add_symbol(model, symbol):
    model["symbols"].add(symbol)

def set_transition_probabilities(model, from_state, to_state, probability):
    model["transitions"][(from_state, to_state)] = float(probability)

def set_emission_probabilities(model, state, symbol, probability):
    model["emissions"][(state, symbol)] = float(probability)
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
    
    # Inicialización
    try:
        idx_sym0 = symbols.index(obs_seq[0])
    except ValueError:
        return 0.0, alpha # Símbolo no observado
        
    for i in range(N):
        alpha[0, i] = start_prob[i] * emit_prob[i, idx_sym0]
    
    # Recurrencia
    for t in range(1, T):
        try:
            idx_sym = symbols.index(obs_seq[t])
        except ValueError:
            return 0.0, alpha
            
        for j in range(N):
            suma = sum(alpha[t-1, i] * trans_prob[i, j] for i in range(N))
            alpha[t, j] = suma * emit_prob[j, idx_sym]
    
    prob = np.sum(alpha[T-1, :])
    return prob, alpha

def backward_algorithm(obs_seq, states, start_prob, trans_prob, emit_prob, symbols):
    N = len(states)
    T = len(obs_seq)
    beta = np.zeros((T, N))
    
    # Inicialización
    beta[T-1, :] = 1
    
    # Recurrencia
    for t in reversed(range(T-1)):
        try:
            idx_sym = symbols.index(obs_seq[t+1])
        except ValueError:
            return 0.0, beta
            
        for i in range(N):
            suma = sum(trans_prob[i, j] * emit_prob[j, idx_sym] * beta[t+1, j] for j in range(N))
            beta[t, i] = suma
            
    # Terminación
    try:
        idx_sym0 = symbols.index(obs_seq[0])
        prob = sum(start_prob[i] * emit_prob[i, idx_sym0] * beta[0, i] for i in range(N))
    except ValueError:
        prob = 0.0

    return prob, beta

def perform_inference(model, observations_seq):
    states, symbols, pi, A, B = build_matrices(model)
    return forward_algorithm(observations_seq, states, pi, A, B, symbols)

def visualizar_hmm(model):
    """
    Visualiza el HMM: Grafos de transiciones + tabla de emisiones.
    Genera una figura matplotlib.
    """
    states = list(model["states"].keys())
    
    G = nx.DiGraph()
    for s in states:
        G.add_node(s)
    
    for (u, v), prob in model["transitions"].items():
        if prob > 0:
            G.add_edge(u, v, weight=prob)
            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Grafo de transiciones
    pos = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightgreen', node_size=2000)
    nx.draw_networkx_labels(G, pos, ax=ax1)
    
    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=edges, connectionstyle='arc3,rad=0.1')
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1)
    ax1.set_title("Transiciones de Estados Ocultos")
    ax1.axis('off')
    
    # Matriz/Tabla de emisiones
    # Preparar datos para tabla
    symbols = list(model["symbols"])
    cell_text = []
    row_labels = states
    col_labels = symbols
    
    for s in states:
        row = []
        for sym in symbols:
            prob = model["emissions"].get((s, sym), 0.0)
            row.append(f"{prob:.2f}")
        cell_text.append(row)
        
    ax2.axis('tight')
    ax2.axis('off')
    table = ax2.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    ax2.set_title("Probabilidades de Emisión")
    
    plt.tight_layout()
    return fig

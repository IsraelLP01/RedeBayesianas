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

def forward_backward(observaciones, pi, A, B):
    """Implementación del algoritmo Forward-Backward"""
    T = len(observaciones)
    N = len(pi)

    # Algoritmo Forward
    alpha = np.zeros((T, N))
    c = np.zeros(T)  # Factores de escala

    # Inicialización
    try:
        alpha[0] = pi * B[:, observaciones[0]]
        c[0] = 1.0 / np.sum(alpha[0])
        alpha[0] *= c[0]
    except ZeroDivisionError:
        # Manejo básico si la probabilidad inicial de la primera observación es 0
        return np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N)), np.zeros((T-1, N, N)), np.zeros(T)

    # Recursión
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, observaciones[t]] * np.sum(alpha[t-1] * A[:, j])
        
        suma = np.sum(alpha[t])
        if suma == 0: 
            c[t] = 1.0 # Evitar div por cero si la prob se vuelve nula
        else:
            c[t] = 1.0 / suma
            
        alpha[t] *= c[t]

    # Algoritmo Backward
    beta = np.zeros((T, N))

    # Inicialización
    beta[T-1] = 1.0
    beta[T-1] *= c[T-1]

    # Recursión hacia atrás
    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[t, i] = np.sum(A[i, :] * B[:, observaciones[t+1]] * beta[t+1, :])
        beta[t] *= c[t]

    # Probabilidades suavizadas
    gamma = np.zeros((T, N))
    for t in range(T):
        gamma[t] = alpha[t] * beta[t]
        denom = np.sum(gamma[t])
        if denom != 0:
            gamma[t] /= denom

    # Probabilidades de transición
    xi = np.zeros((T-1, N, N))
    for t in range(T-1):
        # Calcular denominador común para xi[t]
        # P(O | model) approx 1/prod(c) ? No, xi se normaliza localmente o por P(O)
        # Standard formula: xi_t(i,j) = (alpha_t(i) * a_ij * b_j(O_t+1) * beta_t+1(j)) / P(O|lambda)
        # Con scaling, alpha y beta están escalados.
        # xi_t(i,j) = alpha_scaled_t(i) * a_ij * b_j(O_t+1) * beta_scaled_t+1(j) * (1/c_t+1? no)
        # La formula directa con scaling suele ser simplemente la computada y normalizada
       
        # Implementación del usuario (revisada):
        denom = np.sum(alpha[t] * np.sum(A * B[:, observaciones[t+1]] * beta[t+1], axis=1))
        
        for i in range(N):
            for j in range(N):
                xi[t, i, j] = alpha[t, i] * A[i, j] * B[j, observaciones[t+1]] * beta[t+1, j]
                if denom != 0:
                    xi[t, i, j] /= denom

    return alpha, beta, gamma, xi, c

def perform_inference(model, observations_seq):
    states, symbols, pi, A, B = build_matrices(model)
    
    # 1. Convertir observaciones (strings) a índices
    try:
        obs_indices = [symbols.index(o) for o in observations_seq]
    except ValueError as e:
        print(f"Error: Símbolo no encontrado en el modelo. {e}")
        return 0.0, np.zeros((len(observations_seq), len(states)))

    # 2. Ejecutar Forward-Backward optimizado
    alpha, beta, gamma, xi, c = forward_backward(obs_indices, pi, A, B)
    
    # 3. Calcular Probabilidad Total P(O|lambda)
    # Con scaling: log P(O|lambda) = - sum(log(c_t))
    # Prob = exp(log P)
    # Nota: Si la secuencia es muy larga, Prob puede ser 0 (underflow) incluso si log P es válido.
    # Para la GUI devolvemos el valor real float, sabiendo que puede ser muy pequeño.
    
    log_prob = -np.sum(np.log(c + 1e-300)) # epsilon seguridad
    prob = np.exp(log_prob)
    
    return prob, alpha, gamma, xi

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

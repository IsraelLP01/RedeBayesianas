import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class CadenaMarkov:
    def __init__(self):
        """
        Clase que representa una cadena de Markov.
        Soporta cadenas regulares y absorbentes.
        """
        self.estados = []
        self.transiciones = {}  # Diccionario para almacenar probabilidades de transición
        self.matriz_P = None    # Matriz numpy
        self.es_absorbente = False
        self.estados_transitorios = []
        self.estados_absorbentes = []
        self.submatriz_Q = None
        self.submatriz_R = None
        self.matriz_N = None    # Matriz Fundamental
        self.tiempo_absorcion = None
        self.prob_absorcion = None

    def agregar_estado(self, estado):
        """
        Agrega un estado a la cadena de Markov.
        """
        if estado not in self.estados:
            self.estados.append(estado)
            self.transiciones[estado] = {}
            self._invalidar_cache()

    def establecer_transicion(self, estado_origen, estado_destino, probabilidad):
        """
        Establece la probabilidad de transición entre dos estados.
        """
        if estado_origen in self.estados and estado_destino in self.estados:
            self.transiciones[estado_origen][estado_destino] = float(probabilidad)
            self._invalidar_cache()

    def _invalidar_cache(self):
        self.matriz_P = None
        self.es_absorbente = False
        self.estados_transitorios = []
        self.estados_absorbentes = []
        self.submatriz_Q = None
        self.submatriz_R = None
        self.matriz_N = None
        self.tiempo_absorcion = None
        self.prob_absorcion = None

    def _construir_matriz(self):
        n = len(self.estados)
        P = np.zeros((n, n))
        for i, origen in enumerate(self.estados):
            for j, destino in enumerate(self.estados):
                P[i, j] = self.transiciones[origen].get(destino, 0.0)
        self.matriz_P = P
        return P

    def validar_transiciones(self):
        """
        Valida que las probabilidades de transición de cada estado sumen 1.
        """
        for estado in self.estados:
            suma = sum(self.transiciones[estado].values())
            # Permitir un pequeño error de flotante
            if abs(suma - 1.0) > 0.01:
                # Si es 0 y no tiene transiciones definidas, podria ser manejable, pero
                # para una CM valida debe sumar 1.
                if suma == 0:
                     return False, f"El estado '{estado}' no tiene transiciones salientes (suma 0)."
                return False, f"Las probabilidades desde '{estado}' suman {suma:.2f}, deben sumar 1.0"
        return True, "Validación correcta"

    def analizar_estructura(self):
        """
        Analiza si la cadena es absorbente y calcula sus matrices fundamentales si lo es.
        """
        if self.matriz_P is None:
            self._construir_matriz()
            
        n = len(self.estados)
        P = self.matriz_P
        
        # Identificar estados absorbentes (P[i,i] == 1)
        self.estados_absorbentes = []
        self.estados_transitorios = []
        
        for i in range(n):
            if abs(P[i, i] - 1.0) < 0.000001:
                self.estados_absorbentes.append(self.estados[i])
            else:
                self.estados_transitorios.append(self.estados[i])
                
        # Una CM es absorbente si tiene al menos un estado absorbente y 
        # desde cualquier estado transitorio es posible llegar a un estado absorbente.
        # (La segunda condición es más compleja de verificar rigurosamente sin BFS/DFS, 
        #  pero asumiremos la estructura básica si hay estados absorbentes).
        
        if len(self.estados_absorbentes) > 0 and len(self.estados_transitorios) > 0:
            self.es_absorbente = True
            
            # Reordenar P para forma canónica:
            # | Q  R |
            # | 0  I |
            # Indices: primero transitorios, luego absorbentes
            indices_t = [self.estados.index(e) for e in self.estados_transitorios]
            indices_a = [self.estados.index(e) for e in self.estados_absorbentes]
            
            t = len(indices_t)
            a = len(indices_a)
            
            self.submatriz_Q = P[np.ix_(indices_t, indices_t)]
            self.submatriz_R = P[np.ix_(indices_t, indices_a)]
            
            # Calcular Matriz Fundamental N = (I - Q)^-1
            try:
                I = np.eye(t)
                self.matriz_N = np.linalg.inv(I - self.submatriz_Q)
                
                # Tiempo esperado de absorción t = N * 1
                self.tiempo_absorcion = self.matriz_N @ np.ones(t)
                
                # Probabilidad de absorción B = N * R
                self.prob_absorcion = self.matriz_N @ self.submatriz_R
                
            except np.linalg.LinAlgError:
                print("Error: No se pudo invertir (I - Q). La cadena podría no ser verdaderamente absorbente.")
                self.es_absorbente = False
        else:
            self.es_absorbente = False


    def simular(self, estado_inicial, pasos):
        """
        Simula una secuencia de estados. Detiene si llega a estado absorbente.
        """
        if estado_inicial not in self.estados:
            raise ValueError(f"Estado inicial '{estado_inicial}' no existe.")
        
        estado_actual = estado_inicial
        secuencia = [estado_actual]

        # Pre-check si es absorbente para detenerse
        es_abs = {e: False for e in self.estados}
        for e in self.estados:
            if self.transiciones[e].get(e, 0.0) == 1.0:
                es_abs[e] = True

        for _ in range(pasos):
            # Si estamos en estado absorbente, terminar
            if es_abs.get(estado_actual, False):
                break
                
            if estado_actual not in self.transiciones:
                break
            
            destinos = list(self.transiciones[estado_actual].keys())
            probs = list(self.transiciones[estado_actual].values())
            
            if not destinos:
                break
                
            estado_actual = random.choices(destinos, weights=probs)[0]
            secuencia.append(estado_actual)

        return secuencia

    def calcular_estacionaria(self):
        """
        Calcula la distribución estacionaria (útil para cadenas ergódicas/regulares).
        """
        if self.matriz_P is None:
            self._construir_matriz()
            
        n = len(self.estados)
        P = self.matriz_P
        
        # (P^T - I) * pi = 0
        # Agregar restricción suma(pi) = 1
        A = P.T - np.eye(n)
        A[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1
        
        try:
            pi = np.linalg.solve(A, b)
            return dict(zip(self.estados, pi))
        except np.linalg.LinAlgError:
            return None 

    def visualizar(self):
        """
        Genera un objeto figura de matplotlib con el grafo Y el heatmap.
        """
        if self.matriz_P is None:
            self._construir_matriz()
            
        P = self.matriz_P
        estados = self.estados
        n = len(estados)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- GRAFO ---
        G = nx.DiGraph()
        for i, origen in enumerate(estados):
            G.add_node(origen)
            for j, destino in enumerate(estados):
                prob = P[i, j]
                if prob > 0.01:
                    G.add_edge(origen, destino, weight=prob)

        # Posiciones
        try:
            pos = nx.spring_layout(G, seed=42)
        except:
            pos = nx.circular_layout(G)
        
        # Nodos
        # Colorear absorbentes diferente
        node_colors = []
        for e in estados:
            # Check simple de absorbencia
            if self.transiciones[e].get(e, 0.0) == 1.0:
                node_colors.append('lightcoral')
            else:
                node_colors.append('lightblue')
                
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, node_size=2500, edgecolors='black')
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=9, font_weight='bold')
        
        # Aristas
        edges = G.edges(data=True)
        if edges:
            widths = [d['weight'] * 3 for (u, v, d) in edges]
            nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=edges, width=widths, 
                                 edge_color='gray', connectionstyle='arc3,rad=0.1', arrowsize=15)
            # Etiquetas
            edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1, font_size=8)
            
        ax1.set_title("Diagrama de Transiciones", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # --- HEATMAP ---
        im = ax2.imshow(P, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        
        # Anotaciones
        for i in range(n):
            for j in range(n):
                color = 'white' if P[i, j] > 0.5 else 'black'
                val = P[i, j]
                if val > 0:
                    ax2.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)
        
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        
        # Labels cortos si son muy largos
        labels_cortos = [e[:10]+".." if len(e)>10 else e for e in estados]
        ax2.set_xticklabels(labels_cortos, rotation=45, ha='right')
        ax2.set_yticklabels(labels_cortos)
        
        ax2.set_title("Matriz de Transición P", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig

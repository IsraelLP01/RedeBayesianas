import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class CadenaMarkov:
    def __init__(self):
        """
        Clase que representa una cadena de Markov.
        """
        self.estados = []
        self.transiciones = {}  # Diccionario para almacenar probabilidades de transición
        self.matriz_P = None    # Matriz numpy

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
            if abs(suma - 1.0) > 0.01:
                return False, f"Las probabilidades desde '{estado}' suman {suma:.2f}, deben sumar 1.0"
        return True, "Validación correcta"

    def simular(self, estado_inicial, pasos):
        """
        Simula una secuencia de estados.
        """
        if estado_inicial not in self.estados:
            raise ValueError(f"Estado inicial '{estado_inicial}' no existe.")
        
        estado_actual = estado_inicial
        secuencia = [estado_actual]

        for _ in range(pasos):
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
        Calcula la distribución estacionaria.
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
            return None # Puede pasar si hay múltiples distribuciones estacionarias o singularidad

    def calcular_matriz_transicion_n(self, n):
        """
        Calcula la matriz de transición a n pasos.
        """
        if self.matriz_P is None:
            self._construir_matriz()
        
        Pn = np.linalg.matrix_power(self.matriz_P, n)
        resultado = {}
        for i, origen in enumerate(self.estados):
            resultado[origen] = {}
            for j, destino in enumerate(self.estados):
                resultado[origen][destino] = Pn[i, j]
        return resultado

    def visualizar(self):
        """
        Genera un objeto figura de matplotlib con el grafo.
        """
        G = nx.DiGraph()
        for origen in self.estados:
            G.add_node(origen)
            for destino, prob in self.transiciones[origen].items():
                if prob > 0:
                    G.add_edge(origen, destino, weight=prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        
        # Dibujar nodos
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=2000)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        # Dibujar aristas
        edges = G.edges(data=True)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, arrowsize=20, connectionstyle='arc3,rad=0.1')
        
        # Etiquetas de aristas
        labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
        
        ax.axis('off')
        return fig

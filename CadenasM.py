import random

class CadenaMarkov:
    def __init__(self):
        """
        Clase que representa una cadena de Markov.
        """
        self.estados = []
        self.transiciones = {}  # Diccionario para almacenar probabilidades de transición

    def agregar_estado(self, estado):
        """
        Agrega un estado a la cadena de Markov.
        :param estado: Nombre del estado.
        """
        if estado not in self.estados:
            self.estados.append(estado)
            self.transiciones[estado] = {}

    def establecer_transicion(self, estado_origen, estado_destino, probabilidad):
        """
        Establece la probabilidad de transición entre dos estados.
        :param estado_origen: Estado de origen.
        :param estado_destino: Estado de destino.
        :param probabilidad: Probabilidad de transición (0 <= probabilidad <= 1).
        """
        if estado_origen in self.estados and estado_destino in self.estados:
            self.transiciones[estado_origen][estado_destino] = probabilidad

    def validar_transiciones(self):
        """
        Valida que las probabilidades de transición de cada estado sumen 1.
        """
        for estado, transiciones in self.transiciones.items():
            suma = sum(transiciones.values())
            if abs(suma - 1.0) > 0.001:
                raise ValueError(f"Las probabilidades de transición desde el estado '{estado}' no suman 1.")

    def simular(self, estado_inicial, pasos):
        """
        Simula una secuencia de estados en la cadena de Markov.
        :param estado_inicial: Estado inicial de la simulación.
        :param pasos: Número de pasos a simular.
        :return: Lista con la secuencia de estados.
        """
        if estado_inicial not in self.estados:
            raise ValueError(f"El estado inicial '{estado_inicial}' no existe en la cadena.")
        
        estado_actual = estado_inicial
        secuencia = [estado_actual]

        for _ in range(pasos):
            if estado_actual not in self.transiciones or not self.transiciones[estado_actual]:
                raise ValueError(f"No hay transiciones definidas para el estado '{estado_actual}'.")
            
            estados_destino = list(self.transiciones[estado_actual].keys())
            probabilidades = list(self.transiciones[estado_actual].values())
            estado_actual = random.choices(estados_destino, probabilidades)[0]
            secuencia.append(estado_actual)

        return secuencia
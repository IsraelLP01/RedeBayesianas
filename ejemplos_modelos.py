import RedesBa
import CadenasM
import OcultasM
import numpy as np

def ejemplo_alarma():
    print("\n" + "="*60)
    print(" EJEMPLO 1: RED ALARMA-TERREMOTO-LADRÓN")
    print("="*60)
    red = RedesBa.crear_red_bayesiana()
    
    # 1. Variables
    # B: Burglary (Robo), E: Earthquake (Terremoto), A: Alarm (Alarma)
    # J: JohnCalls (Juan Llama), M: MaryCalls (María Llama)
    RedesBa.agregar_variable(red, "Robo", ["Si", "No"])
    RedesBa.agregar_variable(red, "Terremoto", ["Si", "No"])
    RedesBa.agregar_variable(red, "Alarma", ["Si", "No"])
    RedesBa.agregar_variable(red, "JuanLlama", ["Si", "No"])
    RedesBa.agregar_variable(red, "MariaLlama", ["Si", "No"])
    
    # 2. Estructura (Padres)
    RedesBa.establecer_padres(red, "Alarma", ["Robo", "Terremoto"])
    RedesBa.establecer_padres(red, "JuanLlama", ["Alarma"])
    RedesBa.establecer_padres(red, "MariaLlama", ["Alarma"])
    
    # 3. Probabilidades (Tablas)
    # P(Robo)
    RedesBa.establecer_probabilidad(red, "Robo", (), {"Si": 0.001, "No": 0.999})
    # P(Terremoto)
    RedesBa.establecer_probabilidad(red, "Terremoto", (), {"Si": 0.002, "No": 0.998})
    
    # P(Alarma | Robo, Terremoto)
    # R=Si, T=Si
    RedesBa.establecer_probabilidad(red, "Alarma", ("Si", "Si"), {"Si": 0.95, "No": 0.05})
    # R=Si, T=No
    RedesBa.establecer_probabilidad(red, "Alarma", ("Si", "No"), {"Si": 0.94, "No": 0.06})
    # R=No, T=Si
    RedesBa.establecer_probabilidad(red, "Alarma", ("No", "Si"), {"Si": 0.29, "No": 0.71})
    # R=No, T=No
    RedesBa.establecer_probabilidad(red, "Alarma", ("No", "No"), {"Si": 0.001, "No": 0.999})
    
    # P(JuanLlama | Alarma)
    RedesBa.establecer_probabilidad(red, "JuanLlama", ("Si",), {"Si": 0.90, "No": 0.10})
    RedesBa.establecer_probabilidad(red, "JuanLlama", ("No",), {"Si": 0.05, "No": 0.95})
    
    # P(MariaLlama | Alarma)
    RedesBa.establecer_probabilidad(red, "MariaLlama", ("Si",), {"Si": 0.70, "No": 0.30})
    RedesBa.establecer_probabilidad(red, "MariaLlama", ("No",), {"Si": 0.01, "No": 0.99})
    
    print("Red construida correctamente.")
    
    # 4. Inferencia
    # ¿Probabilidad de Robo dado que Juan y María llaman?
    evidencia = {"JuanLlama": "Si", "MariaLlama": "Si"}
    print(f"\nConsulta: P(Robo | Juan=Si, Maria=Si)")
    res = RedesBa.inferencia_enumeracion("Robo", evidencia, red)
    print(f"Resultado: {res}") 
    # Esperado Approx: P(R=Si|J,M) ~ 0.28
    
    print("\nVisualización de la estructura disponible en consola.")

def ejemplo_medico():
    print("\n" + "="*60)
    print(" EJEMPLO 2: RED MÉDICA (Síntomas-Enfermedades)")
    print("="*60)
    red = RedesBa.crear_red_bayesiana()
    
    # Gripe (G), Abscesso (A), Fiebre (F), Cansancio (C)
    RedesBa.agregar_variable(red, "Gripe", ["Si", "No"])
    RedesBa.agregar_variable(red, "Absceso", ["Si", "No"])
    RedesBa.agregar_variable(red, "Fiebre", ["Alta", "Baja", "Nula"]) # 3 valores
    RedesBa.agregar_variable(red, "Cansancio", ["Si", "No"])
    
    # Padres
    # Fiebre depende de Gripe y Absceso
    # Cansancio depende de Gripe
    RedesBa.establecer_padres(red, "Fiebre", ["Gripe", "Absceso"])
    RedesBa.establecer_padres(red, "Cansancio", ["Gripe"])
    
    # Probabilidades
    RedesBa.establecer_probabilidad(red, "Gripe", (), {"Si": 0.05, "No": 0.95})
    RedesBa.establecer_probabilidad(red, "Absceso", (), {"Si": 0.02, "No": 0.98})
    
    # P(Fiebre | G, A) - 3 estados
    # G=Si, A=Si
    RedesBa.establecer_probabilidad(red, "Fiebre", ("Si", "Si"), {"Alta": 0.90, "Baja": 0.09, "Nula": 0.01})
    # G=Si, A=No
    RedesBa.establecer_probabilidad(red, "Fiebre", ("Si", "No"), {"Alta": 0.60, "Baja": 0.30, "Nula": 0.10})
    # G=No, A=Si
    RedesBa.establecer_probabilidad(red, "Fiebre", ("No", "Si"), {"Alta": 0.70, "Baja": 0.20, "Nula": 0.10})
    # G=No, A=No
    RedesBa.establecer_probabilidad(red, "Fiebre", ("No", "No"), {"Alta": 0.01, "Baja": 0.09, "Nula": 0.90})
    
    # P(Cansancio | Gripe)
    RedesBa.establecer_probabilidad(red, "Cansancio", ("Si",), {"Si": 0.80, "No": 0.20})
    RedesBa.establecer_probabilidad(red, "Cansancio", ("No",), {"Si": 0.10, "No": 0.90})

    evidencia = {"Fiebre": "Alta", "Cansancio": "Si"}
    print(f"\nConsulta: P(Gripe | Fiebre=Alta, Cansancio=Si)")
    res = RedesBa.inferencia_enumeracion("Gripe", evidencia, red)
    print(f"Resultado: {res}")


def ejemplo_fallas():
    print("\n" + "="*60)
    print(" EJEMPLO 3: DIAGNÓSTICO DE FALLAS (Motor)")
    print("="*60)
    red = RedesBa.crear_red_bayesiana()
    
    # Variables: Bateria (B), Combustible (C), Ignicion (I), Arranca (A), MedidorGas (M)
    RedesBa.agregar_variable(red, "BateriaOK", ["Si", "No"])
    RedesBa.agregar_variable(red, "CombustibleOK", ["Si", "No"])
    RedesBa.agregar_variable(red, "IgnicionOK", ["Si", "No"])
    RedesBa.agregar_variable(red, "Arranca", ["Si", "No"])
    RedesBa.agregar_variable(red, "MedidorLleno", ["Si", "No"])
    
    # Estructura
    # Ignicion depende de Bateria
    # Arranca depende de Ignicion y Combustible
    # Medidor depende de Combustible
    RedesBa.establecer_padres(red, "IgnicionOK", ["BateriaOK"])
    RedesBa.establecer_padres(red, "Arranca", ["IgnicionOK", "CombustibleOK"])
    RedesBa.establecer_padres(red, "MedidorLleno", ["CombustibleOK"])
    
    # Probabilidades
    RedesBa.establecer_probabilidad(red, "BateriaOK", (), {"Si": 0.9, "No": 0.1})
    RedesBa.establecer_probabilidad(red, "CombustibleOK", (), {"Si": 0.9, "No": 0.1})
    
    # P(I | B)
    RedesBa.establecer_probabilidad(red, "IgnicionOK", ("Si",), {"Si": 0.95, "No": 0.05})
    RedesBa.establecer_probabilidad(red, "IgnicionOK", ("No",), {"Si": 0.00, "No": 1.00}) # Sin pila no hay ignicion
    
    # P(A | I, C)
    RedesBa.establecer_probabilidad(red, "Arranca", ("Si", "Si"), {"Si": 0.99, "No": 0.01})
    RedesBa.establecer_probabilidad(red, "Arranca", ("Si", "No"), {"Si": 0.0, "No": 1.0})
    RedesBa.establecer_probabilidad(red, "Arranca", ("No", "Si"), {"Si": 0.0, "No": 1.0})
    RedesBa.establecer_probabilidad(red, "Arranca", ("No", "No"), {"Si": 0.0, "No": 1.0})
    
    # P(M | C)
    RedesBa.establecer_probabilidad(red, "MedidorLleno", ("Si",), {"Si": 0.98, "No": 0.02})
    RedesBa.establecer_probabilidad(red, "MedidorLleno", ("No",), {"Si": 0.05, "No": 0.95}) # Medidor roto?

    # Diagnóstico: El coche no arranca y el medidor marca vacío. ¿Probabilidad de que no haya combustible?
    evidencia = {"Arranca": "No", "MedidorLleno": "No"}
    print(f"\nConsulta: P(CombustibleOK | Arranca=No, Medidor=No)")
    res = RedesBa.inferencia_enumeracion("CombustibleOK", evidencia, red)
    print(f"Resultado: {res}")


def ejemplo_clima():
    print("\n" + "="*60)
    print(" EJEMPLO 4: PREDICCIÓN CLIMÁTICA")
    print("="*60)
    red = RedesBa.crear_red_bayesiana()
    
    # Nublado (N), Lluvia (L), Rociador (R), HierbaMojada (H)
    # Clasico ejemplo "Sprinkler"
    RedesBa.agregar_variable(red, "Nublado", ["Si", "No"])
    RedesBa.agregar_variable(red, "Rociador", ["Si", "No"])
    RedesBa.agregar_variable(red, "Lluvia", ["Si", "No"])
    RedesBa.agregar_variable(red, "HierbaMojada", ["Si", "No"])
    
    # Estructura
    # R depende de N (si está nublado no prendo rociador)
    # L depende de N
    # H depende de R y L
    RedesBa.establecer_padres(red, "Rociador", ["Nublado"])
    RedesBa.establecer_padres(red, "Lluvia", ["Nublado"])
    RedesBa.establecer_padres(red, "HierbaMojada", ["Rociador", "Lluvia"])
    
    # Probs
    RedesBa.establecer_probabilidad(red, "Nublado", (), {"Si": 0.5, "No": 0.5})
    
    RedesBa.establecer_probabilidad(red, "Rociador", ("Si",), {"Si": 0.1, "No": 0.9})
    RedesBa.establecer_probabilidad(red, "Rociador", ("No",), {"Si": 0.5, "No": 0.5})
    
    RedesBa.establecer_probabilidad(red, "Lluvia", ("Si",), {"Si": 0.8, "No": 0.2})
    RedesBa.establecer_probabilidad(red, "Lluvia", ("No",), {"Si": 0.2, "No": 0.8})
    
    # Sprinkler=S, Rain=S -> Wet=S high
    RedesBa.establecer_probabilidad(red, "HierbaMojada", ("Si", "Si"), {"Si": 0.99, "No": 0.01})
    RedesBa.establecer_probabilidad(red, "HierbaMojada", ("Si", "No"), {"Si": 0.90, "No": 0.10})
    RedesBa.establecer_probabilidad(red, "HierbaMojada", ("No", "Si"), {"Si": 0.90, "No": 0.10})
    RedesBa.establecer_probabilidad(red, "HierbaMojada", ("No", "No"), {"Si": 0.00, "No": 1.00})
    
    # ¿Llovió si la hierba está mojada?
    evidencia = {"HierbaMojada": "Si"}
    print(f"\nConsulta: P(Lluvia | HierbaMojada=Si)")
    res = RedesBa.inferencia_enumeracion("Lluvia", evidencia, red)
    print(f"Resultado: {res}")


def crear_ejemplo_cm():
    cm = CadenasM.CadenaMarkov()
    cm.agregar_estado("Soleado")
    cm.agregar_estado("Nublado")
    cm.agregar_estado("Lluvioso")
    
    cm.establecer_transicion("Soleado", "Soleado", 0.7)
    cm.establecer_transicion("Soleado", "Nublado", 0.2)
    cm.establecer_transicion("Soleado", "Lluvioso", 0.1)
    
    cm.establecer_transicion("Nublado", "Soleado", 0.3)
    cm.establecer_transicion("Nublado", "Nublado", 0.4)
    cm.establecer_transicion("Nublado", "Lluvioso", 0.3)
    
    cm.establecer_transicion("Lluvioso", "Soleado", 0.2)
    cm.establecer_transicion("Lluvioso", "Nublado", 0.4)
    cm.establecer_transicion("Lluvioso", "Lluvioso", 0.4)
    return cm

def ejemplo_cm():
    print("\n" + "="*60)
    print(" EXTRA: CADENA DE MARKOV (Clima Simple)")
    print("="*60)
    cm = crear_ejemplo_cm()
    
    print("Calculando estacionaria...")
    est = cm.calcular_estacionaria()
    print(f"Distribución a largo plazo: {est}")


def crear_ejemplo_hmm():
    hmm = OcultasM.create_hmm_model()
    OcultasM.add_state(hmm, "Sano", 0.6)
    OcultasM.add_state(hmm, "Febriel", 0.4)
    
    # Transiciones (tiende a quedarse en el mismo estado)
    OcultasM.set_transition_probabilities(hmm, "Sano", "Sano", 0.7)
    OcultasM.set_transition_probabilities(hmm, "Sano", "Febriel", 0.3)
    OcultasM.set_transition_probabilities(hmm, "Febriel", "Sano", 0.4)
    OcultasM.set_transition_probabilities(hmm, "Febriel", "Febriel", 0.6)
    
    # Emisiones
    # Sano -> Normal(0.5), Frio(0.4), Mareado(0.1)
    OcultasM.set_emission_probabilities(hmm, "Sano", "Normal", 0.5)
    OcultasM.set_emission_probabilities(hmm, "Sano", "Frio", 0.4)
    OcultasM.set_emission_probabilities(hmm, "Sano", "Mareado", 0.1)
    
    # Febriel -> Normal(0.1), Frio(0.3), Mareado(0.6)
    OcultasM.set_emission_probabilities(hmm, "Febriel", "Normal", 0.1)
    OcultasM.set_emission_probabilities(hmm, "Febriel", "Frio", 0.3)
    OcultasM.set_emission_probabilities(hmm, "Febriel", "Mareado", 0.6)
    
    # Secuencia recomendada para probar
    obs = ["Normal", "Mareado", "Mareado"]
    return hmm, obs
    

def ejemplo_hmm():
    print("\n" + "="*60)
    print(" EXTRA: HMM (Detección de Fiebre)")
    print("="*60)
    
    hmm, obs = crear_ejemplo_hmm()
    
    print(f"Secuencia observada: {obs}")
    prob, _, _, _ = OcultasM.perform_inference(hmm, obs)
    print(f"Probabilidad de la secuencia: {prob:.4e}")


if __name__ == "__main__":
    ejemplo_alarma()
    ejemplo_medico()
    ejemplo_fallas()
    ejemplo_clima()
    ejemplo_cm()
    ejemplo_hmm()

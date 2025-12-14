import unittest
import RedesBa
import CadenasM
import OcultasM
import numpy as np

class TestModelos(unittest.TestCase):
    
    def test_red_bayesiana(self):
        print("\n--- Test Red Bayesiana ---")
        red = RedesBa.crear_red_bayesiana()
        RedesBa.agregar_variable(red, "Lluvia", ["T", "F"])
        RedesBa.agregar_variable(red, "PastoMojado", ["T", "F"])
        RedesBa.establecer_padres(red, "PastoMojado", ["Lluvia"])
        
        # P(Lluvia=T) = 0.2
        RedesBa.establecer_probabilidad(red, "Lluvia", [], {"T": 0.2, "F": 0.8})
        
        # P(PastoMojado=T | Lluvia=T) = 0.99
        # P(PastoMojado=T | Lluvia=F) = 0.2
        pasto_dado_lluvia = {
            ("T",): {"T": 0.99, "F": 0.01},
            ("F",): {"T": 0.20, "F": 0.80}
        }
        for padres, probs in pasto_dado_lluvia.items():
            RedesBa.establecer_probabilidad(red, "PastoMojado", padres, probs)
            
        # Inferencia: P(Lluvia | PastoMojado=T)
        # Enumeración
        res_enum = RedesBa.inferencia_enumeracion("Lluvia", {"PastoMojado": "T"}, red)
        print("Enumeración P(Lluvia | PastoMojado=T):", res_enum)
        
        # Eliminación
        res_elim = RedesBa.inferencia_eliminacion_variables("Lluvia", {"PastoMojado": "T"}, red)
        print("Eliminación P(Lluvia | PastoMojado=T):", res_elim)
        
        self.assertAlmostEqual(res_enum["T"], res_elim["T"], places=4)
        
        # Valor teórico: P(R|W) = P(W|R)P(R) / P(W)
        # P(W) = 0.99*0.2 + 0.2*0.8 = 0.198 + 0.16 = 0.358
        # P(R|W) = 0.99 * 0.2 / 0.358 = 0.198 / 0.358 ≈ 0.553
        self.assertAlmostEqual(res_enum["T"], 0.55307, places=3)

    def test_cadena_markov(self):
        print("\n--- Test Cadenas Markov ---")
        cm = CadenasM.CadenaMarkov()
        cm.agregar_estado("A")
        cm.agregar_estado("B")
        cm.establecer_transicion("A", "A", 0.9)
        cm.establecer_transicion("A", "B", 0.1)
        cm.establecer_transicion("B", "A", 0.5)
        cm.establecer_transicion("B", "B", 0.5)
        
        est = cm.calcular_estacionaria()
        print("Estacionaria:", est)
        
        # Teórico:
        # pi_A = 0.9 pi_A + 0.5 pi_B
        # pi_B = 0.1 pi_A + 0.5 pi_B  -> 0.5 pi_B = 0.1 pi_A -> pi_B = 0.2 pi_A
        # pi_A + 0.2 pi_A = 1 -> 1.2 pi_A = 1 -> pi_A = 1/1.2 = 0.8333
        self.assertAlmostEqual(est["A"], 0.8333, places=3)

    def test_hmm(self):
        print("\n--- Test HMM ---")
        model = OcultasM.create_hmm_model()
        OcultasM.add_state(model, "Rainy", 0.6)
        OcultasM.add_state(model, "Sunny", 0.4)
        
        # Transiciones
        # Rain -> Rain 0.7, Rain -> Sun 0.3
        OcultasM.set_transition_probabilities(model, "Rainy", "Rainy", 0.7)
        OcultasM.set_transition_probabilities(model, "Rainy", "Sunny", 0.3)
        OcultasM.set_transition_probabilities(model, "Sunny", "Rainy", 0.4)
        OcultasM.set_transition_probabilities(model, "Sunny", "Sunny", 0.6)
        
        # Emisiones
        # Rain -> Walk 0.1, Shop 0.4, Clean 0.5
        OcultasM.set_emission_probabilities(model, "Rainy", "Walk", 0.1)
        OcultasM.set_emission_probabilities(model, "Rainy", "Shop", 0.4)
        OcultasM.set_emission_probabilities(model, "Rainy", "Clean", 0.5)
        
        OcultasM.set_emission_probabilities(model, "Sunny", "Walk", 0.6)
        OcultasM.set_emission_probabilities(model, "Sunny", "Shop", 0.3)
        OcultasM.set_emission_probabilities(model, "Sunny", "Clean", 0.1)
        
        obs = ["Walk", "Shop", "Clean"]
        prob, _ = OcultasM.perform_inference(model, obs)
        print("Probabilidad secuencia:", prob)
        
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)

if __name__ == '__main__':
    unittest.main()

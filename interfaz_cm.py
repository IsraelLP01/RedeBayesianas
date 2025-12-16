import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
import CadenasM

# Crear la cadena de Markov inicial
cm_model = CadenasM.CadenaMarkov()

ventana = tk.Tk()
ventana.title("Constructor de Cadenas de Markov")
ventana.geometry("600x700")

# Título principal
titulo = tk.Label(ventana, text="CONSTRUCTOR DE CADENAS DE MARKOV", font=("Arial", 16, "bold"))
titulo.pack(pady=10)

# Frame principal para configuración
frame_config = tk.Frame(ventana)
frame_config.pack(fill="both", expand=True, padx=10)


ventana.mainloop()

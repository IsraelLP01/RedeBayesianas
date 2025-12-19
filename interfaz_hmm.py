import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt

import OcultasM
import ejemplos_modelos

# Crear el modelo HMM inicial
hmm_model = OcultasM.create_hmm_model()

ventana = tk.Tk()
ventana.title("Constructor de HMM")
ventana.geometry("600x800")

# Título principal
titulo = tk.Label(ventana, text="Constructor de Modelo Oculto de Markov", font=("Arial", 16, "bold"))
titulo.pack(pady=10)

# Frame principal para configuración
frame_config = tk.Frame(ventana)
frame_config.pack(fill="both", expand=True, padx=10)

# --- Columna Izquierda: Estados y Transiciones ---
frame_left = tk.Frame(frame_config)
frame_left.pack(side="left", fill="both", expand=True)

# (Funciones de botones movidas abajo)

# 1. Agregar Estado
lb_estado_frame = tk.LabelFrame(frame_left, text="1. Agregar Estado", padx=5, pady=5)
lb_estado_frame.pack(fill="x", pady=5)

tk.Label(lb_estado_frame, text="Nombre:").pack(anchor="w")
entry_estado = tk.Entry(lb_estado_frame)
entry_estado.pack(fill="x", pady=2)

tk.Label(lb_estado_frame, text="Prob. Inicial:").pack(anchor="w")
entry_prob_ini = tk.Entry(lb_estado_frame)
entry_prob_ini.pack(fill="x", pady=2)

def agregar_estado_btn():
    nombre = entry_estado.get().strip()
    prob = entry_prob_ini.get().strip()
    if nombre and prob:
        try:
            OcultasM.add_state(hmm_model, nombre, float(prob))
            messagebox.showinfo("OK", f"Estado '{nombre}' agregado.")
            entry_estado.delete(0, tk.END)
            entry_prob_ini.delete(0, tk.END)
            actualizar_combos()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showerror("Error", "Faltan datos.")

tk.Button(lb_estado_frame, text="Agregar", command=agregar_estado_btn).pack(pady=5)

# 2. Agregar Transición
lb_trans_frame = tk.LabelFrame(frame_left, text="2. Agregar Transición", padx=5, pady=5)
lb_trans_frame.pack(fill="x", pady=5)

tk.Label(lb_trans_frame, text="De:").pack(anchor="w")
combo_de = ttk.Combobox(lb_trans_frame, state="readonly")
combo_de.pack(fill="x", pady=2)

tk.Label(lb_trans_frame, text="A:").pack(anchor="w")
combo_a = ttk.Combobox(lb_trans_frame, state="readonly")
combo_a.pack(fill="x", pady=2)

tk.Label(lb_trans_frame, text="Probabilidad:").pack(anchor="w")
entry_prob_trans = tk.Entry(lb_trans_frame)
entry_prob_trans.pack(fill="x", pady=2)

def agregar_trans_btn():
    de_e = combo_de.get()
    a_e = combo_a.get()
    prob = entry_prob_trans.get().strip()
    if de_e and a_e and prob:
        try:
            OcultasM.set_transition_probabilities(hmm_model, de_e, a_e, float(prob))
            messagebox.showinfo("OK", "Transición agregada.")
            entry_prob_trans.delete(0, tk.END)
        except Exception as e:
             messagebox.showerror("Error", str(e))
    else:
        messagebox.showerror("Error", "Faltan datos.")

tk.Button(lb_trans_frame, text="Agregar", command=agregar_trans_btn).pack(pady=5)


# --- Columna Derecha: Emisiones ---
frame_right = tk.Frame(frame_config)
frame_right.pack(side="right", fill="both", expand=True, padx=(10, 0))

# 3. Agregar Emisión
lb_emis_frame = tk.LabelFrame(frame_right, text="3. Agregar Emisión", padx=5, pady=5)
lb_emis_frame.pack(fill="x", pady=5)

tk.Label(lb_emis_frame, text="Estado Origen:").pack(anchor="w")
combo_emis_est = ttk.Combobox(lb_emis_frame, state="readonly")
combo_emis_est.pack(fill="x", pady=2)

tk.Label(lb_emis_frame, text="Símbolo Observable:").pack(anchor="w")
entry_simbolo = tk.Entry(lb_emis_frame)
entry_simbolo.pack(fill="x", pady=2)

tk.Label(lb_emis_frame, text="Probabilidad:").pack(anchor="w")
entry_prob_emis = tk.Entry(lb_emis_frame)
entry_prob_emis.pack(fill="x", pady=2)

def agregar_emis_btn():
    est = combo_emis_est.get()
    simb = entry_simbolo.get().strip()
    prob = entry_prob_emis.get().strip()
    if est and simb and prob:
        try:
            OcultasM.set_emission_probabilities(hmm_model, est, simb, float(prob))
            messagebox.showinfo("OK", "Emisión agregada.")
            entry_prob_emis.delete(0, tk.END)
            entry_simbolo.delete(0, tk.END)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    else:
        messagebox.showerror("Error", "Faltan datos.")

tk.Button(lb_emis_frame, text="Agregar", command=agregar_emis_btn).pack(pady=5)


# Helpers
def actualizar_combos():
    estados = list(hmm_model["states"].keys())
    combo_de['values'] = estados
    combo_a['values'] = estados
    combo_emis_est['values'] = estados

# --- AREA DE RESOLUCIÓN ---
frame_resolver = tk.Frame(ventana, bg="#f0f0f0", bd=2, relief="groove")
frame_resolver.pack(fill="x", padx=10, pady=20)

tk.Label(frame_resolver, text="Resolver por Forward-Backward", font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=10)

tk.Label(frame_resolver, text="Ingrese Secuencia de Observaciones (separadas por comas, ej: A,B,A):", bg="#f0f0f0").pack()
entry_seq = tk.Entry(frame_resolver, width=50)
entry_seq.pack(pady=5)

def resolver_problema():
    # 1. Validar modelo básico
    if not hmm_model["states"]:
        messagebox.showwarning("Aviso", "No hay estados en el modelo.")
        return

    seq_txt = entry_seq.get().strip()
    if not seq_txt:
        messagebox.showwarning("Aviso", "Ingrese una secuencia de observaciones.")
        return
    
    observaciones = [x.strip() for x in seq_txt.split(",") if x.strip()]
    
    # 2. Calcular
    try:
        prob, alpha, gamma, xi = OcultasM.perform_inference(hmm_model, observaciones)
        
        # Mostrar resultados en nueva ventana o popup
        res_msg = f"Probabilidad Total de la Secuencia: {prob:.4e}"
        messagebox.showinfo("Resultado", res_msg)
        
        # Opcional: Visualizar al resolver
        try:
            fig = OcultasM.visualizar_hmm(hmm_model)
            plt.show()
        except:
            pass
            
    except Exception as e:
        messagebox.showerror("Error en Cálculo", f"No se pudo resolver: {e}")

btn_resolver = tk.Button(frame_resolver, text="RESOLVER", font=("Arial", 11, "bold"), bg="#4CAF50", fg="white", command=resolver_problema)
btn_resolver.pack(pady=10, ipadx=20, ipady=5)

# --- ACCIONES GLOBALES (INFERIOR) ---
frame_acciones = tk.Frame(ventana, pady=10)
frame_acciones.pack(fill="x", padx=10, side="bottom")

def cargar_ejemplo_btn():
    global hmm_model
    try:
        # Cargar modelo y observaciones de prueba
        hmm_model, obs = ejemplos_modelos.crear_ejemplo_hmm()
        
        # Actualizar la UI
        actualizar_combos()
        
        # Poner la secuencia en el entry
        entry_seq.delete(0, tk.END)
        entry_seq.insert(0, ",".join(obs))
        
        messagebox.showinfo("Éxito", "Ejemplo de Detección de Fiebre cargado.")
        
        # Resolver automáticamente
        resolver_problema()
        
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo cargar el ejemplo: {e}")

def limpiar_todo_btn():
    global hmm_model
    hmm_model = OcultasM.create_hmm_model()
    actualizar_combos()
    entry_seq.delete(0, tk.END)
    messagebox.showinfo("Limpiar", "Modelo HMM reiniciado.")

tk.Button(frame_acciones, text="Cargar Ejemplo (Fiebre)", command=cargar_ejemplo_btn, bg="#8e44ad", fg="white", width=20).pack(side="left", padx=10)
tk.Button(frame_acciones, text="Limpiar Todo", command=limpiar_todo_btn, bg="#e74c3c", fg="white", width=20).pack(side="right", padx=10)


ventana.mainloop()

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

import CadenasM
import ejemplos_modelos

# Crear la cadena de Markov inicial
cm_model = CadenasM.CadenaMarkov()

def iniciar_interfaz():
    ventana = tk.Tk()
    ventana.title("Constructor de Cadenas de Markov")
    ventana.geometry("1100x750")
    ventana.configure(bg="#f4f6f9") # Fondo gris azulado muy suave

    # Estilos globales
    style = ttk.Style()
    style.theme_use('clam')
    style.configure("TCombobox", padding=5)

    # Título principal
    frame_titulo = tk.Frame(ventana, bg="#2c3e50")
    frame_titulo.pack(fill="x")
    titulo = tk.Label(frame_titulo, text="Cadenas de Markov", font=("Helvetica", 18, "bold"), bg="#2c3e50", fg="white")
    titulo.pack(pady=15)

    # Frame principal
    main_frame = tk.Frame(ventana, bg="#f4f6f9")
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    # --- PANELES IZQUIERDA (CONTROLES) ---
    left_panel = tk.Frame(main_frame, width=320, bg="#f4f6f9")
    left_panel.pack(side="left", fill="y", padx=(0, 20))

    # Estilo de LabelFrames
    def crear_labelframe(padre, texto):
        lf = tk.LabelFrame(padre, text=texto, padx=15, pady=15, bg="white", font=("Arial", 10, "bold"), fg="#34495e")
        lf.pack(fill="x", pady=10)
        return lf

    # (Funciones de botones movidas abajo)

    # 1. Agregar Estado
    fr_estado = crear_labelframe(left_panel, "1. Agregar Estado")

    tk.Label(fr_estado, text="Nombre del Estado:", bg="white", fg="#555").pack(anchor="w")
    entry_estado = tk.Entry(fr_estado, relief="flat", bg="#ecf0f1")
    entry_estado.config(highlightbackground="#bdc3c7", highlightthickness=1)
    entry_estado.pack(fill="x", pady=(5, 10), ipady=3)

    def agregar_estado_btn():
        nombre = entry_estado.get().strip()
        if nombre:
            cm_model.agregar_estado(nombre)
            messagebox.showinfo("Éxito", f"Estado '{nombre}' agregado.")
            entry_estado.delete(0, tk.END)
            actualizar_combos()
        else:
            messagebox.showerror("Error", "El nombre no puede estar vacío.")

    tk.Button(fr_estado, text="Agregar Estado", command=agregar_estado_btn, bg="#3498db", fg="white", relief="flat", font=("Arial", 9, "bold")).pack(fill="x", pady=5)

    # 2. Agregar Transición
    fr_trans = crear_labelframe(left_panel, "2. Agregar Transición")

    tk.Label(fr_trans, text="Origen:", bg="white", fg="#555").pack(anchor="w")
    combo_de = ttk.Combobox(fr_trans, state="readonly")
    combo_de.pack(fill="x", pady=(2, 8))

    tk.Label(fr_trans, text="Destino:", bg="white", fg="#555").pack(anchor="w")
    combo_a = ttk.Combobox(fr_trans, state="readonly")
    combo_a.pack(fill="x", pady=(2, 8))

    tk.Label(fr_trans, text="Probabilidad (0.0 - 1.0):", bg="white", fg="#555").pack(anchor="w")
    entry_prob = tk.Entry(fr_trans, relief="flat", bg="#ecf0f1")
    entry_prob.config(highlightbackground="#bdc3c7", highlightthickness=1)
    entry_prob.pack(fill="x", pady=(2, 10), ipady=3)

    def agregar_trans_btn():
        de_e = combo_de.get()
        a_e = combo_a.get()
        prob_s = entry_prob.get().strip()
        if de_e and a_e and prob_s:
            try:
                prob = float(prob_s)
                if 0 <= prob <= 1:
                    cm_model.establecer_transicion(de_e, a_e, prob)
                    messagebox.showinfo("Éxito", f"Transición {de_e} -> {a_e} ({prob}) establecida.")
                    entry_prob.delete(0, tk.END)
                else:
                    messagebox.showerror("Error", "La probabilidad debe estar entre 0 y 1.")
            except ValueError:
                messagebox.showerror("Error", "Probabilidad inválida.")
        else:
            messagebox.showerror("Error", "Faltan datos para la transición.")

    tk.Button(fr_trans, text="Establecer Transición", command=agregar_trans_btn, bg="#3498db", fg="white", relief="flat", font=("Arial", 9, "bold")).pack(fill="x", pady=5)

    def actualizar_combos():
        estados = cm_model.estados
        combo_de['values'] = estados
        combo_a['values'] = estados
    
    # 3. Analizar
    def analizar_btn():
        # Validar
        val, msg = cm_model.validar_transiciones()
        if not val:
            messagebox.showwarning("Advertencia", msg)
            return

        # Analizar
        cm_model.analizar_estructura()
        
        # Limpiar resultados
        txt_resultados.config(state="normal")
        txt_resultados.delete("1.0", tk.END)
        
        # Estilizar salida
        res = "--- INFORME DE ANÁLISIS ---\n\n"
        
        if cm_model.es_absorbente:
            res += "[ TIPO DE CADENA: ABSORBENTE ]\n\n"
            res += f"Estados Transitorios: {', '.join(cm_model.estados_transitorios)}\n"
            res += f"Estados Absorbentes:  {', '.join(cm_model.estados_absorbentes)}\n\n"
            
            res += ">> Matriz Fundamental (N) (Pasos esperados entre transitorios):\n"
            res += str(np.round(cm_model.matriz_N, 3)) + "\n\n"
            
            res += ">> Tiempo Esperado hasta Absorción (Desde cada estado):\n"
            for i, e in enumerate(cm_model.estados_transitorios):
                res += f"   - {e}: {cm_model.tiempo_absorcion[i]:.2f} pasos\n"
            res += "\n"
            
            res += ">> Probabilidades de Absorción Final:\n"
            B = cm_model.prob_absorcion
            for i, et in enumerate(cm_model.estados_transitorios):
                for j, ea in enumerate(cm_model.estados_absorbentes):
                    res += f"   - {et} -> Termina en {ea}: {B[i, j]:.2%}\n"
        else:
            res += "[ TIPO DE CADENA: REGULAR / ERGÓDICA ]\n\n"
            estacionaria = cm_model.calcular_estacionaria()
            if estacionaria:
                res += ">> Distribución Estacionaria (Largo Plazo):\n"
                for e, p in estacionaria.items():
                    res += f"   - {e}: {p:.2%}\n"
            else:
                res += "Nota: No se pudo calcular una distribución estacionaria única.\n"
        
        txt_resultados.insert(tk.END, res)
        txt_resultados.config(state="disabled")

        # Visualizar
        mostrar_graficos()

    btn_analizar = tk.Button(left_panel, text="ANALIZAR CADENA", command=analizar_btn, 
                             bg="#27ae60", fg="white", font=("Arial", 11, "bold"), 
                             relief="flat", pady=10, cursor="hand2")
    btn_analizar.pack(fill="x", pady=20)


    # --- PANEL DERECHO (VISUALIZACIÓN Y RESULTADOS) ---
    right_panel = tk.Frame(main_frame, bg="#f4f6f9")
    right_panel.pack(side="right", fill="both", expand=True)

    # Area de Graficos
    lbl_graficos = tk.Label(right_panel, text="Visualización Gráfica", bg="#f4f6f9", font=("Arial", 10, "bold"), fg="#7f8c8d", anchor="w")
    lbl_graficos.pack(fill="x", pady=(0, 5))

    frame_graficos = tk.Frame(right_panel, bg="white", bd=1, relief="solid")
    frame_graficos.config(highlightbackground="#ccc", highlightthickness=1)
    frame_graficos.pack(fill="both", expand=True, pady=(0, 15))
    
    lbl_placeholder = tk.Label(frame_graficos, text="El diagrama y matriz aparecerán aquí tras analizar.", bg="white", fg="#95a5a6")
    lbl_placeholder.pack(expand=True)

    def mostrar_graficos():
        for widget in frame_graficos.winfo_children():
            widget.destroy()
            
        fig = cm_model.visualizar()
        # Ajustar color de fondo de la figura para que coincida con la UI se ve mejor en blanco
        fig.patch.set_facecolor('white')
        
        canvas = FigureCanvasTkAgg(fig, master=frame_graficos)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # Area de Resultados Texto
    lbl_res = tk.Label(right_panel, text="Resultados Numéricos Detallados", bg="#f4f6f9", font=("Arial", 10, "bold"), fg="#7f8c8d", anchor="w")
    lbl_res.pack(fill="x", pady=(0, 5))

    frame_txt = tk.Frame(right_panel, bd=1, relief="solid", bg="white", height=150)
    frame_txt.pack(fill="x")
    # Evitar que el frame se encoja
    frame_txt.pack_propagate(False) 
    
    txt_resultados = tk.Text(frame_txt, state="disabled", font=("Consolas", 10), bg="#fafafa", relief="flat", padx=10, pady=10)
    txt_resultados.pack(fill="both", expand=True)

    # --- ACCIONES GLOBALES (INFERIOR) ---
    frame_acciones = tk.Frame(ventana, pady=10, bg="#f4f6f9")
    frame_acciones.pack(fill="x", padx=10, side="bottom")

    def cargar_ejemplo_btn():
        global cm_model
        try:
            # Reemplazar el modelo actual con el ejemplo
            cm_model = ejemplos_modelos.crear_ejemplo_cm()
            
            # Actualizar la UI
            actualizar_combos()
            messagebox.showinfo("Éxito", "Ejemplo de Clima cargado correctamente.")
            
            # Ejecutar análisis automáticamente
            analizar_btn()
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el ejemplo: {e}")

    def limpiar_todo_btn():
        global cm_model
        cm_model = CadenasM.CadenaMarkov()
        actualizar_combos()
        
        # Limpiar resultados texto
        txt_resultados.config(state="normal")
        txt_resultados.delete("1.0", tk.END)
        txt_resultados.config(state="disabled")
        
        # Limpiar graficos
        for widget in frame_graficos.winfo_children():
            widget.destroy()
        lbl_placeholder = tk.Label(frame_graficos, text="El diagrama y matriz aparecerán aquí tras analizar.", bg="white", fg="#95a5a6")
        lbl_placeholder.pack(expand=True)
        
        messagebox.showinfo("Limpiar", "Modelo reiniciado.")

    tk.Button(frame_acciones, text="Cargar Ejemplo (Clima)", command=cargar_ejemplo_btn, bg="#8e44ad", fg="white", font=("Arial", 10, "bold"), width=20, relief="flat", pady=5).pack(side="left", padx=20)
    tk.Button(frame_acciones, text="Limpiar Todo", command=limpiar_todo_btn, bg="#e74c3c", fg="white", font=("Arial", 10, "bold"), width=20, relief="flat", pady=5).pack(side="right", padx=20)

    ventana.mainloop()

if __name__ == "__main__":
    iniciar_interfaz()

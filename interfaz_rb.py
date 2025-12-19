import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import RedesBa
import math
import json
import itertools

#Inicialización de la red
red = RedesBa.crear_red_bayesiana()

ventana = tk.Tk()
ventana.title("Constructor y Consultor de Red Bayesiana")
ventana.geometry("800x950") 

def guardar_red():
    ruta = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
    if ruta:
        red_serializable = red.copy()
        #Procesamiento especial para TDP porque JSON no acepta tuplas como llaves
        tdp_serial = {}
        for nodo, tablas in red['TDP'].items():
            tdp_serial[nodo] = {str(k): v for k, v in tablas.items()}
        red_serializable['TDP'] = tdp_serial
        
        with open(ruta, 'w') as f:
            json.dump(red_serializable, f, indent=4)
        messagebox.showinfo("Éxito", "Red guardada correctamente.")

def cargar_red():
    global red
    ruta = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if ruta:
        with open(ruta, 'r') as f:
            datos = json.load(f)
            #Reconstruir tuplas en TDP para recuperar datos del JSON
            for nodo, tablas in datos['TDP'].items():
                nuevatabla = {}
                for k, v in tablas.items():
                    #Convertir el string de la tupla de vuelta a tupla real
                    tupla = tuple(eval(k)) if k != "()" else ()
                    nuevatabla[tupla] = v
                datos['TDP'][nodo] = nuevatabla
            red = datos
        actualizar_interfaz()
        messagebox.showinfo("Éxito", "Red cargada correctamente.")

marco = tk.Frame(ventana)
marco.pack(pady=5)
tk.Button(marco, text="Cargar Red", command=cargar_red).pack(side="left", padx=5)
tk.Button(marco, text="Guardar Red", command=guardar_red).pack(side="left", padx=5)

tk.Label(ventana, text="Nombre del nodo:").pack()
nombreNodo = tk.Entry(ventana)
nombreNodo.pack()

tk.Label(ventana, text="Padres (separados por coma):").pack()
padresNodo = tk.Entry(ventana)
padresNodo.pack()

tk.Label(ventana, text="Estados (ej: True,False):").pack()
estadosNodo = tk.Entry(ventana)
estadosNodo.pack()

canvas = tk.Canvas(ventana, width=450, height=250, bg="white", highlightthickness=1)
canvas.pack(pady=10)

def dibujar_red():
    canvas.delete("all")
    nodos = RedesBa.obtener_nodos(red)
    if not nodos:
        canvas.create_text(225, 125, text="Agregue nodos para visualizar", fill="gray")
        return
    radio_red, centro_x, centro_y = 80, 225, 125
    posiciones = {}
    for i, nodo in enumerate(nodos):
        angulo = 2 * math.pi * i / len(nodos)
        posiciones[nodo] = (centro_x + radio_red * math.cos(angulo), centro_y + radio_red * math.sin(angulo))

    for nodo in nodos:
        for p in red['padres'].get(nodo, []):
            if p in posiciones:
                x1, y1 = posiciones[p]
                x2, y2 = posiciones[nodo]
                canvas.create_line(x1, y1, x2, y2, arrow=tk.LAST, fill="black")

    for nodo, (x, y) in posiciones.items():
        canvas.create_oval(x-20, y-20, x+20, y+20, fill="lightgreen", outline="black")
        canvas.create_text(x, y, text=nodo, font=('Arial', 8, 'bold'))

#Actualizar las listas de las comboboxes (o no lee los nodos agregados) y el dibujo de la red bayesiana
def actualizar_interfaz():
    nodos = RedesBa.obtener_nodos(red)
    combo_nodos['values'] = nodos
    tdpNodo['values'] = nodos
    combo_consulta['values'] = nodos
    dibujar_red()

def agregar_nodo_boton():
    nombre = nombreNodo.get().strip()
    padres = [p.strip() for p in padresNodo.get().split(',') if p.strip()]
    estados = [e.strip() for e in estadosNodo.get().split(',') if e.strip()]
    if nombre and estados:
        RedesBa.agregar_variable(red, nombre, estados)
        if padres: RedesBa.establecer_padres(red, nombre, padres)
        actualizar_interfaz()
        [e.delete(0, tk.END) for e in [nombreNodo, padresNodo, estadosNodo]]

tk.Button(ventana, text="Agregar Nodo", command=agregar_nodo_boton, bg="#4caf50", fg="white").pack(pady=5)


#Ventana de modificacion de las probabilidades por nodo
tk.Label(ventana, text="CONFIGURAR PROBABILIDADES", font=('Arial', 10, 'bold')).pack(pady=5)
tdpNodo = ttk.Combobox(ventana, state="readonly")
tdpNodo.pack()

def modificar_tdp_boton():
    nombre = tdpNodo.get()
    if not nombre or nombre not in red['valores']:
        messagebox.showerror("Error", "Seleccione un nodo primero.")
        return
    
    v_tdp = tk.Toplevel(ventana)
    v_tdp.title(f"TDP: {nombre}")
    
    padres = red['padres'].get(nombre, [])
    estados_nodo = red['valores'].get(nombre, [])
    
    #Generar todas las combinaciones de estados posibles de los padres del nodo
    dominios_padres = [red['valores'][p] for p in padres]
    combinaciones = list(itertools.product(*dominios_padres)) if padres else [()]
    
    entradas = {}
    

    for comb in combinaciones:
        texto = f"Si {list(zip(padres, comb))}:" if padres else "Probabilidades base:"
        tk.Label(v_tdp, text=texto, fg="blue").pack(pady=5)        
        row_frame = tk.Frame(v_tdp)
        row_frame.pack(padx=10, pady=5)
        
        entradas[comb] = {}
        
        #Recuperacion de las probabilidades ya ingresadas en caso de que existan y poder modificarlas, si no existen, se queda en
        tdp_existente = red['TDP'].get(nombre, {}).get(comb, {})
        
        for est in estados_nodo:
            tk.Label(row_frame, text=f"P({est}):").pack(side="left")
            ent = tk.Entry(row_frame, width=8)
            
            previo = tdp_existente.get(est, "0.0")
            ent.insert(0, str(previo))
            
            ent.pack(side="left", padx=5)
            entradas[comb][est] = ent

    def guardar():
        try:
            for comb, est_dic in entradas.items():
                probs = {est: float(e.get()) for est, e in est_dic.items()}
                
                #Validación opcional para la suma de probabilidades = 1 aunque no es obligatorio
                if abs(sum(probs.values()) - 1.0) > 0.01:
                    if not messagebox.askyesno("Aviso", f"Las probabilidades para {comb} no suman 1. ¿Guardar de todos modos?"):
                        return
                
                RedesBa.establecer_probabilidad(red, nombre, comb, probs)
            
            v_tdp.destroy()
            messagebox.showinfo("Éxito", f"TDP de '{nombre}' actualizada correctamente.")
        except ValueError:
            messagebox.showerror("Error", "Asegúrese de ingresar solo números decimales.")

    tk.Button(v_tdp, text="Guardar Cambios", command=guardar, bg="#2196f3", fg="white", font=('Arial', 10, 'bold')).pack(pady=10)
    nombre = tdpNodo.get()
    if not nombre: return

    tk.Button(v_tdp, text="Guardar", command=guardar).pack(pady=10)

tk.Button(ventana, text="Abrir Editor TDP", command=modificar_tdp_boton).pack(pady=5)

tk.Label(ventana, text="INFERENCIA", font=('Arial', 10, 'bold')).pack(pady=10)
f_inf = tk.Frame(ventana)
f_inf.pack()

tk.Label(f_inf, text="Consulta (X):").grid(row=0, column=0)
combo_consulta = ttk.Combobox(f_inf, state="readonly", width=15)
combo_consulta.grid(row=0, column=1)

tk.Label(f_inf, text="Evidencia (Nodo:Valor, ...):").grid(row=1, column=0)
evidencia_ent = tk.Entry(f_inf, width=18)
evidencia_ent.grid(row=1, column=1)

def ejecutar_inferencia(metodo):
    try:
        X = combo_consulta.get()
        evidenciacruda = evidencia_ent.get()
        evidencia = {}
        if evidenciacruda:
            for item in evidenciacruda.split(','):
                k, v = item.split(':')
                evidencia[k.strip()] = v.strip()
        
        if metodo == "enum":
            res = RedesBa.inferencia_enumeracion(X, evidencia, red)
        else:
            res = RedesBa.inferencia_eliminacion_variables(X, evidencia, red)
            
        messagebox.showinfo("Resultado", f"Distribución para {X}:\n{res}")
    except Exception as e:
        messagebox.showerror("Error", f"Error en inferencia: {e}")

btn_frame = tk.Frame(ventana)
btn_frame.pack(pady=5)
tk.Button(btn_frame, text="Enumeración", command=lambda: ejecutar_inferencia("enum")).pack(side="left", padx=5)
tk.Button(btn_frame, text="Eliminación Var", command=lambda: ejecutar_inferencia("elim")).pack(side="left", padx=5)

# --- Eliminar ---
combo_nodos = ttk.Combobox(ventana, state="readonly")
combo_nodos.pack(pady=5)
tk.Button(ventana, text="Eliminar Nodo", command=lambda: [RedesBa.eliminar_variable(red, combo_nodos.get()), actualizar_interfaz()], fg="red").pack()

ventana.mainloop()
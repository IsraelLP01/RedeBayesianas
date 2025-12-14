import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx

# Importar lógica
import RedesBa
import CadenasM
import OcultasM

class AplicacionProbabilistica:
    def __init__(self, root):
        self.root = root
        self.root.title("Herramienta Didáctica de Modelos Probabilísticos")
        self.root.geometry("1400x900")

        # Configuración de estilos
        style = ttk.Style()
        style.theme_use('clam')

        # Notebook (Pestañas)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Tab Redes Bayesianas
        self.tab_rb = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_rb, text='Redes Bayesianas')
        self.setup_rb_tab()

        # Tab Cadenas de Markov
        self.tab_cm = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_cm, text='Cadenas de Markov')
        self.setup_cm_tab()

        # Tab Modelos Ocultos de Markov (HMM)
        self.tab_hmm = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_hmm, text='HMM')
        self.setup_hmm_tab()

    def setup_rb_tab(self):
        # ---------------------------------------------------------
        # Layout principal de RB
        # ---------------------------------------------------------
        self.rb_red = RedesBa.crear_red_bayesiana()
        
        paned = ttk.PanedWindow(self.tab_rb, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Panel Izquierdo: Configuración
        frame_left = ttk.Frame(paned, width=350)
        paned.add(frame_left, weight=1)
        
        # 1. Variables
        frame_vars = ttk.LabelFrame(frame_left, text="1. Variables")
        frame_vars.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(frame_vars, text="Nombre:").grid(row=0, column=0)
        self.ent_rb_chnom = ttk.Entry(frame_vars, width=10)
        self.ent_rb_chnom.grid(row=0, column=1)
        
        ttk.Label(frame_vars, text="Valores (csv):").grid(row=1, column=0)
        self.ent_rb_chval = ttk.Entry(frame_vars, width=15)
        self.ent_rb_chval.insert(0, "T,F")
        self.ent_rb_chval.grid(row=1, column=1)
        
        ttk.Button(frame_vars, text="Agregar Var", command=self.rb_add_var).grid(row=2, column=0, columnspan=2, pady=2)
        
        self.lst_rb_vars = tk.Listbox(frame_vars, height=4)
        self.lst_rb_vars.grid(row=3, column=0, columnspan=2, sticky='ew')
        
        # 2. Estructura (Padres)
        frame_struct = ttk.LabelFrame(frame_left, text="2. Estructura (Padres)")
        frame_struct.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(frame_struct, text="Hijo:").grid(row=0, column=0)
        self.cb_rb_hijo = ttk.Combobox(frame_struct, width=10)
        self.cb_rb_hijo.grid(row=0, column=1)
        
        ttk.Label(frame_struct, text="Padre:").grid(row=1, column=0)
        self.cb_rb_padre = ttk.Combobox(frame_struct, width=10)
        self.cb_rb_padre.grid(row=1, column=1)
        
        ttk.Button(frame_struct, text="Agregar Arco", command=self.rb_add_arco).grid(row=2, column=0, columnspan=2, pady=2)

        # 3. Probabilidades
        frame_cpt = ttk.LabelFrame(frame_left, text="3. Probabilidades (CPT)")
        frame_cpt.pack(fill='both', expand=True, padx=5, pady=5)
        
        ttk.Label(frame_cpt, text="Seleccionar Var:").pack()
        self.cb_rb_cpt_var = ttk.Combobox(frame_cpt)
        self.cb_rb_cpt_var.pack(fill='x')
        self.cb_rb_cpt_var.bind("<<ComboboxSelected>>", self.rb_load_cpt_editor)
        
        # Frame scrollable para entradas de CPT
        self.canvas_cpt_scroll = tk.Canvas(frame_cpt)
        self.scroll_cpt = ttk.Scrollbar(frame_cpt, orient="vertical", command=self.canvas_cpt_scroll.yview)
        self.frame_cpt_editor = ttk.Frame(self.canvas_cpt_scroll)
        
        self.frame_cpt_editor.bind(
            "<Configure>",
            lambda e: self.canvas_cpt_scroll.configure(
                scrollregion=self.canvas_cpt_scroll.bbox("all")
            )
        )
        
        self.canvas_cpt_scroll.create_window((0, 0), window=self.frame_cpt_editor, anchor="nw")
        self.canvas_cpt_scroll.configure(yscrollcommand=self.scroll_cpt.set)
        
        self.canvas_cpt_scroll.pack(side="left", fill="both", expand=True)
        self.scroll_cpt.pack(side="right", fill="y")
        
        ttk.Button(frame_cpt, text="Guardar CPT", command=self.rb_save_cpt).pack(fill='x', side='bottom')

        # Panel Central: Visualización
        frame_viz = ttk.LabelFrame(paned, text="Visualización")
        paned.add(frame_viz, weight=2)
        
        self.fig_rb = plt.Figure(figsize=(5, 4), dpi=100)
        self.canvas_rb = FigureCanvasTkAgg(self.fig_rb, master=frame_viz)
        self.canvas_rb.get_tk_widget().pack(fill='both', expand=True)
        
        ttk.Button(frame_viz, text="Refrescar Grafo", command=self.rb_draw_graph).pack(fill='x')

        # Panel Derecho: Inferencia
        frame_inf = ttk.LabelFrame(paned, text="Inferencia")
        paned.add(frame_inf, weight=1)
        
        ttk.Label(frame_inf, text="Variable Query:").pack()
        self.cb_rb_query = ttk.Combobox(frame_inf)
        self.cb_rb_query.pack(fill='x')
        
        ttk.Label(frame_inf, text="Evidencia (Var=Val, ...):").pack()
        self.ent_rb_evi = ttk.Entry(frame_inf)
        self.ent_rb_evi.pack(fill='x')
        
        self.var_algo = tk.StringVar(value="elim")
        ttk.Radiobutton(frame_inf, text="Eliminación Vars", variable=self.var_algo, value="elim").pack()
        ttk.Radiobutton(frame_inf, text="Enumeración", variable=self.var_algo, value="enum").pack()
        
        ttk.Button(frame_inf, text="Inferir", command=self.rb_infer).pack(pady=5)
        
        self.txt_rb_res = tk.Text(frame_inf, height=10)
        self.txt_rb_res.pack(fill='both', expand=True)
        
        self.cpt_entries = {} # {clave_tupla: widget_entry}

    def rb_add_var(self):
        nombre = self.ent_rb_chnom.get().strip()
        vals = [v.strip() for v in self.ent_rb_chval.get().split(',') if v.strip()]
        if not nombre or not vals:
            return
        
        RedesBa.agregar_variable(self.rb_red, nombre, vals)
        self.lst_rb_vars.insert(tk.END, f"{nombre}: {vals}")
        
        # Actualizar combos
        vars_list = self.rb_red['variables']
        self.cb_rb_hijo['values'] = vars_list
        self.cb_rb_padre['values'] = vars_list
        self.cb_rb_cpt_var['values'] = vars_list
        self.cb_rb_query['values'] = vars_list
        
        self.ent_rb_chnom.delete(0, tk.END)

    def rb_add_arco(self):
        hijo = self.cb_rb_hijo.get()
        padre = self.cb_rb_padre.get()
        if not hijo or not padre or hijo == padre:
            return
        
        padres_actuales = self.rb_red['padres'][hijo]
        if padre not in padres_actuales:
            nuevos_padres = padres_actuales + [padre]
            RedesBa.establecer_padres(self.rb_red, hijo, nuevos_padres)
            self.rb_draw_graph()

    def rb_draw_graph(self):
        self.fig_rb.clear()
        ax = self.fig_rb.add_subplot(111)
        
        G = nx.DiGraph()
        for v in self.rb_red['variables']:
            G.add_node(v)
            for p in self.rb_red['padres'][v]:
                G.add_edge(p, v)
                
        pos = nx.shell_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightyellow', edgecolors='black')
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, arrowsize=20)
        
        ax.set_title("Estructura de la Red")
        ax.axis('off')
        self.canvas_rb.draw()

    def rb_load_cpt_editor(self, event=None):
        var = self.cb_rb_cpt_var.get()
        if not var: return
        
        # Limpiar frame
        for w in self.frame_cpt_editor.winfo_children(): w.destroy()
        self.cpt_entries = {}
        
        padres = self.rb_red['padres'][var]
        valores_var = self.rb_red['valores'][var]
        
        # Generar combinaciones de padres
        import itertools
        dominios_padres = [self.rb_red['valores'][p] for p in padres]
        comb_padres = list(itertools.product(*dominios_padres)) if padres else [()]
        
        # Headers
        col = 0
        for p in padres:
            tk.Label(self.frame_cpt_editor, text=p, font=('bold')).grid(row=0, column=col)
            col += 1
            
        for val in valores_var:
            tk.Label(self.frame_cpt_editor, text=f"P({var}={val})", fg='blue').grid(row=0, column=col)
            col += 1
            
        # Filas
        row = 1
        for comb in comb_padres:
            col = 0
            for val_p in comb:
                tk.Label(self.frame_cpt_editor, text=str(val_p)).grid(row=row, column=col)
                col += 1
            
            # Entradas de probabilidad
            for val_var in valores_var:
                ent = ttk.Entry(self.frame_cpt_editor, width=6)
                # Intentar cargar valor existente
                prob_actual = 0.0
                try:
                    if comb in self.rb_red['TDP'][var]:
                         prob_actual = self.rb_red['TDP'][var][comb].get(val_var, 0.0)
                except: pass
                
                # Default uniforme
                if prob_actual == 0.0 and var not in self.rb_red['TDP']:
                     prob_actual = 1.0 / len(valores_var)

                ent.insert(0, str(prob_actual))
                ent.grid(row=row, column=col)
                
                # Guardar referencia para recuperar dato mas tarde
                # key: (var, comb_padres, val_var)
                self.cpt_entries[(var, comb, val_var)] = ent
                col += 1
            row += 1

    def rb_save_cpt(self):
        var = self.cb_rb_cpt_var.get()
        if not var: return
        
        try:
            # Reorganizar datos para RedesBa: TDP[var][tupla_padres] = {val: prob}
            padres = self.rb_red['padres'][var]
            dominios_padres = [self.rb_red['valores'][p] for p in padres]
            import itertools
            comb_padres = list(itertools.product(*dominios_padres)) if padres else [()]
            
            for comb in comb_padres:
                probs = {}
                for val_var in self.rb_red['valores'][var]:
                    entry = self.cpt_entries.get((var, comb, val_var))
                    if entry:
                        val_float = float(entry.get())
                        probs[val_var] = val_float
                
                # Establecer en la red
                # NOTA: RedesBa.establecer_probabilidad espera argumentos ligeramente distintos
                # red['TDP'][variable][valores_padres_convertidos] = probabilidades_convertidas
                # Podemos asignar directamente si cuidamos los tipos
                
                # Validar suma 1
                suma = sum(probs.values())
                if abs(suma - 1.0) > 0.01:
                    messagebox.showwarning("Advertencia", f"Probabilidades para padres {comb} suman {suma:.2f}")
                
                if var not in self.rb_red['TDP']: self.rb_red['TDP'][var] = {}
                self.rb_red['TDP'][var][comb] = probs
            
            messagebox.showinfo("Info", f"CPT de {var} guardada.")
            
        except ValueError:
            messagebox.showerror("Error", "Valores numéricos inválidos")

    def rb_infer(self):
        X = self.cb_rb_query.get()
        evi_str = self.ent_rb_evi.get()
        
        if not X: return
        
        # Parse evidencia
        evidencia = {}
        if evi_str.strip():
            partes = evi_str.split(',')
            for p in partes:
                if '=' in p:
                    k, v = p.split('=')
                    evidencia[k.strip()] = v.strip() # RedesBa convierte tipos internamente si es necesario
                    # Pero OJO: RedesBa.convertir_valor usa str() comparison, así que strings deberían funcionar
        
        # Ejecutar
        try:
            if self.var_algo.get() == "elim":
                res = RedesBa.inferencia_eliminacion_variables(X, evidencia, self.rb_red)
            else:
                res = RedesBa.inferencia_enumeracion(X, evidencia, self.rb_red)
                
            self.txt_rb_res.delete("1.0", tk.END)
            self.txt_rb_res.insert(tk.END, f"Inferencia P({X} | {evidencia})\n")
            for val, prob in res.items():
                self.txt_rb_res.insert(tk.END, f"  {val}: {prob:.4f}\n")
                
        except Exception as e:
            messagebox.showerror("Error Inferencia", str(e))


    def setup_cm_tab(self):
        # ---------------------------------------------------------
        # Layout principal de CM
        # ---------------------------------------------------------
        paned = ttk.PanedWindow(self.tab_cm, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True, padx=5, pady=5)

        # Panel Izquierdo: Configuración y Matriz
        frame_left = ttk.Frame(paned, width=400)
        paned.add(frame_left, weight=1)

        # Sección: Estados
        frame_estados = ttk.LabelFrame(frame_left, text="1. Definir Estados")
        frame_estados.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(frame_estados, text="Estados (separados por coma):").pack(pady=2)
        self.entry_estados_cm = ttk.Entry(frame_estados)
        self.entry_estados_cm.pack(fill='x', padx=5, pady=2)
        self.entry_estados_cm.insert(0, "A,B,C") # Default
        
        ttk.Button(frame_estados, text="Crear Matriz", command=self.crear_matriz_cm).pack(pady=5)

        # Sección: Matriz P
        self.frame_matriz_cm = ttk.LabelFrame(frame_left, text="2. Matriz de Transición")
        self.frame_matriz_cm.pack(fill='both', expand=True, padx=5, pady=5)
        self.entries_matriz_cm = [] # Lista de listas de Entry widgets

        # Sección: Acciones y Análisis
        frame_acciones = ttk.LabelFrame(frame_left, text="3. Análisis")
        frame_acciones.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(frame_acciones, text="Validar y Visualizar", command=self.validar_visualizar_cm).pack(fill='x', pady=2)
        ttk.Button(frame_acciones, text="Calcular Dist. Estacionaria", command=self.calcular_estacionaria_cm).pack(fill='x', pady=2)
        
        # Simulación
        ttk.Separator(frame_acciones, orient='horizontal').pack(fill='x', pady=5)
        ttk.Label(frame_acciones, text="Simulación (pasos):").pack()
        self.spin_pasos_cm = ttk.Spinbox(frame_acciones, from_=1, to=100, width=5)
        self.spin_pasos_cm.set(10)
        self.spin_pasos_cm.pack()
        ttk.Button(frame_acciones, text="Simular Trayectoria", command=self.simular_cm).pack(fill='x', pady=2)

        # Ejemplos
        frame_ejemplos = ttk.LabelFrame(frame_left, text="Ejemplos")
        frame_ejemplos.pack(fill='x', padx=5, pady=5)
        ttk.Button(frame_ejemplos, text="Cargar: Compras Online", command=self.cargar_ejemplo_compras).pack(fill='x', pady=2)
        ttk.Button(frame_ejemplos, text="Cargar: Videojuegos", command=self.cargar_ejemplo_videojuegos).pack(fill='x', pady=2)

        # Panel Derecho: Visualización y Resultados
        frame_right = ttk.Frame(paned)
        paned.add(frame_right, weight=3)
        
        # Canvas Matplotlib
        self.fig_cm = plt.Figure(figsize=(6, 5), dpi=100)
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=frame_right)
        self.canvas_cm.get_tk_widget().pack(fill='both', expand=True)
        
        # Area de texto para logs/resultados
        self.log_cm = tk.Text(frame_right, height=8)
        self.log_cm.pack(fill='x', padx=5, pady=5)

        self.cm_actual = None # Instancia de CadenaMarkov

    def crear_matriz_cm(self):
        # Limpiar frame anterior
        for widget in self.frame_matriz_cm.winfo_children():
            widget.destroy()
            
        estados_raw = self.entry_estados_cm.get()
        self.lista_estados_cm = [e.strip() for e in estados_raw.split(',') if e.strip()]
        
        n = len(self.lista_estados_cm)
        self.entries_matriz_cm = []
        
        # Headers
        tk.Label(self.frame_matriz_cm, text="Desde \\ Hacia").grid(row=0, column=0)
        for j, est in enumerate(self.lista_estados_cm):
            tk.Label(self.frame_matriz_cm, text=est).grid(row=0, column=j+1)
            
        # Rows
        for i, origen in enumerate(self.lista_estados_cm):
            tk.Label(self.frame_matriz_cm, text=origen).grid(row=i+1, column=0)
            row_entries = []
            for j, destino in enumerate(self.lista_estados_cm):
                ent = ttk.Entry(self.frame_matriz_cm, width=7)
                ent.insert(0, "0.0")
                if i == j: # Opcional: poner 1 en diagonal inicial o algo
                   pass 
                ent.grid(row=i+1, column=j+1)
                row_entries.append(ent)
            self.entries_matriz_cm.append(row_entries)
            
    def validar_visualizar_cm(self):
        if not hasattr(self, 'lista_estados_cm') or not self.lista_estados_cm:
            messagebox.showerror("Error", "Primero crea la matriz de estados")
            return

        cm = CadenasM.CadenaMarkov()
        for e in self.lista_estados_cm:
            cm.agregar_estado(e)
            
        n = len(self.lista_estados_cm)
        try:
            for i in range(n):
                for j in range(n):
                    val = float(self.entries_matriz_cm[i][j].get())
                    cm.establecer_transicion(self.lista_estados_cm[i], self.lista_estados_cm[j], val)
        except ValueError:
            messagebox.showerror("Error", "Asegúrate de que todos los campos sean números")
            return
            
        valido, msg = cm.validar_transiciones()
        if not valido:
            messagebox.showwarning("Advertencia", msg)
            
        self.cm_actual = cm
        
        # Visualizar
        self.fig_cm.clear()
        
        # Reutilizar el metodo visualizar de la clase, pero adaptar para Tkinter canvas
        # El metodo visualizar retorna una figure. Pero quiero dibujar en self.fig_cm existente.
        # Copiar logica de visualización aquí para usar self.fig_cm
        ax = self.fig_cm.add_subplot(111)
        
        G = nx.DiGraph()
        for origen in cm.estados:
            G.add_node(origen)
            for destino, prob in cm.transiciones[origen].items():
                if prob > 0:
                    G.add_edge(origen, destino, weight=prob)
                    
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=1500)
        nx.draw_networkx_labels(G, pos, ax=ax)
        
        edges = G.edges(data=True)
        # Dibujar aristas curvas si es necesario
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edges, arrowsize=15, connectionstyle='arc3,rad=0.1')
        
        labels_edge = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels_edge, ax=ax, label_pos=0.3)
        ax.axis('off')
        ax.set_title("Grafo de Transiciones")
        
        self.canvas_cm.draw()
        self.log("Modelo validado y visualizado.")

    def calcular_estacionaria_cm(self):
        if not self.cm_actual:
            return
        res = self.cm_actual.calcular_estacionaria()
        if res:
            texto = "Distribución Estacionaria:\n"
            for k, v in res.items():
                texto += f"  {k}: {v:.4f}\n"
            self.log(texto)
        else:
            self.log("No se pudo calcular distribución estacionaria única.")

    def simular_cm(self):
        if not self.cm_actual:
            return
        pasos = int(self.spin_pasos_cm.get())
        estado_ini = self.lista_estados_cm[0] # Default al primero
        trayectoria = self.cm_actual.simular(estado_ini, pasos)
        self.log(f"Simulacion ({pasos} pasos): {' -> '.join(trayectoria)}")

    def log(self, msg):
        self.log_cm.insert(tk.END, msg + "\n")
        self.log_cm.see(tk.END)

    def cargar_ejemplo_compras(self):
        self.entry_estados_cm.delete(0, tk.END)
        self.entry_estados_cm.insert(0, "Explorando,ViendoProd,Comparando,Carrito,Compra")
        self.crear_matriz_cm()
        
        # Matriz del ejemplo 1
        # E, V, C, R, F
        vals = [
            [0.20, 0.40, 0.20, 0.15, 0.05],
            [0.10, 0.30, 0.25, 0.25, 0.10],
            [0.05, 0.25, 0.30, 0.30, 0.10],
            [0.00, 0.10, 0.15, 0.50, 0.25],
            [0.00, 0.00, 0.00, 0.00, 1.00]
        ]
        self._llenar_matriz_cm(vals)
        self.validar_visualizar_cm()

    def cargar_ejemplo_videojuegos(self):
        self.entry_estados_cm.delete(0, tk.END)
        self.entry_estados_cm.insert(0, "Explorando,Combate,Inventario,Tienda,Guardando")
        self.crear_matriz_cm()
        
        vals = [
            [0.30, 0.40, 0.15, 0.10, 0.05],
            [0.25, 0.45, 0.00, 0.00, 0.30],
            [0.40, 0.00, 0.35, 0.20, 0.05],
            [0.20, 0.10, 0.30, 0.30, 0.10],
            [0.10, 0.20, 0.10, 0.10, 0.50]
        ]
        self._llenar_matriz_cm(vals)
        self.validar_visualizar_cm()

    def _llenar_matriz_cm(self, valores):
        filas = len(self.entries_matriz_cm)
        cols = len(self.entries_matriz_cm[0])
        for i in range(min(filas, len(valores))):
            for j in range(min(cols, len(valores[i]))):
                self.entries_matriz_cm[i][j].delete(0, tk.END)
                self.entries_matriz_cm[i][j].insert(0, str(valores[i][j]))


    def setup_hmm_tab(self):
        # Layout principal de HMM
        paned = ttk.PanedWindow(self.tab_hmm, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Panel Izq
        frame_left = ttk.Frame(paned, width=450)
        paned.add(frame_left, weight=1)
        
        # Definiciones
        frame_def = ttk.LabelFrame(frame_left, text="Definición del Modelo")
        frame_def.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(frame_def, text="Estados Ocultos (sep. coma):").pack()
        self.entry_hmm_states = ttk.Entry(frame_def)
        self.entry_hmm_states.pack(fill='x')
        self.entry_hmm_states.insert(0, "Lluvioso,Soleado")
        
        ttk.Label(frame_def, text="Símbolos Observables (sep. coma):").pack()
        self.entry_hmm_symbols = ttk.Entry(frame_def)
        self.entry_hmm_symbols.pack(fill='x')
        self.entry_hmm_symbols.insert(0, "Caminar,Limpiar,Tienda")
        
        ttk.Button(frame_def, text="Generar Matrices", command=self.generar_matrices_hmm).pack(pady=5)
        
        # Matrices
        self.notebook_matrices_hmm = ttk.Notebook(frame_left)
        self.notebook_matrices_hmm.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.frame_mat_A = ttk.Frame(self.notebook_matrices_hmm)
        self.notebook_matrices_hmm.add(self.frame_mat_A, text='Transición (A)')
        
        self.frame_mat_B = ttk.Frame(self.notebook_matrices_hmm)
        self.notebook_matrices_hmm.add(self.frame_mat_B, text='Emisión (B)')
        
        self.frame_mat_Pi = ttk.Frame(self.notebook_matrices_hmm)
        self.notebook_matrices_hmm.add(self.frame_mat_Pi, text='Inicial (Pi)')
        
        # Inferencia
        frame_inf = ttk.LabelFrame(frame_left, text="Inferencia")
        frame_inf.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(frame_inf, text="Secuencia Observada (sep. coma):").pack()
        self.entry_hmm_obs = ttk.Entry(frame_inf)
        self.entry_hmm_obs.pack(fill='x')
        self.entry_hmm_obs.insert(0, "Caminar,Tienda,Limpiar")
        
        ttk.Button(frame_inf, text="Calcular Probabilidad", command=self.calcular_hmm).pack(pady=5)
        
        # Panel Der
        frame_right = ttk.Frame(paned)
        paned.add(frame_right, weight=2)
        
        self.fig_hmm = plt.Figure(figsize=(6, 5), dpi=100)
        self.canvas_hmm = FigureCanvasTkAgg(self.fig_hmm, master=frame_right)
        self.canvas_hmm.get_tk_widget().pack(fill='both', expand=True)
        
        self.log_hmm = tk.Text(frame_right, height=6)
        self.log_hmm.pack(fill='x', padx=5, pady=5)

        self.hmm_model = None
        self.entries_A = []
        self.entries_B = []
        self.entries_Pi = []

    def generar_matrices_hmm(self):
        # Limpiar
        for f in [self.frame_mat_A, self.frame_mat_B, self.frame_mat_Pi]:
            for w in f.winfo_children(): w.destroy()
            
        states = [s.strip() for s in self.entry_hmm_states.get().split(',') if s.strip()]
        symbols = [s.strip() for s in self.entry_hmm_symbols.get().split(',') if s.strip()]
        self.hmm_states = states
        self.hmm_symbols = symbols
        
        self.entries_A = []
        self.entries_B = []
        self.entries_Pi = []
        
        # Matriz A (NxN)
        tk.Label(self.frame_mat_A, text="De \\ A").grid(row=0, column=0)
        for j, s in enumerate(states):
            tk.Label(self.frame_mat_A, text=s).grid(row=0, column=j+1)
        
        for i, s_from in enumerate(states):
            tk.Label(self.frame_mat_A, text=s_from).grid(row=i+1, column=0)
            row = []
            for j, s_to in enumerate(states):
                e = ttk.Entry(self.frame_mat_A, width=7)
                e.insert(0, str(1.0/len(states))) # Uniforme por defecto
                e.grid(row=i+1, column=j+1)
                row.append(e)
            self.entries_A.append(row)
            
        # Matriz B (NxM)
        tk.Label(self.frame_mat_B, text="Estado \\ Símbolo").grid(row=0, column=0)
        for j, sym in enumerate(symbols):
            tk.Label(self.frame_mat_B, text=sym).grid(row=0, column=j+1)
            
        for i, s in enumerate(states):
            tk.Label(self.frame_mat_B, text=s).grid(row=i+1, column=0)
            row = []
            for j, sym in enumerate(symbols):
                e = ttk.Entry(self.frame_mat_B, width=7)
                e.insert(0, str(1.0/len(symbols)))
                e.grid(row=i+1, column=j+1)
                row.append(e)
            self.entries_B.append(row)
            
        # Vector Pi (N)
        tk.Label(self.frame_mat_Pi, text="Estado").grid(row=0, column=0)
        tk.Label(self.frame_mat_Pi, text="Prob. Inicial").grid(row=0, column=1)
        for i, s in enumerate(states):
            tk.Label(self.frame_mat_Pi, text=s).grid(row=i+1, column=0)
            e = ttk.Entry(self.frame_mat_Pi, width=7)
            e.insert(0, str(1.0/len(states)))
            e.grid(row=i+1, column=1)
            self.entries_Pi.append(e)

    def calcular_hmm(self):
        if not hasattr(self, 'hmm_states'):
            messagebox.showerror("Error", "Genera las matrices primero")
            return
            
        # Construir modelo desde GUI
        model = OcultasM.create_hmm_model()
        try:
            # Pi
            for i, s in enumerate(self.hmm_states):
                val = float(self.entries_Pi[i].get())
                OcultasM.add_state(model, s, val)
                
            # A
            for i, s_from in enumerate(self.hmm_states):
                for j, s_to in enumerate(self.hmm_states):
                    val = float(self.entries_A[i][j].get())
                    OcultasM.set_transition_probabilities(model, s_from, s_to, val)
                    
            # B
            for i, s in enumerate(self.hmm_states):
                for j, sym in enumerate(self.hmm_symbols):
                    val = float(self.entries_B[i][j].get())
                    OcultasM.set_emission_probabilities(model, s, sym, val)
                    
        except ValueError:
            messagebox.showerror("Error", "Valores numéricos inválidos")
            return
            
        # Inferencia
        obs_seq = [x.strip() for x in self.entry_hmm_obs.get().split(',') if x.strip()]
        prob_fwd, _ = OcultasM.perform_inference(model, obs_seq)
        
        self.log_hmm.insert(tk.END, f"Secuencia: {obs_seq}\n")
        self.log_hmm.insert(tk.END, f"Probabilidad (Forward): {prob_fwd:.6e}\n")
        self.log_hmm.see(tk.END)
        
        # Visualizar
        fig = OcultasM.visualizar_hmm(model)
        
        # Actualizar canvas
        # Limpiar figura actual, copiar contenido o recrear
        self.fig_hmm.clear()
        # Truco: usar el manager de la figura nueva para copiar o rehacer el dibujo
        # Más fácil: reimplementar visualización aquí o adaptar OcultasM para recibir ax
        # OcultasM.visualizar_hmm retorna una nueva figura, la cual no es self.fig_hmm
        # Vamos a extraer los axes de la figura retornada y 'pegarlos' es difícil en matplotlib
        # Mejor: modificar OcultasM para que acepte una figura o axes, O simplemente redibujar aquí.
        # Por simplicidad, haré redibujado manual aquí usando los datos del modelo
        
        ax1 = self.fig_hmm.add_subplot(121)
        ax2 = self.fig_hmm.add_subplot(122)
        
        # 1. Grafo
        G = nx.DiGraph()
        for s in self.hmm_states: G.add_node(s)
        for (u, v), p in model["transitions"].items():
            if p > 0.01: G.add_edge(u, v, weight=p)
            
        pos = nx.circular_layout(G)
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightgreen', node_size=1500)
        nx.draw_networkx_labels(G, pos, ax=ax1)
        edges = G.edges(data=True)
        nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=edges, connectionstyle='arc3,rad=0.1')
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1)
        ax1.set_title("Transiciones")
        ax1.axis('off')
        
        # 2. Tabla Emisiones
        cell_text = []
        for s in self.hmm_states:
            row = []
            for sym in self.hmm_symbols:
                p = model["emissions"].get((s, sym), 0.0)
                row.append(f"{p:.2f}")
            cell_text.append(row)
            
        ax2.axis('tight')
        ax2.axis('off')
        table = ax2.table(cellText=cell_text, rowLabels=self.hmm_states, colLabels=self.hmm_symbols, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        
        self.canvas_hmm.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionProbabilistica(root)
    root.mainloop()

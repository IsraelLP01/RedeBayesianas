import tkinter as tk
from tkinter import ttk, messagebox

import RedesBa
#import OcultasM
#import CadenasM

red = RedesBa.crear_red_bayesiana()
global contador
contador = 0

ventana = tk.Tk()
ventana.title("Proyecto Final")
ventana.geometry("400x600")

titulo = tk.Label(ventana, text="CONSTRUCTOR DE RED BAYESIANA")
titulo.pack(pady=20)

texto1 = tk.Label(ventana, text="Ingrese nombre de nodo:")
texto1.pack()
nombreNodo = tk.Entry(ventana)
nombreNodo.pack(pady=5)

texto2 = tk.Label(ventana, text="Ingrese padres del nodo (separados por coma):")
texto2.pack()
padresNodo = tk.Entry(ventana)
padresNodo.pack(pady=10)

texto3 = tk.Label(ventana, text="Ingrese nombre de estados de nodo(separados por coma):")
texto3.pack()
estadosNodo = tk.Entry(ventana)
estadosNodo.pack(pady=10)


def agregar_nodo_boton():
    global contador
    nombre = nombreNodo.get().strip()
    padres = [p.strip() for p in padresNodo.get().split(',') if p.strip()]
    

    if nombre and nombre not in RedesBa.obtener_nodos(red):
        RedesBa.agregar_variable(red, nombre, estadosNodo.get().strip().split(','))
        if padres:
            RedesBa.establecer_padres(red, nombre, padres)
        contador += 1
        # Actualizar la combobox de nodos
        combo_nodos['values'] = RedesBa.obtener_nodos(red)
        if contador >= 15:
            boton_agregar.config(state=tk.DISABLED)
    elif nombre in RedesBa.obtener_nodos(red):
        tk.messagebox.showerror("Error", "Nombre de nodo ya existe.")
    else:
        tk.messagebox.showerror("Error", "Nombre de nodo no definido.")
    nombreNodo.delete(0, tk.END)
    padresNodo.delete(0, tk.END)
    estadosNodo.delete(0, tk.END)
    combo_nodos.set("")


if True:
    boton_agregar = tk.Button(ventana, text="Agregar nodo", command=agregar_nodo_boton)
    boton_agregar.pack(pady=10)

texto3 = tk.Label(ventana, text="Eliminar nodo:")
texto3.pack()
combo_nodos = tk.ttk.Combobox(ventana, values=RedesBa.obtener_nodos(red), state="readonly")
combo_nodos.pack(pady=5)

def eliminar_nodo_boton():
    global contador
    nombre = combo_nodos.get()
    if nombre:
        RedesBa.eliminar_variable(red, nombre)
        contador -= 1
        if contador < 15:
            boton_agregar.config(state=tk.NORMAL)
        combo_nodos['values'] = RedesBa.obtener_nodos(red) #Actualiza lista de nodos a eliminar
        combo_nodos.set("")
    else:
        tk.messagebox.showerror("Error", "Seleccione un nodo para eliminar.")

boton_eliminar = tk.Button(ventana, text="Eliminar nodo", command=eliminar_nodo_boton)
boton_eliminar.pack(pady=10)


texto2 = tk.Label(ventana, text="Red construida:")
texto2.pack()

ventana.mainloop()

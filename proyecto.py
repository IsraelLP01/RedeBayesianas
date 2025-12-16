import tkinter as tk
import subprocess
import sys
import os

def lanzar_rb():
    # Lanzar interfaz de Redes Bayesianas
    subprocess.Popen([sys.executable, "interfaz_rb.py"])

def lanzar_hmm():
    # Lanzar interfaz de HMM
    subprocess.Popen([sys.executable, "interfaz_hmm.py"])

def lanzar_cm():
    # Lanzar interfaz de Cadenas de Markov
    subprocess.Popen([sys.executable, "interfaz_cm.py"])

ventana = tk.Tk()
ventana.title("Proyecto Final - Modelos Probabilísticos")
ventana.geometry("400x400")

# Estilo
font_titulo = ("Helvetica", 16, "bold")
font_btn = ("Helvetica", 12)

lbl_titulo = tk.Label(ventana, text="MENÚ PRINCIPAL", font=font_titulo)
lbl_titulo.pack(pady=30)

btn_rb = tk.Button(ventana, text="Redes Bayesianas", font=font_btn, width=25, height=2, command=lanzar_rb, bg="#e1f5fe")
btn_rb.pack(pady=10)

btn_hmm = tk.Button(ventana, text="Modelos Ocultos de Markov", font=font_btn, width=25, height=2, command=lanzar_hmm, bg="#e8f5e9")
btn_hmm.pack(pady=10)

btn_cm = tk.Button(ventana, text="Cadenas de Markov", font=font_btn, width=25, height=2, command=lanzar_cm, bg="#fff3e0")
btn_cm.pack(pady=10)

tk.Label(ventana, text="Seleccione una opción para continuar", fg="gray").pack(side="bottom", pady=20)

ventana.mainloop()

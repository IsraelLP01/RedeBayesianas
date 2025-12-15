import tkinter as tk
from tkinter import ttk, messagebox

def convertir_valor(valor, valores_posibles):
    """
    Convierte un valor string al tipo original según los valores posibles definidos en la red.
    """
    for v in valores_posibles:
        if str(v) == str(valor):
            return v
    return valor  # Por si no hace match, dejar como está

def obtener_probabilidad_condicional(Y, y_valor, e, red):
    """
    Obtiene P(Y=y_valor | padres(Y)) de la red bayesiana.
    """
    padres_Y = red['padres'][Y]
    
    # Convertir valores de padres al tipo original
    valores_padres = []
    for padre in padres_Y:
        valor = e[padre]
        valor = convertir_valor(valor, red['valores'][padre])
        valores_padres.append(valor)
    
    valores_padres = tuple(valores_padres)
    
    # Convertir valor de Y también
    y_valor = convertir_valor(y_valor, red['valores'][Y])

    try:
        return red['TDP'][Y][valores_padres][y_valor]
    except KeyError:
        print(f"DEBUG ERROR: Key error en TDP de {Y}")
        print(f"  Padres esperados: {padres_Y}")
        print(f"  Valores padres construidos: {valores_padres}")
        print(f"  Claves en TDP: {list(red['TDP'][Y].keys())}")
        return 0.0

def inferencia_enumeracion(X, e, red):
    """
    Inferencia por enumeración.
    X: Variable de consulta
    e: Evidencia (diccionario)
    red: La red bayesiana
    """
    Q = {}
    for x_i in red['valores'][X]:
        e_extendido = e.copy()
        e_extendido[X] = x_i
        Q[x_i] = enum_aux(red['variables'], e_extendido, red)
    return normaliza(Q)

def enum_aux(vars, e, red):
    if not vars:
        return 1.0
    
    Y = vars[0]
    if Y in e:
        y_valor = e[Y]
        prob = obtener_probabilidad_condicional(Y, y_valor, e, red)
        return prob * enum_aux(vars[1:], e, red)
    else:
        suma = 0.0
        for y_valor in red['valores'][Y]:
            e_extendido = e.copy()
            e_extendido[Y] = y_valor
            prob = obtener_probabilidad_condicional(Y, y_valor, e_extendido, red)
            suma += prob * enum_aux(vars[1:], e_extendido, red)
        return suma

def normaliza(Q):
    total = sum(Q.values())
    return {k: v / total if total > 0 else 0.0 for k, v in Q.items()}

def crear_red_bayesiana():
    return {
        'variables': [],
        'valores': {},
        'padres': {},
        'TDP': {}
    }

def agregar_variable(red, nombre, valores_posibles):
    if nombre not in red['variables']:
        red['variables'].append(nombre)
        red['valores'][nombre] = valores_posibles
        red['padres'][nombre] = []
        red['TDP'][nombre] = {}

def establecer_padres(red, variable, padres):
    if variable not in red['variables']:
        return
    for padre in padres:
        if padre not in red['variables']:
            return
    red['padres'][variable] = padres
    red['TDP'][variable] = {}

def establecer_probabilidad(red, variable, valores_padres, probabilidades):
    if variable not in red['variables']:
        print(f"ERROR: Variable {variable} no existe")
        return
    
    # Debug
    # print(f"DEBUG: Setting probs for {variable} | {valores_padres}")

    # Asegurar que los valores de los padres tengan el tipo correcto
    try:
        valores_padres_convertidos = tuple(
            convertir_valor(v, red['valores'][padre]) 
            for v, padre in zip(valores_padres, red['padres'][variable])
        )
    except Exception as e:
        print(f"ERROR: Fallo al convertir padres: {e}")
        return

    # Asegurar que los valores del diccionario tengan el tipo correcto
    try:
        probabilidades_convertidas = {
            convertir_valor(k, red['valores'][variable]): v
            for k, v in probabilidades.items()
        }
    except Exception as e:
        print(f"ERROR: Fallo al convertir probabilidades: {e}")
        return

    # Validar que sume 1
    total = sum(probabilidades_convertidas.values())
    if abs(total - 1.0) > 0.01:
        print(f"ERROR: Probabilidades suman {total}, se esperaba 1.0 para {variable}|{valores_padres}")
        return

    # Guardar
    if variable not in red['TDP']:
        red['TDP'][variable] = {}
    
    red['TDP'][variable][valores_padres_convertidos] = probabilidades_convertidas
    # print(f"DEBUG: Guardado {variable} -> {valores_padres_convertidos}: {probabilidades_convertidas}")

def eliminar_variable(red, nombre):
    if nombre in red['variables']:
        red['variables'].remove(nombre)
        del red['valores'][nombre]
        del red['padres'][nombre]
        del red['TDP'][nombre]
        
        # Eliminar de padres de otras variables
        for var in red['variables']:
            if nombre in red['padres'][var]:
                red['padres'][var].remove(nombre)
                # También eliminar entradas en TDP relacionadas
                nuevas_tdp = {}
                for key, prob_dict in red['TDP'][var].items():
                    # key es una tupla de valores para los padres
                    idx = red['padres'][var].index(nombre) if nombre in red['padres'][var] else -1
                    if idx != -1:
                        nueva_key = key[:idx] + key[idx+1:]
                    else:
                        nueva_key = key
                    nuevas_tdp[nueva_key] = prob_dict
                red['TDP'][var] = nuevas_tdp

def obtener_nodos(red):
    return red['variables']


# ==========================================
# Implementación de Eliminación de Variables
# ==========================================

class Factor:
    """
    Clase para representar un factor en el algoritmo de eliminación de variables.
    """
    def __init__(self, vars, cpt):
        self.vars = vars
        self.cpt = cpt

    def pointwise_product(self, other, red):
        """
        Multiplica este factor con otro factor.
        """
        # Variables unidas
        new_vars = list(set(self.vars) | set(other.vars))
        new_cpt = {}

        # Iterar sobre todas las combinaciones de valores para las nuevas variables
        # Esto es simplificado; para una implementación robusta se necesita un iterador de combinaciones
        all_values = []
        import itertools
        
        # Obtener dominios de las variables
        dominios = [red['valores'][v] for v in new_vars]
        
        for combination in itertools.product(*dominios):
            # Crear asignación
            asignacion = dict(zip(new_vars, combination))
            
            # Obtener valor de self
            key_self = tuple(asignacion[v] for v in self.vars)
            val_self = self.cpt.get(key_self, 0.0)
            
            # Obtener valor de other
            key_other = tuple(asignacion[v] for v in other.vars)
            val_other = other.cpt.get(key_other, 0.0)
            
            # Multiplicar
            new_cpt[tuple(combination)] = val_self * val_other
            
        return Factor(new_vars, new_cpt)

    def sum_out(self, var, red):
        """
        Suma fuera la variable var de este factor.
        """
        if var not in self.vars:
            return self
        
        vars_restantes = [v for v in self.vars if v != var]
        new_cpt = {}
        
        # Iterar sobre las combinaciones de las variables restantes
        import itertools
        dominios = [red['valores'][v] for v in vars_restantes]
        
        for combination in itertools.product(*dominios):
            asignacion_base = dict(zip(vars_restantes, combination))
            suma = 0.0
            
            # Sumar sobre todos los valores de la variable a eliminar
            for val_elim in red['valores'][var]:
                asignacion_completa = asignacion_base.copy()
                asignacion_completa[var] = val_elim
                
                # Reconstruir clave para self.cpt
                key_self = tuple(asignacion_completa[v] for v in self.vars)
                suma += self.cpt.get(key_self, 0.0)
            
            new_cpt[tuple(combination)] = suma
            
        return Factor(vars_restantes, new_cpt)

def make_factor(var, e, red):
    """
    Crea un factor inicial para una variable dada la evidencia.
    """
    vars_factor = red['padres'][var] + [var]
    cpt_factor = {}
    
    # Iterar sobre todas las combinaciones de var y sus padres
    import itertools
    dominios = [red['valores'][v] for v in vars_factor]
    
    for combination in itertools.product(*dominios):
        asignacion = dict(zip(vars_factor, combination))
        
        # Verificar coherencia con evidencia
        consistente = True
        for k, v in asignacion.items():
            if k in e and e[k] != v:
                consistente = False
                break
        
        if consistente:
            # Obtener probabilidad de la TDP
            val_padres = tuple(asignacion[p] for p in red['padres'][var])
            val_var = asignacion[var]
            
            # Usar la función existente para obtener valor crudo
            try:
                prob = red['TDP'][var][val_padres][val_var]
            except KeyError:
                prob = 0.0
            
            cpt_factor[tuple(combination)] = prob
        else:
            cpt_factor[tuple(combination)] = 0.0
            
    return Factor(vars_factor, cpt_factor)

def inferencia_eliminacion_variables(X, e, red):
    """
    Inferencia por eliminación de variables.
    """
    # 1. Identificar variables ocultas (ni consulta ni evidencia)
    vars_ocultas = [v for v in red['variables'] if v != X and v not in e]
    
    # 2. Crear factores iniciales
    factores = []
    for var in red['variables']:
        factores.append(make_factor(var, e, red))
        
    # 3. Eliminar variables ocultas una por una
    for var in vars_ocultas:
        # Encontrar factores que mencionan var
        factores_con_var = [f for f in factores if var in f.vars]
        factores_sin_var = [f for f in factores if var not in f.vars]
        
        if not factores_con_var:
            continue
            
        # Multiplicar todos los factores que contienen var
        producto = factores_con_var[0]
        for f in factores_con_var[1:]:
            producto = producto.pointwise_product(f, red)
            
        # Sumar fuera var
        nuevo_factor = producto.sum_out(var, red)
        
        # Actualizar lista de factores
        factores = factores_sin_var + [nuevo_factor]
        
    # 4. Multiplicar factores restantes
    if not factores:
        return {} # No debería pasar
    
    resultado = factores[0]
    for f in factores[1:]:
        resultado = resultado.pointwise_product(f, red)
        
    # 5. Normalizar
    # El resultado final tendrá solo var X (y quizás vars de evidencia absorbidas no eliminadas si quedó algo mal, pero debería ser solo X)
    # Como e ya se incorporó fijando valores a 0 si no coinciden, en teoría outcome debería tener solo X
    
    # Extraer distribución para X
    distribucion = {}
    idx_X = -1
    if X in resultado.vars:
        idx_X = resultado.vars.index(X)
        
    if idx_X == -1:
        # X era evidencia? O algo pasó. 
        # Si X está en evidencia, retornamos probabilidad 1.0 para ese valor
        if X in e:
            return {val: 1.0 if val == e[X] else 0.0 for val in red['valores'][X]}
        return {}

    for key, val in resultado.cpt.items():
        # key es una tupla de valores para resultado.vars
        val_X = key[idx_X]
        if val_X not in distribucion:
            distribucion[val_X] = 0.0
        distribucion[val_X] += val

    return normaliza(distribucion)


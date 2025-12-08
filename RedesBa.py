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
        print(f"Error para acceder a la TDP:")
        print(f"  Variable: {Y}")
        print(f"  Padres esperados: {padres_Y}")
        print(f"  Valores de padres dados: {valores_padres}")
        print(f"  Valor consultado: {y_valor}")
        print(f"  TDP disponible: {red['TDP'][Y]}")
        raise ValueError(f"No se encontró probabilidad para {Y}={y_valor} con padres {valores_padres}")

# El resto del archivo se queda igual...

def inferencia_enumeracion(X, e, red):
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
        return
    
    # Asegurar que los valores de los padres tengan el tipo correcto
    valores_padres_convertidos = tuple(
        convertir_valor(v, red['valores'][padre]) 
        for v, padre in zip(valores_padres, red['padres'][variable])
    )

    # Asegurar que los valores del diccionario tengan el tipo correcto
    probabilidades_convertidas = {
        convertir_valor(k, red['valores'][variable]): v
        for k, v in probabilidades.items()
    }

    # Validar
    for valor in red['valores'][variable]:
        if valor not in probabilidades_convertidas:
            return
    
    if abs(sum(probabilidades_convertidas.values()) - 1.0) > 0.001:
        return

    red['TDP'][variable][valores_padres_convertidos] = probabilidades_convertidas
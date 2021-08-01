# MÓDELO MATÉMATICO S.I.R
# Importación de librerías
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.animation as ani

def SIR(y, t, b, k):
    """
    Parameters
    ----------
    y : numpy.ndarray
        Envía las aproximaciones de r, s, i.
    t : array
        Usado en la función odeint.
    b : float
        Tasa de Transmisión.
    k : float
        Tasa de Recuperación.

    Returns
    -------
    dydt : list
        Devuelve la derivada parcial de s(t), r(t), i(t)

    """
    s, i, r = y
    dydt = [(-b*s*i), ((b*s*i)-(k*i)), k*i]
    return dydt

def animate(frames):
    """
    Parameters
    ----------
    frames : integer
        Es el iterador:número de imágenes en las que se va a basar el gráfico

    Returns
    -------
    None.
    """
    ax.clear()    #Cada vez que ingrese borra los ejes y vuelve a gráficar.
    ys = sol[:,0] #numpy.ndarray que contiene la derivada parcial de s(t).
    yi = sol[:,1] #numpy.ndarray que contiene la derivada parcial de i(t).
    yr = sol[:,2] #numpy.ndarray que contiene la derivada parcial de r(t).
    #Presentación de la gráfica
    ax.plot(t[:frames],ys[:frames],label="Susceptibles; s(t)", color = "b")
    ax.plot(t[:frames],yi[:frames],label="Infectados; i(t)", color = "r")
    ax.plot(t[:frames],yr[:frames],label ="Recuperados; r(t)", color = "g")
    ax.set_ylim(0,1.05) # Fijación de ejes
    ax.set_xlim(0,140)  # Fijación de ejes
    plt.legend()
    plt.title('MÓDELO S.I.R\n\n' + '   b = ' +str(b)+ '  '*8 + '  k = ' +str(k))
    plt.xlabel("Tiempo en días (t)")
    plt.ylabel('Valores de s(t),i(t),r(t)')
    plt.grid()
    plt.show()

# Valores iniciales para las variables de la población.
N = int(input('Ingrese número de la población (N): '))
S0 = N #Número de individuos susceptibles.
I0 = int(input('Ingrese número de infectados (I): ')) #Número de individuos infectados.
R0 = 0 #Número de individuos recuperados.

# Fracción 
s0 = S0/N # Fracción susceptible de la población 
i0 = I0/N # Fracción infectada de la población 
r0 = R0/N # Fracción recuperada  de la población 

y0 = [s0, i0, r0] #Lista con los valores iniciales para las variables de la población.

# Parámetros del Módelo S.I.R. (Hong Kong)
# b = 1/2 # Tasa de Transmisión.
# k = 1/3 # Tasa de  Recuperación.
# Variación de parámetros (Pandemia sin control en Ecuador)
# b, k = 1.4, 0.07
# Variación de parámetros (Control de la pandemia en Ecuador)
b, k = 0.61, 0.48

# Variación de parámetros (Datos de Ecuador).
# b, k = 0.28, 0.039 # Tasa de Transmisión, Tasa de  Recuperación.

# Pasos temporales (en días)
t = np.linspace(0, 140) #Arrelo de numpy, que contiene el tiempo inicial y final.

sol = odeint(SIR, y0, t, args=(b,k)) #Arreglo de numpy que va a tener el mismo
                                     #número de columnas como elementos de la lista dydt
                                     
fig = plt.figure(figsize = (16, 7)) #Definición de las dimensiones del Lienzo, en donde va la gráfica
ax = plt.axes(xlim=(0, 140), ylim=(0, 1.05)) #Límites de la gráfica.

# Animación del Módelo S.I.R.
anim=ani.FuncAnimation(fig,animate,frames=range(len(t)),interval=100,repeat=False)

# Guardar la animación de la gráfica.
anim.save('modelo_sir.gif', writer='imagemagick')
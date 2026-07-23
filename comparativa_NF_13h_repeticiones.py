#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from clase_resultados import ResultadosESAR
from scipy.interpolate import interp1d

#%% Lector de resultados
def lector_resultados(path):
    '''
    Para levantar archivos de resultados con columnas :
    Nombre_archivo	Time_m	Temperatura_(ºC)	Mr_(A/m)	Hc_(kA/m)	Campo_max_(A/m)	Mag_max_(A/m)	f0	mag0	dphi0	SAR_(W/g)	Tau_(s)	N	xi_M_0
    '''
    with open(path, 'rb') as f:
        codificacion = chardet.detect(f.read())['encoding']

    # Leer las primeras 20 líneas y crear un diccionario de meta
    meta = {}
    with open(path, 'r', encoding=codificacion) as f:
        for i in range(20):
            line = f.readline()
            if i == 0:
                match = re.search(r'Rango_Temperaturas_=_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)', line)
                if match:
                    key = 'Rango_Temperaturas'
                    value = [float(match.group(1)), float(match.group(2))]
                    meta[key] = value
            else:
                # Patrón para valores con incertidumbre (ej: 331.45+/-6.20 o (9.74+/-0.23)e+01)
                match_uncertain = re.search(r'(.+)_=_\(?([-+]?\d+\.\d+)\+/-([-+]?\d+\.\d+)\)?(?:e([+-]\d+))?', line)
                if match_uncertain:
                    key = match_uncertain.group(1)[2:]  # Eliminar '# ' al inicio
                    value = float(match_uncertain.group(2))
                    uncertainty = float(match_uncertain.group(3))
                    
                    # Manejar notación científica si está presente
                    if match_uncertain.group(4):
                        exponent = float(match_uncertain.group(4))
                        factor = 10**exponent
                        value *= factor
                        uncertainty *= factor
                    
                    meta[key] = ufloat(value, uncertainty)
                else:
                    # Patrón para valores simples (sin incertidumbre)
                    match_simple = re.search(r'(.+)_=_([-+]?\d+\.\d+)', line)
                    if match_simple:
                        key = match_simple.group(1)[2:]
                        value = float(match_simple.group(2))
                        meta[key] = value
                    else:
                        # Capturar los casos con nombres de archivo
                        match_files = re.search(r'(.+)_=_([a-zA-Z0-9._]+\.txt)', line)
                        if match_files:
                            key = match_files.group(1)[2:]
                            value = match_files.group(2)
                            meta[key] = value

    # Leer los datos del archivo (esta parte permanece igual)
    data = pd.read_table(path, header=15,
                         names=('name', 'Time_m', 'Temperatura',
                                'Remanencia', 'Coercitividad','Campo_max','Mag_max',
                                'frec_fund','mag_fund','dphi_fem',
                                'SAR','tau',
                                'N','xi_M_0'),
                         usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13),
                         decimal='.',
                         engine='python',
                         encoding=codificacion)

    files = pd.Series(data['name'][:]).to_numpy(dtype=str)
    time = pd.Series(data['Time_m'][:]).to_numpy(dtype=float)
    temperatura = pd.Series(data['Temperatura'][:]).to_numpy(dtype=float)
    Mr = pd.Series(data['Remanencia'][:]).to_numpy(dtype=float)
    Hc = pd.Series(data['Coercitividad'][:]).to_numpy(dtype=float)
    campo_max = pd.Series(data['Campo_max'][:]).to_numpy(dtype=float)
    mag_max = pd.Series(data['Mag_max'][:]).to_numpy(dtype=float)
    xi_M_0=  pd.Series(data['xi_M_0'][:]).to_numpy(dtype=float)
    SAR = pd.Series(data['SAR'][:]).to_numpy(dtype=float)
    tau = pd.Series(data['tau'][:]).to_numpy(dtype=float)

    frecuencia_fund = pd.Series(data['frec_fund'][:]).to_numpy(dtype=float)
    dphi_fem = pd.Series(data['dphi_fem'][:]).to_numpy(dtype=float)
    magnitud_fund = pd.Series(data['mag_fund'][:]).to_numpy(dtype=float)

    N=pd.Series(data['N'][:]).to_numpy(dtype=int)
    return meta, files, time,temperatura,Mr, Hc, campo_max, mag_max, xi_M_0, frecuencia_fund, magnitud_fund , dphi_fem, SAR, tau, N
#%% LECTOR CICLOS
def lector_ciclos(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()[:8]

    metadata = {'filename': os.path.split(filepath)[-1],
                'Temperatura':float(lines[0].strip().split('_=_')[1]),
        "Concentracion_g/m^3": float(lines[1].strip().split('_=_')[1].split(' ')[0]),
            "C_Vs_to_Am_M": float(lines[2].strip().split('_=_')[1].split(' ')[0]),
            "pendiente_HvsI ": float(lines[3].strip().split('_=_')[1].split(' ')[0]),
            "ordenada_HvsI ": float(lines[4].strip().split('_=_')[1].split(' ')[0]),
            'frecuencia':float(lines[5].strip().split('_=_')[1].split(' ')[0])}

    data = pd.read_table(os.path.join(os.getcwd(),filepath),header=7,
                        names=('Tiempo_(s)','Campo_(Vs)','Magnetizacion_(Vs)','Campo_(kA/m)','Magnetizacion_(A/m)'),
                        usecols=(0,1,2,3,4),
                        decimal='.',engine='python',
                        dtype= {'Tiempo_(s)':'float','Campo_(Vs)':'float','Magnetizacion_(Vs)':'float',
                               'Campo_(kA/m)':'float','Magnetizacion_(A/m)':'float'})
    t     = pd.Series(data['Tiempo_(s)']).to_numpy()
    H_Vs  = pd.Series(data['Campo_(Vs)']).to_numpy(dtype=float) #Vs
    M_Vs  = pd.Series(data['Magnetizacion_(Vs)']).to_numpy(dtype=float)#A/m
    H_kAm = pd.Series(data['Campo_(kA/m)']).to_numpy(dtype=float)*1000 #A/m
    M_Am  = pd.Series(data['Magnetizacion_(A/m)']).to_numpy(dtype=float)#A/m

    return t,H_Vs,M_Vs,H_kAm,M_Am,metadata
#%% funcion extraer SAR, tau y Hc de resultados 
def extraer_SAR_tau(resultados):
    SAR = []
    tau = []
    Hc = []
    for res in resultados:
        meta,_,_,_,_,_,_,_,_,_,_,_,_,_,_ = lector_resultados(res)   
        SAR.append(meta['SAR_W/g'])
        tau.append(meta['tau_ns'])
        Hc.append(meta['Hc_kA/m']) 
    return SAR, tau, Hc
#%% funcion banda temperatura
def banda_temperatura(t, T, N=500, kind='linear'):
    """
    Interpola varias curvas T(t) sobre una grilla temporal común y
    calcula estadísticas punto a punto.

    Parameters
    ----------
    t : list of np.ndarray
        Lista de vectores de tiempo.
    T : list of np.ndarray
        Lista de vectores de temperatura.
    N : int, optional
        Número de puntos de la grilla común.
    kind : str, optional
        Tipo de interpolación (interp1d).

    Returns
    -------
    tt : list of np.ndarray
        Lista original de tiempos.
    TT : list of np.ndarray
        Lista original de temperaturas.
    t_common : np.ndarray
        Grilla temporal común.
    Tmin : np.ndarray
        Temperatura mínima en cada instante.
    Tmax : np.ndarray
        Temperatura máxima en cada instante.
    Tmean : np.ndarray
        Temperatura promedio en cada instante.
    """

    # intervalo temporal común
    tmin = max(tt.min() for tt in t)
    tmax = min(tt.max() for tt in t)

    t_common = np.linspace(tmin, tmax, N)

    # interpolación
    Ti = []
    for tt, TT in zip(t, T):
        f = interp1d(tt, TT, kind=kind)
        Ti.append(f(t_common))

    Ti = np.asarray(Ti)

    # estadísticas
    Tmin  = np.min(Ti, axis=0)
    Tmax  = np.max(Ti, axis=0)
    Tmean = np.mean(Ti, axis=0)

    return t, T, t_common, Tmin, Tmax, Tmean
#%% Obtengo ciclos y resultados para cada concentracion - Todo a 300 kHz

ciclos = glob("data/**/*ciclo_promedio_H_M.txt",recursive=True)
resultados = glob("data/**/*resultados.txt",recursive=True)

ciclos.sort()
resultados.sort()
conc =  19.8 #g/L

for p in ciclos:
    print('  ',p)

for res in resultados:
    print('  ',res)
print('-'*50)    
SAR, tau, Hc = extraer_SAR_tau(resultados)
#%% ploteo ciclos 
fig00, ax =plt.subplots(figsize=(8,6),constrained_layout=True,sharey=True,sharex=True)

ax.set_ylabel('M (A/m)')

for i,e in enumerate(ciclos):
    if '152dA' in e:
        _,_,_, H,M,_ = lector_ciclos(ciclos[i])
        ax.plot(H/1000,M,'-',label=f'{SAR[i]:.2uS}')

ax.grid()
ax.set_xlabel('H (kA/m)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='ESAR (W/g)')
plt.suptitle(f'Comparativa ciclos promedio NF@cit 260527\n300 kHz & 58 kA/m')
plt.savefig('0_ciclos_promedio_NF@cit_260527.png',dpi=300)

#%%
res=[]
print('Resultados primera', '='*80,'\n')
for r in resultados:
    res.append(ResultadosESAR(os.path.dirname(r)))

rates = []
for i,r in enumerate(res):
    dt = r.time[-1]-r.time[0]
    dT = r.temperatura[-1]-r.temperatura[0]
    rate=dT/dt
    print(f'WRate = {rate:.2f} °C/s')
    rates.append(rate)
    Wrate=ufloat(np.mean(rates),np.std(rates)   )
print('-'*50)
print(f"ESAR primera: {np.mean(SAR)}")
print(f" tau primera: {np.mean(tau)}") 
print(f"  Hc primera: {np.mean(Hc)}")
print(f"       WRate: {Wrate:.1uS} °C/s")
print('-'*50)
#%% Ploteo comparativa templogs
t,T=[],[]
fig01, ax = plt.subplots(figsize=(10,6),constrained_layout=True)
for i,r in enumerate(res):
    t.append(r.time)
    T.append(r.temperatura)
    ax.plot(r.time,r.temperatura,'.-',label=f'{rates[i]:.1f} °C/s')
ax.grid()
ax.set_ylabel('T (°C)')
ax.set_xlabel('t (s)')
ax.legend(loc='upper left',frameon=True,shadow=True,title='Warming Rate')
plt.suptitle(f'Templogs NF@cit 260527\n300 kHz & 58 kA/m')    
plt.savefig('0_templogs_NF@cit_260527.png',dpi=300)

tt_1, TT_1, t_common_1, Tmin_1, Tmax_1, Tmean_1 = banda_temperatura(t, T)

fig02,ax = plt.subplots(figsize=(9,4),constrained_layout=True,sharex=True)

for t, T in zip(tt_1, TT_1):
    ax.plot(t, T, '--', c='C0',alpha=0.3)
ax.fill_between(t_common_1, Tmin_1, Tmax_1,alpha=0.3,color='C0')
ax.plot(t_common_1, Tmean_1,'C0-', lw=1.5, label=f'NF@cit 260527 - {Wrate:.1uS} °C/s')

ax.set_ylabel('T (°C)')
ax.grid()
ax.legend(loc='upper left',frameon=True,shadow=True,title='Muestra  -  Warming Rate')
ax.set_xlabel('t (s)')
plt.suptitle('Comparativa templogs - $f=300$ kHz  $H_0=58$ kA/m')
plt.savefig('0_templogs_promedio_NF@cit_260527.png',dpi=300)
plt.show()
#%% ploteo comparativo de errorbars de ESAR
cuadro = '$f=300$ kHz\n$H_0=58$ kA/m'
categorias = ['260527']
x = np.arange(len(categorias))

fig03, ax = plt.subplots(figsize=(9,5),constrained_layout=True)

sep = 0.25

for i,s in enumerate(SAR):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.set_ylabel('ESAR (W/g)')
ax.set_title('ESAR NF@cit 260527')
ax.grid(axis='y', alpha=0.3)

ax.text(0.9,0.9,cuadro, transform=ax.transAxes, 
        va='top', ha='center', fontsize=12,
        bbox=dict(alpha=0.8,facecolor='white'))
plt.show()
#%% ploteo comparativo de errorbars de tau
fig04, ax = plt.subplots(figsize=(9,5),constrained_layout=True)

sep = 0.25

for i,s in enumerate(tau):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

# for i,s in enumerate(tau_13_mala):
#     ax.bar(1+i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C3')

# for i,s in enumerate(tau_13_buena):
#     ax.bar(2+i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C2')

# for i,s in enumerate(tau_13_AV):
#     ax.bar(3+i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C1')

# for i,s in enumerate(tau_13_AN):
#     ax.bar(4+i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C4')

ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.set_ylabel(r'$\tau$ (ns)')
#ax.set_xlabel('Categoría')
ax.set_title(r'$\tau$ NF@cit 260527')
ax.grid(axis='y', alpha=0.3)

ax.text(0.9,0.9,cuadro, transform=ax.transAxes, 
        va='top', ha='center', fontsize=12,
        bbox=dict(alpha=0.8,facecolor='white'))
plt.show()
#%% Idem Hc
fig05, ax = plt.subplots(figsize=(9,5),constrained_layout=True)

sep = 0.25
for i,s in enumerate(Hc):
    ax.bar(i*sep-sep, s.n, yerr=s.s, width=0.2, capsize=5, color='C0')

ax.set_xticks(x)
ax.set_xticklabels(categorias)
ax.set_ylabel('H$_c$ (kA/m)')
ax.set_title('H$_c$ NF@cit 260527')
ax.grid(axis='y', alpha=0.3)

ax.text(0.9,0.9,cuadro, transform=ax.transAxes, 
        va='top', ha='center', fontsize=12,
        bbox=dict(alpha=0.8,facecolor='white'))
plt.show()
#%%
print(f'Muestra = {nombre_M1}')
print(f'Concentracion = {conc_M1:.1f} g/L')
print(f'ESAR = {np.mean(SAR_M1):.2uS} W/g')
print(f'tau = {np.mean(tau_M1):.1uS} ns')
print(f'Hc = {np.mean(Hc_M1):.2uS} kA/m') 
# %%

#%% Salvo figs
fig00.savefig('00_ciclos_promedio_NF@cit_260527.png',dpi=300)
fig01.savefig('01_templogs_NF@cit_260527.png',dpi=300)
fig02.savefig('02_templogs_promedio_NF@cit_260527.png',dpi=300)
fig03.savefig('03_ESAR_NF@cit_260527.png',dpi=300)
fig04.savefig('04_tau_NF@cit_260527.png',dpi=300)
fig05.savefig('05_Hc_NF@cit_260527.png',dpi=300)

# %%





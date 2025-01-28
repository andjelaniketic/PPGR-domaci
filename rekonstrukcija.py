import numpy as np
from numpy import linalg
import math
import plotly.graph_objects as go
import plotly.express as px

# def deveta(P5, P7, P8):
#     duzina = (P5[1] - P8[1])/3
#     P9 = [0, duzina + P8[1]]
    
#     sirina = (P7[0] - P8[0])/3
#     P9[0] = P8[0] + sirina
    
#     return P9

def piksel(tacka):
    x, y, z = tacka
    return [1600 - x, y, z]

# LEVA SLIKA
P1L = piksel([356, 821, 1])
P2L = piksel([653, 778, 1])
P3L = piksel([653, 593, 1])
P4L = piksel([402, 616, 1])
P5L = piksel([253, 622, 1])
P6L = piksel([609, 589, 1])
P7L = piksel([617, 391, 1])
P8L = piksel([313, 416, 1])

Q1L = piksel([733, 454, 1])
Q2L = piksel([988, 576, 1])
Q3L = piksel([1042, 519, 1])
Q4L = piksel([791, 404, 1])
Q5L = piksel([723, 316, 1])
Q6L = piksel([1010, 440, 1])
Q7L = piksel([1067, 384, 1])
Q8L = piksel([786, 271, 1])

R1L = piksel([1039, 785, 1])
R2L = piksel([1350, 1012, 1])
R3L = piksel([1460, 864, 1])
R4L = piksel([1165, 671, 1])
R5L = piksel([1043, 754, 1])
R6L = piksel([1370, 985, 1])
R7L = piksel([1483, 832, 1])
R8L = piksel([1166, 639, 1])

# DESNA SLIKA
P1D = piksel([369, 608, 1])
P2D = piksel([596, 657, 1])
P3D = piksel([683, 516, 1])
P4D = piksel([484, 478, 1])
P5D = piksel([302, 407, 1])
P6D = piksel([553, 452, 1])
P7D = piksel([661, 323, 1])
P8D = piksel([435, 291, 1])

# temena = [p1, p2, p3, p5, p6, p7, p8]
# p4 = osmoteme(temena)
# print(p4)
# p9 = deveta(p5, p7, p8)
# print("p9 desna: ", p9)

Q1D = piksel([816, 440, 1])
Q2D = piksel([973, 587, 1])
Q3D = piksel([1053, 556, 1])
Q4D = piksel([891, 416, 1])
Q5D = piksel([818, 306, 1])
Q6D = piksel([995, 452, 1])
Q7D = piksel([1079, 424, 1])
Q8D = piksel([895, 284, 1])


R1D = piksel([909, 758, 1])
R2D = piksel([1076, 1024, 1])
R3D = piksel([1274, 952, 1])
R4D = piksel([1077, 702, 1])
R5D = piksel([913, 723, 1])
R6D = piksel([1084, 994, 1])
R7D = piksel([1289, 920, 1])
R8D = piksel([1087, 673, 1])

leve8 = np.array([P8L, P7L, P6L, P5L, R8L, R7L, R6L, R5L])
desne8 = np.array([P8D, P7D, P6D, P5D, R8D, R7D, R6D, R5D])

leve8 = np.array(leve8, dtype=float)
desne8 = np.array(desne8, dtype=float)

leve = np.array([P8L, P7L, P6L, P5L, P4L, P3L, P2L, P1L,
                R8L, R7L, R6L, R5L, R4L, R3L, R2L, R1L,
                Q8L, Q7L, Q6L, Q5L, Q4L, Q3L, Q2L, Q1L])
desne = np.array([P8D, P7D, P6D, P5D, P4D, P3D, P2D, P1D,
                R8D, R7D, R6D, R5D, R4D, R3D, R2D, R1D,
                Q8D, Q7D, Q6D, Q5D, Q4D, Q3D, Q2D, Q1D])

# Fundamentalna matrica F
def fundamentalna(u, v):
    a1, a2, a3 = u
    b1, b2, b3 = v

    F = [ 
        a1*b1, a2*b1, a3*b1,
        a1*b2, a2*b2, a3*b2,
        a1*b3, a2*b3, a3*b3
    ]
    return F

jed8 = np.array([fundamentalna(u, v) for u, v in zip(leve8, desne8)])
print(jed8)

U, S, V = np.linalg.svd(jed8)
n = len(V)
# koeficijenti matrice
Fvector = V[n-1]

# FF = Fvector.reshape(-1, 3)
FF = np.array([Fvector[i:i+3] for i in range(0, len(Fvector), 3)])
print("FF",FF)

# testiranje
# def test(x, y, FF):
#     return np.dot(y, np.dot(FF, x))

# testiranje = [float(test(x, y, FF)) for x, y in zip(leve8, desne8)]
# print("TEST", testiranje)

# determinanta = np.linalg.det(FF)
# print("Determinanta", determinanta)

# EPIPOLOVI
U, DD, V = np.linalg.svd(FF)
e1 = V[2, :]
e1 = (1/e1[2])*e1

e2 = U[:, 2]
e2 = (1/e2[2])*e2

diag1 = np.diag([1, 1, 0])
DD1 = np.dot(diag1, np.diag(DD))
# popravka FF
FF1 = np.dot(U, np.dot(DD1, V))

# # popravka FF 
# U, S, V = np.linalg.svd(FF)
# S1 = np.diag([S[0], S[1], 0])
# FF1 = np.dot(U, np.dot(S1, V))

# det1 = np.linalg.det(FF1)
# print("FF1:\n", FF1)
# print("det FF: ", determinanta, " vs det FF1: ", det1)

# osnovna matrica E
K1 = np.array([
    [1300, 0, 800],
    [0, 1300, 600],
    [0, 0, 1]
])
K1T = np.transpose(K1)

EE = np.dot(np.dot(K1T, FF1), K1)
# print("EE:\n", EE)

# dekompozicija EE
Q0 = [
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
]
E0 = [
    [0, 1, 0],
    [-1, 0, 0],
    [0, 0, 0]
]
# print(Q0)
# print(E0)

U, SS, V = np.linalg.svd(EE)
V = -(np.transpose(V))
EC = np.dot(np.dot(U, E0), np.transpose(U))
AA = np.dot(np.dot(U, Q0), np.transpose(V))

def koso2v(A):
    return np.array([A[2, 1], A[0, 2], A[1, 0]])

CC = koso2v(EC)
# print(np.linalg.det(U), np.linalg.det(V))
# razlicitog su mi znaka determinante
# menjam EE u -EE
# EE = -EE
# U, SS, V = np.linalg.svd(EE)
# V = V.T
# print(np.linalg.det(U), np.linalg.det(V))

# EC = np.dot(U, np.dot(E0, np.transpose(U)))
# AA = np.dot(U, np.dot(Q0, V))
# print("EC", EC)
# print("AA", AA)

# print(np.dot(EC, AA), EE)
# ECAA = np.dot(EC, AA)
# print()
# print((EE[0,0] / ECAA[0,0]) * ECAA)

# Matrice kamera (u koordinatnom sistemu druge kamere)
T2 = np.array([
    [1300, 0, 800, 0],
    [0, 1300, 600, 0],
    [0, 0, 1, 0]
])
# print("orijentacija prve kamere u sistemu druge kamere:")
# print(np.transpose(AA))
# print()

CC1 = (np.dot(-(np.transpose(AA)), CC))
print("CC1:")
print(CC1)

temp1 = np.transpose(np.dot(K1, np.transpose(AA)))
T1 = np.transpose(np.vstack([temp1, np.dot(K1, CC1)]))

print("T1")
print(T1)

# triang opsti slucaj
# T1p = np.array([
#     [-2,-1,0,2],
#     [-3, 0, 1,0],
#     [-1, 0, 0, 0]
#     ])
# T2p = np.array([
#     [2, -2, 0, -2],
#     [0, -3, 2, -2],
#     [0, -1, 0, 0]
# ])
# M1 = np.array([5, 3, 1])
# M2 = np.array([-2, 1, 1])

def jednacine(T1, T2, m1, m2):

    eq1 = m1[1] * T1[2] - m1[2] * T1[1]
    eq2 = -m1[0] * T1[2] + m1[2] * T1[0]
    eq3 = m2[1] * T2[2] - m2[2] * T2[1]
    eq4 = -m2[0] * T2[2] + m2[2] * T2[0]
    
    return np.array([eq1, eq2, eq3, eq4])

def UAfine(XX):
    XX = np.array(XX)
    normalizovano = XX / XX[-1]
    return normalizovano[:-1]


def triang(T1, T2, M1, M2):
    jedM = jednacine(T1, T2, M1, M2)
    _, _, V = np.linalg.svd(jedM)
    rez = UAfine(V[3])
    return rez

# triangulacija tacaka sa slike
print("T1:")
print(T1)
print("T2:")
print(T2)

print("AA: ", AA)
print("FF: ", FF)
print("FF1: ", FF1)
print("EE: ", EE)
print("EC: ", EC)
print("E0: ", E0)

# print(tacke3D)

tacke3D = np.array(list(map(lambda x1, x2: triang(T1, T2, x1, x2), leve, desne)))

print("3D Tacke:")
for i, point in enumerate(tacke3D):
    print(f"Point {i+1}: {point}")

temenaKocke = np.array(tacke3D)
# print(temenaKocke)
ivice = [[0, 1], [0, 3], [0, 4],
         [1, 2], [1, 5],
         [2, 3], [2, 6],
         [3, 7],
         [4, 5], [4, 7],
         [5, 6],
         [6, 7],
         [8, 9], [8, 11], [8, 12],
         [9, 10], [9, 13],
         [10, 11], [10, 14],
         [11, 15],
         [12, 15], [12, 13], 
         [13, 14],
         [14, 15],
         [16, 17], [16, 19], [16, 20],
         [17, 18], [17, 21],
         [18, 19], [18, 22],
         [19, 23],
         [20, 21], [21, 22]
         [22, 23]
         ]

# #  prikaz scene - koristi "plotly" biblioteku
def prikazKocke(): 
    # izdvajamo x,y,z koordinate svih tacaka
    xdata = (np.transpose(temenaKocke))[0]
    ydata = (np.transpose(temenaKocke))[1]
    zdata = (np.transpose(temenaKocke))[2]
    # u data1 ubacujemo sve sto treba naccrtati
    data1 = []
    # za svaku ivicu crtamo duz na osnovu koordinata
    for i in range(len(ivice)):
        data1.append(go.Scatter3d(x=[xdata[ivice[i][0]], xdata[ivice[i][1]]], y=[ydata[ivice[i][0]], ydata[ivice[i][1]]],z=[zdata[ivice[i][0]], zdata[ivice[i][1]]]))
    fig = go.Figure(data = data1 )
    # da ne prikazuje legendu
    fig.update_layout(showlegend=False)
    fig.show()
    # pravi html fajl (ako zelite da napravite "rotatable" 3D rekonstruciju)
    # birate kao parametar velicinu apleta. fulhtml=False je vazno da ne bi pravio ogroman fajl
    # ovde stavite neki vas folder
    fig.write_html("C:/Users/Lenovo/Desktop/PPGR/3Drekonstrukcija/jojs.html", include_plotlyjs = 'cdn', default_width = '800px', default_height = '600px', full_html = False) #Modifiy the html file
    fig.show()

prikazKocke()
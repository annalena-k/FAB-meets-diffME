from __future__ import division
from madjax import wavefunctions
from madjax.madjax_patch import complex
from jax.numpy import where


def FFV2_0(F1,F2,V3,COUP):
    TMP0 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+1j*(V3[4])))+F1[3]*(F2[4]*(V3[3]-1j*(V3[4]))+F2[5]*(V3[2]-V3[5])))
    vertex = COUP*-1j * TMP0
    return vertex


def FFV2_5_0(F1,F2,V3,COUP1,COUP2):
    vertex = FFV2_0(F1,F2,V3,COUP1)
    tmp = FFV5_0(F1,F2,V3,COUP2)
    vertex += tmp
    return vertex


def FFV2_1(F2,V3,COUP,M1,W1):
    F1 = wavefunctions.WaveFunction(size=6)
    F1[0] = F2[0]+V3[0]
    F1[1] = F2[1]+V3[1]
    P1 = [-1.0*complex(F1[0]).real, -1.0*complex(F1[1]).real, -1.0*complex(F1[1]).imag, -1.0*complex(F1[0]).imag]
    denom = COUP/(P1[0]**2-P1[1]**2-P1[2]**2-P1[3]**2 - M1 * (M1 -1j* W1))
    F1[2]= denom*1j * M1*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+1j*(V3[4])))
    F1[3]= denom*-1j * M1*(F2[4]*(-V3[3]+1j*(V3[4]))+F2[5]*(-V3[2]+V3[5]))
    F1[4]= denom*(-1j)*(F2[4]*(P1[0]*(V3[2]+V3[5])+(P1[1]*(-V3[3]+1j*(V3[4]))+(P1[2]*(-1)*(+1j*(V3[3])+V3[4])-P1[3]*(V3[2]+V3[5]))))+F2[5]*(P1[0]*(V3[3]+1j*(V3[4]))+(P1[1]*(-V3[2]+V3[5])+(P1[2]*(-1j*(V3[2])+1j*(V3[5]))-P1[3]*(V3[3]+1j*(V3[4]))))))
    F1[5]= denom*(-1j)*(F2[4]*(P1[0]*(V3[3]-1j*(V3[4]))+(P1[1]*(-1)*(V3[2]+V3[5])+(P1[2]*(+1j*(V3[2]+V3[5]))+P1[3]*(V3[3]-1j*(V3[4])))))+F2[5]*(P1[0]*(V3[2]-V3[5])+(P1[1]*(-1)*(V3[3]+1j*(V3[4]))+(P1[2]*(+1j*(V3[3])-V3[4])+P1[3]*(V3[2]-V3[5])))))
    return F1



def FFV2_2(F1,V3,COUP,M2,W2):
    F2 = wavefunctions.WaveFunction(size=6)
    F2[0] = F1[0]+V3[0]
    F2[1] = F1[1]+V3[1]
    P2 = [-1.0*complex(F2[0]).real, -1.0*complex(F2[1]).real, -1.0*complex(F2[1]).imag, -1.0*complex(F2[0]).imag]
    denom = COUP/(P2[0]**2-P2[1]**2-P2[2]**2-P2[3]**2 - M2 * (M2 -1j* W2))
    F2[2]= denom*1j*(F1[2]*(P2[0]*(V3[2]+V3[5])+(P2[1]*(-1)*(V3[3]+1j*(V3[4]))+(P2[2]*(+1j*(V3[3])-V3[4])-P2[3]*(V3[2]+V3[5]))))+F1[3]*(P2[0]*(V3[3]-1j*(V3[4]))+(P2[1]*(-V3[2]+V3[5])+(P2[2]*(+1j*(V3[2])-1j*(V3[5]))+P2[3]*(-V3[3]+1j*(V3[4]))))))
    F2[3]= denom*1j*(F1[2]*(P2[0]*(V3[3]+1j*(V3[4]))+(P2[1]*(-1)*(V3[2]+V3[5])+(P2[2]*(-1)*(+1j*(V3[2]+V3[5]))+P2[3]*(V3[3]+1j*(V3[4])))))+F1[3]*(P2[0]*(V3[2]-V3[5])+(P2[1]*(-V3[3]+1j*(V3[4]))+(P2[2]*(-1)*(+1j*(V3[3])+V3[4])+P2[3]*(V3[2]-V3[5])))))
    F2[4]= denom*-1j * M2*(F1[2]*(-1)*(V3[2]+V3[5])+F1[3]*(-V3[3]+1j*(V3[4])))
    F2[5]= denom*1j * M2*(F1[2]*(V3[3]+1j*(V3[4]))+F1[3]*(V3[2]-V3[5]))
    return F2



def FFV2_3(F1,F2,COUP,M3,W3):
    OM3 = 0.0
    OM3 = where(M3 != 0. , 1.0/M3**2, 0. )
    V3 = wavefunctions.WaveFunction(size=6)
    V3[0] = F1[0]+F2[0]
    V3[1] = F1[1]+F2[1]
    P3 = [-1.0*complex(V3[0]).real, -1.0*complex(V3[1]).real, -1.0*complex(V3[1]).imag, -1.0*complex(V3[0]).imag]
    TMP1 = (F1[2]*(F2[4]*(P3[0]+P3[3])+F2[5]*(P3[1]+1j*(P3[2])))+F1[3]*(F2[4]*(P3[1]-1j*(P3[2]))+F2[5]*(P3[0]-P3[3])))
    denom = COUP/(P3[0]**2-P3[1]**2-P3[2]**2-P3[3]**2 - M3 * (M3 -1j* W3))
    V3[2]= denom*(-1j)*(F1[2]*F2[4]+F1[3]*F2[5]-P3[0]*OM3*TMP1)
    V3[3]= denom*(-1j)*(-F1[2]*F2[5]-F1[3]*F2[4]-P3[1]*OM3*TMP1)
    V3[4]= denom*(-1j)*(-1j*(F1[2]*F2[5])+1j*(F1[3]*F2[4])-P3[2]*OM3*TMP1)
    V3[5]= denom*(-1j)*(-F1[2]*F2[4]-P3[3]*OM3*TMP1+F1[3]*F2[5])
    return V3


def FFV2_4_3(F1,F2,COUP1,COUP2,M3,W3):
    V3 = FFV2_3(F1,F2,COUP1,M3,W3)
    tmp = FFV4_3(F1,F2,COUP2,M3,W3)
    for i in range(2,6):
        V3[i] += tmp[i]
    return V3


def FFV1_0(F1,F2,V3,COUP):
    TMP2 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+1j*(V3[4])))+(F1[3]*(F2[4]*(V3[3]-1j*(V3[4]))+F2[5]*(V3[2]-V3[5]))+(F1[4]*(F2[2]*(V3[2]-V3[5])-F2[3]*(V3[3]+1j*(V3[4])))+F1[5]*(F2[2]*(-V3[3]+1j*(V3[4]))+F2[3]*(V3[2]+V3[5])))))
    vertex = COUP*-1j * TMP2
    return vertex



def FFV1P0_3(F1,F2,COUP,M3,W3):
    V3 = wavefunctions.WaveFunction(size=6)
    V3[0] = F1[0]+F2[0]
    V3[1] = F1[1]+F2[1]
    P3 = [-1.0*complex(V3[0]).real, -1.0*complex(V3[1]).real, -1.0*complex(V3[1]).imag, -1.0*complex(V3[0]).imag]
    denom = COUP/(P3[0]**2-P3[1]**2-P3[2]**2-P3[3]**2 - M3 * (M3 -1j* W3))
    V3[2]= denom*(-1j)*(F1[2]*F2[4]+F1[3]*F2[5]+F1[4]*F2[2]+F1[5]*F2[3])
    V3[3]= denom*(-1j)*(-F1[2]*F2[5]-F1[3]*F2[4]+F1[4]*F2[3]+F1[5]*F2[2])
    V3[4]= denom*(-1j)*(-1j*(F1[2]*F2[5]+F1[5]*F2[2])+1j*(F1[3]*F2[4]+F1[4]*F2[3]))
    V3[5]= denom*(-1j)*(-F1[2]*F2[4]-F1[5]*F2[3]+F1[3]*F2[5]+F1[4]*F2[2])
    return V3



def FFV5_0(F1,F2,V3,COUP):
    TMP0 = (F1[2]*(F2[4]*(V3[2]+V3[5])+F2[5]*(V3[3]+1j*(V3[4])))+F1[3]*(F2[4]*(V3[3]-1j*(V3[4]))+F2[5]*(V3[2]-V3[5])))
    TMP3 = (F1[4]*(F2[2]*(V3[2]-V3[5])-F2[3]*(V3[3]+1j*(V3[4])))+F1[5]*(F2[2]*(-V3[3]+1j*(V3[4]))+F2[3]*(V3[2]+V3[5])))
    vertex = COUP*(-1)*(+1j*(TMP0)+4j*(TMP3))
    return vertex



def FFV4_3(F1,F2,COUP,M3,W3):
    OM3 = 0.0
    OM3 = where(M3 != 0. , 1.0/M3**2, 0. )
    V3 = wavefunctions.WaveFunction(size=6)
    V3[0] = F1[0]+F2[0]
    V3[1] = F1[1]+F2[1]
    P3 = [-1.0*complex(V3[0]).real, -1.0*complex(V3[1]).real, -1.0*complex(V3[1]).imag, -1.0*complex(V3[0]).imag]
    TMP1 = (F1[2]*(F2[4]*(P3[0]+P3[3])+F2[5]*(P3[1]+1j*(P3[2])))+F1[3]*(F2[4]*(P3[1]-1j*(P3[2]))+F2[5]*(P3[0]-P3[3])))
    TMP4 = (F1[4]*(F2[2]*(P3[0]-P3[3])-F2[3]*(P3[1]+1j*(P3[2])))+F1[5]*(F2[2]*(-P3[1]+1j*(P3[2]))+F2[3]*(P3[0]+P3[3])))
    denom = COUP/(P3[0]**2-P3[1]**2-P3[2]**2-P3[3]**2 - M3 * (M3 -1j* W3))
    V3[2]= denom*(-2j)*(OM3*-1/2 * P3[0]*(TMP1+2*(TMP4))+(+1/2*(F1[2]*F2[4]+F1[3]*F2[5])+F1[4]*F2[2]+F1[5]*F2[3]))
    V3[3]= denom*(-2j)*(OM3*-1/2 * P3[1]*(TMP1+2*(TMP4))+(-1/2*(F1[2]*F2[5]+F1[3]*F2[4])+F1[4]*F2[3]+F1[5]*F2[2]))
    V3[4]= denom*2j*(OM3*1/2 * P3[2]*(TMP1+2*(TMP4))+(+1j/2*(F1[2]*F2[5])-1j/2*(F1[3]*F2[4])-1j*(F1[4]*F2[3])+1j*(F1[5]*F2[2])))
    V3[5]= denom*2j*(OM3*1/2 * P3[3]*(TMP1+2*(TMP4))+(+1/2*(F1[2]*F2[4])-1/2*(F1[3]*F2[5])-F1[4]*F2[2]+F1[5]*F2[3]))
    return V3




def calculate(Z, A):
    a1 = 15.75
    a2 = 17.8
    a3 = 0.711
    a4 = 23.7
    a5 = 11.18
    if Z%2 == 0 and (A-Z)%2 == 0:
        delta0 = a5/pow(A,1/2)
    elif Z%2 != 0 and (A-Z)%2 != 0:
        delta0 = -a5/pow(A,1/2)
    else:
        delta0 = 0
    E = a1*A-a2*pow(A,2/3)-a3*Z*(Z-1)/pow(A,1/3)-a4*pow(A-2*Z,2)/A+delta0
    return E/A/10
a = [200, 2  , 20 , 1  , 9  , 000, 000, 1  ]
b = [6  , 50 , 10 , 2  , 20 , 000, 4  , 6  ]
c = [12 , 6  , 300, 5  , 5  , 000, 000, 11 ]
d = [1  , 1  , 3  , 50 , 9  , 000, 000, 18 ]
e = [4  , 2  , 6  , 2  , 300, 7  , 1  , 22 ]
f = [000, 1  , 2  , 1  , 30 , 80 , 2  , 9  ]
g = [1  , 5  , 3  , 2  , 10 , 10 , 24 , 4  ]
h = [1  , 3  , 15 , 5  , 30 , 5  , 2  , 300]

Confusion_Matrix = [
    a, b, c, d, e, f, g, h
]

import copy

def OA(Confusion_Matrix):
    CM = Confusion_Matrix
    xia_list = []
    for i in range(len(CM)):
        xia_list.append(sum(CM[i]))
    xia = sum(xia_list)
    shang_list = []
    for i in range(len(CM)):
        shang_list.append(CM[i][i])
    shang = sum(shang_list)
    OA = shang / xia
    return OA

def KP(Overall_Accuracy, Confusion_Matrix):
    CM = Confusion_Matrix
    p0 = Overall_Accuracy
    pe_xia_list = []
    for i in range(len(CM)):
        pe_xia_list.append(sum(CM[i]))
    pe_xia = sum(pe_xia_list) ** 2
    CMT = list(map(list, zip(*CM))) # Transpose
    pe_shang_list = []
    for i in range(len(CM)):
        pe_shang_list.append(sum(CM[i]) * sum(CMT[i]))
    pe_shang = sum(pe_shang_list)
    pe = pe_shang / pe_xia
    KP = (p0 - pe) / (1 - pe)
    return KP

def CE(Class, Confusion_Matrix):
    CM = Confusion_Matrix
    Class_list = CM[Class]
    Commission_list = copy.deepcopy(Class_list)
    del Commission_list[Class]
    CE = sum(Commission_list) / sum(Class_list)
    return CE

def OE(Class, Confusion_Matrix):
    CM = Confusion_Matrix
    CMT = list(map(list, zip(*CM))) # Transpose
    Class_list = CMT[Class]
    Ommission_list = copy.deepcopy(Class_list)
    del Ommission_list[Class]
    OE = sum(Ommission_list) / sum(Class_list)
    return OE

# Overall accuracy
Overall_Accuracy = OA(Confusion_Matrix)
print(Overall_Accuracy)
# Kappa coefficient
Kappa = KP(Overall_Accuracy, Confusion_Matrix)
print(Kappa)

# For a certain category
# Misclassification error
Commission_Error = CE(0, Confusion_Matrix) # The first parameter is the category, 0 is the first category, 1 is the second category
print(Commission_Error)
# Missing error
Omission_Error = OE(0, Confusion_Matrix) # The first parameter is the category, 0 is the first category, 1 is the second category
print(Omission_Error)
# User Precision
Users_Accuracy = 1 - Commission_Error
print(Users_Accuracy)
# Producer precision/Cartographic precision
Producers_Accuracy = 1 - Omission_Error
print(Producers_Accuracy)

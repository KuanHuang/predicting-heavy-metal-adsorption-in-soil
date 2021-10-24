import numpy as np
import pandas

def readData_cv_ng(K=10):
    #Read excel data
    df = pandas.read_excel('isotherm.xlsx', sheet_name='total')
    #df = pandas.read_excel('adsorption_mg.xlsx', sheet_name='total')
    all_data = np.zeros((2198, 9))
    all_data_Cu = np.zeros((498, 10))
    all_data_Cd = np.zeros((518, 10))
    all_data_Cr = np.zeros((104, 10))
    all_data_Ni = np.zeros((248, 10))
    all_data_Pb = np.zeros((430, 10))
    all_data_Zn = np.zeros((400, 10))
    #load PH
    values = df['pH(H2O)'].values
    all_data_Cu[:, 0] = values[0:498]
    all_data_Cd[:, 0] = values[498:1016]
    all_data_Cr[:, 0] = values[1016:1120]
    all_data_Ni[:, 0] = values[1120:1368]
    all_data_Pb[:, 0] = values[1368:1798]
    all_data_Zn[:, 0] = values[1798:2198]
    #Load OC%
    values = df['OC%'].values
    all_data_Cu[:, 1] = values[0:498]
    all_data_Cd[:, 1] = values[498:1016]
    all_data_Cr[:, 1] = values[1016:1120]
    all_data_Ni[:, 1] = values[1120:1368]
    all_data_Pb[:, 1] = values[1368:1798]
    all_data_Zn[:, 1] = values[1798:2198]

    #Load CEC
    values = df['CEC(cmol/kg)'].values
    all_data_Cu[:, 2] = values[0:498]
    all_data_Cd[:, 2] = values[498:1016]
    all_data_Cr[:, 2] = values[1016:1120]
    all_data_Ni[:, 2] = values[1120:1368]
    all_data_Pb[:, 2] = values[1368:1798]
    all_data_Zn[:, 2] = values[1798:2198]

    #Load clay
    values = df['clay(%)'].values
    all_data_Cu[:, 3] = values[0:498]
    all_data_Cd[:, 3] = values[498:1016]
    all_data_Cr[:, 3] = values[1016:1120]
    all_data_Ni[:, 3] = values[1120:1368]
    all_data_Pb[:, 3] = values[1368:1798]
    all_data_Zn[:, 3] = values[1798:2198]

    #Load equilibrium concentration(mg/L)
    values = df['equilibrium concentration(mg/L)'].values
    all_data_Cu[:, 4] = values[0:498]
    all_data_Cd[:, 4] = values[498:1016]
    all_data_Cr[:, 4] = values[1016:1120]
    all_data_Ni[:, 4] = values[1120:1368]
    all_data_Pb[:, 4] = values[1368:1798]
    all_data_Zn[:, 4] = values[1798:2198]

    #Load electronegativity
    #values = df['electronegativity'].values
    #all_data[:, 5] = values[0:None]
    #Load 1ionization energy(KJ/mol)
    values = df['1ionization energy(KJ/mol)'].values
    all_data_Cu[:, 5] = values[0:498]
    all_data_Cd[:, 5] = values[498:1016]
    all_data_Cr[:, 5] = values[1016:1120]
    all_data_Ni[:, 5] = values[1120:1368]
    all_data_Pb[:, 5] = values[1368:1798]
    all_data_Zn[:, 5] = values[1798:2198]

    #Load 2ionization energy(KJ/mol)
    #values = df['2ionization energy(KJ/mol)'].values
    #all_data[:, 7] = values[0:None]
    #Load ionic radius(A)
    values = df['ionic radius(A)'].values
    all_data_Cu[:, 6] = values[0:498]
    all_data_Cd[:, 6] = values[498:1016]
    all_data_Cr[:, 6] = values[1016:1120]
    all_data_Ni[:, 6] = values[1120:1368]
    all_data_Pb[:, 6] = values[1368:1798]
    all_data_Zn[:, 6] = values[1798:2198]

    #Load hydrated ionic radius(A)
    values = df['hydrated ionic radius(A)'].values
    all_data_Cu[:, 7] = values[0:498]
    all_data_Cd[:, 7] = values[498:1016]
    all_data_Cr[:, 7] = values[1016:1120]
    all_data_Ni[:, 7] = values[1120:1368]
    all_data_Pb[:, 7] = values[1368:1798]
    all_data_Zn[:, 7] = values[1798:2198]

    #Load adsorption(mg/g)
    values = df['adsorption(mg/g)'].values
    all_data_Cu[:, 8] = values[0:498]
    all_data_Cd[:, 8] = values[498:1016]
    all_data_Cr[:, 8] = values[1016:1120]
    all_data_Ni[:, 8] = values[1120:1368]
    all_data_Pb[:, 8] = values[1368:1798]
    all_data_Zn[:, 8] = values[1798:2198]


    #Cross validation split (K-fold)
    #k-1 fold, last fold as test set
    numOfElements = np.zeros((6, ))
    numOfElements[0] = np.shape(all_data_Cu)[0]
    numOfElements[1] = np.shape(all_data_Cd)[0]
    numOfElements[2] = np.shape(all_data_Cr)[0]
    numOfElements[3] = np.shape(all_data_Ni)[0]
    numOfElements[4] = np.shape(all_data_Pb)[0]
    numOfElements[5] = np.shape(all_data_Zn)[0]

    ratio = 1.0/K
    indice = []
    for i in range(6):
        index = np.arange(numOfElements[i])
        np.random.shuffle(index)
        indice.append(index.astype(int))
    #Split Cu
    Cu_data = [np.zeros((0, 9)) for _ in range(K)]
    data_len_normal = int(numOfElements[0] * ratio)
    data_len_last = int(numOfElements[0] - data_len_normal * (K - 1))
    for i in range(K):
        if(i != K-1):
            data_len = data_len_normal
        else:
            data_len = data_len_last
        for j in range(data_len):
            j += i * data_len_normal
            Cu_data[i] = np.append(Cu_data[i], [all_data_Cu[indice[0][j], 0:9]], axis=0)
    cnt = 0
    for i in range(K):
        cnt += np.shape(Cu_data[i])[0]
    print(cnt, np.shape(all_data_Cu))
    #Split Cd
    Cd_data = [np.zeros((0, 9)) for _ in range(K)]
    data_len_normal = int(numOfElements[1] * ratio)
    data_len_last = int(numOfElements[1] - data_len_normal * (K - 1))
    for i in range(K):
        if (i != K - 1):
            data_len = data_len_normal
        else:
            data_len = data_len_last
        for j in range(data_len):
            j += i * data_len_normal
            Cd_data[i] = np.append(Cd_data[i], [all_data_Cd[indice[1][j], 0:9]], axis=0)
    cnt = 0
    for i in range(K):
        cnt += np.shape(Cd_data[i])[0]
    print(cnt, np.shape(all_data_Cd))

    #Split Cr
    groupID = np.unique(all_data_Cr[:, 9])
    Cr_data = [np.zeros((0, 9)) for _ in range(K)]
    data_len_normal = int(numOfElements[2] * ratio)
    data_len_last = int(numOfElements[2] - data_len_normal * (K - 1))
    for i in range(K):
        if (i != K - 1):
            data_len = data_len_normal
        else:
            data_len = data_len_last
        for j in range(data_len):
            j += i * data_len_normal
            Cr_data[i] = np.append(Cr_data[i], [all_data_Cr[indice[2][j], 0:9]], axis=0)

    cnt = 0
    for i in range(K):
        cnt += np.shape(Cr_data[i])[0]
    print(cnt, np.shape(all_data_Cr))

    #Split Ni
    groupID = np.unique(all_data_Ni[:, 9])
    Ni_data = [np.zeros((0, 9)) for _ in range(K)]
    data_len_normal = int(numOfElements[3] * ratio)
    data_len_last = int(numOfElements[3] - data_len_normal * (K - 1))
    for i in range(K):
        if (i != K - 1):
            data_len = data_len_normal
        else:
            data_len = data_len_last
        for j in range(data_len):
            j += i * data_len_normal
            Ni_data[i] = np.append(Ni_data[i], [all_data_Ni[indice[3][j], 0:9]], axis=0)

    cnt = 0
    for i in range(K):
        cnt += np.shape(Ni_data[i])[0]
    print(cnt, np.shape(all_data_Ni))

    #Split Pb
    groupID = np.unique(all_data_Pb[:, 9])
    Pb_data = [np.zeros((0, 9)) for _ in range(K)]
    data_len_normal = int(numOfElements[4] * ratio)
    data_len_last = int(numOfElements[4] - data_len_normal * (K - 1))
    for i in range(K):
        if (i != K - 1):
            data_len = data_len_normal
        else:
            data_len = data_len_last
        for j in range(data_len):
            j += i * data_len_normal
            Pb_data[i] = np.append(Pb_data[i], [all_data_Pb[indice[4][j], 0:9]], axis=0)
    cnt = 0
    for i in range(K):
        cnt += np.shape(Pb_data[i])[0]
    print(cnt, np.shape(all_data_Pb))

    #Split Zn
    Zn_data = [np.zeros((0, 9)) for _ in range(K)]
    data_len_normal = int(numOfElements[5] * ratio)
    data_len_last = int(numOfElements[5] - data_len_normal * (K - 1))
    for i in range(K):
        if (i != K - 1):
            data_len = data_len_normal
        else:
            data_len = data_len_last
        for j in range(data_len):
            j += i * data_len_normal
            Zn_data[i] = np.append(Zn_data[i], [all_data_Zn[indice[5][j], 0:9]], axis=0)
    cnt = 0
    for i in range(K):
        cnt += np.shape(Zn_data[i])[0]
    print(cnt, np.shape(all_data_Zn))

    X_Cu = []
    X_Cd = []
    X_Cr = []
    X_Ni = []
    X_Pb = []
    X_Zn = []
    y_Cu = []
    y_Cd = []
    y_Cr = []
    y_Ni = []
    y_Pb = []
    y_Zn = []
    for i in range(K):
        X_Cu.append(Cu_data[i][:, 0:8])
        X_Cd.append(Cd_data[i][:, 0:8])
        X_Cr.append(Cr_data[i][:, 0:8])
        X_Ni.append(Ni_data[i][:, 0:8])
        X_Pb.append(Pb_data[i][:, 0:8])
        X_Zn.append(Zn_data[i][:, 0:8])
        y_Cu.append(Cu_data[i][:, 8])
        y_Cd.append(Cd_data[i][:, 8])
        y_Cr.append(Cr_data[i][:, 8])
        y_Ni.append(Ni_data[i][:, 8])
        y_Pb.append(Pb_data[i][:, 8])
        y_Zn.append(Zn_data[i][:, 8])


    #X_folds (K-1 + 1)
    X_folds = []
    for i in range(K):
        X_train1 = np.append(X_Cu[i], X_Cd[i], axis=0)
        X_train2 = np.append(X_Cr[i], X_Ni[i], axis=0)
        X_train3 = np.append(X_Pb[i], X_Zn[i], axis=0)
        X_train4 = np.append(X_train1, X_train2, axis=0)
        X_folds.append(np.append(X_train4, X_train3, axis=0))

    #y_folds
    y_folds = []
    for i in range(K):
        y_train1 = np.append(y_Cu[i], y_Cd[i], axis=0)
        y_train2 = np.append(y_Cr[i], y_Ni[i], axis=0)
        y_train3 = np.append(y_Pb[i], y_Zn[i], axis=0)
        y_train4 = np.append(y_train1, y_train2, axis=0)
        y_folds.append(np.append(y_train4, y_train3, axis=0))

    cnt = 0
    for i in range(K):
        cnt += np.shape(X_folds[i])[0]
    print(cnt)
    return X_folds, y_folds

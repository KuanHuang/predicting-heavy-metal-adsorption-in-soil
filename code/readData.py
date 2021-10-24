import numpy as np
import pandas
def readData():
    #Read excel data
    df = pandas.read_excel('grouped_data.xlsx', sheet_name='total')
    #df = pandas.read_excel('adsorption_mg.xlsx', sheet_name='total')
    all_data = np.zeros((802, 9))
    all_data_Cu = np.zeros((249, 10))
    all_data_Cd = np.zeros((218, 10))
    all_data_Cr = np.zeros((48, 10))
    all_data_Ni = np.zeros((45, 10))
    all_data_Pb = np.zeros((117, 10))
    all_data_Zn = np.zeros((125, 10))
    #load PH
    values = df['pH(H2O)'].values
    all_data_Cu[:, 0] = values[0:249]
    all_data_Cd[:, 0] = values[249:467]
    all_data_Cr[:, 0] = values[467:515]
    all_data_Ni[:, 0] = values[515:560]
    all_data_Pb[:, 0] = values[560:677]
    all_data_Zn[:, 0] = values[677:802]
    #Load OC%
    values = df['OC%'].values
    all_data_Cu[:, 1] = values[0:249]
    all_data_Cd[:, 1] = values[249:467]
    all_data_Cr[:, 1] = values[467:515]
    all_data_Ni[:, 1] = values[515:560]
    all_data_Pb[:, 1] = values[560:677]
    all_data_Zn[:, 1] = values[677:802]
    #Load CEC
    values = df['CEC(cmol/kg)'].values
    all_data_Cu[:, 2] = values[0:249]
    all_data_Cd[:, 2] = values[249:467]
    all_data_Cr[:, 2] = values[467:515]
    all_data_Ni[:, 2] = values[515:560]
    all_data_Pb[:, 2] = values[560:677]
    all_data_Zn[:, 2] = values[677:802]
    #Load clay
    values = df['clay(%)'].values
    all_data_Cu[:, 3] = values[0:249]
    all_data_Cd[:, 3] = values[249:467]
    all_data_Cr[:, 3] = values[467:515]
    all_data_Ni[:, 3] = values[515:560]
    all_data_Pb[:, 3] = values[560:677]
    all_data_Zn[:, 3] = values[677:802]
    #Load equilibrium concentration(mg/L)
    values = df['equilibrium concentration(mg/L)'].values
    all_data_Cu[:, 4] = values[0:249]
    all_data_Cd[:, 4] = values[249:467]
    all_data_Cr[:, 4] = values[467:515]
    all_data_Ni[:, 4] = values[515:560]
    all_data_Pb[:, 4] = values[560:677]
    all_data_Zn[:, 4] = values[677:802]
    #Load electronegativity
    #values = df['electronegativity'].values
    #all_data[:, 5] = values[0:None]
    #Load 1ionization energy(KJ/mol)
    values = df['1ionization energy(KJ/mol)'].values
    all_data_Cu[:, 5] = values[0:249]
    all_data_Cd[:, 5] = values[249:467]
    all_data_Cr[:, 5] = values[467:515]
    all_data_Ni[:, 5] = values[515:560]
    all_data_Pb[:, 5] = values[560:677]
    all_data_Zn[:, 5] = values[677:802]
    #Load 2ionization energy(KJ/mol)
    #values = df['2ionization energy(KJ/mol)'].values
    #all_data[:, 7] = values[0:None]
    #Load ionic radius(A)
    values = df['ionic radius(A)'].values
    all_data_Cu[:, 6] = values[0:249]
    all_data_Cd[:, 6] = values[249:467]
    all_data_Cr[:, 6] = values[467:515]
    all_data_Ni[:, 6] = values[515:560]
    all_data_Pb[:, 6] = values[560:677]
    all_data_Zn[:, 6] = values[677:802]
    #Load hydrated ionic radius(A)
    values = df['hydrated ionic radius(A)'].values
    all_data_Cu[:, 7] = values[0:249]
    all_data_Cd[:, 7] = values[249:467]
    all_data_Cr[:, 7] = values[467:515]
    all_data_Ni[:, 7] = values[515:560]
    all_data_Pb[:, 7] = values[560:677]
    all_data_Zn[:, 7] = values[677:802]
    #Load adsorption(mg/g)
    values = df['adsorption(mg/g)'].values
    all_data_Cu[:, 8] = values[0:249]
    all_data_Cd[:, 8] = values[249:467]
    all_data_Cr[:, 8] = values[467:515]
    all_data_Ni[:, 8] = values[515:560]
    all_data_Pb[:, 8] = values[560:677]
    all_data_Zn[:, 8] = values[677:802]

    #Load group number
    values = df['group number'].values
    all_data_Cu[:, 9] = values[0:249]
    all_data_Cd[:, 9] = values[249:467]
    all_data_Cr[:, 9] = values[467:515]
    all_data_Ni[:, 9] = values[515:560]
    all_data_Pb[:, 9] = values[560:677]
    all_data_Zn[:, 9] = values[677:802]

    #Number of groups
    numOfGroup = np.zeros((6,))
    numOfGroup[0] = np.size(np.unique(all_data_Cu[:, 9]))
    numOfGroup[1] = np.size(np.unique(all_data_Cd[:, 9]))
    numOfGroup[2] = np.size(np.unique(all_data_Cr[:, 9]))
    numOfGroup[3] = np.size(np.unique(all_data_Ni[:, 9]))
    numOfGroup[4] = np.size(np.unique(all_data_Pb[:, 9]))
    numOfGroup[5] = np.size(np.unique(all_data_Zn[:, 9]))

    #Split according to group
    ratio = 0.7
    indice = []
    for i in range(6):
        index = np.arange(numOfGroup[i])
        np.random.shuffle(index)
        indice.append(index)
    #Split Cu
    groupID = np.unique(all_data_Cu[:, 9])
    train_len = int(numOfGroup[0]*ratio)
    valid_len = int((numOfGroup[0] - train_len)/2)
    test_len = int(numOfGroup[0] - train_len - valid_len)
    Cu_train = np.zeros((0, 9))
    for i in range(train_len):
        Cu_train = np.append(Cu_train, all_data_Cu[all_data_Cu[:, 9] == groupID[int(indice[0][i])], 0:9], axis=0)
    Cu_valid = np.zeros((0, 9))
    for i in range(valid_len):
        i = i + train_len
        Cu_valid = np.append(Cu_valid, all_data_Cu[all_data_Cu[:, 9] == groupID[int(indice[0][i])], 0:9], axis=0)
    Cu_test = np.zeros((0, 9))
    for i in range(test_len):
        i = i + train_len + valid_len
        Cu_test = np.append(Cu_test, all_data_Cu[all_data_Cu[:, 9] == groupID[int(indice[0][i])], 0:9], axis=0)

    #Split Cd
    groupID = np.unique(all_data_Cd[:, 9])
    train_len = int(numOfGroup[1]*ratio)
    valid_len = int((numOfGroup[1] - train_len)/2)
    test_len = int(numOfGroup[1] - train_len - valid_len)
    Cd_train = np.zeros((0, 9))
    for i in range(train_len):
        Cd_train = np.append(Cd_train, all_data_Cd[all_data_Cd[:, 9] == groupID[int(indice[1][i])], 0:9], axis=0)
    Cd_valid = np.zeros((0, 9))
    for i in range(valid_len):
        i = i + train_len
        Cd_valid = np.append(Cd_valid, all_data_Cd[all_data_Cd[:, 9] == groupID[int(indice[1][i])], 0:9], axis=0)
    Cd_test = np.zeros((0, 9))
    for i in range(test_len):
        i = i + train_len + valid_len
        Cd_test = np.append(Cd_test, all_data_Cd[all_data_Cd[:, 9] == groupID[int(indice[1][i])], 0:9], axis=0)

    #Split Cr
    groupID = np.unique(all_data_Cr[:, 9])
    train_len = int(numOfGroup[2]*ratio)
    valid_len = int((numOfGroup[2] - train_len)/2)
    test_len = int(numOfGroup[2] - train_len - valid_len)
    Cr_train = np.zeros((0, 9))
    for i in range(train_len):
        Cr_train = np.append(Cr_train, all_data_Cr[all_data_Cr[:, 9] == groupID[int(indice[2][i])], 0:9], axis=0)
    Cr_valid = np.zeros((0, 9))
    for i in range(valid_len):
        i = i + train_len
        Cr_valid = np.append(Cr_valid, all_data_Cr[all_data_Cr[:, 9] == groupID[int(indice[2][i])], 0:9], axis=0)
    Cr_test = np.zeros((0, 9))
    for i in range(test_len):
        i = i + train_len + valid_len
        Cr_test = np.append(Cr_test, all_data_Cr[all_data_Cr[:, 9] == groupID[int(indice[2][i])], 0:9], axis=0)

    #Split Ni
    groupID = np.unique(all_data_Ni[:, 9])
    train_len = int(numOfGroup[3]*ratio)
    valid_len = int((numOfGroup[3] - train_len)/2)
    test_len = int(numOfGroup[3] - train_len - valid_len)
    Ni_train = np.zeros((0, 9))
    for i in range(train_len):
        Ni_train = np.append(Ni_train, all_data_Ni[all_data_Ni[:, 9] == groupID[int(indice[3][i])], 0:9], axis=0)
    Ni_valid = np.zeros((0, 9))
    for i in range(valid_len):
        i = i + train_len
        Ni_valid = np.append(Ni_valid, all_data_Ni[all_data_Ni[:, 9] == groupID[int(indice[3][i])], 0:9], axis=0)
    Ni_test = np.zeros((0, 9))
    for i in range(test_len):
        i = i + train_len + valid_len
        Ni_test = np.append(Ni_test, all_data_Ni[all_data_Ni[:, 9] == groupID[int(indice[3][i])], 0:9], axis=0)

    #Split Pb
    groupID = np.unique(all_data_Pb[:, 9])
    train_len = int(numOfGroup[4]*ratio)
    valid_len = int((numOfGroup[4] - train_len)/2)
    test_len = int(numOfGroup[4] - train_len - valid_len)
    Pb_train = np.zeros((0, 9))
    for i in range(train_len):
        Pb_train = np.append(Pb_train, all_data_Pb[all_data_Pb[:, 9] == groupID[int(indice[4][i])], 0:9], axis=0)
    Pb_valid = np.zeros((0, 9))
    for i in range(valid_len):
        i = i + train_len
        Pb_valid = np.append(Pb_valid, all_data_Pb[all_data_Pb[:, 9] == groupID[int(indice[4][i])], 0:9], axis=0)
    Pb_test = np.zeros((0, 9))
    for i in range(test_len):
        i = i + train_len + valid_len
        Pb_test = np.append(Pb_test, all_data_Pb[all_data_Pb[:, 9] == groupID[int(indice[4][i])], 0:9], axis=0)

    #Split Zn
    groupID = np.unique(all_data_Zn[:, 9])
    train_len = int(numOfGroup[5]*ratio)
    valid_len = int((numOfGroup[5] - train_len)/2)
    test_len = int(numOfGroup[5] - train_len - valid_len)
    Zn_train = np.zeros((0, 9))
    for i in range(train_len):
        Zn_train = np.append(Zn_train, all_data_Zn[all_data_Zn[:, 9] == groupID[int(indice[5][i])], 0:9], axis=0)
    Zn_valid = np.zeros((0, 9))
    for i in range(valid_len):
        i = i + train_len
        Zn_valid = np.append(Zn_valid, all_data_Zn[all_data_Zn[:, 9] == groupID[int(indice[5][i])], 0:9], axis=0)
    Zn_test = np.zeros((0, 9))
    for i in range(test_len):
        i = i + train_len + valid_len
        Zn_test = np.append(Zn_test, all_data_Zn[all_data_Zn[:, 9] == groupID[int(indice[5][i])], 0:9], axis=0)

    X_train_Cu = Cu_train[:, 0:8]
    X_valid_Cu = Cu_valid[:, 0:8]
    X_test_Cu = Cu_test[:, 0:8]
    X_train_Cd = Cd_train[:, 0:8]
    X_valid_Cd = Cd_valid[:, 0:8]
    X_test_Cd = Cd_test[:, 0:8]
    X_train_Cr = Cr_train[:, 0:8]
    X_valid_Cr = Cr_valid[:, 0:8]
    X_test_Cr = Cr_test[:, 0:8]
    X_train_Ni = Ni_train[:, 0:8]
    X_valid_Ni = Ni_valid[:, 0:8]
    X_test_Ni = Ni_test[:, 0:8]
    X_train_Pb = Pb_train[:, 0:8]
    X_valid_Pb = Pb_valid[:, 0:8]
    X_test_Pb = Pb_test[:, 0:8]
    X_train_Zn = Zn_train[:, 0:8]
    X_valid_Zn = Zn_valid[:, 0:8]
    X_test_Zn = Zn_test[:, 0:8]

    y_train_Cu = Cu_train[:, 8]
    y_valid_Cu = Cu_valid[:, 8]
    y_test_Cu = Cu_test[:, 8]
    y_train_Cd = Cd_train[:, 8]
    y_valid_Cd = Cd_valid[:, 8]
    y_test_Cd = Cd_test[:, 8]
    y_train_Cr = Cr_train[:, 8]
    y_valid_Cr = Cr_valid[:, 8]
    y_test_Cr = Cr_test[:, 8]
    y_train_Ni = Ni_train[:, 8]
    y_valid_Ni = Ni_valid[:, 8]
    y_test_Ni = Ni_test[:, 8]
    y_train_Pb = Pb_train[:, 8]
    y_valid_Pb = Pb_valid[:, 8]
    y_test_Pb = Pb_test[:, 8]
    y_train_Zn = Zn_train[:, 8]
    y_valid_Zn = Zn_valid[:, 8]
    y_test_Zn = Zn_test[:, 8]
    a = np.array([np.size(y_train_Cu), np.size(y_valid_Cu), np.size(y_test_Cu)])
    print(a/np.sum(a))
    a = np.array([np.size(y_train_Cd), np.size(y_valid_Cd), np.size(y_test_Cd)])
    print(a/np.sum(a))
    a = np.array([np.size(y_train_Cr), np.size(y_valid_Cr), np.size(y_test_Cr)])
    print(a/np.sum(a))
    a = np.array([np.size(y_train_Ni), np.size(y_valid_Ni), np.size(y_test_Ni)])
    print(a/np.sum(a))
    a = np.array([np.size(y_train_Pb), np.size(y_valid_Pb), np.size(y_test_Pb)])
    print(a/np.sum(a))
    a = np.array([np.size(y_train_Zn), np.size(y_valid_Zn), np.size(y_test_Zn)])
    print(a/np.sum(a))


    '''
    #Split train, test(8:1:1)#0.1/0.9=0.111
    X_train_Cu, X_valid_Cu, y_train_Cu, y_valid_Cu = train_test_split(all_data_Cu[:, 0:8], all_data_Cu[:, 8], test_size=0.1, random_state=42)
    X_train_Cu, X_test_Cu, y_train_Cu, y_test_Cu = train_test_split(X_train_Cu, y_train_Cu, test_size=0.111, random_state=42)
    X_train_Cd, X_valid_Cd, y_train_Cd, y_valid_Cd = train_test_split(all_data_Cd[:, 0:8], all_data_Cd[:, 8], test_size=0.1, random_state=42)
    X_train_Cd, X_test_Cd, y_train_Cd, y_test_Cd = train_test_split(X_train_Cd, y_train_Cd, test_size=0.111, random_state=42)
    X_train_Cr, X_valid_Cr, y_train_Cr, y_valid_Cr = train_test_split(all_data_Cr[:, 0:8], all_data_Cr[:, 8], test_size=0.1, random_state=42)
    X_train_Cr, X_test_Cr, y_train_Cr, y_test_Cr = train_test_split(X_train_Cr, y_train_Cr, test_size=0.111, random_state=42)
    X_train_Ni, X_valid_Ni, y_train_Ni, y_valid_Ni = train_test_split(all_data_Ni[:, 0:8], all_data_Ni[:, 8], test_size=0.1, random_state=42)
    X_train_Ni, X_test_Ni, y_train_Ni, y_test_Ni = train_test_split(X_train_Ni, y_train_Ni, test_size=0.111, random_state=42)
    X_train_Pb, X_valid_Pb, y_train_Pb, y_valid_Pb = train_test_split(all_data_Pb[:, 0:8], all_data_Pb[:, 8], test_size=0.1, random_state=42)
    X_train_Pb, X_test_Pb, y_train_Pb, y_test_Pb = train_test_split(X_train_Pb, y_train_Pb, test_size=0.111, random_state=42)
    X_train_Zn, X_valid_Zn, y_train_Zn, y_valid_Zn = train_test_split(all_data_Zn[:, 0:8], all_data_Zn[:, 8], test_size=0.1, random_state=42)
    X_train_Zn, X_test_Zn, y_train_Zn, y_test_Zn = train_test_split(X_train_Zn, y_train_Zn, test_size=0.111, random_state=42)
    '''

    #X_train
    X_train1= np.append(X_train_Cu, X_train_Cd, axis=0)
    X_train2= np.append(X_train_Cr, X_train_Ni, axis=0)
    X_train3= np.append(X_train_Pb, X_train_Zn, axis=0)
    X_train4= np.append(X_train1, X_train2, axis=0)
    X_train= np.append(X_train4, X_train3, axis=0)
    #y_train
    y_train1= np.append(y_train_Cu, y_train_Cd, axis=0)
    y_train2= np.append(y_train_Cr, y_train_Ni, axis=0)
    y_train3= np.append(y_train_Pb, y_train_Zn, axis=0)
    y_train4= np.append(y_train1, y_train2, axis=0)
    y_train= np.append(y_train4, y_train3, axis=0)
    #X_valid
    X_valid1= np.append(X_valid_Cu, X_valid_Cd, axis=0)
    X_valid2= np.append(X_valid_Cr, X_valid_Ni, axis=0)
    X_valid3= np.append(X_valid_Pb, X_valid_Zn, axis=0)
    X_valid4= np.append(X_valid1, X_valid2, axis=0)
    X_valid= np.append(X_valid4, X_valid3, axis=0)
    #y_valid
    y_valid1= np.append(y_valid_Cu, y_valid_Cd, axis=0)
    y_valid2= np.append(y_valid_Cr, y_valid_Ni, axis=0)
    y_valid3= np.append(y_valid_Pb, y_valid_Zn, axis=0)
    y_valid4= np.append(y_valid1, y_valid2, axis=0)
    y_valid= np.append(y_valid4, y_valid3, axis=0)
    #X_test
    X_test1= np.append(X_test_Cu, X_test_Cd, axis=0)
    X_test2= np.append(X_test_Cr, X_test_Ni, axis=0)
    X_test3= np.append(X_test_Pb, X_test_Zn, axis=0)
    X_test4= np.append(X_test1, X_test2, axis=0)
    X_test= np.append(X_test4, X_test3, axis=0)
    #y_test
    y_test1= np.append(y_test_Cu, y_test_Cd, axis=0)
    y_test2= np.append(y_test_Cr, y_test_Ni, axis=0)
    y_test3= np.append(y_test_Pb, y_test_Zn, axis=0)
    y_test4= np.append(y_test1, y_test2, axis=0)
    y_test= np.append(y_test4, y_test3, axis=0)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

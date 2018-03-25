from mylibs import stats as st

def d1(x, y):
    meanX = st.mean(x)
    meanY = st.mean(y)
    somNum = 0.0
    somDen = 0.0
    for i in range(len(x)):
        somNum += (x[i] - meanX) * (y[i] - meanY)
        somDen += (x[i] - meanX) ** 2
    return somNum / somDen

def d0(x, y):
    meanX = st.mean(x)
    meanY = st.mean(y)
    d0 = meanY - d1(x, y) * meanX
    return d0[0]
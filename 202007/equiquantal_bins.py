def eqBins(data,n=20):
    '''
    data - set podatkov
    n - zeljeno stevilo stolpcev
    RETURN:
    bins - array z mejami stolpcev
    '''
    if n > len(data):
        return "število stolpcev mora biti manjše od števila podatkov."
    data0 = data
    data0.sort()
    step = int(len(data)/n)
    bins = [min(data0)]
    i = 0
    while len(data0)>step:
        data0 = data0[step:]
        bins.append(data0[0])
    bins.append(max(data0))
    return bins
        
    
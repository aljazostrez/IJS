from math import pi, e
import numpy as np

## definicija normalne
def norm(x, mu, sigma):
    return (1/((2*pi*sigma**2)**0.5)) * e ** (-(x-mu)**2/(2*sigma**2))

## model gaussovskih mesanic
def GMM(gausses, N=10000, weights=None):
    # gausses: seznam z elementi [mu_i, sigma_i], kjer je mu_i srednja
    #          vrednost i-te normalne porazdelitve in sigma_i standardni
    #          odklon i-te normalne porazdelitve
    # N: stevilo podatkov, ki jih zelimo v nastalem setu
    # weights: obtezitev normalnih utezi. Dolzina mora biti ista kot
    #          dolzina gausses, vsota utezi mora biti enaka 1.
    # RETURN: set podatkov, ki predstavlja model gaussovskih mešanic
    #         glede na gausses, weights  dolzine N.
    if weights == None:
        w = [1/len(gausses) for _ in range(len(gausses))]
    else:
        if (len(weights) != len(gausses)) or (round(sum(weights),5) != 1):
            raise ValueError("weights must be the same length as gausses"
                            " and sum of weights must be equal to 1.")
        w = weights
    data = []
    for i in range(len(gausses)):
        gauss = gausses[i]
        mu = gauss[0]
        sigma = gauss[1]
        var = sigma ** 2
        data.append(np.random.normal(mu, var, round(N*w[i])))
    data = np.concatenate(data)
    np.random.shuffle(data)
    data = list(data)
    return data
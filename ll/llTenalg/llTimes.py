from ..llBase import unfold, fold
import numpy as np



def modeTimes(tensor, matOrVec, mode):
    if matOrVec.ndim == 1:
        newShape = tensor.shape[: mode] + tensor.shape[mode + 1: ]
        return (matOrVec @ llUnfold(tensor, mode)).reshape(newShape)
    else:
        newShape = tensor.shape[: mode] + (matOrVec.shape[0],) + tensor.shape[mode + 1: ]
        return llFold(matOrVec @ llUnfold(tensor, mode), mode, newShape)
    

def khatriRao(matrices, mode=None):
    """
    Khatri-Rao product of a list of matrices
    """
    if mode is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != mode]

    if len(matrices) == 1:
        return matrices[0]

    if matrices[0].ndim == 2:
        nColumns = matrices[0].shape[1]
    else:
        nColumns = 1
        matrices = [np.reshape(m, (-1, 1)) for m in matrices]
        
    nRows = np.prod([m.shape[0] for m in matrices])
    krProduct = np.zeros((nRows, nColumns), dtype=matrices[0].dtype)
    for i in range(nColumns):
        iCol = matrices[0][:, i]
        for mat in matrices[1: ]:
            iCol = np.kron(iCol, mat[:, i])
        krProduct[:, i] = iCol
        
    return krProduct


def unfoldingDotKhatriRao(tensor, weights, factors, mode):
    """
    mode-n unfolding times khatri-rao product of factors
    """ 
    krFactors = khatriRao(factors, mode=mode)
    if weights is None:
        return np.dot(unfold(tensor, mode), krFactors)
    else:
        return np.dot(unfold(tensor, mode), krFactors) * np.reshape(weights, (1, -1))
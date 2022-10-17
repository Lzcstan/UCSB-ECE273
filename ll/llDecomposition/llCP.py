import numpy as np

from ..llBase import unfold, fold
from ..llTenalg import khatriRao, unfoldingDotKhatriRao

def recShape(factors):
    shape = []
    for i, factor in enumerate(factors):
        s = factor.shape
        modeSize = s[0]
        shape.append(modeSize)
    return tuple(shape)


def factorsRank(factors):
    if factors[0].ndim == 2:
        rank = int(factors[0].shape[1])
    elif factors[0].ndim == 1:
        rank = 1
    return rank


def normalization(weights, factors):
    """
    Returns cp_tensor with factors normalised to unit length
    Turns ``factors = [|U_1, ... U_n|]`` into ``[weights; |V_1, ... V_n|]``,
    where the columns of each `V_k` are normalized to unit Euclidean length
    from the columns of `U_k` with the normalizing constants absorbed into
    `weights`.
    """
    rank = factorsRank(factors)

    if weights is None:
        weights = np.ones(rank, dtype=factors[0].dtype)

    normalizedFactors = []
    for i, factor in enumerate(factors):
        # if i == 0:
        #     factor = factor * weights
        #     weights = np.ones(rank, dtype=factor.dtype)

        scales = np.linalg.norm(factor, ord=1, axis=0)
        scalesNonZero = np.where(
            scales == 0, 
            np.ones(np.shape(scales), dtype=factor.dtype), 
            scales
        )
        weights = weights * scales
        normalizedFactors.append(
            factor / np.reshape(scalesNonZero, (1, -1))
        )

    return weights, normalizedFactors


def recTensor(weights, factors):
    """
    Turns the Khatri-product of matrices into a recovered full tensor
    """
    shape = recShape(factors)

    if not shape:  # 0-order tensor
        return np.sum(factors)
    
    if len(shape) == 1:  # just a vector
        return np.sum(weights * factors[0], axis=1)

    if weights is None:
        weights = 1

    fullTensor = np.dot(
        factors[0] * weights, np.transpose(khatriRao(factors, mode=0))
    )

    return fold(fullTensor, 0, shape)


def calcError(tensor, weights, factors):
    """
    Perform the error calculation.
    """
    return np.linalg.norm(tensor - recTensor(weights, factors)) ** 2


def _initializeCP(
    shape,
    rank,
    randomState,
    dtype,
):
    """
    Generates a random (weights, factors)
    """
    factors = [np.array(randomState.random_sample((s, rank)), dtype=dtype) for s in shape]
    weights = np.ones(rank, dtype=dtype)

    return weights, factors

    
def initializeCP(
    tensor,
    rank,
):
    """
    Initialize factors used in `CP`.
    """
    weights, factors = _initializeCP(
        tensor.shape,
        rank,
        np.random.mtrand._rand,
        dtype=tensor.dtype,
    )
    
    return normalization(weights, factors)
    

def CP(
    tensor,
    rank,
    nIterMax=1000,
    tol=1e-6,
):
    """
    CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
    """
    weights, factors = initializeCP(
        tensor,
        rank,
    )

    tensorNorm = np.linalg.norm(tensor)
    modesList = list(range(tensor.ndim))
    recErrors = []
    for iteration in range(nIterMax):
        for mode in modesList:
            # Calculate the pseudo inverse of factors except factors[mode]
            pseudoInverse = np.array(np.ones((rank, rank)), dtype=tensor.dtype)
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudoInverse = pseudoInverse * np.dot(np.conj(np.transpose(factor)), factor)
            pseudoInverse = np.reshape(weights, (-1, 1)) * pseudoInverse * np.reshape(weights, (1, -1))
            unfoldKRProd = unfoldingDotKhatriRao(tensor, weights, factors, mode)
            
            # Get the closed-form solution
            factors[mode] = np.transpose(np.linalg.solve(np.conj(np.transpose(pseudoInverse)), np.transpose(unfoldKRProd)))
            if mode != modesList[-1]:
                weights, factors = normalization(weights, factors)
                
        # Calculate the current unnormalized error if we need it
        unnormalRecError = calcError(tensor, weights, factors)
        recError = unnormalRecError / tensorNorm
        recErrors.append(recError)
         
        if iteration >= 1:
            recErrorDecrease = recErrors[-2] - recErrors[-1]
            # print("iteration {}, reconstruction error: {}, decrease = {}, unnormalized = {}".format(iteration, recError, recErrorDecrease, unnormalRecError))
            
            if abs(recErrorDecrease) < tol:
                print(f"PARAFAC converged after {iteration} iterations")
                break
            elif iteration + 1 == nIterMax:
                print(f"PARAFAC didn't converge after {nIterMax} iterations")
        # else:
        #     print(f"reconstruction error={recErrors[-1]}")
        weights, factors = normalization(weights, factors)
    return weights, factors, recErrors
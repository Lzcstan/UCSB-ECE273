import numpy as np



def _tensorSclideIndex(i, j, mode, baseList):
    sliceList = []
    for base in baseList:
        sliceList.append(j - j // base * base)
        j = j // base
    sliceList.reverse()
    sliceList.insert(mode, i)
    return tuple(sliceList)


def fold(unfoldTensor, mode, shape):
    tensor = np.zeros(shape, dtype=unfoldTensor.dtype)
    baseList = list(shape)
    baseList.pop(mode)
    baseList.reverse()
    for i in range(unfoldTensor.shape[0]):
        for j in range(unfoldTensor.shape[1]):
            tensor[_tensorSclideIndex(i, j, mode, baseList)] = unfoldTensor[i][j]
    return tensor


def unfold(tensor, mode):
    baseList = list(tensor.shape)
    baseList.pop(mode)
    baseList.reverse()
    unfoldTensor = np.zeros(tensor.reshape((tensor.shape[mode], -1)).shape, dtype=tensor.dtype)
    for i in range(unfoldTensor.shape[0]):
        for j in range(unfoldTensor.shape[1]):
            unfoldTensor[i][j] = tensor[_tensorSclideIndex(i, j, mode, baseList)]
    return unfoldTensor
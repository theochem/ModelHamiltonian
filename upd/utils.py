def convert_indices(N, *args):
    """
    Convertin indices from 4d array to 2d numpy array and vice-versa
    :param N: size of corresponding _4d_ matrix: int
    :param args: indices i, j, k, l or p,q: int
    :return: list of converted indices: list
    """
    for elem in args:
        if not isinstance(elem, int):
            raise TypeError("Wrong indices")
        if elem >= N:
            raise TypeError("index is greater than size of the matrix")

    if len(args) == 4:
        i, j, k, l = args
        p = int(i*N+j)
        q = int(k*N+l)
        return [p, q]
    elif len(args) == 2:
        p, q = args
        i, j = p//N, q//N
        j, l = p%N, q%N
        return [i, j, k, l]
    else:
        raise TypeError("Wrong indices")


print(convert_indices(5, 0, 0, 0, 1))

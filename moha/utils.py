def convert_indices(N, *args):
    """
    Convertin indices from 4d array to 2d numpy array and vice-versa
    :param N: size of corresponding _4d_ matrix: int
    :param args: indices i, j, k, l or p,q: int
    :return: list of converted indices: list
    """

    if len(args) == 4:
        # Check if indices are right
        for elem in args:
            if not isinstance(elem, int):
                raise TypeError("Wrong indices")
            if elem >= N:
                raise TypeError("index is greater than size of the matrix")

        # converting indices
        i, j, k, l = args
        p = int(i*N+j)
        q = int(k*N+l)
        return [p, q]

    elif len(args) == 2:
        # check indices
        for elem in args:
            if not isinstance(elem, int):
                raise TypeError("Wrong indices")
            if elem >= N**2:
                raise TypeError("index is greater than size of the matrix")

        # converting indices
        p, q = args
        i, k = p//N, q//N
        j, l = p % N, q % N
        return [i, j, k, l]
    else:
        raise TypeError("Wrong indices")


def get_atom_type(atom):
    """
    Returns atom type and site index
    "C23" -> "C", 23
    :param atom: str
    :return: tuple
    """
    i = 1
    while atom[-i:].isdigit():
        i += 1
    i -= 1
    return atom[:-i], int(atom[-i:])

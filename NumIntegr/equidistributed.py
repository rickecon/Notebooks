'''
------------------------------------------------------------------------
This module generates equidistributed sequences of N points in
d-dimensional as well as identifies the nth point of an equidistributed
sequence in d-dimensional space as described in Judd (1998, chap. 9).
This module is adapted from some code written by Jeremy Bejarano.

This module defines the following functions:
    isPrime()
    primes_ascend()
    equidistr_nth()
    equidistr_seq()
------------------------------------------------------------------------
'''
import numpy as np
import matplotlib.pyplot as plt


def isPrime(n):
    '''
    --------------------------------------------------------------------
    This function returns a boolean indicating whether an integer n is a
    prime number
    --------------------------------------------------------------------
    INPUTS:
    n = scalar, any scalar value

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    i = integer in [2, sqrt(n)]

    FILES CREATED BY THIS FUNCTION: None

    RETURN: boolean
    --------------------------------------------------------------------
    '''
    for i in range(2, int(np.sqrt(n) + 1)):
        if n % i == 0:
            return False

    return True


def primes_ascend(N, min_val=2):
    '''
    --------------------------------------------------------------------
    This function generates an ordered sequence of N consecutive prime
    numbers, the smallest of which is greater than or equal to 1 using
    the Sieve of Eratosthenes algorithm.
    (https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)
    --------------------------------------------------------------------
    INPUTS:
    N       = integer, number of elements in sequence of consecutive
              prime numbers
    min_val = scalar >= 2, the smallest prime number in the consecutive
              sequence must be greater-than-or-equal-to this value

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        isPrime()

    OBJECTS CREATED WITHIN FUNCTION:
    primes_vec     = (N,) vector, consecutive prime numbers greater than
                     min_val
    MinIsEven      = boolean, =True if min_val is even, =False otherwise
    MinIsGrtrThn2  = boolean, =True if min_val is
                     greater-than-or-equal-to 2, =False otherwise
    curr_prime_ind = integer >= 0, running count of prime numbers found

    FILES CREATED BY THIS FUNCTION: None

    RETURN: primes_vec
    --------------------------------------------------------------------
    '''
    primes_vec = np.zeros(N, dtype=int)
    MinIsEven = 1 - min_val % 2
    MinIsGrtrThn2 = min_val > 2
    curr_prime_ind = 0
    if not MinIsGrtrThn2:
        i = 2
        curr_prime_ind += 1
        primes_vec[0] = i
    i = min(3, min_val + (MinIsEven * 1))
    while curr_prime_ind < N:
        if isPrime(i):
            curr_prime_ind += 1
            primes_vec[curr_prime_ind - 1] = i
        i += 2

    return primes_vec


def equidistr_nth(n, d, seq_type='niederreiter'):
    '''
    --------------------------------------------------------------------
    This function returns the nth element of a d-dimensional
    equidistributed sequence. Sequence types available to this function
    are Weyl, Haber, Niederreiter, and Baker. This function follows the
    exposition in Judd (1998, chap. 9).
    --------------------------------------------------------------------
    INPUTS:
    n        = integer >= 1, index of nth value of equidistributed
               sequence where n=1 is the first element
    d        = integer >= 1, number of dimensions in each point of the
               sequence
    seq_type = string, sequence type: "weyl", "haber", "niederreiter",
               or "baker"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        primes_ascend()

    OBJECTS CREATED WITHIN FUNCTION:
    seq_elem  = (d,) vector, coordinates of element of eq'distr seq
    prime_vec = (d,) vector, consecutive prime numbers
    x         = (d,) vector, whole and fractional part of
                equidistributed sequence coordinates
    x_floor   = (d,) vector, whole part of equidistributed sequence
                coordinates
    ar        = (d,) vector, ascending integer values from 1 to d
    error_str = string, error message for unrecognized name error

    FILES CREATED BY THIS FUNCTION: None

    RETURN: seq_elem
    --------------------------------------------------------------------
    '''
    seq_elem = np.empty(d)
    prime_vec = primes_ascend(d)

    if seq_type == 'weyl':
        x = n * (prime_vec ** 0.5)
        x_floor = np.floor(x)
        seq_elem = x - x_floor

    elif seq_type == 'haber':
        x = (n * (n + 1)) / 2 * (prime_vec ** 0.5)
        x_floor = np.floor(x)
        seq_elem = x - x_floor

    elif seq_type == 'niederreiter':
        ar = np.arange(1, d + 1)
        x = n * (2 ** (ar / (d + 1)))
        x_floor = np.floor(x)
        seq_elem = x - x_floor

    elif seq_type == 'baker':
        ar = np.arange(1, d + 1)
        x = n * np.exp(prime_vec)
        x_floor = np.floor(x)
        seq_elem = x - x_floor

    else:
        error_str = ('Equidistributed sequence name in seq_type not ' +
                     'recognized.')
        raise NameError(error_str)

    return seq_elem


def equidistr_seq(N, d, seq_type):
    '''
    --------------------------------------------------------------------
    This function returns a vector of N d-dimensional elements of an
    equidistributed sequence. Sequence types available to this function
    are Weyl, Haber, Niederreiter, and Baker.
    --------------------------------------------------------------------
    INPUTS:
    N        = integer >= 1, number of consecutive elements of
               equidistributed sequence to compute, n=1,2,...N
    d        = integer >= 1, number of dimensions in each point of the
               sequence
    seq_type = string, sequence type: "weyl", "haber", "niederreiter",
               or "baker"

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        equidistr_nth()

    OBJECTS CREATED WITHIN FUNCTION:
    eq_seq = (N, d) matrix, N elements of d-dimensional equidistributed
             sequence
    n_val  = integer in [1, N], index of nth value of equidistributed
             sequence

    FILES CREATED BY THIS FUNCTION: None

    RETURN: eq_seq
    --------------------------------------------------------------------
    '''
    eq_seq = np.zeros((N, d))
    for n_val in range(1, N + 1):
        eq_seq[n_val - 1, :] = equidistr_nth(n_val, d, seq_type)

    return eq_seq


def scatter2d(N):
    '''
    --------------------------------------------------------------------
    Create 4-pane figure of scatter plots for all four equidistributed
    sequence types, each plot with a fixed number of points N
    --------------------------------------------------------------------
    '''
    data_w = equidistr_seq(N, 2, 'weyl')
    data_h = equidistr_seq(N, 2, 'haber')
    data_n = equidistr_seq(N, 2, 'niederreiter')
    data_b = equidistr_seq(N, 2, 'baker')
    x1_w = data_w[:, 0]
    x2_w = data_w[:, 1]
    x1_h = data_h[:, 0]
    x2_h = data_h[:, 1]
    x1_n = data_n[:, 0]
    x2_n = data_n[:, 1]
    x1_b = data_b[:, 0]
    x2_b = data_b[:, 1]
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(x1_w, x2_w, s=1.0, c='blue', label='Weyl')
    axs[0, 0].set_title('Weyl')
    axs[0, 1].scatter(x1_h, x2_h, s=1.0, c='blue', label='Haber')
    axs[0, 1].set_title('Haber')
    axs[1, 0].scatter(x1_n, x2_n, s=1.0, c='blue', label='Niederreiter')
    axs[1, 0].set_title('Niederreiter')
    axs[1, 1].scatter(x1_b, x2_b, s=1.0, c='blue', label='Baker')
    axs[1, 1].set_title('Baker')

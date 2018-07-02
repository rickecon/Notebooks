'''
------------------------------------------------------------------------

------------------------------------------------------------------------
'''
# Import packages
import numpy as np
import numpy.random as rnd
import numpy.linalg as lin
import scipy.stats as sts
import scipy.integrate as intgr
import scipy.optimize as opt
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
cmap1 = matplotlib.cm.get_cmap('summer')

'''
------------------------------------------------------------------------
Read in the data
------------------------------------------------------------------------
claims = (10619,) vector, monthly health claims expenditures (dollars)
------------------------------------------------------------------------
'''
pts = np.loadtxt('Econ381totpts.txt')

'''
------------------------------------------------------------------------
Define functions
------------------------------------------------------------------------
'''

def trunc_norm_pdf(xvals, mu, sigma, cut_lb, cut_ub):
    '''
    --------------------------------------------------------------------
    Generate PDF values from a truncated normal distribution based on a
    normal distribution with mean mu and standard deviation sigma and
    cutoffs (cut_lb, cut_ub).
    --------------------------------------------------------------------
    INPUTS:
    xvals  = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
             value(s) in the support of s~TN(mu, sig, cut_lb, cut_ub)
    mu     = scalar, mean of the nontruncated normal distribution from
             which the truncated normal is derived
    sigma  = scalar > 0, standard deviation of the nontruncated normal
             distribution from which the truncated normal is derived
    cut_lb = scalar or string, ='None' if no lower bound cutoff is
             given, otherwise is scalar lower bound value of
             distribution. Values below this cutoff have zero
             probability
    cut_ub = scalar or string, ='None' if no upper bound cutoff is given
             given, otherwise is scalar lower bound value of
             distribution. Values below this cutoff have zero
             probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()

    OBJECTS CREATED WITHIN FUNCTION:
    cut_ub_cdf = scalar in [0, 1], cdf of N(mu, sigma) at upper bound
                 cutoff of truncated normal distribution
    cut_lb_cdf = scalar in [0, 1], cdf of N(mu, sigma) at lower bound
                 cutoff of truncated normal distribution
    unif2_vals = (N, S) matrix, (N,) vector, or scalar in (0,1),
                 rescaled uniform derived from original.
    pdf_vals   = (N, S) matrix, (N,) vector, or scalar in (0,1), PDF
                 values corresponding to xvals from truncated normal PDF
                 with base normal normal distribution N(mu, sigma) and
                 cutoffs (cut_lb, cut_ub)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: pdf_vals
    --------------------------------------------------------------------
    '''
    # No cutoffs: truncated normal = normal
    if (cut_lb is None) & (cut_ub is None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = 0.0
    # Lower bound truncation, no upper bound truncation
    elif (cut_lb is not None) & (cut_ub is None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    # Upper bound truncation, no lower bound truncation
    elif (cut_lb is None) & (cut_ub is not None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = 0.0
    # Lower bound and upper bound truncation
    elif (cut_lb is not None) & (cut_ub is not None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)

    pdf_vals = (sts.norm.pdf(xvals, loc=mu, scale=sigma) /
                (cut_ub_cdf - cut_lb_cdf))

    return pdf_vals


def trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub):
    '''
    --------------------------------------------------------------------
    Draw (N x S) matrix of random draws from a truncated normal
    distribution based on a normal distribution with mean mu and
    standard deviation sigma and cutoffs (cut_lb, cut_ub). These draws
    correspond to an (N x S) matrix of randomly generated draws from a
    uniform distribution U(0,1).
    --------------------------------------------------------------------
    INPUTS:
    unif_vals = (N, S) matrix, (N,) vector, or scalar in (0,1), random
                draws from uniform U(0,1) distribution
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        scipy.stats.norm()

    OBJECTS CREATED WITHIN FUNCTION:
    cut_ub_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at upper bound
                  cutoff of truncated normal distribution
    cut_lb_cdf  = scalar in [0, 1], cdf of N(mu, sigma) at lower bound
                  cutoff of truncated normal distribution
    unif2_vals  = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  rescaled uniform derived from original.
    tnorm_draws = (N, S) matrix, (N,) vector, or scalar in (0,1),
                  values drawn from truncated normal PDF with base
                  normal distribution N(mu, sigma) and cutoffs
                  (cut_lb, cut_ub)

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: tnorm_draws
    --------------------------------------------------------------------
    '''
    # No cutoffs: truncated normal = normal
    if (cut_lb is None) & (cut_ub is None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = 0.0
    # Lower bound truncation, no upper bound truncation
    elif (cut_lb is not None) & (cut_ub is None):
        cut_ub_cdf = 1.0
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)
    # Upper bound truncation, no lower bound truncation
    elif (cut_lb is None) & (cut_ub is not None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = 0.0
    # Lower bound and upper bound truncation
    elif (cut_lb is not None) & (cut_ub is not None):
        cut_ub_cdf = sts.norm.cdf(cut_ub, loc=mu, scale=sigma)
        cut_lb_cdf = sts.norm.cdf(cut_lb, loc=mu, scale=sigma)

    unif2_vals = unif_vals * (cut_ub_cdf - cut_lb_cdf) + cut_lb_cdf
    tnorm_draws = sts.norm.ppf(unif2_vals, loc=mu, scale=sigma)

    return tnorm_draws


def data_moments(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the two data moments for SMM
    (mean(data), variance(data)) from both the actual data and from the
    simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    mean_data = scalar or (S,) vector, mean value of test scores data
    var_data  = scalar > 0 or (S,) vector, variance of test scores data

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: mean_data, var_data
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        mean_data = xvals.mean()
        var_data = xvals.var()
    elif xvals.ndim == 2:
        mean_data = xvals.mean(axis=0)
        var_data = xvals.var(axis=0)

    return mean_data, var_data


def err_vec(data_vals, sim_vals, mu, sigma, cut_lb, cut_ub, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, test scores data
    sim_vals  = (N, S) matrix, S simulations of test scores data
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments()

    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (2, 1) matrix, column vector of two data moments
    mean_model = scalar, estimated mean value from model
    var_model  = scalar > 0, estimated variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    mean_data, var_data = data_moments(data_vals)
    moms_data = np.array([[mean_data], [var_data]])
    mean_sim, var_sim = data_moments(sim_vals)
    mean_model = mean_sim.mean()
    var_model = var_sim.mean()
    moms_model = np.array([[mean_model], [var_model]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec


def criterion(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the SMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params    = (2,) vector, ([mu, sigma])
    mu        = scalar, mean of the normally distributed random variable
    sigma     = scalar > 0, standard deviation of the normally
                distributed random variable
    args      = length 5 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    xvals, unif_vals, cut_lb, cut_ub, W_hat = args
    sim_vals = trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub)
    err = err_vec(xvals, sim_vals, mu, sigma, cut_lb, cut_ub,
                  simple=False)
    crit_val = np.dot(np.dot(err.T, W_hat), err)

    return crit_val


'''
------------------------------------------------------------------------
1a. asdfasdf
------------------------------------------------------------------------

------------------------------------------------------------------------
'''
N = 161
S = 100
mu_init = 300.0
sig_init = 30.0
params_init = np.array([mu_init, sig_init])
cut_lb = 0.0
cut_ub = 450.0
unif_vals_1 = sts.uniform.rvs(0, 1, size=(N, S))
W_hat_1 = np.eye(2)
# sim_vals_1 = trunc_norm_draws(unif_vals_1, mu_init, sig_init, cut_lb,
#                               cut_ub)
# crit_1 = criterion(params_init, pts, unif_vals_1, cut_lb, cut_ub, W_hat)
# print(crit_1)
smm_args_1 = (pts, unif_vals_1, cut_lb, cut_ub, W_hat_1)
results_1 = opt.minimize(criterion, params_init, args=(smm_args_1),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
mu_SMM1_1, sig_SMM1_1 = results_1.x

# Print results
print('mu_SMM1_1=', mu_SMM1_1, ' sig_SMM1_1=', sig_SMM1_1)
print(results_1)

mean_data, var_data = data_moments(pts)
print('Data mean of scores =', mean_data,
      ', Data variance of scores =', var_data)
sim_vals_1 = trunc_norm_draws(unif_vals_1, mu_SMM1_1, sig_SMM1_1,
                              cut_lb, cut_ub)
mean_sim_1, var_sim_1 = data_moments(sim_vals_1)
mean_model_1 = mean_sim_1.mean()
var_model_1 = var_sim_1.mean()
err_1 = err_vec(pts, sim_vals_1, mu_SMM1_1, sig_SMM1_1, cut_lb, cut_ub,
                False).reshape(2,)

print('Model mean 1 =', mean_model_1, ', Model variance 1 =', var_model_1)
print('Error vector 1 =', err_1)

# Create directory if images directory does not already exist
cur_path = os.path.split(os.path.abspath(__file__))[0]
output_fldr = 'images'
output_dir = os.path.join(cur_path, output_fldr)
if not os.access(output_dir, os.F_OK):
    os.makedirs(output_dir)

# Plot the histogram of the data
count, bins, ignored = plt.hist(pts, 30, normed=True)
plt.title('Econ 381 scores: 2011-2012', fontsize=20)
plt.xlabel('Total points')
plt.ylabel('Percent of scores')
plt.xlim([0, 550])  # This gives the xmin and xmax to be plotted"

# Plot the estimated SMM PDF
dist_pts = np.linspace(0, 450, 500)
plt.plot(dist_pts, trunc_norm_pdf(dist_pts, mu_SMM1_1, sig_SMM1_1,
         cut_lb, cut_ub), linewidth=2, color='k',
         label='1: $\mu_{SMM1}$,$\sigma_{SMM1}$')
plt.legend(loc='upper left')
output_path = os.path.join(output_dir, 'hist_pdf_1.png')
plt.savefig(output_path)
# plt.show()
plt.close()

# Plot criterion function for values of mu and sigma
mu_vals = np.linspace(60, 700, 50)
sig_vals = np.linspace(20, 250, 50)
# mu_vals = np.linspace(600, 610, 50)
# sig_vals = np.linspace(190, 196, 50)
crit_vals = np.zeros((50, 50))
for mu_ind in range(50):
    for sig_ind in range(50):
        crit_vals[mu_ind, sig_ind] = \
            criterion(np.array([mu_vals[mu_ind], sig_vals[sig_ind]]),
                      pts, unif_vals_1, cut_lb, cut_ub, W_hat_1)

mu_mesh, sig_mesh = np.meshgrid(mu_vals, sig_vals)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(sig_mesh, mu_mesh, crit_vals, rstride=8,
                cstride=1, cmap=cmap1)
ax.set_title('Criterion function for values of mu and sigma')
ax.set_xlabel(r'$\sigma$')
ax.set_ylabel(r'$\mu$')
ax.set_zlabel(r'Crit. func.')
output_path = os.path.join(output_dir, 'critvals_1.png')
plt.savefig(output_path)
# plt.show()
plt.close()


# Solve for two step SMM estimator
err1_1 = err_vec(pts, sim_vals_1, mu_SMM1_1, sig_SMM1_1, cut_lb, cut_ub,
                 False)
VCV2_1 = np.dot(err1_1, err1_1.T) / pts.shape[0]
print(VCV2_1)
# Use the pseudo-inverse calculated by SVD because VCV2 is ill-conditioned
W_hat2_1 = lin.pinv(VCV2_1)
print(W_hat2_1)

params_init_2 = np.array([mu_SMM1_1, sig_SMM1_1])
# W_hat3 = np.array([[1. / VCV2[0, 0], 0.], [0., 1. / VCV2[1, 1]]])
smm_args_2 = (pts, unif_vals_1, cut_lb, cut_ub, W_hat2_1)
results_2 = opt.minimize(criterion, params_init_2, args=(smm_args_2),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)))
mu_SMM2_1, sig_SMM2_1 = results_2.x
print('mu_SMM2_1=', mu_SMM2_1, ' sig_SMM2_1=', sig_SMM2_1)


'''
------------------------------------------------------------------------
Change the moments to 4 percentile moments
------------------------------------------------------------------------
'''


def data_moments4(xvals):
    '''
    --------------------------------------------------------------------
    This function computes the four data moments for SMM
    (binpct_1, binpct_2, binpct_3, binpct_4) from both the actual data
    and from the simulated data.
    --------------------------------------------------------------------
    INPUTS:
    xvals = (N, S) matrix, (N,) vector, or scalar in (cut_lb, cut_ub),
            test scores data, either real world or simulated. Real world
            data will come in the form (N,). Simulated data comes in the
            form (N,) or (N, S).

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None

    OBJECTS CREATED WITHIN FUNCTION:
    bpct_1 = scalar in [0, 1] or (S,) vector, percent of observations
             0 <= x < 220
    bpct_2 = scalar in [0, 1] or (S,) vector, percent of observations
             220 <= x < 320
    bpct_3 = scalar in [0, 1] or (S,) vector, percent of observations
             320 <= x < 430
    bpct_4 = scalar in [0, 1] or (S,) vector, percent of observations
             430 <= x <= 450

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: bpct_1, bpct_2, bpct_3, bpct_4
    --------------------------------------------------------------------
    '''
    if xvals.ndim == 1:
        bpct_1 = (xvals < 220).sum() / xvals.shape[0]
        bpct_2 = ((xvals >= 220) & (xvals < 320)).sum() / xvals.shape[0]
        bpct_3 = ((xvals >= 320) & (xvals < 430)).sum() / xvals.shape[0]
        bpct_4 = (xvals >= 430).sum() / xvals.shape[0]
    if xvals.ndim == 2:
        bpct_1 = (xvals < 220).sum(axis=0) / xvals.shape[0]
        bpct_2 = (((xvals >= 220) & (xvals < 320)).sum(axis=0) /
                  xvals.shape[0])
        bpct_3 = (((xvals >= 320) & (xvals < 430)).sum(axis=0) /
                  xvals.shape[0])
        bpct_4 = (xvals >= 430).sum(axis=0) / xvals.shape[0]

    return bpct_1, bpct_2, bpct_3, bpct_4


def err_vec4(data_vals, sim_vals, mu, sigma, cut_lb, cut_ub, simple):
    '''
    --------------------------------------------------------------------
    This function computes the vector of moment errors (in percent
    deviation from the data moment vector) for SMM.
    --------------------------------------------------------------------
    INPUTS:
    data_vals = (N,) vector, test scores data
    sim_vals  = (N, S) matrix, S simulations of test scores data
    mu        = scalar, mean of the nontruncated normal distribution
                from which the truncated normal is derived
    sigma     = scalar > 0, standard deviation of the nontruncated
                normal distribution from which the truncated normal is
                derived
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    simple    = boolean, =True if errors are simple difference, =False
                if errors are percent deviation from data moments

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        data_moments4()

    OBJECTS CREATED WITHIN FUNCTION:
    mean_data  = scalar, mean value of data
    var_data   = scalar > 0, variance of data
    moms_data  = (4, 1) matrix, column vector of two data moments
    mean_model = scalar, mean value from model
    var_model  = scalar > 0, variance from model
    moms_model = (2, 1) matrix, column vector of two model moments
    err_vec    = (2, 1) matrix, column vector of two moment error
                 functions

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: err_vec
    --------------------------------------------------------------------
    '''
    bpct_1_dat, bpct_2_dat, bpct_3_dat, bpct_4_dat = \
        data_moments4(data_vals)
    moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat],
                          [bpct_4_dat]])
    bpct_1_sim, bpct_2_sim, bpct_3_sim, bpct_4_sim = \
        data_moments4(sim_vals)
    bpct_1_mod = bpct_1_sim.mean()
    bpct_2_mod = bpct_2_sim.mean()
    bpct_3_mod = bpct_3_sim.mean()
    bpct_4_mod = bpct_4_sim.mean()
    moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod],
                          [bpct_4_mod]])
    if simple:
        err_vec = moms_model - moms_data
    else:
        err_vec = (moms_model - moms_data) / moms_data

    return err_vec


def criterion4(params, *args):
    '''
    --------------------------------------------------------------------
    This function computes the SMM weighted sum of squared moment errors
    criterion function value given parameter values and an estimate of
    the weighting matrix.
    --------------------------------------------------------------------
    INPUTS:
    params    = (2,) vector, ([mu, sigma])
    mu        = scalar, mean of the normally distributed random variable
    sigma     = scalar > 0, standard deviation of the normally
                distributed random variable
    args      = length 5 tuple,
                (xvals, unif_vals, cut_lb, cut_ub, W_hat)
    xvals     = (N,) vector, values of the truncated normally
                distributed random variable
    unif_vals = (N, S) matrix, matrix of draws from U(0,1) distribution.
                This fixes the seed of the draws for the simulations
    cut_lb    = scalar or string, ='None' if no lower bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    cut_ub    = scalar or string, ='None' if no upper bound cutoff is
                given, otherwise is scalar lower bound value of
                distribution. Values below this cutoff have zero
                probability
    W_hat     = (R, R) matrix, estimate of optimal weighting matrix

    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION:
        norm_pdf()

    OBJECTS CREATED WITHIN FUNCTION:
    err        = (2, 1) matrix, column vector of two moment error
                 functions
    crit_val   = scalar > 0, GMM criterion function value

    FILES CREATED BY THIS FUNCTION: None

    RETURNS: crit_val
    --------------------------------------------------------------------
    '''
    mu, sigma = params
    print(mu, sigma)
    xvals, unif_vals, cut_lb, cut_ub, W_hat = args
    sim_vals = trunc_norm_draws(unif_vals, mu, sigma, cut_lb, cut_ub)
    print(sim_vals.mean(axis=0))
    err = err_vec4(xvals, sim_vals, mu, sigma, cut_lb, cut_ub,
                   simple=False)
    print(err)
    # simple = True
    # bpct_1_dat, bpct_2_dat, bpct_3_dat, bpct_4_dat = \
    #     data_moments4(xvals)
    # moms_data = np.array([[bpct_1_dat], [bpct_2_dat], [bpct_3_dat],
    #                       [bpct_4_dat]])
    # bpct_1_sim, bpct_2_sim, bpct_3_sim, bpct_4_sim = \
    #     data_moments4(sim_vals)
    # bpct_1_mod = bpct_1_sim.mean()
    # bpct_2_mod = bpct_2_sim.mean()
    # bpct_3_mod = bpct_3_sim.mean()
    # bpct_4_mod = bpct_4_sim.mean()
    # moms_model = np.array([[bpct_1_mod], [bpct_2_mod], [bpct_3_mod],
    #                       [bpct_4_mod]])
    # if simple:
    #     err_vec = moms_model - moms_data
    # else:
    #     err_vec = (moms_model - moms_data) / moms_data

    crit_val = np.dot(np.dot(err.T, W_hat), err)

    return crit_val

S = 50
mu_init_3 = 600
sig_init_3 = 200
# sim_vals_3 = trunc_norm_draws(unif_vals_1, mu_init_3, sig_init_3,
#                               cut_lb, cut_ub)
# bpct_1_sim, bpct_2_sim, bpct_3_sim, bpct_4_sim = data_moments4(sim_vals_3)
# # print(bpct_1_sim)
# # print(bpct_2_sim)
# # print(bpct_3_sim)
# # print(bpct_4_sim)
# bpct_1_mod = bpct_1_sim.mean()
# bpct_2_mod = bpct_2_sim.mean()
# bpct_3_mod = bpct_3_sim.mean()
# bpct_4_mod = bpct_4_sim.mean()
# print(bpct_1_mod, bpct_2_mod, bpct_3_mod, bpct_4_mod)
# print(bpct_1_mod + bpct_2_mod + bpct_3_mod + bpct_4_mod)
# err_3 = err_vec4(pts, sim_vals_3, mu_init_3, sig_init_3, cut_lb, cut_ub,
#                  simple=False)
# print(err_3)
# crit_3 = criterion4(np.array([mu_init_3, sig_init_3]),
#                     pts, unif_vals_1, cut_lb, cut_ub, np.eye(4))
# print(crit_3)
params_init_3 = np.array([mu_init_3, sig_init_3])
W_hat1_3 = np.eye(4)
# sim_vals_1 = trunc_norm_draws(unif_vals_1, mu_init, sig_init, cut_lb,
#                               cut_ub)
# crit_1 = criterion(params_init, pts, unif_vals_1, cut_lb, cut_ub, W_hat)
# print(crit_1)
smm_args_3 = (pts, unif_vals_1, cut_lb, cut_ub, W_hat1_3)
results_3 = opt.minimize(criterion4, params_init_3, args=(smm_args_3),
                         method='L-BFGS-B',
                         bounds=((1e-10, None), (1e-10, None)),
                         options={'eps': 1.0})
mu_SMM1_3, sig_SMM1_3 = results_3.x

# Print results
print('mu_SMM1_3=', mu_SMM1_3, ' sig_SMM1_3=', sig_SMM1_3)
print(results_3)

print('Data mean of scores =', mean_data,
      ', Data variance of scores =', var_data)
sim_vals_3 = trunc_norm_draws(unif_vals_1, mu_SMM1_3, sig_SMM1_3,
                              cut_lb, cut_ub)
mean_sim_3, var_sim_3 = data_moments(sim_vals_3)
mean_model_3 = mean_sim_3.mean()
var_model_3 = var_sim_3.mean()
err_3 = err_vec4(pts, sim_vals_3, mu_SMM1_3, sig_SMM1_3, cut_lb, cut_ub,
                 False).reshape(4,)

print('Model mean 3 =', mean_model_3, ', Model variance 3 =', var_model_3)
print('Error vector 3 =', err_3)

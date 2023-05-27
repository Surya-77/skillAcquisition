import yaml
import argparse
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from scipy.stats import gamma

# Define warning level for debug purposes.
from numpy import seterr as numpy_seterr

numpy_seterr(divide='ignore', invalid='ignore')


def hmmPower(times, params, stay, n_iteration=100, tolerance=0.01, debug=False, max_iter=100000, optimizer='BFGS'):
    """
    Identify the best fitting parameters

    input: -times: Raw latencies (practice opportunities x unique problems)
           -params: initial parameters for power law
           -stay: intial stay values (i.e. probability of staying in a phase)

    output: -lkh: log-likelihood of the latency data
            -params: the parameters estimated for the power function
            -stay: the estimated probability of staying in a phase
            -occupancy: (practice opportunities x unique problems x cognitive
                phase x residence within phase) for example (18,13,2,10) is the
                probability the 18th trial of the 13th problem has been in the
                2nd phase for 10 trials. The hmm states are reflected in the
                nxp cross product of the last two dimentions
    """
    nObs = times.shape
    nState = stay.shape
    occupancy = np.zeros((nObs[0], nObs[1], nState[1], nObs[0]),
                         dtype=np.float64)
    lkh1, stay1, params1, occupancy1 = hmmTime(times, occupancy, params, stay,
                                               debug, max_iter, optimizer)
    count = 1
    while True:
        occupancy1 = np.nan_to_num(occupancy1)
        lkh = lkh1
        if np.nanmax(stay) <= 1:
            stay = stay1
        params = params1
        occupancy = occupancy1
        lkh1, stay1, params1, occupancy1 = hmmTime(times, occupancy, params, None,
                                                   debug, max_iter, optimizer)
        print(f"Count: {count:<5} Likelihood Difference: {round(lkh1 - lkh, 2):>10}")
        if (count >= n_iteration or (lkh1 - lkh) <= tolerance) and np.isfinite(lkh):
            break
        count += 1

    return lkh, stay, params, occupancy


def hmmTime(times, occupancy, params, stay=None, debug=True, max_iter=100000, optimizer='BFGS'):
    """
    Implementation of the forward backwards algorithm to identify where phase
    boundries occur.
    """
    occupancy2 = occupancy.copy()
    occupancy2 = np.nan_to_num(occupancy2)
    nObs = times.shape
    nState = occupancy2.shape
    timeprob = np.zeros((nObs[0], nObs[1], nState[2], nObs[0]),
                        dtype=np.float64)
    if stay is None:
        stay = estimateProbs(occupancy2, debug, max_iter, optimizer)
        params = estimateParams(times, occupancy2, params, debug, max_iter, optimizer)

    for i in range(0, nState[2]):
        temp = np.append(np.array(params[0][i]),
                         np.array(params[0][nState[2]:]))
        timeprob[:, :, i, :] = timeProbs(times, temp)

    logprob = np.log(timeprob)
    forward = (np.ones((nObs[0], nObs[1], nState[2], nObs[0]),
                       dtype=np.float64) * float('-inf'))
    forward[0, :, 0, 0] = logprob[0, :, 0, 0]
    backward = (np.ones((nObs[0], nObs[1], nState[2], nObs[0]),
                        dtype=np.float64) * float('-inf'))
    backward[nObs[0] - 1, :, :, :] = 0

    if type(stay) is int:
        print("Unexpected conversion to Int type, converting to numpy array of shape [1,1]")
        stay = np.array(stay, dtype=np.int).reshape(1, 1)

    for i in range(1, nObs[0]):

        forward[i, :, 0, i] = (forward[i - 1, :, 0, i - 1] + np.log(stay[0, 0]) +
                               logprob[i, :, 0, i])

        for j in range(1, nState[2]):
            forward[i, :, j, 0] = np.log(np.sum(np.exp(forward[i - 1, :, j - 1, :]), axis=1) *
                                         timeprob[i, :, j, 0] *
                                         (1 - stay[0, j - 1]))
            forward[i, :, j, np.arange(1, nObs[0])] = (forward[i - 1, :, j, np.arange(0, nObs[0] - 1)] +
                                                       np.log(stay[0, j]) +
                                                       logprob[i, :, j, np.arange(1, nObs[0])])

    for i in range(nObs[0] - 2, -1, -1):
        backward[i, :, nState[2] - 1, np.arange(0, nObs[0] - 1)] = (
                backward[i + 1, :, nState[2] - 1, np.arange(1, nObs[0])] +
                logprob[i + 1, :, nState[2] - 1, np.arange(1, nObs[0])])
        for j in range(nState[2] - 2, -1, -1):
            a = (backward[i + 1, :, j, np.arange(1, nObs[0])] +
                 np.log(stay[0, j]) +
                 logprob[i + 1, :, j, np.arange(1, nObs[0])])
            b = np.tile((backward[i + 1, :, j + 1, 0] +
                         np.log(1 - stay[0, j]) +
                         logprob[i + 1, :, j + 1, 0]), (nObs[0] - 1, 1))
            backward[i, :, j, np.arange(0, nObs[0] - 1)] = np.log(np.exp(a) +
                                                                  np.exp(b))

    temp = backward + forward
    temp[temp > float('-inf')] = np.maximum([-700], temp[temp > float('-inf')])
    likes = np.exp(temp)
    lkh = np.sum(np.log(np.sum(np.sum(likes[nObs[0] - 1, :, :, :], axis=2),
                               axis=1)))
    occ = np.zeros((nObs[0], nObs[1], nState[2], nObs[0]), dtype=np.float64)
    for stateN in range(0, nState[2]):
        occ[:, :, stateN, :] = (likes[:, :, stateN, :] /
                                (np.tile(np.transpose(np.sum(np.sum(likes,
                                                                    axis=2),
                                                             axis=2)),
                                         (nObs[0], 1, 1))))
    return lkh, stay, params, occ


def timeProbs(times, params):
    """
    Calculate the probability of each response latency for each trial given
    its phase.
    """
    nObs = times.shape
    shape = 3
    preds = powerIntercept(np.arange(1, nObs[0] + 1), params)
    matOnes = np.ones((nObs[0], nObs[1], 1), dtype=np.float64)
    scales = np.kron(matOnes, np.reshape([x / shape for x in preds],
                                         (1, 1, nObs[0])))
    matOnes2 = np.ones((nObs[0], nObs[1], nObs[0]), dtype=np.float64)
    probs = gamma.pdf(np.tile(times[:, :, np.newaxis],
                              (1, 1, nObs[0])),
                      np.kron(matOnes2, shape), 0, scales)
    return probs


def estimateProbs(occupancy3, debug=True, max_iter=100000, optimizer='BFGS'):
    """
    Estimate the  likelihood of being in each phase
    """
    nObs = occupancy3.shape
    occupancy3 = np.nan_to_num(occupancy3)
    if nObs[2] == 1:
        probs = 1
    else:
        occupancy4 = np.sum(np.sum(occupancy3, axis=1), axis=2)
        funCall = lambda x: fitDist(occupancy4, x)
        start = np.tile(.8, (1, nObs[2] - 1))
        options_dict = {
            "disp": debug,
            "maxiter": max_iter,
        }
        probs_fit = minimize(
            fun=funCall,
            x0=start[0, 0:],
            method=optimizer,
            options=options_dict
        ).x
        probs = np.array([np.append(probs_fit, [1])])
    return probs


def fitDist(occupancy, params):
    """
    For each phase, calculate likelihood of being in that phase
    """
    nObs = occupancy.shape
    occupancy = np.nan_to_num(occupancy)
    if np.nanmax(params) < .999 and np.nanmin(params) > .001:
        params = np.append(params, [1])
        probs = np.zeros((nObs[0], nObs[1]))
        probs[:, 0] = np.power(params[0], np.arange(0, nObs[0]))
        for i in range(1, nObs[0]):
            probs[i, np.arange(1, nObs[1])] = (probs[i - 1, np.arange(0, nObs[1] - 1)]
                                               * (1 - params[np.arange(0, nObs[1] - 1)])
                                               + probs[i - 1, np.arange(1, nObs[1])]
                                               * params[1:])
        temp = occupancy * np.log(probs)
        val = -np.sum(temp[probs[:, :] > 0])
    else:
        val = np.inf
    return val


def estimateParams(times, occupancy2, params, debug=True, max_iter=100000, optimizer='BFGS'):
    """
    Estimate or re-estimate the parameters of the power law function.
    """
    nObs = times.shape
    nStates = occupancy2.shape
    occupancy2 = np.nan_to_num(occupancy2)
    rts = np.transpose(np.kron(np.ones((nObs[0], 1), dtype=np.float64),
                               np.reshape(np.transpose(times),
                                          (nObs[0] * nObs[1]))))
    occupancy3 = np.reshape(np.transpose(occupancy2, [0, 1, 3, 2]),
                            ((nObs[0] * nObs[1]), nObs[0], nStates[2]),
                            order="F")
    funCall = lambda x: fitParams(rts, occupancy3, x)
    options_dict = {
        "disp": debug,
        "maxiter": max_iter,
    }
    pfit = minimize(
        fun=funCall,
        x0=params[0, 0:],
        method=optimizer,
        options=options_dict
    ).x
    params = np.expand_dims(pfit, axis=0)
    return params


def fitParams(rts, occupancy5, params2):
    """
    Refit the parameters based on phase occupancy.

    Args:
        rts (array-like): The array of reaction time values.
        occupancy5 (array-like): The array representing phase occupancy.
        params2 (array-like): The array of parameters.

    Returns:
        float: The calculated value after refitting the parameters.
    """
    nObs = rts.shape
    nState = occupancy5.shape
    occupancy5 = np.nan_to_num(occupancy5)
    shape = 3
    val = 0
    test = (params2[np.arange(0, nState[2] - 1)] < params2[np.arange(1, nState[2])])
    test_if_lt = np.all(test)
    test_if_gt_zero = np.min(params2[np.arange(0, nState[2])]) <= 0

    if test_if_lt or test_if_gt_zero:
        val = np.inf
    else:
        for i in range(0, nState[2]):
            preds = powerIntercept(
                np.arange(1, nObs[1] + 1),
                np.concatenate((np.array([params2[i], ]), params2[nState[2]:]))
            )
            scales = np.kron(np.ones((nObs[0], 1)),
                             np.array([x / shape for x in preds]))
            probs = gamma.pdf(rts, shape, 0, scales)
            val = val - np.sum(occupancy5[:, :, i] * np.log(probs))
    if np.isnan(val):
        val = np.inf
    return val


def powerIntercept(trials, params):
    """
    Calculate the time values based on the power law function.

    Args:
        trials (array-like): The array of trial values.
        params (array-like): The array of parameters for the power law function.

    Returns:
        array-like: The calculated time values based on the power law function.
    """
    if params[2] < 0:
        times = np.zeros(len(trials))
    else:
        times = params[2] + params[0] * np.power(trials, params[1])
    return times


# ----------------------------------------

# Default Command Line Arguments --stay 0.5 0.5 0.5 --params 10.0 5.0 1.0 -0.1 0.1 --sampling_interval 1
# --n_iteration 100 --tolerance 0.01 --max_iter 1000000 --optimizer BFGS --filepath data/simulationTest3phase.mat

# Parse Command Line Arguments
parser = argparse.ArgumentParser()

# Add the --config argument to specify the YAML config file
parser.add_argument('--config', type=str, help='Path to YAML config file')

# Parse the command-line arguments
args = parser.parse_args()

# If --config argument is provided, load the data from the YAML file
if args.config:
    # Load the YAML config file
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # Access the configuration values
    stay = config['stay']
    params = config['params']
    filepath = config['filepath']
    sampling_interval = config['sampling_interval']
    n_iteration = config.get('n_iteration', 100)
    tolerance = config.get('tolerance', 0.01)
    debug = config.get('debug', False)
    max_iter = config.get('max_iter', 100000)
    optimizer = config.get('optimizer', 'BFGS')

# If --config argument is not provided, use command-line arguments
else:
    # Add the command-line arguments
    parser.add_argument('--stay', nargs='+', type=float, help='Array for stay')
    parser.add_argument('--params', nargs='+', type=float, help='Array for params')
    parser.add_argument('--filepath', type=str, help='File path')
    parser.add_argument('--sampling_interval', type=int, help='Sampling interval')
    parser.add_argument('--n_iteration', type=int, default=100, help='Number of iterations')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance value')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--max_iter', type=int, default=100000, help='Maximum number of iterations')
    parser.add_argument('--optimizer', type=str, default='BFGS', help='Optimizer name')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the argument values
    stay = args.stay
    params = args.params
    filepath = args.filepath
    sampling_interval = args.sampling_interval
    n_iteration = args.n_iteration
    tolerance = args.tolerance
    debug = args.debug
    max_iter = args.max_iter
    optimizer = args.optimizer

# Check the condition
stay_size = len(stay)
params_size = len(params)
if params_size != stay_size + 2:
    print('Error: "params" array size should be "stay" array size + 2')

dataset_path = filepath
mat_data = loadmat(dataset_path)
times = mat_data.get("simRTs", mat_data.get("RTs"))
# params = mat_data["paramI"]
# stay = mat_data["stayI"]

# Testing Parameters
stay = np.array([stay])
params = np.array([params])
times = times[np.arange(0, times.shape[0], sampling_interval)]

stay_init, params_init = stay, params
lkh = None
occupancy = None

print(f"Optimizer Start:\n")
lkhFIN, stayFIN, paramsFIN, occupancyFIN = hmmPower(times, params, stay,
                                                    n_iteration=n_iteration,
                                                    tolerance=tolerance,
                                                    debug=debug,
                                                    max_iter=max_iter,
                                                    optimizer=optimizer
                                                    )
test_data = {
    "lkhFIN": lkhFIN,
    "stayFIN": stayFIN,
    "paramsFIN": paramsFIN,
    "occupancyFIN": occupancyFIN,
}
bic = -2 * lkhFIN + params.shape[1] * np.log(np.prod(times.shape))
print(f"\n{'BIC:':<10} {np.round(bic, 2):>10}")
print(f"{'Stay:':<15}\n\t{'Initial:':<10}{np.array2string(np.round(stay_init.flatten(), 2))}"
      f"\n\t{'Final:':<10}{np.array2string(np.round(stayFIN.flatten(), 2))}")
print(f"{'Parameters:':<15}\n\t{'Initial:':<10}{np.array2string(np.round(params_init.flatten(), 2))}"
      f"\n\t{'Final:':<10}{np.array2string(np.round(paramsFIN.flatten(), 2))}")

import numpy as np
import tensorflow as tf
import gym
import multiprocessing as mp
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from get_laplace_data import read_laplace_data, read_normalized_laplace_data



# in this programme, we use iteration to be the reward r.
# so eval_threshold would be 0 in this programme any more.
# we shouldn't set the eval_threshold.
# we just have to set a generation time.

file_name = "ESwNN_cuda_128_b2_ratioes.txt"
N_KID = 2                  # half of the training population
N_GENERATION = 60         # training step
LR = .05                    # learning rate
SIGMA = .1                 # mutation strength or step size
N_CORE = mp.cpu_count()-1


b_r1 = -100000
b_r2 = -100000
b_s1 = 0
b_s2 = 0

_, _, normalize_data = read_normalized_laplace_data()

#~!!!!!! if you change the value of n
# you should change the bdim or gdim in line 138,139
n = 128

CONFIG_laplace = dict(n_feature = 13, n_action = 10, continuous_a = [False], ep_max_step = 10000000, eval_threshold = 10000000)





def sign(k_id): return -1. if k_id % 2 == 0 else 1.  # mirrored sampling


# here is some function for input data

def transfrom_input_of_NN( err, r):
    if (err == 0):
            o = 19
    else:
        o = int(-np.ceil(np.log10(err)))

    if ( o <=0):
        o =1
    m = r    
    mn = n / 100
    mm = (m+1) / 10
        
    #input_of_NN = [o, o**2, o**3, o**mn, mn*o, mn, mn**2, mn**3, mn, mm**mn, mm**2, mm**o, mm*o*mn, o**mm]
    input_of_NN = [o, o**2, o**3, o**mn, mn**o, mn**2, mn**3, mn, mm**mn, mm**2, mm**o, mm*o*mn, o**mm]
    return To_normalize_data(input_of_NN, normalize_data)

def To_normalize_data(input_of_NN, normalize_data):
    input_of_NN_normalize = []
    normalize_data = np.array(normalize_data)
    normalize_data = np.log(normalize_data + 1)
    for i, n in zip(input_of_NN, normalize_data):
        input_of_NN_normalize.append((i - n[0]) / n[1])
    return input_of_NN_normalize





class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v


def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p


def get_reward(shapes, params, ep_max_step, seed_and_id=None,):
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += sign(k_id) * SIGMA * np.random.randn(params.size)
    p = params_reshape(shapes, params)
    
    ############## gpu setting ####################

    T = np.zeros((n,n))
    T[:,0] = 10

    T = T.astype(np.float64)
    err = np.zeros_like(T.flatten())

    _err = cuda.mem_alloc(T.nbytes)
    T1 = cuda.mem_alloc(T.nbytes)
    T2 = cuda.mem_alloc(T.nbytes)
    cuda.memcpy_htod(T1,T)
    cuda.memcpy_htod(T2,T)


    mod = SourceModule("""
    __global__ void Laplace(double *T_old, double *T_new, double *err, double ratio, int n)
    {
        // compute the "i" and "j" location of the node point
        // handled by this thread

        int i = blockIdx.x * blockDim.x + threadIdx.x ;
        int j = blockIdx.y * blockDim.y + threadIdx.y ;

        // get the natural index values of node (i,j) and its neighboring nodes
                                    //                         N 
        int P = i + j*n;           // node (i,j)              |
        int N = i + (j+1)*n;       // node (i,j+1)            |
        int S = i + (j-1)*n;       // node (i,j-1)     W ---- P ---- E
        int E = (i+1) + j*n;       // node (i+1,j)            |
        int W = (i-1) + j*n;       // node (i-1,j)            |
                                    //                         S 

        // only update "interior" node points
        if(i>0 && i<n-1 && j>0 && j<n-1) {
            T_new[P] = 0.25*( T_old[E] + T_old[W] + T_old[N] + T_old[S] );
            T_new[P] = ratio*T_new[P] + (1-ratio)*T_old[P];
        }
        __syncthreads();
        err[P] = abs( (T_new[P] - T_old[P]));
    }

      """)

    bdim = (16, 16, 1)
    gdim = (8,8,1)
    func = mod.get_function("Laplace")

    ############## create the first observation ##################### 
    ratio = 1.0
    for i in range(3):
        func(T1, T2, _err, np.float64(ratio), np.int32(n), block=bdim, grid=gdim)
        func(T2, T1, _err, np.float64(ratio), np.int32(n), block=bdim, grid=gdim)
    cuda.memcpy_dtoh(err, _err)
    max_err = max(err)
    s = transfrom_input_of_NN(max_err, r = ratio)
    ep_r = 0

    # set some variable to improve the learning step
    # i am going to store 3 error
    err_1 = max_err
    err_2 = 10

    ############# run by the network ######################### 
    ratioes = []
    for step in range(ep_max_step):
        a = get_action(p, s)
        ratio = float((a+1)) / 10
        ratioes.append(ratio)
        for i in range(1000):
            func(T1, T2, _err, np.float64(ratio), np.int32(n), block=bdim, grid=gdim)
            func(T2, T1, _err, np.float64(ratio), np.int32(n), block=bdim, grid=gdim)
        cuda.memcpy_dtoh(err, _err)
        
        max_err = max(err)
        s = transfrom_input_of_NN(max_err, r = ratio)           

        if (err_1 < max_err and err_2 < err_1):
            ep_r -= 20000
        ep_r -= 1

        if max_err < 10e-16: break
    with open(file_name, "a") as text_file:
        text_file.write(np.array_str(np.array(ratioes)) + "\n\n\n")

    return ep_r, ratioes




def get_action(params, x, layer_size = 3):
    #x = x[np.newaxis, :]
    x = np.atleast_2d(x)
    # this code determine how many layers in this netwrok
    # in this case, there is three layers
    for i in range(layer_size - 1):
        x = np.tanh(x.dot(params[2*i]) + params[2*i +1])
    x = x.dot(params[4]) + params[5]
    return np.argmax(x, axis=1)[0]      # for discrete action


def build_net(layer_size = 3):
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    # this code also determine the layer sizes of this network
    # it means that when we going to change the layer sizes,
    # we have to change function build_net and get_action
    
    s = []
    p = []
    M1 = CONFIG_laplace['n_feature']
    M2 = 30
    for size in range(layer_size):
        print(M1, M2)
        sn, pn = linear(M1, M2)
        s.append(sn)
        p.append(pn)
        M1 = M2
        M2 = M2 - 10 if (not size == layer_size -2) else CONFIG_laplace['n_action']

    p_out = p[0]
    for pi in range(1, len(p)):
        p_out = np.concatenate((p_out, p[pi]))
    return s, p_out


def train(net_shapes, net_params, optimizer, utility, b2_r, b2_s):
    # pass seed instead whole noise matrix to parallel will save your time
    noise_seed = np.random.randint(0, 2 ** 32 - 1, size=N_KID, dtype=np.uint32).repeat(2)    # mirrored sampling

    rs = []
    b_p = None

    for k_id in range(N_KID*2):
        print("getting the %d child's performance!"% k_id)
        reward, _ = get_reward(net_shapes, net_params, CONFIG_laplace['ep_max_step'], [noise_seed[k_id], k_id])
        rs.append(reward)

    # append the last two best reward
    rs.append(b2_r[0])
    rs.append(b2_r[1])

    noise_seed = np.append(noise_seed, b2_s[0])
    noise_seed = np.append(noise_seed, b2_s[1])
    noise_seed = noise_seed.astype(np.uint32)

    rewards = np.array([r for r in rs])
    kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward
    with open(file_name, "a") as text_file:
        text_file.write(np.array_str(np.array(b2_r)) + "\n")
        text_file.write(np.array_str(rewards)+"\n\n")

    cumulative_update = np.zeros_like(net_params)       # initialize update values
    for ui, k_id in enumerate(kids_rank):
        np.random.seed(noise_seed[k_id])                # reconstruct noise using seed

        # save the best 2 
        if(ui == 0):
            b2_r[0] = rewards[k_id]
            b2_s[0] = noise_seed[k_id]
            b_p = net_params + sign(k_id) * SIGMA * np.random.randn(net_params.size)
        if(ui == 1):
            b2_r[1] = rewards[k_id]
            b2_s[1] = noise_seed[k_id]
        # in this method, the way we used to update the params isn't calculating they weight by reward percentage,
        # but argsort the reward firstly, and find the value in utility.

        # in the update funciton, the term np.random.randn(net_params.size) somehow use in function get_reward,
        # so we can see the relationship between update and try.
        cumulative_update += utility[ui] * sign(k_id) * np.random.randn(net_params.size)

    gradients = optimizer.get_gradients(cumulative_update/(2*N_KID*SIGMA))
    return net_params + gradients, rewards, b2_r, b2_s, b_p


def ESwNN_train():
    # utility instead reward for update parameters (rank transformation)
    base = N_KID * 2 +2    # *2 for mirrored sampling
    rank = np.arange(1, base + 1)
    util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
    utility = util_ / util_.sum() - 1 / base

    # training
    net_shapes, net_params = build_net()
    b_p = net_params
    b2_r = np.zeros(2)
    b2_r[0] = -100000
    b2_r[1] = -100000
    b2_s = np.ones(2).astype(np.uint32)
    b2_s[0] = 1
    b2_s[1] = 1


    optimizer = SGD(net_params, LR)
    mar = None      # moving average reward
    two_third_N_GENERATION = int(N_GENERATION*2/3)
    for g in range(two_third_N_GENERATION):
        net_params, kid_rewards, b2_r, b2_s, b_p = train(net_shapes, net_params, optimizer, utility, b2_r, b2_s)
    lower_point = bs_r[0]
    for g in range(two_third_N_GENERATION, N_GENERATION):
        net_params, kid_rewards, b2_r, b2_s, b_p = train(net_shapes, net_params, optimizer, utility, b2_r, b2_s)
    higher_point = bs_r[0]

    line = [higher_point, lower_point] 

    return net_shapes, net_params, b_p, line




if __name__ == "__main__":
    net_shapes, net_params, b_p, line = ESwNN_train()

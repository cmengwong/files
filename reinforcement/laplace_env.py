import numpy 
import sys, time
import math
import numpy as np
import tensorflow as tf
#from plot_try import plot_3d
from dropout_final_load import create_NN
from laplace_get_data_new2 import read_laplace_data, read_normalized_laplace_data

# This function just delete all the stuff that is not necessary.
# We just do rtest and plot all the error of the path every 10000 iters,
# and find out the features from the data which are the transformation of fourier
# of the error of the path. (in solve_all_path)

class Grid:
    
    """A simple grid class that stores the details and solution of the
    computational grid."""
    
    def __init__(self, nx=10, ny=10, xmin=0.0, xmax=1.0,
                 ymin=0.0, ymax=1.0):
        self.xmin, self.xmax, self.ymin, self.ymax = xmin, xmax, ymin, ymax
        self.dx = float(xmax-xmin)/(nx-1)
        self.dy = float(ymax-ymin)/(ny-1)
        self.x = np.linspace(xmin, xmax, nx)
        self.y = np.linspace(ymin, ymax, ny)
        self.u = numpy.zeros((nx, ny), 'd')
        # used to compute the change in solution in some of the methods.
        self.old_u = self.u.copy()
        self.nx = nx
        self.ny = ny
        print("init: nx = %d, ny = %d" % (nx, ny))        

    def reset(self):
        self.u = numpy.zeros((self.nx, self.ny), 'd')
        self.old_u = self.u.copy()

    def setBC(self, l, r, b, t):        
        """Sets the boundary condition given the left, right, bottom
        and top values (or arrays)"""        
        self.u[0, :] = l
        self.u[-1, :] = r
        self.u[:, 0] = b
        self.u[:,-1] = t
        self.old_u = self.u.copy()

    def setBCFunc(self, func):
        """Sets the BC given a function of two variables."""
        xmin, ymin = self.xmin, self.ymin
        xmax, ymax = self.xmax, self.ymax
        x = numpy.arange(xmin, xmax + self.dx*0.5, self.dx)
        y = numpy.arange(ymin, ymax + self.dy*0.5, self.dy)
        self.u[0 ,:] = func(xmin,y)
        self.u[-1,:] = func(xmax,y)
        self.u[:, 0] = func(x,ymin)
        self.u[:,-1] = func(x,ymax)

    def computeError(self):        
        """Computes absolute error using an L2 norm for the solution.
        This requires that self.u and self.old_u must be appropriately
        setup."""        
        v = (self.u - self.old_u).flat
        return numpy.sqrt(numpy.dot(v,v))

    def computeError_all_path(self):
        v = (self.u - self.old_u)
        return v

class LaplaceSolver:
    def __init__(self, stepper='numeric',ratio =0.5, n = 200, NN = None, normailize_data = 0):
        self.grid = Grid(nx = n, ny = n)
        self.n = n
        self.setTimeStepper(stepper)
        self.ratio = ratio
        self.prev_ratio = self.ratio
        self.grid.setBCFunc(self.BC)
        self.prev_error = 100
        self.trained_NN = NN
        self.normailize_data = normailize_data
        self.d_n = 0


    def reset_grid(self):
        self.grid.reset()  
        self.grid.setBCFunc(self.BC)
        self.prev_error = 100
        self.prev_prev_error = 101
        self.d_n =0
        return 10

        # print(self.grid.u)

    def numericTimeStep(self, dt=0.0, all_path = False):
        """Takes a time step using a numeric expressions."""
        g = self.grid
        dx2, dy2 = g.dx**2, g.dy**2
        dnr_inv = 0.5/(dx2 + dy2)
        u = g.u
        g.old_u = u.copy()

        # The actual iteration
        u[1:-1, 1:-1] = ((u[0:-2, 1:-1] + u[2:, 1:-1])*dy2 + 
                         (u[1:-1,0:-2] + u[1:-1, 2:])*dx2)*dnr_inv
        u = self.ratio*u + (1-self.ratio)* g.old_u
        g.u = u
        self.grid = g
        output = g.computeError()
        if all_path:
            return g.computeError_all_path()
        return output

    def setTimeStepper(self, stepper='numeric'):        
        """Sets the time step scheme to be used while solving given a
        string which should be one of ['slow', 'numeric', 'blitz',
        'inline', 'fastinline', 'fortran']."""        

        self.timeStep = self.numericTimeStep

    def solve_output_all(self, n_iter = 1e6, eps = 1.0e-16, an = 1000):
        err = self.timeStep()
        error = []
        error_m =[] 
        count = 1
        while (err > eps) & (count < n_iter) :
            err = self.timeStep()
            if err != 0:
                error.append(err)
            count = count + 1

        error_all_path = self.timeStep(all_path = True)
        # print(error_all_path)
        return count, err, error, error_all_path

    def ratio_change(self, r):
        self.ratio = r

    def solve_step(self, r = 1, n_iter=1000, stepper='numeric', eps = 1.0e-16):
        done = False        
        self.ratio_change(r)
        self.ratio = r
        self.d_n = 0

        _, err, error, _ = self.solve_output_all(n_iter = 1000, eps =eps)
        #print("solve step, error:", err,"prev_error:", self.prev_error, "\t ratio :", r)
        input_of_NN = self.from_error_to_input_of_NN(err, r)

        if err >= self.prev_error and self.prev_error >= self.prev_prev_error and self.ratio == self.prev_ratio:
            done = True
            reward = -5000
            print('diverge!!!  err:', err, 'prev_err:', self.prev_error, 'ratio:', r)
            return input_of_NN, reward, done, err
        if err < eps:
            done = True
            reward = 5000
            print "err ", err, "prev_err:", self.prev_error
            print'reach!!!!'
            return input_of_NN, reward, done, err

        self.prev_prev_error = self.prev_error
        self.prev_error = err
        self.prev_ratio = self.ratio

        reward = self.trained_NN.predict_out(input_of_NN)
        reward = reward.item(0)
        reward -= 11
        if err >= self.prev_error:
            reward = -200
        '''
        print("\n\n")
        print(input_of_NN)
        print()
        print("err :" ,err)
        '''



        return input_of_NN, reward, done, err


    def from_error_to_input_of_NN(self, err, r):
        return self.transfrom_input_of_NN(err, r) 


    def BC(self, x, y):
        return (x**2 - y**2)

    def transfrom_input_of_NN(self, err, r):
        if (err == 0):
            o = 19
        else:
            o = int(-np.ceil(np.log10(err)))

        if ( o <=0):
            o =1
        m = r
        n = self.n    
        mn = n / 100
        mm = (m+1) / 10
        
        #input_of_NN = [o, o**2, o**3, o**mn, mn*o, mn, mn**2, mn**3, mn, mm**mn, mm**2, mm**o, mm*o*mn, o**mm]
        input_of_NN = [o, o**2, o**3, o**mn, mn**o, mn**2, mn**3, mn, mm**mn, mm**2, mm**o, mm*o*mn, o**mm]
        return To_normalize_data(input_of_NN, self.normailize_data)



def find_order_number(error_order):
    order_number = np.zeros(20)
    for e in error_order:
        me = -e
        if me >= 0 and me <= 20:
            order_number[int(me)] += 1
    return order_number




def To_normalize_data(input_of_NN, normalize_data):
    input_of_NN_normalize = []
    normalize_data = np.array(normalize_data)
    normalize_data = np.log(normalize_data + 1)
    for i, n in zip(input_of_NN, normalize_data):
        input_of_NN_normalize.append((i - n[0]) / n[1])
    return input_of_NN_normalize

def create_LaplaceSolver(session, n = 200):
    _, _, normailize_data = read_normalized_laplace_data()
    NN = create_NN(session)
    print("set n = %d  \n\n\n", n)
    s = LaplaceSolver(NN = NN, normailize_data = normailize_data, n = n)
    return s

def main():
    session = tf.InteractiveSession()
    _, _, normailize_data = read_normalized_laplace_data()
    NN = create_NN(session)
    s = LaplaceSolver(NN = NN, normailize_data = normailize_data)
    #plot_3d(s.grid.x, s.grid.y, s.grid.u)

    for i in range(10):
        print(s.solve_step(r = 0.5))






if __name__ == '__main__':
    main()
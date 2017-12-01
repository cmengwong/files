import numpy 
import sys, time
import math
#import matplotlib.pyplot as plt
import numpy as np
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
    def __init__(self, grid, stepper='numeric',ratio =0.5):
        self.grid = grid
        self.setTimeStepper(stepper)
        self.ratio = ratio

    def reset_grid(self):
        self.grid.reset()
        # print(self.grid.u)
        self.grid.setBCFunc(BC)
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


    def solve_output_all(self, n_iter = 1e10, eps = 1.0e-16, an = 1000):
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


def BC(x, y):    
    """Used to set the boundary condition for the grid of points.
    Change this as you feel fit."""    
    return (x**2 - y**2)

def test(nmin=5, nmax=30, dn=5, eps=1.0e-16, n_iter=0, stepper='numeric', ratio = 0.5):
    print("with ratio = %f" % (ratio))
    iters = []
    n_grd = numpy.arange(nmin, nmax, dn)
    times = []
    count = 0
    for i in n_grd:
        g = Grid(nx=i, ny=i)
        g.setBCFunc(BC)
        s = LaplaceSolver(g, stepper, ratio)
        t1 = time.clock()
        iters.append(s.solve(n_iter=n_iter, eps=eps))
        dt = time.clock() - t1
        times.append(dt)
        print("Solution for nx = ny = %d, iters = %d, took %f seconds"%(i,iters[count], dt))
        count += 1
    return (n_grd**2, iters, times)


def find_order_number(error_order):
    order_number = np.zeros(20)
    for e in error_order:
        me = -e
        if me >= 0 and me <= 20:
            order_number[int(me)] += 1
    return order_number

def rtest_and_error(n = 100, rmin=0, rmax=1.0, dr=0.1, eps=1.0e-20, n_iter=1e6, stepper='numeric'):
    iters = []
    r_grd = np.arange(rmin, rmax, dr) + 0.1
    times = []
    error = []
    errord = []
    count = 0
    r = np.random.choice(r_grd)
    g = Grid(nx=n, ny=n)
    g.setBCFunc(BC)
    s = LaplaceSolver(g, stepper, r)
    err = 50
    evr_0 = []
    evr_1 = []
    evr_2 = []
    old_explained_variance_ratio = np.zeros(3)
    total_iter = 0
    err_up_or_down = []
    error_order = []

    error.append(0)
    r = 1.0
    while err > eps and count < n_iter:
        print("with ratio = %f" % (r))
        t1 = time.clock()
        # c, err, errorEach = s.solve2(n_iter=n_iter, eps=eps)
        c, err, errorEach, error_all_path = s.solve_output_all(n_iter = 100, eps = eps)
        # plot_3d(g.x, g.y, error_all_path)
        explained_variance_ratio_, _ = pca_3d_to_1d(error_all_path, n_components = 3)
        print("explanied_variance_ratio :\n", explained_variance_ratio_[:2])
        print("old_explanied_variance_ratio :\n", old_explained_variance_ratio[:2])
        print("new - old:\n",(explained_variance_ratio_[:2] - old_explained_variance_ratio[:2]) / old_explained_variance_ratio[:2] )
        print(np.abs(np.mean( (explained_variance_ratio_[:2] - old_explained_variance_ratio[:2]) / old_explained_variance_ratio[:2] ) ))
        errord.append(errorEach)
        evr_0.append(explained_variance_ratio_[0])
        evr_1.append(explained_variance_ratio_[1])
        evr_2.append(explained_variance_ratio_[2])
        iters.append(c)
        error.append(err)
        dt = time.clock() - t1
        times.append(dt)
        err_up_or_down.append(1 if (error[-2] > error[-1]) else 0)

        print("iters = %d, took %f seconds. "%(iters[count], dt))
        print("Error*1e10 = %.17f \n" % (error[count]*1e10))
        #error_order.append(np.ceil(np.log10(err)))
        error_order.append(np.log10(err))
        '''
        plt.plot(errorEach)
        plt.title("error in each iteration when ratio = %f  (log)" % (r))
        plt.show()
        '''
        old_explained_variance_ratio = explained_variance_ratio_
        total_iter += c
        count += 1
    print("total iteratins : ", total_iter)
    '''
    plt.plot(evr_0, label = 'number 0')
    plt.plot(evr_1, label = 'number 1')
    plt.plot(evr_2, label = 'number 2')
    plt.plot(error[1:], label = 'error')
    plt.plot(err_up_or_down[1:], label = 'error up or down')
    plt.plot(error_order, label = 'error order')
    plt.legend()
    plt.show()
    '''
    #plt.plot(error_order)
    #plt.plot(find_order_number(error_order))
    #plt.title("order number ")
    #plt.show()

    return ( iters, times)


def time_test(nx=500, ny=500, eps=1.0e-16, n_iter=1e6, stepper='numeric', ratio = 0.5):
    print("with ratio = %f" % (ratio))
    g = Grid(nx, ny )
    g.setBCFunc(BC)
    s = LaplaceSolver(g, stepper, ratio)
    t = time.clock()
    count, err, _, _ =s.solve_output_all(eps = eps)
    return time.clock() - t, count


def main():
    #rtest_and_error(n = 200)
    t, count = time_test(nx = 256, ny = 256, ratio = 1)
    print("time use:", t)
    print("\n")
    print("count : ", count)


if __name__ == '__main__':
    main()
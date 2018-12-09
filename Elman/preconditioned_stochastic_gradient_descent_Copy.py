# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:58:57 2017

Python functions for preconditioned SGD

@author: XILIN
"""
import numpy as np
import scipy
import scipy.linalg

_tiny = np.finfo('float32').tiny   # to avoid dividing by zero
_diag_loading = 1e-9   # to avoid numerical difficulty when solving triangular systems


###############################################################################
def update_precond_dense(Q, dx, dg, step=0.01, diag=0):
    """
    update dense preconditioner P = Q^T*Q
    Q: Cholesky factor of preconditioner with positive diagonal entries 
    dx: perturbation of parameter
    dg: perturbation of gradient
    step: normalized step size in [0, 1]
    diag: see the code for details
    """
    dx = np.reshape(dx, [-1, 1])    # reshape to column vector
    dg = np.reshape(dg, [-1, 1])
    max_diag = np.max(np.diag(Q))
    Q = Q + np.diag(np.clip(_diag_loading*max_diag - np.diag(Q), 0.0, max_diag))
    
    a = Q.dot(dg)
    b = scipy.linalg.solve_triangular(Q, dx, trans=1, lower=False)
    
    grad = np.triu(a.dot(a.T) - b.dot(b.T))
    if diag:
        step0 = step/(np.max(np.abs(np.diag(grad))) + _tiny)
    else:
        step0 = step/(np.max(np.abs(grad)) + _tiny)
        
    return Q - step0*grad.dot(Q)


def precond_grad_dense(Q, grad):
    """
    return preconditioned gradient using dense preconditioner
    Q: Cholesky factor of preconditioner
    grad: gradient
    """
    grad_shape = grad.shape
    return np.reshape( Q.T.dot(Q.dot( np.reshape(grad, [-1, 1]) )), grad_shape )



###############################################################################
def update_precond_kron(Ql, Qr, dX, dG, step=0.01, diag=0):
    """
    update Kronecker product preconditioner P = kron_prod(Qr^T*Qr, Ql^T*Ql)
    Ql: (left side) Cholesky factor of preconditioner with positive diagonal entries
    Qr: (right side) Cholesky factor of preconditioner with positive diagonal entries
    dX: perturbation of (matrix) parameter
    dG: perturbation of (matrix) gradient
    step: normalized step size in range [0, 1] 
    diag: see the code for details
    """
    max_diag_l = np.max(np.diag(Ql))
    max_diag_r = np.max(np.diag(Qr))
    
    Ql = Ql + np.diag(np.clip(_diag_loading*max_diag_l - np.diag(Ql), 0.0, max_diag_l))
    Qr = Qr + np.diag(np.clip(_diag_loading*max_diag_r - np.diag(Qr), 0.0, max_diag_r))
    
    rho = np.sqrt(max_diag_l/max_diag_r)
    Ql = Ql/rho
    Qr = rho*Qr
    
    A = Ql.dot( dG.dot( Qr.T ) )
    Bt = scipy.linalg.solve_triangular(Ql, 
                                       np.transpose(scipy.linalg.solve_triangular(Qr, dX.T, trans=1, lower=False)), 
                                       trans=1, lower=False)
    
    grad1 = np.triu(A.dot(A.T) - Bt.dot(Bt.T))
    grad2 = np.triu(A.T.dot(A) - Bt.T.dot(Bt))
    
    if diag:
        step1 = step/(np.max(np.abs(np.diag(grad1))) + _tiny)
        step2 = step/(np.max(np.abs(np.diag(grad2))) + _tiny)
    else:
        step1 = step/(np.max(np.abs(grad1)) + _tiny)
        step2 = step/(np.max(np.abs(grad2)) + _tiny)
        
    return Ql - step1*grad1.dot(Ql), Qr - step2*grad2.dot(Qr)
    

def precond_grad_kron(Ql, Qr, Grad):
    """
    return preconditioned gradient using Kronecker product preconditioner
    Ql: (left side) Cholesky factor of preconditioner
    Qr: (right side) Cholesky factor of preconditioner
    Grad: (matrix) gradient
    """
    return Ql.T.dot( Ql.dot( Grad.dot( Qr.T.dot(Qr) ) ) )
    


"""
Other forms of preconditioners, e.g., banded Q (P is banded too), can be useful as well. 
In many deep learning models, affine map 

    f(x) = W*[x; 1]
    
is the building block with matrix W containing the parameters to be optimized. 
Kronecker product preconditioners are particularly useful for such applications.  
"""





### Testing code 
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    """
    Verification of update_precond_dense()
    Eigenvalues of the preconditioned system should be close to 1 or -1
    """
    dim = 3
    hess0 = np.random.randn(dim, dim)
    hess0 = hess0 + hess0.T     # the true Hessian
    Q = np.eye(dim)
    all_eigs = []
    for _ in range(10000):
        dx = 1e-3*np.random.randn(dim)
        dg = hess0.dot(dx) # dg is noiseless here
        Q = update_precond_dense(Q, dx, dg)
        eigs, _ = np.linalg.eig( Q.T.dot(Q).dot(hess0) )
        eigs.sort()
        all_eigs.append(list(eigs))
        if np.max(np.abs(np.abs(eigs) - 1)) < 0.1:
            break
        
    plt.figure(1)
    plt.plot(np.array(all_eigs))
    plt.xlabel('Number of iterations')
    plt.ylabel('Eigenvalues of preconditioned system')
    plt.title('Dense preconditioner estimation')
    
    
    
    """
    Verification of update_precond_kron()
    Eigenvalues of the preconditioned system should be close to 1 or -1 as the 
    true Hessian is decomposable 
    """
    dim1, dim2 = 2, 3
    dim = dim1*dim2
    hess1 = np.random.randn(dim1, dim1)
    hess2 = np.random.randn(dim2, dim2)
    hess0 = np.kron(hess2 + hess2.T, hess1 + hess1.T)   # the true Hessian
    Ql, Qr = np.eye(dim1), np.eye(dim2)
    all_eigs = []
    for _ in range(10000):
        dx = 1e-3*np.random.randn(dim)
        dg = hess0.dot(dx) # dg is noiseless here
        dX = dx.reshape(dim2, dim1).T   
        dG = dg.reshape(dim2, dim1).T # numpy assumes row-major order, so dG is defined in this way
        Ql, Qr = update_precond_kron(Ql, Qr, dX, dG)
        eigs, _ = np.linalg.eig( np.kron(Qr.T.dot(Qr), Ql.T.dot(Ql)).dot(hess0) )
        eigs.sort()
        all_eigs.append(list(eigs))
        if np.max(np.abs(np.abs(eigs) - 1)) < 0.1:
            break
        
    plt.figure(2)
    plt.plot(np.array(all_eigs))
    plt.xlabel('Number of iterations')
    plt.ylabel('Eigenvalues of preconditioned system')
    plt.title('Kron product preconditioner estimation')



    """
    Verification of update_precond_kron()
    Eigenvalues of the preconditioned system should be well normalized 
    """
    dim1, dim2 = 2, 3
    dim = dim1*dim2
    hess0 = np.random.randn(dim, dim)
    hess0 = hess0 + hess0.T     # this is an arbitrary Hessian
    Ql, Qr = np.eye(dim1), np.eye(dim2)
    all_eigs = []
    for i in range(1000):
        dx = 1e-3*np.random.randn(dim)
        dg = hess0.dot(dx) # dg is noiseless here
        dX = dx.reshape(dim2, dim1).T
        dG = dg.reshape(dim2, dim1).T
        Ql, Qr = update_precond_kron(Ql, Qr, dX, dG)
        eigs, _ = np.linalg.eig( np.kron(Qr.T.dot(Qr), Ql.T.dot(Ql)).dot(hess0) )
        eigs.sort()
        all_eigs.append(list(eigs))
        
#        if i==0 or i==999: # remove this comment to see the improvement on eigenvalue spread reduction
#            eigs=np.log(np.abs(eigs))
#            print(np.std(eigs))
        
    plt.figure(3)
    plt.plot(np.array(all_eigs))
    plt.xlabel('Number of iterations')
    plt.ylabel('Eigenvalues of preconditioned system')
    plt.title('Normalizing eigenvalues using Kronecker product preconditioner')
    
    
    
    """
    direct sum approximation
    Eigenvalues of the preconditioned system should be well normalized 
    """
    dim1, dim2 = 2, 3
    dim = dim1 + dim2
    hess0 = np.random.randn(dim, dim)
    hess0 = hess0 + hess0.T     # this is an arbitrary Hessian
    Q1, Q2 = np.eye(dim1), np.eye(dim2)
    all_eigs = []
    for i in range(1000):
        dx = 1e-3*np.random.randn(dim)
        dg = hess0.dot(dx) # dg is noiseless here
        Q1 = update_precond_dense(Q1, dx[:dim1], dg[:dim1])
        Q2 = update_precond_dense(Q2, dx[dim1:], dg[dim1:])
        eigs, _ = np.linalg.eig( scipy.linalg.block_diag(Q1.T.dot(Q1), Q2.T.dot(Q2)).dot(hess0) )
        eigs.sort()
        all_eigs.append(list(eigs))
        
#        if i==0 or i==999:  # remove this comment to see the improvement on eigenvalue spread reduction
#            eigs=np.log(np.abs(eigs))
#            print(np.std(eigs))
        
    plt.figure(4)
    plt.plot(np.array(all_eigs))
    plt.xlabel('Number of iterations')
    plt.ylabel('Eigenvalues of preconditioned system')
    plt.title('Normalizing eigenvalues using direct sum preconditioner')
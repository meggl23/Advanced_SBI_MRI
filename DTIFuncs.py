import numpy as np

def ComputeDTI(params):
    L = fill_lower_diag(params)
    
    np.fill_diagonal(L, np.abs(np.diagonal(L)))

    A = L @ L.T
    return A

def fill_lower_diag(a):
    b = [a[0],a[3],a[1],a[4],a[5],a[2]]
    n = 3
    mask = np.tri(n,dtype=bool) 
    out = np.zeros((n,n),dtype=float)
    out[mask] = b
    return out

def ForceLowFA(dt):
    # Modify the matrix to ensure low FA (more isotropic)
    eigenvalues, eigenvectors = np.linalg.eigh(dt)
    
    # Make the eigenvalues more similar to enforce low FA
    mean_eigenvalue = np.mean(eigenvalues)

    adjusted_eigenvalues = np.clip(eigenvalues, mean_eigenvalue * np.random.rand(), mean_eigenvalue * 1.0)
    
    # Reconstruct the matrix with the adjusted eigenvalues
    dt_low_fa = eigenvectors @ np.diag(adjusted_eigenvalues) @ eigenvectors.T
    
    return dt_low_fa

def vals_to_mat(dt):
    DTI = np.zeros((3,3))
    DTI[0,0] = dt[0]
    DTI[0,1],DTI[1,0] =  dt[1],dt[1]
    DTI[1,1] =  dt[2]
    DTI[0,2],DTI[2,0] =  dt[3],dt[3]
    DTI[1,2],DTI[2,1] =  dt[4],dt[4]
    DTI[2,2] =  dt[5]
    return DTI

def mat_to_vals(DTI):
    dt = np.zeros(6)
    dt[0] = DTI[0,0]
    dt[1] = DTI[0,1]
    dt[2] = DTI[1,1]
    dt[3] = DTI[0,2]
    dt[4] = DTI[1,2]
    dt[5] = DTI[2,2]
    return dt

def FracAni(evals,MD):
    numerator = np.sqrt(3 * np.sum((evals - MD) ** 2))
    denominator = np.sqrt(2) * np.sqrt(np.sum(evals ** 2))
    
    return numerator / denominator
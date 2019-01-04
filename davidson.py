# Davidson Algorithm
# Based on Crouzeix et al, SIAM J. Sci. Comput. 15-1 (1994), pp. 62-76

import numpy as np
from scipy.linalg import eigh

# Build a sparse Hermitian matrix A
def build_A(N, sparsity):
	A = np.diag(range(1,N+1)).astype(np.float64)
	A = A + sparsity * np.random.rand(N,N)
	A = (A + A.T)/2
	return A

# Main procedure of Davidson algorithm
def davidson(A, l, m, max_iter, tol):
	N = A.shape[0]
	V = np.eye(N,l)
	w_old = 10
	for i in range(max_iter):
		H = reduce(np.dot, [V.T, A, V])
		w, S = np.linalg.eigh(H)
		idx = w.argsort()
		w = w[idx][:l]	# Find the smallest l eigenvalues
		if np.linalg.norm(w-w_old, ord=np.inf) < tol:
			print 'Davidson converged.'
			break
		W = np.diag(w)
		S = S[:,idx][:,:l]
		x = np.dot(V, S)
		r = np.dot(A, x) - np.dot(x, W)
		t = np.zeros((N,l))
		for j in range(l):
			for k in range(N):
				t[k,j] = r[k,j]/(w[j] - A[k,k]) # Here the preconditioner is (wI - D)^{-1}, where D is the diagonal of A
		if V.shape[1] + l < m:
			V = np.hstack((V, t))
		else:
			V = np.hstack((x, t))	# If number of search vectors exceeds m, restart with the Ritz vectors plus the additional search vectors in this step 
		V = np.linalg.qr(V)[0]
		w_old = w
	return w

if __name__ == '__main__':
	import time
	
	N = 1000			# Dimension of matrix
	sparsity = 0.01		# Prefactor of the perturbation matrix
	max_iter = 10000	# Max number of iterations
	tol = 1e-8			# Convergene tolerance
	l = 4				# Number of eigevalues needed
	m = N//2			# Max dimension of search space
	
	A = build_A(N, sparsity)
	t1 = time.time()
	
	w_davidson = davidson(A, l, m, max_iter, tol)
	t2 = time.time()

	w_np = np.linalg.eigh(A)[0]
	idx = w_np.argsort()[:l]
	t3 = time.time()
	
	print 'Davidson eigenvalues', w_davidson, ';', t2-t1, 's'
	print 'Numpy eigenvalues   ', w_np[idx], ';', t3-t2, 's'
	

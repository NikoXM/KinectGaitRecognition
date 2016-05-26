import numpy as np
from sklearn.metrics.pairwise import manhattan_distances

def poly_func(x, a1, a2, a3, a4, a5, a6, a7):
    return a7 * x ** 7 + a6 * x ** 6 + a5 * x ** 5 + a4 * x ** 4 + a3 * x ** 3 + a2 * x ** 2 + a1 * x

def dtw(xs,ys,count=[0]):
	if(len(xs) == 10):
		return 0
	print count[0]
	count[0] += 1
	xs = xs.reshape(-1,7)
	ys = ys.reshape(-1,7)
	dist = 0
	for i in range(len(xs)):
		xdata = np.arange(30)
		ytrainData = poly_func(xdata,xs[i][0],xs[i][1],xs[i][2],xs[i][3],xs[i][4],xs[i][5],xs[i][6])
		ytestData = poly_func(xdata,ys[i][0],ys[i][1],ys[i][2],ys[i][3],ys[i][4],ys[i][5],ys[i][6])
		dist += dtw_single(ytrainData,ytestData)
	#print "calciulateing..."
	return dist

def dtw_single(x, y):
	# print x.shape
	# print y.shape

	dist = manhattan_distances
 	r, c = len(x), len(y)
	D0 = np.zeros((r + 1, c + 1))
	D0[0, 1:] = np.inf
	D0[1:, 0] = np.inf
	D1 = D0[1:, 1:]
	for i in range(r):
		for j in range(c):
			D1[i, j] = dist([x[i]], [y[j]])
	C = D1.copy()
	for i in range(r):
		for j in range(c):
			D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
	if len(x)==1:
		path = np.zeros(len(y)), range(len(y))
	elif len(y) == 1:
		path = range(len(x)), np.zeros(len(x))
	else:
		path = _traceback(D0)
	#return D1[-1, -1] / sum(D1.shape), C, D1, path
	return D1[-1, -1] / sum(D1.shape)

def _traceback(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)
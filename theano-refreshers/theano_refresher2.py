import theano
a = theano.tensor.vector()
b = theano.tensor.vector()

c = a ** 2 + b ** 2 + 2 * a * b

f = theano.function([a,b], c)
print (f([0,1,2], [1,2,3]))
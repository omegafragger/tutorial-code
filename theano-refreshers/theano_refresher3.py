import theano

shared_var = theano.shared(0)
inc = theano.tensor.iscalar('inc')

func = theano.function([inc], shared_var, updates=[(shared_var, shared_var + inc)])
func(0)
print (shared_var.get_value())
func(2)
print (shared_var.get_value())

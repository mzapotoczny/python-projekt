import numpy as np
from copy import deepcopy

def compute_error_rate(net, stream):
    num_errs = 0.0
    num_examples = 0
    for X, Y in stream.get_epoch_iterator():
        O = net.fprop(X)
        num_errs += (O.argmax(0) != Y).sum()
        num_examples += X.shape[1]
    return num_errs/num_examples

def SGD(net, train_stream, validation_stream):
    i=0
    e=0
    
    velocities = [np.zeros_like(P) for P in net.parameters]
    
    best_valid_error_rate = np.inf
    best_params = deepcopy(net.parameters)
    best_params_epoch = 0
    
    train_erros = []
    train_loss = []
    validation_errors = []
    
    number_of_epochs = 3
    patience_expansion = 1.5
    
    try:
        while e<number_of_epochs:
            e += 1
            for X,Y in train_stream.get_epoch_iterator(): 
                i += 1
                L, O, gradients = net.get_cost_and_gradient(X, Y)
                err_rate = (O.argmax(0) != Y).mean()
                train_loss.append((i,L))
                train_erros.append((i,err_rate))
                if i % 100 == 0:
                    print "At minibatch %d, batch loss %f, batch error rate %f%%" % (i, L, err_rate*100)
                for P, V, G, N in zip(net.parameters, velocities, gradients, net.parameter_names):
                    if N=='W':
                        G += -1e-9 * np.sum(P**2)
                    alpha = -1e-1 * (1-np.tanh(i/18000.0))
                    epsilon = 1.0
                    V = epsilon*V + alpha*G
                    P += V

            # After an epoch compute validation error
            val_error_rate = compute_error_rate(net, validation_stream)
            if val_error_rate < best_valid_error_rate:
                number_of_epochs = np.maximum(number_of_epochs, e * patience_expansion+1)
                best_valid_error_rate = val_error_rate
                best_params = deepcopy(net.parameters)
                best_params_epoch = e
                validation_errors.append((i,val_error_rate))
            print "After epoch %d: valid_err_rate: %f%% currently going to do %d epochs" %(
                e, val_error_rate*100.0, number_of_epochs)
            
    except KeyboardInterrupt:
        print "Setting network parameters from after epoch %d" %(best_params_epoch)
        net.parameters = best_params
        
        subplot(2,1,1)
        train_loss = np.array(train_loss)
        semilogy(train_loss[:,0], train_loss[:,1], label='batch train loss')
        legend()
        
        subplot(2,1,2)
        train_erros = np.array(train_erros)
        plot(train_erros[:,0], train_erros[:,1], label='batch train error rate')
        validation_errors = np.array(validation_errors)
        plot(validation_errors[:,0], validation_errors[:,1], label='validation error rate', color='r')
        ylim(0,0.2)
        legend()

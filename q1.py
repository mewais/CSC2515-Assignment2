import numpy as np

alpha = 0.01                # learning rate
delta = 0.5                 # Huber's delta

def huber_error(y, t):
    # Calculate the correct ranges first
    error = y - t
    abs_error = np.abs(error)
    
    # Get index of inputs inside range and outside it
    inside_indices = np.where(abs_error <= delta)
    outside_indices = np.where(abs_error > delta)

    # Inside range is ((y-t)^2)/2
    # we already have y-t
    huber_error = error
    huber_error[inside_indices] = np.square(huber_error[inside_indices])
    huber_error[inside_indices] = huber_error[inside_indices]/2

    # Outside range is d(abs(y-t)-d/2)
    # we already have abs(y-t)
    huber_error[outside_indices] = abs_error[outside_indices]-(delta/2)
    huber_error[outside_indices] = abs_error[outside_indices]*delta

    # We're done
    return huber_error

def mean_huber_error(y, t):
    huber_loss = huber_error(y, t)
    return np.mean(huber_loss)

def huber_gradient(y, t):
    # Calculate the correct ranges first
    error = y - t

    # Get index of inputs inside range and outside it
    inside_indices = np.where(np.abs(error) <= delta)
    after_indices = np.where(error > delta)
    before_indices = np.where(error < -delta)

    # Inside range is x(y-t)
    # we have (y-t), leave it for now
    huber_gradient = error

    # Outside positive range is dx
    # set it to d for now
    huber_gradient[after_indices] = delta
    
    # Inside positive range is dx
    # set it to -d for now
    huber_gradient[before_indices] = -delta

    return huber_gradient

def huber_weight_gradient_descent(y, t, w, x):
    huber_weight_gradient = huber_gradient(y, t)
    
    # This does the multiplication and summation for us
    huber_weight_gradient = np.matmul(huber_weight_gradient, x)
    
    # Then we use the learning rate
    update = huber_weight_gradient*alpha/x.shape[0]

    # And finally update
    w = w - update

    return w

def huber_bias_gradient_descent(y, t, b):

    huber_bias_gradient = huber_gradient(y, t)

    # Sum, apply learning rate
    update = np.sum(huber_bias_gradient)*alpha/huber_bias_gradient.shape[0]

    # Update
    b = b - update

    return b

def prediction(w, x, b):
    y = np.matmul(x, w) + b
    return y

def init_params(num_variables):
    return np.zeros(num_variables), 0

def train(t, x, iterations=200000000):
    num_points, num_variables = x.shape
    # Initialize the weights and bias with zeros
    w, b = init_params(num_variables)

    # start gradient descent
    for i in range(iterations):
        # first calculate y based on w and b
        y = prediction(w, x, b)
      
        loss = mean_huber_error(y, t)
        print('Iteration: ' + str(i), end=', ')
        print('Loss: ' + str(round(loss,2)), end='\r')
        if loss < 0.3:
            return w, b
        
        # Then compute error and update weights
        w = huber_weight_gradient_descent(y, t, w, x)
        b = huber_bias_gradient_descent(y, t, b)
        
    return w, b

def init_test_input(num_variables, num_points, coefficients, bias, noise_std = 0.75):
    if len(coefficients.shape) != 1 or num_variables != coefficients.shape[0]:
        raise ValueError('Length of coefficients must be the same as num_variables')
    
    # Define X
    x = np.random.uniform(low=-10, high=10, size=(num_points, num_variables))

    # define Y
    y = np.dot(x, coefficients) + bias

    # Add a little noise
    y = y + np.random.normal(0, noise_std, num_points)

    return y, x

if __name__ == '__main__':
    print('Single or multivariate linear regression with gradient descent')

    # single variable
    # y = 4x + 10
    y, x = init_test_input(1, 2000, np.array([4]), 10)
    # train
    w, b = train(y, x)
    # Calculate loss
    trained_y = np.matmul(x, w) + b
    print('\nUnivariate loss after training')
    print(mean_huber_error(trained_y, y))
    print('Expected: [4], 10')
    print('Got:', w, b)
    print('\n')
    
    # multi variable
    # y = 7x0 + 2x1 + 5x2 + 3x3 + 4x4 + 20
    y, x = init_test_input(5, 1000, np.array([7, 2, 5, 3, 4]), 20)
    # train
    w, b = train(y, x)
    # Calculate loss
    trained_y = np.matmul(x, w) + b
    print('\nMultivariate loss after training')
    print(mean_huber_error(trained_y, y))
    print('Expected: [7, 2, 5, 3, 4], 20')
    print('Got:', w, b)

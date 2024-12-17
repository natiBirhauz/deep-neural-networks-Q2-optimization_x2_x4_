
import numpy as np
import matplotlib.pyplot as plt
"""
Exercise 0: Basic Gradient Descent Implementation
------------------------------------------------
In this exercise, you'll implement basic gradient descent for two functions:
x² and x⁴. This will help you understand the fundamental concepts of optimization.
"""


def x2(x):
    """Function that returns x²"""
    return x * x


def x2_(x):
    """Derivative of x²
    Hint: The derivative of x² is 2x
    """
    return 2*x  # Replace with your implementation


def x4(x):
    """Function that returns x⁴"""
    return x ** 4


def x4_(x):
    """Derivative of x⁴
    Hint: The derivative of x⁴ is 4x³
    """
    return 4*x**3  # Replace with your implementation

def GD(gradient1,gradient2,hiddenLayer_Size,input,learningRate):
    x1=input*0.01 #i normalized the size with *001 so the resault will be smaller and returned the real resault with *100
    x2=input *0.01
    x1_history, x2_history = [input], [input] #save history points for the grid

    for i in range (hiddenLayer_Size):
        grad1=gradient1(x1)
        grad2=gradient2(x2)
        if (np.abs(grad1) < 1e-3) and (np.abs(grad2) < 1e-3):
            print(f"Iteration stoped at {i+1}: x1 = {x1*100:.4f}, x2 = {x2*100:.4f}")
            break
        x1=x1-learningRate*grad1
        x2=x2-learningRate*grad2
        x1_history.append(x1*100)
        x2_history.append(x2*100)
        print(f"Iteration {i+1}: x1 = {x1*100:.4f}, x2 = {x2*100:.4f}")
    return x1_history, x2_history
"""
Exercise 1: Momentum Method Implementation
----------------------------------------
Compare the convergence of standard gradient descent with the momentum method
for both x² and x⁴ functions.
"""

def momentum_update(velocity, graident,learning_rate,momentum=0.9):
    velocity = (momentum * velocity) + (learning_rate * graident)
    return velocity

def momentumGD(gradient1, gradient2, hiddenLayer_Size, input, learningRate):
    x1, x2 = input*0.01, input*0.01
    x1_history, x2_history = [input], [input]
    velocity1, velocity2 = 0.0, 0.0    
    for i in range(hiddenLayer_Size):
        grad1 = gradient1(x1)
        grad2 = gradient2(x2)
        if (np.abs(grad1) < 1e-3) and (np.abs(grad2) < 1e-3):
            print(f"Iteration stoped at {i+1}: x1 = {x1*100:.4f}, x2 = {x2*100:.4f}")
            break
        velocity1 = momentum_update(velocity1,grad1,learningRate)
        velocity2 = momentum_update(velocity2,grad2,learningRate)
        x1 -= velocity1
        x2 -= velocity2
        x1_history.append(x1*100)
        x2_history.append(x2*100)

        print(f"Iteration {i+1}: x1 = {x1*100:.4f}, x2 = {x2*100:.4f}")
    return x1_history, x2_history

"""
Exercise 2: Advanced Optimization Methods
---------------------------------------
Implement and compare four different optimization methods:
1. Gradient Descent (SGD)
2. Momentum
3. Nesterov Accelerated Gradient (NAG)
4. AdaGrad
"""


def nesterov_update(x,velocity, gradient_func,learning_rate, momentum=0.9):
   
    """
    Hint: Look ahead using current velocity before computing gradient
    """
    x_ahead = x + momentum * velocity
    grad = gradient_func(x_ahead)
    velocity = momentum * velocity - learning_rate * grad
    x = x + velocity 
    
    return velocity, x

def NAG(gradient1, gradient2, hiddenLayer_Size, input, learningRate):
    x1, x2 = input * 0.01, input * 0.01
    x1_history, x2_history = [input], [input]
    velocity1, velocity2 = 0.0, 0.0
    for i in range(hiddenLayer_Size):
        grad1 = gradient1(x1)
        grad2 = gradient2(x2)
        print(f"Iteration {i+1}: grad1 = {grad1}, grad2 = {grad2}")
        
        # Check for early stopping if gradients are small
        if (np.abs(grad1) < 1e-3) and (np.abs(grad2) < 1e-3):
            print(f"Iteration stopped at {i+1}: x1 = {x1*100:.4f}, x2 = {x2*100:.4f}")
            break

        # Update velocities and positions using Nesterov momentum
        velocity1, x1 = nesterov_update(x1, velocity1, gradient1, learningRate)
        velocity2, x2 = nesterov_update(x2, velocity2, gradient2, learningRate)

        # Store the history for plotting
        x1_history.append(x1 * 100)
        x2_history.append(x2 * 100)

        print(f"Iteration {i+1}: x1 = {x1*100:.4f}, x2 = {x2*100:.4f}")
    return x1_history, x2_history


def adagrad_update(gradient, historical_gradient, learning_rate):    
    """
    Hint: Use historical gradient to adjust learning rate
    """
    historical_gradient=gradient**2
    gradient-=(learning_rate*gradient)/(np.sqrt(historical_gradient)+1e-3)#very small epsilon
    return gradient, historical_gradient

def adagrad(gradient1, gradient2, hiddenLayer_Size, input, learningRate):
    x1,x2=input*0.01,input*0.01
    x1_history, x2_history = [input], [input]
    historical_gradient1,historical_gradient2=0,0
    for i in range(hiddenLayer_Size):
        grad1 = gradient1(x1)
        grad2 = gradient2(x2)
        if (np.abs(grad1) < 1e-3) and (np.abs(grad2) < 1e-3):
            print(f"Iteration stoped at {i+1}: x1 = {x1*100:.4f}, x2 = {x2*100:.4f}")
            break
        grad1,historical_gradient1= adagrad_update(grad1, historical_gradient1, learningRate)
        grad2,historical_gradient2= adagrad_update(grad2, historical_gradient2, learningRate)
        x1-=grad1
        x2-=grad2

        x1_history.append(x1*100)
        x2_history.append(x2*100)

        print(f"Iteration {i+1}: x1 = {x1*100:.4f}, x2 = {x2*100:.4f}")
    return x1_history, x2_history

def plotGrid (x1,x2):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(x1)), x1, label="x1=(x^2)" ,marker='o')
    plt.plot(range(len(x2)), x2, label="x2=(x^4)", marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("x")
    plt.legend()
    plt.grid()
    plt.show()
# Example usage and testing code
if __name__ == "__main__":
    # Starting points
    X= 10.0
    # Hyperparameters (you should tune these)
    lr = 0.01  # Learning rate for basic gradient descent
    num_steps = 100  # Number of optimization steps

    # Storage for plotting
    history = {
        'sgd_x2': [], 'sgd_x4': [],
        'momentum_x2': [], 'momentum_x4': [],
        'nag_x2': [], 'nag_x4': [],
        'adagrad_x2': [], 'adagrad_x4': []
    }
    
def plotGrid(x1_history, x2_history, optimizer): #PLOTTING
    plt.plot(x1_history, label=f'{optimizer} x1')
    plt.plot(x2_history, label=f'{optimizer} x2')
    plt.title("optimizers plot")
    plt.xlabel('x')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()



# Gradient Descent
x1GD,x2GD=GD(x2_, x4_,num_steps, X, lr)
history['sgd_x2'] = x1GD
history['sgd_x4'] = x2GD
print(f"GD: {history}")
plotGrid(x1GD, x2GD, "GD")

# S Momentum
x1Momentum,x2Momentum=momentumGD(x2_, x4_,num_steps, X, lr)
history['momentum_x2'] = x1Momentum
history['momentum_x4'] = x2Momentum
plotGrid(x1Momentum, x2Momentum, "Momentum")

#  NAG
x1NAG,x2NAG = NAG(x2_, x4_,num_steps, X, lr)
history['nag_x2'] = x1NAG
history['nag_x4'] = x2NAG
plotGrid(x1NAG, x2NAG, "NAG")

#  AdaGrad
x1adagrad, x2adagrad = adagrad(x2_, x4_,num_steps, X, lr)
history['adagrad_x2'] = x1adagrad
history['adagrad_x4'] = x2adagrad
plotGrid(x1adagrad, x2adagrad, "AdaGrad")


"""
Submission Requirements:
1. Implement all TODO sections
2. Include visualizations comparing the convergence of different methods
3. Write a brief analysis (1-2 paragraphs) explaining your observations
4. Submit your code and a brief report in PDF format



"""
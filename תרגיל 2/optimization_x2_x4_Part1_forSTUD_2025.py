
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
    TODO: Implement the derivative of x²
    Hint: The derivative of x² is 2x
    """
    return 2*x  # Replace with your implementation


def x4(x):
    """Function that returns x⁴"""
    return x ** 4


def x4_(x):
    """Derivative of x⁴
    TODO: Implement the derivative of x⁴
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
    velocity1, velocity2 = 0, 0    
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


def nesterov_update(x, velocity, gradient_func, momentum=0.9):
    """
    TODO: Implement Nesterov update
    Hint: Look ahead using current velocity before computing gradient
    """
    return None  # Replace with your implementation


def adagrad_update(gradient, historical_gradient):
    """
    TODO: Implement AdaGrad update
    Hint: Use historical gradient to adjust learning rate
    """
    return None  # Replace with your implementation

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
    X2 = X4 = X2m = X4m = X2n = X4n = X2g = X4g = 10.0

    # Hyperparameters (you should tune these)
    lr = 0.01  # Learning rate for basic gradient descent
    momentum = 0.9  # Momentum coefficient
    num_steps = 100  # Number of optimization steps

    # Storage for plotting
    history = {
        'sgd_x2': [], 'sgd_x4': [],
        'momentum_x2': [], 'momentum_x4': [],
        'nag_x2': [], 'nag_x4': [],
        'adagrad_x2': [], 'adagrad_x4': []
    }
   # print(x4_(1))

    x1,x2=GD(x2_, x4_, 50, 5, 0.01)
    plotGrid(x1,x2)
    x1,x2=momentumGD(x2_, x4_, 50, 5, 0.01)
    plotGrid(x1,x2)
    # Plotting

    # TODO: Implement the training loops for each method

    # TODO: Create visualization of convergence paths

    # TODO: Compare and analyze the results

"""
Submission Requirements:
1. Implement all TODO sections
2. Include visualizations comparing the convergence of different methods
3. Write a brief analysis (1-2 paragraphs) explaining your observations
4. Submit your code and a brief report in PDF format



"""
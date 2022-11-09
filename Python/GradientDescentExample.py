from numpy import *
import time


def print_run_time(func):
    def wrapper(*args, **kw):
        local_time = time.time()
        func(*args, **kw)
        print("Current Function [%s] run time is %.2f" % (func.__name__, time.time() - local_time))
    return wrapper


# y = a + bx + cx^2
# m is slope, b is y-intercept
def compute_error_for_line_given_points(a, b, c, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (a + b * x + c * x * x)) ** 2
    return totalError


def step_gradient(a_current, b_current, c_current, points, learningRate):
    a_gradient = 0
    b_gradient = 0
    c_gradient = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        a_gradient += (((c_current * x * x) + (b_current * x) + a_current) - y)
        b_gradient += (((c_current * x * x) + (b_current * x) + a_current) - y) * x
        c_gradient += (((c_current * x * x) + (b_current * x) + a_current) - y) * x * x
    new_a = a_current - (learningRate * 2 * a_gradient)
    new_b = b_current - (learningRate * 2 * b_gradient)
    new_c = c_current - (learningRate * 2 * c_gradient)
    return [new_a, new_b, new_c]


@print_run_time
def gradient_descent_runner(points, starting_a, starting_b, starting_c, learning_rate, num_iterations,
                            sum_squared_error):
    a = starting_a
    b = starting_b
    c = starting_c
    for i in range(num_iterations):
        a, b, c = step_gradient(a, b, c, array(points), learning_rate)
        error = compute_error_for_line_given_points(a, b, c, points)
        if i % 50000 == 0:
            print("After {0} iterations a = {1}, b = {2}, c = {3}, error = {4}".format(i, a, b, c, error))
        if error < sum_squared_error:
            print("After {0} iterations a = {1}, b = {2}, c = {3}, error = {4}".format(i, a, b, c, error))
            break
    return [a, b, c]


def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.00003
    initial_a = 3
    initial_b = 3
    initial_c = 3
    initial_error = compute_error_for_line_given_points(initial_a, initial_b, initial_c, points)
    num_iterations = 1000000
    sum_squared_error = 1.e-10
    print("Starting gradient descent at a = {0}, b = {1}, c = {2}, error = {3}".format(initial_a, initial_b, initial_c,
                                                                                       initial_error))
    print("Running...")
    gradient_descent_runner(points, initial_a, initial_b, initial_c, learning_rate, num_iterations, sum_squared_error)


if __name__ == '__main__':
    run()

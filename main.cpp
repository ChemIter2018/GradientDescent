#include <iostream>
#include <cmath>
#include <vector>
#include <cstdio>
#include <ctime>

using namespace std;

double compute_error_for_line_given_points(double a, double b, double c, vector<double> x, vector<double> y) {
    double totalError = 0.0;
    unsigned int length_x = x.size();
    for (int i = 0; i < length_x; i++) {
        totalError += pow((y[i] - (a + b * x[i] + c * x[i] * x[i])), 2.0);
    }
    return totalError;
}

vector<double> step_gradient(double a_current, double b_current, double c_current, vector<double> x, vector<double> y,
                             double learningRate) {
    vector<double> newParameters(3);
    double a_gradient  = 0.0;
    double b_gradient  = 0.0;
    double c_gradient  = 0.0;
    unsigned int length_x = x.size();
    for (int i = 0; i < length_x; ++i) {
        a_gradient += (((c_current * x[i] * x[i]) + (b_current * x[i]) + a_current) - y[i]);
        b_gradient += (((c_current * x[i] * x[i]) + (b_current * x[i]) + a_current) - y[i]) * x[i];
        c_gradient += (((c_current * x[i] * x[i]) + (b_current * x[i]) + a_current) - y[i]) * x[i] * x[i];
    }
    newParameters[0] = a_current - (learningRate * 2 * a_gradient);
    newParameters[1] = b_current - (learningRate * 2 * b_gradient);
    newParameters[2] = c_current - (learningRate * 2 * c_gradient);
    return newParameters;
}

vector<double> gradient_descent_runner(const vector<double>& x, const vector<double>& y, double starting_a,
                                       double starting_b, double starting_c, double learning_rate, int num_iterations,
                                       double sum_squared_error) {
    vector<double> finalParameters(3);
    finalParameters[0] = starting_a;
    finalParameters[1] = starting_b;
    finalParameters[2] = starting_c;

    double totalError;
    for (int i = 0; i < num_iterations; ++i) {
        finalParameters = step_gradient(finalParameters[0], finalParameters[1],
                                        finalParameters[2], x, y, learning_rate);
        totalError = compute_error_for_line_given_points(finalParameters[0], finalParameters[1],
                                                         finalParameters[2], x, y);
        if (i % 50000 == 0) {
            printf("After %d iterations a = %.10f, b = %.10f, c = %.10f, error = %.10f\n", i, finalParameters[0],
                   finalParameters[1], finalParameters[2], totalError);
        }
        if (totalError < sum_squared_error) {
            printf("After %d iterations a = %.10f, b = %.10f, c = %.10f, error = %.10f\n", i, finalParameters[0],
                   finalParameters[1], finalParameters[2], totalError);
            break;
        }
    }
    return finalParameters;
}

int main() {
    // Run time
    clock_t startTime, endTime;
    startTime = clock();
    // Initial value of parameters
    vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    vector<double> y = {3, 7, 13, 21, 31, 43, 57, 73, 91, 111};
    double learningRate = 0.00003;
    double initial_a = 3;
    double initial_b = 3;
    double initial_c = 3;
    int numIterations = 1000000;
    double sumSquaredError = 1.0e-10;
    double initialError = compute_error_for_line_given_points(initial_a, initial_b, initial_c, x, y);
    // Run Gradient Descent
    vector<double> finalParameters(3);
    printf("starting gradient descent at a = %.10f, b = %.10f, c = %.10f, error = %.10f\n", initial_a, initial_b,
           initial_c, initialError);
    printf("Running...\n");
    finalParameters = gradient_descent_runner(x, y, initial_a, initial_b, initial_c,
                                              learningRate,numIterations,
                                              sumSquaredError);
    printf("a = %.10f, b = %.10f, c = %.10f\n", finalParameters[0], finalParameters[1], finalParameters[2]);
    // Run time
    endTime = clock();
    printf("Current Function [Gradient Descent] run time is %.2fs.\n", double(endTime - startTime)/CLOCKS_PER_SEC);
    return 0;
}

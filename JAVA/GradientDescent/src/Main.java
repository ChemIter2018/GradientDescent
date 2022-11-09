public class Main {
    public static void main(String[] args) {
        // Run time
        long startTime=System.currentTimeMillis();
        // Initial value of parameters
        double[] x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        double[] y = {3, 7, 13, 21, 31, 43, 57, 73, 91, 111};
        double learningRate = 0.00003;
        double initial_a = 3;
        double initial_b = 3;
        double initial_c = 3;
        int numIterations = 1000000;
        double sumSquaredError = 1.0e-10;
        double initialError = compute_error_for_line_given_points(initial_a, initial_b, initial_c, x, y);
        // Run Gradient Descent
        double[] finalParameters;
        System.out.printf("starting gradient descent at a = %.10f, b = %.10f, c = %.10f, error = %.10f\n", initial_a, initial_b,
                initial_c, initialError);
        System.out.print("Running...\n");
        finalParameters = gradient_descent_runner(x, y, initial_a, initial_b, initial_c,
                learningRate,numIterations,
                sumSquaredError);
        System.out.printf("a = %.10f, b = %.10f, c = %.10f\n", finalParameters[0], finalParameters[1], finalParameters[2]);
        // Run time
        long endTime=System.currentTimeMillis();
        System.out.printf("Current Function [Gradient Descent] run time is %.2fs.\n", (endTime - startTime) / 1000.0);

    }

    private static double compute_error_for_line_given_points(double a, double b, double c, double[] x, double[] y) {
        double totalError = 0.0;
        int length_x = x.length;
        for (int i = 0; i < length_x; i++) {
            totalError += Math.pow((y[i] - (a + b * x[i] + c * x[i] * x[i])), 2.0);
        }
        return totalError;
    }

    private static double[] step_gradient(double a_current, double b_current, double c_current, double[] x, double[] y,
                                 double learningRate) {
        double[] newParameters = new double[3];
        double a_gradient  = 0.0;
        double b_gradient  = 0.0;
        double c_gradient  = 0.0;
        int length_x = x.length;
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

    private static double[] gradient_descent_runner(double[] x, double[] y, double starting_a,
                                           double starting_b, double starting_c, double learning_rate, int num_iterations,
                                           double sum_squared_error) {
        double[] finalParameters = new double[3];
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
                System.out.printf("After %d iterations a = %.10f, b = %.10f, c = %.10f, error = %.10f\n", i, finalParameters[0],
                        finalParameters[1], finalParameters[2], totalError);
            }
            if (totalError < sum_squared_error) {
                System.out.printf("After %d iterations a = %.10f, b = %.10f, c = %.10f, error = %.10f\n", i, finalParameters[0],
                        finalParameters[1], finalParameters[2], totalError);
                break;
            }
        }
        return finalParameters;
    }
}



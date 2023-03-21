/**
 * @brief Test functions
*/

#include <cmath>
#include <vector>
#include "Eigen/Eigen"
#include "ekf.h"
#include "simulation_helper.h"
#include "matplotlibcpp/matplotlibcpp.h"

int main()
{
    Eigen::VectorXd x_est(4);
    x_est << 0, 0, 0, 0;
    Eigen::MatrixXd P_est(4,4);
    P_est << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1;
    std::vector<double> px, py;
    plotCovEllipse(x_est, P_est, px, py);
    plt::plot(px, py, "--r");
    // Add graph title
    plt::title("Sample figure");
    // Enable legend.
    plt::legend();
    plt::pause(10);

    return 0;
}
/**
 * @brief Functions related to running simulation
*/

#include <cmath>
#include <vector>
#include <Eigen/Eigen>

using namespace Eigen;

/**
 * @brief Get control inputs for the simulation
 * @param t current time
 * @param u current control inputs
*/
void calcInput(const double &t, VectorXd &u)
{
    double v = 1.0;
    double yaw_rate = 0.1;
    u << v, yaw_rate;
}
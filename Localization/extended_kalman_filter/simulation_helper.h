/**
 * @brief Functions related to running simulation
*/

#include <cmath>
#include <vector>
#include <Eigen/Eigen>
#include <Eigen/Eigenvalues>

#include "matplotlibcpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;

using namespace std;
using namespace Eigen;

/* Data structs for plotting */
struct state_x
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> yaw;
    std::vector<double> v;
};

struct obs_z
{
    std::vector<double> x;
    std::vector<double> y;
};

struct input_u
{
    std::vector<double> v;
    std::vector<double> yaw_rate;
};

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

void plotCovEllipse(const VectorXd &x_est, const MatrixXd &cov, std::vector<double> &px, std::vector<double> &py)
{
    // eigenvalue reference: https://eigen.tuxfamily.org/dox/classEigen_1_1EigenSolver.html#a0ccaeb4f7d44c18af60a7b3a1dd91f7a
    MatrixXd A = cov.block(0,0,2,2);
    EigenSolver<MatrixXd> es(A);
    complex<double> lambda = es.eigenvalues()[0];
    // cout << "Consider the first eigenvalue, lambda = " << lambda << endl;
    std::vector<double> e_val({es.eigenvalues()[0].real(), es.eigenvalues()[1].real()});

    VectorXcd v = es.eigenvectors().col(0);
    // cout << "If v is the corresponding eigenvector, then lambda * v = " << endl << lambda * v << endl;
    // cout << "... and A * v = " << endl << A.cast<complex<double> >() * v << endl << endl;
    
    int bigind, smallind;
    bigind = e_val[0] > e_val[1] ? 0 : 1;
    smallind = 1 - bigind;
    VectorXcd vv = es.eigenvectors().col(bigind);
    std::vector<double> e_vec_big({vv(0).real(), vv(1).real()});

    double a = sqrt(e_val[bigind]);
    double b = sqrt(e_val[smallind]);

    const int samples = 32;
    MatrixXd sample_xy(samples, 2);
    for(int i = 0; i < samples; i++)
    {
        sample_xy(i, 0) = a * cos(i * 2 * M_PI / samples);
        sample_xy(i, 1) = b * sin(i * 2 * M_PI / samples);
    }

    double angle = atan2(e_vec_big[1], e_vec_big[0]);
    Rotation2D<double> rot2(angle);

    for(int i = 0; i < samples; i++)
    {
        Vector2d xy = rot2 * sample_xy.row(i).transpose();
        cout << xy << endl;
        px.push_back(xy(0) + x_est(0));
        py.push_back(xy(1) + x_est(1));
    }
    if(px.size() > 0 && py.size() > 0)
    {
        px.push_back(px[0]);
        py.push_back(py[0]);
    } 


    // plt::plot(px, py, "--r");

}
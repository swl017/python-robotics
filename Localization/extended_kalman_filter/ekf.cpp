/**
 * @brief EKF class implementation
 * @date 2023-03-18
*/

#include <iostream>
#include <vector>
#include <cmath>
// #include <Eigen/Dense>
// #include <Eigen/Geometry>
#include <Eigen/Eigen>

#include "ekf.h"
#include "simulation_helper.h"
#include "matplotlibcpp/matplotlibcpp.h"

using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;

EKF::EKF(const int &state_dim, const int &input_dim, const int &obs_dim, const double &dt) :
_state_dim(state_dim),
_input_dim(input_dim),
_obs_dim(obs_dim),
_dt(dt)
{
    _Q.resize(state_dim, state_dim);
    _R.resize(obs_dim, obs_dim);
    _F.resize(state_dim, state_dim);
    _P_pred.resize(state_dim, state_dim);
    _B.resize(state_dim, input_dim);
    _H.resize(obs_dim, state_dim);
    _I_ss.resize(state_dim, state_dim);
    
    _x_true.resize(state_dim);
    _x_pred.resize(state_dim);
    _y.resize(obs_dim);
    _z.resize(obs_dim);
    _z_pred.resize(obs_dim);
    _u.resize(input_dim);

    _input_noise.resize(input_dim, input_dim);
    _sensor_noise.resize(input_dim, input_dim);

    _I_ss = MatrixXd::Identity(state_dim, state_dim);
}

EKF::~EKF()
{
}

MatrixXd EKF::getCovMat(const std::vector<double> &cov_vec)
{
    MatrixXd A(cov_vec.size(), cov_vec.size());
    for(size_t i = 0; i < cov_vec.size(); i++)
    {
        A(i,i) = cov_vec[i]*cov_vec[i];
    }

    return A;
}

MatrixXd EKF::getCovMat(const MatrixXd &cov_vec)
{
    MatrixXd A = cov_vec.array().square().matrix().asDiagonal();

    return A;
}

void EKF::setProcessNoiseCov(const MatrixXd &Q)
{
    _Q = Q;
}

void EKF::setObservationNoiseCov(const MatrixXd &R)
{
    _R = R;
}

void EKF::setInputNoise(const std::vector<double> &cov_vec)
{
    _input_noise = getCovMat(cov_vec);
}

void EKF::setSensorNoise(const std::vector<double> &cov_vec)
{
    _sensor_noise = getCovMat(cov_vec);
}

// /**
//  * @todo sensor measurement callback when implemented in real-time
// */
// MatrixXd EKF::getObservation()
// {
//     return MatrixXd::Identity(_obs_dim, _state_dim); // temp
// }

// void EKF::setObservation(const MatrixXd &mat)
// {
//     _z = mat;
// }

// void EKF::setControlInput(const VectorXd &vec)
// {
//     u = vec;
// }

VectorXd EKF::motionModel(const VectorXd &x, const VectorXd &u)
{
    MatrixXd A(_state_dim, _state_dim);
    A << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0;

    MatrixXd B(_state_dim, _input_dim);
    B << _dt*cos(x(INDEX::STATE_YAW)), 0.0,
         _dt*sin(x(INDEX::STATE_YAW)), 0.0,
         0.0, _dt,
         1.0, 0.0;

    return A * x + B * u;
}

/**
 * @brief Jacobian of Motion Model
 * @note
        motion model
        x_{t+1} = x_t+v*dt*cos(yaw)
        y_{t+1} = y_t+v*dt*sin(yaw)
        yaw_{t+1} = yaw_t+omega*dt
        v_{t+1} = v{t}
        so
        dx/dyaw = -v*dt*sin(yaw)
        dx/dv = dt*cos(yaw)
        dy/dyaw = v*dt*cos(yaw)
        dy/dv = dt*sin(yaw)
*/
MatrixXd EKF::jacob_f(const VectorXd &x, const VectorXd &u)
{
    double yaw = x(INDEX::STATE_YAW);
    double v = u(INDEX::INPUT_V);
    MatrixXd jacob_f(_state_dim, _state_dim);
    jacob_f << 1.0, 0.0, -v*_dt*sin(yaw), _dt*cos(yaw),
               0.0, 1.0,  v*_dt*cos(yaw), _dt*sin(yaw),
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0;
            
    return jacob_f;
}

/**
 * @brief Jacobian of Motion Model
 * @note  Observation model H
 *        z = H * x + v
 *        in this case z = [x_est, y_est]
 *        thus, H = [1 0 0 0; 0 1 0 0;]
*/
VectorXd EKF::observationModel(const VectorXd &x)
{
    MatrixXd H(_obs_dim, _state_dim);
    H << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0;

    return H * x;
}

MatrixXd EKF::jacob_h(const VectorXd &x)
{
    MatrixXd jacob_h(_obs_dim, _state_dim);
    jacob_h << 1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0;
            
    return jacob_h;
}

// template <typename T>
// void EKF::jacob_h(const VectorXd &x, T &H)
// {
//     H << 1.0, 0.0, 0.0, 0.0,
//          0.0, 1.0, 0.0, 0.0;
// }

/**
 * @brief Compute EKF estimation.
 * @note  Prediction Stage:
 *        1. Predicted State Estimate
 *           x_pred = f(x_est, u)
 *        2. Predicted Covariance Matrix
 *           P_pred = F * P_est * F^T + Q
 *           where F = Jacob(f) over x given x_est and u
 * 
 *        Update Stage:
 *        1. Measurement residual
 *           y = z - h(x_pred)
 *        2. Residual covariance
 *           S = H * P_pred * H^T + R
 *           where H = Jacob(h) over x given x_pred
 *        3. Near-Optimal Kalman gain
 *           K = P_pred * H^T * inv(S)
 *        4. Updated state estimate
 *           x_est = x_pred + K * y
 *        5. Updated covariance estimate
 *           P_est = (I - K * H) * P_pred
 * 
*/
void EKF::ekfEstimation(VectorXd &x_est, MatrixXd &P_est, const VectorXd &z, const MatrixXd &u)
{
    /* Predict */
    _x_pred = motionModel(x_est, u);
    _F = jacob_f(x_est, u);
    _P_pred = _F * P_est * _F.transpose() + _Q;
    // cout << "x_est" << x_est << endl;
    // cout << "u" << u << endl;
    // cout << "_x_pred" << _x_pred << endl;
    // cout << "_F" << _F << endl;
    // cout << "_Q" << _Q << endl;
    // cout << "P_est" << P_est << endl;
    // cout << "_P_pred" << _P_pred << endl;

    /* Update */
    _z_pred = observationModel(_x_pred);
    _y = z - _z_pred;
    MatrixXd H = jacob_h(_x_pred);
    MatrixXd S = H * _P_pred * H.transpose() + _R;
    MatrixXd K = _P_pred * H.transpose() * S.inverse();
    x_est = _x_pred + K * _y;
    P_est = (_I_ss - K * H) * _P_pred;
    // cout << "z" << z << endl;
    // cout << "_z_pred" << _z_pred << endl;
    // cout << "_y" << _y << endl;
    // cout << "H" << H << endl;
    // cout << "S" << S << endl;
    // cout << "K" << K << endl;
    // cout << "x_est" << x_est << endl;
    // cout << "_I_ss" << _I_ss << endl;
    // cout << "P_est" << P_est << endl;
    // cout << "================" << endl;


}


/**
 * @note The dimensions of the problem(4,2,2)
 *       x = [x, y, yaw, v]
 *       u = [v, yaw_rate]
 *       z = [x, y]
*/
int main()
{
    /* EKF parameters */
    int state_dim = 4;
    int input_dim = 2;
    int obs_dim = 2;
    double dt = 0.1;
    EKF ekf(state_dim, input_dim, obs_dim, dt);

    /* EKF Tunning factors */
    MatrixXd Q = ekf.getCovMat(vector<double>({0.1, 0.1, 0.01, 1.0}));
    MatrixXd R = ekf.getCovMat(vector<double>({1.0, 1.0}));
    ekf.setProcessNoiseCov(Q);
    ekf.setObservationNoiseCov(R);

    /* Parameters for generating a simulation */
    // MatrixXd input_noise = ekf.getCovMat(vector<double>({1.0, 0.5}));
    // MatrixXd observation_noise = ekf.getCovMat(vector<double>({0.5, 0.5}));
    MatrixXd input_noise = ekf.getCovMat(vector<double>({1.0, 0.5}));
    MatrixXd observation_noise = ekf.getCovMat(vector<double>({0.5, 0.5}));

    double dt_sim = 0.1;
    double sim_duration = 500.0;
    double sim_time = 0.0;
    double ekf_timer = 0.0;

    /* Initial values */
    // simulation ground truth
    VectorXd x_true(state_dim);
    x_true << 0.0, 0.0, 0.0, 0.0;
        // x_true << 1.0, 0.0, M_PI_2, 0.0;
    VectorXd u_true(input_dim);
    u_true << 0.0, 0.0;
    // No EKF estimation
    VectorXd x_deadreck(state_dim);
    x_deadreck << 0.0, 0.0, 0.0, 0.0;
        // x_deadreck << 1.0, 0.0, M_PI_2, 0.0;
    // EKF estimation
    VectorXd x_est(state_dim);
    x_est << 0.0, 0.0, 0.0, 0.0;
        // x_est << 1.0, 0.0, M_PI_2, 0.0;
    MatrixXd P_est(state_dim, state_dim);
    P_est = MatrixXd::Identity(state_dim, state_dim);
    // Measured control input containing noise
    VectorXd u(input_dim);
    u << 0.0, 0.0;
    // Observation
    VectorXd z(obs_dim);
    z << 0.0, 0.0;

    /* Variables for plotting */
    state_x plt_x_true, plt_x_deadreck, plt_x_est;
    obs_z plt_z;
    input_u plt_u;
    int i = 0;

    /* Simulation loop */
    while(sim_time < sim_duration)
    {
        // Update the simulation one step
        sim_time += dt_sim;
        ekf_timer += dt_sim;
        calcInput(sim_time, u_true);
        x_true = ekf.motionModel(x_true, u_true);

        if(true)
        // if(ekf_timer >= dt)
        {
            // Generate measurements
            z = ekf.observationModel(x_true) + observation_noise * VectorXd::Random(obs_dim);
            u = u_true + input_noise * VectorXd::Random(input_dim);

            // Update dead-reckoning(for comparison)
            x_deadreck = ekf.motionModel(x_deadreck, u);

            // Perform EKF estimation
            ekf.ekfEstimation(x_est, P_est, z, u);

            // Reset timer
            ekf_timer = 0;
        }

        /* matplotlibcpp */
        plt_x_true.x.push_back(x_true(EKF::INDEX::STATE_X));
        plt_x_true.y.push_back(x_true(EKF::INDEX::STATE_Y));
        plt_x_true.yaw.push_back(x_true(EKF::INDEX::STATE_YAW));
        plt_x_true.v.push_back(x_true(EKF::INDEX::STATE_V));
        plt_x_deadreck.x.push_back(x_deadreck(EKF::INDEX::STATE_X));
        plt_x_deadreck.y.push_back(x_deadreck(EKF::INDEX::STATE_Y));
        plt_x_deadreck.yaw.push_back(x_deadreck(EKF::INDEX::STATE_YAW));
        plt_x_deadreck.v.push_back(x_deadreck(EKF::INDEX::STATE_V));
        plt_x_est.x.push_back(x_est(EKF::INDEX::STATE_X));
        plt_x_est.y.push_back(x_est(EKF::INDEX::STATE_Y));
        plt_x_est.yaw.push_back(x_est(EKF::INDEX::STATE_YAW));
        plt_x_est.v.push_back(x_est(EKF::INDEX::STATE_V));

        plt_z.x.push_back(z(EKF::INDEX::OBS_X));
        plt_z.y.push_back(z(EKF::INDEX::OBS_Y));

        plt_u.v.push_back(u(EKF::INDEX::INPUT_V));
        plt_u.yaw_rate.push_back(u(EKF::INDEX::INPUT_YAWRATE));

        if (i % 10 == 0) {
			// Clear previous plot
			plt::clf();
			// Plot line from given x and y data. Color is selected automatically.
			plt::named_plot("x_true", plt_x_true.x, plt_x_true.y);
			plt::named_plot("x_deadreck", plt_x_deadreck.x, plt_x_deadreck.y);
			plt::named_plot("x_est", plt_x_est.x, plt_x_est.y);
			plt::plot(plt_z.x, plt_z.y, ".");

            vector<double> px, py;
            plotCovEllipse(x_est, P_est, px, py);
            plt::plot(px, py, "--r");

			// Set x-axis to interval [0,1000000]
			// plt::xlim(0, n*n);

			// Add graph title
			plt::title("Dead Reckoning vs. EKF");
			// Enable legend.
			plt::legend();
            plt::axis("equal");
			// Display plot continuously
			plt::pause(0.1);
		}
        i++;


    }
    
    cout << "Simulation Completed." << endl;

    return 0;
}
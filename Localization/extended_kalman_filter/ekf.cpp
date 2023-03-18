/**
 * @brief EKF class implementation
 * @date 2023-03-18
*/

#include <iostream>
#include <vector>
#include <cmath>
// #include <Eigen/Dense>
// #include <Eigen/Geometry>
#include "Eigen/Eigen"

#include "ekf.h"

using namespace std;
using namespace Eigen;

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

    _input_noise.resize(input_dim, input_dim);
    _sensor_noise.resize(input_dim, input_dim);

    MatrixXd I_ss = MatrixXd::Identity(state_dim, state_dim);
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

MatrixXd EKF::setInputNoise(const std::vector<double> &cov_vec)
{
    _input_noise = getCovMat(cov_vec);
}

MatrixXd EKF::setSensorNoise(const std::vector<double> &cov_vec)
{
    _sensor_noise = getCovMat(cov_vec);
}

/**
 * @todo sensor measurement callback when implemented in real-time
*/
MatrixXd EKF::getObservation()
{
    return MatrixXd::Identity(_obs_dim, _state_dim); // temp
}

void EKF::setObservation(const MatrixXd &mat)
{
    _z = mat;
}

MatrixXd EKF::getControlInput()
{

}

void EKF::setControlInput(const VectorXd &vec)
{
    _ud = vec;
}

VectorXd EKF::motionModel(const VectorXd &x, const VectorXd &u)
{
    MatrixXd F(_state_dim, _state_dim);
    F << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 0.0;

    _F = F;

    MatrixXd B(_state_dim, _input_dim);
    B << _dt*cos(x(STATE_YAW)), 0.0,
         _dt*sin(x(STATE_YAW)), 0.0,
         0.0, _dt,
         1.0, 0.0;

    return F * x + B * u;
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
    double yaw = x(STATE_YAW);
    double v = u(INPUT_V);
    MatrixXd jacob_f(_state_dim, _state_dim);
    jacob_f << 1.0, 0.0, -v*_dt*sin(yaw), _dt*cos(yaw),
               0.0, 1.0,  v*_dt*cos(yaw), _dt*sin(yaw),
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 0.0;
            
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

// MatrixXd EKF::jacob_h(const VectorXd &x)
// {
//     MatrixXd jacob_h(_obs_dim, _state_dim);
//     jacob_h << 1.0, 0.0, 0.0, 0.0,
//               0.0, 1.0, 0.0, 0.0;
            
//     return jacob_h;
// }

template <typename T>
void EKF::jacob_h(const VectorXd &x, T &H)
{
    H << 1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0;
}

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

    /* Update */
    _z_pred = observationModel(_x_pred);
    _y = _z - _z_pred;
    MatrixXd H(_obs_dim, _state_dim);
    jacob_h(_x_pred, H);
    MatrixXd S = H * _P_pred * H.transpose() + _R;
    MatrixXd K = _P_pred * H.transpose() * S.inverse();
    x_est = _x_pred + K * _y;
    P_est = (_I_ss - K * H) * _P_pred;

}


/**
 * @note The dimensions of the problem(4,2)
 *       x = [x, y, yaw, v]
 *       u = [v, yaw_rate]
*/
int main()
{
    int state_dim = 4;
    int input_dim = 2;
    int obs_dim = 2;
    double dt = 0.1;
    EKF ekf(state_dim, input_dim, obs_dim, dt);
    vector<double> q({0.1, 0.1, 0.01, 1.0});
    
    /* Process Noise Covariance */
    MatrixXd Q = ekf.getCovMat(q);

    cout << "Q:\n" << Q << endl;

    /* Observation Noise Covariance*/
    MatrixXd R = ekf.getCovMat(vector<double>({
                    pow(1.0, 2),
                    pow(1.0, 2)
                    }));
    cout << "R:\n" << R << endl;

    MatrixXd input_noise = ekf.getCovMat(vector<double>({
                                pow(0.1, 2),
                                pow(0.1, 2),
                                pow(0.01, 2),
                                pow(1.0, 2)
                                }));

    

    return 0;
}
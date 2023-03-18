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

using namespace std;
using namespace Eigen;

class EKF
{
public:
    EKF(const int &state_dim, const int &input_dim, const int &obs_dim, const double &dt);
    ~EKF();

    MatrixXd getCovMat(const std::vector<double> &cov_vec);
    MatrixXd getCovMat(const MatrixXd &cov_vec);
    void setProcessNoiseCov(const MatrixXd &Q);
    void setObservationNoiseCov(const MatrixXd &R);
    void setInputNoise(const std::vector<double> &cov_vec);
    void setSensorNoise(const std::vector<double> &cov_vec);

    /* Observations(Measurements)*/
    // MatrixXd getObservation(); // get measurement 
    // void setObservation(const MatrixXd &mat); // set measurement 
    // MatrixXd getControlInput(); // get u
    // void setControlInput(const VectorXd &vec); // set u

    /* Predictions */
    VectorXd motionModel(const VectorXd &x, const VectorXd &u);
    MatrixXd jacob_f(const VectorXd &x, const VectorXd &u);

    /* Update */
    VectorXd observationModel(const VectorXd &x);
    template <typename T>
    void jacob_h(const VectorXd &x, T &H);

    /* EKF */
    void ekfEstimation(VectorXd &x_est, MatrixXd &P_est, const VectorXd &z, const MatrixXd &u);

    /* Types */
    enum STATE_INDEX
    {
        STATE_X = 0,
        STATE_Y = 1,
        STATE_YAW = 2,
        STATE_V = 3,
    };

    enum INPUT_INDEX
    {
        INPUT_V = 0,
        INPUT_YAWRATE = 1,
    };

    /* Variables */
    MatrixXd _input_noise; 
    MatrixXd _sensor_noise; 
    double _sim_time;
    int _state_dim;
    int _input_dim;
    int _obs_dim;
    double _dt;

private:
    MatrixXd _Q; // Process noise covariance 
    MatrixXd _R; // Observation noise covariance
    MatrixXd _F; // State transition model
    MatrixXd _P_pred; // State transition model
    MatrixXd _B; // Control input model
    MatrixXd _H; // Observation model
    MatrixXd _I_ss; // Identity Matrices(state_dim*state_dim)

    VectorXd _x_true; // True state
    VectorXd _x_pred; // Predicted state
    VectorXd _y; // Residual
    VectorXd _z; // Observation(measurement)
    VectorXd _z_pred; // Observation(measurement) prediction
    VectorXd _u; // Control input with noise
};

#include <iostream>
#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/2;

  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  lambda_ = 3 - n_aug_;
  time_us_ = 0.0;

  weights_ = VectorXd(2 * n_aug_ + 1);
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_)
  {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {

      double radial_disance = meas_package.raw_measurements_[0];
      double angle = meas_package.raw_measurements_[1];
      double radial_change = meas_package.raw_measurements_[2];
      double px = radial_change * cos(angle);
      double py = radial_change * sin(angle);
      x_ << px, py, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
    }
    P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 0.03, 0,
        0, 0, 0, 0, 0.03;
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;

    return;
  }
  double dt = (double)(meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(dt);

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
  else
  {
    UpdateLidar(meas_package);
  }
}

void UKF::Prediction(double delta_t)
{
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  MatrixXd aug_state_matrix = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  AugmentedSigmaPoints(&aug_state_matrix);
  SigmaPointPrediction(&aug_state_matrix, delta_t);
  PredictMeanAndCovariance();
}

void UKF::UpdateLidar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   * 
   */
  int n_z = 2;
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  PredictLidarMeasurement(&z_pred, &S, &Zsig, n_z);
  UpdateLidarState(&z_pred,n_z, &S,&Zsig,meas_package);
}

void UKF::UpdateRadar(MeasurementPackage meas_package)
{
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // mean predicted measurement

  int n_z = 3;
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  PredictRadarMeasurement(&z_pred, &S, &Zsig, n_z);
  UpdateRadarState(&z_pred, n_z, &S, &Zsig, meas_package);
}

void UKF::AugmentedSigmaPoints(MatrixXd *Xsig_out)
{

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  x_aug.head(5) = x_;
  x_aug[5] = 0;
  x_aug[6] = 0;
  // create augmented mean state
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5, 5) = P_;
  P_aug(5, 5) = std_a_ * std_a_;
  P_aug(6, 6) = std_yawdd_ * std_yawdd_;

  // create augmented covariance matrix
  MatrixXd A = P_aug.llt().matrixL();
  // create square root matrix
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i)
  {
    Xsig_aug.col(1 + i) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(n_aug_ + i + 1) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  *Xsig_out = Xsig_aug;
}

void UKF::SigmaPointPrediction(MatrixXd *Xsig_aug_input, double delta_t)
{

  // create matrix with predicted sigma points as columns

  MatrixXd Xsig_aug = *Xsig_aug_input;

  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double pxk = Xsig_aug(0, i);
    double pyk = Xsig_aug(1, i);
    double vk = Xsig_aug(2, i);
    double phik = Xsig_aug(3, i);
    double phidotk = Xsig_aug(4, i);
    double acc_noise = Xsig_aug(5, i);
    double ang_noise = Xsig_aug(6, i);

    double pxk_ = 0.0;
    double pyk_ = 0.0;
    double vk_ = 0.0;
    double phik_ = 0.0;
    double phidotk_ = 0.0;
    if (fabs(phidotk) > 0.001)
    {
      pxk_ = pxk + vk / phidotk * (sin(phik + phidotk * delta_t) - sin(phik));
      pyk_ = pyk + vk / phidotk * (cos(phik) - cos(phik + phidotk * delta_t));
    }
    else
    {
      pxk_ = pxk + vk * cos(phik) * delta_t;
      pyk_ = pyk + vk * sin(phik) * delta_t;
    }
    vk_ = vk;
    phik_ = phik + phidotk * delta_t;
    phidotk_ = phidotk;

    //adding noise parameter
    double dtsqr = delta_t * delta_t;
    pxk_ = pxk_ + 0.5 * dtsqr * cos(phik) * acc_noise;
    pyk_ = pyk_ + 0.5 * dtsqr * sin(phik) * acc_noise;
    vk_ = vk_ + delta_t * acc_noise;
    phik_ = phik_ + 0.5 * dtsqr * ang_noise;
    phidotk_ = phidotk_ + delta_t * ang_noise;

    Xsig_pred_(0, i) = pxk_;
    Xsig_pred_(1, i) = pyk_;
    Xsig_pred_(2, i) = vk_;
    Xsig_pred_(3, i) = phik_;
    Xsig_pred_(4, i) = phidotk_;
  }

  //std::cout << "Xsig_pred = " << std::endl << Xsig_pred_ << std::endl;
}

void UKF::PredictMeanAndCovariance()
{

  // set weights
  double weight = lambda_ / (lambda_ + n_aug_);
  weights_[0] = weight;
  for (int i = 1; i < 2 * n_aug_ + 1; ++i)
  {
    weight = 1 / ((lambda_ + n_aug_) * 2);
    weights_[i] = weight;
  }
  // predict state mean
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    x_ = x_ + weights_[i] * Xsig_pred_.col(i);
  }
  // predict state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd diff = Xsig_pred_.col(i) - x_;
    while (diff(3) > M_PI)
      diff(3) -= 2. * M_PI;
    while (diff(3) < -M_PI)
      diff(3) += 2. * M_PI;

    P_ = P_ + weights_(i) * diff * diff.transpose();
  }

  // print result
  // std::cout << "Predicted state" << std::endl;
  // std::cout << x << std::endl;
  // std::cout << "Predicted covariance matrix" << std::endl;
  // std::cout << P << std::endl;
}

void UKF::UpdateRadarState(VectorXd *z_pred_input, int n_z, MatrixXd *S_input, MatrixXd *Zsig_input, MeasurementPackage meas_package)
{

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  VectorXd z_pred = *z_pred_input;
  MatrixXd S = *S_input;
  MatrixXd Zsig = *Zsig_input;

  VectorXd z = VectorXd(n_z);

  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];

  Tc.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    // angle normalization
    while (x_diff(3) > M_PI)
      x_diff(3) -= 2. * M_PI;
    while (x_diff(3) < -M_PI)
      x_diff(3) += 2. * M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  // calculate cross correlation matrix
  MatrixXd K = Tc * S.inverse();
  // calculate Kalman gain K;
  x_ = x_ + K * (z - z_pred);
  // update state mean and covariance matrix
  P_ = P_ - K * S * K.transpose();

  VectorXd nis_z = VectorXd(n_z);
  nis_z = z - z_pred;
  double NIS = nis_z.transpose() * S.inverse() * nis_z;
  std::cout << "NIS :" << NIS << std::endl;
}

void UKF::PredictRadarMeasurement(VectorXd *z_out, MatrixXd *S_out, MatrixXd *Zsig_out, int n_z)
{
  int n_z_ = n_z;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);

  MatrixXd R = MatrixXd(n_z_, n_z_);
  R << std_radr_ * std_radr_, 0, 0,
      0, std_radphi_ * std_radphi_, 0,
      0, 0, std_radrd_ * std_radrd_;

  /**
   * Student part begin
   */
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double phi = Xsig_pred_(3, i);
    double phidot = Xsig_pred_(4, i);

    double rho = sqrt(px * px + py * py);
    double r_phi = atan2(py, px);
    double rhodot = (px * v * cos(phi) + py * v * sin(phi)) / rho;

    Zsig(0, i) = rho;
    Zsig(1, i) = r_phi;
    Zsig(2, i) = rhodot;
  }
  z_pred.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  // transform sigma points into measurement space
    S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd z_diff = VectorXd(n_z_);
    z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI)
      z_diff(1) -= 2. * M_PI;
    while (z_diff(1) < -M_PI)
      z_diff(1) += 2. * M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  // calculate mean predicted measurement

  // calculate innovation covariance matrix S
  S = S + R;
  /**
   * Student part end
   */

  // print result
  // std::cout << "z_pred: " << std::endl
  // << z_pred << std::endl;
  //std::cout << "S: " << std::endl
  //  << S << std::endl;

  // write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}

void UKF::PredictLidarMeasurement(VectorXd *z_out, MatrixXd *S_out, MatrixXd *Zsig_out, int n_z)
{
  int n_z_ = n_z;
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);

  MatrixXd R = MatrixXd(n_z_, n_z_);
  R << std_laspx_ * std_laspx_, 0,
      0, std_laspy_ * std_laspy_;

  /**
   * Student part begin
   */
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    // double v = Xsig_pred_(2, i);
    // double phi = Xsig_pred_(3, i);
    // double phidot = Xsig_pred_(4, i);

    Zsig(0, i) = px;
    Zsig(1, i) = py;
  }

  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd z_diff = VectorXd(n_z_);
    z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  // calculate mean predicted measurement

  // calculate innovation covariance matrix S
  S = S + R;

  // write result
  *z_out = z_pred;
  *S_out = S;
  *Zsig_out = Zsig;
}

void UKF::UpdateLidarState(VectorXd *z_pred_input, int n_z, MatrixXd *S_input, MatrixXd *Zsig_input, MeasurementPackage meas_package)
{
  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  VectorXd z_pred = *z_pred_input;
  MatrixXd S = *S_input;
  MatrixXd Zsig = *Zsig_input;

  VectorXd z = VectorXd(n_z);

  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

  Tc.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  // calculate cross correlation matrix
  MatrixXd K = Tc * S.inverse();
  // calculate Kalman gain K;
  x_ = x_ + K * (z - z_pred);
  // update state mean and covariance matrix
  P_ = P_ - K * S * K.transpose();

  VectorXd nis_z = VectorXd(n_z);
  nis_z = z - z_pred;
  double NIS = nis_z.transpose() * S.inverse() * nis_z;
  std::cout << "NIS :" << NIS << std::endl;
}



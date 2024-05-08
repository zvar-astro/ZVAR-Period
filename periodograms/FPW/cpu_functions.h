#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>
#include "Eigen/Dense"

using namespace std;
using Eigen::VectorXf;
using Eigen::VectorXd;
using Eigen::VectorXi;
using Eigen::MatrixXf;
using std::vector;

VectorXi makeIndices(const VectorXd& times, const double f, const int N_bins);
double deltaChi2(const VectorXd& y, const VectorXd& ivar, const VectorXi& ind, int N_bins, int N_dat);
vector<double> runFPW(const vector<double>& t, const vector<double>& y, const vector<double>& ivar, const vector<double>& freqs, int N_bins);

#endif
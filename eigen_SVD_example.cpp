#include <iostream>

#include <Eigen/Dense>
#include <Eigen/Sparse>

int main(int argc, char *argv[])
{
	Eigen::MatrixXf m = Eigen::MatrixXf::Random(3,2);
	cout << "Here is the matrix m:" << endl << m << endl;
	
	// compute SVD decomposition of a matrix m
	// SVD: m = U * S * V^T
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
	
	// get the decomposed matrices
	const Eigen::Matrix3f U = svd.matrixU();
	// note that this is actually V^T!!
	const Eigen::Matrix3f V = svd.matrixV();
	const Eigen::VectorXf S = svd.singularValues();
	
	return 0;
}
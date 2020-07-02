#include <Rcpp.h>
#include <cmath>
#include <RcppEigen.h>

using namespace Rcpp;
using namespace Eigen;
using namespace std;
typedef Eigen::MappedSparseMatrix<double, Eigen::RowMajor> MSM;
typedef Eigen::MappedSparseMatrix<double> MSM2;

// [[Rcpp::export]]
double floglik(SEXP & TM, Eigen::Map<Eigen::VectorXd> values, const Eigen::Map<Eigen::VectorXd> rparams,
                         const Eigen::Map<Eigen::MatrixXd> transmatrix, const Eigen::Map<Eigen::VectorXd> init,
                         const SEXP & EM, const IntegerVector & obs) {
  MSM tmat(as<MSM> (TM));
  const MSM2 emat(as<MSM2> (EM));
  SparseVector<double> svector;
  VectorXd init2;

  double isum, output = 0;
  int i = 0;

  values = transmatrix.rightCols(1) + transmatrix.leftCols(rparams.size()) * rparams ;

  if (IntegerVector::is_na(obs(0))) {
    for (init2 = init; (i < obs.length()) && (IntegerVector::is_na(obs(i))); ++i)
      init2 = tmat.transpose() * init2;
    if (i == obs.length())
      return 0;
    svector = emat.col(obs(i)).cwiseProduct(init2);
  }
  else
    svector = emat.col(obs(0)).cwiseProduct(init);

  isum = svector.sum();
  output += log(isum);
  svector = svector / isum;

  for (++i; i < obs.length(); ++i) {
    svector = tmat.transpose() * svector;
    if (!IntegerVector::is_na(obs(i))) {
      svector = svector.cwiseProduct(emat.col(obs(i)));
      isum = svector.sum();
      output += log(isum);
      svector = svector / isum;
    }
  }

  return(-output);
}

// [[Rcpp::export]]
double fmultiloglik(SEXP & TM, Eigen::Map<Eigen::VectorXd> values, const Eigen::Map<Eigen::VectorXd> rparams,
               const Eigen::Map<Eigen::MatrixXd> transmatrix, const SEXP & init,
               const SEXP & EM, const IntegerMatrix & obs) {
  MSM tmat(as<MSM> (TM));
  const MSM2 emat(as<MSM2> (EM));
  const MSM2 smatinit(as<MSM2> (init));
  SparseMatrix<double> smat(smatinit.rows(), smatinit.cols());
  VectorXd isum;

  double output = 0;
  int i = 0, j;

  for (j = 0; j < obs.nrow(); ++j) {
    if (IntegerVector::is_na(obs(j, i)))
      smat.col(j) = VectorXd::Ones(smat.rows()).cwiseProduct(smatinit.col(j));
    else
      smat.col(j) = emat.col(obs(j,i)).cwiseProduct(smatinit.col(j));
  }

  values = transmatrix.rightCols(1) + transmatrix.leftCols(rparams.size()) * rparams ;

  isum = VectorXd::Ones(smat.cols()) * smat;
  output += isum.log().sum();
  for (j = 0; j < obs.nrow(); ++j)
    smat.col(j) = smat.col(j) / isum(j);

  for (++i; i < obs.ncol(); ++i) {
    smat = tmat.transpose() * smat;
    for (j = 0; j < obs.nrow(); ++j)
      if (!IntegerVector::is_na(obs(j, i)))
        smat.col(j) = emat.col(obs(j, i)).cwiseProduct(emat.col(j));

    isum = VectorXd::Ones(smat.cols()) * smat;
    output += isum.log().sum();
    for (j = 0; j < obs.nrow(); ++j)
      smat.col(j) = smat.col(j) / isum(j);
  }

  return(-output);
}


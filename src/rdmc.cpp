/*
 * Author: Andreas Alfons
 *         Erasmus Universiteit Rotterdam
 * 
 * based on R code by Aurore Archimbaud
 */

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>


// For now, there are multiple copies of the function where the only difference
// is how the loss function is computed in step 2 (specifically, update L for 
// cells with observed values in X). This is because the loss function is 
// computed for every observed cell in every iteration, so we should avoid if 
// statements inside those loops. We should look into object-oriented 
// implementations using class templates, which could be an efficient way of 
// handling the loss function inside the loops. That is, we should have 
// different classes for the different loss function, with a method that 
// computes the loss for a given cell.


// -----------------
// pseudo-Huber loss
// -----------------

// loss function
double pseudo_huber(const double& x, const double& delta) {
  return std::pow(delta, 2.0) * (sqrt(1 + std::pow(x/delta, 2.0)) - 1);
}

// workhorse function for a single value of the regularization parameter lambda
void rdmc_pseudo_huber(const arma::mat& X, const arma::uword& n, 
                       const arma::uword& p, const arma::umat& idx_NA, 
                       const arma::umat& idx_observed, const arma::mat& values, 
                       const double& lambda, const arma::uword& rank_max, 
                       const std::string& type, const double& svd_tol, 
                       const double& loss_const, const double& delta, 
                       double mu, const double& conv_tol, 
                       const int& max_iter,
                       // output to be returned through arguments
                       arma::mat& L, arma::mat& Z, arma::mat& Theta,
                       double& objective, bool& converged, 
                       int& nb_iter) {
  
  Rprintf("\nlambda = %f\n", lambda);
  
  // initializations
  objective = R_PosInf;
  converged = false;
  nb_iter = 0;
  
  // iterate update steps of Z, L, and Theta
  arma::uword i, j, k, l, nb_values = values.n_rows, rank, which_min;
  arma::mat U, V, L_minus_Z;
  arma::vec d;
  arma::uword nb_elem_change;
  double nuclear_norm, tmp, objective_step2, objective_step2_min, values_min, 
  loss_norm, loss, loss_min, previous_objective, change;
  while (!converged && nb_iter < max_iter) {
    
    // step 1: update Z keeping L fixed
    // soft-thresholding of L + 1/mu * Theta
    
    // compute SVD
    // TODO: add option to use SVD-ALS instead of the SVD
    arma::svd(U, d, V, L + Theta/mu);
    // adjust singular values for our parametrization
    d -= lambda / mu;
    // compute rank of soft-thresholded singular values
    j = 0;
    rank = 0;
    nuclear_norm = 0;
    while ((j < d.n_elem) & (rank < rank_max)) {
      if (d(j) > svd_tol) {
        rank++;
        nuclear_norm += d(j);
      }
      j++;
    }
    // initialize Z with zeros
    Z.zeros();
    // if we have a nonzero soft-thresholded singular value, update Z from
    // the soft-thresholded SVD
    if (rank > 0) {
      // efficient implementation of matrix multiplications without copying parts
      for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
          // add contributions from the different rows of U and columns of V'
          for (k = 0; k < rank; k++) Z(i, j) += U(i, k) * d(k) * V(j, k);
        }
      }
    }
    
    // step 2: update L keeping Z fixed
    // separable problem for missing and observed values in X
    nb_elem_change = 0;
    
    // update L for cells with missing values in X
    for (l = 0; l < idx_NA.n_rows; l++) {
      // row and column indices of current cell to be updated
      i = idx_NA(l, 0);
      j = idx_NA(l, 1);
      // save some computation time in loop
      tmp = -Z(i, j) + Theta(i, j)/mu;
      // initialize the minimum
      which_min = 0;
      objective_step2_min = R_PosInf;
      // loop over the different values and choose the one that minimizes the
      // objective function
      for (k = 0; k < nb_values; k++) {
        // The paper says to take the argmin of the squared expression. But
        // this is equivalent to taking the argmin of the absolute value,
        // which is faster to compute.
        objective_step2 = std::abs(values(k, j) + tmp);
        if (objective_step2 < objective_step2_min) {
          which_min = k;
          objective_step2_min = objective_step2;
        }
      }
      // update the element of L with the argmin of the objective function
      values_min = values(which_min, j);
      if (L(i, j) != values_min) {
        L(i, j) = values_min;
        nb_elem_change++;
      }
    }
    
    // update L for cells with observed values in X
    loss_norm = 0;
    for (l = 0; l < idx_observed.n_rows; l++) {
      // row and column indices of current cell to be updated
      i = idx_observed(l, 0);
      j = idx_observed(l, 1);
      // save some computation time in loop
      tmp = -Z(i, j) + Theta(i, j)/mu;
      // initialize the minimum
      which_min = 0;
      objective_step2_min = R_PosInf;
      // loop over the different values and choose the one that minimizes the
      // objective function
      for (k = 0; k < nb_values; k++) {
        loss = pseudo_huber(values(k, j) - X(i, j), loss_const);
        objective_step2 = loss + mu * std::pow(values(k, j) + tmp, 2.0)/2.0;
        if (objective_step2 < objective_step2_min) {
          which_min = k;
          loss_min = loss;
          objective_step2_min = objective_step2;
        }
      }
      // update the element of L with the argmin of the objective function
      // values_min = values(which_min, j);
      // if (L(i, j) != values_min) {
      //   L(i, j) = values_min;
      //   nb_elem_change++;
      // }
      L(i, j) = values(which_min, j);
      // update the norm given by loss function
      loss_norm += loss_min;
    }
    
    // step 3: update Lagrange multiplier Theta and parameter mu
    L_minus_Z = L - Z;
    Theta = Theta + mu * L_minus_Z;
    mu = delta * mu;
    
    // update iteration counter
    nb_iter++;
    // update objective function for convergence criterion
    previous_objective = objective;
    objective = loss_norm + lambda * nuclear_norm + arma::dot(Theta, L_minus_Z) +
      mu * arma::norm(L_minus_Z, "fro") / 2.0;
    // compute relative change and check convergence
    if (nb_iter > 1) {
      // we can't compute relative change in the first iteration since the
      // objective function is initialized with infinity
      change = std::abs((objective - previous_objective) / previous_objective);
      converged = change < conv_tol;
    }
    
    // print information on current iteration
    Rprintf("iteration = %d, nb_elem_change = %d, obj_change = %f\n", 
            nb_iter, nb_elem_change, change);
    
  }
  
}


// -------------
// absolute loss
// -------------

// workhorse function for a single value of the regularization parameter lambda
void rdmc_absolute(const arma::mat& X, const arma::uword& n, 
                   const arma::uword& p, const arma::umat& idx_NA, 
                   const arma::umat& idx_observed, const arma::mat& values, 
                   const double& lambda, const arma::uword& rank_max, 
                   const std::string& type, const double& svd_tol, 
                   const double& delta, double mu, const double& conv_tol, 
                   const int& max_iter,
                   // output to be returned through arguments
                   arma::mat& L, arma::mat& Z, arma::mat& Theta,
                   double& objective, bool& converged, 
                   int& nb_iter) {
  
  // initializations
  objective = R_PosInf;
  converged = false;
  nb_iter = 0;
  
  // iterate update steps of Z, L, and Theta
  arma::uword i, j, k, l, nb_values = values.n_rows, rank, which_min;
  arma::mat U, V, L_minus_Z;
  arma::vec d;
  double nuclear_norm, tmp, objective_step2, objective_step2_min,
  loss_norm, loss, loss_min, previous_objective, change;
  while (!converged && nb_iter < max_iter) {
    
    // step 1: update Z keeping L fixed
    // soft-thresholding of L + 1/mu * Theta
    
    // compute SVD
    // TODO: add option to use SVD-ALS instead of the SVD
    arma::svd(U, d, V, L + Theta/mu);
    // adjust singular values for our parametrization
    d -= lambda / mu;
    // compute rank of soft-thresholded singular values
    j = 0;
    rank = 0;
    nuclear_norm = 0;
    while ((j < d.n_elem) & (rank < rank_max)) {
      if (d(j) > svd_tol) {
        rank++;
        nuclear_norm += d(j);
      }
      j++;
    }
    // initialize Z with zeros
    Z.zeros();
    // if we have a nonzero soft-thresholded singular value, update Z from
    // the soft-thresholded SVD
    if (rank > 0) {
      // efficient implementation of matrix multiplications without copying parts
      for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
          // add contributions from the different rows of U and columns of V'
          for (k = 0; k < rank; k++) Z(i, j) += U(i, k) * d(k) * V(j, k);
        }
      }
    }
    
    // step 2: update L keeping Z fixed
    // separable problem for missing and observed values in X
    
    // update L for cells with missing values in X
    for (l = 0; l < idx_NA.n_rows; l++) {
      // row and column indices of current cell to be updated
      i = idx_NA(l, 0);
      j = idx_NA(l, 1);
      // save some computation time in loop
      tmp = -Z(i, j) + Theta(i, j)/mu;
      // initialize the minimum
      which_min = 0;
      objective_step2_min = R_PosInf;
      // loop over the different values and choose the one that minimizes the
      // objective function
      for (k = 0; k < nb_values; k++) {
        // The paper says to take the argmin of the squared expression. But
        // this is equivalent to taking the argmin of the absolute value,
        // which is faster to compute.
        objective_step2 = std::abs(values(k, j) + tmp);
        if (objective_step2 < objective_step2_min) {
          which_min = k;
          objective_step2_min = objective_step2;
        }
      }
      // update the element of L with the argmin of the objective function
      L(i, j) = values(which_min, j);
    }
    
    // update L for cells with observed values in X
    loss_norm = 0;
    for (l = 0; l < idx_observed.n_rows; l++) {
      // row and column indices of current cell to be updated
      i = idx_observed(l, 0);
      j = idx_observed(l, 1);
      // save some computation time in loop
      tmp = -Z(i, j) + Theta(i, j)/mu;
      // initialize the minimum
      which_min = 0;
      objective_step2_min = R_PosInf;
      // loop over the different values and choose the one that minimizes the
      // objective function
      for (k = 0; k < nb_values; k++) {
        loss = std::abs(values(k, j) - X(i, j));
        objective_step2 = loss + mu * std::pow(values(k, j) + tmp, 2.0)/2.0;
        if (objective_step2 < objective_step2_min) {
          which_min = k;
          loss_min = loss;
          objective_step2_min = objective_step2;
        }
      }
      // update the element of L with the argmin of the objective function
      L(i, j) = values(which_min, j);
      // update the norm given by loss function
      loss_norm += loss_min;
    }
    
    // step 3: update Lagrange multiplier Theta and parameter mu
    L_minus_Z = L - Z;
    Theta = Theta + mu * L_minus_Z;
    mu = delta * mu;
    
    // update iteration counter
    nb_iter++;
    // update objective function for convergence criterion
    previous_objective = objective;
    objective = loss_norm + lambda * nuclear_norm + arma::dot(Theta, L_minus_Z) +
      mu * arma::norm(L_minus_Z, "fro") / 2.0;
    // compute relative change and check convergence
    if (nb_iter > 1) {
      // we can't compute relative change in the first iteration since the
      // objective function is initialized with infinity
      change = std::abs((objective - previous_objective) / previous_objective);
      converged = change < conv_tol;
    }
    
  }
  
}


// ---------------------
// bounded absolute loss
// ---------------------

// loss function
double bounded(const double& x, const double& bound) {
  return std::min(std::abs(x), bound);
}

// workhorse function for a single value of the regularization parameter lambda
void rdmc_bounded(const arma::mat& X, const arma::uword& n, 
                  const arma::uword& p, const arma::umat& idx_NA, 
                  const arma::umat& idx_observed, const arma::mat& values, 
                  const double& lambda, const arma::uword& rank_max, 
                  const std::string& type, const double& svd_tol, 
                  const double& loss_const, const double& delta, 
                  double mu, const double& conv_tol, 
                  const int& max_iter,
                  // output to be returned through arguments (passed on to R)
                  arma::mat& L, arma::mat& Z, arma::mat& Theta,
                  double& objective, bool& converged, 
                  int& nb_iter) {
  
  // initializations
  objective = R_PosInf;
  converged = false;
  nb_iter = 0;
  
  // iterate update steps of Z, L, and Theta
  arma::uword i, j, k, l, nb_values = values.n_rows, rank, which_min;
  arma::mat U, V, L_minus_Z;
  arma::vec d;
  double nuclear_norm, tmp, objective_step2, objective_step2_min,
  loss_norm, loss, loss_min, previous_objective, change;
  while (!converged && nb_iter < max_iter) {
    
    // step 1: update Z keeping L fixed
    // soft-thresholding of L + 1/mu * Theta
    
    // compute SVD
    // TODO: add option to use SVD-ALS instead of the SVD
    arma::svd(U, d, V, L + Theta/mu);
    // adjust singular values for our parametrization
    d -= lambda / mu;
    // compute rank of soft-thresholded singular values
    j = 0;
    rank = 0;
    nuclear_norm = 0;
    while ((j < d.n_elem) & (rank < rank_max)) {
      if (d(j) > svd_tol) {
        rank++;
        nuclear_norm += d(j);
      }
      j++;
    }
    // initialize Z with zeros
    Z.zeros();
    // if we have a nonzero soft-thresholded singular value, update Z from
    // the soft-thresholded SVD
    if (rank > 0) {
      // efficient implementation of matrix multiplications without copying parts
      for (i = 0; i < n; i++) {
        for (j = 0; j < p; j++) {
          // add contributions from the different rows of U and columns of V'
          for (k = 0; k < rank; k++) Z(i, j) += U(i, k) * d(k) * V(j, k);
        }
      }
    }
    
    // step 2: update L keeping Z fixed
    // separable problem for missing and observed values in X
    
    // update L for cells with missing values in X
    for (l = 0; l < idx_NA.n_rows; l++) {
      // row and column indices of current cell to be updated
      i = idx_NA(l, 0);
      j = idx_NA(l, 1);
      // save some computation time in loop
      tmp = -Z(i, j) + Theta(i, j)/mu;
      // initialize the minimum
      which_min = 0;
      objective_step2_min = R_PosInf;
      // loop over the different values and choose the one that minimizes the
      // objective function
      for (k = 0; k < nb_values; k++) {
        // The paper says to take the argmin of the squared expression. But
        // this is equivalent to taking the argmin of the absolute value,
        // which is faster to compute.
        objective_step2 = std::abs(values(k, j) + tmp);
        if (objective_step2 < objective_step2_min) {
          which_min = k;
          objective_step2_min = objective_step2;
        }
      }
      // update the element of L with the argmin of the objective function
      L(i, j) = values(which_min, j);
    }
    
    // update L for cells with observed values in X
    loss_norm = 0;
    for (l = 0; l < idx_observed.n_rows; l++) {
      // row and column indices of current cell to be updated
      i = idx_observed(l, 0);
      j = idx_observed(l, 1);
      // save some computation time in loop
      tmp = -Z(i, j) + Theta(i, j)/mu;
      // initialize the minimum
      which_min = 0;
      objective_step2_min = R_PosInf;
      // loop over the different values and choose the one that minimizes the
      // objective function
      for (k = 0; k < nb_values; k++) {
        loss = bounded(values(k, j) - X(i, j), loss_const);
        objective_step2 = loss + mu * std::pow(values(k, j) + tmp, 2.0)/2.0;
        if (objective_step2 < objective_step2_min) {
          which_min = k;
          loss_min = loss;
          objective_step2_min = objective_step2;
        }
      }
      // update the element of L with the argmin of the objective function
      L(i, j) = values(which_min, j);
      // update the norm given by loss function
      loss_norm += loss_min;
    }
    
    // step 3: update Lagrange multiplier Theta and parameter mu
    L_minus_Z = L - Z;
    Theta = Theta + mu * L_minus_Z;
    mu = delta * mu;
    
    // update iteration counter
    nb_iter++;
    // update objective function for convergence criterion
    previous_objective = objective;
    objective = loss_norm + lambda * nuclear_norm + arma::dot(Theta, L_minus_Z) +
      mu * arma::norm(L_minus_Z, "fro") / 2.0;
    // compute relative change and check convergence
    if (nb_iter > 1) {
      // we can't compute relative change in the first iteration since the
      // objective function is initialized with infinity
      change = std::abs((objective - previous_objective) / previous_objective);
      converged = change < conv_tol;
    }
    
  }
  
}


// ----------------------------
// function to be called from R
// ----------------------------

// [[Rcpp::export]]
Rcpp::List rdmc_cpp(const arma::mat& X, 
                    const arma::umat& idx_NA,
                    const arma::umat& idx_observed, 
                    const arma::mat& values, 
                    const Rcpp::NumericVector& lambda, 
                    const double& d_max,
                    const arma::uword& rank_max, 
                    const std::string& type, 
                    const double& svd_tol, 
                    const std::string& loss, 
                    const double& loss_const, 
                    const double& delta, 
                    double mu, 
                    const double& conv_tol, 
                    const int& max_iter, 
                    arma::mat L, arma::mat Theta) {
  
  // extract number of rows and columns
  arma::uword n = X.n_rows, p = X.n_cols, nb_lambda = lambda.length();
  
  // initialize Z matrix and variables related to convergence
  // (to be updated by workhorse function)
  arma::mat Z(n, p);
  double objective;
  bool converged;
  int nb_iter;

  // different behavior depending on whether we have one value of the
  // regularization parameter lambda or multiple values
  if (nb_lambda == 1) {
    
    // call workhorse function with initial values
    if (loss == "pseudo_huber") {
      rdmc_pseudo_huber(X, n, p, idx_NA, idx_observed, values, 
                        lambda(0) * d_max, rank_max, type, svd_tol, 
                        loss_const, delta, mu, conv_tol, max_iter, 
                        L, Z, Theta, objective, converged, nb_iter);
    } else if (loss == "absolute") {
      rdmc_absolute(X, n, p, idx_NA, idx_observed, values, lambda(0) * d_max,
                    rank_max, type, svd_tol, delta, mu, conv_tol, max_iter,
                    L, Z, Theta, objective, converged, nb_iter);
    } else if (loss == "bounded") {
      rdmc_bounded(X, n, p, idx_NA, idx_observed, values, lambda(0) * d_max,
                   rank_max, type, svd_tol, loss_const, delta, mu,
                   conv_tol, max_iter, L, Z, Theta, objective,
                   converged, nb_iter);
    } else Rcpp::stop("loss function not implemented");  // shouldn't happen
    // return list of results
    return Rcpp::List::create(Rcpp::Named("lambda") = lambda(0),
                              Rcpp::Named("d_max") = d_max,
                              Rcpp::Named("L") = L,
                              Rcpp::Named("Z") = Z,
                              Rcpp::Named("Theta") = Theta,
                              Rcpp::Named("objective") = objective,
                              Rcpp::Named("converged") = converged,
                              Rcpp::Named("nb_iter") = nb_iter);
    
  } else {
    
    // loop over values of the regularization parameter lambda
    Rcpp::List L_list, Z_list, Theta_list;
    Rcpp::NumericVector objective_vec(nb_lambda);
    Rcpp::LogicalVector converged_vec(nb_lambda);
    Rcpp::IntegerVector nb_iter_vec(nb_lambda);
    for (arma::uword l = 0; l < nb_lambda; l++) {
      // call workhorse function with starting values: note that solutions
      // for previous value of lambda are used as starting values
      if (loss == "pseudo_huber") {
        rdmc_pseudo_huber(X, n, p, idx_NA, idx_observed, values, 
                          lambda(l) * d_max, rank_max, type, svd_tol, 
                          loss_const, delta, mu, conv_tol, max_iter, 
                          L, Z, Theta, objective, converged, nb_iter);
      } else if (loss == "absolute") {
        rdmc_absolute(X, n, p, idx_NA, idx_observed, values, lambda(l) * d_max,
                      rank_max, type, svd_tol, delta, mu, conv_tol, max_iter,
                      L, Z, Theta, objective, converged, nb_iter);
      } else if (loss == "bounded") {
        rdmc_bounded(X, n, p, idx_NA, idx_observed, values, lambda(l) * d_max,
                     rank_max, type, svd_tol, loss_const, delta, mu,
                     conv_tol, max_iter, L, Z, Theta, objective,
                     converged, nb_iter);
      } else Rcpp::stop("loss function not implemented");  // shouldn't happen
      // add results for current value of the regularization parameter:
      // note that a copy of the objects that are stored in the list so that 
      // they are not modified in future calls to the workhorse functions
      L_list.push_back(L);
      Z_list.push_back(Z);
      Theta_list.push_back(Theta);
      objective_vec(l) = objective;
      converged_vec(l) = converged;
      nb_iter_vec(l) = nb_iter;
      
    }
    // return list of results
    return Rcpp::List::create(
      Rcpp::Named("lambda") = lambda,
      Rcpp::Named("d_max") = d_max,
      Rcpp::Named("L") = L_list,
      Rcpp::Named("Z") = Z_list,
      Rcpp::Named("Theta") = Theta_list,
      Rcpp::Named("objective") = objective_vec,
      Rcpp::Named("converged") = converged_vec,
      Rcpp::Named("nb_iter") = nb_iter_vec
    );
    
  }
  
}

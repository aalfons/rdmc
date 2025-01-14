# ************************************
# Author: Andreas Alfons
#         Erasmus University Rotterdam
# ************************************


#' Extract the imputed data matrix
#' 
#' Extract the imputed data matrix from an object returned by a matrix 
#' completion algorithm.
#' 
#' @param object  an object returned by a matrix completion algorithm.
#' @param discretized  a logical indicating if the imputed data matrix with or 
#' without the discretization step should be extracted.
#' @param \dots  additional arguments to be passed down to methods.
#' 
#' @return  The imputed data matrix.
#' 
#' @author Andreas Alfons
#' 
#' @seealso \code{\link{rdmc_tune}()}, \code{\link{soft_impute_tune}()}, 
#' \code{\link{median_impute}()}, \code{\link{mode_impute}()}
#' 
#' @keywords utilities
#' 
#' @export

get_X <- function(object, ...) UseMethod("get_X")


#' @rdname get_X
#' @export
get_X.rdmc_tuned <- function(object, ...) {
  object$fit$X
}

#' @rdname get_X
#' @export
get_X.soft_impute_tuned <- function(object, discretized = FALSE, ...) {
  if (isTRUE(discretized)) object$fit$X_discretized
  else object$fit$X
}

#' @rdname get_X
#' @export
get_X.median_impute <- function(object, discretized = FALSE, ...) {
  if (isTRUE(discretized)) object$X_discretized
  else object$X
}

#' @rdname get_X
#' @export
get_X.mode_impute <- function(object, ...) {
  object$X
}


#' Extract the optimal value of the regularization parameter
#' 
#' Extract the optimal value of the regularization parameter from an object 
#' returned by a matrix completion algorithm.
#' 
#' @param object  an object returned by a matrix completion algorithm with a 
#' regularization parameter.
#' @param relative  logical; in case the values of the regularization parameter 
#' were given relative to a certain reference value computed from the data at 
#' hand, this allows to return the optimal value before or after multiplication 
#' with that reference value.
#' @param \dots  additional arguments to be passed down to methods.
#' 
#' @return  The optimal value of the regularization parameter.
#' 
#' @author Andreas Alfons
#' 
#' @seealso \code{\link{rdmc_tune}()}, \code{\link{soft_impute_tune}()}
#' 
#' @keywords utilities
#' 
#' @export

get_lambda <- function(object, ...) UseMethod("get_lambda")


#' @rdname get_lambda
#' @export
get_lambda.rdmc_tuned <- function(object, ...) {
  get_lambda(object$fit, ...)
}

#' @rdname get_lambda
#' @export
get_lambda.rdmc <- function(object, relative = TRUE, ...) {
  lambda <- object$lambda
  if (isTRUE(relative)) lambda 
  else lambda * object$d_max
}

#' @rdname get_lambda
#' @export
get_lambda.soft_impute_tuned <- function(object, ...) {
  get_lambda(object$fit, ...)
}

#' @rdname get_lambda
#' @export
get_lambda.soft_impute <- function(object, relative = TRUE, ...) {
  lambda <- object$lambda
  if (isTRUE(relative)) lambda 
  else lambda * object$lambda0
}


#' Extract the number of iterations
#' 
#' Extract the number of iterations from an object returned by a matrix 
#' completion algorithm.
#' 
#' @param object  an object returned by an iterative matrix completion 
#' algorithm.
#' @param \dots  currently ignored.
#' 
#' @return  The number of iterations performed in the iterative algorithm.
#' 
#' @author Andreas Alfons
#' 
#' @seealso \code{\link{rdmc_tune}()}
#' 
#' @keywords utilities
#' 
#' @export

get_nb_iter <- function(object, ...) UseMethod("get_nb_iter")


#' @rdname get_nb_iter
#' @export
get_nb_iter.rdmc_tuned <- function(object, ...) {
  object$fit$nb_iter
}

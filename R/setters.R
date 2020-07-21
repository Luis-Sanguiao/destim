#' Set reduced parameters
#'
#' Sets the paramaters selected by minparams function.
#'
#' The function minparams selects a minimal set of parameters, that fully determine the
#' transition probabilities. This function sets those parameters and recalculates all
#' transition probabilities from them.
#'
#' The model is initialized with initparams and minparams when required.
#'
#' @param x A HMM model.
#' @param value A numeric vector with the new parameters.
#'
#' @return Changes parameters$reducedparams$params and parameters$transitions in the object x.
#'
#' @seealso \link{minparams}, \link{rparams}, \link{initparams}
#'
#' @examples
#' model <- HMMrectangle(3,3)
#' rparams(model)<-c(0.3, 0.03)
#' ptransition(model)
#'
#' @export
`rparams<-` <- function(x, value) {
  if (is.null(x$parameters$reducedparams)) {
    if (is.null(x$parameters$states))
      x <- initparams(x)
    x <- minparams(x)
  }
  x$parameters$reducedparams$params <- value
  x$parameters$transitions <-
    (x$parameters$reducedparams$transmatrix %*%
      matrix(c(x$parameters$reducedparams$params, 1), ncol = 1)) [,]
  if (length(which(x$parameters$transitions < 0)) > 0) {
    warning(paste0("Parameters outside the feasible region. Negative transition ",
                   "probabilities will be set to zero."))
    x$parameters$transitions[which(x$parameters$transitions < 0)] <- 0
  }
  return(x)
}

#' Set the names of the states.
#'
#' Sets the names of the states.
#'
#' The length of the character vector must match the number of states of the model.
#'
#' @param x A HMM model.
#' @param value A character vector with the names.
#'
#' @return Changes states names in the object x.
#'
#' @seealso \link{HMM}
#'
#' @examples
#' model <- HMM(3)
#' setsnames(model) <- c("a","b","c")
#' model$states$names
#'
#' @export
`setsnames<-` <- function(x,value) {
  if (length(value) != nstates(x))
    stop("Wrong number of states")
  x$states$names <- value
  return(x)
}

#' Set the emissions of the model
#'
#' Sets the emissions of the model.
#'
#' The number of rows must match the number of states. If a matrix is provided, it is converted
#' to column major sparse matrix (dgCMatrix).
#'
#' @param x A HMM model.
#' @param value A (sparse column major) matrix with the likelihoods of each emission (column)
#' conditioned on the state (row).
#'
#' @return Changes the emissions matrix in the model.
#'
#' @seealso \link{HMM}, \link{emissions}
#'
#' @examples
#' model <- HMMrectangle(10,10)
#' tws <- matrix(c(3.2, 6.1, 2.2, 5.7, 5.9, 9.3, 5.4,
#' 4.0, 2.9, 8.6, 6.9, 6.2, 9.7, 1.3),
#' nrow = 2, ncol = 7)
#' S <- function(x) if (x > 5) return(0) else return(20*log(5/x))
#' emissions(model)<-createEM(c(10,10), tws, S)
#' emissions(model)
#'
#' @export
`emissions<-` <- function(x, value) {
  if (is.matrix(value))
    value <- as(value, "dgCMatrix")
  if (!is(value,"dgCMatrix"))
    stop("The value is suposed to be of class dgCMatrix.")
  if (nrow(value) != nstates(x))
    stop("The number of rows does not match the number of states.")
  x$emissions <- value
  return(x)
}

#' Set the initial states
#'
#' Sets the initial distribution of states.
#'
#' The length must match the number of states, and the sum of the vector must be one.
#'
#' @param x A HMM model.
#' @param value A numeric vector with a probability for each state that represents the
#' initial distribution of states.
#'
#' @return Changes the initial distribution of states in the model.
#'
#' @seealso \link{HMM}, \link{initparams}, \link{initsteady}, \link{fit}
#'
#' @examples
#' model <- HMMrectangle(3,3)
#' model <- initparams(model)
#' istates(model)
#' istates(model) <- (1:9) / sum(1:9)
#' istates(model)
#'
#' @export
`istates<-` <- function(x,value) {
  if (length(value) != nstates(x))
    stop("Wrong number of states")
  x$parameters$states <- value
  return(x)
}

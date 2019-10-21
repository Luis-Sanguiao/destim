#' Adds a transition to the model
#'
addtransition <- function(x,t) {
  if (class(x) != "HMM")
    stop("This function only works with HMM objects.")
  TL <- x[["transitions"]]
  CT <- x[["constraints"]]
  CTrow <- which(sapply(1:nrow(CT), function(z) {
                return(all(as.numeric(TL[1,] == t[1]) ==
                             CT[z, -ncol(CT)]))
    }))
  TL <- cbind(matrix(t, ncol = 1), TL)
  CT <- cbind(matrix(0,nrow = nrow(CT), ncol = 1), CT)
  CT[CTrow, 1] <- 1
  x[["transitions"]] <- TL
  x[["constraints"]] <- CT
  x[["parameters"]] <- list(transitions = NULL, states = NULL)
  return(x)
}
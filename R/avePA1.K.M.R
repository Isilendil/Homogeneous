
# 2015.03.25
# avePA1: Average online passive-aggressive algorithm
#----------------------------------------------------
# Input:
#   Y: the vector of labels
#   K: precomputed kernel for all the example, i.e., K_{ij} = K(x_i, x_j)
#   id.list: a randomized ID list
#   options: a struct containing rho, sigma, C, n.label and n.tick;
# Output:
#   err.count: total number of training errors
#   run.time: time consumed by this algorithm once
#   mistakes: a vector of mistake rate
#   mistake.idx: a vector of number, in which every number corresponds to a mistake rate in the vector above
#   SVs: a vector records the number of support vectors
#   size.SV: the size of final support set
#------------------------------------------------------

avePA1.K.M <- function(Y, K, options, id.list)
{
  # initialize parameters
  C <- options$C # 1 by default
  t.tick <- options$t.tick
  alpha <- vector()
  ave.alpha <- vector()
  SV <- vector()
  ID <- id.list
  err.count <- 0
  mistakes <- vector()
  mistakes.idx <- vector()
  SVs <- vector()
  TMs <- vector()
  
  # loop
  t1 <- proc.time()
  
  for (t in 1 : (length(ID)))
  {
    id <- ID[t]
    if (length(alpha) == 0)
    {
      f.t <- 0
    }
    else
    {
      k.t <- K[id, SV]
      f.t <- alpha %*% k.t  # decision function
    }
    loss.t <- max(0, 1 - Y[id]*f.t)  # hinge loss
    hat.y.t <- sign(f.t)  # prediction
    if (hat.y.t == 0)
    {
      hat.y.t <- 1
    }
    
    # count accumulative mistakes
    if (hat.y.t ~= Y[id])
    {
      err.count <- err.count + 1
    }
    
    if (loss.t > 0)
    {
      # update
      s.t <- K(id, id)
      gamma.t <- min(C, loss.t / s.t)
      alpha <- c(alpha, Y[id]*gamma.t)
      SV <- c(SV, id)
      
      ave.alpha <- c(ave.alpha, 0)
      ave.alpha <- ((t-1)/t) * ave.alpha + (1/t) * alpha
    }
    else
    {
      ave.alpha <- ((t-1)/t) * ave.alpha + (1/t) * alpha
    }
    
    t2 <- proc.time()
    run.time <- t2[3] - t1[3]
    if ((t%%t.tick) == 0)
    {
      mistakes <- c(mistakes, err.count/t)
      mistakes.idx <- c(mistakes.idx, t)
      SVs <- c(SVs, length(SV))
      TMs <- c(TMs, run.time)
    }
  }
  
  classifier <- list(SV = SV, alpha = ave.alpha)
  
  model <- list(classifier = classifier, err.count = err.count, mistakes = mistakes, mistakes.idx = mistakes.idx,
                SVs = SVs, TMs = TMs)
  
  print(paste('The number of mistakes =', err.count))
  
  t3 <- proc.time()
  run.time <- t3[3] - t1[3]
  
  model$run.time <- run.time
  
  return(model)
  
}

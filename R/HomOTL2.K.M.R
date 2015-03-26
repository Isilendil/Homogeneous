
# 2015.03.26
# HomOTL2: HomOTL-II
#-----------------------------------
# Input:
#   Y: the vector of labels
#   K: precomputed kernel for all the example, i.e., K_{ij} = K(x_i, x_j)
#   id.list : a randomized ID list
#   options: a struct containing rho, sigma, C, n.label and t.tick
# Output:
#   err.count: total nubmer of training errors
#   run.time: time consumed by this algorithm once
#   mistakes: a vector of mistake rate
#   mistake.idx: a vector of number, in which every number corresponds to a mistake rate in the vector above
#   SVs: a vector records the number of support vectors
#   size.SV: the size of final support set
#-------------------------------------------

HomOTLF.K.M <- function(Y, K1, K2, options, id.list, classifier)
{
  beta <- options$beta
  number.Old <- options$number.Old
  C <- options$C  # 1 by default
  t.tick.const <- options$t.tick
  alpha1 <- classifier.alpha
  SV1 <- classifier.SV
  alpha2 <- vector()
  SV2 <- vector()
  ID <- id.list
  err.count = 0
  mistakes <- vector()
  mistakes.idx <- vector()
  SVs <- vector()
  TMs <- vector()
  w.1t <- 1/2
  w.2t <- 1/2
  
  t.tick <- t.tick.const
  
  # loop
  t1 <- proc.time()
  
  for (t in 1 : (length(ID)) )
  {
    id <- ID[t]
    if (length(alpha1) == 0)  # init stage
    {
      f1.t <- 0
    }
    else
    {
      k1.t <- K1[id, SV1]
      f1.t <- alpha1 * k1.t  # decision function
    }
    
    id2 <- id - number.Old
    if (length(alpha2) == 0)
    {
      f2.t <- 0
    }
    else
    {
      k2.t <- K2[id2, SV2]
      f2.t <- alpha2 * k2.t 
    }
    
    f.t <- w.1t * sign(f1.t) + w.2t * sign(f2.t)
    hat.y.t <- sign(f.t)   # prediction
    if (hat.y.t == 0)
    {
      hat.y.t <- 1
    }
    
    # count accumulative mistakes
    if (hat.y.t != Y[id])
    {
      err.count <- err.count + 1
    }
    
    z1 <- ((Y[id]*f1.t) <= 0)
    z2 <- ((Y[id]*f2.t) <= 0)
    w.1t <- w.1t * (beta^z1)
    w.2t <- w.2t * (beta^z2)
    sum.w <- w.1t + w.2t
    w.1t <- w.1t / sum.w
    w.2t <- w.2t / sum.w
    
    loss2.t <- max(0, 1 - Y[id]*f2.t)  # hinge loss
    if (loss2.t > 0)
    {
      # update
      s2.t <- K2[id2, id2]
      gamma.t <- min(C, loss2.t/s2.t)
      alpha2 <- c(alpha2, Y[id] * gamma.t)
      SV2 <- c(SV2, id2)
    }
    
    t2 <- proc.time()
    run.time <- t2[3] - t1[3]
    
    if (t < t.tick.const)
    {
      if (t == t.tick)
      {
        mistakes <- c(mistakes, err.count/t)
        mistakes.idx <- c(mistakes.idx, t)
        SVs <- c(SVs, length(SV1) + length(SV2))
        TMs <- c(TMs, run.time)
        
        t.tick <- 2 * t.tick
        if (t.tick >= t.tick.const)
        {
          t.tick <- t.tick.const
        }
      }
    }
    else
    {
      if ((t %% t.tick) == 0)
      {
        mistakes <- c(mistakes, err.count/t)
        mistakes.idx <- c(mistakes.idx, t)
        SVs <- c(SVs, length(SV1) + length(SV2))
        TMs <- c(TMs, run.time)
      }
    }
  }
  
  classifier$SV1 <- SV1
  classifier$SV2 <- SV2
  classifier$alpha1 <- alpha1
  classifier$alpha2 <- alpha2
  #classifier$alpha1 <- alpha2
  
  model <- list(classifier = classifier, err.count = err.count, mistakes = mistakes,
                mistakes.idx = mistakes.idx, SVs = SVs, TMs = TMs)
  
  print('The number of mistakes =', err.count)
  
  t3 <- proc.time()
  run.time <- t3[3] - t2[3]
  
  model$run.time <- rum.time
  
  return(model)
  
}
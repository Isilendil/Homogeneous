# 2015.03.23

# Experiment.OTL.K.M: the main function used to compare all the online algorithms
#--------------------------------------------------------------------------------
# Input:
#   ataset.name, name of the dataset, e.g. 'books_dvd'
#
# Output:
#   a table containing the accuracies,
#   the number of support vectors,
#   the running times of all the online learning algorithms on the inputed datasets
#   a figure for the online average accuracies of all the online learning algorithms
#   a figure for the online numbers of SVs of all the online learning algorithms
#   a figure for the online running time of all the online learning algorithms
#--------------------------------------------------------------------------------

Experiment.OTL.K.M <- function(dataset.name)
{
  
  # load dataset
  require(R.matlab)
  data.mat <- readMat(paste('../data/', dataset.name, sep = ''))
  
  size <- dim(data.mat$data)
  n <- size[1]
  d <- size[2]
  
  # set parameters
  options <- list()
  options$C <- 5
  
  # set parameters: 'sigma' (kernel width) and 't.tick' (step size for plotting figures)
  # default: sigma <- 10
  options$sigma <- 4
  options$sigma2 <- 8
  options$t.tick <- round(dim(data.mat$ID.new)[2]/15)
  
  if (dataset.name == 'usenet2')
  {
    options$sigma <- 4
    options$sigma2 <- 8
    options$t.tick <- 100
  }
  if (dataset.name == 'newsgroup4')
  {
    options$sigma <- 4
    options$sigma2 <- 8
    options$t.tick <- 100
  }
  
  m <- dim(data.mat$ID.new)[2]
  options$beta <- sqrt(m) / (sqrt(m) + sqrt(log(2)))
  options$number.Old <- n - m
  
  Y <- data.mat$data[, 1]
  X <- data.mat$data[, 2:d]
  
  # scale
  MaxX <- matrix(apply(X, 1, max), nrow = dim(X)[1])
  MinX <- matrix(apply(X, 1, min), nrow = dim(X)[1])
  DifX <- MaxX - MinX
  idx.DifNonZero <- (DifX != 0)
  DifX.2 <- matrix(1, nrow = dim(X)[1])
  DifX.2[idx.DifNonZero, ] <- DifX[idx.DifNonZero, ]
  X <- t(apply(X, 1, 
               function(x) { 
                 (x-min(x)) / (if ( (max(x)-min(x)) == 0 ) 1 else (max(x)-min(x)) ) } ))
  
  # K = X * X'
  # Gaussian Kernel Matrix 
  # K(x1, x2) = exp(- norm(x1-x2)^2 * sigma))
  print('Pre-computing kernel matrix...')
  require(kernlab)
  rbf <- rbfdot(sigma = 1/(2*options$sigma^2))
  K <- kernelMatrix(rbf, X)
  
  # K2 = X2 * X2
  X2 <- X[n-m+1:n, ]
  Y2 <- Y[n-m+1:n, ]
  rbf2 <- rbfdot(sigma = 1/(2*options$sigma2^2))
  K2 <- kernelMatrix(rbf2, X2)
  
  # learn the old classifier
  source('avePA1.K.M.R')
  model.old <- avePA1.K.M(Y, K, options, data.mat$ID.old)
  
  print(paste('The old classifier has', length(model.old$SV), 'support vectors'))
  
  # run experiments:
  nGroup <- dim(data.mat$ID.new)[1]
  nColumn <- dim(data.mat$ID.new)[2] %/% options$t.tick
  
  for (i in 1 : nGroup )
  {
    printf(paste('running on the ', i, '-th trial...', sep = '' ))
    ID <- data.mat$ID.new[i, ]
    
    # 1. PA-I
    nSV.PA1 <- vector(length = nGroup)
    err.PA1 <- vector(length = nGroup)
    time.PA1 <- vector(length = nGroup)
    mistakes.list.PA1 <- matrix(nrow = nGroup, ncol = nColumn)
    SVs.PA1 <- matrix(nrow = nGroup, ncol = nColumn)
    TMs.PA1 <- matrix(nrow = nGroup, ncol = nColumn)
    
    model.PA1 <- PA1.K.M(Y2, K2, options, ID)
    
    nSV.PA1[i] <- length(model.PA1$classifier$SV)
    err.PA1[i] <- model.PA1$err.count
    time.PA1[i] <- model.PA1$run.time
    mistakes.list.PA1[i, ] <- model.PA1$mistakes
    SVs.PA1[i, ] <- model.PA1$SVs
    TMs.PA1[i, ] <- model.PA1$TMs
  
    # 2. PAIO
    nSV.PAIO <- vector(length = nGroup)
    err.PAIO <- vector(length = nGroup)
    time.PAIO <- vector(length = nGroup)
    mistakes.list.PAIO <- matrix(nrow = nGroup, ncol = nColumn)
    SVs.PAIO <- matrix(nrow = nGroup, ncol = nColumn)
    TMs.PAIO <- matrix(nrow = nGroup, ncol = nColumn)
    
    #model.PAIO <- PAIO.K.M(Y, K, options, ID, model.old$classifier)
    
    nSV.PAIO[i] <- length(model.PAIO$classifier$SV)
    err.PAIO[i] <- model.PAIO$err.count
    time.PAIO[i] <- model.PAIO$run.time
    mistakes.list.PAIO[i, ] <- model.PAIO$mistakes
    SVs.PAIO[i, ] <- model.PAIO$SVs
    TMs.PAIO[i, ] <- model.PAIO$TMs
  
  
    # 3. HomOTL(fixed)
    nSV.OTLF <- vector(length = nGroup)
    err.OTLF <- vector(length = nGroup)
    time.OTLF <- vector(length = nGroup)
    mistakes.list.OTLF <- matrix(nrow = nGroup, ncol = nColumn)
    SVs.OTLF <- matrix(nrow = nGroup, ncol = nColumn)
    TMs.OTLF <- matrix(nrow = nGroup, ncol = nColumn)
    
    #model.OTLF <- HomOTLF.K.M(Y, K, K2, options, ID, model.old$classifier)
    
    nSV.OTLF[i] <- length(model.OTLF$classifier$SV)
    err.OTLF[i] <- model.OTLF$err.count
    time.OTLF[i] <- model.OTLF$run.time
    mistakes.list.OTLF[i, ] <- model.OTLF$mistakes
    SVs.OTLF[i, ] <- model.OTLF$SVs
    TMs.OTLF[i, ] <- model.OTLF$TMs
  
  
    # 4. HomOTL-I
    nSV.OTL <- vector(length = nGroup)
    err.OTL <- vector(length = nGroup)
    time.OTL <- vector(length = nGroup)
    mistakes.list.OTL <- matrix(nrow = nGroup, ncol = nColumn)
    SVs.OTL <- matrix(nrow = nGroup, ncol = nColumn)
    TMs.OTL <- matrix(nrow = nGroup, ncol = nColumn)
    
    model.OTL <- HomOTL1.K.M(Y, K, K2, options, ID, old.model$classifier)
    
    nSV.OTL[i] <- length(model.OTL$classifier$SV)
    err.OTL[i] <- model.OTL$err.count
    time.OTL[i] <- model.OTl$run.time
    mistakes.list.OTL[i, ] <- model.OTL$mistakes
    SVs.OTL[i, ] <- model.OTL$SVs
    TMs.OTL[i, ] <- model.OTL$TMs
  
   
    # 5. HomOTL-II
    nSV.OTL2 <- vector(length = nGroup)
    err.OTL2 <- vector(length = nGroup)
    time.OTL2 <- vector(length = nGroup)
    mistakes.list.OTL2 <- matrix(nrow = nGroup, ncol = nColumn)
    SVs.OTl2 <- matrix(nrow = nGroup, ncol = nColumn)
    TMs.OTl2 <- matrix(nrow = nGroup, ncol = nColumn)
    
    model.OTL2 <- HomOTL2.K.M(Y, K, K2, options, ID, model.old$classifier)
    
    nSV.OTL2[i] <- length(model.OTL2$classifier$SV)
    err.OTL2[i] <- model.OTl2$err.count
    time.OTL2[i] <- model.OTL2$run.time
    mistakes.list.OTL2[i, ] <- model.OTL2$mistakes
    SVs.OTL2[i, ] <- model.OTL2$SVs
    TMs.OTL2[i, ] <- model.OTL2$TMs
  
  
  }
  
  # print and plot results
  
}

setwd('~/Workspace/OnlineTransferLearning/Homogeneous/R')

dataset.name <- 'books_dvd.mat'

Experiment.OTL.K.M('books_dvd.mat')




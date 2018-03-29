
library('SuperLearner')
library('parallel')

args = commandArgs(trailingOnly=TRUE)

working_dir <- args[1]
taskn <- as.numeric(args[2])

print(working_dir)
print(taskn)

load_data <- function(foldn, taskn) {
    xtf <- sprintf('%s/input_train_%d_%d.csv', working_dir, foldn, taskn)
    ytf <- sprintf('%s/output_train_%d_%d.csv', working_dir, foldn, taskn)
    xtef <- sprintf('%s/input_test_%d_%d.csv', working_dir, foldn, taskn)
    ytef <- sprintf('%s/output_test_%d_%d.csv', working_dir, foldn, taskn)
    Xtrain <- as.data.frame(read.csv(xtf, header=FALSE))
    Ytrain <- as.data.frame(read.csv(ytf, header=FALSE))[,taskn+1]
    Xtest <- as.data.frame(read.csv(xtef, header=FALSE))
    Ytest <- as.data.frame(read.csv(ytef, header=FALSE))[,taskn+1]
    res <- list(Xtrain, Ytrain, Xtest, Ytest)
    return(res)
}

validate_fold <- function(ftn) {
    foldn <- ftn[1]
    taskn <- ftn[2]
    data <- load_data(foldn, taskn)
    Xtrain <- data[[1]]
    Ytrain <- data[[2]]
    Xtest <- data[[3]]
    Ytest <- data[[4]]
    summary(Ytrain)
    SL.library <-c("SL.glmnet","SL.glm","SL.stepAIC","SL.nnet","SL.polymars","SL.randomForest","SL.gam","SL.rpartPrune", "SL.bayesglm")
    fitSL<-SuperLearner(Y=Ytrain, X=Xtrain, newX=Xtest, family = binomial(), SL.library=SL.library, method = "method.NNLS", id = NULL, verbose = TRUE, cvControl=list(stratifyCV=TRUE,shuffle=TRUE))
    return(fitSL)
}

validate_task <- function(taskn, foldnum=5) {
    args <- list()
    for (foldn in c(1:foldnum)) {
        args[[foldn]] <- c(foldn-1, taskn)
    }
    fitResults <- mclapply(args, validate_fold, mc.preschedule=FALSE, mc.cores=32)
    return(fitResults)
}

fitResults <- validate_task(taskn)

foldn <- 0
results <- list()
for (fitSL in fitResults) {
    ytef <- sprintf('%s/output_test_%d_%d.csv', working_dir, foldn, taskn)
    Ytest <- as.data.frame(read.csv(ytef, header=FALSE))[,taskn+1]
    predictSL<- fitSL$SL.predict
    library_predictSL = fitSL$library.predict
    coef <- fitSL$coef
    results[[foldn+1]] <- list(Ytest, predictSL, library_predictSL, coef)
    foldn <- foldn + 1
}

results_dir <- sprintf('%s/results_%d.rds', working_dir, taskn)
saveRDS(results, results_dir)

results <- readRDS(results_dir)

library(cvAUC)
foldn <- 0
aucs <- c()
for (fitSL in fitResults) {
    Ytest <- results[[foldn+1]][1]
    predictSL<- results[[foldn+1]][2]
    aucresults = cvAUC(predictions=predictSL, labels=Ytest)
    aucresults_ci = ci.cvAUC(predictions=predictSL, labels=Ytest, confidence=0.95)
    aucs <- append(aucs, aucresults_ci$cvAUC[1])
    foldn <- foldn + 1
}

print(aucs)
print(mean(aucs))
stderr <- sd(aucs)/sqrt(length(aucs))
print(stderr)
print(mean(aucs) - 1.96*stderr)
print(mean(aucs) + 1.96*stderr)

library(MLmetrics)

foldn <- 0
aucs <- c()
for (fitSL in fitResults) {
    Ytest <- results[[foldn+1]][1]
    predictSL<- results[[foldn+1]][2]
    aucs <- append(aucs, PRAUC(predictSL, Ytest))
    foldn <- foldn + 1
}

print(aucs)
print(mean(aucs))
stderr <- sd(aucs)/sqrt(length(aucs))
print(stderr)
print(mean(aucs) - 1.96*stderr)
print(mean(aucs) + 1.96*stderr)

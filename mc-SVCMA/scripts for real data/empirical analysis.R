#################### Loading packages ######################
library(caret)
library(ggplot2)
library(reshape2)
library(pROC)
library(gridExtra)
library(RColorBrewer)
library(PRROC)
library(LiblineaR)
library(nloptr)
library(mvtnorm)

set.seed(123)
nism <- 100

#################### Functions ################
standardize_data <- function(X_train, X_test) {
  train_mean <- apply(X_train, 2, mean)
  train_std <- apply(X_train, 2, sd)
  
  X_train_scaled <- scale(X_train, center = train_mean, scale = train_std)
  X_test_scaled <- scale(X_test, center = train_mean, scale = train_std)
  return(list(train = X_train_scaled, test = X_test_scaled))
}

multi_hinge_loss <- function(y_true, scores) {
  n <- length(y_true)
  J <- ncol(scores)
  loss <- 0
  for (i in 1:n) {
    true_c <- as.integer(as.character(y_true[i]))
    true_score <- scores[i, true_c]
    max_margin <- 0
    for (c in 1:J) {
      if (c == true_c) next
      margin <- max(0, 1 - true_score + scores[i, c])
      if (margin > max_margin) max_margin <- margin
    }
    loss <- loss + max_margin
  }
  return(loss / n)
}

get_min_loss <- function(beta_matrix, X_test, y_test) {
  J <- nrow(beta_matrix)
  S <- dim(beta_matrix)[3]
  X_test_total <- X_test
  
  w_min_fun <- function(w) {
    wbeta <- matrix(0, J, ncol(beta_matrix))
    for (s in 1:S) wbeta <- wbeta + w[s] * beta_matrix[, , s]
    scores <- X_test_total %*% t(wbeta)
    return(multi_hinge_loss(y_test, scores))
  }
  
  heq <- function(w) sum(w) - 1
  p0 <- rep(1 / S, S)
  
  res <- nloptr(x0 = p0, eval_f = w_min_fun,
                lb = rep(0, S), ub = rep(1, S),
                eval_g_eq = heq,
                opts = list(algorithm = "NLOPT_LN_COBYLA",
                            xtol_rel = 1e-7, maxeval = 3000, print_level = 0))
  return(w_min_fun(res$solution))
}

svm_model <- function(X_train, Y_train, C_val) {
  Y_train <- as.factor(Y_train)
  J <- length(levels(Y_train))
  clf <- LiblineaR(data = X_train, 
                   target = Y_train,  
                   type = 4,    
                   cost = C_val, 
                   bias = 1)
  
  W <- clf$W 
  beta <- cbind(W[, ncol(W)], W[, -ncol(W)])
  
  target_order <- levels(Y_train)
  class_order <- as.character(clf$ClassNames)
  match_idx <- match(target_order, class_order)
  beta <- beta[match_idx, , drop = FALSE]
  
  X_s <- cbind(1, X_train)
  scores <- X_s %*% t(beta)
  loss <- multi_hinge_loss(Y_train, scores)
  
  return(list(
    model = clf,
    beta = beta,
    scores = scores,
    loss = loss,
    class_order = target_order 
  ))
}

class_evaluate <- function(y_true, y_pred, y_prob) {
  all_levels <- unique(c(as.character(y_true), as.character(y_pred)))
  y_true <- factor(y_true, levels = all_levels)
  y_pred <- factor(y_pred, levels = all_levels)
  c <- levels(y_true)
  J <- length(c)
  n <- length(y_true)
  
  TP <- numeric(J)
  FP <- numeric(J)
  FN <- numeric(J)
  TN <- numeric(J)
  precision <- numeric(J)
  recall <- numeric(J)
  F1 <- numeric(J)
  roc_auc_j <- numeric(J)
  pr_auc_j <- numeric(J)
  names(TP) <- names(FP) <- names(FN) <- names(TN) <- c
  names(precision) <- names(recall) <- names(F1) <- names(roc_auc_j) <- names(pr_auc_j) <- c
  
  cm <- table(y_pred, y_true)
  for (j in 1:J) {
    idx <- which(rownames(cm) == c[j])
    TP[j] <- cm[idx, idx]
    FP[j] <- sum(cm[, idx]) - TP[j]
    FN[j] <- sum(cm[idx, ]) - TP[j]
    TN[j] <- sum(cm) - TP[j] - FP[j] - FN[j]
    
    precision[j] <- TP[j] / (TP[j] + FP[j])
    recall[j] <- TP[j] / (TP[j] + FN[j])
    F1[j] <- 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    
    y_bin <- as.numeric(y_true == c[j])
    prob_bin <- y_prob[, c[j]]
    roc_auc_j[j] <- roc(y_bin, prob_bin, quiet = TRUE)$auc
    pr_obj <- pr.curve(scores.class0 = prob_bin[y_bin == 1],
                       scores.class1 = prob_bin[y_bin == 0],
                       curve = FALSE)
    pr_auc_j[j] <- pr_obj$auc.integral
  }
  
  accuracy <- sum(TP) / n
  
  micro_precision <- sum(TP) / sum(TP + FP)
  micro_recall <- sum(TP) / sum(TP + FN)
  micro_F1 <- (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
  
  macro_precision <- mean(precision)
  macro_recall <- mean(recall)
  macro_F1 <- mean(F1)
  macro_roc_auc <- mean(roc_auc_j)
  macro_pr_auc <- mean(pr_auc_j)
  
  n_j <- table(y_true)
  weighted_precision <- sum(n_j / n * precision)
  weighted_recall <- sum(n_j / n * recall)
  weighted_F1 <- sum(n_j / n * F1)
  weighted_roc_auc <- sum(n_j / n * roc_auc_j)
  weighted_pr_auc <- sum(n_j / n * pr_auc_j)
  
  y_ic <- matrix(0, nrow = n, ncol = J)
  for (i in 1:n) {
    y_ic[i, which(c == y_true[i])] <- 1
  }
  log_loss <- -mean(rowSums(y_ic * log(y_prob)))
  mse <- mean((y_ic - y_prob)^2)
  mae <- mean(abs(y_ic - y_prob))
  
  return(list(
    n = n,
    J = J,
    c = c,
    confusion_matrix = cm,
    TP = TP, FP = FP, FN = FN, TN = TN,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    F1 = F1,
    roc_auc = roc_auc_j,
    pr_auc = pr_auc_j,
    micro_precision = micro_precision,
    micro_recall = micro_recall,
    micro_F1 = micro_F1,
    macro_precision = macro_precision,
    macro_recall = macro_recall,
    macro_F1 = macro_F1,
    macro_roc_auc = macro_roc_auc,
    macro_pr_auc = macro_pr_auc,
    weighted_precision = weighted_precision,
    weighted_recall = weighted_recall,
    weighted_F1 = weighted_F1,
    weighted_roc_auc = weighted_roc_auc,
    weighted_pr_auc = weighted_pr_auc,
    log_loss = log_loss,
    mse = mse,
    mae = mae
  ))
}

SVMIC <- function(y_train, svm_result, X_train_scaled) {
  n <- length(y_train)
  J <- length(levels(y_train))
  hinge_loss <- svm_result$loss
  p_s <- ncol(X_train_scaled)
  SVMICLs <- hinge_loss * n + J * p_s * log(n)
  SVMICHs <- hinge_loss * n + J * p_s * ((log(n))^(3/2))
  return(list(SVMICLs = SVMICLs, SVMICHs = SVMICHs))
}

softmax <- function(x) {
  exp(x - max(x)) / sum(exp(x - max(x)))
}

weight_pred <- function(beta_matrix, X_test_total, y_test, weights) {
  wbeta <- matrix(0, nrow(beta_matrix), ncol(beta_matrix))
  for (s in 1:length(weights)) {
    wbeta <- wbeta + weights[s] * beta_matrix[, , s]
  }
  test_scores <- X_test_total %*% t(wbeta)
  
  y_pred <- apply(test_scores, 1, which.max) 
  y_pred <- as.factor(y_pred)
  
  y_prob <- t(apply(test_scores, 1, softmax))
  colnames(y_prob) <- as.character(1:(nrow(beta_matrix)))
  eval_result <- class_evaluate(y_test, y_pred, y_prob)
  return(eval_result)
}

############################### Load data #####################
data_folder <- "data/"
output_folder <- "data/"

if (!dir.exists(output_folder)) {
  dir.create(output_folder, recursive = TRUE)
}

X_list <- list()
labels_list <- list()
n_methods <- 7
C_val <- 1
data_id <- "CH12"
train_ratio <- 0.8
J <- 9

for (i in 0:8) {
  file_name <- file.path(data_folder, sprintf('CH12_%d.csv', i))
  data <- read.csv(file_name, header = TRUE)
  X_list[[i + 1]] <- as.matrix(data)
  labels_list[[i + 1]] <- rep(i + 1, nrow(data))
}

X1 <- do.call(rbind, X_list)
X1 <- X1[, 1:26]
Y1 <- unlist(labels_list)

model <- LiblineaR(data = X1, target = Y1, type = 4, cost = C_val, bias = 1)
W <- model$W
importance <- apply(abs(W[, -ncol(W), drop = FALSE]), 2, sum)
sorted_idx <- order(importance, decreasing = TRUE)
importance_sorted <- importance[sorted_idx]
W_total <- sum(importance)
theta <- 1
cum_importance <- cumsum(importance_sorted)
m_selected <- which(cum_importance >= theta * W_total)[1]
selected_idx <- sorted_idx[1:m_selected]

imp_df <- data.frame(Feature = 1:length(importance), Importance = importance)
imp_df <- imp_df[order(imp_df$Importance, decreasing = TRUE), ]
imp_df$Rank <- 1:nrow(imp_df)

X <- X1[, selected_idx, drop = FALSE]
n <- nrow(X)
p <- ncol(X)
Y <- as.factor(Y1)
S <- p
K <- 5

############################ Train candidate models ####################
method_names <- c("SVMICL", "SVMICH", "SCL", "SCH", "EW", "OvR-SVCMA", "mc-SVCMA")
methods <- c("SVMICL", "SVMICH", "SCL", "SCH", "EW", "mc-SVCMA")

results_acc <- matrix(0, nrow = nism, ncol = n_methods)
results_nhl <- matrix(0, nrow = nism, ncol = length(methods))
results_hinge <- matrix(0, nrow = nism, ncol = length(methods))

colnames(results_nhl) <- colnames(results_hinge) <- methods

pb <- txtProgressBar(min = 0, max = nism, style = 3)

for (m in 1:nism) {
  train_idx <- caret::createDataPartition(Y, p = train_ratio, list = FALSE)
  
  X_train <- X[train_idx, ]
  y_train <- Y[train_idx]
  X_test <- X[-train_idx, ]
  y_test <- Y[-train_idx]
  
  n_train <- nrow(X_train)
  n_test <- nrow(X_test)
  C_val <- 1
  vars <- 1:p
  feat_cols <- list()
  for (i in 1:p) {
    feat_cols[[i]] <- vars[1:i]
  }
  
  X_train_list <- list()
  X_test_list <- list()
  beta_matrix <- array(0, dim = c(J, p + 1, S))
  SVMICLs <- numeric(S)
  SVMICHs <- numeric(S)
  
  for (s in 1:S) {
    fcols <- feat_cols[[s]]
    X_train_s <- X_train[, fcols, drop = FALSE]
    X_test_s <- X_test[, fcols, drop = FALSE]
    scaled_s <- standardize_data(X_train_s, X_test_s)
    svm_result_s <- svm_model(scaled_s$train, y_train, C_val)
    X_train_list[[s]] <- scaled_s$train
    X_test_list[[s]] <- scaled_s$test
    beta <- svm_result_s$beta
    beta_matrix[, 1, s] <- beta[, 1]
    beta_matrix[, feat_cols[[s]] + 1, s] <- beta[, -1]
    SVMIC_result <- SVMIC(y_train, svm_result_s, X_train_list[[s]])
    SVMICLs[s] <- SVMIC_result$SVMICLs
    SVMICHs[s] <- SVMIC_result$SVMICHs
  }
  
  cv_beta_matrix <- array(0, dim = c(J, p + 1, S, K))
  folds <- createFolds(y_train, k = K, list = TRUE, returnTrain = FALSE)
  cv_X_test <- list()
  cv_y_test <- list()
  X_train_total <- X_train_list[[S]]
  X_test_total <- cbind(1, X_test_list[[S]])
  
  for (k in 1:K) {
    A_k <- folds[[k]]
    B_k <- setdiff(1:n_train, A_k)
    y_train_fold <- y_train[B_k]
    y_test_fold <- y_train[A_k]
    X_train_fold <- X_train_total[B_k, , drop = FALSE]
    X_test_fold <- cbind(1, X_train_total[A_k, ])
    cv_y_test[[k]] <- y_test_fold
    cv_X_test[[k]] <- X_test_fold
    
    for (s in 1:S) {
      X_train_s <- X_train_fold[, feat_cols[[s]], drop = FALSE]
      svm_result_fold <- svm_model(X_train_s, y_train_fold, 1)
      cv_beta <- svm_result_fold$beta
      cv_beta_matrix[, 1, s, k] <- cv_beta[, 1]
      cv_beta_matrix[, feat_cols[[s]] + 1, s, k] <- cv_beta[, -1]
    }
  }
  
  w_obj <- function(w) {
    cv_loss <- 0
    for (k in 1:K) {
      beta_k <- cv_beta_matrix[, , , k]
      wbeta_k <- matrix(0, J, p + 1)
      for (s in 1:S) {
        wbeta_k <- wbeta_k + w[s] * beta_k[, , s]
      }
      X_test_s <- cv_X_test[[k]]
      y_test_fold <- cv_y_test[[k]]
      cv_scores <- X_test_s %*% t(wbeta_k)
      cv_loss <- cv_loss + multi_hinge_loss(y_test_fold, cv_scores) * length(y_test_fold)
    }
    return(cv_loss / n_train)
  }
  
  heq <- function(w) { sum(w) - 1 }
  
  p0 <- rep(1 / S, S)
  res <- nloptr(
    x0 = p0,
    eval_f = w_obj,
    lb = rep(0, S),
    ub = rep(1, S),
    eval_g_eq = heq,
    opts = list(
      algorithm = "NLOPT_LN_COBYLA",
      xtol_rel = 1e-6,
      maxeval = 2000,
      print_level = 0
    )
  )
  cv_weight <- res$solution
  
  best_SVMICLs <- which.min(SVMICLs)
  SVMICL_beta <- beta_matrix[, , best_SVMICLs]
  SVMICL_test_scores <- X_test_total %*% t(SVMICL_beta)
  SVMICL_y_pred <- apply(SVMICL_test_scores, 1, which.max) 
  SVMICL_y_pred <- as.factor(SVMICL_y_pred)
  SVMICL_y_prob <- t(apply(SVMICL_test_scores, 1, softmax))
  colnames(SVMICL_y_prob) <- as.character(1:J)
  SVMICL_eval_result <- class_evaluate(y_test, SVMICL_y_pred, SVMICL_y_prob)
  
  best_SVMICHs <- which.min(SVMICHs)
  SVMICH_beta <- beta_matrix[, , best_SVMICHs]
  SVMICH_test_scores <- X_test_total %*% t(SVMICH_beta)
  SVMICH_y_pred <- apply(SVMICH_test_scores, 1, which.max) 
  SVMICH_y_pred <- as.factor(SVMICH_y_pred)
  SVMICH_y_prob <- t(apply(SVMICH_test_scores, 1, softmax))
  colnames(SVMICH_y_prob) <- as.character(1:J)
  SVMICH_eval_result <- class_evaluate(y_test, SVMICH_y_pred, SVMICH_y_prob)
  
  hinge_svmicl <- multi_hinge_loss(y_test, SVMICL_test_scores)
  hinge_svmich <- multi_hinge_loss(y_test, SVMICH_test_scores)
  min_loss <- get_min_loss(beta_matrix, X_test_total, y_test)
  nhl_svmicl <- hinge_svmicl / min_loss
  nhl_svmich <- hinge_svmich / min_loss
  
  cv_eval_result <- weight_pred(beta_matrix, X_test_total, y_test, cv_weight)
  
  eq_weight <- rep(1 / S, S)
  eq_eval_result <- weight_pred(beta_matrix, X_test_total, y_test, eq_weight)
  
  SCL_weight <- (exp(-(SVMICLs - min(SVMICLs)) / 2)) / (sum(exp(-(SVMICLs - min(SVMICLs)) / 2)))
  SCL_eval_result <- weight_pred(beta_matrix, X_test_total, y_test, SCL_weight)
  
  SCH_weight <- (exp(-(SVMICHs - min(SVMICHs)) / 2)) / (sum(exp(-(SVMICHs - min(SVMICHs)) / 2)))
  SCH_eval_result <- weight_pred(beta_matrix, X_test_total, y_test, SCH_weight)
  
  wbeta_scl <- matrix(0, J, p + 1)
  for (s in 1:S) wbeta_scl <- wbeta_scl + SCL_weight[s] * beta_matrix[, , s]
  scores_scl <- X_test_total %*% t(wbeta_scl)
  hinge_scl <- multi_hinge_loss(y_test, scores_scl)
  nhl_scl <- hinge_scl / min_loss
  
  wbeta_sch <- matrix(0, J, p + 1)
  for (s in 1:S) wbeta_sch <- wbeta_sch + SCH_weight[s] * beta_matrix[, , s]
  scores_sch <- X_test_total %*% t(wbeta_sch)
  hinge_sch <- multi_hinge_loss(y_test, scores_sch)
  nhl_sch <- hinge_sch / min_loss
  
  wbeta_eq <- matrix(0, J, p + 1)
  for (s in 1:S) wbeta_eq <- wbeta_eq + (1 / S) * beta_matrix[, , s]
  scores_eq <- X_test_total %*% t(wbeta_eq)
  hinge_eq <- multi_hinge_loss(y_test, scores_eq)
  nhl_eq <- hinge_eq / min_loss
  
  wbeta_cv <- matrix(0, J, p + 1)
  for (s in 1:S) wbeta_cv <- wbeta_cv + cv_weight[s] * beta_matrix[, , s]
  scores_cv <- X_test_total %*% t(wbeta_cv)
  hinge_cv <- multi_hinge_loss(y_test, scores_cv) 
  nhl_cv <- hinge_cv / min_loss
  
  ######################### OvR-SVMMA ############################
  classes <- levels(as.factor(y_train))
  iter_results <- list(tp = numeric(J))
  X_train_bin_scaled <- X_train_list[[S]]
  
  for (j in 1:J) {
    y_binary <- ifelse(y_train == classes[j], 1, -1)
    y_binary <- as.factor(y_binary)
    y_test_binary <- ifelse(y_test == classes[j], 1, -1)
    y_test_binary <- as.factor(y_test_binary)
    
    y_train_bin <- y_binary
    X_train_bin <- X_train_bin_scaled
    
    n1_bin <- length(y_train_bin)
    p_bin <- ncol(X_train_bin)
    M_bin <- p_bin
    
    feat_bin <- list()
    for (i in 1:p_bin) {
      feat_bin[[i]] <- 1:i
    }
    
    beta_bin <- array(0, dim = c(p_bin + 1, M_bin))
    
    for (m_idx in 1:M_bin) {
      bin_fcols <- feat_bin[[m_idx]]
      X_train_s_bin <- X_train_bin[, bin_fcols, drop = FALSE]
      
      clf_bin <- LiblineaR(data = X_train_s_bin, target = y_train_bin, 
                           type = 3, cost = C_val, bias = 1)
      
      W_bin <- clf_bin$W
      betab <- c(W_bin[, ncol(W_bin)], W_bin[, -ncol(W_bin)])
      
      beta_bin[1, m_idx] <- betab[1]
      beta_bin[bin_fcols + 1, m_idx] <- betab[-1]
    }
    
    folds_bin <- createFolds(y_train_bin, k = K, list = TRUE, returnTrain = FALSE)
    
    cv_beta_bin <- array(0, dim = c(p_bin + 1, M_bin, K))
    cv_X_bin <- list()
    cv_y_bin <- list()
    
    for (k in 1:K) {
      val_idx <- folds_bin[[k]]
      train_idx_fold <- setdiff(1:n1_bin, val_idx)
      
      y_train_fold_bin <- y_train_bin[train_idx_fold]
      y_test_fold_bin <- y_train_bin[val_idx]
      
      X_train_fold_bin <- X_train_bin[train_idx_fold, , drop = FALSE]
      X_test_fold_bin <- cbind(1, X_train_bin[val_idx, ])
      
      cv_y_bin[[k]] <- y_test_fold_bin
      cv_X_bin[[k]] <- X_test_fold_bin
      
      for (m_idx in 1:M_bin) {
        bin_fcols <- feat_bin[[m_idx]]
        X_train_s_bin <- X_train_fold_bin[, bin_fcols, drop = FALSE]
        
        cv_clf <- LiblineaR(data = X_train_s_bin, target = y_train_fold_bin,
                            type = 3, cost = C_val, bias = 1)
        
        cv_W_bin <- cv_clf$W
        cv_betab <- c(cv_W_bin[, ncol(cv_W_bin)], cv_W_bin[, -ncol(cv_W_bin)])
        
        cv_beta_bin[1, m_idx, k] <- cv_betab[1]
        cv_beta_bin[bin_fcols + 1, m_idx, k] <- cv_betab[-1]
      }
    }
    
    w_obj_bin <- function(w_b) {
      cv_loss_bin <- 0
      for (k in 1:K) {
        beta_kb <- cv_beta_bin[, , k]
        wbeta_kb <- matrix(0, 1, p_bin + 1)
        for (m_idx in 1:M_bin) {
          wbeta_kb <- wbeta_kb + w_b[m_idx] * beta_kb[, m_idx]
        }
        X_test_s_bin <- cv_X_bin[[k]]
        y_test_fold_bin <- cv_y_bin[[k]]
        
        cv_scores_bin <- X_test_s_bin %*% t(wbeta_kb)
        y_num <- as.numeric(as.character(y_test_fold_bin))
        cv_loss_bin <- cv_loss_bin + sum(pmax(0, 1 - y_num * cv_scores_bin[, 1]))
      }
      return(cv_loss_bin / n1_bin)
    }
    
    heq_bin <- function(w_b) sum(w_b) - 1
    p0 <- rep(1 / M_bin, M_bin)
    
    res_bin <- nloptr(
      x0 = p0,
      eval_f = w_obj_bin,
      lb = rep(0, M_bin),
      ub = rep(1, M_bin),
      eval_g_eq = heq_bin,
      opts = list(
        algorithm = "NLOPT_LN_COBYLA",
        xtol_rel = 1e-7,
        maxeval = 3000,
        print_level = 0
      )
    )
    
    w_bin <- res_bin$solution
    
    wbeta_bin <- matrix(0, 1, p_bin + 1)
    for (m_idx in 1:M_bin) {
      wbeta_bin <- wbeta_bin + w_bin[m_idx] * beta_bin[, m_idx]
    }
    
    test_scores_bin <- X_test_total %*% t(wbeta_bin)
    pred_bin <- ifelse(test_scores_bin[, 1] >= 0, 1, -1)
    
    y_true <- as.numeric(as.character(y_test_binary))
    tp <- sum(y_true == 1 & pred_bin == 1)
    iter_results$tp[j] <- tp
  }
  bin_accuracy <- sum(iter_results$tp / n_test)
  setTxtProgressBar(pb, m)
  
  results_acc[m, ] <- c(SVMICL_eval_result$accuracy, SVMICH_eval_result$accuracy, 
                        SCL_eval_result$accuracy, SCH_eval_result$accuracy, 
                        eq_eval_result$accuracy, bin_accuracy, cv_eval_result$accuracy)
  
  results_nhl[m, ] <- c(nhl_svmicl, nhl_svmich, nhl_scl, nhl_sch, nhl_eq, nhl_cv)
  results_hinge[m, ] <- c(hinge_svmicl, hinge_svmich, hinge_scl, hinge_sch, hinge_eq, hinge_cv)
  setTxtProgressBar(pb, m)
}

close(pb)

################################### Plot ##################################
acc_df <- as.data.frame(results_acc)
colnames(acc_df) <- method_names
acc_melted <- reshape2::melt(acc_df, id.vars = NULL, variable.name = "Method", value.name = "Accuracy")
acc_melted$Method <- factor(acc_melted$Method, levels = c("SVMICL", "SVMICH", "SCL", "SCH", "EW", "OvR-SVCMA", "mc-SVCMA"))

nhl_df <- as.data.frame(results_nhl)
colnames(nhl_df) <- methods
nhl_melted <- reshape2::melt(nhl_df, id.vars = NULL, variable.name = "Method", value.name = "NHL")
nhl_melted$Method <- factor(nhl_melted$Method, levels = methods)

hinge_df <- as.data.frame(results_hinge)
colnames(hinge_df) <- methods
hinge_melted <- reshape2::melt(hinge_df, id.vars = NULL, variable.name = "Method", value.name = "HingeLoss")
hinge_melted$Method <- factor(hinge_melted$Method, levels = methods)

method_colors <- c("SVMICL" = "#F4A582",
                   "SVMICH" = "#E76F51",
                   "SCL" = "#C2A5CF",
                   "SCH" = "#9A8C98",
                   "EW" = "#A6DBA0",
                   "OvR-SVCMA" = "#FDBF6F",
                   "mc-SVCMA" = "#92C5DE")

pl1 <- ggplot(acc_melted, aes(x = Method, y = Accuracy, fill = Method)) +
  stat_boxplot(geom = "errorbar", width = 0.25) +
  geom_boxplot(outlier.colour = "black", outlier.shape = 1, outlier.size = 1.5, alpha = 0.7, fatten = 1) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black") +
  scale_fill_manual(values = method_colors) +
  labs(title = "(a) Accuracy", x = NULL, y = "Accuracy") +
  theme_bw() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 16, angle = 30, hjust = 1),
    axis.text.y = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
  )

pl2 <- ggplot(nhl_melted, aes(x = Method, y = NHL, fill = Method)) +
  stat_boxplot(geom = "errorbar", width = 0.25) +
  geom_boxplot(outlier.colour = "black", outlier.shape = 1, outlier.size = 1.5, alpha = 0.7, fatten = 1) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black") +
  scale_fill_manual(values = method_colors) +
  labs(title = "(b) NHL", x = NULL, y = "NHL") +
  theme_bw() +
  theme(
    legend.position = "none", 
    axis.text.x = element_text(size = 16, angle = 30, hjust = 1),
    axis.text.y = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
  )

pl3 <- ggplot(hinge_melted, aes(x = Method, y = HingeLoss, fill = Method)) +
  stat_boxplot(geom = "errorbar", width = 0.25) +
  geom_boxplot(outlier.colour = "black", outlier.shape = 1, outlier.size = 1.5, alpha = 0.7, fatten = 1) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "black") +
  scale_fill_manual(values = method_colors) +
  labs(title = "(c) Hinge Loss", x = NULL, y = "Hinge Loss") +
  theme_bw() +
  theme(
    legend.position = "none", 
    axis.text.x = element_text(size = 16, angle = 30, hjust = 1),
    axis.text.y = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    panel.grid.major.x = element_blank(),
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold")
  )

filename <- file.path(output_folder, paste0(data_id, "_theta", theta, "_n", n, "_K", K, "_boxplot.pdf"))
p_all <- arrangeGrob(pl1, pl2, pl3, ncol = 3)
ggsave(filename, p_all, width = 12, height = 4)

save.image(file.path(output_folder, paste0("theta", theta, "_", data_id, "_n", n, "_K", K, ".RData")))
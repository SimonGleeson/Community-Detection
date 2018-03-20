rm(list = ls())
setwd('~/Documents/network/facebook/results/')
library(ggplot2)
library(reshape2)

get_results <- function(n, k, enc){
  mainDir <- getwd()
  subDir <- paste('plots/plots_n', n, '_k', k, '_enc', enc, sep = '')
  if(file.exists(paste('results_n', n, '_k', k, '_enc', enc, '.csv', sep = ''))){
    df <- read.csv(paste('results_n', n, '_k', k, '_enc', enc, '.csv', sep = ''), header = TRUE)
    dir.create(file.path(mainDir, subDir), showWarnings = FALSE)
    setwd(file.path(mainDir, subDir))
    
    ggplot(df, aes(x = Iteration, y = LL*k)) + geom_line(color = 'red') + xlim(1, max(df$Iteration)) + 
      ylab("Log Likelihood")
    ggsave(paste('LogLikelihood_n', n, '_k', k, '_enc', enc,  '.png', sep = ''), plot = last_plot())
    
    ignore = c("Alpha", "BER", "LL", "Theta")
    assignments <- df[,!(names(df) %in% ignore)]
    assignments <- melt(assignments, id="Iteration")
    
    ggplot(assignments, aes(x = Iteration, y = value, color = variable)) + geom_line() 
    ggsave(paste('Assignments_n', n, '_k', k, '_enc', enc, '.png', sep = ''), plot = last_plot())
    
    tempdf <- df
    tempdf$BER <- ifelse(tempdf$BER == 0, NA, tempdf$BER)
    ggplot(tempdf, aes(x = Iteration, y = BER)) + geom_line(color = 'red') + 
      ylim(0, 1)
    ggsave(paste('BER_n', n, '_k', k, '_enc', enc, '.png', sep = ''), plot = last_plot()) 
    
    ggplot(df, aes(x = Iteration, y = Theta)) + geom_line(color = 'red') 
    ggsave(paste('Theta_n', n, '_k', k, '_enc', enc, '.png', sep = ''), plot = last_plot())
    
    ggplot(df, aes(x = Iteration, y = Alpha)) + geom_line(color = 'red') 
    ggsave(paste('Alpha_n', n, '_k', k, '_enc', enc, '.png', sep = ''), plot = last_plot())
    setwd(file.path(mainDir))
  }
}
merge_dfs_bar <- function(encodings, networks, ks){
  data = list()
  k. = c()
  e. = c()
  n. = c()
  ber. = c()
  ll. = c()
  for(e in encodings){
    for(k in ks){
      for(n in networks){
        if(file.exists(paste('results_n', n, '_k', k, '_enc', e, '.csv', sep = ''))){
          dftemp <- read.csv(paste('results_n', n, '_k', k, '_enc', e, '.csv', sep = ''), header = TRUE)
          ll <-  dftemp$LL[nrow(dftemp) - 1]
          ber <- dftemp$BER[nrow(dftemp) - 1]
          k. = c(k., k)
          e. = c(e., e)
          n. = c(n., n)
          ber. = c(ber., ber)
          ll. = c(ll., ll)
        }
      }
    }
  }
  df = data.frame(k., e., n., ber., ll.)
  colnames(df) <- c('K', 'Encoding', 'Network', 'BER', 'Log Likelihood')
  return(df)
}
get_bar <- function(df, n, v){
  if(v == 'LL'){
    df$BER <- NULL
    df$`Log Likelihood` <- df$`Log Likelihood` * df$K * -1
  }else{
    df$`Log Likelihood` <- NULL
  }
  df$K <- as.factor(df$K)
  df <- df[(df$Network == n),]
  df $Network <- NULL
  df.long <- melt(df, id.vars = c('Encoding', 'K'))
  df.long$variable <- NULL
  p = ggplot(df.long, aes(x=K, y=value, fill = Encoding)) + geom_bar(stat = 'identity', position='dodge') 
  if(v == 'BER'){
    p = p + ylab("BER")
  }else{
    p = p + ylab("Negative Log Likelihood")
  }
  nm = paste('Bar_network_', n, '_', v, '.eps', sep='')
  ggsave(nm, plot = p)
  # return(p)
}
get_total_df <- function(encodings, networks, ks){
  for(enc in encodings){
    for(k in ks){
      for(n in networks){
        if(file.exists(paste('results_n', n, '_k', k, '_enc', enc, '.csv', sep = ''))){
          dft <- read.csv(paste('results_n', n, '_k', k, '_enc', enc, '.csv', sep = ''), header = TRUE)
          dft <- dft[, (colnames(dft) %in% c('Alpha', 'BER', 'Iteration', 'LL', 'Theta'))]
          # dft$LL <- dft$LL / -dft$LL[2]
          dft$K = k
          dft$LL <- k*dft$LL
          dft$Encode = enc
          dft$Network = n
          if(is.null(get0('dfa'))){
            dfa = dft
          }else{
            dfa = rbind(dfa, dft)
          }
        }
      }
    }
  }
  dfa$K <- as.factor(dfa$K)
  dfa$Network <- as.factor(dfa$Network)
  dfa$BER <- ifelse(dfa$BER == 0, NA, dfa$BER)
  return(dfa)
}

encodings = c('phi1', 'w1', 'w2')
ks = c(2, 4, 6, 8)
networks = c(0, 698, 414)

for(e in encodings){
  for(k in ks){
    for(n in networks){
      get_results(n, k, e)
    }
  }
}

df = merge_dfs_bar(encodings, networks, ks)
for(n in networks){
  get_bar(df, n, 'BER')
  get_bar(df, n, 'LL')
}

dfall <- get_total_df(encodings, networks, ks)
temp <- dfall[(dfall$Network == 0), ]
ggplot(temp, aes(x=Iteration, y = LL, color = Encode, linetype = as.factor(K))) + geom_line() + xlim(1, 5)


  
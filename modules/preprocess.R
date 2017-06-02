# read data
fb = read.csv('E:/Features_Variant_1.csv')

# pick the best transformation
corrs = c(rep(0,6))
best = c(rep(0,51))
for( n in c(1:24,29:33,35:51) ){
  corrs[1] = cor(fb[,n]^3,fb[,52]) # cubic
  corrs[2] = cor(fb[,n]^2,fb[,52]) # square
  corrs[3] = cor(fb[,n],fb[,52]) # constant
  corrs[4] = cor(sqrt(fb[,n]),fb[,52]) # square root
  corrs[5] = cor(log(fb[,n]+1),fb[,52]) # log
  corrs[6] = cor(1/(fb[,n]+1),fb[,52]) # reciprocal
  cand = 0
  for( i in 1:6 ){
    if( abs(corrs[i]) > cand ){
      cand = corrs[i]
      best[n] = i
    }  
  }
}
best

# transform according to 'best'
fb_new = fb

# cubic
fb_new[which(best %in% 1)] = fb_new[which(best %in% 1)]^3
# square
fb_new[which(best %in% 2)] = fb_new[which(best %in% 2)]^2
# square root
fb_new[which(best %in% 4)] = sqrt(fb_new[which(best %in% 4)])
# log
fb_new[which(best %in% 5)] = log(fb_new[which(best %in% 5)] + 1)
# reciprocal
fb_new[which(best %in% 6)] = 1 / (fb_new[which(best %in% 6)] + 1)

# write to file
write.csv(fb_new, "E:/Features_Variant_1_transformed.csv", row.names=F)

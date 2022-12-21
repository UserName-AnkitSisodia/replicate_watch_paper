library(dplyr)
s <- c(1,2,3,4,5,6,7,8,9,10)
b <- c(1,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50)
m <- c(0,1,5,10,12,14,16,18,20,25,30,35,40,45,50)
df <- expand.grid(s,b,m)
colnames(df) <- c("s","b","m")
df$train_losses_file <- paste0("s",df$s,"b",df$b,"m",df$m,"_circa_train_losses.csv")
df$validation_loss <- NA

for(i in 1:nrow(df))
{
loss_file <- read.csv(df$train_losses_file[i],stringsAsFactors = FALSE) %>% filter(Epoch==max(Epoch)) %>% filter(grepl("mse_loss_validation",Loss))
df$validation_loss[i] <- loss_file$Value
}

df <- df %>% filter(m!=0)
df$validation_loss <- df$validation_loss/df$m

df_summary <- df %>% group_by(paste(b,m)) %>% summarise(validation_loss=mean(validation_loss)) %>% as.data.frame()
print(df_summary[which.min(df_summary$validation_loss),])

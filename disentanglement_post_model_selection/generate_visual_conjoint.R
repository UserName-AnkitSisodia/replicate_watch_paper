x1 <- c(-2,0,2) + 0.01
x2 <- c(-2,0,2) + 0.01
x3 <- c(-2,0,2) + 0.01
x4 <- c(-2,0,2) + 0.01
x5 <- c(-2,0,2) + 0.01
x6 <- c(-2,0,2) + 0.01
x7 <- c(-2,0,2) + 0.01
x8 <- c(-2,0,2) + 0.01
x9 <- c(-2,0,2) + 0.01
x10 <- c(-2,0,2) + 0.01

x <- expand.grid(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)
colnames(x) <- c("x1","x2","x3","x4","x5","x6","x7","x8","x9","x10")

x$v1 <- ifelse(x$x1==-1.99,"L",ifelse(x$x1==0.01,"M","H"))
x$v2 <- ifelse(x$x2==-1.99,"L",ifelse(x$x2==0.01,"M","H"))
x$v3 <- ifelse(x$x3==-1.99,"L",ifelse(x$x3==0.01,"M","H"))
x$v4 <- ifelse(x$x4==-1.99,"L",ifelse(x$x4==0.01,"M","H"))
x$v5 <- ifelse(x$x5==-1.99,"L",ifelse(x$x5==0.01,"M","H"))
x$v6 <- ifelse(x$x6==-1.99,"L",ifelse(x$x6==0.01,"M","H"))
x$v7 <- ifelse(x$x7==-1.99,"L",ifelse(x$x7==0.01,"M","H"))
x$v8 <- ifelse(x$x8==-1.99,"L",ifelse(x$x8==0.01,"M","H"))
x$v9 <- ifelse(x$x9==-1.99,"L",ifelse(x$x9==0.01,"M","H"))
x$v10 <- ifelse(x$x10==-1.99,"L",ifelse(x$x10==0.01,"M","H"))

x$file <- paste0("x1_",x$v1,"_x2_",x$v2,"_x3_",x$v3,"_x4_",x$v4,"_x5_",x$v5,"_x6_",x$v6,"_x7_",x$v7,"_x8_",x$v8,"_x9_",x$v9,"_x10_",x$v10,".jpg")
x$seq <- seq(100001,100000+nrow(x),1)
x$cmd1 <- paste0("                mean_before_decoder",x$seq,"=torch.tensor([[",x$x1,",",x$x2,",",x$x3,",",x$x4,",",x$x5,",",x$x6,",",x$x7,",",x$x8,",",x$x9,",",x$x10,"]]) # Jun 15 2022")
x$cmd2 <- paste0("                filename",x$seq,"='",x$file,"' # Jun 15 2022")
x$cmd3 <- paste0("                mean_before_decoder",x$seq,"=mean_before_decoder",x$seq,".to(self.device) # Jun 15 2022")
x$cmd4 <- paste0("                decoded_traversal",x$seq,"=self.model.decoder(mean_before_decoder",x$seq,").cpu() # Jun 15 2022")
x$cmd5 <- paste0("                save_image(decoded_traversal",x$seq,",filename",x$seq,") # Jun 15 2022")

x$cmdx <- paste0("                save_image(self.model.decoder(torch.tensor([[",x$x1,",",x$x2,",",x$x3,",",x$x4,",",x$x5,",",x$x6,",",x$x7,",",x$x8,",",x$x9,",",x$x10,"]]).to(self.device)).cpu()",",x1_",x$v1,"_x2_",x$v2,"_x3_",x$v3,"_x4_",x$v4,"_x5_",x$v5,"_x6_",x$v6,"_x7_",x$v7,"_x8_",x$v8,"_x9_",x$v9,"_x10_",x$v10,".jpg) # Jun 15 2022")

write.csv(x,"jun15_2022.csv",row.names = FALSE)

library(dplyr)

rm(list=ls())

file_names <- read.csv("preprocessed_structured_data.csv",stringsAsFactors = FALSE)
file_names <- file_names  %>% filter(Year>=2010)
file_names <- file_names %>% filter(file_name!="15363-83-0.jpg" & file_name!="16375-114-0.jpg" & file_name!="16375-189-0.jpg" & file_name!="17563-78-0.jpg" & file_name!="17566-118-0.jpg" & file_name!="22642-230-0.jpg" & file_name!="22805-1-0.jpg" & file_name!="22805-3-0.jpg" & file_name!="22805-6-0.jpg" & file_name!="22805-7-0.jpg" & file_name!="22807-139-0.jpg" & file_name!="23102-178-0.jpg" & file_name!="23754-3529-0.jpg" & file_name!="23886-4762-0.jpg" & file_name!="23896-2661-0.jpg" & file_name!="24307-2644-0.jpg" & file_name!="24307-2645-0.jpg" & file_name!="24319-2341-0.jpg" & file_name!="24319-2644-0.jpg" & file_name!="24771-2505-0.jpg" & file_name!="24775-2489-0.jpg" & file_name!="24775-2649-0.jpg" & file_name!="25020-19-0.jpg" & file_name!="25628-2417-0.jpg" & file_name!="25628-2589-0.jpg" & file_name!="25628-2644-0.jpg" & file_name!="26310-2426-0.jpg" & file_name!="27677-2243-0.jpg" & file_name!="27677-2405-0.jpg")
file_names <- file_names %>% filter(location!="Amsterdam" & location!="London")
file_names <- file_names %>% filter(real_wtp >= 1000)

model_names <- file_names %>% group_by(Model_Name) %>% summarise(count=n())
set.seed(123)

model_names$rand <- runif(nrow(model_names))
model_names$dataset <- ifelse(model_names$rand < 0.05,"test1",ifelse(model_names$rand > 0.95,"test2","train_valid"))
file_names <- merge(file_names,model_names %>% select(Model_Name,dataset))
file_names$command <- paste0("mv *",file_names$file_name," ",file_names$dataset,"/")
write.csv(file_names %>% select(command),"command.csv",row.names=FALSE)

library(magick)
library(dplyr)
library(jpeg)
library(doParallel)

rm(list=ls())

file_names <- read.csv("preprocessed_structured_data.csv",stringsAsFactors = FALSE)
file_names <- file_names  %>% filter(Year>=2010)
file_names <- file_names %>% filter(file_name!="15363-83-0.jpg" & file_name!="16375-114-0.jpg" & file_name!="16375-189-0.jpg" & file_name!="17563-78-0.jpg" & file_name!="17566-118-0.jpg" & file_name!="22642-230-0.jpg" & file_name!="22805-1-0.jpg" & file_name!="22805-3-0.jpg" & file_name!="22805-6-0.jpg" & file_name!="22805-7-0.jpg" & file_name!="22807-139-0.jpg" & file_name!="23102-178-0.jpg" & file_name!="23754-3529-0.jpg" & file_name!="23886-4762-0.jpg" & file_name!="23896-2661-0.jpg" & file_name!="24307-2644-0.jpg" & file_name!="24307-2645-0.jpg" & file_name!="24319-2341-0.jpg" & file_name!="24319-2644-0.jpg" & file_name!="24771-2505-0.jpg" & file_name!="24775-2489-0.jpg" & file_name!="24775-2649-0.jpg" & file_name!="25020-19-0.jpg" & file_name!="25628-2417-0.jpg" & file_name!="25628-2589-0.jpg" & file_name!="25628-2644-0.jpg" & file_name!="26310-2426-0.jpg" & file_name!="27677-2243-0.jpg" & file_name!="27677-2405-0.jpg")
file_names <- file_names %>% filter(location!="Amsterdam" & location!="London")
file_names <- file_names %>% filter(real_wtp >= 1000)

plist = c("doParallel","foreach"); sapply(plist,require,character=TRUE)
ncores = detectCores();
regcores = min(ncores-5)
registerDoParallel(regcores)
print(paste("Number of Cores Registered: ",regcores))
getDoParWorkers()

table(file_names$brand_name)
table(file_names$location)
table(file_names$circa_range)
table(file_names$movement)
table(file_names$case_material)

vertical_center <- function(file_name,real_wtp,brand_name,location,date,circa,movement,diameter,material,timetrend,model_name)
{
  read_file <- image_read(file_name)
  # edited_file <- image_contrast(read_file)
  # edited_file <- image_transparent(read_file, 'white',fuzz = 0.10)
  # edited_file <- image_background(edited_file, "white", flatten = TRUE)
  # edited_file <- image_trim(edited_file)
  edited_file <- read_file
  max_dim <- as.character(max(image_info(edited_file)$width,image_info(edited_file)$height))
  edited_file <- image_extent(edited_file,paste0(max_dim,"x",max_dim),"center","white") 
  edited_file <- image_resize(edited_file,"128x128")
  image_write(edited_file, path = paste0("new%",real_wtp,"%$",date,"$#",location,"#^",brand_name,"^@",circa,"@!",movement,"!*",diameter,"*<",material,"<>",timetrend,">|",model_name,"|",file_name))
}

foreach(i=1:nrow(file_names),.packages="magick")%dopar%
{ # parallizable code here 
  vertical_center(file_names$file_name[i],file_names$log_real_wtp[i],file_names$brand_name[i],file_names$location[i],file_names$date[i],file_names$circa_range[i],file_names$movement[i],file_names$case_diameter[i],file_names$case_material[i],file_names$timetrend[i],file_names$Model_Name[i])
}


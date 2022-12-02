for i in *.jpg; do magick $i -background white -flatten -fuzz 40% -transparent white  \( +clone -white-threshold 50% -set option:cropvals "%@" +delete \) -crop "%[cropvals]" +repage $i; done

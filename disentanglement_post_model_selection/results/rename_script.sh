#!/bin/sh

mv full_s1b18m50/train_losses.csv full_s1b18m50/full_s1b18m50_train_losses.csv
mv full_s2b18m50/train_losses.csv full_s2b18m50/full_s2b18m50_train_losses.csv
mv full_s3b18m50/train_losses.csv full_s3b18m50/full_s3b18m50_train_losses.csv
mv full_s4b18m50/train_losses.csv full_s4b18m50/full_s4b18m50_train_losses.csv
mv full_s5b18m50/train_losses.csv full_s5b18m50/full_s5b18m50_train_losses.csv
mv full_s6b18m50/train_losses.csv full_s6b18m50/full_s6b18m50_train_losses.csv
mv full_s7b18m50/train_losses.csv full_s7b18m50/full_s7b18m50_train_losses.csv
mv full_s8b18m50/train_losses.csv full_s8b18m50/full_s8b18m50_train_losses.csv
mv full_s9b18m50/train_losses.csv full_s9b18m50/full_s9b18m50_train_losses.csv
mv full_s10b18m50/train_losses.csv full_s10b18m50/full_s10b18m50_train_losses.csv

ldconfig

echo "Done!"

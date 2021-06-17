conda activate dense_matching_env
cd data/
mkdir megadepth_test_set
gdown https://drive.google.com/uc?id=1SikcOvCJ-zznOyCRJCTGtpKtTp01Jx5g
unzip MegaDepth.zip -d megadepth_test_set
rm MegaDepth.zip
cd megadepth_test_set/MegaDepth
rm -r MegaDepth_Train
rm -r MegaDepth_Train_Org
rm -r Val
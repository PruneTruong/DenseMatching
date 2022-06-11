mkdir -p data
cd data/
wget http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset-PASCAL.zip
wget http://www.di.ens.fr/willow/research/cnngeometric/other_resources/test_pairs_pf_pascal.csv
wget http://www.di.ens.fr/willow/research/cnngeometric/other_resources/val_pairs_pf_pascal.csv
gdown https://drive.google.com/uc?id=111tpXshLiJ4qudBHoGK3HbMSNr9vVRq9
unzip PF-dataset-PASCAL.zip -d .
rm PF-dataset-PASCAL.zip
mv test_pairs_pf_pascal.csv PF-dataset-PASCAL
mv val_pairs_pf_pascal.csv PF-dataset-PASCAL
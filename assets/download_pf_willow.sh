mkdir -p data
cd data/
wget http://www.di.ens.fr/willow/research/proposalflow/dataset/PF-dataset.zip
wget http://www.di.ens.fr/willow/research/cnngeometric/other_resources/test_pairs_pf.csv
unzip PF-dataset.zip -d .
rm PF-dataset.zip
mv test_pairs_pf.csv PF-dataset
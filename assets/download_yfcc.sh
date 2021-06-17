mkdir -p data
cd data/
git clone https://github.com/zjhthu/OANet
cd OANet
bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8
tar -xvf raw_data_yfcc.tar.gz
mv raw_data/yfcc100m ../
cd ../
rm -r OANet
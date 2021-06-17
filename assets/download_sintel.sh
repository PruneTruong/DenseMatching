mkdir -p data
cd data/
wget https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip
mkdir -p sintel
unzip MPI-Sintel-complete.zip -d sintel
rm MPI-Sintel-complete.zip
# data-extractor.\*

used to generate dataset. intended to be called only through `make gen-dataset` command, that will generate `src/dataset/` folder in your build directory

# extract_metrics_from_random_file.\*

used to extract features from any given `.c` file. only `.sh` file is intended to be used by end-user. use: `./data-extractor.sh file.c`. it will generate folder, containing `data.csv` - select features that are accepted by classifier

# tree.py

main body of classifier. first, trains on given dataset, than predicts values for given objects. objects can be generated via `extract_metrics_from_random_file.sh`

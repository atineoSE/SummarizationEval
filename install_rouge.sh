#!/bin/bash
# See here for instructions and workaround
# Instructions: https://github.com/bheinzerling/pyrouge?tab=readme-ov-file#installation
# Workaround: https://github.com/bheinzerling/pyrouge/issues/14#issuecomment-351642652
DIR=`pwd`
pip install pyrouge
git clone git@github.com:andersjo/pyrouge.git
cd ./pyrouge/tools/ROUGE-1.5.5/data/WordNet-2.0-Exceptions/
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ..
rm WordNet-2.0.exc.db
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
cd DIR
pyrouge_set_rouge_path $DIR/pyrouge/tools/ROUGE-1.5.5
python -m pyrouge.test

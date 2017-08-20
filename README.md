# AADE

The python implementation of AADE models as described in the paper:
Associated Activation-Driven Enrichment: Understanding Implicit Information from a Cognitive Perspective

Five text enrichment methods mentioned in this paper (Global equal base, Global Actv, Local Actv, Similarity and Random) are implemented in aademodel.py

activate_test.py provides a test case on how to use the models.


To run activate_test.py or aademodel.py, you need to install python 2.7.3 first, as well as gensim 0.12.4 along with the other packages needed.

Run activate_test.py by executing the following command from the project home directory:
python src/activate_test.py --w2vmodel_path model/w2vmodel --stopwords_path data/stopwords.txt --input data/input.txt --output data/output.txt

or executing the following command from the src directory:
python activate_test.py

You can also train word2vec models through the gensim toolkit according to your needs.

For more information, please see the source codes and the paper.

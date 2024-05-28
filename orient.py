#!/usr/local/bin/python3
#
# orient.py : a image classifier
#
# Submitted by : Paventhan Vivekanandan, username : pvivekan
#
#

import sys
from decision_tree_classifier import tree_classifier
from neural_networks import nnet_classifier



# Main Function
if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: \n./orient.py [train/test] [training_file.txt/testfile.txt] model_file.txt [model]")
        sys.exit()

    if sys.argv[4] == "tree":
        tree_classifier(sys.argv[1], sys.argv[2], sys.argv[3])
    elif sys.argv[4] == "nnet":
        nnet_classifier(sys.argv[1], sys.argv[2], sys.argv[3])
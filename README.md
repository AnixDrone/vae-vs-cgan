# CVAE vs CGAN

Project that shows the difference between CVAE and CGAN.

Using the MNIST data set, we will train a CVAE and a CGAN to generate handwritten digits. We will then compare the two models and see which one is better. We will also see the difference in the generated digits.

To train the CVAE use the command:

python vae/cvae_experiment.py

To train the CGAN use the command:

python cgan/train.py


Furthermore to generate data there are generate data scripts for the CVAE and CGAN.

To compare the generated data use the testing_models.ipynb file in the classifer folder. First you will need to train the classifier using the mnist_classifier.ipynb file.

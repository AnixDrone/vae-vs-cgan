# CVAE vs CGAN

Project that shows the difference between CVAE and CGAN.

Using the MNIST data set, we will train a CVAE and a CGAN to generate handwritten digits. We will then compare the two models and see which one is better. We will also see the difference in the generated digits.

To train the CVAE use the command:

python vae/cvae_experiment.py

To train the CGAN use the command:

python cgan/train.py


Furthermore to generate data there are generate data scripts for the CVAE and CGAN.

To compare the generated data use the testing_models.ipynb file in the classifer folder. First you will need to train the classifier using the mnist_classifier.ipynb file.

# Results

The classifier results when tested on MNIST regular data:

                precision    recall  f1-score   support

           0       0.99      1.00      0.99       980
           1       1.00      0.99      1.00      1135
           2       0.98      0.99      0.99      1032
           3       0.99      0.99      0.99      1010
           4       0.98      0.99      0.99       982
           5       0.97      0.99      0.98       892
           6       1.00      0.98      0.99       958
           7       0.99      0.98      0.99      1028
           8       0.99      0.98      0.99       974
           9       0.99      0.97      0.98      1009

    accuracy                           0.99     10000
    macro avg       0.99      0.99      0.99     10000
    weighted avg       0.99      0.99      0.99     10000

![alt origina_confusion_matrix](https://github.com/AnixDrone/vae-vs-cgan/blob/main/assets/original_conf_matrix.png)

The classifier results when tested on CGAN generated data:

                precision    recall  f1-score   support

           0       0.94      0.98      0.96       100
           1       0.97      0.91      0.94       100
           2       0.93      0.90      0.91       100
           3       0.90      0.93      0.92       100
           4       0.93      0.97      0.95       100
           5       0.84      0.85      0.85       100
           6       0.93      0.99      0.96       100
           7       0.92      0.97      0.95       100
           8       0.96      0.82      0.89       100
           9       0.92      0.93      0.93       100

    accuracy                           0.93      1000
    macro avg       0.93      0.93      0.92      1000
    weighted avg       0.93      0.93      0.92      1000

![alt cgan_confusion_matrix](https://github.com/AnixDrone/vae-vs-cgan/blob/main/assets/cgan_conf_matrix.png)

The classifier results when tested on CVAE generated data:

                precision    recall  f1-score   support

           0       0.96      0.92      0.94       100
           1       0.66      0.38      0.48       100
           2       0.45      0.24      0.31       100
           3       0.03      0.06      0.04       100
           4       0.05      0.01      0.02       100
           5       0.17      0.22      0.19       100
           6       0.31      0.13      0.18       100
           7       0.17      0.35      0.22       100
           8       0.30      0.28      0.29       100
           9       0.37      0.41      0.39       100

    accuracy                           0.30      1000
    macro avg       0.35      0.30      0.31      1000
    weighted avg       0.35      0.30      0.31      1000

![alt cvae_confusion_matrix](https://github.com/AnixDrone/vae-vs-cgan/blob/main/assets/cvae_conf_matrix.png)


# Samples

## CGAN

### Generated images

![alt cgan_generated_images](https://github.com/AnixDrone/vae-vs-cgan/blob/main/assets/cgan_fake_images.png)

### Real images

![alt cgan_real_images](https://github.com/AnixDrone/vae-vs-cgan/blob/main/assets/cgan_real_images.png)

## CVAE

### Generated images

![alt cvae_generated_images](https://github.com/AnixDrone/vae-vs-cgan/blob/main/assets/cvae_fake_images.png)

### Real images

![alt cvae_real_images](https://github.com/AnixDrone/vae-vs-cgan/blob/main/assets/cvae_real_images.png)
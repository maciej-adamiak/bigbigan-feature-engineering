# Aerial imagery feature engineering using bidirectional generative adversarial networks 
## A case study of the Pilica river region, Poland

Maciej Adamiak (SoftwareMill, 02-791 Warsaw, Poland, [OCRID](https://orcid.org/0000-0002-8229-9661), [LinkedIn](https://www.linkedin.com/in/maciejadamiak/))

Krzysztof Będkowski University of Lodz, Faculty of Geographical Sciences, Institute of Urban Geography, Tourism and Geoinformation, 90-139 Łódź, Poland, [ORCID](0000-0001-7945-343X))

Anna Majchrowska (University of Lodz, Faculty of Geographical Sciences, Department of Physical Geography, 90-139 Łódź, Poland, [OCRID](https://orcid.org/0000-0002-1611-6118)) 

![](header.png)

## Abstract
Generative adversarial networks (GANs) are a type of neural network that are characterized by their unique construction and training process. Utilizing the concept of the latent space and exploiting the results of a duel between different GAN components opens up interesting opportunities for computer vision (CV) activities, such as image inpainting, style transfer, or even generative art. GANs have great potential to support aerial and satellite image interpretation activities. Carefully crafting a GAN and applying it to a high-quality dataset can result in nontrivial feature enrichment. In this study, we have designed and tested an unsupervised procedure capable of engineering new features by shifting real orthophotos into the GAN’s underlying latent space. Latent vectors are a low-dimensional representation of the orthophoto patches that hold information about the strength, occurrence, and interaction between spatial features discovered during the network training. Latent vectors were combined with geographical coordinates to bind them to their original location in the orthophoto. In consequence, it was possible to describe the whole research area as a set of latent vectors and perform further spatial analysis not on RGB images but on their lower-dimensional representation. To accomplish this goal, a modified version of the big bidirectional generative adversarial network (BigBiGAN) has been trained on a fine-tailored orthophoto imagery dataset covering the area of the Pilica River region in Poland. Trained models, precisely the generator and encoder, have been utilized during the processes of model quality assurance and feature engineering, respectively. Quality assurance was performed by measuring model reconstruction capabilities and by manually verifying artificial images produced by the generator. The feature engineering use case, on the other hand, has been presented in a real research scenario that involved splitting the orthophoto into a set of patches, encoding the patch set into the GAN latent space, grouping similar patches latent codes by utilizing hierarchical clustering, and producing a segmentation map of the orthophoto.

##

```python
from tensorflow.keras.models import load_model, Model
import tensorflow as tf

from train.blocks import pool_and_double_channels

self.model: Model = load_model('bignigan_encoder.h5', compile=False, custom_objects={
                        'tf': tf,
                        'pool_and_double_channels': pool_and_double_channels
                    })
```

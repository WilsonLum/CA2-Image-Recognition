# CA2-Image-Recognition
 NUS-ISS CA2 Project


 - Identifying insects correctly is critically important in entomological research, pest control, and insect-resource utilization. 
 - However, to identify an unknown species using traditional methods is very time consuming due to the vast number of insect species that must be identified and limited taxonomist resources. 
 - It is a problem that has been recognised for quite a while and there have been attempts for computer-based recognition systems to identify insect species automatically [1, 2].
 - In this project, we have chosen to focus on three insects that are vital for nature’s pollination – Butterflies, bees and moths. This is a challenging problem due to the large intra-class variations (e.g., pose and appearance changes), as well as small inter-class variations (i.e., there are only subtle differences in the overall appearance between classes). 
 - A sample of 1300 images of butterflies, moths and bees each, was split into training, test and validation sets. With this dataset, we employed deep convolutional neural networks (CNN) and Residual Network (ResNets) to extract the features of the 3 classes, and thereby reliably classify the species. We compared and evaluated several variations of the architecture with different depth of layers configurations for the CNN and ResNet frameworks and tuned the hyper-parameters to achieve the highest possible accuracy. 
 - In addition, to set a baseline to compare against, we also used pre-trained models for popular image architectures such as VGG16, ResNet50 and Inception V3, left out the last fully connected layer, froze all existing layers and added our custom classification layer to our number of classes.

# X-ray-image-recognition
Self-project

The project to give a True False tag given the product X-ray image and title text.
The model architecture is as follows:
1. Image engine: converts the given image to a vetor after fine-tuning pretrained weights.
2. Text engine: converts the title into a vector of fixed length.
3. Classifier: Combines the output of above two engines to get a concatenated vector of fix length and feeds it to a fully connected artificial neural network.

# Instructions to run 
The resnet18.py and resnet_model_states folder is used.
The resnet_model_states contains 3 models, for floor-1, floor1 and floor2.

1. The resnet18.py contains 3 functions. First, load the model states by indicating the path to the model state using load_model(path).
2. The loading of model should only be done once during model selection.
3. Now that the model is loaded, localisation can be done by passing a list of RSSI values containing 345 APs. This is done using the localisation()

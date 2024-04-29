"""
Model with ff layers only. Trained on PCA and normed data, so should not require preprocessing for prediction.
Taken from TearingAvoidance Repo. Model is written in Keras.
Preprocessing is defined in the reward function. This just predicts for preprocessed data.

Author: Rohit Sonker
"""

from typing import Tuple, Dict, Any
import numpy 
import tf_keras as keras
import os

class FFTearingProbabilityModel():
    def __init__(self,  
                 model_path:str, #= "/zfsauton2/home/rsonker/TearingAvoidance/tm_prediction_model",
                 is_ensemble:bool = True,
                 use_model_seed = 0):
        self.is_ensemble = is_ensemble
        self.model_path = model_path

        if self.is_ensemble:
            self.model = []
            for seed in range(5):
                model = keras.saving.load_model(os.path.join(model_path,f'best_model_{seed}'),
                                                compile=False)
                self.model.append(model)
        else:
            self.model = keras.saving.load_model(os.path.join(model_path,f'best_model_{use_model_seed}'), compile = False)

    
    def predict(self, x0: numpy.ndarray, x1:numpy.ndarray, combine_predictions = 'mean') -> numpy.ndarray:
        x = numpy.hstack([x0, x1])
        if self.is_ensemble:
            predictions = []
            for model in self.model:
                predictions.append(model.predict(x, verbose=0))
            predictions = numpy.array(predictions)
            return numpy.mean(predictions, axis=0)
        else:
            return self.model.predict(x, verbose = 0)
        


    

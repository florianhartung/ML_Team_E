import numpy as np
import pandas as pd
from pathlib import Path
from src.common import *

def mse(predict, actual):
    return np.mean((predict - actual) ** 2)

def r2(predict, actual):
    return 1 - np.sum((predict - actual) ** 2) / np.sum((actual - np.mean(actual)) ** 2)

def regression_comparison(
        modules:list,
        model_names:list,
        test_data:pd.DataFrame,
        models_dir:Path,
        particle_type:str
) -> pd.DataFrame:
    """
    param modules: A list of python modules that include a method load(path)->model and evaluate(dir, name, test_data_df, image_features, additional_features, target_features)->float
    param model_names: The names of the models.
    param particle_type: "protons" or "gammas"

    return: A table that includes a comparison between the models on different datasets
    """
    assert len(modules) == len(model_names)


    categories = ["m1", "m2", "m1_m2", "hillas", "stereo", "hillas_stereo"]
    categories_clear = ["Cleaned image m1", 
                        "Cleaned image m2", 
                        "Both cleaned images", 
                        "Both cleaned images and Hillas parameters", 
                        "Both cleaned images and Stereo parameters", 
                        "Both cleaned images, Hillas and Stereo parameters"]
    img_combinations = [[FEATURES_CLEAN_IMAGE_M1], 
                        [FEATURES_CLEAN_IMAGE_M2], 
                        [FEATURES_CLEAN_IMAGE_M1, FEATURES_CLEAN_IMAGE_M2],
                        [FEATURES_CLEAN_IMAGE_M1, FEATURES_CLEAN_IMAGE_M2],
                        [FEATURES_CLEAN_IMAGE_M1, FEATURES_CLEAN_IMAGE_M2],
                        [FEATURES_CLEAN_IMAGE_M1, FEATURES_CLEAN_IMAGE_M2]]
    add_combinations = [[], [], [],
                        FEATURES_HILLAS,
                        FEATURES_STEREO,
                        FEATURES_HILLAS + FEATURES_STEREO]

    results = pd.DataFrame(index=categories_clear, columns=model_names)

    for module, name in zip(modules, model_names):
        for category, cat_title, img_features, add_features in zip(categories, categories_clear, img_combinations, add_combinations):
            full_name = f"regression_{name}_{particle_type}_{category}"
            r2 = module.evaluate(
                models_dir,
                full_name,
                test_data,
                img_features,
                add_features,
                FEATURES_TRUE_SHOWER
            )
            results.loc[cat_title, name] = float(r2)

    return results
            
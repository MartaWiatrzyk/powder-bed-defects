import pandas as pd
import os
import cv2
import numpy as np
import xgboost as xgb
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import time
import logging
import pickle
from typing import Any, List, Tuple
import warnings


def load_images(categories: List[str], input_dir: str) -> Tuple[list, list]:
    """
    Load images and converts to lists: flatten images and categories
    :param categories: labels for images (ok/ nok)
    :param input_dir: input data directory
    :return: flatten images and corresponding labels
    """
    x_val = []
    y_val = []
    for i in categories:
        path = os.path.join(input_dir, i)
        for img in os.listdir(path):
            if img != '.gitkeep':
                image = cv2.imread(os.path.join(path, img))
                image_resized = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
                image_mono = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
                histogram, bin_edges = np.histogram(image_mono, bins=256, range=(0, 255))
                x_val.append(histogram)
                y_val.append(categories.index(i))
    return x_val, y_val


def load_parameters(model_type: str) -> dict:
    """
    Load parameters depending on selected model
    :param model_type: type of model (knn, random forest, svc, xgboost, adaboost)
    :return: parameters grid
    """
    model_conf: dict = {
        'rf': {'n_estimators': [10, 50, 100, 500, 1000], 'max_depth': [10], 'max_features': ['sqrt', 'log2'],
               'min_samples_split': [2, 5, 10]},
        'svm': {'C': [0.5, 1], 'gamma': [0.005, 0.01], 'kernel': ['poly']},
        'knn': {'n_neighbors': [5, 11, 15, 19, 21, 25], 'weights': ['uniform', 'distance'],
                'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'], 'leaf_size': [20, 30, 50], 'p': [1, 2]},
        'xgb': {'max_depth': [3, 4, 5, 6, 7, 8], 'learning_rate': [0.1, 0.01, 0.05, 0.005, 0.001, 0.0005],
                'n_estimators': [10, 50, 100, 100, 500, 1000], 'booster': ['gbtree', 'gblinear', 'dart'],
                'tree_metod': ['exact', 'approx', 'hist']},
        'abc': {}
    }
    try:
        param_grid = model_conf[model_type]
    except KeyError as ex:
        raise TypeError(f'model not recognised: {str(ex)}')
    return param_grid


def prepare_data(categories: List[str], input_dir: str, test_size: float) -> Tuple[Any, Any, Any, Any]:
    """
    Convert arrays to dataframe and splits dataset into train and test sets
    :param input_dir: input data directory
    :param categories: labels for images (ok/ nok)
    :param test_size: fractionating parameter for declaring test dataset
    :return: train and tests set (x - flattened image data, y - categories)
    """
    flat_data_arr, target_arr = load_images(categories, input_dir)
    flat_data = np.array(flat_data_arr)
    target = np.array(target_arr)
    df = pd.DataFrame(flat_data)
    df['Target'] = target

    # input data
    x = df.iloc[:, :-1]
    # output data
    y = df.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=50, stratify=y)
    return x_train, y_train, x_test, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame, param: dict, model_type: str) -> Any:
    """
    Train model using given hyperparameters
    :param x_train: train test (flattened image data)
    :param y_train: train categories
    :param param: hyperparameters
    :param model_type: type of model
    :return: trained model
    """
    if model_type == 'rf':
        model = RandomForestClassifier()
    elif model_type == 'svm':
        model = svm.SVC(probability=True)
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'xgb':
        model = xgb.XGBClassifier()
    elif model_type == 'abc':
        rf = RandomForestClassifier()
        model = AdaBoostClassifier(random_state=42, base_estimator=rf)
    else:
        print('model not found')

    best_model = RandomizedSearchCV(model, param_distributions=param, n_iter=5, cv=5, verbose=10)
    best_model.fit(x_train, y_train)
    return best_model


def test_model(x_test: pd.DataFrame, y_test: pd.DataFrame, model: Any, categories: List[str]) -> None:
    """
    Test the accuracy of trained model
    :param x_test: test set (flattened image data)
    :param y_test: test categories
    :param model: trained model
    :param categories: categories (from train and test sets)
    """
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)
    logging.info(f"The model is {accuracy * 100}% accurate")
    logging.info(classification_report(y_test, y_pred, target_names=categories))
    logging.info(('best params: ', model.best_params_))


def save_model(model: Any, filename: str):
    """
    Save trained model to a file
    :param model: trained model
    :param filename: name of the file in which the model is saved
    """
    model_path = os.path.join("models", filename)
    pickle.dump(model, open(model_path, 'wb'))


def validate_model(filename: str, categories: List[str], val_dir: str) -> float:
    """
    Validate trained model on validation set; load images from given directory, resize and get accuracy for each image
    and whole validation dataset
    :param filename: name of the file in which the model is saved
    :param categories: categories (ok/ nok)
    :param val_dir: path to a validation set
    :return: score of the model (based on the validation set)
    """
    model_path = os.path.join("models", filename)
    model = pickle.load(open(model_path, 'rb'))
    x_val, y_val = load_images(categories, val_dir)
    x_val = np.array(x_val)
    model_score = model.score(x_val, y_val)
    logging.info(('validation score: ', model_score))
    return model_score


def main():
    # choose model type: 'knn', 'svm', 'rf' (default), 'xgb' - xgboost or 'abc' - adaboost
    model_type: str = 'rf'

    categories: list[str] = ['ok', 'nok']
    input_dir: str = os.path.join('images', 'aug_test_train')
    val_dir = os.path.join('images', 'val')
    test_size: float = 0.40
    paths: list[str] = ['logs', 'models']

    # crete directory if it doesn't exist
    created_path: list[str] = []
    for path in paths:
        if not os.path.exists(path):
            created_path.append(path)
            os.makedirs(path)

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.pyplot').disabled = True
    warnings.filterwarnings('ignore')

    filename = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join("logs", filename + '_' + model_type + '_large_hist.log')
    logging.basicConfig(filename=file_path, encoding='utf-8', level=logging.DEBUG)

    logging.info(time.ctime())

    for new_paths in created_path:
        logging.info(f"Created directory: {new_paths}")

    logging.info(('model: ', model_type))
    start = time.time()

    param_grid = load_parameters(model_type)
    logging.info(('param: ', param_grid))

    x_train, y_train, x_test, y_test = prepare_data(categories, input_dir, test_size)
    logging.info(('prepare data: ', time.time() - start, ' s'))

    model = train_model(x_train, y_train, param_grid, model_type)
    logging.info(('train model: ', time.time() - start, ' s'))

    test_model(x_test, y_test, model, categories)
    logging.info(('test model: ', time.time() - start, ' s'))

    save_model(model, filename)
    logging.info(('save model: ', time.time() - start, ' s'))

    val = validate_model(filename, categories, val_dir)
    logging.info(('validate model: ', time.time() - start, ' s'))
    logging.info(time.ctime())


if __name__ == '__main__':
    main()

import torch
import cv2
import os
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Any
import time
import logging
import warnings
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


def load_images(categories: list[str], input_dir: str) -> tuple[list, list]:
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
            image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # mono
            image_resized = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
            image_tensor = torch.tensor(image_resized, dtype=torch.float32)  # Convert to tensor
            image_tensor = image_tensor.unsqueeze(0)
            x_val.append(image_tensor)
            y_val.append(categories.index(i))

    return x_val, y_val


def prepare_data(categories: list[str], input_dir: str, test_size: float, batch_size: int) -> \
        tuple[Any, Any, Any, int, Any, Any, Any, Any]:
    """
    Convert arrays to dataframe and splits dataset into train and test sets
    :param batch_size: size of batch in train_loader and test_loader
    :param input_dir: input data directory
    :param categories: labels for images (ok/ nok)
    :param test_size: fractionating parameter for declaring test dataset
    :return: train and tests set (x - flattened image data, y - categories)
    """
    flat_data_arr, target_arr = load_images(categories, input_dir)

    x_train, x_test, y_train, y_test = train_test_split(flat_data_arr, target_arr, test_size=test_size, random_state=50,
                                                        stratify=target_arr)
    x_train = torch.stack(x_train).cuda()
    x_test = torch.stack(x_test).cuda()
    y_train = torch.tensor(y_train, dtype=torch.long).cuda()
    y_test = torch.tensor(y_test, dtype=torch.long).cuda()

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    test_len = len(y_test)

    return x_train, y_train, x_test, test_len, train_loader, test_loader, train_dataset, test_dataset


def parametrize_model(model, lr):
    """
    Set criterion and optimizer
    :param model: type of model
    :param lr: learning rate value
    :return: criterion and optimizer
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    return criterion, optimizer


def train_model(model, criterion, optimizer, train_loader, test_loader, epochs, filename):
    """
    Train selected type of model
    :param model: type of model
    :param criterion: chosen type of criterion
    :param optimizer: chosen algorithm for optimization
    :param train_loader: training data
    :param test_loader: testing data
    :param epochs: number of epochs
    :param filename: name of log file
    :return: trained model
    """
    runs_dir = os.path.join('runs', filename)
    writer = SummaryWriter(log_dir=runs_dir)
    epochs = epochs
    train_losses = []
    train_correct = []

    best_val_accuracy = 0.0
    for i in range(epochs):
        trn_corr = 0
        model.train()

        for b, (X_train, y_train) in enumerate(train_loader):
            y_pred = model(X_train)
            loss_train = criterion(y_pred, y_train)
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

        train_losses.append(loss_train.item())
        train_correct.append(trn_corr.item() / len(train_loader.dataset))

        # run testing batches
        model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for X_test, y_test in test_loader:
                y_val = model(X_test)
                test_loss += criterion(y_val, y_test).item()

                predicted = torch.max(y_val.data, 1)[1]
                correct += (predicted == y_test).sum().item()

        current_val_accuracy = correct / len(test_loader.dataset)
        # if accuracy is higher, save model
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            save_dir = os.path.join('models', filename + '.pt')
            torch.save(model, save_dir)

        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': loss_train.item(), 'Validation': test_loss / len(test_loader.dataset)},
                           i * len(train_loader.dataset) + i)

        logging.info(('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(i, loss_train.item(), test_loss
                                                                                     / len(test_loader.dataset))))
        logging.info(f'test: {correct} out of {len(test_loader.dataset)} = {100 * correct / len(test_loader.dataset):.2f}% correct')

    # load the best model after training
    best_save_dir = os.path.join('models', filename + '.pt')
    best_model = torch.load(best_save_dir)
    return best_model


def test_model(model, test_len, test_loader):
    """
    Test final model
    :param model: trained model
    :param test_len: test set size
    :param test_loader: test data
    """
    correct = 0
    with torch.no_grad():
        for b, (test_data, test_target) in enumerate(test_loader):
            y_predicted = model(test_data)
            predicted = torch.max(y_predicted.data, 1)[1]
            correct += (predicted == test_target).sum()
    logging.info(f'test: {correct} out of {test_len} = {100 * correct / test_len:.2f}% correct')


def save_model(model: Any, filename: str):
    """
    Save model to a file
    :param model: trained model
    :param filename: name of the file in which the model is saved
    """
    save_path = os.path.join("models", filename + ".pt")
    torch.save(model, save_path)


def validate_model(filename: str, categories: list[str], val_dir: str, batch_size) -> float:
    """
    Validate trained model on validation set; load images from given directory, resize and get accuracy for each image
    and whole validation dataset
    :param batch_size: size of batch
    :param filename: name of the file in which the model is saved
    :param categories: categories (ok/ nok)
    :param val_dir: path to a validation set
    :return: score of the model (based on the validation set)
    """
    load_dir = os.path.join('models', filename + '.pt')
    model = torch.load(load_dir)

    model.eval()
    x_val, y_val = load_images(categories, val_dir)

    x_val = torch.stack(x_val).cuda()
    y_val = torch.tensor(y_val, dtype=torch.long).cuda()

    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    correct_val = 0
    with torch.no_grad():
        for b, (val_data, val_target) in enumerate(val_loader):
            y_predicted = model(val_data)
            predicted = torch.max(y_predicted.data, 1)[1]
            correct_val += (predicted == val_target).sum()

    logging.info(f'val: {correct_val} out of {len(y_val)} = {100 * correct_val / len(y_val):.2f}% correct')

    return correct_val


def main():
    # choose type of model: 'resnet18', 'resnet50' or 'vgg16'
    model_type: str = 'resnet18'
    categories: list[str] = ['ok', 'nok']
    input_dir: str = os.path.join('images', 'aug_test_train')
    val_dir: str = os.path.join('images', 'val')

    test_size: float = 0.40
    epochs: int = 100
    batch_size: int = 50

    # crete directory if it doesn't exist
    paths: list[str] = ['logs', 'models', 'runs']
    created_path: list[str] = []
    for path in paths:
        if not os.path.exists(path):
            created_path.append(path)
            os.makedirs(path)

    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.pyplot').disabled = True
    warnings.filterwarnings('ignore')

    filename = time.strftime("%Y%m%d-%H%M%S")
    file_path = os.path.join("logs", filename + '_' + model_type + '_img_mono_gpu_batch' + str(batch_size) + '.log')
    logging.basicConfig(filename=file_path, encoding='utf-8', level=logging.DEBUG)

    logging.info((time.ctime()))
    for new_paths in created_path:
        logging.info(f"Created directory: {new_paths}")

    start = time.time()
    logging.info(('gpu: ', torch.cuda.get_device_name(0)))

    seed = 40
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    learning_rate = [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
    for lr in learning_rate:
        if model_type == 'resnet18' or model_type == 'resnet50':
            if model_type == 'resnet18':
                model = models.resnet18(pretrained=True).cuda()
            else:
                model = models.resnet50(pretrained=True).cuda()
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # mono images
            model.state_dict()['conv1.weight'] = model.state_dict()['conv1.weight'].sum(dim=1, keepdim=True)
            ft = model.fc.in_features
            model.fc = nn.Linear(ft, 2)  # set number of outputs to 2 (ok or nok)
            model = model.to('cuda')

        elif model_type == 'vgg16':
            model = models.vgg16(pretrained=True).cuda()
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, 2)  # set number of outputs to 2 (ok or nok)
            # mono images
            model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
            pretrained_state_dict = models.vgg16(pretrained=True).state_dict()
            modified_state_dict = model.state_dict()
            modified_state_dict['features.0.weight'][:, 0] = pretrained_state_dict['features.0.weight'].sum(dim=1)
            model = model.to('cuda')

        logging.info(('model: ', model))
        logging.info(('model time: ', time.time() - start))

        x_train, y_train, x_test, test_len, train_loader, test_loader, train_dat, test_dat = prepare_data(
            categories, input_dir, test_size, batch_size)

        criterion, optimizer = parametrize_model(model, lr)
        logging.info(('prepare data time: ', time.time() - start))

        trained_model = train_model(model, criterion, optimizer, train_loader, test_loader, epochs, filename)

        logging.info(('training time: ', time.time() - start))

        test_model(trained_model, test_len, test_loader)

        save_model(trained_model, filename)
        correct_val = validate_model(filename, categories, val_dir, batch_size)
        logging.info(('learning rate: ', lr))

    logging.info((time.ctime()))
    logging.info(('time: ', time.time() - start))


if __name__ == '__main__':
    main()

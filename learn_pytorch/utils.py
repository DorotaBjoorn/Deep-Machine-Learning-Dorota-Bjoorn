# https://www.learnpytorch.io

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Dict

#------------ data---------------------------------------------------------------------------
class make_dataset(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)
    self.length = self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]

  def __len__(self):
    return self.length # len(self.y)
#-------------------------------------------------------------------------------------------
  

# ---------------metrics--------------------------------------------------------------------
# check also https://lightning.ai/docs/torchmetrics/stable/

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred))
    return acc
# ------------------------------------------------------------------------------------------

# ---------------training and evaluation loops-------------------------------------------------------------------------------------------
# inspired by https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
# train loop to be run for training with train_loader and several epochs
# possibly y_pred.squeeze() instad of batch_y.unsqueeze(1) as in https://www.learnpytorch.io/02_pytorch_classification/
def train_loop(dataloader, model, loss_fn, optimizer):
  model.train()
  num_samples = len(dataloader.dataset)
  running_train_loss, correct_predictions = 0.0, 0
  
  for batch, (batch_X, batch_y) in enumerate(dataloader):
      y_pred = model.forward(batch_X) # fit model; y_pred has shape [batchsize, 1]
      train_loss = loss_fn(y_pred, batch_y.unsqueeze(1)) # calculate loss; unsqueeze(1): shape [batchsize, ] -> shape [batchsize, 1]
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()
  
      running_train_loss += train_loss.item() * len(batch_X) # alt train_loss.detach().numpy()
      correct_predictions += torch.sum(y_pred.round() == batch_y.unsqueeze(1)).item() # for binary classification
      # correct_predictions += (y_pred.argmax(1) == batch_y).type(torch.float).sum().item() for multiple classes classification
  
  epoch_train_loss = running_train_loss / num_samples
  epoch_train_accuracy = correct_predictions / num_samples

  return epoch_train_loss, epoch_train_accuracy


# eval_loop to be run for validation with val_loader and several epochs
# eval_loop to be run for test with test_loader and not in epoch loop
def eval_loop(dataloader, model, loss_fn):
    model.eval()
    num_samples = len(dataloader.dataset)
    running_loss, correct_predictions = 0.0, 0

    with torch.inference_mode():           # new syntax replacing torch.no_grad()
        for batch_X, batch_y in dataloader:
            y_pred = model.forward(batch_X)
            loss = loss_fn(y_pred, batch_y.unsqueeze(1))

            running_loss += loss.item() * len(batch_X) # alt train_loss.detach().numpy()
            correct_predictions += torch.sum(y_pred.round() == batch_y.unsqueeze(1)).item() # for binary classification
            # correct_predictions += (y_pred.argmax(1) == batch_y).type(torch.float).sum().item() for multiple classes classification

    epoch_loss = running_loss / num_samples
    epoch_accuracy = correct_predictions / num_samples

    return epoch_loss, epoch_accuracy


# Alt from https://www.learnpytorch.io/03_pytorch_computer_vision/
def train_step(model, data_loader, loss_fn, optimizer, accuracy_fn, device):
    accum_batch_loss, accum_batch_acc = 0, 0
    model.to(device)
    # model.train()
    
    for batch, (batch_X, batch_y) in enumerate(data_loader):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        y_pred = model(batch_X)
        loss = loss_fn(y_pred, batch_y) # batch loss
        accum_batch_loss += loss * len(batch_y) 
        accum_batch_acc += accuracy_fn(y_pred.argmax(dim=1), batch_y) * len(batch_y) # batch acc * batch size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    step_train_loss = accum_batch_loss / len(data_loader.dataset)
    step_train_acc = accum_batch_acc / len(data_loader.dataset)
    print(f"Train loss: {step_train_loss:.5f} | Train accuracy: {step_train_acc:.3f}")
    return step_train_loss, step_train_acc


def test_step(data_loader, model, loss_fn, accuracy_fn, device):
    test_loss, test_acc = 0, 0
    model.to(device)
    model.eval()

    with torch.inference_mode(): 
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            y_pred = model(batch_X)
            test_loss += loss_fn(y_pred, batch_y) * len(batch_y)
            test_acc += accuracy_fn(y_pred.argmax(dim=1), batch_y) * len(batch_y) # logits -> pred labels
        
        step_test_loss = test_loss / len(data_loader.dataset)
        step_test_acc = test_acc / len(data_loader.dataset)
        print(f"Test loss: {step_test_loss:.5f} | Test accuracy: {step_test_acc:.3f}\n")
    
    return step_test_loss, step_test_acc


# Alt simpel from https://www.learnpytorch.io/04_pytorch_custom_datasets/
def train_step2(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step2(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device):
    model.eval() 
    test_loss, test_acc = 0, 0
    model.to(device)
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # verage loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    history = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_step2(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step2(model=model, dataloader=test_dataloader, loss_fn=loss_fn)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    return history

def pred_and_plot_image(model: torch.nn.Module, 
                        image_path: str,
                        device: torch.device, 
                        class_names: List[str] = None, 
                        transform=None):
    """Makes a prediction on a target image and plots the image with its prediction."""
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image = target_image / 255. 
    if transform:
        target_image = transform(target_image)
    
    model.to(device)
    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0) # add batch dim to image
        target_image_pred = model(target_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    plt.imshow(target_image.squeeze().permute(1, 2, 0)) # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False);
#-----------------------------------------------------------------------------------------------------



#-----------visualization--------------------------------------------------------
def plot_data_and_preds_regression(x_train, y_train, x_val, y_val, x_test, y_test, y_pred=None):
  """
  Plots training data, test data and compares predictions where x and y are corrdinates.
  Train, val, test and predicitoins are plotted in different colours.
  """
  plt.figure(figsize=(10, 7))
  plt.scatter(x_train, y_train, c="b", s=4, label="Training data")
  plt.scatter(x_val, y_val, c="b", s=4, label="Validerings data")
  plt.scatter(x_test, y_test, c="g", s=4, label="Testing data")
  if y_pred is not None:
    plt.scatter(x_test, y_pred, c="r", s=4, label="Predictions")
  plt.legend(prop={"size": 14});


def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, test_loss, test_accuracy):
  """
  Visualization of training behaviour and test results
  """    
  plt.figure(figsize=(10, 3))

  plt.subplot(1, 2, 1)
  plt.plot(train_losses, label='train')
  plt.plot(val_losses, label='val')
  plt.title(f'Loss (test loss = {test_loss:.4f})')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')

  plt.subplot(1, 2, 2)
  plt.plot(train_accuracies, label='train')
  plt.plot(val_accuracies, label='val')
  plt.title(f'Accuracy (test accuracy = {test_accuracy:.4f})')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')

  plt.legend()
  plt.show()

  # alt from https://www.learnpytorch.io/04_pytorch_custom_datasets/
  def plot_training_history2(history: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    loss = history['train_loss']
    test_loss = history['test_loss']
    accuracy = history['train_acc']
    test_accuracy = history['test_acc']
    epochs = range(len(history['train_loss']))

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

#---------------------------------------------------------------------------------
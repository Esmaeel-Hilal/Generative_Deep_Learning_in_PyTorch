# Resources
# https://github.com/mrdbourke/pytorch-deep-learning/blob/main/going_modular/going_modular
import torch
from typing import Dict, List, Tuple
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               compute_accuracy: bool = True,
               target_fn = lambda X, y: y) -> Tuple[float, float]:
    """
    Trains a PyTorch model for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader providing (X, y) batches.
        loss_fn: The loss function to minimize.
        optimizer: The optimizer for updating model parameters.
        device: Device to run training on ("cpu" or "cuda").
        compute_accuracy: Whether to compute accuracy (skip for models like autoencoders).
        target_fn: Function to extract the appropriate target from (X, y). 
                   Default is identity for standard supervised learning.

    Returns:
        A tuple of (average training loss, average training accuracy).
    """
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0.0, 0.0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        target = target_fn(X, y)

        # 1. Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, target) # Calculate the loss
        train_loss += loss.item() # Accumulate the loss

        # 2. Back Propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        if compute_accuracy:
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader) if compute_accuracy else 0.0

    return train_loss, train_acc


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              compute_accuracy: bool = True,
              target_fn = lambda X, y: y) -> Tuple[float, float]:
    
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        compute_accuracy: Whether to compute accuracy (skip for models like autoencoders).
        target_fn: Function to extract the appropriate target from (X, y). 
                   Default is identity for standard supervised learning.

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:
        
        (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0.0, 0.0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            target = target_fn(X, y)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, target)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            if compute_accuracy:
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader) if compute_accuracy else 0.0

    return test_loss, test_acc


def save_best_model(model, optimizer, loss, epoch, best_loss, path="./checkpoint.pth"):
    if loss < best_loss:
        best_loss = loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, path)
        print(f"Saved better model at epoch {epoch} with loss {loss:.4f}")
    return best_loss


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          compute_accuracy: bool = True,
          target_fn = lambda X, y: y,
          save_best: bool = False,
          checkpoint_path: str = "./checkpoint.pth",
          use_writer: bool = False,
          writer_log_dir: str = "./logs") -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      
    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for 
      each epoch.
      In the form: {train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...]} 
      For example if training for epochs=2: 
              {train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973]} 
    """
    # Move the model to the device
    model.to(device)

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Create a writer with all default settings
    writer = SummaryWriter()

    writer = SummaryWriter(log_dir=writer_log_dir) if use_writer else None


    # Initialize best_loss variable
    best_loss = float("inf")

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           compute_accuracy=compute_accuracy,
                                           target_fn=target_fn)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device,
                                        compute_accuracy=compute_accuracy,
                                        target_fn=target_fn)

        if save_best:
            best_loss = save_best_model(model=model, 
                                        optimizer=optimizer, 
                                        loss=test_loss, 
                                        epoch=epoch, 
                                        best_loss = best_loss,
                                        path=checkpoint_path)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        ### New: Experiment tracking ###
        # Add loss results to SummaryWriter
        if use_writer:
            writer.add_scalars(main_tag="Loss", 
                            tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                            global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars(main_tag="Accuracy", 
                            tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                            global_step=epoch)
            
            # Track the PyTorch model architecture
            # Only add graph on the first epoch to avoid redundancy and performance issues
            if epoch == 0:
                X, _ = next(iter(train_dataloader))
                dummy_input = X[:1].to(device)  # batch of size 1
                writer.add_graph(model=model, input_to_model=dummy_input)

    
    # Close the writer
    writer.close()
    
    ### End new ###

    # Return the filled results at the end of the epochs
    return results
import torch
import torch.nn as nn
from pathlib import Path
import argparse
from pi_learning_for_collaborative_carrying.mann_pytorch.utils import read_from_file
import numpy as np
import os


def convert_model(model_path: Path, onnx_model_path: Path, normalization_folder: Path, opset_version: int):
    # Restore the model with the trained weights
    mann_restored = torch.load(str(model_path))

    # Set dropout and batch normalization layers to evaluation mode before running inference
    mann_restored.eval()
    input_size = next(mann_restored.parameters()).size()[1]

    # Here we create two layes for normalization and denormalization
    X_mean = read_from_file(str(normalization_folder / "X_mean.txt"))
    X_std = read_from_file(str(normalization_folder / "X_std.txt"))
    X_std[np.where(X_std <= 1e-4)] = 1

    # Clip inputs to exclude the stuff only used for loss function calculations
    X_mean = X_mean[:input_size]
    X_std = X_std[:input_size]

    Y_mean = read_from_file(str(normalization_folder / "Y_mean.txt"))
    Y_std = read_from_file(str(normalization_folder / "Y_std.txt"))
    Y_std[np.where(Y_std <= 1e-4)] = 1

    # Choose the device to use
    device = torch.device("cuda:0" if (next(mann_restored.parameters()).get_device() == 0) else "cpu")

    # the normalization is
    # x_norm = (x - x_mean) / x_std
    # it is possible to convert it in a linear layer by massaging the equation
    # x_norm = x / x_std - x_mean / x_std
    lin_normalization = nn.Linear(input_size, input_size)
    with torch.no_grad():
        lin_normalization.weight.copy_(torch.tensor(np.diag(np.reciprocal(X_std))))
        lin_normalization.bias.copy_(torch.tensor(-X_mean / X_std))

    # the denormalization is
    # y = y_std * y_norm + y_mean
    lin_output_denormalization = nn.Linear(Y_mean.size, Y_mean.size)
    with torch.no_grad():
        lin_output_denormalization.weight.copy_(torch.diag(torch.tensor(Y_std)))
        lin_output_denormalization.bias.copy_(torch.tensor(Y_mean))

    # The extended model contains the normalization and the denormalization
    extended_model = nn.Sequential(lin_normalization,
                                   mann_restored,
                                   lin_output_denormalization)

    # Input to the model
    batch_size = 1
    x = torch.randn(batch_size, input_size, requires_grad=True)

    # Ensure all tensors are on the same device
    x = x.to(device)
    lin_normalization.to(device)
    lin_output_denormalization.to(device)

    # Export the model
    torch.onnx.export(extended_model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      str(onnx_model_path),  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=opset_version,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})


def main():
    parser = argparse.ArgumentParser(description='Convert mann-pytorch model into a onnx model.')
    parser.add_argument('--torch_training_path', '-i', type=lambda p: Path(p).absolute(),
                        default="../datasets/trained_models/training_test_collab_20241122-110309/",
                        required=False,
                        help='Pytorch training folder location.')
    parser.add_argument('--model_epoch', type=int, required=False, default=149,
                        help='Model epoch to convert.')
    parser.add_argument('--onnx_opset_version', type=int, default=12, required=False,
                        help='The ONNX version to export the model to. At least 12 is required.')
    args = parser.parse_args()

    training_path = args.torch_training_path
    ep = args.model_epoch
    model_path = training_path / Path("models/model_" + str(ep) + ".pth")
    normalization_path = training_path / "normalization"
    onnx_filename = os.path.basename(os.path.normpath(training_path)) + "_ep" + str(ep) + ".onnx"
    onnx_model_path = training_path / Path(onnx_filename)

    print("Converting model from:", model_path)
    print("Normalization folder:", normalization_path)
    print("Saving onnx model to:", onnx_model_path)
    input("continue")

    convert_model(model_path=model_path,
                  onnx_model_path=onnx_model_path,
                  normalization_folder=normalization_path,
                  opset_version=args.onnx_opset_version)


if __name__ == "__main__":
    main()

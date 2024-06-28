import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import _functional as F
import json
import copy
from PIL import Image
from random import randint
from tqdm import tqdm
import wandb
import os
from datetime import datetime
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, create_gaussian_cloud, get_random_element, load_timestep_captures


class Configuration:
    def __init__(self):
        self.data_directory_path = "./data/"
        self.output_directory_path = "./output/"
        self.sequence_name = "basketball"
        self.experiment_id = datetime.utcnow().isoformat() + "Z"

    def set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)
        self.output_directory_path = os.path.abspath(self.output_directory_path)

    def update_from_arguments(self, args):
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        self.set_absolute_paths()

    def update_from_json(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        self.set_absolute_paths()


def parse_arguments(configuration: Configuration):
    argument_parser = argparse.ArgumentParser(description="???")
    argument_parser.add_argument(
        "--config_file", type=str, help="Path to the JSON config file"
    )

    for key, value in vars(configuration).items():
        t = type(value)
        if t == bool:
            argument_parser.add_argument(f"--{key}", default=value, action="store_true")
        else:
            argument_parser.add_argument(f"--{key}", default=value, type=t)

    return argument_parser.parse_args()


def calculate_loss(gaussian_cloud_parameters: dict[str, torch.nn.Parameter], batch):
    gaussian_cloud = create_gaussian_cloud(gaussian_cloud_parameters)
    # gaussian_cloud['means2D'].retain_grad()

    (
        rendered_image,
        _,
        _,
    ) = Renderer(
        raster_settings=batch["cam"]
    )(**gaussian_cloud)
    return torch.nn.functional.l1_loss(rendered_image, batch["im"])


def load_and_freeze_parameters(path: str):
    parameters = torch.load(path)
    for parameter in parameters.values():
        parameter.requires_grad = False
    return parameters


class MLP(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size + 2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, input_size)

        self.relu = nn.ReLU()

        self.embedding = nn.Embedding(embedding_size, 2)

    def forward(self, input_tensor, timestep):
        batch_size = input_tensor.shape[0]
        initial_input_tensor = input_tensor
        embedding_tensor = self.embedding(timestep).repeat(batch_size, 1)
        input_with_embedding = torch.cat((input_tensor, embedding_tensor), dim=1)

        x = self.relu(self.fc1(input_with_embedding))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        return initial_input_tensor + x


def train():
    dataset_metadata = json.load(
        open(
            os.path.join(
                config.data_directory_path, config.sequence_name, "train_meta.json"
            ),
            "r",
        )
    )
    sequence_length = 20  # len(dataset_metadata["fn"])
    parameters = load_and_freeze_parameters("params.pth")

    mlp = MLP(7, sequence_length).cuda()
    optimizer = torch.optim.Adam(params=mlp.parameters(), lr=1e-3)

    for timestep in range(sequence_length):
        dataset = load_timestep_captures(timestep, dataset_metadata)
        timestep_capture_buffer = []

        for _ in tqdm(range(10_000)):
            capture = get_random_element(input_list=timestep_capture_buffer, fallback_list=dataset)
            updated_parameters = update_parameters(mlp, parameters, timestep)
            loss = calculate_loss(updated_parameters, capture)
            wandb.log(
                {
                    f"loss-{timestep}": loss.item(),
                }
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        timestep_capture_buffer = dataset.copy()
        losses = []
        while timestep_capture_buffer:
            with torch.no_grad():
                capture = get_random_element(input_list=timestep_capture_buffer, fallback_list=dataset)
                loss = calculate_loss(updated_parameters, capture)
                losses.append(loss.item())

        wandb.log({f"mean-losses": sum(losses) / len(losses)})

    # Random Training
    datasets = []
    for timestep in range(sequence_length):
        datasets.append(load_timestep_captures(timestep, dataset_metadata))
    for _ in tqdm(range(10_000)):
        di = torch.randint(0, len(datasets), (1,))
        si = torch.randint(0, len(datasets[0]), (1,))
        X = datasets[di][si]

        updated_parameters = update_parameters(mlp, parameters, di)

        loss = calculate_loss(updated_parameters, X)

        wandb.log(
            {
                f"loss-new": loss.item(),
            }
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    for dataset in datasets:
        losses = []
        with torch.no_grad():
            for X in dataset:
                loss = calculate_loss(updated_parameters, X)
                losses.append(loss.item())

        wandb.log({f"mean-losses-new": sum(losses) / len(losses)})


def update_parameters(mlp: MLP, parameters, timestep):
    delta = mlp(
        torch.cat((parameters["means"], parameters["rotations"]), dim=1),
        torch.tensor(timestep).cuda(),
    )
    means_delta = delta[:, :3]
    rotations_delta = delta[:, 3:]
    learning_rate = 0.01
    updated_parameters = copy.deepcopy(parameters)
    updated_parameters["means"] = updated_parameters["means"].detach()
    updated_parameters["means"] += means_delta * learning_rate
    updated_parameters["rotations"] = updated_parameters["rotations"].detach()
    updated_parameters["rotations"] += rotations_delta * learning_rate
    return updated_parameters


if __name__ == "__main__":
    config = Configuration()
    arguments = parse_arguments(config)

    if arguments.config_file:
        config.update_from_json(arguments.config_json)
    else:
        config.update_from_arguments(arguments)

    wandb.init(project="new-dynamic-gaussians")
    train()

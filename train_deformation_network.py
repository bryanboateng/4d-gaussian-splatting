import os
from dataclasses import dataclass, MISSING

import numpy as np
import torch
from torch import nn
import json
import copy
from PIL import Image
from random import randint
from tqdm import tqdm
import wandb

from command import Command
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera
from train_commons import load_timestep_captures, get_random_element, Capture


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


class DeformationNetwork(nn.Module):
    def __init__(self, input_size, sequence_length) -> None:
        super(DeformationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size + 2, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, input_size)

        self.relu = nn.ReLU()

        self.embedding = nn.Embedding(sequence_length, 2)

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


@dataclass
class Xyz(Command):
    data_directory_path: str = MISSING
    sequence_name: str = MISSING

    @staticmethod
    def _load_and_freeze_parameters(path: str):
        parameters = torch.load(path)
        for parameter in parameters.values():
            parameter.requires_grad = False
        return parameters

    @staticmethod
    def _update_parameters(
        deformation_network: DeformationNetwork, parameters, timestep
    ):
        delta = deformation_network(
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

    @staticmethod
    def _create_gaussian_cloud(parameters):
        return {
            "means3D": parameters["means"],
            "colors_precomp": parameters["colors"],
            "rotations": torch.nn.functional.normalize(parameters["rotations"]),
            "opacities": torch.sigmoid(parameters["opacities"]),
            "scales": torch.exp(parameters["scales"]),
            "means2D": torch.zeros_like(
                parameters["means"], requires_grad=True, device="cuda"
            )
            + 0,
        }

    @staticmethod
    def get_loss(parameters, target_capture: Capture):
        gaussian_cloud = Xyz._create_gaussian_cloud(parameters)
        # gaussian_cloud['means2D'].retain_grad()
        (
            rendered_image,
            _,
            _,
        ) = Renderer(
            raster_settings=target_capture.camera.gaussian_rasterization_settings
        )(**gaussian_cloud)
        return torch.nn.functional.l1_loss(rendered_image, target_capture.image)

    def _set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)

    def run(self):
        wandb.init(project="new-dynamic-gaussians")
        dataset_metadata = json.load(
            open(
                os.path.join(
                    self.data_directory_path,
                    self.sequence_name,
                    "train_meta.json",
                ),
                "r",
            )
        )
        sequence_length = len(dataset_metadata["fn"])
        parameters = self._load_and_freeze_parameters("params.pth")
        deformation_network = DeformationNetwork(7, sequence_length).cuda()
        optimizer = torch.optim.Adam(params=deformation_network.parameters(), lr=1e-3)

        for timestep in range(sequence_length):
            timestep_captures = load_timestep_captures(
                dataset_metadata, timestep, self.data_directory_path, self.sequence_name
            )
            timestep_capture_buffer = []

            for _ in tqdm(range(10_000)):
                capture = get_random_element(
                    input_list=timestep_capture_buffer, fallback_list=timestep_captures
                )
                updated_parameters = self._update_parameters(
                    deformation_network, parameters, timestep
                )
                loss = self.get_loss(updated_parameters, capture)

                wandb.log(
                    {
                        f"loss-{timestep}": loss.item(),
                    }
                )

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            timestep_capture_buffer = timestep_captures.copy()
            losses = []
            while timestep_capture_buffer:
                with torch.no_grad():
                    capture = get_batch(timestep_capture_buffer, timestep_captures)

                    loss = self.get_loss(updated_parameters, capture)
                    losses.append(loss.item())

            wandb.log({f"mean-losses": sum(losses) / len(losses)})
        ## Random Training
        timestep_captures = []
        for timestep in range(sequence_length):
            timestep_captures += [
                load_timestep_captures(
                    dataset_metadata,
                    timestep,
                    self.data_directory_path,
                    self.sequence_name,
                )
            ]
        for i in tqdm(range(10_000)):
            di = torch.randint(0, len(timestep_captures), (1,))
            si = torch.randint(0, len(timestep_captures[0]), (1,))
            capture = timestep_captures[di][si]

            updated_parameters = self._update_parameters(
                deformation_network, parameters, timestep
            )

            loss = self.get_loss(updated_parameters, capture)

            wandb.log(
                {
                    f"loss-new": loss.item(),
                }
            )

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        for d in timestep_captures:
            losses = []
            with torch.no_grad():
                for capture in d:
                    loss = self.get_loss(updated_parameters, capture)
                    losses.append(loss.item())

            wandb.log({f"mean-losses-new": sum(losses) / len(losses)})


def main():
    Xyz().parse_args().run()


if __name__ == "__main__":
    main()

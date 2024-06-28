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


def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md["fn"][t])):
        w, h, k, w2c = md["w"], md["h"], md["k"][t][c], md["w2c"][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md["fn"][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(
            copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))
        ).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({"cam": cam, "im": im, "seg": seg_col, "id": c})
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def params2rendervar(params):
    rendervar = {
        "means3D": params["means"],
        "colors_precomp": params["colors"],
        "rotations": torch.nn.functional.normalize(params["rotations"]),
        "opacities": torch.sigmoid(params["opacities"]),
        "scales": torch.exp(params["scales"]),
        "means2D": torch.zeros_like(params["means"], requires_grad=True, device="cuda")
        + 0,
    }
    return rendervar


def get_loss(params, batch):
    rendervar = params2rendervar(params)
    # rendervar['means2D'].retain_grad()

    (
        im,
        _,
        _,
    ) = Renderer(
        raster_settings=batch["cam"]
    )(**rendervar)
    loss = torch.nn.functional.l1_loss(im, batch["im"])

    return loss


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
            dataset = get_dataset(timestep, dataset_metadata, self.sequence_name)
            dataset_queue = []

            for i in tqdm(range(10_000)):
                X = get_batch(dataset_queue, dataset)

                delta = deformation_network(
                    torch.cat((parameters["means"], parameters["rotations"]), dim=1),
                    torch.tensor(timestep).cuda(),
                )
                delta_means = delta[:, :3]
                delta_rotations = delta[:, 3:]

                l = 0.01
                updated_params = copy.deepcopy(parameters)
                updated_params["means"] = updated_params["means"].detach()
                updated_params["means"] += delta_means * l
                updated_params["rotations"] = updated_params["rotations"].detach()
                updated_params["rotations"] += delta_rotations * l

                loss = get_loss(updated_params, X)

                wandb.log(
                    {
                        f"loss-{timestep}": loss.item(),
                    }
                )

                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

            dataset_queue = dataset.copy()
            losses = []
            while dataset_queue:
                with torch.no_grad():
                    X = get_batch(dataset_queue, dataset)

                    loss = get_loss(updated_params, X)
                    losses.append(loss.item())

            wandb.log({f"mean-losses": sum(losses) / len(losses)})
        ## Random Training
        dataset = []
        for timestep in range(sequence_length):
            dataset += [get_dataset(timestep, dataset_metadata, self.sequence_name)]
        for i in tqdm(range(10_000)):
            di = torch.randint(0, len(dataset), (1,))
            si = torch.randint(0, len(dataset[0]), (1,))
            X = dataset[di][si]

            delta = deformation_network(
                torch.cat((parameters["means"], parameters["rotations"]), dim=1),
                torch.tensor(timestep).cuda(),
            )
            delta_means = delta[:, :3]
            delta_rotations = delta[:, 3:]

            l = 0.01
            updated_params = copy.deepcopy(parameters)
            updated_params["means"] = updated_params["means"].detach()
            updated_params["means"] += delta_means * l
            updated_params["rotations"] = updated_params["rotations"].detach()
            updated_params["rotations"] += delta_rotations * l

            loss = get_loss(updated_params, X)

            wandb.log(
                {
                    f"loss-new": loss.item(),
                }
            )

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        for d in dataset:
            losses = []
            with torch.no_grad():
                for X in d:
                    loss = get_loss(updated_params, X)
                    losses.append(loss.item())

            wandb.log({f"mean-losses-new": sum(losses) / len(losses)})


def main():
    Xyz().parse_args().run()


if __name__ == "__main__":
    main()

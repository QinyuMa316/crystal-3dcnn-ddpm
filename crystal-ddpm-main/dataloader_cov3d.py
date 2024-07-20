from pymatgen.core import Structure
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

GRID_SIZE = 64
# NUM_ATOM = 84
def get_3d_grid(structure, grid_size=GRID_SIZE): #, num_atom=NUM_ATOM):
    # num_atom = num_atom
    atom_coords = structure.frac_coords
    atom_types = structure.atomic_numbers
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)
    for i, coord in enumerate(atom_coords):
        atomic_number = atom_types[i]
        x, y, z = coord
        grid_x = int(np.round(x * (grid_size - 1)))
        grid_y = int(np.round(y * (grid_size - 1)))
        grid_z = int(np.round(z * (grid_size - 1)))
        grid[grid_x, grid_y, grid_z] = atomic_number
    return grid


class CrystalDataset(Dataset):
    def __init__(self, structures, labels = None, grid_size=GRID_SIZE):
        self.structures = structures
        self.labels = labels
        self.grid_size = grid_size

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        structure = self.structures[idx]
        grid = get_3d_grid(structure, self.grid_size)
        grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0) # 增加channel=1
        if self.labels:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float32)
            return grid, label
        return grid

# if __name__ == '__main__':
    # import json
    # def read_entry_dict_from_json(filename):
    #     with open(filename, 'r', encoding='utf-8') as file:
    #         entry_dict = json.load(file)
    #     new_entry_dict = {}  # entry_dict['structure_type'].type() = pymatgen.obj
    #     for key, entry in entry_dict.items():
    #         new_entry_dict[key] = {'structure': Structure.from_dict(entry['structure']),
    #                                'formation_energy': entry['formation_energy'],
    #                                'ehull': entry['ehull']}
    #     return new_entry_dict
    # type_ = 'P2'
    # filename = f'{type_}_NaTMO2_2tm_final.json'
    # entry_dict = read_entry_dict_from_json(filename)
    # print(f'Number of entries: {len(entry_dict)}')
    #
    # structures = []
    # for key, entry in entry_dict.items():
    #     struct = entry['structure']  # pymatgen.core.structure obj
    #     structures.append(struct)


    # structures = [Structure.from_file("P2_NaNiO2.cif") for _ in range(5)]  # 替换为实际文件路径
    # dataset = CrystalDataset(structures)#, labels)
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # print(f'len of dataloader : {len(dataloader)}')
    # for grids in dataloader:
    #     print(grids.shape)
    #     # len of dataloader : 3
    #     # torch.Size([2, 1, 64, 64, 64])
    #     break




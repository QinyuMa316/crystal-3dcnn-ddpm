import numpy as np
from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

GRID_SIZE = 64

def get_3d_grid(structure, grid_size=GRID_SIZE):
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

def visualize_3d_grid(grid):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 获取网格中非零元素的索引
    atom_indices = np.argwhere(grid > 0)
    atom_types = grid[atom_indices[:, 0], atom_indices[:, 1], atom_indices[:, 2]]
    # 为不同原子类型分配颜色
    colors = plt.cm.jet((atom_types - atom_types.min()) / (atom_types.max() - atom_types.min()))
    ax.scatter(atom_indices[:, 0], atom_indices[:, 1], atom_indices[:, 2], c=colors, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def grid_to_structure(grid, grid_size=GRID_SIZE, lattice=None):  # 使用相同的网格分辨率
    frac_coords = []
    atom_types = []

    x, y, z = np.nonzero(grid)
    for i in range(len(x)):
        frac_coords.append([x[i] / (grid_size - 1), y[i] / (grid_size - 1), z[i] / (grid_size - 1)])
        atom_types.append(Element.from_Z(grid[x[i], y[i], z[i]]))

    if lattice is None:
        lattice = Lattice.cubic(10.0)  # 可以调整晶胞参数
    structure = Structure(lattice, atom_types, frac_coords)
    return structure

if __name__ == '__main__':
    cif_file_path = "../str2grid_reverse/P2_NaNiO2.cif"
    structure = Structure.from_file(cif_file_path)
    structure.make_supercell([3, 3, 1])
    structure.to_file(filename="../str2grid_reverse/P2_NaNiO2_661.cif")
    # origin for p2 [3,1,1] 6Na 6Ni 12O * 12(2*6)
    # [3,1,1]*[2,6,1]
    # now [6,6,1] 72Na 72Ni 144O
    grid = get_3d_grid(structure)
    print(grid.shape)  # (64, 64, 64)
    # visualize_3d_grid(grid)
    structure_from_grid = grid_to_structure(grid, lattice=structure.lattice)
    structure_from_grid.to(filename="../str2grid_reverse/reversed_str_v2_661.cif")
    #
    print("structure information of original structure : ")
    analyzer = SpacegroupAnalyzer(structure)
    print(f"Space group: {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")
    print(f"Crystal system: {analyzer.get_crystal_system()}")
    print(f"Lattice type: {analyzer.get_lattice_type()}")
    # 检查生成的结构的空间群对称性
    print("structure information of reserved structure : ")
    analyzer = SpacegroupAnalyzer(structure_from_grid)
    print(f"Space group: {analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})")
    print(f"Crystal system: {analyzer.get_crystal_system()}")
    print(f"Lattice type: {analyzer.get_lattice_type()}")
    '''
    supercell[3,3,1]+grid_size=64
    structure information of original structure : 
    Space group: P6_3/mmc (194)
    Crystal system: hexagonal
    Lattice type: hexagonal
    structure information of reserved structure : 
    Space group: P31m (157)
    Crystal system: trigonal
    Lattice type: hexagonal
    '''
    gen_grid = get_3d_grid(structure_from_grid)
    # visualize_3d_grid(gen_grid)
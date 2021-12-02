import numpy as np
from path import Path


def combine_npz_files(path, output_path):
    id, primary, evolutionary, secondary, tertiary, ter_mask = [], [], [], [], [], []
    for f in Path(path).walkfiles():
        data = np.load(f, allow_pickle=True)
        id.append(data['id'])
        primary.append(data['primary'])
        secondary.append(data['secondary'])
        evolutionary.append(data['evolutionary'])
        tertiary.append(data['tertiary'])
        ter_mask.append(data['ter_mask'])

    np.savez(f'{output_path}/protein_combined.npz', id=np.concatenate(id), primary=np.concatenate(primary),
             secondary=np.concatenate(secondary),
             tertiary=np.concatenate(tertiary), evolutionary=np.concatenate(evolutionary),
             ter_mask=np.concatenate(ter_mask))
    return np.load(f'{output_path}/protein_combined.npz', allow_pickle=True)


def contact_map(tertiary, ter_mask):
    b, l, n = tertiary.shape
    c_map = np.zeros((b, l, l, 1))
    for i in range(b):
        for j in range(l):
            for k in range(l):
                if ter_mask[i][j][k] == 0:  # pad or cordinates not known
                    continue

                x1, y1, z1 = tertiary[i][j]
                x2, y2, z2 = tertiary[i][k]
                distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
                if distance <= 800:
                    c_map[i][j][k] = 1
    return c_map


def chunk_and_save(path, id, pair_wise, evo, cmap, mask):
    indices = list(range(1000, id.shape[0], 1000))
    ids = np.split(id, indices_or_sections=indices)
    pairs = np.split(pair_wise, indices_or_sections=indices)
    evos = np.split(evo, indices_or_sections=indices)
    cmaps = np.split(cmap, indices_or_sections=indices)
    masks = np.split(mask, indices_or_sections=indices)
    for i in range(len(ids)):
        np.savez(f'{path}/{i}.npz', id=ids[i], pair=pairs[i], evo=evos[i], cmap=cmaps[i], mask=masks[i])
    return


if __name__ == '__main__':
    id, primary, evolutionary, secondary, tertiary, ter_mask = combine_npz_files('C:\\Users\\amitk\\Downloads\\npz',
                                                                                 'C:\\Users\\amitk\\Downloads')

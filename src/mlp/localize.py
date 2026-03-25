"""
Ava Powelson
B00802243
March 25, 2026

See README for references.
"""
import numpy as np
from sklearn.manifold import MDS

def get_coords(gbl_data):
    distances = gbl_data[['id', 'who_CH', 'Dist_To_CH']]
    distances['id'] = distances['id'] % 100
    distances['who_CH'] = distances['who_CH'] % 100
    distances = distances.drop_duplicates()
    distances = distances.drop(distances[distances.Dist_To_CH == 0].index)

    """
    Drop duplicate rows irrespective of cell order by sorting before dropping.
    """
    row_keys = distances.apply(lambda row: tuple(sorted(row)), axis=1)
    distances = distances.loc[row_keys.drop_duplicates().index]
    # print(distances[((distances['id'] == 4) | (distances['who_CH'] == 4)) & ((distances['id'] == 23) | (distances['who_CH'] == 23))])

    """
    Find average distance between every node pair.
    """
    row_keys = distances[["id", "who_CH"]].apply(sorted, axis=1).apply(tuple)
    max_ds = [-1, (0, 0)]
    pairs = []
    for i in range(0, 100):
        for j in range(i+1, 100):
            tuple_ij = (i, j)
            pairs.append([i, j, 0])
            matches = row_keys[row_keys == tuple_ij]
            # print(matches)
            pairs[-1][-1] = (distances.loc[matches.index]['Dist_To_CH'].mean())

    pairs = [(i, j, d) for i, j, d in pairs if not np.isnan(d)]

    """
    Construct a layout on a 100x00 grid that maintains (best effort) 
    inter-node distances using Multi-Dimensional Scaling (MDS).
    """
    NUM_NODES = 100

    # make grid w/ distances
    D = np.zeros((NUM_NODES, NUM_NODES))
    for n1, n2, d in pairs:
        D[n1, n2] = d
        D[n2, n1] = d

    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i != j and D[i, j] == 0:
                D[i, j] = D.max()

    mds = MDS(
        n_init=4,
        n_components=2,
        init='random',
        metric="precomputed",
        random_state=42
    )

    # fit distances into coordinates & scale to 100x100
    coords = mds.fit_transform(D)
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    scaled = (coords - min_vals) / (max_vals - min_vals) * 100
    x = scaled[:, 0]
    y = scaled[:, 1]

    return x, y

    # import matplotlib.pyplot as plt
    # plot nodes
    # plt.figure(figsize=(6, 6))
    # plt.scatter(x, y)

    # for i in range(NUM_NODES):
    #     plt.text(x[i], y[i], i)

    # plt.grid(True)
    # plt.show()

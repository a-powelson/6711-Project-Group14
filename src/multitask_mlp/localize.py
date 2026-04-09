"""
Ava Powelson
B00802243
March 25, 2026

See README for references.
"""
import numpy as np
from sklearn.manifold import MDS

def get_coords(gbl_data, N=100):
    distances = gbl_data[['id', 'who_CH', 'Dist_To_CH']]
    distances['id'] = distances['id'] % N
    distances['who_CH'] = distances['who_CH'] % N
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
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            tuple_ij = (i, j)
            pairs.append([i, j, 0])
            matches = row_keys[row_keys == tuple_ij]
            # print(matches)
            d = distances.loc[matches.index]['Dist_To_CH'].mean()
            if not np.isnan(d):
                pairs[-1][-1] = d
            else:
                pairs[-1][-1] = 0

    """
    Construct a layout on a 100x00 grid that maintains (best effort) 
    inter-node distances using Multi-Dimensional Scaling (MDS).
    """
    # make grid w/ distances
    D = np.zeros((N, N))
    for n1, n2, d in pairs:
        D[n1, n2] = d
        D[n2, n1] = d

    # Get rid of (non-diagonal) 0s
    for i in range(N):
        for j in range(N):
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

# Test
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from preprocess import load_data

    gbl_data = load_data('data/wsn-ds.csv')
    x, y = get_coords(gbl_data)
    
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=300, alpha=0.2, edgecolors='black')

    for i in range(100):
        plt.text(x[i], y[i], i, size=12, ha='center', va='center')

    plt.grid(True)
    # plt.title("MDS Locations of Nodes in WSN-DS")
    plt.xticks([0, 25, 50, 75, 100], fontsize=18)
    plt.yticks([0, 25, 50, 75, 100], fontsize=18)
    plt.grid(False)
    plt.savefig('./charts/dataset/MDS_grid.png', dpi=600)
    # plt.show()

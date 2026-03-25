from preprocess import *

"""
Load & prepare data
"""
gbl_data = load_data('data/wsn-ds.csv')
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
Node pairs and distances in dataset
"""
row_keys = distances[["id", "who_CH"]].apply(sorted, axis=1).apply(tuple)
max_ds = [-1, (0, 0)]
pairs = []
for i in range(0, 100):
    for j in range(i+1, 100):
        tuple_ij = (i, j)
        pairs.append([tuple_ij, []])
        matches = row_keys[row_keys == tuple_ij]
        # print(matches)
        pairs[-1][1].append(distances.loc[matches.index]['Dist_To_CH'].values.tolist())
        if len(pairs[-1][1][0]) > max_ds[0]:
            max_ds[0] = len(pairs[-1][1][0])
            max_ds[1] = pairs[-1][0]

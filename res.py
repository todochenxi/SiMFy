import numpy as np
import scipy.sparse as sp

all_data = [[0, 0, 1, 0],  # (实体0, 关系0, 实体1) - 正向关系
            [1, 1, 2, 0],  # (实体1, 关系1, 实体2) - 正向关系
            [2, 0, 0, 1],  # (实体2, 关系0, 实体0) - 这是关系0的反向关系
            [0, 1, 2, 1]]  # (实体0, 关系1, 实体2) - 这是关系1的反向关系

num_e = 3
num_r = 2
num_r_2 = 4

row = np.array(all_data)[:, 0] * num_r_2 + np.array(all_data)[:, 1]
col_rel = np.array(all_data)[:, 1]
d_ = np.ones(len(row))

tail_rel = sp.csr_matrix((d_, (row, col_rel)), shape=(num_e*num_r_2, num_r_2))

print(tail_rel.toarray())

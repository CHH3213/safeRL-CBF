import torch
import numpy as np
state_batch = np.random.random((256,7))
state_batch = torch.FloatTensor(state_batch)
state_batch = torch.unsqueeze(state_batch, -1)
print(np.shape(state_batch))
# print(state_batch)

thetas = state_batch[:, 2, :].squeeze()
print(np.shape(thetas))
# print(thetas)

batch_size = 256
Rs = torch.zeros((batch_size, 2, 2))
Rs[:, 0, 0] = 0.5
Rs[:, 0, 1] = -0.6
Rs[:, 1, 0] = 0.5
Rs[:, 1, 1] = 0.6
print(np.shape(Rs))


a=torch.Tensor([[[3,4],[1,2]],[[3,4],[1,2]],[[3,4],[1,2]]])
b=torch.Tensor([[[1,2],[3,4]],[[1,2],[3,4]],[[3,4],[1,2]]])
print(a.shape)
print(b.shape)
print(torch.bmm(a,b))
print(np.shape(torch.bmm(a,b)))
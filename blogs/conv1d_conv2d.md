# Conv1d and Conv2d in PyTroch



<pre></code>
import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
x = Variable(torch.FloatTensor(np.random.rand(128,1,300*20)))
conv = nn.Conv1d(in_channels = 1,
                      out_channels =32,
                      kernel_size = 600,stride= 300)

y = conv(x)
print(y.size())

x= x.view(x.size()[0],x.size()[1],-1,300)
conv = nn.Conv2d(in_channels = 1,
                      out_channels =32,
                      kernel_size = (2,300))
y=conv(x)
print(y.size())
y.squeeze().size()
</code>
</pre>
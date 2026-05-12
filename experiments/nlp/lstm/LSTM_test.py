import os
# matplotlib notebook
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
LABELS = ['AdaBelief','SGD']#'Adam', 'AdaBound','SGD','AmsVar','AmsBound','AmsVar4']

params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
         }
         #'axes.legend.fontsize':'medium',}
#plt.rcParams.keys()
plt.rcParams.update(params)

import os
# matplotlib notebook
import matplotlib.pyplot as plt
import torch
import numpy as np
import math
LABELS = ['AdaBelief','SGD']#'Adam', 'AdaBound','SGD','AmsVar','AmsBound','AmsVar4']

params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
         }
         #'axes.legend.fontsize':'medium',}
#plt.rcParams.keys()
plt.rcParams.update(params)

names = [
    'PTB.pt-niter-200-optimizer-adabelief-nlayers3-lr0.01-clip-0.25-eps1e-12-epsqrt0.0-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
    'PTB.pt-niter-200-optimizer-sgd-nlayers3-lr30.0-clip-0.25-eps1e-08-epsqrt1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
    'PTB.pt-niter-200-optimizer-adabound-nlayers3-lr0.01-clip-0.25-eps1e-08-epsqrt1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
    'PTB.pt-niter-200-optimizer-adam-nlayers3-lr0.01-clip-0.25-eps1e-08-epsqrt1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
    'PTB.pt-niter-200-optimizer-adamw-nlayers3-lr0.01-clip-0.25-eps1e-08-epsqrt1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
    'PTB.pt-niter-200-optimizer-msvag-nlayers3-lr30.0-clip-0.25-eps1e-08-epsqrt1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
    'PTB.pt-niter-200-optimizer-radam-nlayers3-lr0.01-clip-0.25-eps1e-08-epsqrt1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',
    'PTB.pt-niter-200-optimizer-yogi-nlayers3-lr0.01-clip-0.25-eps0.001-epsqrt1e-08-betas-0.9-0.999-run0-wdecay1.2e-06-when-[100, 145]',

]
labels = ['AdaBelief',
          'SGD',
          'AdaBound',
          'Adam',
          'AdamW',
          'MSVAG',
          'RAdam',
          'Yogi'
          ]
plt.plot(names, 'Train', labels=labels)
plt.savefig('Train_lstm_3layer.png', dpi=600)
plt.plot(names, 'Test', ylim=(50, 90), labels=labels)
plt.savefig('Test_lstm_3layer.png', dpi=600)

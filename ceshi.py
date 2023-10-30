from torch import distributed as dist
from utils.ddp_utils import init_distributed_mode

print(dist.is_available())
init_distributed_mode()
print(dist.is_initialized())

    

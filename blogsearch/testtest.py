from bitsandbytes.cextension import LIBBINSANDBYTES_AVAILABLE
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print(LIBBINSANDBYTES_AVAILABLE)  # should be True

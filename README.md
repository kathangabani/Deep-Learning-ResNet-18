# ProjectKS

## Description
This project implements a neural network model using PyTorch for a deep learning task. It includes necessary data loading steps, model definitions (including a custom `BottleneckBlock`), and installation steps.

## Setup & Installation

1. **Clone the Repository**
   - Clone this repository to your local machine using:
     ```bash
     git clone <https://github.com/kathangabani/Deep-Learning-ResNet-18>
     ```

2. **Data Preparation**
   - The script assumes that data is stored in a directory accessible by the code. Modify the file paths as needed:
     ```python
     for dirname, _, filenames in os.walk('/kaggle/input'):
         for filename in filenames:
             print(os.path.join(dirname, filename))
     ```

## Code Overview

### Imports
- **numpy**: For linear algebra operations.
- **pandas**: For data processing (e.g., reading CSV files).
- **torch, torchvision**: For deep learning functionalities, such as model training and data transformations.
- **PIL**: For image manipulation.

### Model
A custom `BottleneckBlock` for ResNet is implemented as part of the model architecture. The block uses a typical expansion factor of 4, commonly used in ResNet-50/101/152 models.

```python
class BottleneckBlock(nn.Module):
    expansion = 4  # Typical expansion factor

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

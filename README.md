# s12erav3

This repository contains the implementation of a custom transformer model for text generation, utilizing the GPT-2 tokenizer from the `tiktoken` library. The project includes scripts for model definition, training, and an example input text file.
## Demo
[Huggingface Space app](https://huggingface.co/spaces/piyushgrover/Decoder124M-S12ERAv3)
## Repository Structure

- **`model.py`**: Defines the architecture of the custom transformer model.
- **`train.py`**: Contains the training loop and procedures for model training.
- **`input.txt`**: Example input text file used for training or evaluation.
- **`requirements.txt`**: Lists the dependencies required to run the project.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.

## Getting Started

### Prerequisites

Ensure you have Python installed. It's recommended to use a virtual environment to manage dependencies.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/piygr/s12erav3.git
   cd s12erav3
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Training the Model

To train the model, run:
```bash
python train.py
```
This script will initiate the training process using the configurations defined within.

#### Training Logs

Training was run for 150 epochs to achieve the target loss (<0.0999)


```
1 epoch = 41 batches (batch size: 256)

step0, loss: 10.993502616882324
step41, loss: 6.356048583984375
step82, loss: 5.995212078094482
step123, loss: 5.77719259262085
step164, loss: 5.522269248962402
step205, loss: 5.350070476531982
step246, loss: 5.2119574546813965
step287, loss: 5.123021125793457
step328, loss: 4.9851884841918945
step369, loss: 4.9308085441589355
.
.
.
step410, loss: 4.812866687774658
step451, loss: 4.726990222930908
step492, loss: 4.65977144241333
step533, loss: 4.575459003448486
step574, loss: 4.508343696594238
step615, loss: 4.396072864532471
step656, loss: 4.40117073059082
step697, loss: 4.354644298553467
step738, loss: 4.173995018005371
step779, loss: 4.0909504890441895
step820, loss: 4.019453048706055
step861, loss: 3.985774517059326
step902, loss: 3.92303204536438
step943, loss: 3.8618218898773193
step984, loss: 3.7643375396728516
step1025, loss: 3.708109140396118
step1066, loss: 3.5901601314544678
step1107, loss: 3.4955663681030273
step1148, loss: 3.5195417404174805
step1189, loss: 3.3989686965942383
step1230, loss: 3.216315507888794
step1271, loss: 3.229893922805786
step1312, loss: 3.1352314949035645
step1353, loss: 3.0728228092193604
step1394, loss: 3.0254807472229004
step1435, loss: 2.880225419998169
step1476, loss: 2.883176326751709
step1517, loss: 2.7413694858551025
step1558, loss: 2.711130142211914
step1599, loss: 2.6285529136657715
step1640, loss: 2.5204429626464844
step1681, loss: 2.494588613510132
step1722, loss: 2.337798595428467
step1763, loss: 2.2047386169433594
step1804, loss: 2.126124143600464
step1845, loss: 2.1257786750793457
step1886, loss: 2.0513439178466797
step1927, loss: 1.932582139968872
step1968, loss: 1.9159703254699707
step2009, loss: 1.8414294719696045
step2050, loss: 1.7314482927322388
step2091, loss: 1.697108268737793
step2132, loss: 1.5796211957931519
.
.
.
step2542, loss: 0.6851575374603271
step2583, loss: 0.7083978056907654
step2624, loss: 0.6137531399726868
step2665, loss: 0.4813488721847534
step2706, loss: 0.4696134030818939
step2747, loss: 0.4351591467857361
step2788, loss: 0.3966071605682373
step2829, loss: 0.38692599534988403
step2870, loss: 0.32090145349502563
step2911, loss: 0.270128071308136
step2952, loss: 0.24507303535938263
step2993, loss: 0.2236078828573227
step3034, loss: 0.20706918835639954
step3075, loss: 0.1983974128961563
step3116, loss: 0.18023601174354553
step3157, loss: 0.16102169454097748
step3198, loss: 0.15884876251220703
.
.
.
step4715, loss: 0.10485374927520752
step4756, loss: 0.10589515417814255
step4797, loss: 0.10833167284727097
step4838, loss: 0.10170894116163254
step4879, loss: 0.10313621163368225
step4920, loss: 0.10321711748838425
step4961, loss: 0.10241436958312988
step5002, loss: 0.10417047142982483
step5043, loss: 0.10369696468114853
step5084, loss: 0.10560239851474762
step5125, loss: 0.11791584640741348
step5166, loss: 0.12627579271793365
step5207, loss: 0.15985025465488434
step5248, loss: 0.2812890112400055
step5289, loss: 0.40623608231544495
step5330, loss: 0.3732604682445526
step5371, loss: 0.23899592459201813
step5412, loss: 0.17271515727043152
step5453, loss: 0.12999747693538666
step5494, loss: 0.11217284947633743
step5535, loss: 0.11175640672445297
step5576, loss: 0.10542663931846619
step5617, loss: 0.10294816642999649
step5658, loss: 0.09804287552833557
step5699, loss: 0.09575089812278748
step5740, loss: 0.10066627711057663
step5781, loss: 0.10006291419267654
step5822, loss: 0.09813326597213745
step5863, loss: 0.09689625352621078
step5904, loss: 0.09795211255550385
```
## Usage

After training, you can utilize the model for text generation tasks. Ensure that the `model.py` and `train.py` scripts are properly configured to load the trained model weights and perform inference.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Note: For detailed information on the implementation and usage, please refer to the comments within the respective Python scripts.*

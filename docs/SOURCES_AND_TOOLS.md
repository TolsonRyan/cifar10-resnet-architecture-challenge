# Sources and tool use

Sheng Tao is the sole author. Codex assisted with implementation, debugging, experiment execution, analysis, figures, and manuscript preparation. The author is responsible for reviewing and understanding the submitted work. All reported experiments derive from saved completed runs; notebook outputs derive from actual execution. Official hidden-test accuracy was confirmed by the author on November 9, 2026.

The PyramidNet residual-block design follows Han et al., *Deep Pyramidal Residual Networks* (CVPR 2017), with block ordering cross-checked against the authors' implementation. ShakeDrop follows Yamada et al., *ShakeDrop Regularization for Deep Residual Learning* (IEEE Access 2019). No pretrained tensors were imported.

RandAugment operation definitions and equalization behavior are adapted from torchvision 0.21.0 (`torchvision/transforms/autoaugment.py` and `_functional_tensor.py`). The implementation shares policy choices within groups of 32 images and vectorizes equalization. The upstream BSD-3-Clause notice is included in `third_party/torchvision-LICENSE.txt`.

The AAAI 2026 style files are unchanged copies of the official author kit; see `report/TEMPLATE_SOURCE.md`. Mathematical methods are cited in the report. PyTorch, NumPy, Pillow, Matplotlib, and Jupyter retain their respective licenses.

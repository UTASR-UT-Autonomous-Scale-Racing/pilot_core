"""
vision — drivable-track segmentation.

    models      candidate networks (U-Net, Fast-SCNN, LR-ASPP, DeepLabV3)
    data        dataset layout, augmentation, loss and IoU metric
    training    training loop and ONNX export
    segmenters  runtime segmenters used by VisionWorker (ONNX network,
                training-free RANSAC ground-plane free space)

Every segmenter outputs a uint8 mask at frame resolution: 1 = drivable track,
0 = everything else.  torch is only needed for training and export.
"""

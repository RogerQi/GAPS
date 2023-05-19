def dispatcher(cfg):
    network_name = cfg.BACKBONE.network
    if network_name == "lenet":
        from backbone.lenet import net
        return net
    elif network_name == "resnet18":
        from backbone.resnet import resnet18
        return resnet18
    elif network_name == "resnet32_cifar":
        from backbone.resnet_cifar import resnet32
        return resnet32
    elif network_name == "vgg16":
        # Note that there is NO batch normalization in this VGG16
        # implementation, which hurts performance. This is for
        # consistency with the PANet implementation
        from backbone.vgg import vgg16
        return vgg16
    elif network_name == "vgg16_seg":
        from backbone.vgg import vgg16_seg
        return vgg16_seg
    elif network_name == "panet_vgg16":
        from backbone.panet_vgg import panet_vgg
        return panet_vgg
    elif network_name == "refinenet_lw50":
        from backbone.lw_refinenet import rf_lw50
        return rf_lw50
    elif network_name == "deeplabv3_resnet50":
        from backbone.deeplabv3 import deeplabv3_resnet50
        return deeplabv3_resnet50
    elif network_name == "deeplabv3_resnet101":
        from backbone.deeplabv3 import deeplabv3_resnet101
        return deeplabv3_resnet101
    elif network_name == "deeplabv3_resnet101_renorm":
        from backbone.deeplabv3_renorm import deeplabv3_resnet101
        return deeplabv3_resnet101
    elif network_name == "deeplabv3_resnet101_new":
        from backbone.deeplabv3_new import deeplabv3_resnet101
        return deeplabv3_resnet101
    elif network_name == "conv4":
        from backbone.conv4 import ConvNet
        return ConvNet
    else:
        raise NotImplementedError

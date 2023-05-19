import torch
import torch.nn as nn
import torch.nn.functional as F

class identity_mod(nn.Module):
    def forward(self, x):
        return x

def dispatcher(cfg, feature_shape, num_classes = -1):
    classifier_name = cfg.CLASSIFIER.classifier
    if num_classes != -1:
        # Specified by user. Manually override.
        num_classes = num_classes
    else:
        # Usual task
        num_classes = cfg.num_classes
    assert num_classes != -1, "Did you forget to specify num_classes in cfg?"
    assert classifier_name != "none"
    if classifier_name == "fc":
        import classifier.fc as fc
        fc_classifier = fc.fc(cfg, feature_shape, num_classes)
        return fc_classifier
    elif classifier_name == "cos":
        import classifier.cos as cos
        cos_classifier = cos.cos(cfg, feature_shape, num_classes)
        return cos_classifier
    elif classifier_name == "euclidean":
        import classifier.euclidean as euclidean
        l2_classifier = euclidean.euclidean(cfg, feature_shape, num_classes)
        return l2_classifier
    elif classifier_name == "c1":
        import classifier.c1 as c1
        c1_seghead = c1.c1(cfg, feature_shape, num_classes)
        return c1_seghead
    elif classifier_name == "identity":
        identity_module = identity_mod()
        return identity_module
    elif classifier_name == 'seg_cos':
        import classifier.seg_cos as seg_cos
        seg_cos_head = seg_cos.seg_cos(cfg, feature_shape, num_classes)
        return seg_cos_head
    elif classifier_name == "plain_c1":
        import classifier.plain_c1 as plain_c1
        plain_c1_head = plain_c1.plain_c1(cfg, feature_shape, num_classes)
        return plain_c1_head
    elif classifier_name == "seg_cos_decoder":
        import classifier.seg_cos_decoder as seg_cos_decoder
        seg_cos_decoder_head = seg_cos_decoder.seg_cos_decoder(cfg, feature_shape, num_classes)
        return seg_cos_decoder_head
    else:
        raise NotImplementedError
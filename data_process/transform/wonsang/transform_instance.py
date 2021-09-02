from torchvision import transforms

from .custom_transform import TwoClassifierOneRegressionTargetTransform


ws_cc_384_288_rhf_norm55 = transforms.Compose([
        transforms.CenterCrop((384, 288)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


ws_cc_384_288_norm55 = transforms.Compose([
        transforms.CenterCrop((384, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


ws_target_two_classifier_one_regression_age60p10 = TwoClassifierOneRegressionTargetTransform(10)
ws_target_two_classifier_one_regression_age60p20 = TwoClassifierOneRegressionTargetTransform(20)

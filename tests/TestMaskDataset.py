from torchvision import transforms, utils

from data_loader.data_sets import MaskDataset


if __name__ == "__main__":
    data_set = MaskDataset("/opt/ml/mask_data",
                                train=True,
                                num_folds=5,
                                folds=(2,3),
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                ])
                           )
    print(len(data_set))
    img, info = data_set[5]
    print(info)
    # utils.save_image(img, "./test_img.jpg")


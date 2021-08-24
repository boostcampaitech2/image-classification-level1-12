from data_loader.data_sets import MaskDataset

if __name__ == "__main__":
    data_set = MaskDataset("/opt/ml/mask_data", num_folds=5, folds=(2,3))
    print(len(data_set))
    print(data_set[5])

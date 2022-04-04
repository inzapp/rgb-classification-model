from rgb_classification_model import RGBClassificationModel

if __name__ == '__main__':
    RGBClassificationModel(
        # pretrained_model_path=r'checkpoints\model_20000_iter.h5',
        lr=0.001,
        momentum=0.9,
        batch_size=8,
        iterations=100000,
        training_view=False).fit()

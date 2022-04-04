from rgb_classification_model import RGBClassificationModel

if __name__ == '__main__':
    RGBClassificationModel(
        lr=0.001,
        momentum=0.9,
        batch_size=32,
        iterations=100000).fit()

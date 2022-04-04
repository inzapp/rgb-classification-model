from rgb_classification_model import RGBClassificationModel

if __name__ == '__main__':
    RGBClassificationModel(
        # pretrained_model_path=r'checkpoints\model_42000_iter_val_loss_0.0017_yuv_2.h5'
        input_shape=(3,),
        lr=0.001,
        momentum=0.9,
        batch_size=8,
        iterations=300000,
        training_view=False).fit()

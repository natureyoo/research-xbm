import torchvision.transforms as T
import albumentations as ab


# def build_transforms(cfg, is_train=True):
#     normalize_transform = T.Normalize(
#         mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
#     )
#     if is_train:
#         transform = T.Compose(
#             [
#                 T.Resize(size=cfg.INPUT.ORIGIN_SIZE[0]),
#                 T.RandomResizedCrop(
#                     scale=cfg.INPUT.CROP_SCALE, size=cfg.INPUT.CROP_SIZE[0]
#                 ),
#                 T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
#                 T.ToTensor(),
#                 normalize_transform,
#             ]
#         )
#     else:
#         transform = T.Compose(
#             [
#                 T.Resize(size=cfg.INPUT.ORIGIN_SIZE[0]),
#                 T.CenterCrop(cfg.INPUT.CROP_SIZE[0]),
#                 T.ToTensor(),
#                 normalize_transform,
#             ]
#         )
#     return transform


def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )
    if is_train:
        transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=cfg.INPUT.FLIP_PROB),
                T.Resize(size=(cfg.INPUT.ORIGIN_SIZE[0], cfg.INPUT.ORIGIN_SIZE[0])),
                T.ToTensor(),
                normalize_transform,
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(size=(cfg.INPUT.ORIGIN_SIZE[0], cfg.INPUT.ORIGIN_SIZE[0])),
                T.ToTensor(),
                normalize_transform,
            ]
        )

    return transform
#
#
# def build_transforms(cfg, is_train=True):
#     if is_train:
#         augment = ab.Compose([
#             ab.augmentations.transforms.RandomCropNearBBox(max_part_shift=0.3),
#             ab.OneOf([
#                 ab.HorizontalFlip(p=1),
#                 ab.Rotate(border_mode=1, p=1)
#             ], p=0.8),
#             ab.OneOf([
#                 ab.MotionBlur(p=1),
#                 ab.OpticalDistortion(p=1),
#                 ab.GaussNoise(p=1)
#             ], p=1),])
#     else:
#         augment = None
#
#     transform = ab.Compose([
#         ab.Resize(cfg.INPUT.ORIGIN_SIZE[0], cfg.INPUT.ORIGIN_SIZE[0]),
#         ab.augmentations.transforms.Normalize(),
#     ])
#
#     return augment, transform


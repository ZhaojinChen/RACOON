"""Classification dataset construction backed by the ResParam data pipeline."""

from ResParams_utils.data_utils_image_cropping_logic import Normalization
from ResParams_utils.data_utils_ori_affine_multiloss import MRIDatasetsAugMultiLoss


class ClassificationMRIDataset(MRIDatasetsAugMultiLoss):
    """Classification dataset with deterministic extra-affine parameters.

    Existing transform parameters are read from the CSV. If the CSV has no
    transform-parameter column, identity affine parameters are used and the
    ResParam training-time random parameter generator is never called.
    """

    def _sample_train_params(self, row):
        return self._load_extra_params_from_row(row)


def build_augmentation_config(args):
    return {
        "enabled": args.use_augmentation,
        "unchanged_prob": args.augmentation_unchanged_prob,
        "use_rician_noise": args.augmentation_use_rician,
        "use_bias_field": args.augmentation_use_bias,
        "use_deface": args.augmentation_use_deface,
        "rician_noise_prob": args.augmentation_rician_prob,
        "bias_field_prob": args.augmentation_bias_prob,
        "deface_prob": args.augmentation_deface_prob,
        "rician_sigma_range": tuple(args.augmentation_rician_sigma),
        "bias_coefficient_range": tuple(args.augmentation_bias_coefficients),
        "bias_order": args.augmentation_bias_order,
        "face_mask_hdf5_path": args.face_mask_hdf5_path,
        "augmentation_mixture": args.augmentation_mixture,
        "validate_face_mask_hdf5": True,
    }


def build_dataset(csv_path, args, train):
    """Build a train/evaluation dataset with ResParam augmentations."""
    return ClassificationMRIDataset(
        csv_path,
        args.hdf5_root,
        transform=Normalization(args.normalization),
        TrainFlag=train,
        LazyLoading=True,
        template=False,
        template_path=args.template_path,
        image_size=tuple(args.image_size),
        add_coord=False,
        add_noise=False,
        use_sitk_hdf5_resample=True,
        sitk_resample_template_path=args.template_path,
        ori_affine_column="ori_affine",
        force_identity_extra_params=False,
        augmentation_config=build_augmentation_config(args) if train else {"enabled": False},
        return_similarity_data=False,
        return_reference_image=False,
        face_mask_hdf5_path=args.face_mask_hdf5_path,
        deface_mask_to_icbm_xfm=args.deface_mask_to_icbm_xfm,
    )

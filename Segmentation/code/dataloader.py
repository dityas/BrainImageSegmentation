import numpy
import nibabel
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import logging


class fMRIDataset(Dataset):

    """
    PyTorch wrapper for dataset.
    """

    def __init__(self, filenames, name="Unnamed_dataset"):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.files = filenames
        self.name = name
        self.logger.info(f"Dataset {self.name} initialised with {len(self.files)} samples.")

    def __get_files(self, folder):
        return list(folder.iterdir())

    def __len__(self):
        return len(self.files)

    def __read_single_sample(self, image_folder):
        image_files = self.__get_files(image_folder)

        t1 = numpy.array(nibabel.load(str(list(filter(lambda x: "t1.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading T1 image")

        t2 = numpy.array(nibabel.load(str(list(filter(lambda x: "t2.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading T2 image")

        flair = numpy.array(nibabel.load(str(list(filter(lambda x: "flair.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading flair image")

        t1ce = numpy.array(nibabel.load(str(list(filter(lambda x: "t1ce.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading T1CE image")

        seg = numpy.array(nibabel.load(str(list(filter(lambda x: "seg.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading segmentation map")

        _input = numpy.stack([t1, t2, t1ce, flair], axis=3)
        label = seg

        self.logger.debug("Transposing matrices")
        _input = numpy.transpose(_input, axes=[3, 2, 0, 1]) / \
                                (numpy.max(_input) + 0.0001)
        label = numpy.transpose(label, axes=[2, 0, 1])

        # A really weird and possibly incorrect way of doing one-hot encoding
        # for segmentation maps. (I should have probably used keras)
        # Decisions... Decisions...
        label1 = 1.0 * (label == 1.0)
        label2 = 1.0 * (label == 2.0)
        label3 = 1.0 * (label == 3.0)
        label4 = 1.0 * (label == 4.0)
        label = numpy.stack([label1, label2, label3, label4], axis=0)

        return (_input, label)

    def __getitem__(self, idx):
        image_folder = self.files[idx]
        sample = self.__read_single_sample(image_folder)

        return sample


class Dataset2d(Dataset):

    """
    PyTorch wrapper for dataset.
    """

    def __init__(self, filenames, name="Unnamed_dataset"):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.files = filenames
        self.name = name
        self.logger.info(f"Dataset {self.name} initialised with {155 * len(self.files)} samples.")

        self.prev_input = None
        self.prev_label = None
        self.prev_file_no = None

        self.scaler = MinMaxScaler()

    def __get_files(self, folder):
        return list(folder.iterdir())

    def __len__(self):
        return len(self.files) * 155

    def __read_single_image(self, image_folder):
        image_files = self.__get_files(image_folder)

        t1 = numpy.array(nibabel.load(str(list(filter(lambda x: "t1.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading T1 image")

        t2 = numpy.array(nibabel.load(str(list(filter(lambda x: "t2.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading T2 image")

        flair = numpy.array(nibabel.load(str(list(filter(lambda x: "flair.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading flair image")

        t1ce = numpy.array(nibabel.load(str(list(filter(lambda x: "t1ce.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading T1CE image")

        seg = numpy.array(nibabel.load(str(list(filter(lambda x: "seg.nii" in str(x), image_files))[0])).get_data(), dtype=numpy.float32)
        self.logger.debug(f"Finished reading segmentation map")

        _input = numpy.stack([t1, t2, t1ce, flair], axis=3)
        label = 1.0 * (seg > 0)

        self.logger.debug("Transposing matrices")
        # _input = numpy.transpose(_input, axes=[3, 2, 0, 1]) / numpy.max(_input)
        # label = numpy.transpose(label, axes=[2, 0, 1])

        # A really weird and possibly incorrect way of doing one-hot encoding
        # for segmentation maps. (I should have probably used keras)
        # Decisions... Decisions...
        # label1 = 1.0 * (label == 1.0)
        # label2 = 1.0 * (label == 2.0)
        # label3 = 1.0 * (label == 3.0)
        # label4 = 1.0 * (label == 4.0)
        # label = numpy.stack([label1, label2, label3, label4], axis=0)

        return (_input, label)

    def get_sample(self, idx):
        file_no = idx // 155
        slice_idx = idx % 155

        if file_no == self.prev_file_no:
            _input, label = self.prev_input, self.prev_label
        else:
            image_folder = self.files[file_no]
            _input, label = self.__read_single_image(image_folder)
            self.prev_input, self.prev_label = _input, label
            self.prev_file_no = file_no

        sample = []
        for channels in range(_input.shape[-1]):
            sample.append(_input[:, :, slice_idx, channels])

        sample = numpy.stack(sample, axis=2)
        label = label[:, :, slice_idx]

        sample = numpy.transpose(sample, [2, 0, 1])

        return (sample, label)

    def __getitem__(self, idx):

        return self.get_sample(idx)

from pathlib import Path
import numpy
import nibabel
import matplotlib.pyplot as plotter
from torch.utils.data import Dataset
import logging

def get_data_files(root, name_filter):

    files = []
    root = Path(root)

    def visitor(node, name_filter, file_list):

        for file in node.iterdir():
            if name_filter in str(file) and file.is_file():
                file_list.append(str(file))
            elif file.is_dir():
                visitor(file, name_filter, file_list)

    visitor(root, name_filter, files)

    return files


def temp_loader(files):

    for file in sorted(files[-7:]):
        img = numpy.array(nibabel.load(file).get_data())
        print(img.shape)
        print(file)
        plotter.imshow(img[:, :, 50])
        plotter.show()


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

    def __getitem__(self, idx):
        image_folder = self.files[idx]
        image_files = self.__get_files(image_folder)

        t1 = numpy.array(nibabel.load(str(list(filter(lambda x: "t1.nii" in str(x), image_files))[0])).get_data())

        t2 = numpy.array(nibabel.load(str(list(filter(lambda x: "t2.nii" in str(x), image_files))[0])).get_data())

        flair = numpy.array(nibabel.load(str(list(filter(lambda x: "flair.nii" in str(x), image_files))[0])).get_data())

        t1ce = numpy.array(nibabel.load(str(list(filter(lambda x: "t1ce.nii" in str(x), image_files))[0])).get_data())

        seg = numpy.array(nibabel.load(str(list(filter(lambda x: "seg.nii" in str(x), image_files))[0])).get_data())

        return (numpy.stack([t1, t2, t1ce, flair], axis=3), seg)

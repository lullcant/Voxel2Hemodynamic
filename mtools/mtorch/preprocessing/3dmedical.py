

from mtool.mio import get_medical_image
from matplotlib import pyplot as plt
from mtool.mutils.mbinary import get_binary_mask,get_largest_n_connected_region
from mtool.mutils.mboxes import get_bounding_box,ge


if __name__ == '__main__':

    image, _ = get_medical_image('../../../data/images/04929753/c_1.nii.gz')

    masks,_ = get_binary_mask(image,mode='otsu')
    masks = get_largest_n_connected_region(masks,n=1)[0]

    bboxs = get_bounding_box(image,masks)
    print(bboxs)


    plt.imshow(image[100])
    plt.show()


    plt.imshow(masks[100])
    plt.show()

    pass



import skimage.segmentation as seg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import timeit
from PIL import Image

def test_parameters(input_image, _segmentation_fn, segmentation_type, **kwargs):
    """
    Show examples of the given segmentation function
    :param input_image: the image to be segmented
    :param _segmentation_fn: the segmentation function
    :param segmentation_type: the type of segmentation being used
    :param kwargs: additional parameters for the segmentation function
    :return: nothing, the examples are shown
    """
    input = np.squeeze(input_image)
    start = timeit.default_timer()
    test = _segmentation_fn(input)
    stop = timeit.default_timer()
    time_taken = stop - start
    print(f"Segmentation: {segmentation_type}, Time: {time_taken}, Parameters: {kwargs}")
    
    return seg.mark_boundaries(input, test)

def display_images(images, titles):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    current_image = 0

    def update_image(index):
        ax.clear()
        ax.imshow(images[index])
        ax.set_title(titles[index])
        plt.draw()

    def next_image(event):
        nonlocal current_image
        current_image = (current_image + 1) % len(images)
        update_image(current_image)

    def prev_image(event):
        nonlocal current_image
        current_image = (current_image - 1) % len(images)
        update_image(current_image)

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(next_image)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(prev_image)

    update_image(current_image)
    plt.show()

if __name__ == '__main__':
    # Load image
    image_path = r"input_images\0594b389-b9dc31c1.jpg"
    image = Image.open(image_path)
    image = np.array(image)

    # Choose segmentation algorithm
    segmentation = "quickshift"

    images = []
    titles = []

    if segmentation == "quickshift":
        for i in range(0, 4):
            for j in range(1, 5):
                for k in range(1, 4):
                    kernel_size = 1 + i
                    max_dist = j * kernel_size
                    ratio = k * 0.33

                    segmentation_fn = (lambda x : seg.quickshift(x, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio, convert2lab=False))

                    segmented_image = test_parameters(image, segmentation_fn, segmentation,
                                                      kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
                    images.append(segmented_image)
                    titles.append(f"{segmentation} Segmentation\nKernel Size: {kernel_size}, Max Dist: {max_dist}, Ratio: {ratio}")

    elif segmentation == "felzenswalb":
        for i in range(0, 4):
            for j in range(0, 9):
                for k in range(0, 6):
                    scale = 1 + 20 * i
                    sigma = j * 0.25
                    min_size = k

                    segmentation_fn = (
                        lambda x: seg.felzenszwalb(x, scale=scale, sigma=sigma, min_size=min_size))

                    segmented_image = test_parameters(image, segmentation_fn, segmentation,
                                                      scale=scale, sigma=sigma, min_size=min_size)
                    images.append(segmented_image)
                    titles.append(f"{segmentation} Segmentation\nScale: {scale}, Sigma: {sigma}, Min Size: {min_size}")

    elif segmentation == "slic":
        for i in range(0, 6):
            for j in range(0, 5):
                for k in range(0, 5):
                    n_segments = 40 + (i * 40)
                    sigma = j * 0.25
                    compactness = 0.001 * (10**k)

                    segmentation_fn = (
                        lambda x: seg.slic(x, n_segments=n_segments, compactness=compactness, sigma=sigma))

                    segmented_image = test_parameters(image, segmentation_fn, segmentation,
                                                      n_segments=n_segments, compactness=compactness, sigma=sigma)
                    images.append(segmented_image)
                    titles.append(f"{segmentation} Segmentation\nSegments: {n_segments}, Compactness: {compactness}, Sigma: {sigma}")

    display_images(images, titles)
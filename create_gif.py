import os
import re
import imageio.v3 as iio


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    """
    from: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    """
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def create_gif(image_list, gif_name):
    frames = [iio.imread(image_name) for image_name in image_list]
    # save them as frames into a gif
    iio.imwrite(gif_name, frames, fps=60)


def main(folder, output_gif):
    episodes = os.listdir(folder)
    episodes.sort(key=natural_keys)
    image_list = []
    for ep in episodes:
        images = os.listdir(os.path.join(folder, ep))
        images.sort(key=natural_keys)
        image_list.extend([os.path.join(folder, ep, image) for image in images])

    create_gif(image_list, output_gif)


if __name__ == "__main__":
    main("./results", "results.mp4")

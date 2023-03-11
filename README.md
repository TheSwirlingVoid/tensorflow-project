# My tensorflow project

This in theory should've classified images of birds among 450 possible classes, but the model right now is ridiculously horrible.

Here is how I am meeting the requirements:

I use data augmentation to randomly rotate the image, cache the image input, and I have a script (work in progress) to
equalize the amount of images for each class in the training dataset.

For the other part of the requirement I have a really crap "live" classifier that can predict any bird image with sub-one-percent accuracy.
It's only SLIGHTLY better than random. Wonderful! I used pickling to read and write class names here.

import matplotlib.pyplot as plt

def show_imgs(arrayImages):
    _, axs = plt.subplots(len(arrayImages), 1, figsize=(24, 16))

    for i, image in enumerate(arrayImages):
        axs[i].imshow(image)
        axs[i].axis('off')
        

    plt.tight_layout()
    plt.show()
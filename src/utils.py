import pandas as pd
import matplotlib.pyplot as plt


def check_for_name(l):

    def check(pos):
        while l[pos] == '11':
            pos -= 1
            check(pos)

        return l[pos]

    return [check(i) for i in range(len(l))]


def return_coords_table(coords):

    sorted_coords = sorted(coords, key=lambda x: [x[0][0], x[0][1]])

    all_cols = [sorted_coords[i::6][:24] for i in range(6)]

    data = {str(i): all_cols[i] for i in range(6)}

    df = pd.DataFrame(all_cols).T
    df = df[df.columns[::-1]]
    df.columns = ['Col1', 'Col2', 'col3', 'col4', 'col5', 'col6']

    return df


def plot_image(img):

    plt.imshow(img, cmap='gray')
    plt.show()

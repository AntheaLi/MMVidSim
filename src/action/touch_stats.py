import numpy as np
import matplotlib.pyplot as plt

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_size_and_color_for_plt_scatter(size, lim_low, lim_high):

    s = 75 
    size = np.clip(size, lim_low, lim_high)
    c = (size - lim_low) / (lim_high - lim_low)

    return s, c



def get_grid_mapping(grid_bbox, plot_box, data):

    g0, g1 = grid_bbox
    xs, ys = plot_box 

    num_x, num_y = int(g1[0]  - g0[0]), int(g1[1] - g0[1])

    mapping = np.zeros((num_x+1, num_y+1, 2))
    heatmap_x = []
    heatmap_y = []
    color = []
    mapping_x = []
    mapping_y = []
    for xx in range(num_x+1):
        for yy in range(num_y+1):

            xs_ratio = xx / num_x
            ys_ratio = yy / num_y

            l0 = [(xs[1] - xs[0]) * xs_ratio + xs[0], (ys[1] - ys[0]) * xs_ratio + ys[0]]
            l1 = [(xs[2] - xs[1]) * ys_ratio + xs[1], (ys[2] - ys[1]) * ys_ratio + ys[1]]
            l2 = [(xs[2] - xs[3]) * xs_ratio + xs[3], (ys[2] - ys[3]) * xs_ratio + ys[3]]
            l3 = [(xs[3] - xs[0]) * ys_ratio + xs[0], (ys[3] - ys[0]) * ys_ratio + ys[0]]

            cur_x, cur_y = line_intersection((l0, l2), (l1, l3))

            mapping[xx, yy] =  (cur_x, cur_y)
            mapping_x.append(cur_x)
            mapping_y.append(cur_y)
            heatmap_x.append(g0[0] + xx)
            heatmap_y.append(g0[1] + yy)
            color.append(data[xx, yy])

    return (heatmap_x, heatmap_y), mapping_x, mapping_y, color



def map_hand(ax, grid_bbox, plot_coords, data):
    # grid_bbox = ((0, 11), (6, 15))
    # plot_coords = ([340, ], [])

    _, x, y, c = get_grid_mapping(grid_bbox, plot_coords, data)

    ax.scatter(x, y, s=10, c=c, cmap='viridis_r', linewidths=.2, edgecolors='k')


    return ax

def map_hands(ax, data, left=True):
    # grid_bbox = ((0, 11), (6, 15))
    # plot_coords = ([340, ], [])

    # left hand
    if left:
        # grid_bboxs = [([0, 9], [2, 15]), ([3, 9], [5, 15]), ([6, 9], [8, 15]), ([9, 9], [11, 15]), ([11, 0], [15, 6]), ([0, 0], [9, 10])]
        grid_bboxs= []
        grid_bboxs.append(([0, 0], [2, 6]))
        grid_bboxs.append(([3, 0], [5, 6]))
        grid_bboxs.append(([6, 0], [8, 6]))
        grid_bboxs.append(([9, 0], [11, 6]))
        grid_bboxs.append(([10, 12], [13, 15]))
        grid_bboxs.append(([7, 0], [15, 10]))
        plot_coords = [([106, 114, 133, 111], [147, 143, 242, 268]), ([154, 172, 185, 160], [79, 75, 224, 236]), ([202, 226, 236, 208], [43, 40, 214, 220]), \
            ([285, 305, 298, 269], [69, 68, 234, 229]), ([356, 340, 406, 422], [357, 335, 298, 302]), ([123, 291, 280, 151], [281, 261, 416, 439])]


    # right_hand
    else:
        # grid_bboxs = [([0, 0], [2, 6]), ([3, 0], [5, 6]), ([6, 0], [8, 6]), ([9, 0], [11, 6]), ([12,12], [15, 13]), ([0, 0],  [7, 8])  ]
        # grid_bboxs = [([0, 0], [2, 6]), ([3, 0], [5, 6]), ([6, 0], [8, 6]), ([9, 0], [11, 6]), ([12,12], [15, 13]), ([0, 0],  [7, 8])  ]
        # grid_bboxs = [([10, 13], [15, 15]), ([10, 10], [15, 12]), ([10, 7], [15, 9]), ([10, 4], [15, 6]), ([0, 0], [ 3, 5]), ([0, 5], [9, 15]) ]
        grid_bboxs= []
        grid_bboxs.append(([0, 13], [6, 15]))
        grid_bboxs.append(([0, 10], [6, 12]))
        grid_bboxs.append(([0, 7], [6, 9]))
        grid_bboxs.append(([0, 4], [6, 6]))
        grid_bboxs.append(([10, 0], [13, 3]))
        grid_bboxs.append(([7, 4], [15, 15]))
        
        plot_coords = [([403, 420, 428, 422], [228, 143, 150, 231]), ([350, 363, 370, 374], [220, 77, 82, 226]), ([298, 308, 328, 331], [214, 41, 42, 212]), \
            ([233, 230, 250, 261], [209, 75, 69, 207]), ([113, 128, 186, 175], [303, 287, 333, 357]), ([256, 237, 413, 387], [434, 258, 273, 443])]


    xl, yl, gx_l, gy_l, cl = [], [], [], [], []
    for i in range(len(grid_bboxs)):
        (gx, gy), x, y, c = get_grid_mapping(grid_bboxs[i], plot_coords[i], data)
        xl.extend(x)
        yl.extend(y)
        gx_l.extend(gx)
        gy_l.extend(gy)
        ax.scatter(x, y, s=10, c=c, cmap='viridis_r', linewidths=.2, edgecolors='k')

    print(len(xl), len(yl), len(gx_l), len(gy_l))
    left_hand_mapping = np.array([gx_l, gy_l, xl, yl])

    return ax, left_hand_mapping


if __name__ == '__main__':
    contour_left = np.load('./left.npy')
    data = np.random.random((32, 32))
    axs = plt.figure(constrained_layout=True).subplots(1, 2, sharex=False, sharey=False)
    axs[0].imshow(contour_left)

    _, left_hand_mapping = map_hands(axs[0], data, left=True)

    np.save("left_mapping.npy", left_hand_mapping.T)
    # grid_bbox1 = ([0, 9], [2, 15])
    # plot_coords1 = ([106, 114, 133, 111], [147, 143, 242, 268])

    # axs[0] = map_hand(axs[0], grid_bbox1, plot_coords1, data)

    # grid_bbox2 = ([3, 9], [5, 15])
    # plot_coords2 = ([154, 172, 185, 160], [79, 75, 224, 236])

    # axs[0] = map_hand(axs[0], grid_bbox2, plot_coords2, data)

    # grid_bbox3 = ([6, 9], [8, 15])
    # plot_coords3 = ([202, 226, 236, 208], [43, 40, 214, 220])

    # axs[0] = map_hand(axs[0], grid_bbox3, plot_coords3, data)

    # grid_bbox4 = ([9, 9], [11, 15])
    # plot_coords4 = ([285, 305, 298, 269], [69, 68, 234, 229])

    # axs[0] = map_hand(axs[0], grid_bbox4, plot_coords4, data)

    # grid_bbox5 = ([11, 0], [15, 6])
    # plot_coords5 = ([356, 340, 406, 422], [357, 335, 298, 302])

    # axs[0] = map_hand(axs[0], grid_bbox5, plot_coords5, data)

    # grid_bbox6 = ([0, 0], [9, 10])
    # plot_coords6 = ([123, 291, 280, 151], [281, 261, 416, 439])

    # axs[0] = map_hand(axs[0], grid_bbox6, plot_coords6, data)





    contour_right = np.load('./right.npy')
    axs[1].imshow(contour_right)

    _, right_hand_mapping = map_hands(axs[1], data, left=False)
    np.save("right_mapping.npy", right_hand_mapping.T)


    # grid_bbox1 = ([10, 13], [15, 15])
    # ([0, 0], [2, 6])
    # plot_coords1 = ([403, 420, 428, 422], [228, 143, 150, 231])

    # axs[1] = map_hand(axs[1], grid_bbox1, plot_coords1, data)

    # grid_bbox2 = ([10, 10], [15, 12])
    # ([3, 0], [5, 6])
    # plot_coords2 = ([350, 363, 370, 374], [220, 77, 82, 226])

    # axs[1] = map_hand(axs[1], grid_bbox2, plot_coords2, data)

    # grid_bbox3 = ([10, 7], [15, 9])
    # ([6, 0], [8, 6])
    # plot_coords3 = ([298, 308, 328, 331], [214, 41, 42, 212])

    # axs[1] = map_hand(axs[1], grid_bbox3, plot_coords3, data)

    # grid_bbox4 = ([10, 4], [15, 6])
    # ([9, 0], [11, 6])
    # plot_coords4 = ([233, 230, 250, 261], [209, 75, 69, 207])   

    # axs[1] = map_hand(axs[1], grid_bbox4, plot_coords4, data)

    # grid_bbox5 = ([0, 0], [ 3, 5])
    # ([12,12], [15, 13])
    # plot_coords5 = ([113, 128, 186, 175], [303, 287, 333, 357])

    # axs[1] = map_hand(axs[1], grid_bbox5, plot_coords5, data)

    # grid_bbox6 = ([0, 5], [9, 15])
    # ([0, 0],  [7, 8])
    # plot_coords6 = ([256, 237, 413, 387], [434, 258, 273, 443])

    # axs[1] = map_hand(axs[1], grid_bbox6, plot_coords6, data)

    plt.show()

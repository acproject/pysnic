from pkg_resources import resource_stream
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from skimage.segmentation import mark_boundaries
from timeit import default_timer as timer
from itertools import chain
from pysnic.algorithms.snic import snic, compute_grid
from pysnic.ndim.operations_collections import nd_computations
from pysnic.metric.snic import create_augmented_snic_distance

# load image
color_image = np.array(Image.open(resource_stream(__name__, "../data/5x.png")).convert("RGB"))
number_of_pixels = color_image.shape[0] * color_image.shape[1]

# SNIC parameters
numSegments = 100
compactness = 10.00

# compute grid
grid = compute_grid(color_image.shape, numSegments)
seeds = list(chain.from_iterable(grid))
seed_len = len(seeds)

# choose a distance metric
distance_metric = create_augmented_snic_distance(color_image.shape, seed_len, compactness)


start = timer()

segmentation, distances, numSegments, centroids = snic(
    skimage.color.rgb2lab(color_image).tolist(),
    seeds,
    compactness, nd_computations["3"], distance_metric,
    update_func=lambda num_pixels: print("processed %05.2f%%" % (num_pixels * 100 / number_of_pixels)))

end = timer()
print("superpixelation took: %fs" % (end - start))
labels = np.array(segmentation.copy())
# show the output of SNIC
plt.figure("SNIC with %d segments" % numSegments)
plt.imshow(mark_boundaries(color_image, np.array(segmentation)))
plt.show()

# show the distance map
plt.figure("Distances")
plt.imshow(distances, cmap="gray")
plt.show()

from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np

def _weight_mean_color(graph, src, dst, n):
    """
        Callback to handle merging nodes by recomputing mean color.
        The method expects that the mean color of `dst` is already computed.

        Parameters
        ----------
        graph : RAG(Region Adjacency Graph)
            the graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbot of `src` or `dst` or both.
        Returns:
        data: dict
        A dictionary with the `weight` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """
    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src ,dst):
    """
        Callback called before merging two nodes of a mean color distance graph.
        This method computes the mean color of 'dst'.
        Parameters
        ----------
        graph: RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
    """

    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count'])



# print(labels)
g = graph.rag_mean_color(color_image, labels)
labels2 = graph.merge_hierarchical(labels, g, thresh=40, rag_copy=False, in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)
g2 = graph.rag_mean_color(color_image, labels2)
out2 = color.label2rgb(labels2, color_image, kind='avg')

plt.imshow(out2)
plt.show()
